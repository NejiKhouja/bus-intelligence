"""Enrichissement des coordonnées d'arrêts — combine `OpenData` (dictionnaire nom -> coord),
`Historique_Tickets` (ordre des arrêts + triangulation horaire) et la fondation GPS existante.

Le pipeline actuel (`foundation._stop_station_frame`) échoue pour 266 des 402 lignes
`ligne` (< 4 ancres géocodées) faute de coordonnées. Ce module fournit une SECONDE source,
utilisée en repli avant l'abandon pur et simple d'une ligne :

1. `opendata_dict()`      : dictionnaire {nom normalisé -> [(lat, lon, source), ...]} à partir des
                            4 collections `OpenData.Station*` (~3 358 noms uniques). GARDE TOUS
                            LES CANDIDATS par nom : des noms génériques (« STADE MUNICIPALE »,
                            « MEUBLATEX ») désignent des arrêts RÉELS DIFFÉRENTS à des endroits du
                            pays distants de >100 km — un dictionnaire premier-trouvé-gagne
                            choisirait silencieusement la mauvaise coordonnée pour ~10% des noms
                            dupliqués (210/2109 conflits mesurés, voir notebook 09 §0.5).
2. `ticket_ordered_stops()` : pour une (ligne, société), la séquence d'arrêts nommés et
                            ordonnée dérivée des codes `origine`/`Distination` des tickets —
                            agrégée sur TOUTES les années disponibles (2019-2026).
3. `_disambiguate_sequential()` : quand un nom a plusieurs candidats, on retient celui géographiquement
                            cohérent avec les arrêts déjà résolus (sans ambiguïté) les plus proches
                            dans la séquence de la ligne — un vrai trajet de bus ne fait pas de
                            saut de 100 km entre deux arrêts consécutifs.
4. `triangulate_stop()`   : pour un arrêt sans coordonnée (ni GPS ni OpenData), on utilise
                            l'horodatage des tickets qui le référencent + le bus concerné pour
                            retrouver la position GPS du bus à ce moment précis, puis on prend
                            le centroïde sur plusieurs tickets indépendants.
5. `build_enriched_stops()` : assemble les sources en un `stops_frame` compatible avec
                            `foundation.stops_frame` (colonnes seq/route_seq/name/lat/lon/s_m),
                            avec une colonne `source` de provenance pour audit.

Note sur la portée société — les collections `STOP<societe>` (`foundation._STOP_COL_MAP`) et la
table `winicari.station` (différente de `OpenData.Station`) donnent déjà une correspondance
propre PAR société quand elles couvrent la ligne. Ce module n'est utilisé QUE pour les lignes où
cette source manque — il n'y a donc, par construction, aucune table société-spécifique à
recouper : la désambiguïsation doit s'appuyer sur la géométrie de la route elle-même.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pymongo.database import Database

from src.data.foundation import haversine

TICKET_YEARS = ["2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]
_OD_PRIORITY = ["Station", "Station_new", "Station2", "Station_sts"]


def norm(name) -> str:
    """Clé de comparaison : NFKD, majuscule, alphanumérique uniquement (insensible aux accents/espaces)."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", s.upper())


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — dictionnaire OpenData
# ─────────────────────────────────────────────────────────────────────────────

def opendata_dict(od_db: Database) -> dict[str, list[tuple]]:
    """{nom_normalisé -> [(lat, lon, collection_source), ...]} — TOUS les candidats gardés.

    Ne PAS dédupliquer par « premier trouvé » : un nom générique peut légitimement pointer
    vers plusieurs arrêts réels distants. La désambiguïsation se fait au moment de la
    résolution (`_disambiguate_sequential`), avec le contexte de la ligne.
    """
    out: dict[str, list[tuple]] = {}
    for col in _OD_PRIORITY:
        for d in od_db[col].find({}, {"nom_fr": 1, "lat": 1, "lng": 1, "_id": 0}):
            try:
                lat, lng = float(d.get("lat")), float(d.get("lng"))
            except (TypeError, ValueError):
                continue
            if abs(lat) > 1 and abs(lng) > 1 and d.get("nom_fr"):
                key = norm(d["nom_fr"])
                cands = out.setdefault(key, [])
                # éviter les doublons quasi-identiques (même source réelle citée 2x)
                if not any(abs(c[0] - lat) < 1e-4 and abs(c[1] - lng) < 1e-4 for c in cands):
                    cands.append((lat, lng, col))
    return out


def audit_name_conflicts(od_dict: dict, min_km: float = 1.0) -> pd.DataFrame:
    """Noms avec >= 2 candidats dont la distance dépasse `min_km` — mesure la véritable ambiguïté
    du dictionnaire OpenData (ex. « STADE MUNICIPALE » désigne des arrêts réels différents)."""
    rows = []
    for name, cands in od_dict.items():
        if len(cands) < 2:
            continue
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                d_km = haversine(np.array([cands[i][0]]), np.array([cands[i][1]]),
                                  cands[j][0], cands[j][1])[0] / 1000
                if d_km >= min_km:
                    rows.append({"nom": name, "candidat_1": cands[i], "candidat_2": cands[j], "distance_km": round(d_km, 1)})
    return pd.DataFrame(rows).sort_values("distance_km", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def fuzzy_resolve(name: str, od_dict: dict, cutoff: float = 0.82) -> Optional[list[tuple]]:
    """Repli approximatif quand la clé normalisée exacte est absente (variantes/abréviations).

    Retourne la liste de candidats (potentiellement ambigus, comme `opendata_dict`) pour la
    clé la plus proche, ou None si aucune correspondance suffisante.
    """
    import difflib
    key = norm(name)
    match = difflib.get_close_matches(key, od_dict.keys(), n=1, cutoff=cutoff)
    if match:
        return [(lat, lon, f"{src}_fuzzy") for lat, lon, src in od_dict[match[0]]]
    return None


def _nearest_candidate(candidates: list[tuple], ref_lat: float, ref_lon: float) -> tuple:
    """Le candidat le plus proche d'une position de référence (ancre géographique)."""
    return min(candidates, key=lambda c: haversine(np.array([ref_lat]), np.array([ref_lon]), c[0], c[1])[0])


def _disambiguate_sequential(stop_candidates: list[dict]) -> list[Optional[tuple]]:
    """Choisit UN candidat par arrêt en utilisant l'ordre de la route comme contrainte géographique.

    Les arrêts au nom UNIQUE (un seul candidat dans tout le dictionnaire) servent d'ancres de
    confiance. Pour un arrêt ambigu, on retient le candidat le plus proche de l'ancre non-ambiguë
    la plus proche dans la séquence — un trajet de bus réel ne saute pas de 100 km entre deux
    arrêts voisins. S'il n'existe aucune ancre sur toute la ligne (rare), repli sur le premier
    candidat (comportement d'avant, avec provenance `_ambigu` pour signaler la moindre confiance).
    """
    n = len(stop_candidates)
    resolved: list[Optional[tuple]] = [None] * n
    anchors = [i for i, s in enumerate(stop_candidates) if len(s["candidates"]) == 1]

    if not anchors:
        for i, s in enumerate(stop_candidates):
            if s["candidates"]:
                lat, lon, src = s["candidates"][0]
                resolved[i] = (lat, lon, f"{src}_ambigu" if len(s["candidates"]) > 1 else src)
        return resolved

    for i in anchors:
        resolved[i] = stop_candidates[i]["candidates"][0]

    for i, s in enumerate(stop_candidates):
        if resolved[i] is not None or not s["candidates"]:
            continue
        if len(s["candidates"]) == 1:
            resolved[i] = s["candidates"][0]
            continue
        nearest_anchor = min(anchors, key=lambda j: abs(j - i))
        ref_lat, ref_lon, _ = resolved[nearest_anchor]
        resolved[i] = _nearest_candidate(s["candidates"], ref_lat, ref_lon)
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — ordre des arrêts depuis les tickets
# ─────────────────────────────────────────────────────────────────────────────

def _ticket_stop_votes(tk_db: Database) -> pd.DataFrame:
    """Un passage par collection annuelle : compte les votes (societe, line, code, name) pour
    origine->NomFR1 ET Distination->NomFR2, sur TOUTES les années. Beaucoup plus rapide qu'une
    agrégation par ligne (une seule passe par collection au lieu de 402 x 8)."""
    frames = []
    for yr in TICKET_YEARS:
        col = tk_db[f"Ticket{yr}"]
        for code_field, name_field in (("origine", "NomFR1"), ("Distination", "NomFR2")):
            pipe = [
                {"$group": {
                    "_id": {"soc": "$Societe", "line": "$CodeRoute", "code": f"${code_field}", "name": f"${name_field}"},
                    "n": {"$sum": 1},
                }},
            ]
            rows = [{"societe": d["_id"]["soc"], "line": d["_id"]["line"],
                     "code": d["_id"]["code"], "name": d["_id"]["name"], "n": d["n"]}
                    for d in col.aggregate(pipe)
                    if d["_id"].get("soc") and d["_id"].get("line") and d["_id"].get("code") and d["_id"].get("name")]
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["societe", "line", "code", "name", "n"])
    return pd.concat(frames, ignore_index=True)


def build_ticket_stop_index(tk_db: Database) -> pd.DataFrame:
    """Index complet : une ligne par (societe, line, code) avec le nom dominant et le total de votes.

    Coûteux (8 collections x 2 passes) — calculer une fois et mettre en cache sur disque.
    """
    votes = _ticket_stop_votes(tk_db)
    if votes.empty:
        return votes
    agg = (votes.groupby(["societe", "line", "code", "name"])["n"].sum().reset_index())
    agg["code_int"] = pd.to_numeric(agg["code"], errors="coerce")
    agg = agg.dropna(subset=["code_int"])
    agg["code_int"] = agg["code_int"].astype(int)
    # nom dominant par (societe, line, code)
    idx = agg.groupby(["societe", "line", "code_int"])["n"].idxmax()
    best = agg.loc[idx, ["societe", "line", "code_int", "name", "n"]].rename(columns={"code_int": "code"})
    return best.sort_values(["societe", "line", "code"]).reset_index(drop=True)


def ticket_ordered_stops(ticket_index: pd.DataFrame, line: str, societe: str) -> pd.DataFrame:
    """Sous-ensemble ordonné de `build_ticket_stop_index` pour une (ligne, société) donnée."""
    sub = ticket_index[(ticket_index["line"] == str(line)) & (ticket_index["societe"] == societe)]
    return sub.sort_values("code").reset_index(drop=True)


def load_or_build_ticket_index(tk_db: Database, cache_path) -> pd.DataFrame:
    """Charge l'index depuis le cache parquet s'il existe, sinon le calcule et le sauvegarde."""
    from pathlib import Path
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    idx = build_ticket_stop_index(tk_db)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(cache_path, index=False)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — triangulation ticket-heure x ping GPS
# ─────────────────────────────────────────────────────────────────────────────

def _ticket_day_str(jour_service: str) -> Optional[str]:
    """'2025/02/03' -> 'd20250203'."""
    if not jour_service:
        return None
    parts = str(jour_service).split("/")
    if len(parts) != 3:
        return None
    y, m, d = parts
    return f"d{y}{m.zfill(2)}{d.zfill(2)}"


def triangulate_stop(tk_db: Database, gps_db: Database, societe: str, line: str, code,
                      max_tickets: int = 150, window_min: float = 3.0,
                      max_spread_m: float = 800.0, min_used: int = 5) -> Optional[dict]:
    """Position d'un arrêt inféré à partir des pings GPS proches de l'heure des tickets qui le référencent.

    Pour chaque ticket citant `code` comme origine sur cette (ligne, société), on cherche le ping GPS
    du même bus le plus proche dans le temps (fenêtre +/- `window_min` min). Le centroïde de ces
    positions est retenu si suffisamment de tickets indépendants convergent (`min_used`) et que la
    dispersion spatiale reste faible (`max_spread_m`) — sinon la triangulation est jugée peu fiable.

    Échantillonne un quota FIXE PAR ANNÉE (pas « le plus récent d'abord ») : la fenêtre GPS
    utile diffère par société et n'est pas toujours la plus récente -- ex. S.R.T.K a été équipé
    tardivement (bus 6039 : aucun ping avant nov. 2025, donc récent=mieux), mais S.T.S a une
    couverture GPS quasi nulle sur les tout derniers jours alors qu'elle était dense en
    2022-2024 (donc récent=pire). Pour un arrêt à très fort volume (ex. 1,78M tickets), un
    remplissage glouton « récent d'abord » épuise le quota sur 2025-2026 seul et ne voit JAMAIS
    les années où le GPS de CETTE société était bon. Répartir le quota sur toutes les années
    laisse chaque période une vraie chance, quelle que soit la direction du biais GPS.
    """
    tickets: list[dict] = []
    per_year_quota = max(1, max_tickets // len(TICKET_YEARS))
    for yr in TICKET_YEARS:
        col = tk_db[f"Ticket{yr}"]
        cur = col.find(
            {"CodeRoute": str(line), "Societe": societe, "origine": str(code)},
            {"CodeBus": 1, "date": 1, "jour_service": 1, "_id": 0},
        ).limit(per_year_quota)
        tickets.extend(list(cur))
    # si le quota par année n'a pas rempli max_tickets (années sans données), compléter avec
    # les années restantes sans se soucier de l'ordre -- mieux qu'un budget gaspillé
    if len(tickets) < max_tickets:
        for yr in TICKET_YEARS:
            if len(tickets) >= max_tickets:
                break
            col = tk_db[f"Ticket{yr}"]
            cur = col.find(
                {"CodeRoute": str(line), "Societe": societe, "origine": str(code)},
                {"CodeBus": 1, "date": 1, "jour_service": 1, "_id": 0},
            ).skip(per_year_quota).limit(max_tickets - len(tickets))
            tickets.extend(list(cur))

    positions = []
    window = timedelta(minutes=window_min)
    for tk in tickets:
        bus, date_s, jour = tk.get("CodeBus"), tk.get("date"), tk.get("jour_service")
        if not (bus and date_s and jour):
            continue
        day = _ticket_day_str(jour)
        if day is None or day not in gps_db.list_collection_names():
            continue
        try:
            t = pd.Timestamp(date_s.replace("/", "-"))
        except (ValueError, AttributeError):
            continue
        bus_variants = [bus]
        try:
            bus_variants.append(int(bus))
        except (TypeError, ValueError):
            pass
        ping = None
        for bv in bus_variants:
            cur = gps_db[day].find(
                {"bus.code": bv, "date": {"$gte": t - window, "$lte": t + window}},
                {"date": 1, "localisation": 1, "_id": 0},
            )
            cands = list(cur)
            if cands:
                ping = min(cands, key=lambda d: abs((d["date"] - t).total_seconds()))
                break
        if ping is None:
            continue
        loc = ping.get("localisation") or {}
        lat, lon = loc.get("x"), loc.get("y")
        if lat and lon and abs(lat) > 1 and abs(lon) > 1:
            positions.append((float(lat), float(lon)))

    if len(positions) < min_used:
        return None

    # Le bus peut réellement charger à 2+ points distincts près de l'arrêt nominal (portes
    # différentes d'un grand complexe, léger détour un jour donné...). Une simple moyenne +
    # rejet-si-écart-max pénalise à tort un groupe majoritaire TRÈS serré simplement parce
    # qu'une minorité de points est ailleurs (observé : 10/18 points quasi identiques pour
    # HOP.SAHLOUL, mais rejeté par la moyenne globale à cause de 5 points ~1.2km plus loin).
    # On cherche donc le plus grand sous-groupe SERRÉ (`max_spread_m` de rayon) plutôt que
    # d'imposer que TOUS les points convergent.
    lats_all = np.array([p[0] for p in positions])
    lons_all = np.array([p[1] for p in positions])
    best_members, best_clat, best_clon = [], None, None
    for i in range(len(positions)):
        d = haversine(lats_all, lons_all, lats_all[i], lons_all[i])
        members = np.where(d <= max_spread_m)[0]
        if len(members) > len(best_members):
            best_members = members
            best_clat, best_clon = float(lats_all[members].mean()), float(lons_all[members].mean())

    if len(best_members) < min_used:
        return None
    spread = float(haversine(lats_all[best_members], lons_all[best_members], best_clat, best_clon).max())
    return {"lat": best_clat, "lon": best_clon, "n_used": len(best_members), "n_total_candidates": len(positions),
            "spread_m": spread, "source": "ticket_triangulation"}


# ─────────────────────────────────────────────────────────────────────────────
# Assemblage — une ligne enrichie complète
# ─────────────────────────────────────────────────────────────────────────────

def build_enriched_stops(ticket_index: pd.DataFrame, od_dict: dict, line: str, societe: str,
                          tk_db: Optional[Database] = None, gps_db: Optional[Database] = None,
                          triangulate_gaps: bool = False) -> pd.DataFrame:
    """Table d'arrêts pour (ligne, société) : ordre des tickets + coordonnées OpenData
    (avec repli approximatif), et triangulation optionnelle pour les arrêts résiduels.

    Retourne un DataFrame compatible `stops_frame` (seq, route_seq, name, lat, lon, s_m) +
    colonne `source` de provenance, ou un DataFrame vide si < 4 arrêts résolus.
    """
    ordered = ticket_ordered_stops(ticket_index, line, societe)
    if ordered.empty:
        return pd.DataFrame()

    # Passe 1 : rassembler TOUS les candidats par arrêt (sans choisir encore)
    stop_candidates = []
    for _, r in ordered.iterrows():
        key = norm(r["name"])
        cands = od_dict.get(key)
        if cands is None:
            cands = fuzzy_resolve(r["name"], od_dict)
        stop_candidates.append({"code": r["code"], "name": r["name"], "candidates": cands or []})

    # Passe 2 : désambiguïser en utilisant la géométrie de la route comme contrainte
    resolved = _disambiguate_sequential(stop_candidates)

    rows = []
    for s, hit in zip(stop_candidates, resolved):
        if hit is None and triangulate_gaps and tk_db is not None and gps_db is not None:
            tri = triangulate_stop(tk_db, gps_db, societe, line, s["code"])
            if tri is not None:
                hit = (tri["lat"], tri["lon"], tri["source"])
        if hit is None:
            continue
        lat, lon, source = hit
        rows.append({"route_seq": int(s["code"]), "name": s["name"], "lat": lat, "lon": lon, "source": source})

    if len(rows) < 4:
        return pd.DataFrame()

    rows.sort(key=lambda x: x["route_seq"])
    st = pd.DataFrame(rows)
    seg = haversine(st["lat"].values[:-1], st["lon"].values[:-1], st["lat"].values[1:], st["lon"].values[1:])
    st["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
    st.insert(0, "seq", range(len(st)))
    return st
