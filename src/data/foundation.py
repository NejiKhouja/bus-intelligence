
"""Reconstruction de trajets GPS — la couche de fondation partagée.

Transforme les pings GPS bruts (`Historique_pos`) en trajets reconstruits avec l'heure
d'arrivée réelle dérivée à chaque arrêt. Utilisée par la prédiction de retard (labels),
la détection d'anomalies et le repli GPS. Ce module est la source de vérité unique ;
le notebook et le CLI batch (`build_foundation.py`) importent tous deux depuis ici.

Comment la segmentation est réalisée (chaîne en 4 étapes)
----------------------------------------------------------
1. NETTOYAGE  (`clean_pings`)
   Supprime les pings consécutifs aux coordonnées identiques (un bus stationné ping encore
   toutes les ~5 s, ~10% des lignes), en gardant le PREMIER contact pour préserver
   l'horodatage d'arrivée. Annote le délai entre les pings et signale `signal_gap`
   quand le bus est passé en silence.

2. CORRESPONDANCE CARTOGRAPHIQUE  (`project_to_route`)
   Projette chaque ping sur la polyligne d'ancrage de la ligne pour obtenir `s` = distance
   le long du trajet en mètres (puis lissée). La correspondance est *séquentielle-fenêtrée* —
   chaque ping n'est mis en correspondance que près du segment du ping précédent — ce qui
   empêche `s` de reculer quand les ancres sont clairsemées. Transforme un tracé lat/lon
   désordonné en UN seul nombre propre qui monte vers le terminus éloigné et descend au retour.

3. SEGMENTATION  (`segment_trips`) — surveiller `s` dans le temps :
   - UN DEMI-TOUR = `s` s'inverse de plus qu'un seuil d'hystérésis
     (`reversal_frac * route_len`, donc adaptatif : grand pour une ligne de 192 km,
     petit pour une boucle de 6 km). Chaque tronçon entre demi-tours est un trajet ;
     direction = ALLER si `s` monte, RETOUR si elle descend.
   - Une course est divisée UNIQUEMENT à une *pause stationnée* (long écart temporel
     où `s` a à peine bougé), de sorte qu'une interruption de signal en milieu de route
     ne crée pas un faux nouveau trajet.
   - Les trajets plus courts que `min_span` / `min_trip_min` sont supprimés ; chacun est
     étiqueté `full` (couvre les deux extrémités) ou PARTIEL (bus revenu prématurément
     ou journée terminée en cours de route).

4. ARRIVÉES  (`derive_arrivals`)
   Pour chaque trajet, accroche les arrêts dans sa plage couverte au ping le plus proche,
   dans l'ordre de voyage avec des temps strictement croissants ; `matched` indique si le
   bus est passé dans un rayon `arrival_thresh_m` (350 m). Le taux de correspondance par
   ligne est le signal principal de qualité des données.

Ce que cette couche fait et ne calcule PAS encore
--------------------------------------------------
FAIT   - heure d'ARRIVÉE réelle à chaque arrêt (`arrival`), structure du trajet, écarts de signal.
FAIT   - STATIONNEMENT / IMMOBILISATION par arrêt (`departure`, `dwell_s`) : arrivée = premier
         ping dans la zone, départ = dernier ping consécutif encore dans la zone avant que le bus
         ne reparte, dwell_s = leur écart. Un long stationnement est un signal d'anomalie fort
         (panne / incident / arrêt non prévu). NOTE : nécessite de ré-exécuter build_foundation
         pour apparaître dans le jeu de données persisté.
À FAIRE - RETARD : retard = arrivée réelle - arrivée prévue. Nous avons `réelle` mais il n'y a
          pas d'horaires par arrêt — `ligne.horaires` ne stocke que les heures de DÉPART à
          l'origine. Construit dans 03_delay sur une base de référence pilotée par les données
          (médiane observée du temps écoulé jusqu'à l'arrêt) plutôt qu'un horaire officiel.
Ce module est la COUCHE prérequise ; le retard (03_delay) est la couche suivante au-dessus.

Hypothèses / limites
---------------------
- La géométrie de la ligne est constituée des ancres `array_lat/lng_opendata` ordonnées avec
  les espaces réservés `0.0` supprimés ; les routes nécessitent >= `min_anchors` vraies ancres.
  Avec des ancres clairsemées, la polyligne est une approximation grossière de la route,
  c'est pourquoi la segmentation utilise des oscillations de distance avec hystérésis plutôt
  qu'une correspondance cartographique exacte.
- `code` n'est pas unique entre entreprises ; toujours clé par `(code, societe)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pymongo.database import Database


# configuration
@dataclass(frozen=True)
class Config:
    # géométrie
    min_anchors: int = 4              # nombre min d'ancres réelles pour qu'une ligne soit utilisable
    # sélection des candidats
    min_pings: int = 300              # pings min pour qu'un (jour, ligne, bus) vaille la peine d'être traité
    first_usable_day: str = "d20220601"   # le lien de service (service.codeLigne) commence ~ici
    # nettoyage
    dedup_round: int = 6              # arrondi des coordonnées pour la suppression des doublons stationnaires
    signal_gap_s: int = 600           # annoter les écarts supérieurs à ceci
    # projection (correspondance cartographique)
    proj_window: int = 3              # fenêtre de recherche séquentielle (segments) autour du dernier match
    proj_gap_reset_s: int = 900       # après un écart supérieur à ceci, rechercher globalement
    smooth_window: int = 15           # fenêtre de médiane glissante sur la distance le long de la route
    # segmentation (invariante par rapport à la longueur de la route)
    reversal_frac: float = 0.15       # hystérésis de demi-tour en fraction de la longueur de la route
    reversal_floor_m: float = 2000.0
    min_span_frac: float = 0.06       # longueur min du trajet en fraction de la longueur de la route
    min_span_floor_m: float = 1500.0
    min_trip_min: float = 8.0         # durée min du trajet
    layover_gap_s: int = 2400         # diviser une course uniquement aux écarts >= ceci ...
    park_frac: float = 0.05           # ... et seulement si le bus a à peine bougé pendant l'écart
    full_frac: float = 0.10           # le trajet est "complet" s'il couvre les deux extrémités dans cette bande
    # accrochage des arrivées
    arrival_thresh_m: float = 350.0   # distance max ping-à-arrêt pour compter comme une arrivée

    @property
    def out_columns(self) -> list:
        return ["day", "line", "societe", "bus", "trip_id", "dir", "full",
                "trip_start", "trip_end", "seq", "route_seq", "stop",
                "arrival", "departure", "dwell_s", "dark_s", "had_gap",
                "dist_m", "matched"]


# géométrie
def haversine(lat1, lon1, lat2, lon2):
    """Distance orthodromique en mètres (scalaires ou tableaux numpy)."""
    R = 6371000.0
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def real_anchor_stops(ligne: dict) -> list:
    """Arrêts géocodés uniquement, dans l'ordre de la route, en conservant la position originale de chaque arrêt."""
    la = ligne.get("array_lat_opendata") or []
    lo = ligne.get("array_lng_opendata") or []
    names = ligne.get("stationnames") or []
    rows = []
    for i in range(min(len(la), len(lo))):
        try:
            lat, lon = float(la[i]), float(lo[i])
        except (TypeError, ValueError):
            continue
        if abs(lat) > 1 and abs(lon) > 1:                 # supprimer les espaces réservés 0.0
            rows.append({"route_seq": i,
                         "name": names[i] if i < len(names) else f"stop{i}",
                         "lat": lat, "lon": lon})
    return rows


def stops_frame(ligne: dict) -> pd.DataFrame:
    """Arrêts d'ancrage avec un `seq` compact et une distance cumulée le long de la route `s_m`."""
    rows = real_anchor_stops(ligne)
    st = pd.DataFrame(rows)
    seg = haversine(st["lat"].values[:-1], st["lon"].values[:-1],
                    st["lat"].values[1:], st["lon"].values[1:])
    st["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
    st.insert(0, "seq", range(len(st)))
    return st


def usable_geometry(ligne: dict, cfg: Config) -> bool:
    return len(real_anchor_stops(ligne)) >= cfg.min_anchors


# Per-company stop-list collections and their matching societe names in `ligne`.
_STOP_COL_MAP: dict[str, str] = {
    "S.R.T.K":       "STOPS.R.T.K",
    "S.R.T.M":       "STOPS.R.T.M",
    "S.R.T.BIZERTE": "STOPS.R.T.BIZERTE",
    "S.R.T.SELIANA": "STOPS.R.T.SELIANA",
    "TCV":           "STOPTCV",
    "SORETRAS":      "STOPSORETRAS",
    "SRT.ELGOUAFEL": "STOPSRT.ELGOUAFEL",
    "S.T.C.I":       "STOPS.T.C.I",
    "S.T.S":         "STOPS.T.S",
    "EPE-TVE":       "STOPEPE-TVE",
}


def _stop_station_frame(db: Database, code, societe: str) -> Optional[pd.DataFrame]:
    """Build a stops_frame from STOP<societe> (ordered) + station (coords).

    Returns None when the collection doesn't exist, the line isn't in it, or
    fewer than 4 geocoded stops are found — caller falls back to ligne anchors.
    """
    col_name = _STOP_COL_MAP.get(str(societe) if societe else "")
    if col_name is None:
        return None
    try:
        route_nr = int(code)
    except (TypeError, ValueError):
        return None

    stops_raw = list(db[col_name].find(
        {"ROUTENR": route_nr, "NAMENRnew": {"$exists": True}},
        {"STOPNR": 1, "NAMENRnew": 1, "_id": 0},
    ))
    if not stops_raw:
        return None

    stop_ids = [str(d["NAMENRnew"]) for d in stops_raw]
    station_docs = list(db["station"].find(
        {"stop_id": {"$in": stop_ids}, "societe": societe},
        {"stop_id": 1, "stop_lat": 1, "stop_lon": 1, "stop_name_fr": 1, "_id": 0},
    ))
    station_map: dict[str, tuple] = {}
    for s in station_docs:
        try:
            lat, lon = float(s["stop_lat"]), float(s["stop_lon"])
        except (TypeError, ValueError):
            continue
        if abs(lat) > 1 and abs(lon) > 1:
            station_map[str(s["stop_id"])] = (lat, lon, s.get("stop_name_fr") or "")

    rows = []
    for d in stops_raw:
        sid = str(d["NAMENRnew"])
        if sid not in station_map:
            continue
        lat, lon, name = station_map[sid]
        try:
            seq_n = int(str(d["STOPNR"]))
        except (TypeError, ValueError):
            continue
        rows.append({"route_seq": seq_n, "name": name or sid, "lat": lat, "lon": lon})

    if len(rows) < 4:
        return None

    rows.sort(key=lambda r: r["route_seq"])
    st = pd.DataFrame(rows)
    seg = haversine(st["lat"].values[:-1], st["lon"].values[:-1],
                    st["lat"].values[1:], st["lon"].values[1:])
    st["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
    st.insert(0, "seq", range(len(st)))
    return st


def build_usable_lines(db: Database, cfg: Config) -> dict:
    """Table {(code, societe) -> stops_frame} pour chaque ligne utilisable.

    Tries STOP<societe>+station first (ordered, named stops with real coords);
    falls back to ligne.array_lat_opendata anchors where that coverage is absent.
    """
    out: dict = {}
    for linge in db["ligne"].find({}):
        code = linge["code"]
        soc = linge.get("societe")
        sf = _stop_station_frame(db, code, soc)
        if sf is None:
            if not usable_geometry(linge, cfg):
                continue
            sf = stops_frame(linge)
        out[(str(code), soc)] = sf
    return out


# pings
def load_pings(gps_db: Database, day: str, line: str, bus) -> pd.DataFrame:
    """Un bus-jour pour une ligne. Requête filtrée + projetée = économe en mémoire.

    Vitesse : préférer `speed` au niveau supérieur ; utiliser `bus.vitesse` en secours uniquement
    quand absent (dans les données 2025+, `bus.vitesse` est souvent un 0 obsolète,
    donc `speed` fait autorité quand présent).
    """
    cur = gps_db[day].find(
        {"service.codeLigne": line, "bus.code": bus},
        {"date": 1, "localisation": 1, "speed": 1, "bus.vitesse": 1, "service.voyage": 1, "_id": 0},
    )
    rows = []
    for d in cur:
        loc = d.get("localisation") or {}
        b = d.get("bus") or {}
        svc = d.get("service") or {}
        speed = d.get("speed")
        if speed is None:
            speed = b.get("vitesse")
        rows.append({"t": d.get("date"), "lat": loc.get("x"),
                     "lon": loc.get("y"), "speed": speed,
                     "voyage": svc.get("voyage")})
    g = pd.DataFrame(rows)
    if len(g) == 0 or "t" not in g:
        return pd.DataFrame(columns=["t", "lat", "lon", "speed", "gap_s", "signal_gap", "voyage"])
    return g.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)


def clean_pings(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Supprime les pings consécutifs aux coordonnées identiques (spam stationnaire), en gardant le premier
    contact pour préserver l'horodatage d'arrivée ; annote `gap_s` et `signal_gap`."""
    if len(g) == 0:
        return g
    r = cfg.dedup_round
    same = (g["lat"].round(r).eq(g["lat"].round(r).shift())
            & g["lon"].round(r).eq(g["lon"].round(r).shift()))
    g = g[~same].reset_index(drop=True)
    g["gap_s"] = g["t"].diff().dt.total_seconds()
    g["signal_gap"] = g["gap_s"] > cfg.signal_gap_s
    return g


# --------------------------------------------------------------------------- projection
def project_to_route(g: pd.DataFrame, stops: pd.DataFrame, cfg: Config):
    """Correspondance cartographique séquentielle (fenêtrée) : distance le long de la route `s` et hors-route `off`.

    Contraindre le segment correspondant de chaque ping à une petite fenêtre autour du match
    précédent maintient `s` physiquement lisse malgré des ancres clairsemées.
    Retourne (g_with_s_off, route_len_m).
    """
    slat, slon = stops["lat"].values, stops["lon"].values
    cum = stops["s_m"].values
    mlat = 111320.0
    mlon = 111320.0 * np.cos(np.radians(slat.mean()))
    ax, ay = slon[:-1] * mlon, slat[:-1] * mlat
    bx, by = slon[1:] * mlon, slat[1:] * mlat
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    L2[L2 == 0] = 1e-9
    nseg = len(ax)

    px, py = g["lon"].values * mlon, g["lat"].values * mlat
    gap = g["gap_s"].values
    s = np.empty(len(g))
    off = np.empty(len(g))
    s_raw = np.empty(len(g))
    prevk: Optional[int] = None
    for i in range(len(g)):
        if prevk is None or (not np.isnan(gap[i]) and gap[i] > cfg.proj_gap_reset_s):
            cand = range(nseg)
        else:
            cand = range(max(0, prevk - cfg.proj_window), min(nseg, prevk + cfg.proj_window + 1))
        best_k, best_d, best_t = 0, np.inf, 0.0
        for k in cand:
            t = min(1.0, max(0.0, ((px[i] - ax[k]) * vx[k] + (py[i] - ay[k]) * vy[k]) / L2[k]))
            dx = px[i] - (ax[k] + t * vx[k])
            dy = py[i] - (ay[k] + t * vy[k])
            d = np.hypot(dx, dy)
            if d < best_d:
                best_d, best_k, best_t = d, k, t
        prevk = best_k
        s_raw[i] = cum[best_k] + best_t * (cum[best_k + 1] - cum[best_k])
        off[i] = best_d
    g = g.copy()
    g["s_raw"] = s_raw
    g["s"] = pd.Series(s_raw).rolling(cfg.smooth_window, center=True, min_periods=1).median()
    g["off_m"] = off
    return g, float(cum[-1])


# --------------------------------------------------------------------------- segmentation
def segment_trips(g: pd.DataFrame, route_len: float, cfg: Config) -> pd.DataFrame:
    """Segmentation par oscillation de direction. Capture les trajets complets et partiels ;
    divise une course à un écart uniquement quand le bus était stationné pendant celui-ci
    (une vraie pause entre courses)."""
    if len(g) < 5 or route_len <= 0:
        return pd.DataFrame()
    rev = max(cfg.reversal_floor_m, cfg.reversal_frac * route_len)
    min_span = max(cfg.min_span_floor_m, cfg.min_span_frac * route_len)
    park = cfg.park_frac * route_len
    s = g["s"].values
    s_raw = g["s_raw"].values
    tm = g["t"].values
    gp = g["gap_s"].values
    N = len(g)

    # pivots en zigzag avec hystérésis sur la distance le long de la route
    piv = [0]
    direction = 0
    ext = 0
    for i in range(1, N):
        if direction >= 0 and s[i] > s[ext]:
            ext = i
        elif direction <= 0 and s[i] < s[ext]:
            ext = i
        if direction >= 0 and s[i] <= s[ext] - rev:
            piv.append(ext); direction = -1; ext = i
        elif direction <= 0 and s[i] >= s[ext] + rev:
            piv.append(ext); direction = 1; ext = i
        elif direction == 0:
            if s[i] >= s[0] + rev:
                direction = 1; ext = i
            elif s[i] <= s[0] - rev:
                direction = -1; ext = i
    piv.append(N - 1)
    piv = sorted(set(piv))

    trips = []
    for a, b in zip(piv[:-1], piv[1:]):
        # diviser une course dans la même direction uniquement aux pauses stationnées
        cuts = [a] + [i for i in range(a + 1, b + 1)
                      if gp[i] > cfg.layover_gap_s and abs(s_raw[i] - s_raw[i - 1]) < park] + [b]
        for sa, se in zip(sorted(set(cuts))[:-1], sorted(set(cuts))[1:]):
            if se <= sa:
                continue
            span = abs(s[se] - s[sa])
            dur = (pd.Timestamp(tm[se]) - pd.Timestamp(tm[sa])).total_seconds() / 60
            if span < min_span or dur < cfg.min_trip_min:
                continue
            lo, hi = float(min(s[sa], s[se])), float(max(s[sa], s[se]))
            trips.append({
                "dir": "ALLER" if s[se] > s[sa] else "RETOUR",
                "start": pd.Timestamp(tm[sa]), "end": pd.Timestamp(tm[se]),
                "s_lo": lo, "s_hi": hi,
                "full": lo <= cfg.full_frac * route_len and hi >= (1 - cfg.full_frac) * route_len,
            })
    return pd.DataFrame(trips).reset_index(drop=True)


# --------------------------------------------------------------------------- arrivées
def derive_arrivals(g: pd.DataFrame, trip: pd.Series, stops: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Accroche les arrêts dans la plage couverte du trajet aux pings, dans l'ordre de voyage,
    en imposant des temps d'arrivée monotones. Retourne une ligne par arrêt couvert (correspondant ou non).

    Dérive aussi le STATIONNEMENT/IMMOBILISATION : `arrival` = premier ping dans la zone,
    `departure` = dernier ping consécutif encore dans la zone avant que le bus ne reparte,
    `dwell_s` = leur écart. Un long stationnement est un signal d'anomalie fort
    (panne / incident / arrêt non prévu).
    """
    seg = g[(g["t"] >= trip["start"]) & (g["t"] <= trip["end"])]
    if len(seg) == 0:
        return pd.DataFrame()
    lat = seg["lat"].values
    lon = seg["lon"].values
    t = seg["t"].values
    gap_flag = seg["signal_gap"].values                    # True = dark period before this ping
    gap_arr  = seg["gap_s"].fillna(0).values               # seconds of that dark period
    margin = cfg.arrival_thresh_m
    covered = stops[(stops["s_m"] >= trip["s_lo"] - margin) & (stops["s_m"] <= trip["s_hi"] + margin)]
    order = covered.sort_values("s_m", ascending=(trip["dir"] == "ALLER"))

    out = []
    ptr = 0                                   # imposer des arrivées monotones le long du trajet
    for _, st in order.iterrows():
        if ptr >= len(seg):
            d_arr, j_local, matched = np.inf, None, False
        else:
            d = haversine(lat[ptr:], lon[ptr:], st["lat"], st["lon"])
            j_local = int(np.argmin(d)) + ptr
            d_arr = float(d.min())
            matched = d_arr <= cfg.arrival_thresh_m

        departure, dwell_s, dark_s, had_gap = pd.NaT, None, 0.0, False
        if matched:
            # départ = dernier ping *consécutif* encore dans la zone, sans franchir un écart de signal.
            # Un écart de signal fait partie de `gap_flag`; on s'arrête avant lui pour ne pas
            # compter la durée de l'écart comme immobilisation (ce qui déclencherait des anomalies).
            d_fwd = haversine(lat[j_local:], lon[j_local:], st["lat"], st["lon"])
            within = d_fwd <= cfg.arrival_thresh_m
            last = 0
            for k in range(len(within)):
                if k > 0 and gap_flag[j_local + k]:   # signal perdu avant ce ping → s'arrêter
                    had_gap = True
                    dark_s = float(gap_arr[j_local + k])
                    break
                if within[k]:
                    last = k
                else:
                    break
            dep_idx = j_local + last
            departure = pd.Timestamp(t[dep_idx])
            dwell_s = round(float((t[dep_idx] - t[j_local]) / np.timedelta64(1, "s")), 1)
            ptr = dep_idx + 1
        out.append({
            "seq": int(st["seq"]),
            "route_seq": int(st["route_seq"]),
            "stop": str(st["name"])[:24],
            "arrival": pd.Timestamp(t[j_local]) if matched else pd.NaT,
            "departure": departure,
            "dwell_s": dwell_s,
            "dark_s": dark_s,
            "had_gap": had_gap,
            "dist_m": int(d_arr) if np.isfinite(d_arr) else None,
            "matched": bool(matched),
        })
    return pd.DataFrame(out)


def correct_direction_from_voyage(trips: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Override geometric ALLER/RETOUR using per-ping service.voyage parity.

    voyage even → ALLER, odd → RETOUR.  Falls back to geometric direction when
    voyage is absent (older GPS eras) or tied.
    """
    if "voyage" not in g.columns or g["voyage"].isna().all():
        return trips
    trips = trips.copy()
    for i, tr in trips.iterrows():
        seg_v = g[(g["t"] >= tr["start"]) & (g["t"] <= tr["end"])]["voyage"].dropna()
        if len(seg_v) == 0:
            continue
        seg_v = seg_v.astype(int)
        n_aller  = int((seg_v % 2 == 0).sum())
        n_retour = int((seg_v % 2 == 1).sum())
        if n_aller > n_retour:
            trips.at[i, "dir"] = "ALLER"
        elif n_retour > n_aller:
            trips.at[i, "dir"] = "RETOUR"
        # tie → keep geometric guess
    return trips


def reconstruct_bus_day(gps_db: Database, day: str, line: str, societe, bus,
                        stops: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pipeline complet pour un (jour, ligne, societe, bus). Retourne les lignes d'arrivée aux arrêts."""
    g = clean_pings(load_pings(gps_db, day, line, bus), cfg)
    if len(g) < 20:
        return pd.DataFrame()
    g, route_len = project_to_route(g, stops, cfg)
    trips = segment_trips(g, route_len, cfg)
    trips = correct_direction_from_voyage(trips, g)
    frames = []
    for tid, tr in trips.iterrows():
        a = derive_arrivals(g, tr, stops, cfg)
        if a.empty:
            continue
        a.insert(0, "day", day[1:]); a.insert(1, "line", line)
        a.insert(2, "societe", societe); a.insert(3, "bus", bus)
        a.insert(4, "trip_id", tid); a.insert(5, "dir", tr["dir"])
        a.insert(6, "full", bool(tr["full"]))
        a.insert(7, "trip_start", tr["start"]); a.insert(8, "trip_end", tr["end"])
        frames.append(a)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- candidats
def candidates_for_day(gps_db: Database, day: str, usable: dict, cfg: Config) -> list:
    """Couples distincts (jour, ligne, societe, bus) actifs ce jour sur des lignes à géométrie utilisable."""
    pipe = [{"$group": {"_id": {"l": "$service.codeLigne", "s": "$service.societe",
                                "b": "$bus.code"}, "n": {"$sum": 1}}}]
    out = []
    for a in gps_db[day].aggregate(pipe):
        l, s, b = a["_id"].get("l"), a["_id"].get("s"), a["_id"].get("b")
        if b is not None and a["n"] >= cfg.min_pings and (str(l), s) in usable:
            out.append((day, str(l), s, b))
    return out


def gps_days(gps_db: Database, cfg: Config) -> list:
    """Collections GPS journalières triées à partir du premier jour avec lien de service."""
    return sorted(x for x in gps_db.list_collection_names()
                  if re.fullmatch(r"d\d{8}", x) and x >= cfg.first_usable_day)
