
"""Reconstruction de trajets GPS — la couche de fondation partagée.

Transforme les pings GPS bruts (`Historique_pos`) en trajets reconstruits avec l'heure
d'arrivée réelle dérivée à chaque arrêt. Utilisée par la prédiction de retard (labels),
la détection d'anomalies et le repli GPS. Ce module est la source de vérité unique ;
le notebook et le CLI batch (`build_foundation.py`) importent tous deux depuis ici.

Comment la segmentation est réalisée (chaîne en 4 étapes)
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


"""
from __future__ import annotations

import re
import warnings
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
    geom_outlier_mult: float = 3.0    # une ancre est "pontée" (voir bridge_geometry_outliers) si SES
                                      # DEUX tronçons dépassent ce multiple de la médiane des tronçons
                                      # de la ligne -- confirmé sur S.R.T.K/217 : "EL GARAA" partage un
                                      # nom avec des arrêts d'~9 AUTRES lignes S.R.T.K et a hérité d'une
                                      # coordonnée juste pour quelques-unes, ~25-27km fausse pour la
                                      # plupart -- non corrigé, route_len gonfle de 55km/21.6% sur la
                                      # 217 SEULE, ce qui fausse TOUS les seuils dérivés de route_len
                                      # plus bas (full_frac/reversal_frac/min_span_frac) : un trajet
                                      # complet réel mesure alors court de la bande full_frac (gonflée)
                                      # et se retrouve étiqueté "partiel" -- exactement le symptôme
                                      # "trajets partiels qui polluent une ligne à trajets complets".
    geom_outlier_floor_m: float = 8000.0  # ... ET l'un des deux tronçons doit dépasser ce plancher
                                          # absolu, sinon une ligne courte/dense (médiane faible) se
                                          # déclencherait sur un espacement pourtant normal.
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
    loop_frac: float = 0.15           # 1er et dernier arrêt à moins de cette fraction de route_len -> route en boucle
    max_trip_h: float = 24.0          # au-delà : horodatages corrompus (pings d'UNE collection journalière), rejeté
    idle_trim_m: float = 350.0        # rayon "immobile" pour rogner le stationnement terminus des bornes du trajet
    split_stationary_h: float = 3.0   # bloc immobile plus long que ceci EN COURS de trajet = pause de service
                                      # (stationnement nocturne/dépôt), coupe le trajet -- un vrai incident
                                      # d'exploitation (panne 30-120 min) reste EN-DESSOUS et reste entier
    # accrochage des arrivées
    arrival_thresh_m: float = 350.0   # distance max ping-à-arrêt pour compter comme une arrivée

    @property
    def out_columns(self) -> list:
        return ["day", "line", "societe", "bus", "trip_id", "dir", "full",
                "trip_start", "trip_end", "seq", "route_seq", "stop",
                "arrival", "departure", "dwell_s", "dark_s", "had_gap",
                "dist_m", "matched", "terminus_idle_min",
                "origin_idle_min", "end_idle_min", "origin_idle_from", "end_idle_to",
                "origin_idle_stop", "end_idle_stop"]


# géométrie
def haversine(lat1, lon1, lat2, lon2):
    """Distance orthodromique en mètres (scalaires ou tableaux numpy)."""
    R = 6371000.0
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def bridge_geometry_outliers(lat: np.ndarray, lon: np.ndarray, seg: np.ndarray, cfg: Config,
                             max_passes: int = 5) -> np.ndarray:
    """Corrige `seg` (distances entre ancres consécutives, longueur = len(lat)-1) pour les
    ancres dont les DEUX tronçons environnants sont des valeurs aberrantes par rapport à
    l'espacement typique de la ligne -- voir `Config.geom_outlier_mult` pour le cas
    S.R.T.K/217 "EL GARAA" qui a motivé ce correctif : la coordonnée de cet arrêt est
    fausse pour la plupart des lignes qui le référencent (nom partagé, voir
    `get_coord_suspect` côté API), ce qui gonfle `route_len` de dizaines de km et fausse
    tous les seuils dérivés dans `segment_trips`.

    Une ancre interne k est "pontée" (ses deux tronçons remplacés par la moitié de la
    distance DIRECTE entre ses voisins, comme si elle était exactement entre eux) quand ses
    deux tronçons dépassent À LA FOIS `geom_outlier_mult` x la médiane des tronçons de la
    ligne ET le plancher absolu `geom_outlier_floor_m` -- le plancher évite qu'une ligne
    courte/dense (médiane faible) se déclenche sur un espacement pourtant normal.

    PLUSIEURS passes (jusqu'à `max_passes`) : deux ancres consécutives TOUTES DEUX fausses
    peuvent se masquer l'une l'autre au premier passage (le tronçon ENTRE elles n'est pas
    forcément un outlier si les deux sont égarées dans la même direction) -- constaté sur
    des lignes EPE-TVE où route_len restait à >1000 km après un seul passage (plusieurs
    ancres corrompues d'affilée) contre quelques dizaines de km attendues. Recalcule la
    médiane à chaque passage (sur les tronçons déjà pontés) et s'arrête dès qu'un passage
    ne change plus rien.
    """
    n = len(seg)
    if n < 3:
        return seg
    seg = seg.copy()
    for _ in range(max_passes):
        med = float(np.median(seg))
        thresh = max(cfg.geom_outlier_floor_m, cfg.geom_outlier_mult * med)
        changed = False
        for k in range(1, len(lat) - 1):           # ancres internes uniquement (pas les termini)
            if seg[k - 1] > thresh and seg[k] > thresh:
                direct = haversine(lat[k - 1], lon[k - 1], lat[k + 1], lon[k + 1])
                if not np.isclose(seg[k - 1], direct / 2.0) or not np.isclose(seg[k], direct / 2.0):
                    seg[k - 1] = direct / 2.0
                    seg[k] = direct / 2.0
                    changed = True
        if not changed:
            break
    return seg


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


def stops_frame(ligne: dict, cfg: Config = Config()) -> pd.DataFrame:
    """Arrêts d'ancrage avec un `seq` compact et une distance cumulée le long de la route `s_m`."""
    rows = real_anchor_stops(ligne)
    st = pd.DataFrame(rows)
    seg = haversine(st["lat"].values[:-1], st["lon"].values[:-1],
                    st["lat"].values[1:], st["lon"].values[1:])
    seg = bridge_geometry_outliers(st["lat"].values, st["lon"].values, seg, cfg)
    st["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
    st.insert(0, "seq", range(len(st)))
    return st


def usable_geometry(ligne: dict, cfg: Config) -> bool:
    return len(real_anchor_stops(ligne)) >= cfg.min_anchors


def detect_loop_route(stops: pd.DataFrame, cfg: Config) -> bool:
    """Repère une ligne en forme de boucle : premier et dernier arrêt physiquement proches
    (à moins de `loop_frac` de route_len l'un de l'autre) malgré une longue distance `s_m`
    entre eux dans l'ordre de la route.
    """
    if len(stops) < 2:
        return False
    route_len = float(stops["s_m"].iloc[-1])
    if route_len <= 0:
        return False
    first, last = stops.iloc[0], stops.iloc[-1]
    d = float(haversine(first["lat"], first["lon"], last["lat"], last["lon"]))
    return d <= cfg.loop_frac * route_len


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


def _stop_station_frame(db: Database, code, societe: str, cfg: Config) -> Optional[pd.DataFrame]:
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
    seg = bridge_geometry_outliers(st["lat"].values, st["lon"].values, seg, cfg)
    st["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
    st.insert(0, "seq", range(len(st)))
    return st


def build_usable_lines(db: Database, cfg: Config, ticket_index=None, od_dict=None,
                        tk_db: Optional[Database] = None, gps_db: Optional[Database] = None,
                        triangulate_gaps: bool = False) -> dict:
    """Table {(code, societe) -> stops_frame} pour chaque ligne utilisable.

    Ordre de priorité des sources de coordonnées :
    1. STOP<societe>+station (ordonné, coords géocodées — la source la plus fiable).
    2. Enrichissement OpenData+tickets (`stations.build_enriched_stops`), si `ticket_index`
       et `od_dict` sont fournis — récupère des lignes qui seraient sinon abandonnées.
    3. Ancres brutes `ligne.array_lat_opendata` (dernier recours, seulement si >= min_anchors).

    Les paramètres d'enrichissement sont optionnels (défaut None) : sans eux, le comportement
    est identique à l'ancienne signature (rétrocompatible).
    """
    out: dict = {}
    for linge in db["ligne"].find({}):
        code = linge["code"]
        soc = linge.get("societe")
        sf = _stop_station_frame(db, code, soc, cfg)
        if sf is None and ticket_index is not None and od_dict is not None:
            from src.data import stations as _st
            enriched = _st.build_enriched_stops(ticket_index, od_dict, code, soc,
                                                 tk_db=tk_db, gps_db=gps_db,
                                                 triangulate_gaps=triangulate_gaps)
            if not enriched.empty:
                sf = enriched
        if sf is None:
            if not usable_geometry(linge, cfg):
                continue
            sf = stops_frame(linge, cfg)
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


# projection
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


# segmentation
def segment_trips(g: pd.DataFrame, route_len: float, cfg: Config, is_loop: bool = False) -> pd.DataFrame:
    """Segmentation par oscillation de direction. Capture les trajets complets et partiels ;
    divise une course à un écart uniquement quand le bus était stationné pendant celui-ci
    (une vraie pause entre courses).

    `is_loop` (voir `detect_loop_route`) : pour une ligne en boucle, `full` basé sur route_len
    n'est PAS fiable -- observé sur TCV/3, la trace GPS oscille en boucles courtes sans jamais
    couvrir toute la géométrie, ce qui pourrait autant signifier « boucle courte complète »
    que « boucle longue abandonnée en route » : rien ne permet de trancher avec la géométrie
    actuelle. Plutôt que deviner (et risquer de réintroduire la surestimation de l'ancienne
    géométrie éparse), `full` est mis à None (inconnu) explicitement pour ces lignes.
    """
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
        # diviser une course dans la même direction : aux pauses stationnées pendant un
        # ÉCART de signal (comportement historique), ET aux blocs immobiles multi-heures
        # SANS écart de signal -- un bus garé la nuit au dépôt qui continue de pinger ne
        # produit aucun gap, et sans cette coupe le stationnement nocturne se fondait dans
        # le trajet (constaté : « trajet » de 15h23 sur une ligne à médiane 4h, dont ~13h
        # de stationnement nocturne pingé en milieu de fenêtre). Un vrai incident (panne
        # 30-120 min) reste sous `split_stationary_h` et reste un trajet entier -- c'est
        # exactement le signal que le module d'anomalies doit voir.
        cuts = {a, b}
        for i in range(a + 1, b + 1):
            if gp[i] > cfg.layover_gap_s and abs(s_raw[i] - s_raw[i - 1]) < park:
                cuts.add(i)
        # blocs immobiles (|Δs| < idle_trim_m entre pings consécutifs) plus longs que
        # split_stationary_h : couper au début ET à la fin du bloc pour que l'immobilité
        # n'appartienne à AUCUN des deux trajets
        run_start = a
        i = a + 1
        while i <= b:
            if abs(s_raw[i] - s_raw[run_start]) > cfg.idle_trim_m:
                # ping isolé (bruit GPS) si le suivant revient près de run_start : ne pas
                # casser un long bloc immobile pour un seul point aberrant (constaté : un
                # stationnement nocturne >3h scindé en deux moitiés <3h par un unique ping
                # à >350m qui revient à sa position juste après)
                if i + 1 <= b and abs(s_raw[i + 1] - s_raw[run_start]) <= cfg.idle_trim_m:
                    i += 1
                    continue
                block_min = (pd.Timestamp(tm[i - 1]) - pd.Timestamp(tm[run_start])).total_seconds() / 60
                if block_min > cfg.split_stationary_h * 60:
                    cuts.add(run_start)
                    cuts.add(i - 1)
                run_start = i
            i += 1
        block_min = (pd.Timestamp(tm[b]) - pd.Timestamp(tm[run_start])).total_seconds() / 60
        if block_min > cfg.split_stationary_h * 60:
            cuts.add(run_start)

        for sa, se in zip(sorted(cuts)[:-1], sorted(cuts)[1:]):
            if se <= sa:
                continue
            # Rogner le stationnement terminus des deux bornes : départ = dernier ping
            # encore au point de départ, arrivée = premier ping au point d'arrivée. Sans
            # ça, le temps garé au terminus origine (le bus pinge mais n'est pas parti)
            # compte dans la durée du trajet (constaté : « trajet » de 235 min dont 170 min
            # garé au terminus SOUSSE avant le vrai départ).
            i, j = sa, se
            while i + 1 <= se and abs(s[i + 1] - s[sa]) < cfg.idle_trim_m:
                i += 1
            while j - 1 >= i and abs(s[j - 1] - s[se]) < cfg.idle_trim_m:
                j -= 1
            if j <= i:
                continue
            span = abs(s[j] - s[i])
            dur = (pd.Timestamp(tm[j]) - pd.Timestamp(tm[i])).total_seconds() / 60
            if span < min_span or dur < cfg.min_trip_min:
                continue
            # > max_trip_h = bug de données prouvé, pas un vrai trajet lent : les pings
            # viennent d'UNE collection journalière, donc une étendue pareille implique des
            # horodatages corrompus (constaté : « trajet » de 4 668h dans d20230603).
            # Ceinture+bretelles avec le filtre de fenêtre-jour dans reconstruct_bus_day.
            if dur > cfg.max_trip_h * 60:
                continue
            lo, hi = float(min(s[i], s[j])), float(max(s[i], s[j]))
            full = (None if is_loop else
                    lo <= cfg.full_frac * route_len and hi >= (1 - cfg.full_frac) * route_len)
            # Le stationnement rogné n'est PAS jeté : il devient une caractéristique du
            # trajet (« bus resté X min au terminus avant de partir ») -- signal
            # opérationnel réel (service non clôturé) que le module d'anomalies doit voir,
            # sans plus gonfler la durée du trajet.
            # exclure le temps traceur-silencieux (gp > layover_gap_s) du stationnement :
            # sinon un cut posé juste après un trou de signal compte ce trou comme de
            # l'attente "active" au terminus, alors qu'il est déjà représenté par
            # max_dark_s -- double comptage du même trou sous deux features différentes
            def _dark_min(lo: int, hi: int) -> float:
                return sum(gp[k] / 60.0 for k in range(lo + 1, hi + 1) if gp[k] > cfg.layover_gap_s)
            origin_idle = max(0.0, (pd.Timestamp(tm[i]) - pd.Timestamp(tm[sa])).total_seconds() / 60
                               - _dark_min(sa, i))
            end_idle = max(0.0, (pd.Timestamp(tm[se]) - pd.Timestamp(tm[j])).total_seconds() / 60
                            - _dark_min(j, se))
            trips.append({
                "dir": "ALLER" if s[j] > s[i] else "RETOUR",
                "start": pd.Timestamp(tm[i]), "end": pd.Timestamp(tm[j]),
                "s_lo": lo, "s_hi": hi,
                "full": full,
                "terminus_idle_min": round(origin_idle + end_idle, 1),
                # Séparés (pas seulement sommés) + horodatages réels + position -- pour que le
                # diagnostic puisse nommer LE terminus concerné (voir `_nearest_stop_name` dans
                # reconstruct_bus_day) plutôt que rapporter un seul nombre sans repère (constaté :
                # « stationnement terminus 84 min » sans dire où, alors que c'est nommable -- ici
                # MAHDIA, dernier arrêt de la ligne). `s_start`/`s_end` sont les positions RÉELLES
                # de début/fin de trajet (contrairement à s_lo/s_hi qui sont min/max, donc perdent
                # l'info de quel côté est le début quand le trajet est RETOUR).
                "origin_idle_min": round(origin_idle, 1),
                "origin_idle_from": pd.Timestamp(tm[sa]),
                "end_idle_min": round(end_idle, 1),
                "end_idle_to": pd.Timestamp(tm[se]),
                "s_start": float(s[i]), "s_end": float(s[j]),
            })
    return pd.DataFrame(trips).reset_index(drop=True)


# arrivées
def derive_arrivals(g: pd.DataFrame, trip: pd.Series, stops: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Accroche les arrêts dans la plage couverte du trajet aux pings, dans l'ordre de voyage,
    en imposant des temps d'arrivée monotones. Retourne une ligne par arrêt couvert (correspondant ou non).
    """
    seg = g[(g["t"] >= trip["start"]) & (g["t"] <= trip["end"])]
    if len(seg) == 0:
        return pd.DataFrame()
    lat = seg["lat"].values
    lon = seg["lon"].values
    t = seg["t"].values
    gap_flag = seg["signal_gap"].values                    # True = dark period before this ping
    gap_arr  = seg["gap_s"].fillna(0).values               # seconds of that dark period
    # Plus grand trou de signal du TRAJET ENTIER, indépendant des arrêts -- le scan par-arrêt
    # plus bas ne détecte un trou QUE s'il interrompt l'attente après un arrêt déjà matché ; un
    # trou qui survient EN ROUTE (le bus quitte un arrêt matché, puis le traceur se tait avant
    # qu'aucun autre arrêt suivant ne soit jamais matché) reste invisible à `dark_s`/`had_gap`
    # ci-dessous, alors qu'il peut être LE signal dominant du trajet (constaté : trou de 4h41 sur
    # 230km avec dark_s=0 sur tous les arrêts -- le trajet était classé "mauvais suivi GPS" au
    # lieu de "trou de signal", et invisible au modèle puisque max_dark_s ressortait à 0).
    trip_dark_idx = int(np.argmax(gap_arr)) if len(gap_arr) else None
    trip_dark_s = float(gap_arr[trip_dark_idx]) if trip_dark_idx is not None else 0.0
    trip_dark_t = pd.Timestamp(t[trip_dark_idx]) if trip_dark_idx is not None and trip_dark_s > 0 else None
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
            # Premier passage dans le rayon, pas le minimum global : le minimum global
            # peut tomber tard dans le trajet (bruit GPS près des terminus, géométrie
            # en aller-retour) et faire sauter `ptr` au-delà des pings nécessaires aux
            # arrêts suivants — qui deviennent alors tous « non desservis » à tort.
            within = np.where(d <= cfg.arrival_thresh_m)[0]
            if len(within):
                k0 = int(within[0])                     # premier ping dans le rayon
                k1 = k0                                 # fin de la fenêtre contiguë
                while k1 + 1 < len(d) and d[k1 + 1] <= cfg.arrival_thresh_m:
                    k1 += 1
                k_best = k0 + int(np.argmin(d[k0:k1 + 1]))   # ping le plus proche de CE passage
                j_local = k_best + ptr
                d_arr = float(d[k_best])
                matched = True
            else:
                j_local = None
                d_arr = float(d.min())
                matched = False

        departure, dwell_s, dark_s, had_gap = pd.NaT, None, 0.0, False
        if matched:
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
    if not out:
        return pd.DataFrame(out)
    # Encadre le plus grand trou EN ROUTE (voir trip_dark_s ci-dessus) par les arrêts matchés
    # juste avant/après, pour une puce lisible (« perte de signal entre X et Y ») plutôt qu'un
    # simple chiffre sans repère géographique.
    before_stop = after_stop = None
    if trip_dark_t is not None:
        matched_rows = [r for r in out if r["matched"]]
        before = [r for r in matched_rows if pd.notna(r["departure"]) and r["departure"] <= trip_dark_t]
        after = [r for r in matched_rows if pd.notna(r["arrival"]) and r["arrival"] >= trip_dark_t]
        before_stop = before[-1]["stop"] if before else None
        after_stop = after[0]["stop"] if after else None
    for r in out:
        r["trip_dark_s"] = trip_dark_s
        r["trip_dark_before_stop"] = before_stop
        r["trip_dark_after_stop"] = after_stop
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


def _nearest_stop_name(stops: pd.DataFrame, s_val: float) -> str:
    """Nom de l'arrêt d'ancrage le plus proche d'une position `s` (mètres le long de la
    route) -- utilisé pour nommer le terminus concerné par un stationnement plutôt que de
    rapporter juste une durée sans repère (voir `segment_trips`, `s_start`/`s_end`)."""
    idx = (stops["s_m"] - s_val).abs().idxmin()
    return str(stops.loc[idx, "name"])


def reconstruct_bus_day(gps_db: Database, day: str, line: str, societe, bus,
                        stops: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Pipeline complet pour un (jour, ligne, societe, bus). Retourne les lignes d'arrivée aux arrêts."""
    raw = load_pings(gps_db, day, line, bus)
    if len(raw):
        # Rejette les pings dont l'horodatage tombe loin du jour calendaire de la collection
        # (horloges d'appareil corrompues -- constaté : pings estampillés 6 MOIS plus tard
        # dans d20230603, produisant un « trajet » de 4 668h). Marge -2h/+30h : tolère la
        # dérive d'ingestion autour de minuit, rejette les aberrations franches.
        day0 = pd.Timestamp(day[1:])
        in_window = (raw["t"] >= day0 - pd.Timedelta(hours=2)) & \
                    (raw["t"] <= day0 + pd.Timedelta(hours=30))
        raw = raw[in_window]
    g = clean_pings(raw, cfg)
    if len(g) < 20:
        return pd.DataFrame()
    g, route_len = project_to_route(g, stops, cfg)
    is_loop = detect_loop_route(stops, cfg)
    trips = segment_trips(g, route_len, cfg, is_loop=is_loop)
    trips = correct_direction_from_voyage(trips, g)
    frames = []
    for tid, tr in trips.iterrows():
        a = derive_arrivals(g, tr, stops, cfg)
        if a.empty:
            continue
        a.insert(0, "day", day[1:]); a.insert(1, "line", line)
        a.insert(2, "societe", societe); a.insert(3, "bus", bus)
        a.insert(4, "trip_id", tid); a.insert(5, "dir", tr["dir"])
        # None = « inconnu » pour une ligne en boucle (voir segment_trips) -- NE PAS caster en
        # bool(), ça convertirait silencieusement None -> False et annulerait le signal
        a.insert(6, "full", tr["full"] if pd.isna(tr["full"]) or tr["full"] is None else bool(tr["full"]))
        a.insert(7, "trip_start", tr["start"]); a.insert(8, "trip_end", tr["end"])
        a["terminus_idle_min"] = float(tr.get("terminus_idle_min", 0.0) or 0.0)
        # Stationnement terminus détaillé : séparé origine/fin (pas juste sommé) + horodatage
        # réel + NOM du terminus concerné -- répond à « 84 min stationné où ? » et « à quelle
        # heure le bus a-t-il vraiment démarré ? » plutôt qu'un seul chiffre sans repère.
        a["origin_idle_min"] = float(tr.get("origin_idle_min", 0.0) or 0.0)
        a["end_idle_min"] = float(tr.get("end_idle_min", 0.0) or 0.0)
        a["origin_idle_from"] = tr.get("origin_idle_from", pd.NaT)
        a["end_idle_to"] = tr.get("end_idle_to", pd.NaT)
        a["origin_idle_stop"] = (_nearest_stop_name(stops, tr["s_start"])
                                 if pd.notna(tr.get("s_start")) else None)
        a["end_idle_stop"] = (_nearest_stop_name(stops, tr["s_end"])
                              if pd.notna(tr.get("s_end")) else None)
        frames.append(a)
    if not frames:
        return pd.DataFrame()
    # certains trajets produisent des colonnes entièrement NA (ex. "full" pour une ligne en
    # boucle -- voir le commentaire sur is_full plus haut) ; anodin ici, seul le futur
    # comportement de concat change, pas le résultat actuel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat(frames, ignore_index=True)


# candidats
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
