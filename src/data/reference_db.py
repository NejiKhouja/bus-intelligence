"""Base de données de référence canonique (SQLite) — la couche « propre » construite au-dessus
des sources MongoDB éparpillées (`winicari`, `Historique_Tickets`, `OpenData`, `Historique_pos`).

Contrairement à la fondation GPS (`foundation.py`), qui reste un pipeline dérivé des pings bruts
et n'est PAS reconstruite, cette base couvre les données de RÉFÉRENCE (sociétés, arrêts, lignes)
qui souffrent d'incohérences entre collections (noms de société différents selon la source,
mêmes noms d'arrêts désignant des lieux physiques différents, etc.).

Schéma complet créé dès le départ (voir SCHEMA_SQL) pour que les tables futures s'intègrent
proprement, mais chaque table n'est peuplée qu'au moment où on la construit explicitement —
on commence par `companies`.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "reference" / "winicari_reference.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    company_id     INTEGER PRIMARY KEY,
    canonical_name TEXT UNIQUE NOT NULL,
    aliases        TEXT,                        -- JSON: variantes brutes vues dans les sources
    has_gps        BOOLEAN,                      -- actif dans la fenêtre RÉCENTE (30j) -- pas un fait figé
    gps_first_seen TEXT,                         -- premier jour GPS jamais observé (échantillon historique)
    gps_last_seen  TEXT,                         -- dernier jour GPS jamais observé
    gps_days_sampled INTEGER,                    -- nb de jours échantillonnés où la société est présente
    nom_complet    TEXT,                         -- winicari.societe.nomComplet
    gouvernorat    TEXT,                         -- winicari.societe.Gouvernorat
    active_plateforme BOOLEAN,                   -- winicari.societe.active -- statut ADMIN, distinct de has_gps
    notes          TEXT,                         -- ex. absente de winicari.societe malgré activité réelle ailleurs
    last_synced_at TEXT
);

CREATE TABLE IF NOT EXISTS stops (
    stop_id      INTEGER PRIMARY KEY,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    primary_name TEXT NOT NULL,
    aliases      TEXT,
    source       TEXT,
    confidence   TEXT,
    notes        TEXT              -- ex. coordonnée réelle mais nom probablement incorrect
);

CREATE TABLE IF NOT EXISTS lines (
    line_id    INTEGER PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    line_code  TEXT NOT NULL,
    source     TEXT,                     -- ligne / ticket_only / gps_only
    notes      TEXT,                      -- e.g. low ticket volume, code pattern looks admin/special
    UNIQUE(company_id, line_code)
);

CREATE TABLE IF NOT EXISTS line_stops (
    line_id           INTEGER REFERENCES lines(line_id),
    stop_id           INTEGER REFERENCES stops(stop_id),
    seq               INTEGER NOT NULL,
    company_stop_code TEXT,
    PRIMARY KEY (line_id, seq)
);

CREATE TABLE IF NOT EXISTS trips (
    trip_id           INTEGER PRIMARY KEY,
    line_id           INTEGER REFERENCES lines(line_id),
    company_id        INTEGER REFERENCES companies(company_id),
    bus               TEXT NOT NULL,
    day               TEXT NOT NULL,
    dir               TEXT,
    is_full           BOOLEAN,
    trip_start        TEXT,
    trip_end          TEXT,
    n_stops           INTEGER,
    match_rate        REAL,
    max_dwell_s       REAL,
    mean_dwell_s      REAL,
    total_elapsed_min REAL,
    dist_m_max        REAL,
    max_dark_s        REAL,
    terminus_idle_min REAL,
    trip_dark_before_stop TEXT,
    trip_dark_after_stop  TEXT,
    origin_idle_min   REAL,
    origin_idle_from  TEXT,
    origin_idle_stop  TEXT,
    end_idle_min      REAL,
    end_idle_to       TEXT,
    end_idle_stop     TEXT
);

CREATE TABLE IF NOT EXISTS trip_stops (
    trip_id    INTEGER REFERENCES trips(trip_id),
    stop_id    INTEGER REFERENCES stops(stop_id),
    seq        INTEGER NOT NULL,
    arrival    TEXT,
    departure  TEXT,
    dwell_s    REAL,
    dist_m     REAL,
    matched    BOOLEAN,
    dark_s     REAL,
    had_gap    BOOLEAN,
    PRIMARY KEY (trip_id, seq)
);

CREATE TABLE IF NOT EXISTS tickets_daily (
    company_id INTEGER REFERENCES companies(company_id),
    line_id    INTEGER REFERENCES lines(line_id),
    bus        TEXT,
    day        TEXT,
    nbr_ticket INTEGER,
    recette    REAL,
    PRIMARY KEY (company_id, line_id, bus, day)
);

-- Billetterie PAR ARRÊT D'ORIGINE (Phase 2 de la détection d'anomalies billetterie, voir
-- docs/WEBSERVICES_NEEDED.md -- service 3). Source : tickets INDIVIDUELS
-- (Historique_Tickets.Ticket{annee}, ~5,5M documents), PAS `winicari.details` (qui n'a pas
-- de grain arrêt) -- voir populate_tickets_station_daily. `station_name` = NomFR1 du ticket
-- (nom lisible de l'arrêt d'origine), pas un stop_id résolu -- la résolution vers `stops`
-- (pour la carte) se fait par nom au moment de servir, en best-effort (voir
-- api.main._resolve_station_coords), certains noms peuvent ne pas correspondre.
CREATE TABLE IF NOT EXISTS tickets_station_daily (
    company_id   INTEGER REFERENCES companies(company_id),
    line_id      INTEGER REFERENCES lines(line_id),
    bus          TEXT,
    station_name TEXT,
    day          TEXT,
    nbr_ticket   INTEGER,
    recette      REAL,
    PRIMARY KEY (company_id, line_id, bus, station_name, day)
);

-- Répartition ALLER/RETOUR de la même billetterie, voir populate_tickets_station_trip_daily.
-- Table SÉPARÉE de tickets_station_daily -- PAS lue par le modèle d'anomalie billetterie
-- (qui reste au grain bus-jour, décision utilisateur 2026-07-11 : découper l'ANOMALIE par
-- trajet est un changement bien plus gros et casse tout bus-jour dont l'appareil ne
-- distingue pas les trajets -- voir la colonne `direction`) -- sert UNIQUEMENT à
-- l'affichage du détail par arrêt (« voir le détail par arrêt » côté dashboard), pour
-- éviter de mélanger dans une même table un aller et un retour qui n'ont pas la même
-- pause terminus/normale. `direction` dérivée de la PARITÉ de `voyage` sur le ticket
-- (pair -> ALLER, impair -> RETOUR) -- MÊME convention que `foundation.
-- correct_direction_from_voyage` côté GPS (déjà validée), pas une règle inventée pour
-- l'occasion. Vaut 'UNKNOWN' quand `voyage` est absent/toujours à 0 sur tout le
-- bus-jour (l'appareil ne le suit pas) -- l'API/dashboard replient alors sur la vue
-- combinée existante (tickets_station_daily) plutôt que d'afficher un faux ALLER seul.
CREATE TABLE IF NOT EXISTS tickets_station_trip_daily (
    company_id   INTEGER REFERENCES companies(company_id),
    line_id      INTEGER REFERENCES lines(line_id),
    bus          TEXT,
    direction    TEXT,
    station_name TEXT,
    day          TEXT,
    nbr_ticket   INTEGER,
    recette      REAL,
    PRIMARY KEY (company_id, line_id, bus, direction, station_name, day)
);

-- Sessions chauffeur (ouverture/fermeture), voir populate_driver_services. Source : tickets
-- INDIVIDUELS (Historique_Tickets.Ticket{annee}), PAS la collection `service` de MongoDB --
-- celle-ci ne retient QUE les sessions actuellement ouvertes (14 documents au total au
-- 2026-07-15, toutes current=True), aucun historique de fermeture n'y est conservé, donc
-- inutilisable pour du rattachement rétroactif. `service_start`/`service_end` sont dérivées
-- de `date_debut_service`+`heure_debut_ticket` / `date_fin_service`+`heure_fin_ticket`,
-- champs dénormalisés identiques sur chaque ticket de la même session -- une approximation
-- (bornée par le premier/dernier ticket ÉMIS, pas l'instant exact d'ouverture/fermeture)
-- mais la seule source qui existe.
CREATE TABLE IF NOT EXISTS driver_services (
    company_id    INTEGER REFERENCES companies(company_id),
    line_id       INTEGER REFERENCES lines(line_id),
    bus           TEXT,
    day           TEXT,
    driver_code   TEXT,
    service_start TEXT,
    service_end   TEXT,
    n_tickets     INTEGER,
    PRIMARY KEY (company_id, line_id, bus, day, driver_code)
);

CREATE TABLE IF NOT EXISTS anomaly_scores (
    trip_id         INTEGER REFERENCES trips(trip_id),
    model_version   TEXT NOT NULL,
    if_score        REAL,
    is_anomaly_if   BOOLEAN,
    lstm_error      REAL,
    is_anomaly_lstm BOOLEAN,
    computed_at     TEXT,
    PRIMARY KEY (trip_id, model_version)
);
"""


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Crée le fichier SQLite et le schéma complet (idempotent) ; retourne la connexion."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    # migration idempotente : colonne ajoutée 2026-07-07 (stationnement terminus rogné,
    # voir foundation.segment_trips) -- CREATE IF NOT EXISTS n'altère pas une table existante
    try:
        conn.execute("ALTER TABLE trips ADD COLUMN terminus_idle_min REAL")
    except sqlite3.OperationalError:
        pass  # colonne déjà présente
    # migration idempotente : colonnes ajoutées 2026-07-09 (repère du trou de signal EN ROUTE
    # le plus important du trajet, voir foundation.derive_arrivals -- max_dark_s pouvait
    # rester à 0 sur un trajet contenant plusieurs heures de silence GPS)
    for col in ("trip_dark_before_stop", "trip_dark_after_stop"):
        try:
            conn.execute(f"ALTER TABLE trips ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass  # colonne déjà présente
    # migration idempotente : colonnes ajoutées 2026-07-10 (stationnement terminus détaillé
    # origine/fin -- voir foundation.segment_trips) -- séparé de terminus_idle_min (qui reste
    # la SOMME) pour pouvoir nommer LEQUEL des deux termini et donner l'heure réelle de
    # départ/arrivée dans le diagnostic, au lieu d'un seul chiffre sans repère
    for col in ("origin_idle_min", "end_idle_min"):
        try:
            conn.execute(f"ALTER TABLE trips ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # colonne déjà présente
    for col in ("origin_idle_from", "origin_idle_stop", "end_idle_to", "end_idle_stop"):
        try:
            conn.execute(f"ALTER TABLE trips ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass  # colonne déjà présente
    # migration idempotente : colonne `bus` ajoutée à tickets_station_daily le 2026-07-11
    # (voir populate_tickets_station_daily) -- change la CLÉ PRIMAIRE, pas ALTER-able en
    # SQLite. La table est entièrement dérivée/régénérée par populate_tickets_station_daily
    # (DELETE + réinsertion complète), donc un DROP est sans perte -- CREATE TABLE IF NOT
    # EXISTS juste avant ne recrée PAS une table déjà existante avec l'ancien schéma.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tickets_station_daily)")}
    if cols and "bus" not in cols:
        conn.execute("DROP TABLE tickets_station_daily")
        conn.executescript(SCHEMA_SQL)
    # Index ajoutés 2026-07-13 pour query_foundation_slice() : les endpoints qui chargeaient
    # foundation_arrivals_full.parquet en entier interrogent maintenant trips/trip_stops à la
    # demande (voir la note dans query_foundation_slice) -- sans ces index, chaque requête
    # scannerait les 47k lignes de `trips` en entier.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trips_scope ON trips(company_id, line_id, bus, day)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trips_line ON trips(company_id, line_id)")
    # migration idempotente : colonne ajoutée 2026-07-15 (rattachement chauffeur, voir
    # populate_driver_services/attach_driver_codes_to_trips) -- NULL pour tout trajet sans
    # session chauffeur correspondante (pas de tickets Historique_Tickets couvrant ce
    # bus-jour, ou hors de la fenêtre d'années ingérée).
    try:
        conn.execute("ALTER TABLE trips ADD COLUMN driver_code TEXT")
    except sqlite3.OperationalError:
        pass  # colonne déjà présente
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trips_driver ON trips(driver_code)")
    conn.commit()
    return conn


# Table 1 — companies
# Regroupement canonique décidé manuellement après audit des variantes brutes
# (voir notebook/scratch d'inventaire) : un seul cas ambigu (Winicari/winicari,
# tranché par l'utilisateur comme une même entité) + un artefact billetterie
# (S.R.T.K0 : mêmes numéros de série d'appareil BS-18-000x et mêmes lignes que
# S.R.T.K -- fusionné comme alias, pas une société distincte).
CANONICAL_COMPANIES: dict[str, list[str]] = {
    "EPE-TVE":       ["EPE-TVE"],
    "S.R.T.BIZERTE": ["S.R.T.BIZERTE"],
    "S.R.T.K":       ["S.R.T.K", "S.R.T.K0"],
    "S.R.T.M":       ["S.R.T.M"],
    "S.R.T.SELIANA": ["S.R.T.SELIANA"],
    "S.T.C.I":       ["S.T.C.I"],
    "S.T.S":         ["S.T.S"],
    "SORETRAS":      ["SORETRAS"],
    "SRT.ELGOUAFEL": ["SRT.ELGOUAFEL"],
    "TCV":           ["TCV"],
    "TUS":           ["TUS"],
    "Winicari":      ["Winicari", "winicari"],
}


def _gps_history(gps_db, canonical_companies: dict[str, list[str]],
                  sample_stride: int = 3, recent_n_days: int = 30) -> dict[str, dict]:
    """Scanne TOUT l'historique GPS (pas seulement les derniers jours) pour chaque société.

    Un booléen "a du GPS" figé sur une fenêtre récente est trompeur : une société peut avoir eu
    une couverture GPS réelle et soutenue (ex. S.R.T.BIZERTE : ~11 mois, mai 2022 -> avril 2023)
    puis avoir été débranchée -- une fenêtre de 10 jours la classerait à tort comme "jamais de GPS".
    On échantillonne 1 jour sur `sample_stride` sur toute la plage pour capturer ces fenêtres
    historiques, et on calcule séparément un indicateur "actif récemment" (30 derniers jours).

    Retourne {canonical_name -> {first_seen, last_seen, days_seen, has_gps_recent}}.
    """
    import re
    days = sorted(n for n in gps_db.list_collection_names() if re.fullmatch(r"d\d{8}", n))
    recent_days = days[-recent_n_days:]
    sample = sorted(set(days[::sample_stride]) | set(recent_days))
    recent = set(recent_days)

    alias_to_canonical = {alias: canon for canon, aliases in canonical_companies.items() for alias in aliases}
    hist = {canon: {"first_seen": None, "last_seen": None, "days_seen": 0, "has_gps_recent": False}
            for canon in canonical_companies}

    for day in sample:
        seen_raw = set(s for s in gps_db[day].distinct("service.societe") if s)
        for raw in seen_raw:
            canon = alias_to_canonical.get(raw)
            if canon is None:
                continue
            h = hist[canon]
            h["days_seen"] += 1
            if h["first_seen"] is None:
                h["first_seen"] = day
            h["last_seen"] = day
            if day in recent:
                h["has_gps_recent"] = True
    return hist


def populate_companies(conn: sqlite3.Connection, gps_db=None,
                        canonical_companies: dict[str, list[str]] = None) -> None:
    """Peuple `companies` à partir du regroupement canonique. Idempotent (REPLACE).

    `has_gps` reflète l'activité RÉCENTE (30j) ; `gps_first_seen`/`gps_last_seen` couvrent
    tout l'historique 2022-2026 échantillonné, pour ne pas rater une société débranchée
    (voir `_gps_history`).
    """
    canonical_companies = canonical_companies or CANONICAL_COMPANIES
    hist = ({canon: {"first_seen": None, "last_seen": None, "days_seen": 0, "has_gps_recent": False}
             for canon in canonical_companies} if gps_db is None
            else _gps_history(gps_db, canonical_companies))
    now = datetime.now(timezone.utc).isoformat()

    rows = [
        (canon, json.dumps(aliases, ensure_ascii=False), hist[canon]["has_gps_recent"],
         hist[canon]["first_seen"], hist[canon]["last_seen"], hist[canon]["days_seen"], now)
        for canon, aliases in canonical_companies.items()
    ]
    conn.executemany(
        """INSERT INTO companies (canonical_name, aliases, has_gps, gps_first_seen, gps_last_seen,
                                   gps_days_sampled, last_synced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(canonical_name) DO UPDATE SET
             aliases=excluded.aliases, has_gps=excluded.has_gps,
             gps_first_seen=excluded.gps_first_seen, gps_last_seen=excluded.gps_last_seen,
             gps_days_sampled=excluded.gps_days_sampled, last_synced_at=excluded.last_synced_at""",
        rows,
    )
    conn.commit()


def enrich_companies_from_societe(conn: sqlite3.Connection, wi_db,
                                   canonical_companies: dict[str, list[str]] = None) -> None:
    """Ajoute nom_complet/gouvernorat/active_plateforme depuis `winicari.societe` -- registre
    admin réel (nomComplet, Gouvernorat, statut actif de la plateforme), distinct de `has_gps`
    (présence GPS mesurée) qu'on avait déjà. Les deux se recoupent bien : S.R.T.BIZERTE/TUS/
    EPE-TVE sont `active=False` ici ET ont une fenêtre GPS historique désormais arrêtée --
    signal cohérent, pas contradictoire.

    SORETRAS et S.T.C.I n'ont AUCUNE entrée dans `winicari.societe` malgré une activité réelle
    ailleurs (lignes, tickets, sav) -- flagué en `notes` plutôt que masqué.
    """
    canonical_companies = canonical_companies or CANONICAL_COMPANIES
    alias_to_canonical = {alias.lower(): canon for canon, aliases in canonical_companies.items() for alias in aliases}

    matched_canon = set()
    for d in wi_db["societe"].find({}, {"Nom": 1, "nomComplet": 1, "Gouvernorat": 1, "active": 1, "_id": 0}):
        canon = alias_to_canonical.get(str(d.get("Nom", "")).lower())
        if canon is None:
            continue
        matched_canon.add(canon)
        active = str(d.get("active", "")).strip().lower() == "true"
        conn.execute(
            """UPDATE companies SET nom_complet = ?, gouvernorat = ?, active_plateforme = ?
               WHERE canonical_name = ?""",
            (d.get("nomComplet"), d.get("Gouvernorat"), active, canon),
        )

    missing = set(canonical_companies) - matched_canon
    for canon in missing:
        conn.execute(
            """UPDATE companies SET notes = ? WHERE canonical_name = ?""",
            ("Absente de winicari.societe (registre admin) malgré une activité réelle "
             "dans ligne/tickets/sav -- statut plateforme inconnu.", canon),
        )
    conn.commit()
    print(f"Enrichies depuis winicari.societe : {len(matched_canon)} ; absentes (flaguées) : {sorted(missing)}")


# Table 2 — lines
# Codes qui, par leur forme (bloc 900-999) ou leur volume quasi nul, ressemblent à des
# codes administratifs/spéciaux (charter, test, service non régulier) plutôt qu'à de
# vraies lignes numérotées -- gardés (l'utilisateur veut TOUTES les lignes) mais annotés
# pour rester visibles/filtrables en aval plutôt que silencieusement acceptés comme lignes normales.
def _line_note(code: str, n_tickets: int) -> str | None:
    if re.fullmatch(r"9\d{2}", code):
        return f"code motif 9XX -- probable service administratif/spécial (n_tickets={n_tickets})"
    if n_tickets is not None and n_tickets <= 2:
        return f"volume de tickets quasi nul (n={n_tickets}) -- possible bruit/typo"
    return None


def populate_lines(conn: sqlite3.Connection, wi_db, tk_db=None, gps_db=None,
                    canonical_companies: dict[str, list[str]] = None) -> None:
    """Peuple `lines` avec l'UNION des lignes vues dans `ligne`, les tickets, et le GPS
    (l'utilisateur veut toutes les lignes, pas seulement celles enregistrées dans `ligne`).

    `ligne` reste la source la plus fiable (source='ligne'). Les lignes vues UNIQUEMENT
    dans les tickets ou UNIQUEMENT dans le GPS sont ajoutées avec `source` et `notes`
    pour rester traçables -- notamment les codes à motif 9XX qui ressemblent à des
    services spéciaux plutôt qu'à des lignes régulières (voir `_line_note`).
    """
    canonical_companies = canonical_companies or CANONICAL_COMPANIES
    alias_to_canon = {alias: canon for canon, aliases in canonical_companies.items() for alias in aliases}
    company_id = {row[1]: row[0] for row in conn.execute("SELECT company_id, canonical_name FROM companies")}

    # 1. `ligne` source la plus fiable
    ligne_pairs: dict[tuple, dict] = {}
    for d in wi_db["ligne"].find({}, {"code": 1, "societe": 1}):
        canon = alias_to_canon.get(d.get("societe"))
        if canon is None:
            continue
        key = (canon, str(d["code"]).strip())
        ligne_pairs.setdefault(key, {"source": "ligne", "notes": None})

    # 2. tickets avec volume, pour distinguer signal réel de bruit
    tk_pairs: dict[tuple, int] = {}
    if tk_db is not None:
        for yr in ["2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]:
            pipe = [{"$group": {"_id": {"s": "$Societe", "l": "$CodeRoute"}, "n": {"$sum": 1}}}]
            for d in tk_db[f"Ticket{yr}"].aggregate(pipe):
                soc, code = d["_id"].get("s"), d["_id"].get("l")
                if not (soc and code):
                    continue
                canon = alias_to_canon.get(soc)
                if canon is None:
                    continue
                key = (canon, str(code).strip())
                tk_pairs[key] = tk_pairs.get(key, 0) + d["n"]

    # 3. GPS échantillon (juste pour découvrir des codes, pas pour des stats)
    gps_pairs: set = set()
    if gps_db is not None:
        days = sorted(n for n in gps_db.list_collection_names() if re.fullmatch(r"d\d{8}", n))
        for day in days[::20]:
            pipe = [{"$group": {"_id": {"s": "$service.societe", "l": "$service.codeLigne"}}}]
            for d in gps_db[day].aggregate(pipe):
                soc, code = d["_id"].get("s"), d["_id"].get("l")
                if soc and code:
                    canon = alias_to_canon.get(soc)
                    if canon:
                        gps_pairs.add((canon, str(code).strip()))

    all_pairs = dict(ligne_pairs)
    for key, n in tk_pairs.items():
        if key not in all_pairs:
            all_pairs[key] = {"source": "ticket_only", "notes": _line_note(key[1], n)}
    for key in gps_pairs:
        if key not in all_pairs:
            all_pairs[key] = {"source": "gps_only", "notes": _line_note(key[1], None)}

    rows = [
        (company_id[soc], code, info["source"], info["notes"])
        for (soc, code), info in all_pairs.items() if soc in company_id
    ]
    conn.executemany(
        """INSERT INTO lines (company_id, line_code, source, notes)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(company_id, line_code) DO UPDATE SET
             source=excluded.source, notes=excluded.notes""",
        rows,
    )
    conn.commit()
    return all_pairs


# Table 3 — stops (la table « propre » : arrêts canoniques par coordonnée)
EARTH_R_M = 6371000.0
STOP_CLUSTER_EPS_M = 150.0   

_PLACEHOLDER_NAME_RE = re.compile(r"^(stop\d+|section\d+)$", re.IGNORECASE)


# Boîte englobante Maghreb (Tunisie + Algérie) -- PAS juste la Tunisie : EPE-TVE opère des
# lignes transfrontalières avec une société algérienne, donc des arrêts réels existent bien
# en Algérie (ex. Alger 36.09°N/4.32°E, région Djelfa/Laghouat 33.64°N/1.01°E -- tous deux
# initialement signalés à tort comme "hors zone" avant cette correction). Reste assez étroit
# pour rejeter les ancres `ligne.array_lat_opendata` réellement aberrantes détectées cette
# session : Égypte (31.04°N/31.38°E), Israël/Jordanie (32.52°N/35.52°E) -- bien au-delà de la
# frontière libyenne (~25°E), donc aucun risque de rejeter un arrêt maghrébin légitime.
_MAGHREB_LAT = (28.0, 38.0)
_MAGHREB_LON = (-9.0, 12.0)


def _in_maghreb_bbox(lat: float, lon: float) -> bool:
    return _MAGHREB_LAT[0] <= lat <= _MAGHREB_LAT[1] and _MAGHREB_LON[0] <= lon <= _MAGHREB_LON[1]


def _gather_raw_stop_points(wi_db, od_db) -> list[tuple]:
    """Rassemble TOUS les points géocodés bruts : OpenData (4 collections), winicari.station,
    et les ancres `ligne.array_lat_opendata`. Chaque point garde son nom brut ET sa source,
    pour permettre le vote majoritaire et l'audit après clustering."""
    points = []
    for col in ["Station", "Station_new", "Station2", "Station_sts"]:
        for d in od_db[col].find({}, {"nom_fr": 1, "lat": 1, "lng": 1, "_id": 0}):
            try:
                lat, lng = float(d.get("lat")), float(d.get("lng"))
            except (TypeError, ValueError):
                continue
            if abs(lat) > 1 and abs(lng) > 1 and d.get("nom_fr"):
                points.append((lat, lng, d["nom_fr"], f"OpenData.{col}"))

    for d in wi_db["station"].find({}, {"stop_name_fr": 1, "stop_lat": 1, "stop_lon": 1, "_id": 0}):
        try:
            lat, lon = float(d["stop_lat"]), float(d["stop_lon"])
        except (TypeError, ValueError, KeyError):
            continue
        if abs(lat) > 1 and abs(lon) > 1 and d.get("stop_name_fr"):
            points.append((lat, lon, d["stop_name_fr"], "winicari.station"))

    n_rejected_oob = 0
    for lg in wi_db["ligne"].find({}, {"array_lat_opendata": 1, "array_lng_opendata": 1, "stationnames": 1}):
        la = lg.get("array_lat_opendata") or []
        lo = lg.get("array_lng_opendata") or []
        names = lg.get("stationnames") or []
        for i in range(min(len(la), len(lo))):
            try:
                lat, lon = float(la[i]), float(lo[i])
            except (TypeError, ValueError):
                continue
            if not (abs(lat) > 1 and abs(lon) > 1):
                continue  # (0,0)/nul -- pas une ancre réelle, pas comptée comme "hors zone"
            # Rejette les ancres clairement hors zone Maghreb (ex. Égypte, Israël/Jordanie
            # détectés cette session) -- voir _in_maghreb_bbox.
            if not _in_maghreb_bbox(lat, lon):
                n_rejected_oob += 1
                continue
            nm = names[i] if i < len(names) else f"stop{i}"
            points.append((lat, lon, nm, "ligne.array_lat_opendata"))
    if n_rejected_oob:
        print(f"    (ligne.array_lat_opendata : {n_rejected_oob} ancre(s) hors zone Maghreb rejetée(s))")
    return points


def _pick_primary_name(names: list[str], sources: list[str]) -> str:
    """Vote par SOURCE DISTINCTE d'accord, pas par nombre brut de lignes.

    Piège découvert en session : `ligne.array_lat_opendata` répète le même nom une fois par
    ligne qui dessert l'arrêt (parfois des dizaines de fois, ex. un terminus desservi par
    30 lignes), alors qu'OpenData ne contribue qu'une entrée par collection (4 max). Un vote
    par ligne brute laisse `ligne` gagner par pur volume avec un nom générique/paresseux
    (ex. « TUNIS » pour un terminus qu'OpenData nomme précisément « BAB SAADOUN »), même quand
    3 sources indépendantes s'accordent sur le nom précis. On déduplique donc par
    (source, nom_normalisé) avant de compter : chaque COLLECTION ne vote qu'une fois par nom,
    peu importe combien de lignes internes répètent ce nom.
    """
    from collections import Counter
    pairs = [(s, n.strip()) for n, s in zip(names, sources)
             if n and not _PLACEHOLDER_NAME_RE.match(n.strip())]
    if not pairs:
        return names[0] if names else "INCONNU"
    # une source de base (ex. "ligne", "OpenData.Station") ne vote qu'une fois par nom normalisé
    dedup = {(s, n.strip().upper()) for s, n in pairs}
    counts = Counter(n for _, n in dedup)
    best_count = max(counts.values())
    winners_norm = {k for k, v in counts.items() if v == best_count}
    tied = sorted({n for _, n in pairs if n.strip().upper() in winners_norm})
    return tied[0] if tied else (names[0] if names else "INCONNU")


def populate_stops(conn: sqlite3.Connection, wi_db, od_db, eps_m: float = STOP_CLUSTER_EPS_M) -> pd.DataFrame:
    """Peuple `stops` en regroupant TOUS les points géocodés bruts par proximité physique
    (pas par nom) -- résout simultanément : même nom/lieux différents (ex. STADE MUNICIPALE),
    noms différents/même lieu, variantes orthographiques mineures, et doublons.

    Chaque cluster devient UN arrêt canonique (centroïde + alias + provenance). Les entrées
    isolées (un seul point, ex. un placeholder `stopN` sans aucun nom réel dans le rayon)
    restent avec `confidence='non_nomme'` plutôt que de recevoir un faux nom.
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN

    points = _gather_raw_stop_points(wi_db, od_db)
    df = pd.DataFrame(points, columns=["lat", "lon", "name", "source"])

    coords_rad = np.radians(df[["lat", "lon"]].values)
    eps_rad = eps_m / EARTH_R_M
    labels = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords_rad).labels_
    df["cluster"] = labels

    rows = []
    for cid, g in df.groupby("cluster"):
        lat, lon = float(g["lat"].mean()), float(g["lon"].mean())
        names = list(g["name"])
        point_sources = list(g["source"])
        real_names = [n for n in names if not _PLACEHOLDER_NAME_RE.match(str(n).strip())]
        primary = _pick_primary_name(names, point_sources)
        aliases = sorted(set(str(n).strip() for n in names))
        sources = sorted(set(point_sources))
        n_independent_sources = len(sources)
        if not real_names:
            confidence = "non_nomme"          # seulement des placeholders stopN/sectionN
        elif n_independent_sources >= 2:
            confidence = "verifie"            # >=2 sources indépendantes s'accordent
        else:
            confidence = "inferee"            # une seule source
        rows.append((lat, lon, primary, json.dumps(aliases, ensure_ascii=False),
                      json.dumps(sources, ensure_ascii=False), confidence))

    conn.execute("DELETE FROM stops")   # reconstruction complète -- pas de fusion incrémentale pour l'instant
    conn.executemany(
        """INSERT INTO stops (lat, lon, primary_name, aliases, source, confidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return df


def merge_triangulated_stops(conn: sqlite3.Connection, results: list[dict],
                              merge_radius_m: float = STOP_CLUSTER_EPS_M) -> dict:
    """Intègre les résultats de `stations.triangulate_stop` dans `stops`.

    Si un point triangulé tombe à moins de `merge_radius_m` d'un arrêt EXISTANT (cas d'un
    alias qu'on n'avait pas su rapprocher par le nom), on ajoute juste le nom comme alias
    plutôt que de créer un doublon. Sinon, nouvel arrêt avec confidence='triangule_non_verifie'.

    IMPORTANT (découvert par vérification terrain de l'utilisateur, cas HOP.SAHLOUL) : le plus
    GRAND sous-groupe de pings correspondants n'est PAS forcément le bon arrêt -- un bus peut
    s'attarder plus longtemps près d'un point de congestion voisin (carrefour, feu) que sur
    l'arrêt désigné lui-même, qui n'est qu'un arrêt bref. Pour HOP.SAHLOUL, le sous-groupe
    correct ne comptait que 4 points (sous le seuil `min_used=5`) contre 10 pour le mauvais.
    Donc TOUTE triangulation reste non-vérifiée par construction -- `confidence` le reflète
    explicitement plutôt que de laisser croire à une position confirmée.

    `results` : liste de dicts avec au moins name/lat/lon/n_used/spread_m (voir
    `stations.triangulate_stop`). Retourne {"merged": n, "created": n}.
    """
    import numpy as np
    from src.data.foundation import haversine

    existing = conn.execute("SELECT stop_id, lat, lon, aliases FROM stops").fetchall()
    ex_lat = np.array([r[1] for r in existing]) if existing else np.array([])
    ex_lon = np.array([r[2] for r in existing]) if existing else np.array([])

    merged, created = 0, 0
    for r in results:
        if not r.get("ok") or r.get("lat") is None:
            continue
        name, lat, lon = r["name"], r["lat"], r["lon"]
        if len(existing):
            d = haversine(ex_lat, ex_lon, lat, lon)
            nearest = int(np.argmin(d))
            if d[nearest] <= merge_radius_m:
                stop_id, _, _, aliases_json = existing[nearest]
                aliases = json.loads(aliases_json)
                if name not in aliases:
                    aliases.append(name)
                    conn.execute("UPDATE stops SET aliases = ? WHERE stop_id = ?",
                                 (json.dumps(aliases, ensure_ascii=False), stop_id))
                merged += 1
                continue
        conn.execute(
            """INSERT INTO stops (lat, lon, primary_name, aliases, source, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (lat, lon, name, json.dumps([name], ensure_ascii=False),
             json.dumps(["ticket_triangulation"], ensure_ascii=False), "triangule_non_verifie"),
        )
        created += 1
    conn.commit()
    return {"merged": merged, "created": created}


# Table 4 — line_stops : résolution multi-niveaux, la source la plus fiable d'abord
# Un seul mécanisme de désaccord potentiel : chaque niveau produit un lat/lon brut, qu'on
# rattache au `stops` canonique déjà construit par PROXIMITÉ (pas par nom) -- cohérent avec
# la façon dont `stops` a lui-même été bâti, donc la correspondance est quasi exacte pour
# les niveaux 1 à 4 (ces points ont été des ENTRÉES du clustering initial).

class _StopIndex:
    """Recherche du stop canonique le plus proche -- vectorisé, pas d'index spatial nécessaire
    à cette échelle (~3000 points)."""

    def __init__(self, conn: sqlite3.Connection):
        import numpy as np
        rows = conn.execute("SELECT stop_id, lat, lon FROM stops").fetchall()
        self.ids = np.array([r[0] for r in rows])
        self.lats = np.array([r[1] for r in rows])
        self.lons = np.array([r[2] for r in rows])

    def nearest(self, lat: float, lon: float, max_m: float = STOP_CLUSTER_EPS_M):
        from src.data.foundation import haversine
        if len(self.ids) == 0:
            return None
        d = haversine(self.lats, self.lons, lat, lon)
        i = int(d.argmin())
        return int(self.ids[i]) if d[i] <= max_m else None


def _load_station_coords(wi_db) -> dict:
    """{(societe, stop_id_str) -> (lat, lon)} depuis winicari.station."""
    out = {}
    for d in wi_db["station"].find({}, {"societe": 1, "stop_id": 1, "stop_lat": 1, "stop_lon": 1, "_id": 0}):
        try:
            lat, lon = float(d["stop_lat"]), float(d["stop_lon"])
        except (TypeError, ValueError, KeyError):
            continue
        if abs(lat) > 1 and abs(lon) > 1:
            out[(d.get("societe"), str(d.get("stop_id")))] = (lat, lon)
    return out


def _load_opendata_code_station(od_db) -> dict:
    """{code_station -> (lat, lon)} across les 4 collections OpenData."""
    out = {}
    for col in ["Station", "Station_new", "Station2", "Station_sts"]:
        for d in od_db[col].find({"code_station": {"$exists": True, "$ne": ""}},
                                  {"code_station": 1, "lat": 1, "lng": 1, "_id": 0}):
            try:
                lat, lng = float(d["lat"]), float(d["lng"])
            except (TypeError, ValueError, KeyError):
                continue
            if abs(lat) > 1 and abs(lng) > 1:
                out.setdefault(d["code_station"], (lat, lng))
    return out


def _load_stop_societe_rows(wi_db) -> dict:
    """{(societe, ROUTENR) -> [rows STOPNR/NAMENRnew/NAMENR ordonnés]} depuis STOP<societe>."""
    from src.data.foundation import _STOP_COL_MAP
    out: dict = {}
    for societe, col_name in _STOP_COL_MAP.items():
        for d in wi_db[col_name].find({}, {"ROUTENR": 1, "STOPNR": 1, "NAMENRnew": 1, "NAMENR": 1, "_id": 0}):
            key = (societe, d.get("ROUTENR"))
            out.setdefault(key, []).append(d)
    for key in out:
        out[key].sort(key=lambda d: str(d.get("STOPNR", "")))
    return out


def _tier1_stop_station(societe, code, stop_rows, station_coords, idx: _StopIndex):
    """STOP<societe> (ordre STOPNR) + winicari.station (coords par NAMENRnew) -- le plus fiable."""
    try:
        route_nr = int(code)
    except (TypeError, ValueError):
        return None
    rows = stop_rows.get((societe, route_nr))
    if not rows:
        return None
    out = []
    for d in rows:
        coord = station_coords.get((societe, str(d.get("NAMENRnew"))))
        if coord is None:
            continue
        stop_id = idx.nearest(*coord)
        if stop_id is None:
            continue
        try:
            seq = int(str(d["STOPNR"]))
        except (TypeError, ValueError):
            continue
        out.append({"seq": seq, "stop_id": stop_id, "company_stop_code": str(d.get("NAMENRnew"))})
    return out if len(out) >= 4 else None


def _tier2_stop_opendata_code(societe, code, stop_rows, od_code_station, idx: _StopIndex):
    """STOP<societe>.NAMENR -> OpenData.code_station -- direct, mais dépend de la société
    (vérifié : marche pour S.R.T.BIZERTE/S.T.S/TCV, pas pour SRT.ELGOUAFEL/EPE-TVE)."""
    try:
        route_nr = int(code)
    except (TypeError, ValueError):
        return None
    rows = stop_rows.get((societe, route_nr))
    if not rows:
        return None
    out = []
    for d in rows:
        coord = od_code_station.get(str(d.get("NAMENR")))
        if coord is None:
            continue
        stop_id = idx.nearest(*coord)
        if stop_id is None:
            continue
        try:
            seq = int(str(d["STOPNR"]))
        except (TypeError, ValueError):
            continue
        out.append({"seq": seq, "stop_id": stop_id, "company_stop_code": str(d.get("NAMENR"))})
    return out if len(out) >= 4 else None


def _load_names_registry(wi_db) -> dict:
    """{(societe, code) -> NomFr} depuis winicari.Names -- registre de noms société-large,
    confirmé équivalent à STOP<societe>.NAMENRnew pour S.R.T.BIZERTE (Names.Code=47 -> 'TUNIS'
    == STOP.NAMENRnew=47). Existe aussi pour SRT.ELGOUAFEL (451 entrées) et S.R.T.SELIANA (91)
    -- ces deux sociétés n'ont NI winicari.station NI de NAMENR compatible OpenData, donc ce
    registre est leur SEULE voie de résolution de coordonnées en dehors des tickets."""
    out = {}
    for d in wi_db["Names"].find({}, {"societe": 1, "Code": 1, "NomFr": 1, "_id": 0}):
        if d.get("societe") and d.get("Code") is not None and d.get("NomFr"):
            out[(d["societe"], str(d["Code"]))] = d["NomFr"]
    return out


def _tier2b_stop_names_registry(societe, code, stop_rows, names_registry, stops_od_dict, idx: _StopIndex):
    """STOP<societe> (ordre STOPNR) + NAMENRnew -> Names.Code (même société) -> NomFr, puis
    résolution/désambiguïsation par nom contre `stops` (même mécanisme validé que tier 5,
    mais ordre STOPNR officiel au lieu de l'ordre déduit des tickets -- plus propre quand
    disponible). Comble le vrai trou : sociétés avec STOP<societe> mais sans winicari.station
    ni NAMENR compatible OpenData (SRT.ELGOUAFEL, S.R.T.SELIANA)."""
    from src.data import stations as st
    try:
        route_nr = int(code)
    except (TypeError, ValueError):
        return None
    rows = stop_rows.get((societe, route_nr))
    if not rows:
        return None

    stop_candidates = []
    for d in rows:
        name = names_registry.get((societe, str(d.get("NAMENRnew"))))
        if not name:
            continue
        try:
            seq = int(str(d["STOPNR"]))
        except (TypeError, ValueError):
            continue
        cands = stops_od_dict.get(st.norm(name))
        stop_candidates.append({"code": seq, "name": name, "candidates": cands or []})
    if len(stop_candidates) < 4:
        return None

    resolved = st._disambiguate_sequential(stop_candidates)
    out = []
    for s, hit in zip(stop_candidates, resolved):
        if hit is None:
            continue
        lat, lon, _src = hit
        stop_id = idx.nearest(lat, lon, max_m=STOP_CLUSTER_EPS_M * 2)
        if stop_id is None:
            continue
        out.append({"seq": s["code"], "stop_id": stop_id, "company_stop_code": s["name"]})
    return out if len(out) >= 4 else None


def _tier3_ligne_stations(ligne_doc, societe, station_coords, idx: _StopIndex):
    """ligne.stations (ordonné) -> winicari.station.stop_id."""
    ids = ligne_doc.get("stations") or []
    if not ids:
        return None
    out = []
    for i, sid in enumerate(ids):
        coord = station_coords.get((societe, str(sid)))
        if coord is None:
            continue
        stop_id = idx.nearest(*coord)
        if stop_id is None:
            continue
        out.append({"seq": i, "stop_id": stop_id, "company_stop_code": str(sid)})
    return out if len(out) >= 4 else None


def _tier4_ligne_anchors(ligne_doc, idx: _StopIndex):
    """ligne.array_lat_opendata / stationnames -- ancres brutes, déjà les entrées du clustering."""
    la = ligne_doc.get("array_lat_opendata") or []
    lo = ligne_doc.get("array_lng_opendata") or []
    out = []
    for i in range(min(len(la), len(lo))):
        try:
            lat, lon = float(la[i]), float(lo[i])
        except (TypeError, ValueError):
            continue
        if abs(lat) <= 1 or abs(lon) <= 1:
            continue
        stop_id = idx.nearest(lat, lon)
        if stop_id is None:
            continue
        out.append({"seq": i, "stop_id": stop_id, "company_stop_code": None})
    return out if len(out) >= 4 else None


def _stops_as_od_dict(conn: sqlite3.Connection) -> dict:
    """`stops` reformaté comme `stations.opendata_dict()` : {nom_normalisé -> [(lat,lon,source)]}
    -- permet de réutiliser `stations.build_enriched_stops` (désambiguïsation séquentielle déjà
    validée, +44 lignes prouvées en notebook) contre le registre canonique désormais plus propre
    que le dictionnaire OpenData brut d'origine, au lieu de réinventer une résolution plus
    faible ici (une première version en alias-exact-unique n'en récupérait que 22)."""
    import json as _json
    out: dict = {}
    from src.data import stations as st
    for stop_id, aliases_json, primary_name, lat, lon in conn.execute(
            "SELECT stop_id, aliases, primary_name, lat, lon FROM stops"):
        for a in _json.loads(aliases_json):
            out.setdefault(st.norm(a), []).append((lat, lon, f"stops.{stop_id}"))
    return out


def _tier5_ticket_order(ticket_index, stops_od_dict: dict, idx: _StopIndex, societe, code):
    """Ordre des tickets + résolution/désambiguïsation déjà validées (`stations.py`), puis
    rattachement au stop canonique le plus proche du point résolu."""
    from src.data import stations as st
    sf = st.build_enriched_stops(ticket_index, stops_od_dict, code, societe)
    if sf.empty:
        return None
    out = []
    for _, r in sf.iterrows():
        stop_id = idx.nearest(r["lat"], r["lon"], max_m=STOP_CLUSTER_EPS_M * 2)
        if stop_id is None:
            continue
        out.append({"seq": int(r["seq"]), "stop_id": stop_id, "company_stop_code": str(r["route_seq"])})
    return out if len(out) >= 4 else None


# Assignations ligne/arrêt connues comme fausses -- trouvées en vérifiant systématiquement
# TOUTE la géométrie résolue pour des sauts consécutifs invraisemblables (>50km ET >8x la
# distance médiane des segments de la ligne), puis en testant le retrait de chaque extrémité
# du segment pour identifier laquelle des deux fait réellement baisser le ratio de saut de
# CETTE ligne -- pas une supposition manuelle. Dans TOUS ces cas sauf EL GARAA (qui a deux
# candidats OpenData réels et distincts, ~24km d'écart), le nom n'a qu'UNE seule coordonnée
# réelle connue dans TOUTES les sources (OpenData, winicari.station, ligne.array_lat_opendata)
# -- il n'existe donc pas de "deuxième coordonnée" à garder : soit ce nom est utilisé sur la
# bonne ligne quelque part (qu'on garde), soit il ne l'est nulle part pour CETTE ligne (qu'on
# exclut ici plutôt que de fabriquer une coordonnée qu'on n'a pas). N'exclut QUE cette ligne
# précise -- le stop reste pleinement utilisable sur toute autre ligne où il est cohérent.
#
# Clé = (societe, line_code, nom NORMALISÉ via stations.norm) -- PAS stop_id : les stop_id
# sont regénérés à chaque populate_stops (clustering + autoincrément), un ID épinglé ici
# casserait silencieusement au premier rebuild dont le clustering change. Le nom normalisé
# est stable tant que la collision existe -- c'est précisément LA propriété qu'on exclut.
_KNOWN_BAD_LINE_STOP_ASSIGNMENTS: set[tuple[str, str, str]] = {
    ("S.R.T.K", "204", "OUEDRAMAL"),      # incohérent avec SBEITLA/EL GONNA sur cette ligne
    ("S.T.S", "325", "ELKEF"),            # incohérent avec CHEBIKA sur cette ligne
    ("S.T.S", "227", "SAIDA"),            # homonyme nord (36.83N) sur une ligne région Sousse
    ("S.T.S", "232", "SAIDA"),            # même homonyme nord, même problème
    ("S.T.S", "232", "SOUASSI"),          # homonyme nord (36.83N) -- le vrai Souassi est en région Mahdia
    ("S.R.T.BIZERTE", "550", "DKHILA"),   # homonyme région Monastir (35.52N/10.97E) sur une ligne Tunis/nord
    ("S.R.T.BIZERTE", "560", "DKHILA"),
    ("SRT.ELGOUAFEL", "8", "SIDIAHMED"),  # incohérent avec Kalla Khesba sur cette ligne
    ("SRT.ELGOUAFEL", "57", "ECHIBIKA"),  # collision de nom (homonyme lointain)
    ("SRT.ELGOUAFEL", "83", "ECHIBIKA"),
    ("SRT.ELGOUAFEL", "97", "ECHIBIKA"),
    ("SRT.ELGOUAFEL", "66", "GABES"),     # incohérent avec Kbili sur cette ligne précise
    ("SRT.ELGOUAFEL", "2213", "FATNASSA"),# incohérent avec KM 105 sur cette ligne
    ("SRT.ELGOUAFEL", "17", "ELGARAA"),   # les DEUX candidats EL GARAA sont à 80-100km du reste
                                          # de la ligne (GAFSA/Majnni/El Gtar, tous ~34.3-34.4N)
    ("S.R.T.K", "505", "MADRESSA1"),      # incohérent avec EL BATEN sur ces 3 lignes
    ("S.R.T.K", "506", "MADRESSA1"),
    ("S.R.T.K", "507", "MADRESSA1"),
}


def populate_line_stops(conn: sqlite3.Connection, wi_db, od_db, tk_db=None,
                         ticket_index=None) -> dict:
    """Peuple `line_stops` en essayant chaque niveau dans l'ordre de fiabilité, ligne par
    ligne, et s'arrête au premier niveau qui résout >= 4 arrêts. Retourne un décompte par
    niveau utilisé, pour audit."""
    idx = _StopIndex(conn)
    station_coords = _load_station_coords(wi_db)
    od_code_station = _load_opendata_code_station(od_db)
    stop_rows = _load_stop_societe_rows(wi_db)
    stops_od_dict = _stops_as_od_dict(conn)
    names_registry = _load_names_registry(wi_db)

    if ticket_index is None and tk_db is not None:
        from src.data import stations as st
        ticket_index = st.load_or_build_ticket_index(
            tk_db, Path(__file__).resolve().parents[2] / "models" / "cache" / "ticket_stop_index.parquet")

    ligne_by_key = {(d.get("societe"), str(d.get("code")).strip()): d for d in wi_db["ligne"].find({})}
    lines = conn.execute("SELECT line_id, company_id, line_code FROM lines").fetchall()
    company_name = {row[0]: row[1] for row in conn.execute("SELECT company_id, canonical_name FROM companies")}

    # stop_id -> nom normalisé, pour la liste d'exclusions (voir
    # _KNOWN_BAD_LINE_STOP_ASSIGNMENTS : clé par nom, stable entre rebuilds)
    from src.data.stations import norm as _norm_name
    stop_norm_name = {sid: _norm_name(name) for sid, name in
                      conn.execute("SELECT stop_id, primary_name FROM stops")}

    conn.execute("DELETE FROM line_stops")
    tier_counts = {"stop_station": 0, "stop_opendata_code": 0, "stop_names_registry": 0,
                   "ligne_stations": 0, "ligne_anchors": 0, "ticket_order": 0, "unresolved": 0}

    for line_id, company_id, line_code in lines:
        societe = company_name.get(company_id)
        if societe is None:
            continue
        ligne_doc = ligne_by_key.get((societe, line_code))

        result, tier_name = None, None
        for name, fn in [
            ("stop_station", lambda: _tier1_stop_station(societe, line_code, stop_rows, station_coords, idx)),
            ("stop_opendata_code", lambda: _tier2_stop_opendata_code(societe, line_code, stop_rows, od_code_station, idx)),
            ("stop_names_registry", lambda: _tier2b_stop_names_registry(societe, line_code, stop_rows, names_registry, stops_od_dict, idx)),
            ("ligne_stations", lambda: _tier3_ligne_stations(ligne_doc, societe, station_coords, idx) if ligne_doc else None),
            ("ligne_anchors", lambda: _tier4_ligne_anchors(ligne_doc, idx) if ligne_doc else None),
            ("ticket_order", lambda: _tier5_ticket_order(ticket_index, stops_od_dict, idx, societe, line_code) if ticket_index is not None else None),
        ]:
            result = fn()
            if result is not None:
                tier_name = name
                break

        if result is None:
            tier_counts["unresolved"] += 1
            continue
        tier_counts[tier_name] += 1
        # certaines sources brutes répètent un STOPNR (données sales en amont) -- on garde la
        # première occurrence par seq plutôt que de planter sur la contrainte UNIQUE
        seen_seq: set = set()
        deduped = []
        for r in result:
            if r["seq"] in seen_seq:
                continue
            if (societe, line_code, stop_norm_name.get(r["stop_id"], "")) in _KNOWN_BAD_LINE_STOP_ASSIGNMENTS:
                continue  # voir _KNOWN_BAD_LINE_STOP_ASSIGNMENTS -- assignation vérifiée fausse
            seen_seq.add(r["seq"])
            deduped.append(r)
        conn.executemany(
            "INSERT INTO line_stops (line_id, stop_id, seq, company_stop_code) VALUES (?, ?, ?, ?)",
            [(line_id, r["stop_id"], r["seq"], r["company_stop_code"]) for r in deduped],
        )
    conn.commit()
    return tier_counts


# Table 5/6 — trips + trip_stops : reconstruction GPS réelle sur la géométrie line_stops
def _usable_lines_from_line_stops(conn: sqlite3.Connection) -> dict:
    """{(line_code, societe) -> DataFrame(seq, route_seq, name, lat, lon, s_m, stop_id)}
    -- remplace `foundation.build_usable_lines` comme source de géométrie de ligne pour la
    reconstruction de trajets, en s'appuyant sur `line_stops` (résolu par les 6 niveaux les
    plus fiables disponibles, voir populate_line_stops) plutôt que sur la logique ad-hoc de
    `foundation.py`. Même forme de sortie que `foundation.stops_frame` + colonne `stop_id`
    supplémentaire pour rattacher chaque arrivée à son arrêt canonique après reconstruction.

    C'est la source de géométrie RÉELLEMENT utilisée par `build_foundation.py` (le script de
    reconstruction en production) -- `bridge_geometry_outliers` y est donc appliqué comme dans
    `foundation.stops_frame`/`_stop_station_frame` : un nom d'arrêt partagé entre plusieurs
    lignes (ex. S.R.T.K "EL GARAA", correcte pour quelques lignes, ~25-27km fausse pour la
    plupart des autres -- voir `api.main.get_coord_suspect`) gonflait `route_len` de dizaines
    de km sur les lignes affectées, ce qui faussait tous les seuils dérivés de route_len dans
    `segment_trips` (`full_frac`/`reversal_frac`/`min_span_frac`) -- confirmé : +55km/21.6% sur
    S.R.T.K/217 à lui seul, assez pour faire classer "partiel" un trajet réellement complet.
    """
    import numpy as np
    import pandas as pd
    from src.data.foundation import haversine, bridge_geometry_outliers, Config

    rows = conn.execute("""
        SELECT l.line_code, c.canonical_name AS societe, ls.seq, ls.stop_id, s.primary_name, s.lat, s.lon
        FROM line_stops ls
        JOIN lines l ON l.line_id = ls.line_id
        JOIN companies c ON c.company_id = l.company_id
        JOIN stops s ON s.stop_id = ls.stop_id
        ORDER BY l.line_id, ls.seq
    """).fetchall()

    by_line: dict = {}
    for line_code, societe, seq, stop_id, name, lat, lon in rows:
        by_line.setdefault((line_code, societe), []).append((seq, stop_id, name, lat, lon))

    cfg = Config()
    out = {}
    for key, entries in by_line.items():
        entries.sort(key=lambda x: x[0])
        df = pd.DataFrame(entries, columns=["route_seq", "stop_id", "name", "lat", "lon"])
        seg = haversine(df["lat"].values[:-1], df["lon"].values[:-1], df["lat"].values[1:], df["lon"].values[1:])
        seg = bridge_geometry_outliers(df["lat"].values, df["lon"].values, seg, cfg)
        df["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
        df.insert(0, "seq", range(len(df)))
        out[key] = df
    return out


def populate_trips(conn: sqlite3.Connection, gps_db, since_day: str = None, until_day: str = None,
                    companies: list = None, additive: bool = False) -> dict:
    """Reconstruit les trajets réels (pings GPS) sur la géométrie `line_stops`, peuple
    `trips`/`trip_stops`. Réutilise `foundation.reconstruct_bus_day` (même pipeline que
    `build_foundation.py`, seule la source de géométrie de ligne change) et
    `anomaly.trip_features` pour l'agrégation par trajet (au lieu de la réimplémenter).

    `companies` restreint la géométrie/les candidats à ces sociétés (ex. pour reconstruire
    la fenêtre GPS historique propre à UNE société sans retraiter tout le reste).
    `additive=True` n'efface PAS `trips`/`trip_stops` avant d'insérer -- utile pour cumuler
    plusieurs fenêtres (une par société) sans écraser un run précédent.

    Reconstruction complète sur la plage demandée à chaque appel (pas de fragments
    incrémentaux comme `build_foundation.py` -- portée volontairement plus restreinte pour
    l'instant). Retourne des statistiques agrégées pour comparaison avant/après.
    """
    from src.data import foundation as fdn
    from src.data import anomaly as an
    import pandas as pd
    import numpy as np

    cfg = fdn.Config()
    acfg = an.AnomalyConfig()
    usable = _usable_lines_from_line_stops(conn)
    if companies:
        usable = {k: v for k, v in usable.items() if k[1] in companies}
    days = fdn.gps_days(gps_db, cfg)
    if since_day:
        days = [d for d in days if d >= since_day]
    if until_day:
        days = [d for d in days if d <= until_day]

    line_id_map, company_id_map = {}, {}
    for line_id, company_id, societe, line_code in conn.execute(
            "SELECT l.line_id, l.company_id, c.canonical_name, l.line_code "
            "FROM lines l JOIN companies c ON c.company_id = l.company_id"):
        line_id_map[(line_code, societe)] = line_id
        company_id_map[(line_code, societe)] = company_id

    if not additive:
        conn.execute("DELETE FROM trip_stops")
        conn.execute("DELETE FROM trips")
    cur = conn.cursor()

    n_candidates = n_busdays_with_trips = n_stop_rows = 0
    frames = []
    from tqdm import tqdm
    for day in tqdm(days, desc="Trajets GPS", unit="jour"):
        for (dy, line, soc, bus) in fdn.candidates_for_day(gps_db, day, usable, cfg):
            n_candidates += 1
            key = (line, soc)
            if key not in usable:
                continue
            try:
                f = fdn.reconstruct_bus_day(gps_db, dy, line, soc, bus, usable[key], cfg)
            except Exception:
                continue
            if len(f):
                frames.append(f)
                n_busdays_with_trips += 1

    if not frames:
        return {"days": len(days), "n_candidates": n_candidates, "n_trips": 0,
                "n_full_trips": 0, "n_stop_rows": 0}

    fa = pd.concat(frames, ignore_index=True)
    feats = an.trip_features(fa, acfg)
    stop_id_lookup = {key: dict(zip(df["seq"], df["stop_id"])) for key, df in usable.items()}

    n_full = n_loop_unknown = 0
    for _, t in feats.iterrows():
        key = (t["line"], t["societe"])
        line_id = line_id_map.get(key)
        company_id = company_id_map.get(key)
        if line_id is None:
            continue
        # None = ligne en boucle, "complet" non fiable pour cette géométrie (voir
        # foundation.detect_loop_route) -- NE PAS caster en bool(), ça donnerait False à tort
        is_full = None if t["full"] is None or pd.isna(t["full"]) else bool(t["full"])
        if is_full is None:
            n_loop_unknown += 1
        n_full += int(is_full) if is_full else 0
        origin_idle_from = t.get("origin_idle_from")
        end_idle_to = t.get("end_idle_to")
        cur.execute(
            """INSERT INTO trips (line_id, company_id, bus, day, dir, is_full, trip_start, trip_end,
                                   n_stops, match_rate, max_dwell_s, mean_dwell_s, total_elapsed_min,
                                   dist_m_max, max_dark_s, terminus_idle_min,
                                   trip_dark_before_stop, trip_dark_after_stop,
                                   origin_idle_min, origin_idle_from, origin_idle_stop,
                                   end_idle_min, end_idle_to, end_idle_stop)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (line_id, company_id, str(t["bus"]), str(t["day"]), t["dir"], is_full,
             str(t["trip_start"]), str(t["trip_end"]), int(t["n_stops"]), float(t["match_rate"]),
             float(t["max_dwell_s"]), float(t["mean_dwell_s"]), float(t["total_elapsed"]),
             float(t["dist_m_max"]), float(t["max_dark_s"]), float(t.get("terminus_idle_min", 0.0) or 0.0),
             t.get("trip_dark_before_stop"), t.get("trip_dark_after_stop"),
             float(t.get("origin_idle_min", 0.0) or 0.0),
             str(origin_idle_from) if pd.notna(origin_idle_from) else None,
             t.get("origin_idle_stop"),
             float(t.get("end_idle_min", 0.0) or 0.0),
             str(end_idle_to) if pd.notna(end_idle_to) else None,
             t.get("end_idle_stop")),
        )
        trip_pk = cur.lastrowid

        seq_map = stop_id_lookup.get(key, {})
        g = fa[(fa["day"] == t["day"]) & (fa["line"] == t["line"]) & (fa["societe"] == t["societe"])
               & (fa["bus"] == t["bus"]) & (fa["trip_id"] == t["trip_id"])]
        stop_rows = []
        for _, r in g.iterrows():
            stop_id = seq_map.get(r["seq"])
            if stop_id is None:
                continue
            dwell = None if pd.isna(r["dwell_s"]) else float(r["dwell_s"])
            dist = None if pd.isna(r["dist_m"]) else float(r["dist_m"])
            dark = None if pd.isna(r.get("dark_s")) else float(r["dark_s"])
            arrival = None if pd.isna(r["arrival"]) else str(r["arrival"])
            departure = None if pd.isna(r["departure"]) else str(r["departure"])
            stop_rows.append((trip_pk, stop_id, int(r["seq"]), arrival, departure, dwell, dist,
                               bool(r["matched"]), dark, bool(r.get("had_gap", False))))
        if stop_rows:
            cur.executemany(
                """INSERT INTO trip_stops (trip_id, stop_id, seq, arrival, departure, dwell_s, dist_m,
                                            matched, dark_s, had_gap) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                stop_rows,
            )
            n_stop_rows += len(stop_rows)

    conn.commit()
    return {"days": len(days), "n_candidates": n_candidates, "n_busdays_with_trips": n_busdays_with_trips,
            "n_trips": len(feats), "n_full_trips": n_full, "n_loop_unknown_full": n_loop_unknown,
            "n_stop_rows": n_stop_rows}


# Table 7 — tickets_daily (agrégats journaliers, depuis winicari.details)
def populate_tickets_daily(conn: sqlite3.Connection, wi_db,
                            canonical_companies: dict[str, list[str]] = None) -> dict:
    """Peuple `tickets_daily` depuis `winicari.details` (agrégats journaliers billetterie,
    PAS les 5,4M tickets individuels -- ceux-là ne servent qu'une fois, comme entrée ETL pour
    stops/line_stops, pas comme table durable -- voir la note dans le schéma).

    `details` contient des doublons bruts sur (societe, CodeLigne, codeBus, date) --
    7 881 documents pour 7 171 combinaisons distinctes -- sommés ici plutôt qu'insérés tels
    quels. ~1,3% des documents ont `societe=None` (inattribuable) et sont ignorés.
    """
    canonical_companies = canonical_companies or CANONICAL_COMPANIES
    alias_to_canon = {alias: canon for canon, aliases in canonical_companies.items() for alias in aliases}
    company_id_by_name = {row[1]: row[0] for row in conn.execute("SELECT company_id, canonical_name FROM companies")}
    line_id_by_key = {(row[2], row[1]): row[0] for row in conn.execute(
        "SELECT l.line_id, c.canonical_name, l.line_code FROM lines l JOIN companies c ON c.company_id = l.company_id")}

    agg: dict = {}
    n_null_societe = n_unmatched_company = n_unmatched_line = 0
    for d in wi_db["details"].find({}, {"societe": 1, "CodeLigne": 1, "codeBus": 1, "date": 1,
                                          "nbrTicket": 1, "recette": 1, "_id": 0}):
        raw_soc = d.get("societe")
        if not raw_soc:
            n_null_societe += 1
            continue
        canon = alias_to_canon.get(raw_soc)
        company_id = company_id_by_name.get(canon) if canon else None
        if company_id is None:
            n_unmatched_company += 1
            continue
        line_code = str(d.get("CodeLigne", "")).strip()
        line_id = line_id_by_key.get((line_code, canon))
        if line_id is None:
            n_unmatched_line += 1
            continue
        parts = str(d.get("date", "")).split("/")
        day = f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}" if len(parts) == 3 else None
        if day is None:
            continue
        bus = str(d.get("codeBus", ""))
        key = (company_id, line_id, bus, day)
        entry = agg.setdefault(key, {"nbr_ticket": 0, "recette": 0.0})
        entry["nbr_ticket"] += int(d.get("nbrTicket") or 0)
        entry["recette"] += float(d.get("recette") or 0.0)

    conn.execute("DELETE FROM tickets_daily")
    conn.executemany(
        """INSERT INTO tickets_daily (company_id, line_id, bus, day, nbr_ticket, recette)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(cid, lid, bus, day, v["nbr_ticket"], v["recette"]) for (cid, lid, bus, day), v in agg.items()],
    )
    conn.commit()
    return {"rows_inserted": len(agg), "n_null_societe": n_null_societe,
            "n_unmatched_company": n_unmatched_company, "n_unmatched_line": n_unmatched_line}


def populate_tickets_station_daily(conn: sqlite3.Connection, tk_db,
                                   canonical_companies: dict[str, list[str]] = None,
                                   years: list[str] = None) -> dict:
    """Peuple `tickets_station_daily` depuis les tickets INDIVIDUELS
    (`Historique_Tickets.Ticket{annee}`, ~5,5M documents au total sur 2019-2026) --
    seule source qui porte l'arrêt d'ORIGINE par ticket (`origine`/`NomFR1`), absent de
    `winicari.details` (voir `populate_tickets_daily`).

    Champs vérifiés directement sur des documents réels (pas devinés) :
    - `CodeRoute` = code de ligne réel (PAS `Codeligne`, qui vaut '00'/'99' -- un code de
      catégorie, pas la ligne).
    - `CodeBus` (ex. 6044, même numérotation que le GPS) -- inclus dans le groupement
      depuis le 2026-07-11 : sans lui, une ligne desservie par plusieurs bus le même jour
      mélangeait leurs ventes dans UNE seule ligne par (ligne, arrêt, jour), ce qui ne
      pouvait pas se recouper avec le total d'un bus-jour précis affiché ailleurs
      (`tickets_daily`) -- constaté : la somme par arrêt d'un jour donné dépassait très
      largement la recette du bus-jour signalé, parce qu'elle incluait TOUS les bus de la
      ligne ce jour-là, pas seulement celui affiché.
    - `jour_service` (format "YYYY/MM/DD") = jour de service, cohérent entre années,
      même format que `winicari.details.date` -- utilisé plutôt que `date`/`date_ticket`
      (l'heure d'émission du ticket, pas forcément le même jour calendaire qu'un service
      commencé juste avant minuit) ou `date_service` (horodatage, pas juste le jour).
    - `Societe` a les mêmes variantes que `winicari.details` (ex. "S.R.T.K0"), résolues
      via le même `canonical_companies`/`alias_to_canon` que `populate_tickets_daily`.
    - `requisition` (\"O\"/\"N\") -- EXCLU de `recette` (mais PAS du compte `nbr_ticket`).
      Constaté (2026-07-11) en comparant bus-jour par bus-jour contre `winicari.details` :
      un ticket `requisition=O` est bien ÉMIS (compte dans `nbrTicket` des deux côtés) mais
      son `Prix` n'entre PAS dans la `recette` de `winicari.details` -- vérifié exact sur
      plusieurs bus-jours (ex. S.R.T.K/217/6028/20260621 : 94 tickets des deux côtés, somme
      brute 704.36 DT contre recette réelle 561.26 DT ; en excluant les 15 tickets
      requisition=O -- 143.10 DT -- on retombe EXACTEMENT sur 561.26 DT). Sans cette
      exclusion, `recette` par arrêt était gonflée et ne se recoupait pas avec les totaux
      bus-jour déjà utilisés par le modèle billetterie existant.
      NOTE : ça ne réconcilie pas 100% des bus-jours (un résidu plus petit subsiste sur
      certains, cause non identifiée avec les champs disponibles -- voir la note dans
      docs/WEBSERVICES_NEEDED.md, à clarifier avec l'équipe plateforme plutôt que deviner
      plus loin) -- mais c'est un progrès net et vérifié, pas une supposition.

    Agrégation faite CÔTÉ MONGO (`$group`) plutôt qu'en ramenant 5,5M documents en Python --
    un ordre de grandeur plus rapide, et le volume par (ligne, arrêt, jour) est bien plus
    petit une fois groupé.
    """
    canonical_companies = canonical_companies or CANONICAL_COMPANIES
    alias_to_canon = {alias: canon for canon, aliases in canonical_companies.items() for alias in aliases}
    company_id_by_name = {row[1]: row[0] for row in conn.execute("SELECT company_id, canonical_name FROM companies")}
    line_id_by_key = {(row[2], row[1]): row[0] for row in conn.execute(
        "SELECT l.line_id, c.canonical_name, l.line_code FROM lines l JOIN companies c ON c.company_id = l.company_id")}

    years = years or sorted(c for c in tk_db.list_collection_names() if c.startswith("Ticket"))
    pipeline = [
        {"$group": {
            "_id": {"societe": "$Societe", "line": "$CodeRoute", "bus": "$CodeBus",
                    "station": "$NomFR1", "day": "$jour_service"},
            "nbr_ticket": {"$sum": 1},
            "recette": {"$sum": {"$cond": [{"$eq": ["$requisition", "O"]}, 0, "$Prix"]}},
        }}
    ]

    agg: dict = {}
    n_null_societe = n_unmatched_company = n_unmatched_line = n_null_station = 0
    for yr in years:
        for row in tk_db[yr].aggregate(pipeline, allowDiskUse=True):
            key_raw = row["_id"]
            raw_soc = key_raw.get("societe")
            if not raw_soc:
                n_null_societe += 1
                continue
            canon = alias_to_canon.get(raw_soc)
            company_id = company_id_by_name.get(canon) if canon else None
            if company_id is None:
                n_unmatched_company += 1
                continue
            line_code = str(key_raw.get("line", "")).strip()
            line_id = line_id_by_key.get((line_code, canon))
            if line_id is None:
                n_unmatched_line += 1
                continue
            station = (key_raw.get("station") or "").strip()
            if not station:
                n_null_station += 1
                continue
            bus = str(key_raw.get("bus", "")).strip()
            parts = str(key_raw.get("day", "")).split("/")
            day = f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}" if len(parts) == 3 else None
            if day is None:
                continue
            key = (company_id, line_id, bus, station, day)
            entry = agg.setdefault(key, {"nbr_ticket": 0, "recette": 0.0})
            entry["nbr_ticket"] += int(row.get("nbr_ticket") or 0)
            entry["recette"] += float(row.get("recette") or 0.0)

    conn.execute("DELETE FROM tickets_station_daily")
    conn.executemany(
        """INSERT INTO tickets_station_daily (company_id, line_id, bus, station_name, day,
                                              nbr_ticket, recette)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [(cid, lid, bus, station, day, v["nbr_ticket"], v["recette"])
         for (cid, lid, bus, station, day), v in agg.items()],
    )
    conn.commit()
    return {"rows_inserted": len(agg), "years": years, "n_null_societe": n_null_societe,
            "n_unmatched_company": n_unmatched_company, "n_unmatched_line": n_unmatched_line,
            "n_null_station": n_null_station}


def populate_tickets_station_trip_daily(conn: sqlite3.Connection, tk_db,
                                        canonical_companies: dict[str, list[str]] = None,
                                        years: list[str] = None) -> dict:
    """Peuple `tickets_station_trip_daily` -- même source que `populate_tickets_station_daily`
    (tickets individuels), mais groupée EN PLUS par direction ALLER/RETOUR, dérivée de la
    PARITÉ de `voyage` (pair -> ALLER, impair -> RETOUR) -- même convention que
    `foundation.correct_direction_from_voyage` côté GPS, pas une règle nouvelle.

    Repli 'UNKNOWN' : certains appareils ne suivent pas `voyage` (reste à 0 toute la
    journée) -- dans ce cas, séparer par parité donnerait un faux ALLER-seul (tout à 0 est
    pair) qui ferait croire à une info de direction inexistante. Détecté PAR BUS-JOUR
    (pas par ticket) via une première passe qui vérifie si `voyage` prend au moins une
    valeur non nulle ce jour-là pour ce bus, toutes lignes confondues -- si non, toutes
    les lignes de ce bus-jour sont étiquetées 'UNKNOWN' plutôt que 'ALLER'.
    """
    canonical_companies = canonical_companies or CANONICAL_COMPANIES
    alias_to_canon = {alias: canon for canon, aliases in canonical_companies.items() for alias in aliases}
    company_id_by_name = {row[1]: row[0] for row in conn.execute("SELECT company_id, canonical_name FROM companies")}
    line_id_by_key = {(row[2], row[1]): row[0] for row in conn.execute(
        "SELECT l.line_id, c.canonical_name, l.line_code FROM lines l JOIN companies c ON c.company_id = l.company_id")}

    years = years or sorted(c for c in tk_db.list_collection_names() if c.startswith("Ticket"))

    has_voyage_pipeline = [
        {"$group": {
            "_id": {"societe": "$Societe", "bus": "$CodeBus", "day": "$jour_service"},
            "has_voyage": {"$max": {"$cond": [{"$gt": [{"$ifNull": ["$voyage", 0]}, 0]}, 1, 0]}},
        }}
    ]
    trip_pipeline = [
        {"$group": {
            "_id": {"societe": "$Societe", "line": "$CodeRoute", "bus": "$CodeBus",
                    "station": "$NomFR1", "day": "$jour_service",
                    "parity": {"$mod": [{"$ifNull": ["$voyage", 0]}, 2]}},
            "nbr_ticket": {"$sum": 1},
            "recette": {"$sum": {"$cond": [{"$eq": ["$requisition", "O"]}, 0, "$Prix"]}},
        }}
    ]

    agg: dict = {}
    n_null_societe = n_unmatched_company = n_unmatched_line = n_null_station = 0
    for yr in years:
        has_voyage: dict = {}
        for row in tk_db[yr].aggregate(has_voyage_pipeline, allowDiskUse=True):
            k = row["_id"]
            soc = k.get("societe")
            bus_raw = k.get("bus")
            if not isinstance(soc, str) or isinstance(bus_raw, list):
                continue  # champ mal formé sur ce doc (rare) -- ignoré, pas deviné
            parts = str(k.get("day", "")).split("/")
            day = f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}" if len(parts) == 3 else None
            if day is None:
                continue
            has_voyage[(soc, str(bus_raw or "").strip(), day)] = bool(row["has_voyage"])

        for row in tk_db[yr].aggregate(trip_pipeline, allowDiskUse=True):
            key_raw = row["_id"]
            raw_soc = key_raw.get("societe")
            if not raw_soc:
                n_null_societe += 1
                continue
            canon = alias_to_canon.get(raw_soc)
            company_id = company_id_by_name.get(canon) if canon else None
            if company_id is None:
                n_unmatched_company += 1
                continue
            line_code = str(key_raw.get("line", "")).strip()
            line_id = line_id_by_key.get((line_code, canon))
            if line_id is None:
                n_unmatched_line += 1
                continue
            station = (key_raw.get("station") or "").strip()
            if not station:
                n_null_station += 1
                continue
            bus = str(key_raw.get("bus", "")).strip()
            parts = str(key_raw.get("day", "")).split("/")
            day = f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}" if len(parts) == 3 else None
            if day is None:
                continue
            if has_voyage.get((raw_soc, bus, day)):
                direction = "ALLER" if key_raw.get("parity") == 0 else "RETOUR"
            else:
                direction = "UNKNOWN"
            key = (company_id, line_id, bus, direction, station, day)
            entry = agg.setdefault(key, {"nbr_ticket": 0, "recette": 0.0})
            entry["nbr_ticket"] += int(row.get("nbr_ticket") or 0)
            entry["recette"] += float(row.get("recette") or 0.0)

    conn.execute("DELETE FROM tickets_station_trip_daily")
    conn.executemany(
        """INSERT INTO tickets_station_trip_daily (company_id, line_id, bus, direction,
                                                    station_name, day, nbr_ticket, recette)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(cid, lid, bus, direction, station, day, v["nbr_ticket"], v["recette"])
         for (cid, lid, bus, direction, station, day), v in agg.items()],
    )
    conn.commit()
    return {"rows_inserted": len(agg), "years": years, "n_null_societe": n_null_societe,
            "n_unmatched_company": n_unmatched_company, "n_unmatched_line": n_unmatched_line,
            "n_null_station": n_null_station}


def populate_driver_services(conn: sqlite3.Connection, tk_db,
                             canonical_companies: dict[str, list[str]] = None,
                             years: list[str] = None) -> dict:
    """Peuple `driver_services` depuis les tickets INDIVIDUELS (Historique_Tickets.Ticket{annee}),
    même source que `populate_tickets_station_daily` -- seule à porter un code chauffeur
    (`CodeCh`) avec un horodatage de session exploitable (`date_debut_service`+
    `heure_debut_ticket` / `date_fin_service`+`heure_fin_ticket`, dénormalisés identiques sur
    chaque ticket de la même session -- voir la note sur `driver_services` dans SCHEMA_SQL).

    `years` par défaut restreint à 2025+ (décision utilisateur 2026-07-15 : les années
    antérieures ne sont pas backfillées pour ce premier passage) -- passer explicitement
    une liste plus large pour étendre plus tard sans changer le code.

    $min/$max plutôt que $first sur les champs de session : ils sont censés être constants
    sur tout le groupe (dénormalisés), mais $min/$max encaisse sans risque une éventuelle
    incohérence ponctuelle au lieu de dépendre de l'ordre de renvoi de Mongo.
    """
    canonical_companies = canonical_companies or CANONICAL_COMPANIES
    alias_to_canon = {alias: canon for canon, aliases in canonical_companies.items() for alias in aliases}
    company_id_by_name = {row[1]: row[0] for row in conn.execute("SELECT company_id, canonical_name FROM companies")}
    line_id_by_key = {(row[2], row[1]): row[0] for row in conn.execute(
        "SELECT l.line_id, c.canonical_name, l.line_code FROM lines l JOIN companies c ON c.company_id = l.company_id")}

    years = years or ["Ticket2025", "Ticket2026"]
    pipeline = [
        {"$group": {
            "_id": {"societe": "$Societe", "line": "$CodeRoute", "bus": "$CodeBus",
                    "driver": "$CodeCh", "day": "$jour_service"},
            "date_debut_service": {"$min": "$date_debut_service"},
            "heure_debut_ticket": {"$min": "$heure_debut_ticket"},
            "date_fin_service": {"$max": "$date_fin_service"},
            "heure_fin_ticket": {"$max": "$heure_fin_ticket"},
            "n_tickets": {"$sum": 1},
        }}
    ]

    agg: dict = {}
    n_null_societe = n_unmatched_company = n_unmatched_line = n_null_driver = 0
    for yr in years:
        if yr not in tk_db.list_collection_names():
            continue
        for row in tk_db[yr].aggregate(pipeline, allowDiskUse=True):
            key_raw = row["_id"]
            raw_soc = key_raw.get("societe")
            if not raw_soc:
                n_null_societe += 1
                continue
            canon = alias_to_canon.get(raw_soc)
            company_id = company_id_by_name.get(canon) if canon else None
            if company_id is None:
                n_unmatched_company += 1
                continue
            line_code = str(key_raw.get("line", "")).strip()
            line_id = line_id_by_key.get((line_code, canon))
            if line_id is None:
                n_unmatched_line += 1
                continue
            driver = str(key_raw.get("driver") or "").strip()
            if not driver:
                n_null_driver += 1
                continue
            bus = str(key_raw.get("bus", "")).strip()
            parts = str(key_raw.get("day", "")).split("/")
            day = f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}" if len(parts) == 3 else None
            if day is None:
                continue
            d0, h0 = row.get("date_debut_service"), row.get("heure_debut_ticket")
            d1, h1 = row.get("date_fin_service"), row.get("heure_fin_ticket")
            if not (d0 and h0 and d1 and h1):
                continue
            service_start = f"{d0.replace('/', '-')} {h0}:00"
            service_end = f"{d1.replace('/', '-')} {h1}:00"
            key = (company_id, line_id, bus, day, driver)
            entry = agg.setdefault(key, {"service_start": service_start, "service_end": service_end, "n_tickets": 0})
            entry["service_start"] = min(entry["service_start"], service_start)
            entry["service_end"] = max(entry["service_end"], service_end)
            entry["n_tickets"] += int(row.get("n_tickets") or 0)

    conn.execute("DELETE FROM driver_services")
    conn.executemany(
        """INSERT INTO driver_services (company_id, line_id, bus, day, driver_code,
                                         service_start, service_end, n_tickets)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(cid, lid, bus, day, driver, v["service_start"], v["service_end"], v["n_tickets"])
         for (cid, lid, bus, day, driver), v in agg.items()],
    )
    conn.commit()
    return {"rows_inserted": len(agg), "years": years, "n_null_societe": n_null_societe,
            "n_unmatched_company": n_unmatched_company, "n_unmatched_line": n_unmatched_line,
            "n_null_driver": n_null_driver}


def attach_driver_codes_to_trips(conn: sqlite3.Connection) -> dict:
    """Rattache `driver_code` à chaque trajet de `trips` par recoupement de fenêtre horaire
    avec `driver_services` : un trajet est attribué au chauffeur dont la session (même
    société/ligne/bus/jour) englobe son `trip_start`. Approximatif (fenêtre dérivée des
    horodatages de tickets, pas d'un événement d'ouverture/fermeture explicite) mais c'est
    la seule source qui existe -- voir la note sur `driver_services` dans SCHEMA_SQL.

    NULL laissé tel quel pour un trajet sans session correspondante (pas de tickets
    Historique_Tickets sur ce bus-jour, hors fenêtre d'années ingérée, ou aucune session ne
    couvre exactement ce trip_start) -- pas de repli deviné.
    """
    conn.execute(
        """UPDATE trips SET driver_code = (
             SELECT ds.driver_code FROM driver_services ds
             WHERE ds.company_id = trips.company_id AND ds.line_id = trips.line_id
               AND ds.bus = trips.bus AND ds.day = trips.day
               AND trips.trip_start BETWEEN ds.service_start AND ds.service_end
             LIMIT 1
           )
           WHERE EXISTS (
             SELECT 1 FROM driver_services ds
             WHERE ds.company_id = trips.company_id AND ds.line_id = trips.line_id
               AND ds.bus = trips.bus AND ds.day = trips.day
               AND trips.trip_start BETWEEN ds.service_start AND ds.service_end
           )"""
    )
    conn.commit()
    n_attached = conn.execute("SELECT COUNT(*) FROM trips WHERE driver_code IS NOT NULL").fetchone()[0]
    n_total = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    return {"n_attached": n_attached, "n_total": n_total}


# Export reconstitue la forme `foundation_arrivals_full.parquet` depuis trips/trip_stops
def export_foundation_parquet(conn: sqlite3.Connection, out_path) -> dict:
    """Reconstruit un parquet avec EXACTEMENT les colonnes de l'ancien
    `foundation_arrivals_full.parquet` (day, line, societe, bus, trip_id, dir, full,
    trip_start, trip_end, seq, route_seq, stop, arrival, departure, dwell_s, dark_s,
    had_gap, dist_m, matched) mais depuis `trips`/`trip_stops` (géométrie line_stops,
    match_rate corrigé, routes en boucle correctement marquées `full=NULL`).

    Rétrocompatible par construction : `src/data/delay.py`, `src/data/fallback.py`,
    `src/data/anomaly.py` et `src/train_pipeline.py` lisent tous ce chemin sans
    modification -- ne change QUE la source de données, pas le code des modules.
    `trip_id` utilise l'ID SQL globalement unique de `trips` (au lieu de l'ancien entier
    local par bus-jour) -- fonctionnellement équivalent pour tout groupby sur TRIP_KEYS
    puisqu'il reste unique par trajet.
    """
    import pandas as pd

    df = pd.read_sql("""
        SELECT t.day, l.line_code AS line, c.canonical_name AS societe, t.bus,
               t.trip_id, t.dir, t.is_full AS full, t.trip_start, t.trip_end,
               COALESCE(t.terminus_idle_min, 0) AS terminus_idle_min,
               -- max_dark_s DEJA CORRIGE (voir anomaly.trip_features) : ré-exposé ici sous
               -- trip_dark_s, répété par ligne, pour que le second appel à trip_features() au
               -- moment de l'entraînement (sur CE parquet) réapplique la même correction au
               -- lieu de retomber sur le scan par-arrêt seul (qui ratait les trous EN ROUTE).
               COALESCE(t.max_dark_s, 0) AS trip_dark_s,
               t.trip_dark_before_stop, t.trip_dark_after_stop,
               -- Stationnement terminus détaillé (voir foundation.segment_trips) : séparé
               -- origine/fin + nom du terminus + horodatage réel, pour nommer LEQUEL des
               -- deux termini et donner l'heure de départ/arrivée réelle dans le diagnostic.
               COALESCE(t.origin_idle_min, 0) AS origin_idle_min,
               t.origin_idle_from, t.origin_idle_stop,
               COALESCE(t.end_idle_min, 0) AS end_idle_min,
               t.end_idle_to, t.end_idle_stop,
               ts.seq, ts.seq AS route_seq, s.primary_name AS stop,
               ts.arrival, ts.departure, ts.dwell_s, ts.dark_s, ts.had_gap,
               ts.dist_m, ts.matched
        FROM trip_stops ts
        JOIN trips t ON t.trip_id = ts.trip_id
        JOIN lines l ON l.line_id = t.line_id
        JOIN companies c ON c.company_id = t.company_id
        JOIN stops s ON s.stop_id = ts.stop_id
        ORDER BY t.trip_id, ts.seq
    """, conn)

    # format='mixed' -- str(timestamp) omet parfois les microsecondes quand elles sont nulles,
    # donnant un mélange de formats dans la même colonne (ex. "...11:03:13" vs "...11:03:13.500000")
    df["arrival"] = pd.to_datetime(df["arrival"], format="mixed")
    df["departure"] = pd.to_datetime(df["departure"], format="mixed")
    df["trip_start"] = pd.to_datetime(df["trip_start"], format="mixed")
    df["trip_end"] = pd.to_datetime(df["trip_end"], format="mixed")
    df["origin_idle_from"] = pd.to_datetime(df["origin_idle_from"], format="mixed")
    df["end_idle_to"] = pd.to_datetime(df["end_idle_to"], format="mixed")
    df["matched"] = df["matched"].astype(bool)
    df["had_gap"] = df["had_gap"].astype(bool)
    # `full` reste nullable (None = ligne en boucle, voir foundation.detect_loop_route) --
    # PAS de cast en bool ici, ça écraserait le signal "inconnu" en False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return {"rows": len(df), "trips": df["trip_id"].nunique(), "lines": df["line"].nunique(),
            "companies": df["societe"].nunique(), "out_path": str(out_path)}


_FOUNDATION_SLICE_SQL = """
    SELECT t.day, l.line_code AS line, c.canonical_name AS societe, t.bus,
           t.trip_id, t.dir, t.is_full AS full, t.trip_start, t.trip_end,
           COALESCE(t.terminus_idle_min, 0) AS terminus_idle_min,
           COALESCE(t.max_dark_s, 0) AS trip_dark_s,
           t.trip_dark_before_stop, t.trip_dark_after_stop,
           COALESCE(t.origin_idle_min, 0) AS origin_idle_min,
           t.origin_idle_from, t.origin_idle_stop,
           COALESCE(t.end_idle_min, 0) AS end_idle_min,
           t.end_idle_to, t.end_idle_stop,
           ts.seq, ts.seq AS route_seq, s.primary_name AS stop,
           ts.arrival, ts.departure, ts.dwell_s, ts.dark_s, ts.had_gap,
           ts.dist_m, ts.matched
    FROM trip_stops ts
    JOIN trips t ON t.trip_id = ts.trip_id
    JOIN lines l ON l.line_id = t.line_id
    JOIN companies c ON c.company_id = t.company_id
    JOIN stops s ON s.stop_id = ts.stop_id
"""


def query_foundation_slice(conn: sqlite3.Connection, societe: str, line: str = None,
                           bus=None, day: str = None, trip_id: int = None) -> pd.DataFrame:
    """Même forme EXACTE que `foundation_arrivals_full.parquet` (voir
    `export_foundation_parquet`), mais interrogée À LA DEMANDE pour un périmètre précis au
    lieu de charger les 637k lignes en mémoire en permanence -- voir la discussion
    2026-07-13 : `foundation_arrivals_full.parquet` a été retiré du déploiement Render slim
    (486 Mo même réduit, encore trop pour 512 Mo) alors que `trips`/`trip_stops` (la même
    donnée, dérivée en SQL) étaient déjà dans `winicari_reference_slim.db`, déjà poussée en
    production -- inutile de choisir entre "moins d'historique" et "pas de détail du tout",
    la BDD relationnelle EST la source, on peut simplement l'interroger au lieu de tout
    garder en RAM.

    `societe` est obligatoire (toutes les requêtes existantes filtraient déjà au minimum par
    société) pour ne jamais scanner toute la table par erreur. Index voir
    `idx_trips_scope`/`idx_trips_day` dans `init_db`.
    """
    import pandas as pd

    where = ["c.canonical_name = ?"]
    params: list = [societe]
    if line is not None:
        where.append("l.line_code = ?"); params.append(line)
    if bus is not None:
        where.append("t.bus = ?"); params.append(str(bus))
    if day is not None:
        where.append("t.day = ?"); params.append(day)
    if trip_id is not None:
        where.append("t.trip_id = ?"); params.append(int(trip_id))

    df = pd.read_sql(_FOUNDATION_SLICE_SQL + " WHERE " + " AND ".join(where) +
                     " ORDER BY t.trip_id, ts.seq", conn, params=params)
    if len(df) == 0:
        return df
    for col in ("arrival", "departure", "trip_start", "trip_end", "origin_idle_from", "end_idle_to"):
        df[col] = pd.to_datetime(df[col], format="mixed")
    df["matched"] = df["matched"].astype(bool)
    df["had_gap"] = df["had_gap"].astype(bool)
    # `full` reste nullable (None = ligne en boucle) -- pas de cast direct en bool, ça
    # écraserait le signal "inconnu" en False (même remarque que export_foundation_parquet)
    df["full"] = df["full"].map({1: True, 0: False}, na_action="ignore")
    df["bus"] = df["bus"].astype(int)
    return df


def query_trip_scopes(conn: sqlite3.Connection, societe: str = None, line: str = None) -> pd.DataFrame:
    """Trip-level metadata only (societe, line, bus, day, dir) -- no `trip_stops` join, for
    the "list available lines/buses/days" endpoints (options, lines-ranked, buses-for-line,
    etc.) that never needed the per-stop detail `query_foundation_slice` joins in for.
    Reads `trips` alone (47k rows, vs the 558k-row `trip_stops` join) -- `societe` is
    optional here (unlike `query_foundation_slice`) since listing "all companies" is a
    real, lightweight use case at this grain.
    """
    import pandas as pd

    where, params = [], []
    if societe is not None:
        where.append("c.canonical_name = ?"); params.append(societe)
    if line is not None:
        where.append("l.line_code = ?"); params.append(line)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    return pd.read_sql(f"""
        SELECT c.canonical_name AS societe, l.line_code AS line, t.bus, t.day, t.dir, t.trip_id
        FROM trips t
        JOIN lines l ON l.line_id = t.line_id
        JOIN companies c ON c.company_id = t.company_id
        {where_sql}
    """, conn, params=params).assign(bus=lambda d: d["bus"].astype(int))
