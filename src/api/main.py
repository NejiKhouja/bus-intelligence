"""FastAPI serving layer for WiniCari AI"""
from __future__ import annotations

import functools
import gzip
import json
import logging
import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import model modules
from src.models import delay, gps_fallback, anomaly, chatbot, ticket_anomaly
from src.data import foundation as fdn
from src.data import reference_db as rdb
from src.data import model_version as mv
from src.data import webservices as ws
from src.data.db import get_db

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
api_logger = logging.getLogger("winicari.api")
api_logger.setLevel(logging.INFO)
if not api_logger.handlers:
    _handler = RotatingFileHandler(_LOG_DIR / "api.log", maxBytes=10_000_000, backupCount=5)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    api_logger.addHandler(_handler)

_process_start_time = time.time()
_request_count = 0
_model_version_info = mv.read_version_file()
_app_ready = False

# Déploiement config lue depuis l'environnement (voir docs/DEPLOYMENT.md)
API_KEY = os.getenv("API_KEY")  # None = auth désactivée (dev local uniquement)
ENABLE_CHATBOT = os.getenv("ENABLE_CHATBOT", "false").lower() == "true"

# Sélection des modules à charger : ENABLED_MODULES="anomaly" ou "delay,anomaly" ou "all".
# Défaut = comportement historique (tout sauf le chatbot, qui reste piloté par
# ENABLE_CHATBOT pour compatibilité). Un module non chargé ne coûte AUCUNE RAM : les
# imports lourds (torch, prophet, chromadb...) sont locaux aux fonctions load() de chaque
# module — c'est ce qui permet un déploiement anomaly-only sur 512 Mo (voir DEPLOYMENT.md).
ALL_MODULES = ("delay", "fallback", "anomaly", "ticket_anomaly", "chatbot")
_modules_env = os.getenv("ENABLED_MODULES", "").strip().lower()
if _modules_env in ("", None):
    ENABLED_MODULES = {"delay", "fallback", "anomaly", "ticket_anomaly"} | (
        {"chatbot"} if ENABLE_CHATBOT else set())
elif _modules_env == "all":
    ENABLED_MODULES = set(ALL_MODULES)
else:
    ENABLED_MODULES = {m.strip() for m in _modules_env.split(",") if m.strip()}
    _unknown = ENABLED_MODULES - set(ALL_MODULES)
    if _unknown:
        print(f"  ! ENABLED_MODULES contient des modules inconnus, ignorés : {sorted(_unknown)} "
              f"(valides : {', '.join(ALL_MODULES)})")
        ENABLED_MODULES -= _unknown
    if ENABLE_CHATBOT:
        ENABLED_MODULES.add("chatbot")

_allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if _allowed_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = ["*"]
    print("  ! ALLOWED_ORIGINS non défini -- CORS grand ouvert (\"*\"). "
          "À restreindre au domaine réel avant la mise en production (voir docs/DEPLOYMENT.md).")


# Pydantic Models
class DelayPredictionManualRequest(BaseModel):
    societe: str
    line: str
    direction: str
    dep_time: str
    current_seq: int = Field(ge=0)
    current_delay_min: float
    model_type: str = "hgbm"

class GPSFallbackRequest(BaseModel):
    day: str
    line: str
    societe: str
    bus: int
    query_time: str

class AnomalyRequest(BaseModel):
    day: str
    line: str
    societe: str
    bus: Optional[int] = None

class GpsPingRow(BaseModel):
    """One raw GPS ping. Filled in by whoever fetched them (your own code calling
    getPingsForDay, see docs/WEBSERVICES_NEEDED.md service 1) -- this endpoint never
    talks to MongoDB itself, same principle as TicketDayRow below. `t` is an ISO
    timestamp string; `voyage` is the raw service.voyage field when your source has
    it (used to disambiguate ALLER/RETOUR), omit it if unavailable."""
    t: str
    lat: float
    lon: float
    speed: Optional[float] = None
    voyage: Optional[int] = None

class AnomalyLiveScoreRequest(BaseModel):
    day: str
    line: str
    societe: str
    bus: int
    pings: List[GpsPingRow]

class TicketDayRow(BaseModel):
    """One (societe, line, bus, day) ticket-sales total -- same grain/fields as
    winicari.details, already aggregated. Filled in by whoever fetched them (your own
    code calling a company webservice, or the company pushing them directly) -- this
    endpoint never talks to a database itself, see docs/WEBSERVICES_NEEDED.md."""
    societe: str
    line: str
    bus: str
    day: str
    nbr_ticket: int
    recette: float

class TicketAnomalyScoreRequest(BaseModel):
    rows: List[TicketDayRow]

class ChatbotRequest(BaseModel):
    query: str
    k: int = 5

class ListResponse(BaseModel):
    companies: List[str]
    lines: List[str]
    directions: List[str]
    buses: List[int]
    days: List[str]

# Model Manager
class ModelManager:
    _instance = None
    _models = {}
    _stops_data = {}
    _stop_coords = {}   # f"{societe}_{line}" -> {stop_name: (lat, lon)}
    _stop_coord_suspect = {}   # f"{societe}_{line}" -> {stop_name: bool, chronically wrong coords}
    _gps_trip_counts = {}   # societe -> {(line, str(bus), day): n_trips}, see get_gps_trip_count
    _latest_day = None
    _trips_count = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_all(self):
        print("Loading models...")

        # Trip/stop detail (route_seq<->stop, lat/lon, "today", per-trip sequences) used to
        # come from foundation_arrivals_full.parquet loaded whole into memory (~486MB even
        # after shrinking -- see git history 2026-07-13) -- it never needed to be a
        # permanently-resident DataFrame, since trips/trip_stops (the same data, and the
        # ACTUAL source _load_usable_lines/line_stops already reads for route geometry) are
        # already in the reference DB. Per-request detail now comes from
        # `_foundation_slice()` (queries trips/trip_stops on demand, see
        # reference_db.query_foundation_slice); what's built here at startup is just the
        # small, line-geometry-shaped pieces (stop names/order/coords), from `line_stops`
        # directly -- not per-trip observations, and not requiring foundation data at all.
        try:
            usable = _load_usable_lines()
            for (line_code, soc), sf in usable.items():
                key = f"{soc}_{line_code}"
                self._stops_data[key] = (
                    sf.sort_values("route_seq")[["route_seq", "name"]]
                    .rename(columns={"name": "stop"}).drop_duplicates().to_dict("records"))
                self._stop_coords[key] = {
                    str(row["name"]): (float(row["lat"]), float(row["lon"]))
                    for _, row in sf.iterrows()
                    if pd.notna(row.get("lat")) and pd.notna(row.get("lon"))
                }
            print(f"Stops mapping + coordinates built for {len(self._stops_data)} lines "
                  f"(from line_stops, no foundation data needed)")
        except Exception as e:
            print(f"Stops mapping/coordinates not available: {e}")

        # Demo clock ("today" = most recent day with real data) + trips count for /health --
        # both computed once here rather than per-request (health checks can be frequent).
        try:
            conn = rdb.init_db()
            try:
                row = conn.execute("SELECT MAX(day), COUNT(*) FROM trips").fetchone()
            finally:
                conn.close()
            self._latest_day = str(row[0]) if row and row[0] else None
            self._trips_count = int(row[1]) if row and row[1] else 0
            print(f"Demo 'today' set to latest data day: {self._latest_day} "
                  f"({self._trips_count:,} trips in reference DB)")
        except Exception as e:
            print(f"Failed to determine latest day/trips count: {e}")

        loaders = {"delay": delay.load, "fallback": gps_fallback.load,
                   "anomaly": anomaly.load, "ticket_anomaly": ticket_anomaly.load,
                   "chatbot": chatbot.load}
        disabled = [m for m in ALL_MODULES if m not in ENABLED_MODULES]
        if disabled:
            print(f"  - Modules désactivés (ENABLED_MODULES) : {', '.join(disabled)} "
                  f"-- non chargés, leurs endpoints répondront 503")

        for model_name in ALL_MODULES:
            if model_name not in ENABLED_MODULES:
                continue
            try:
                self._models[model_name] = loaders[model_name]()
                print(f"{model_name.capitalize()} models loaded")
            except Exception as e:
                print(f"Failed to load {model_name} models: {e}")

        return self._models

    def get(self, model_name: str) -> Dict:
        if model_name not in self._models:
            if model_name not in ENABLED_MODULES:
                raise KeyError(f"Module '{model_name}' désactivé sur ce déploiement "
                               f"(ENABLED_MODULES={','.join(sorted(ENABLED_MODULES))})")
            raise KeyError(f"Model '{model_name}' not loaded")
        return self._models[model_name]

    def foundation_slice(self, societe: str, line: str = None, bus=None,
                        day: str = None, trip_id: int = None,
                        include_live: bool = True) -> Optional[pd.DataFrame]:
        """Trip/stop detail for a scoped request, queried on demand from `trips`/
        `trip_stops` (see `reference_db.query_foundation_slice`) -- replaces the old
        permanently-resident `foundation_arrivals_full.parquet` DataFrame (see git history
        2026-07-13: it cost ~486MB even after shrinking, more than this deployment's whole
        512MB budget, while this same data was already sitting in the reference DB).
        Returns `None` only if the reference DB itself can't be reached (never "empty" --
        an empty-but-real result is an empty DataFrame, not None; keeps the same
        `if df is None` guard callers already used for "data unavailable" everywhere).

        Also merges in yesterday's LIVE per-stop data when relevant (`day` unspecified or
        equal to the live day) -- see _score_all_gps_live. Without this, "view details"/the
        map for a trip flagged from live data would come up empty: that trip was never
        written to trips/trip_stops, it only exists in the in-memory live cache.

        `include_live=False` skips that merge entirely -- for a caller that's explicitly
        after HISTORICAL/typical behaviour (e.g. `reference_trip`, hunting for a normal
        PAST trip, never today's), `day=None` would otherwise trigger a full company-wide
        live-GPS webservice fetch + trip reconstruction + scoring for NO benefit, on top of
        the line's own (potentially large) historical slice -- confirmed 2026-07-19: OOM on
        `/api/reference-trip` for a big line (S.R.T.K/212, 2000+ trips), the two costs
        stacking in one request.
        """
        try:
            conn = rdb.init_db()
            try:
                result = rdb.query_foundation_slice(conn, societe, line=line, bus=bus,
                                                    day=day, trip_id=trip_id)
            finally:
                conn.close()
        except Exception as e:
            print(f"foundation_slice query failed: {e}")
            return None

        potential_live_day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        if include_live and (day is None or day == potential_live_day):
            live = _score_all_gps_live(societe, potential_live_day)
            if live is not None and len(live["stops"]):
                live_stops = live["stops"]
                if line:
                    live_stops = live_stops[live_stops["line"] == line]
                if bus is not None:
                    live_stops = live_stops[live_stops["bus"].astype(str) == str(bus)]
                if day == potential_live_day:
                    result = live_stops
                elif len(live_stops):
                    result = pd.concat([result, live_stops], ignore_index=True, sort=False)

        # `fa` (live, from reconstruct_bus_day) carries these as object-dtype Timestamp
        # scalars, not datetime64 -- concat with the DB-sourced (already datetime64) result,
        # or even the live-only case alone, can leave the column as `object` rather than a
        # proper datetime dtype. Confirmed by a real crash (2026-07-13):
        # "TypeError: unsupported operand type(s) for -: 'numpy.ndarray' and 'Timestamp'"
        # in _trip_sequence's `df["trip_start"] - ts`. Normalize once here so every caller
        # gets a consistent dtype regardless of source (DB/live/merged).
        if result is not None and len(result):
            for col in ("arrival", "departure", "trip_start", "trip_end",
                       "origin_idle_from", "end_idle_to"):
                if col in result.columns:
                    result[col] = pd.to_datetime(result[col], errors="coerce")
        return result

    def trip_scopes(self, societe: str = None, line: str = None) -> Optional[pd.DataFrame]:
        """Lightweight trip metadata (societe/line/bus/day/dir, no per-stop join) for the
        "list available X" endpoints -- see `reference_db.query_trip_scopes`. Same `None`
        convention as `foundation_slice`: only for an unreachable DB, not an empty result.
        """
        try:
            conn = rdb.init_db()
            try:
                return rdb.query_trip_scopes(conn, societe=societe, line=line)
            finally:
                conn.close()
        except Exception as e:
            print(f"trip_scopes query failed: {e}")
            return None

    def get_stops(self, societe: str, line: str) -> List[Dict]:
        key = f"{societe}_{line}"
        return self._stops_data.get(key, [])

    def get_stop_coords(self, societe: str, line: str) -> dict:
        """Maps stop_name -> (lat, lon) for a given line."""
        return self._stop_coords.get(f"{societe}_{line}", {})

    def get_gps_trip_count(self, societe: str, line: str, bus: str, day: str) -> Optional[int]:
        """Nombre de trajets GPS réels pour ce (société, ligne, bus, jour) exact -- ou
        `None` si le cross-check GPS n'est pas disponible DU TOUT sur ce déploiement (la
        BDD de référence est absente/inaccessible). `None` DOIT rester distinct de `0` :
        `0` veut dire "vérifié, aucun trajet GPS ce jour-là" (vrai signal, voir
        is_no_service dans `_ticket_rows_with_reasons`), alors que `None` veut dire "pas
        vérifiable ici" -- les confondre ferait passer CHAQUE jour-bus signalé pour "pas
        d'anomalie réelle" et les ferait tous disparaître à tort de la vue client.

        Sert à distinguer une panne de machine à tickets (le bus a bien circulé --
        confirmé côté GPS -- mais quasi aucun ticket enregistré) d'une vraie anomalie de
        recette/fraude. Mis en cache PAR SOCIÉTÉ (pas globalement) : une requête SQL sur
        les trajets de cette société au premier appel, puis des lectures dict pour tous
        les jours-bus suivants de la même société -- pas de requête par ligne du tableau.
        """
        if societe not in self._gps_trip_counts:
            fa = self.foundation_slice(societe)
            if fa is None:
                return None  # BDD injoignable -- pas vérifiable, distinct de "0 trajet"
            if len(fa) == 0:
                self._gps_trip_counts[societe] = {}
            else:
                counts = (fa.assign(bus=fa["bus"].astype(str))
                          .groupby(["line", "bus", "day"], observed=True)["trip_id"].nunique())
                self._gps_trip_counts[societe] = counts.to_dict()
        return int(self._gps_trip_counts[societe].get((line, str(bus), day), 0))

    def get_coord_suspect(self, societe: str, line: str) -> dict:
        """Maps stop_name -> True when its geocoded coordinates are chronically wrong.

        A raw match-rate threshold alone misses the mixed case: a stop matched on ~40%
        of trips but tens of km off the other ~60% (confirmed on S.R.T.K "EL GARAA" --
        the same stop name resolves to a good coordinate on some lines but one that's
        tens of km off on most others, and lands at ~40% match / ~11-20 km miss distance
        on a couple of lines in between). A bare match-rate cutoff misses that middle
        ground; requiring a large mean miss-distance too catches it -- but only once
        there are enough misses to trust the average (n_unmatched >= 10), otherwise a
        line with just a handful of observations and one stray far ping (e.g. a partial
        trip fragment) gets wrongly condemned.
        """
        key = f"{societe}_{line}"
        if key not in self._stop_coord_suspect:
            sub = self.foundation_slice(societe, line=line)
            if sub is None or len(sub) == 0:
                return {}
            grp = sub.groupby("stop", observed=True).agg(
                match_rate=("matched", "mean"), n=("matched", "size"))
            unmatched = sub[~sub["matched"]].groupby("stop", observed=True)["dist_m"]
            grp = grp.join(unmatched.mean().rename("miss_dist_m"))
            grp = grp.join(unmatched.size().rename("n_unmatched"))
            self._stop_coord_suspect[key] = {
                str(stop): bool(row["match_rate"] < 0.10
                               or (row["match_rate"] < 0.50
                                   and row.get("n_unmatched", 0) >= 10
                                   and pd.notna(row["miss_dist_m"])
                                   and row["miss_dist_m"] > 3000))
                for stop, row in grp.iterrows() if row["n"] >= 20
            }
        return self._stop_coord_suspect.get(key, {})

    def get_latest_day(self) -> Optional[str]:
        """Demo 'today' -- most recent day present in `trips`, computed once at startup."""
        return self._latest_day

    def get_trips_count(self) -> int:
        return self._trips_count

    def is_loaded(self) -> bool:
        return len(self._models) > 0

    def get_loaded_models(self) -> List[str]:
        return list(self._models.keys())

model_manager = ModelManager()

_usable_lines_cache: Optional[dict] = None


def _load_usable_lines() -> dict:
    """Line/stop geometry {(line_code, societe) -> stops_frame}, from the reference DB
    (data/reference/winicari_reference.db). Built once and cached for the app's lifetime --
    replaces the old per-call rebuild from live MongoDB via `foundation.build_usable_lines`.
    """
    global _usable_lines_cache
    if _usable_lines_cache is None:
        conn = rdb.init_db()
        try:
            _usable_lines_cache = rdb._usable_lines_from_line_stops(conn)
        finally:
            conn.close()
    return _usable_lines_cache

# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_ready  
    print("=" * 60)
    print("WiniCari AI API Starting...")
    print("=" * 60)
    model_manager.load_all()
    _app_ready = True  
    print("Application ready to serve requests")  
    yield
    print("Shutting down...")

def _sanitize_non_finite(obj):
    """Recursively replaces NaN/inf/-inf with `None` -- strict JSON doesn't allow them
    (confirmed by a real crash, 2026-07-14: "ValueError: Out of range float values are not
    JSON compliant" from /api/ticket-anomaly-history, traced to a z-score/if_score blowing
    up on a freshly live-scored day with too little history for a stable scaler std). Fixed
    at the serialization boundary rather than chasing every individual field that computes
    a ratio/z-score/model score -- this guards the whole API against the same class of bug
    wherever it next shows up, not just this one endpoint.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


class SanitizedJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_sanitize_non_finite(content))


app = FastAPI(
    title="WiniCari AI API",
    description="API for bus delay prediction, GPS fallback, anomaly detection, and RAG chatbot",
    version="2.0.0",
    lifespan=lifespan,
    default_response_class=SanitizedJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if API_KEY:
    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        """Simple shared-secret auth for /api/* routes (see docs/PHP_INTEGRATION.md) --
        `/`, `/health`, and the auto-generated docs stay open for uptime monitors."""
        if request.url.path.startswith("/api/") and request.headers.get("X-API-Key") != API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid X-API-Key"})
        return await call_next(request)
else:
    print("  ! API_KEY non défini -- authentification désactivée (dev local uniquement). "
          "À définir avant la mise en production (voir docs/DEPLOYMENT.md).")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Journalise chaque requête (méthode/chemin/params/statut/latence) dans logs/api.log
    et compte les requêtes pour /health -- voir la section journalisation en tête de fichier."""
    global _request_count
    _request_count += 1
    t0 = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - t0) * 1000, 1)
    api_logger.info(
        f"{request.method} {request.url.path} query={dict(request.query_params)} "
        f"status={response.status_code} latency_ms={latency_ms} "
        f"model_commit={(_model_version_info or {}).get('git_commit', 'n/a')}"
    )
    return response

# Limite la concurrence des handlers les plus lourds (scans historique/live à l'échelle
# d'une société entière) -- Starlette envoie les `def` (voir get_anomaly_history) dans son
# threadpool SANS plafond propre, donc plusieurs visiteurs qui arrivent à la même minute
# (plusieurs opérateurs qui ouvrent le widget) peuvent chacun garder des DataFrames
# entiers en mémoire EN MÊME TEMPS et dépasser les 512MB de l'instance même si AUCUNE
# requête individuelle n'est anormalement grosse (constaté 2026-07-20 : OOM immédiatement
# après une ligne de log "webservice injoignable" qui, elle, est bénigne -- le disjoncteur
# de src/data/webservices.py la rend quasi instantanée après le 1er échec ; la vraie cause
# est la charge concurrente, pas cette requête-là). Au-delà de la limite, on attend
# quelques secondes puis on échoue vite avec un 503 réessayable plutôt que de s'entasser
# et de se faire tuer silencieusement par l'OOM killer.
_HEAVY_ENDPOINT_CONCURRENCY = int(os.getenv("HEAVY_ENDPOINT_CONCURRENCY", "2"))
_heavy_endpoint_semaphore = threading.BoundedSemaphore(_HEAVY_ENDPOINT_CONCURRENCY)


def _limit_concurrency(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _heavy_endpoint_semaphore.acquire(timeout=8):
            raise HTTPException(status_code=503,
                                detail="Serveur chargé, réessayez dans quelques secondes.")
        try:
            return fn(*args, **kwargs)
        finally:
            _heavy_endpoint_semaphore.release()
    return wrapper

# Helper Functions
def get_available_options(societe: Optional[str] = None, line: Optional[str] = None) -> Dict:
    df = model_manager.trip_scopes(societe=None)  # companies list always needs the full scope
    if df is None or len(df) == 0:
        return {"companies": [], "lines": [], "directions": [], "buses": [], "days": []}

    result = {}
    result["companies"] = sorted(df["societe"].unique().tolist())

    df_filtered = df
    if societe:
        df_filtered = df_filtered[df_filtered["societe"] == societe]
    result["lines"] = sorted(df_filtered["line"].unique().tolist())

    if line:
        df_filtered = df_filtered[df_filtered["line"] == line]
    result["directions"] = sorted(df_filtered["dir"].unique().tolist())
    result["buses"] = sorted(df_filtered["bus"].unique().tolist())
    days = sorted(df_filtered["day"].unique().tolist(), reverse=True)
    result["days"] = days[:30]

    return result

# Health & Options Endpoints
@app.get("/")
@app.get("/health")
async def health_check():
    if not _app_ready:  
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "detail": "Models still loading"}
        )
    return {
        "status": "healthy" if _app_ready else "starting",
        "ready": _app_ready,
        "models_loaded": model_manager.is_loaded() if _app_ready else [],
        "models": model_manager.get_loaded_models() if _app_ready else [],
        "enabled_modules": sorted(ENABLED_MODULES),
        "foundation_data": model_manager.get_latest_day() is not None if _app_ready else False,
        "rows": model_manager.get_trips_count() if _app_ready else 0,
        "latest_day": model_manager.get_latest_day() if _app_ready else None,
        "timestamp": datetime.now().isoformat(),
        "model_version": _model_version_info,
        "uptime_seconds": round(time.time() - _process_start_time, 1),
        "request_count": _request_count,
    }



def demo_today() -> str:
    """Demo 'today' — the latest day with real data (data ends 2026-06-21)."""
    return model_manager.get_latest_day() or datetime.now().strftime("%Y%m%d")


def latest_day_for(societe: str, line: Optional[str] = None) -> str:
    """Most recent day a given operator/line actually ran (for live demos).

    The literal last calendar day is sparse, so live views snap to the most recent
    day the selected scope has real trips — keeping the 'today' demo populated.
    """
    df = model_manager.trip_scopes(societe, line=line)
    if df is None or len(df) == 0:
        return demo_today()
    days = df["day"]
    return str(days.max()) if len(days) else demo_today()


# ─────────────────────────────────────────────────────────────────────────────
# Scoring EN DIRECT depuis le web service GPS (pas MongoDB) pour "aujourd'hui" (voir
# docs/webservice_fields.txt) -- "Trajets signalés" doit refléter le VRAI jour courant, pas
# le dernier jour du dataset précalculé (toujours en retard, voir demo_today() -- constaté
# 2026-07-13 : dataset arrêté au 21 juin alors que le service GPS a des données du 12
# juillet). `_current_gps_cache` évite de refaire l'appel webservice + la reconstruction
# (coûteuse : projection + segmentation par bus) à chaque rerun Streamlit -- TTL court,
# pas un cache de longue durée comme les artefacts entraînés.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Magasin de données EN DIRECT POUSSÉES (relais) -- les webservices de la plateforme
# tournent sur un réseau local SANS URL publique : un serveur cloud (Render, Allemagne)
# ne peut pas les atteindre (timeout, constaté 2026-07-17), et l'utilisateur n'a pas la
# main sur le serveur des webservices. Solution : inverser le sens -- un petit script
# relais (scripts/push_live_day.py) tourne sur une machine DU réseau local (elle, elle
# atteint les webservices), tire la journée et la POUSSE ici via POST /api/ingest/*
# (sortie HTTPS simple, marche depuis n'importe quel réseau). Les scoreurs live lisent
# ce magasin D'ABORD, le webservice ensuite (le pull direct reste le chemin naturel en
# dev local, où le réseau le permet).
# Stocké sur disque (gzip) : survit aux redémarrages du process sur la même instance ;
# sur une instance neuve le magasin repart vide et se remplit au prochain push -- même
# philosophie éphémère que le reste des artefacts non committés.
# ─────────────────────────────────────────────────────────────────────────────
_INGEST_DIR = Path("data/live_ingest")
_INGEST_KEEP_DAYS = 3


def _ingest_safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


def _ingest_path(kind: str, day: str, societe: str = None) -> Path:
    name = f"{kind}_{_ingest_safe(societe)}_{day}.json.gz" if societe else f"{kind}_{day}.json.gz"
    return _INGEST_DIR / name


def _ingest_write(path: Path, rows: list) -> None:
    _INGEST_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    # Rétention courte : seule la journée "live" (hier) sert au scoring ; au-delà, les
    # données sont déjà couvertes par le pipeline historique au prochain réentraînement.
    cutoff = (datetime.now() - timedelta(days=_INGEST_KEEP_DAYS)).strftime("%Y%m%d")
    for old in _INGEST_DIR.glob("*.json.gz"):
        day_part = old.stem.replace(".json", "").rsplit("_", 1)[-1]
        if day_part.isdigit() and day_part < cutoff:
            try:
                old.unlink()
            except OSError:
                pass


def _ingest_read(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ingest illisible ({path.name}): {e}")
        return None


_current_gps_cache: dict = {}   # (societe, day) -> (timestamp, {"trips":.., "stops":..} ou None)
_CURRENT_GPS_CACHE_TTL_S = 90
# Nombre max d'entrées retenues par cache de scoring en direct. Une entrée GPS porte les
# DataFrames trips+stops d'une journée entière d'une société -- sans borne, naviguer entre
# plusieurs sociétés accumule tout (11 sociétés possibles) et grignote la RAM de l'instance
# 512MB déjà tendue (voir l'OOM du scoring live, 2026-07-17). Éviction du plus ancien.
_LIVE_CACHE_MAX_ENTRIES = 3


def _live_cache_put(cache: dict, key, value) -> None:
    cache[key] = (time.time(), value)
    while len(cache) > _LIVE_CACHE_MAX_ENTRIES:
        oldest = min(cache, key=lambda k: cache[k][0])
        del cache[oldest]


def _score_all_gps_live(societe: str, day: str) -> Optional[dict]:
    """Reconstruit + score TOUS les (ligne, bus) que le web service GPS rapporte pour cette
    société/ce jour -- même pipeline que /api/anomaly/score-live, juste appliqué à chaque
    bus-jour trouvé plutôt qu'à un seul. Toujours au grain (societe, day) -- PAS filtré par
    ligne ici (un seul appel webservice couvre déjà toutes les lignes de la société ce
    jour-là ; filtrer par ligne serait juste un `df[df.line==...]` après coup, pas la peine
    de refaire l'appel/la reconstruction par ligne).

    Retourne {"trips": <1 ligne/trajet, même forme que models["trips"]>,
              "stops": <1 ligne/arrêt, même forme que query_foundation_slice>} ou `None` si
    le web service n'est pas configuré/joignable/pas prêt -- l'appelant retombe alors sur la
    vue historique. `stops` est ce qui manquait à la version précédente : sans lui, "voir le
    détail"/la carte d'un trajet EN DIRECT n'avait nulle part où lire la séquence par arrêt
    (jamais écrite dans trips/trip_stops, ce jour n'existant que dans ce cache en mémoire).
    """
    cache_key = (societe, day)
    cached = _current_gps_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CURRENT_GPS_CACHE_TTL_S:
        return cached[1]

    result = None
    # Magasin poussé D'ABORD (aucun réseau, voir le relais scripts/push_live_day.py),
    # webservice ensuite (chemin naturel en dev local où le réseau le permet).
    pings = _ingest_read(_ingest_path("gps", day, societe))
    if pings is None and ws.WEBSERVICE_URL:
        try:
            if ws.is_day_ready(day):
                pings = ws.get_pings_for_day(day, societe=societe)
        except Exception as e:
            print(f"Live GPS webservice unavailable ({societe}/{day}): {e}")
    if pings:
        try:
            groups = ws.group_pings_by_bus_line(pings)
            usable = _load_usable_lines()
            models = model_manager.get("anomaly")
            cfg = fdn.Config()
            trip_frames, stop_frames = [], []
            for (grp_line, grp_bus), grp_pings in groups.items():
                key = (grp_line, societe)
                if key not in usable:
                    continue
                rows = ws.pings_to_score_live_rows(grp_pings)
                if not rows:
                    continue
                raw_df = pd.DataFrame(rows)
                # Le service renvoie des horodatages avec offset (+00:00) -> tz-aware,
                # alors que le reste du pipeline (day0 = pd.Timestamp(...) dans
                # reconstruct_bus_day) compare avec des Timestamp NAÏFS, comme les
                # datetime bruts de pymongo -- confirmé par erreur réelle (2026-07-13,
                # "Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp").
                # On retire juste l'info de fuseau (les valeurs restent identiques),
                # pas de vraie conversion de fuseau nécessaire.
                raw_df["t"] = pd.to_datetime(raw_df["t"]).dt.tz_localize(None)
                raw_df = raw_df.sort_values("t").reset_index(drop=True)
                try:
                    fa = fdn.reconstruct_bus_day(
                        None, f"d{day}", grp_line, societe, int(grp_bus),
                        usable[key], cfg, raw_pings=raw_df)
                except Exception as e:
                    print(f"  live reconstruct failed for {societe}/{grp_line}/{grp_bus}: {e}")
                    continue
                if fa.empty:
                    continue
                try:
                    trip_frames.append(anomaly.score(models, fa))
                except Exception as e:
                    print(f"  live scoring failed for {societe}/{grp_line}/{grp_bus}: {e}")
                    continue
                stop_frames.append(fa)
            if trip_frames:
                result = {
                    "trips": pd.concat(trip_frames, ignore_index=True),
                    "stops": pd.concat(stop_frames, ignore_index=True),
                }
        except Exception as e:
            print(f"Live GPS scoring failed ({societe}/{day}): {e}")

    _live_cache_put(_current_gps_cache, cache_key, result)
    return result


def _live_gps_day(societe: str) -> Optional[str]:
    """Yesterday's date if the live GPS web service actually has (usable) data for this
    société, else `None` -- see _score_all_gps_live for why "yesterday" not "today"."""
    day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    return day if _score_all_gps_live(societe, day) is not None else None


class GpsDayIngest(BaseModel):
    day: str = Field(..., description="YYYYMMDD")
    societe: str
    pings: List[Dict[str, Any]] = Field(..., description="pings compacts: line/bus/t/lat/lon/speed/voyage")


class TicketDayIngest(BaseModel):
    day: str = Field(..., description="YYYYMMDD")
    rows: List[Dict[str, Any]] = Field(..., description="lignes brutes de getTicketTotalsForDay (toutes sociétés)")


@app.post("/api/ingest/gps-day")
async def ingest_gps_day(body: GpsDayIngest):
    """Reçoit la journée GPS d'une société, POUSSÉE par le relais tournant sur le réseau
    local de la plateforme (scripts/push_live_day.py) -- ce serveur ne peut pas atteindre
    les webservices lui-même (réseau privé sans URL publique, voir _INGEST_DIR). Protégé
    par la même X-API-Key que le reste de /api/*."""
    if not (len(body.day) == 8 and body.day.isdigit()):
        raise HTTPException(status_code=422, detail="day doit être au format YYYYMMDD")
    keep = [
        {k: p.get(k) for k in ("line", "bus", "t", "lat", "lon", "speed", "voyage")}
        for p in body.pings
        if p.get("line") and p.get("bus") and p.get("t") and p.get("lat") is not None and p.get("lon") is not None
    ]
    _ingest_write(_ingest_path("gps", body.day, body.societe), keep)
    # Invalider le cache de scoring pour que le prochain affichage reparte des données
    # fraîchement poussées (sinon un résultat None mis en cache 90s masquerait le push).
    _current_gps_cache.pop((body.societe, body.day), None)
    return {"stored": len(keep), "dropped": len(body.pings) - len(keep),
            "societe": body.societe, "day": body.day}


@app.get("/api/ingest/status")
async def ingest_status():
    """Ce que le magasin poussé contient (jour/société/taille/date de push) -- permet au
    déclencheur côté page (embed/relay.php en mode auto) de savoir si la journée d'hier
    est déjà là AVANT de lancer un push : indispensable parce que ce magasin est
    ÉPHÉMÈRE (instance Render redémarrée = magasin vide) -- un marqueur local côté PHP
    dirait "déjà poussé" à tort, seul CE serveur sait ce qu'il a réellement."""
    items = []
    if _INGEST_DIR.exists():
        for p in sorted(_INGEST_DIR.glob("*.json.gz")):
            stem = p.name[:-8]
            parts = stem.split("_")
            items.append({
                "kind": parts[0],
                "societe": "_".join(parts[1:-1]) or None,
                "day": parts[-1],
                "size": p.stat().st_size,
                "pushed_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
    return {"files": items}


@app.post("/api/ingest/ticket-day")
async def ingest_ticket_day(body: TicketDayIngest):
    """Reçoit les totaux billetterie d'une journée (toutes sociétés, forme brute de
    getTicketTotalsForDay), poussés par le même relais que /api/ingest/gps-day."""
    if not (len(body.day) == 8 and body.day.isdigit()):
        raise HTTPException(status_code=422, detail="day doit être au format YYYYMMDD")
    _ingest_write(_ingest_path("tickets", body.day), body.rows)
    for key in [k for k in _current_ticket_cache if k[1] == body.day]:
        _current_ticket_cache.pop(key, None)
    return {"stored": len(body.rows), "day": body.day}


_current_ticket_cache: dict = {}   # (societe, day) -> (timestamp, days_df ou None)


def _score_all_tickets_live(societe: str, day: str) -> Optional[pd.DataFrame]:
    """Score la billetterie EN DIRECT depuis getTicketTotalsForDay pour cette société/ce
    jour -- même pipeline que /api/ticket-anomaly/score-live, appliqué à toutes les lignes
    de la société d'un coup. `day` au format YYYYMMDD (converti en YYYY-MM-DD pour l'appel,
    voir docs/webservice_fields.txt -- format différent de getPingsForDay, confirmé sur le
    service réel). Pas de isDayReady équivalent fourni pour ce service -- on tente l'appel
    directement et on traite une réponse vide/l'échec comme "pas encore prêt".
    """
    cache_key = (societe, day)
    cached = _current_ticket_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CURRENT_GPS_CACHE_TTL_S:
        return cached[1]

    result = None
    # Magasin poussé d'abord (fichier = journée entière toutes sociétés, comme la réponse
    # du service), webservice ensuite -- même logique que _score_all_gps_live.
    raw = _ingest_read(_ingest_path("tickets", day))
    if raw is None and ws.WEBSERVICE_URL:
        try:
            day_dashed = f"{day[:4]}-{day[4:6]}-{day[6:]}"
            raw = ws.get_ticket_totals_for_day(day_dashed)
        except Exception as e:
            print(f"Live ticket webservice unavailable ({societe}/{day}): {e}")
    if raw:
        try:
            sub = [r for r in raw if r.get("societe") == societe]
            rows = ws.ticket_totals_to_rows(sub, day)
            if rows:
                models = model_manager.get("ticket_anomaly")
                result = ticket_anomaly.score(models, pd.DataFrame(rows))
        except Exception as e:
            print(f"Live ticket scoring failed ({societe}/{day}): {e}")

    _live_cache_put(_current_ticket_cache, cache_key, result)
    return result


def _live_ticket_day(societe: str) -> Optional[str]:
    """Yesterday's date if the live ticket web service has data for this société, else
    `None` -- same "yesterday, not today" reasoning as _live_gps_day."""
    day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    return day if _score_all_tickets_live(societe, day) is not None else None


@app.get("/api/options", response_model=ListResponse)
async def get_options(
    societe: Optional[str] = Query(None),
    line: Optional[str] = Query(None)
):
    options = get_available_options(societe, line)
    return ListResponse(**options)

@app.get("/api/lines")
async def get_lines(societe: str):
    options = get_available_options(societe=societe)
    return {"lines": options["lines"]}

@app.get("/api/lines-ranked")
async def get_lines_ranked(societe: str):
    """Lines for an operator, busiest first — so demos default to a line with data."""
    df = model_manager.trip_scopes(societe)
    if df is None or len(df) == 0:
        return {"lines": []}
    counts = df.groupby("line")["trip_id"].nunique().sort_values(ascending=False)
    return {"lines": counts.index.tolist()}

@app.get("/api/directions")
async def get_directions(societe: str, line: str):
    options = get_available_options(societe=societe, line=line)
    return {"directions": options["directions"]}

@app.get("/api/prophet-lines")
async def get_prophet_lines(societe: str):
    """Only the lines/directions that actually have a trained Prophet model."""
    try:
        prophet = model_manager.get("delay").get("prophet", {})
    except KeyError:
        return {"societe": societe, "lines": [], "by_line": {}}
    df = model_manager.trip_scopes(societe)
    if df is None or len(df) == 0:
        return {"societe": societe, "lines": [], "by_line": {}}
    sub = df[["line", "dir"]].drop_duplicates()
    by_line: Dict[str, list] = {}
    for _, r in sub.iterrows():
        if f"{societe}_{r['line']}_{r['dir']}" in prophet:
            by_line.setdefault(r["line"], []).append(r["dir"])
    return {"societe": societe, "lines": sorted(by_line.keys()),
            "by_line": {k: sorted(v) for k, v in by_line.items()}}

@app.get("/api/buses")
async def get_buses(societe: str, line: str):
    options = get_available_options(societe=societe, line=line)
    return {"buses": options["buses"]}

@app.get("/api/days")
async def get_days(
    societe: Optional[str] = Query(None), 
    line: Optional[str] = Query(None)
):
    df = model_manager.trip_scopes(societe or None, line=line or None)
    if df is None or len(df) == 0:
        return {"days": []}
    days = sorted(df["day"].unique().tolist(), reverse=True)
    return {"days": days[:30]}

@app.get("/api/buses-for-line")
async def get_buses_for_line(societe: str, line: str):
    """All unique buses that have ever run on a given line (across all days in the foundation)."""
    df = model_manager.trip_scopes(societe, line=line)
    if df is None or len(df) == 0:
        return {"buses": []}
    buses = sorted(int(b) for b in df["bus"].unique())
    return {"buses": buses}


@app.get("/api/buses-for-day")
async def get_buses_for_day(
    societe: str,
    line: str,
    day: str
):
    df = model_manager.trip_scopes(societe, line=line)
    if df is None or len(df) == 0:
        return {"buses": []}
    buses = sorted(df.loc[df["day"] == day, "bus"].unique().tolist())
    return {"buses": buses}

@app.get("/api/days-for-line")
async def get_days_for_line(
    societe: str,
    line: str
):
    df = model_manager.trip_scopes(societe, line=line)
    days = sorted(df["day"].unique().tolist(), reverse=True) if df is not None and len(df) else []

    # Yesterday, if the live GPS web service actually has trips for THIS line -- otherwise
    # "Expliquer un bus" only ever offers historical days, even when live data exists (see
    # _live_gps_day) -- an admin has no way to pick it explicitly.
    live_day = _live_gps_day(societe)
    if live_day and live_day not in days:
        live = _score_all_gps_live(societe, live_day)
        if live is not None and len(live["trips"][live["trips"]["line"] == line]):
            days = [live_day] + days

    return {"days": days[:30]}

@app.get("/api/stops")
async def get_stops(
    societe: str,
    line: str
):
    """Get all stops for a line with their sequences."""
    stops = model_manager.get_stops(societe, line)
    return {"stops": stops}

@app.get("/api/route-info")
async def get_route_info(
    societe: str,
    line: str,
    day: str,
    bus: int
):
    """Get complete route information for a bus."""
    bus_data = model_manager.foundation_slice(societe, line=line, bus=bus, day=day)
    if bus_data is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    bus_data = bus_data.sort_values("trip_start")

    if len(bus_data) == 0:
        raise HTTPException(status_code=404, detail="No data found for this bus")
    
    # Get the latest trip
    bus_data["trip_end_dt"] = pd.to_datetime(bus_data["trip_end"])
    latest_trip = bus_data.loc[bus_data["trip_end_dt"].idxmax()]
    trip_id = latest_trip["trip_id"]
    trip_data = bus_data[bus_data["trip_id"] == trip_id].sort_values("seq")
    
    # Get stops with sequence
    stops = []
    for _, row in trip_data.iterrows():
        stop_info = {
            "seq": int(row["seq"]),
            "stop": row["stop"],
            "arrival": row["arrival"].isoformat() if pd.notna(row["arrival"]) else None,
            "departure": row["departure"].isoformat() if pd.notna(row["departure"]) else None,
            "dwell_s": float(row["dwell_s"]) if pd.notna(row["dwell_s"]) else 0,
            "matched": bool(row["matched"])
        }
        stops.append(stop_info)
    
    # Find current position (last arrived stop)
    arrived = [s for s in stops if s["arrival"] is not None]
    current_stop = arrived[-1] if arrived else None
    next_stop = None
    
    if current_stop:
        current_idx = current_stop["seq"]
        next_stops = [s for s in stops if s["seq"] > current_idx and s["matched"]]
        next_stop = next_stops[0] if next_stops else None
    
    return {
        "trip_id": int(trip_id),
        "direction": latest_trip["dir"],
        "trip_start": latest_trip["trip_start"].isoformat(),
        "trip_end": latest_trip["trip_end"].isoformat(),
        "total_stops": len(stops),
        "stops": stops,
        "current_stop": current_stop,
        "next_stop": next_stop,
        "first_stop": stops[0] if stops else None,
        "last_stop": stops[-1] if stops else None,
        "bus": bus,
        "line": line,
        "societe": societe
    }

@app.get("/api/bus-status")
async def get_bus_status(
    societe: str,
    line: str,
    bus: int,
    day: str
):
    """Get current status of a bus with stop names."""
    bus_data = model_manager.foundation_slice(societe, line=line, bus=bus, day=day)
    if bus_data is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    bus_data = bus_data.sort_values("trip_start")

    if len(bus_data) == 0:
        raise HTTPException(status_code=404, detail="No data found for this bus")
    
    # Get the most recent completed trip
    bus_data["trip_end_dt"] = pd.to_datetime(bus_data["trip_end"])
    latest_trip = bus_data.loc[bus_data["trip_end_dt"].idxmax()]
    trip_id = latest_trip["trip_id"]
    trip_data = bus_data[bus_data["trip_id"] == trip_id].sort_values("seq")
    
    # Find the last stop the bus arrived at
    arrived_stops = trip_data[trip_data["arrival"].notna()]
    
    if len(arrived_stops) > 0:
        last_stop = arrived_stops.iloc[-1]
        current_seq = int(last_stop["seq"])
        current_stop = last_stop["stop"]
        last_arrival = last_stop["arrival"]
        
        # Calculate delay
        trip_start = pd.Timestamp(trip_data.iloc[0]["trip_start"])
        expected_min = current_seq * 2.5
        expected_time = trip_start + pd.Timedelta(minutes=expected_min)
        delay = (pd.Timestamp(last_arrival) - expected_time).total_seconds() / 60
        
        # Get all stops for this trip
        all_stops = trip_data[["seq", "stop", "arrival", "departure"]].to_dict('records')
        
        return {
            "trip_id": int(trip_id),
            "current_seq": current_seq,
            "current_stop": current_stop,
            "last_arrival": last_arrival.isoformat(),
            "current_delay_min": round(max(0, delay), 1),
            "trip_start": trip_data.iloc[0]["trip_start"].isoformat(),
            "trip_end": trip_data.iloc[-1]["trip_end"].isoformat(),
            "total_stops": len(trip_data),
            "remaining_stops": len(trip_data) - current_seq - 1,
            "direction": trip_data.iloc[0]["dir"],
            "all_stops": all_stops
        }
    else:
        return {
            "trip_id": int(trip_id),
            "current_seq": -1,
            "current_stop": "Not started",
            "last_arrival": None,
            "current_delay_min": 0,
            "trip_start": trip_data.iloc[0]["trip_start"].isoformat(),
            "trip_end": trip_data.iloc[-1]["trip_end"].isoformat(),
            "total_stops": len(trip_data),
            "remaining_stops": len(trip_data),
            "direction": trip_data.iloc[0]["dir"],
            "all_stops": trip_data[["seq", "stop", "arrival", "departure"]].to_dict('records')
        }

# Anomaly Detection — served from precomputed trip scores, with explainability
# ── Classification qualité des trajets (bug de données vs artefact de suivi vs réel) ──
# Trois classes masquées par défaut côté service (l'admin les affiche via
# include_data_bugs=true) ; ce qui reste visible est jugeable en confiance :
#
#   is_data_bug      -- durée > 24h : PROUVABLEMENT corrompu (les pings viennent d'UNE
#                       collection journalière ; constaté : « trajet » de 4 668h dans
#                       d20230603). Correction racine dans foundation.reconstruct_bus_day
#                       (filtre fenêtre-jour) -- effective au prochain rebuild.
#   is_fragment      -- durée < 30% de la médiane de la ligne ET <= 4 arrêts suivis : le
#                       GPS n'a vu qu'un éclat du service ; incomparable à un trajet complet.
#   is_dark_inflated -- durée > 1.5x la médiane de la ligne ET dont > 50% est un trou de
#                       signal : le bus a pu finir normalement traceur éteint -- durée non
#                       fiable. NB : un trajet long dont l'excès est CORROBORÉ par une
#                       immobilisation observée (pings présents, bus immobile, ex. 204 min
#                       à SFAX) reste AFFICHÉ -- c'est une vraie anomalie opérationnelle,
#                       pas un artefact.
#   is_implausible   -- durée > 2x la médiane de la ligne SANS observation GPS qui la
#                       corrobore (immobilisation + perte de signal observées < 50% de
#                       l'excès) : l'excès s'est produit hors de tout arrêt reconnu --
#                       stationnement terminus/dépôt fondu dans le trajet (corrigé à la
#                       racine par le rognage/la coupe stationnaire de foundation.py au
#                       prochain rebuild). Un trajet long dont l'excès EST corroboré
#                       (ex. 204 min immobile observées à SFAX) reste affiché : c'est un
#                       vrai incident, précisément ce que le module doit montrer.
#   is_partial_coverage -- nettement moins d'arrêts couverts que la normale pour cette
#                       ligne/direction (ex. 5 arrêts vs 11 médians) : le bus a réellement
#                       parcouru une distance plus courte, donc une durée plus courte est
#                       NORMALE pour CE trajet -- ce n'est pas une anomalie de vitesse.
#                       Détecté empiriquement (2026-07-08, ligne 212/S.R.T.K) : les trajets
#                       à faible n_stops ont une durée médiane ~153 min contre ~210 min pour
#                       les trajets à couverture complète -- les comparer à la médiane pleine
#                       ligne les signalait à tort comme « anormalement rapides ».
DATA_BUG_TRIP_MIN = 24 * 60.0
FRAGMENT_FRAC = 0.30
FRAGMENT_MAX_STOPS = 4
DARK_INFLATED_MIN_FRAC = 1.5
DARK_DOMINANT_SHARE = 0.5
IMPLAUSIBLE_MIN_FRAC = 2.0
CORROBORATION_SHARE = 0.5
PARTIAL_COVERAGE_FRAC = 0.6

_line_median: Optional[pd.DataFrame] = None


def _trip_quality_flags(trips: pd.DataFrame, models) -> pd.DataFrame:
    """Ajoute is_data_bug / is_fragment / is_dark_inflated / is_implausible /
    is_partial_coverage (voir constantes ci-dessus).

    Médianes calculées PAR DIRECTION (societe, line, dir) -- un trajet ALLER et son RETOUR
    peuvent couvrir des distances/temps différents ; les mélanger biaiserait la comparaison
    dans les deux sens.
    """
    global _line_median
    if _line_median is None:
        allt = models["trips"]
        ok = allt[allt["total_elapsed"] <= DATA_BUG_TRIP_MIN]
        # Un trajet PARTIEL (couverture réduite, voir `full` dans foundation.segment_trips)
        # a intrinsèquement une durée/un nombre d'arrêts plus faible qu'un trajet complet --
        # confirmé empiriquement : médiane 30.6 min / 4 arrêts (partiel) contre 39.0 min /
        # 14 arrêts (complet). Les mélanger dans la référence de la ligne tire la médiane vers
        # le bas et fausse is_fragment/is_dark_inflated/is_implausible/is_partial_coverage --
        # un trajet complet un peu long se comparerait à tort à une référence "partielle".
        # Repli sur tous les trajets (comme `full` en boucle = None) si trop peu de trajets
        # complets pour une (societe, line, dir) donnée -- même seuil que `reference_trip`.
        full_only = ok[ok["full"] == True]  # noqa: E712 -- `full` est nullable (lignes en boucle)
        med_full = (full_only.groupby(["societe", "line", "dir"])
                    .agg(line_median_elapsed=("total_elapsed", "median"),
                         line_median_n_stops=("n_stops", "median"),
                         n_full=("total_elapsed", "size"))
                    .reset_index())
        med_all = (ok.groupby(["societe", "line", "dir"])
                   .agg(line_median_elapsed=("total_elapsed", "median"),
                        line_median_n_stops=("n_stops", "median"))
                   .reset_index())
        merged = med_all.merge(med_full, on=["societe", "line", "dir"], how="left",
                               suffixes=("_all", "_full"))
        enough_full = merged["n_full"].fillna(0) >= 3
        merged["line_median_elapsed"] = merged["line_median_elapsed_full"].where(
            enough_full, merged["line_median_elapsed_all"])
        merged["line_median_n_stops"] = merged["line_median_n_stops_full"].where(
            enough_full, merged["line_median_n_stops_all"])
        _line_median = merged[["societe", "line", "dir", "line_median_elapsed", "line_median_n_stops"]]
    t = trips.merge(_line_median, on=["societe", "line", "dir"], how="left")
    med = t["line_median_elapsed"]
    med_stops = t["line_median_n_stops"]
    dark_min = t.get("max_dark_s", pd.Series(0.0, index=t.index)).fillna(0) / 60
    t["is_data_bug"] = t["total_elapsed"] > DATA_BUG_TRIP_MIN
    t["is_fragment"] = (~t["is_data_bug"] & med.notna()
                        & (t["total_elapsed"] < FRAGMENT_FRAC * med)
                        & (t["n_stops"] <= FRAGMENT_MAX_STOPS))
    # Distinction clé : l'immobilisation OBSERVÉE (pings présents, bus immobile) corrobore
    # une durée longue -- un vrai incident. Un trou de signal ne corrobore RIEN : c'est du
    # temps inconnu. Excès expliqué surtout par du noir -> durée non fiable (dark_inflated) ;
    # excès expliqué par rien du tout -> stationnement fondu dans le trajet (implausible).
    dwell_min = t.get("max_dwell_s", pd.Series(0.0, index=t.index)).fillna(0) / 60
    excess = t["total_elapsed"] - med
    long_trip = med.notna() & (t["total_elapsed"] > DARK_INFLATED_MIN_FRAC * med)
    t["is_dark_inflated"] = (~t["is_data_bug"] & long_trip
                             & ((dark_min > DARK_DOMINANT_SHARE * t["total_elapsed"])
                                | (dark_min > CORROBORATION_SHARE * excess)))
    t["is_implausible"] = (~t["is_data_bug"] & ~t["is_dark_inflated"] & med.notna()
                           & (t["total_elapsed"] > IMPLAUSIBLE_MIN_FRAC * med)
                           & (dwell_min < CORROBORATION_SHARE * excess))
    # Restreint aux trajets pas plus longs que la médiane pleine ligne : moins d'arrêts
    # explique une durée COURTE, pas une durée LONGUE -- un trajet à couverture réduite qui
    # est QUAND MÊME anormalement long (ex. immobilisé sur son court trajet) reste une vraie
    # anomalie et ne doit pas être masqué sous prétexte de couverture partielle.
    t["is_partial_coverage"] = (~t["is_data_bug"] & ~t["is_fragment"] & med_stops.notna()
                                & (t["n_stops"] < PARTIAL_COVERAGE_FRAC * med_stops)
                                & (med.notna() & (t["total_elapsed"] <= med)))
    return t


def _filter_trips(societe: str, line: Optional[str] = None,
                  bus: Optional[int] = None, day: Optional[str] = None,
                  include_data_bugs: bool = False, dir: Optional[str] = None,
                  include_live: bool = True):
    """Filter the scored-trips table -- precomputed (historical) PLUS, when relevant,
    yesterday's data scored live from the GPS web service (see _score_all_gps_live).

    Only checks live data when it could actually matter (`day` unspecified, i.e. "show
    everything", or `day` explicitly equal to yesterday) -- an explicit OTHER historical
    day never needs a webservice round-trip. `day=None` merges live trips in alongside
    history (so "Historique récent"/patterns naturally include yesterday); `day=<live
    day>` returns ONLY the live trips for that exact day (an explicit pick).

    `include_live=False` skips the merge outright, for callers that are explicitly after
    HISTORICAL/typical behaviour and never today's (e.g. `reference_trip`) -- avoids
    paying for a full company-wide live-GPS fetch + reconstruction for no benefit. See the
    matching note on `ModelManager.foundation_slice`.
    """
    models = model_manager.get("anomaly")
    trips = models["trips"]

    potential_live_day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    if include_live and (day is None or day == potential_live_day):
        live = _score_all_gps_live(societe, potential_live_day)
        if live is not None:
            if day == potential_live_day:
                trips = live["trips"]
            else:
                trips = pd.concat([trips, live["trips"]], ignore_index=True, sort=False)
            # Même normalisation que foundation_slice -- `trip_features()` sur des données
            # EN DIRECT peut laisser trip_start/trip_end en dtype object plutôt que
            # datetime64, cassant les comparaisons/isoformat() en aval (confirmé par un
            # crash réel, voir la note dans foundation_slice).
            if len(trips):
                trips = trips.copy()
                for col in ("trip_start", "trip_end"):
                    if col in trips.columns:
                        trips[col] = pd.to_datetime(trips[col], errors="coerce")

    mask = trips["societe"] == societe
    if line:
        mask &= trips["line"] == line
    if bus is not None:
        mask &= trips["bus"].astype(str) == str(bus)
    if day:
        mask &= trips["day"] == day
    if dir:
        mask &= trips["dir"] == dir
    trips = _trip_quality_flags(trips[mask].copy(), models)
    if not include_data_bugs:
        # Décision utilisateur (2026-07-07) : les trajets longs douteux (dark_inflated,
        # implausible) restent VISIBLES par défaut, avec leur explication -- masqués sans
        # explication, l'admin conclut que le modèle/la segmentation est cassé ; expliqués
        # (« le boîtier GPS a continué d'émettre après la fin du service »), ils deviennent
        # un signal opérationnel compréhensible. Seuls restent masqués : les bugs de données
        # prouvés (>24h), les fragments trop courts pour être jugés, et les trajets à
        # couverture partielle (2026-07-08 : durée courte non comparable à un trajet complet,
        # voir is_partial_coverage -- pas une vraie anomalie, juste une distance différente).
        trips = trips[~(trips["is_data_bug"] | trips["is_fragment"] | trips["is_partial_coverage"])]
    return models, trips


def _rows_with_reasons(models, trips, anomalies_only: bool = True,
                       limit: Optional[int] = None) -> List[Dict]:
    """Attach human-readable reasons to (anomalous) trips and serialize to dicts."""
    if anomalies_only:
        trips = trips[trips["anomaly"]]
    if len(trips) == 0:
        return []
    # Trié par JOUR d'abord (le plus récent en tête), gravité en départage -- pas l'inverse.
    # `limit` coupe la liste APRÈS ce tri ; trier par seule gravité pouvait faire disparaître
    # une anomalie du jour même (ex. la seule anomalie "en direct" du jour) d'une liste de
    # 300 si des milliers d'anomalies historiques plus sévères passaient devant elle --
    # constaté 2026-07-20 : le bandeau de fraîcheur comptait 1 anomalie "aujourd'hui" côté
    # /api/current-anomalies, introuvable dans la liste fusionnée /api/anomaly-history
    # pourtant censée déjà l'inclure (voir _filter_trips, day=None -> merge live). Les deux
    # interfaces (dashboard Streamlit ET widget embarqué) affichent de toute façon le plus
    # récent en premier par défaut -- ce tri correspond à ce qu'elles montrent réellement.
    ex = anomaly.explain_trips(models, trips).sort_values(
        ["day", "anomaly_strength"], ascending=[False, False])
    if limit:
        ex = ex.head(limit)

    # Isolation Forest et l'autoencodeur LSTM ont 3 paliers de spécificité (voir
    # models/anomaly.py train()) : dédié à la (société, ligne) si assez de trajets sur
    # CETTE ligne (>=30 pour l'IF, >=200 pour le LSTM) -- sinon dédié à l'opérateur entier
    # (mêmes seuils) -- sinon repli sur un modèle GLOBAL entraîné sur tout le réseau.
    # `model_line_dedicated` distingue le meilleur cas (comparé à cette ligne précisément)
    # du cas intermédiaire (comparé à l'opérateur entier, toutes lignes poolées), pour que
    # le dashboard puisse nuancer son avertissement au lieu de traiter les deux pareil.
    if_models = models.get("if_models", {})
    lstm_models = models.get("lstm_models") or {}
    # LSTM absent ≠ LSTM pas assez entraîné : le déploiement slim (Render) n'installe PAS
    # torch, donc lstm_models est VIDE quel que soit l'historique -- avant ce correctif,
    # une ligne avec largement assez de trajets (constaté 2026-07-18 : S.R.T.K/202, 876
    # trajets, modèle LSTM dédié bien présent dans les artefacts) affichait quand même
    # "pas assez de données pour un LSTM dédié". Quand le LSTM est indisponible au niveau
    # du DÉPLOIEMENT, il est exclu du calcul de model_low_data au lieu d'être imputé aux
    # données ; les scores LSTM précalculés de l'entraînement restent servis tels quels.
    lstm_enabled = bool(lstm_models)
    driver_lookup = _driver_code_lookup()
    schedules = _scheduled_departures()

    rows = []
    for _, r in ex.iterrows():
        soc = r.get("societe")
        line = r.get("line")
        if_line_dedicated = (soc, line) in if_models
        lstm_line_dedicated = lstm_enabled and (soc, line) in lstm_models
        if_dedicated = if_line_dedicated or soc in if_models
        lstm_dedicated = lstm_enabled and (lstm_line_dedicated or soc in lstm_models)
        driver_code = driver_lookup.get((soc, line, str(int(r["bus"])), r["day"], r["trip_start"]))

        # Départ prévu (horaire publié, voir _scheduled_departures) vs départ réel
        # (trip_start, déjà net de l'immobilisation terminus -- voir foundation.segment_trips)
        # -- purement informationnel (pas une feature entraînée), disponible seulement pour
        # les lignes qui publient un horaire (45.6% des lignes suivies, confirmé 2026-07-15).
        scheduled_departure = departure_delay_min = None
        schedule_multi_variant = False
        sched = schedules.get((soc, line))
        ts = r.get("trip_start")
        if sched and pd.notna(ts):
            scheduled_departure = sched.get(r.get("dir"))
            if scheduled_departure:
                schedule_multi_variant = bool(sched.get("multi_variant"))
                sh, sm = int(scheduled_departure[:2]), int(scheduled_departure[3:5])
                sched_min = sh * 60 + sm
                actual_min = ts.hour * 60 + ts.minute
                delta = actual_min - sched_min
                if delta > 720:
                    delta -= 1440
                elif delta < -720:
                    delta += 1440
                departure_delay_min = delta

        rows.append({
            "driver_code": driver_code,
            "scheduled_departure": scheduled_departure,
            "departure_delay_min": departure_delay_min,
            "schedule_multi_variant": schedule_multi_variant,
            "model_if_dedicated": bool(if_dedicated),
            "model_lstm_dedicated": bool(lstm_dedicated),
            "model_lstm_enabled": lstm_enabled,
            "model_line_dedicated": bool(if_line_dedicated and (lstm_line_dedicated or not lstm_enabled)),
            "model_low_data": not (if_dedicated and (lstm_dedicated or not lstm_enabled)),
            "societe": soc,
            "day": r["day"],
            "trip_id": int(r["trip_id"]),
            "bus": int(r["bus"]),
            "line": r["line"],
            "dir": r["dir"],
            "if_score": float(r["if_score"]),
            "lstm_score": float(r["lstm_score"]),
            "dual_anomaly": bool(r["dual_anomaly"]),
            "severity": str(r["severity"]),
            "anomaly_strength": float(r["anomaly_strength"]),
            "reasons": list(r["reasons"]),
            "reason_features": list(r.get("reason_features", [])),
            "top_feature": r.get("top_feature"),
            "max_dwell_min": round(float(r["max_dwell_s"]) / 60, 1),
            "max_dark_min": round(float(r.get("max_dark_s", 0) or 0) / 60, 1),
            # Arrêts encadrant le plus grand trou de signal EN ROUTE (voir
            # foundation.derive_arrivals / anomaly.trip_features) -- None si le trou (s'il y en
            # a un) est capté par le scan par-arrêt classique (déjà visible via problem_stops).
            "dark_gap_before_stop": (r.get("trip_dark_before_stop")
                                     if pd.notna(r.get("trip_dark_before_stop")) else None),
            "dark_gap_after_stop": (r.get("trip_dark_after_stop")
                                    if pd.notna(r.get("trip_dark_after_stop")) else None),
            "worst_dwell_stop": r.get("worst_dwell_stop"),
            "trip_duration_min": round(float(r["total_elapsed"]), 1),
            "total_elapsed_min": round(float(r["total_elapsed"]), 1),  # kept for back-compat
            "n_stops": int(r["n_stops"]),
            "trip_start": r["trip_start"].isoformat() if pd.notna(r["trip_start"]) else None,
            "trip_end": r["trip_end"].isoformat() if pd.notna(r["trip_end"]) else None,
            # classes qualité (voir _trip_quality_flags) -- visibles seulement quand
            # l'admin a demandé include_data_bugs=true, sinon filtrées en amont.
            # Repli sur le seul critère prouvable quand les colonnes manquent (score-live).
            "is_data_bug": bool(r.get("is_data_bug", float(r["total_elapsed"]) > DATA_BUG_TRIP_MIN)),
            "is_fragment": bool(r.get("is_fragment", False)),
            "is_dark_inflated": bool(r.get("is_dark_inflated", False)),
            "is_implausible": bool(r.get("is_implausible", False)),
            "is_partial_coverage": bool(r.get("is_partial_coverage", False)),
            "line_median_n_stops": (round(float(r["line_median_n_stops"]), 1)
                                    if pd.notna(r.get("line_median_n_stops")) else None),
            # Durée totale moins les blocs immobiles/noirs DÉTECTÉS (max par arrêt) --
            # estimation basse du vrai temps de conduite pour contextualiser les durées
            # gonflées. Le stationnement hors de tout arrêt reconnu reste invisible ici
            # (corrigé à la racine au prochain rebuild).
            "driving_time_est_min": round(max(
                0.0, float(r["total_elapsed"])
                     - float(r["max_dwell_s"]) / 60
                     - float(r.get("max_dark_s", 0) or 0) / 60), 1),
            # médiane de la ligne (jours sains) -- permet au client de ne contextualiser
            # QUE les durées gonflées : soustraire l'immobilisation d'un trajet déjà court
            # produirait un temps « plus rapide que physiquement possible », trompeur
            "line_median_elapsed_min": (round(float(r["line_median_elapsed"]), 1)
                                        if pd.notna(r.get("line_median_elapsed")) else None),
            "match_rate": round(float(r["match_rate"]), 3),
            "terminus_idle_min": round(float(r.get("terminus_idle_min", 0) or 0), 1),
            # Détail par terminus (voir foundation.segment_trips) -- nomme LEQUEL des deux
            # termini et donne l'heure réelle de départ/arrivée, pour remplacer un chiffre
            # sans repère ("84 min au terminus") par quelque chose de vérifiable (« immobile
            # à MAHDIA de 13h54 à 15h18, départ effectif à 15h18 »).
            "origin_idle_min": round(float(r.get("origin_idle_min", 0) or 0), 1),
            "origin_idle_stop": r.get("origin_idle_stop") if pd.notna(r.get("origin_idle_stop")) else None,
            "origin_idle_from": (r["origin_idle_from"].isoformat()
                                 if pd.notna(r.get("origin_idle_from")) else None),
            "end_idle_min": round(float(r.get("end_idle_min", 0) or 0), 1),
            "end_idle_stop": r.get("end_idle_stop") if pd.notna(r.get("end_idle_stop")) else None,
            "end_idle_to": (r["end_idle_to"].isoformat()
                           if pd.notna(r.get("end_idle_to")) else None),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Ticket/billing anomaly -- COMPLEMENTARY signal to the GPS one above, not merged
# with it. Daily grain (societe, line, bus, day), not per-trip -- see
# src/data/ticket_anomaly.py for why. Static history comes from days_scored.parquet
# (computed at training time); score-live takes rows the CALLER already fetched (this
# deployment never talks to MongoDB itself, see docs/WEBSERVICES_NEEDED.md).
# ─────────────────────────────────────────────────────────────────────────────

def _filter_ticket_days(societe: str, line: Optional[str] = None,
                        bus: Optional[str] = None, day: Optional[str] = None):
    """Filter the scored-days table -- precomputed (historical) PLUS, when relevant,
    yesterday's billetterie scored live from the web service (see
    _score_all_tickets_live). Same "only check live when it could matter" reasoning as
    _filter_trips (GPS side)."""
    models = model_manager.get("ticket_anomaly")
    days = models["days"]

    potential_live_day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    if day is None or day == potential_live_day:
        live = _score_all_tickets_live(societe, potential_live_day)
        if live is not None:
            if day == potential_live_day:
                days = live
            else:
                days = pd.concat([days, live], ignore_index=True, sort=False)

    mask = days["societe"] == societe
    if line:
        mask &= days["line"] == line
    if bus is not None:
        mask &= days["bus"].astype(str) == str(bus)
    if day:
        mask &= days["day"] == day
    return models, days[mask].copy()


def _ticket_rows_with_reasons(models, days, anomalies_only: bool = True,
                              limit: Optional[int] = None,
                              client_safe: bool = False) -> List[Dict]:
    """Attach human-readable reasons to (anomalous) ticket-days and serialize to dicts.

    `client_safe` : n'inclut que les jours dont l'anomalie n'est PAS explicable par un
    problème de NOTRE côté -- voir `is_machine_issue`/`is_no_service` ci-dessous. Pensé
    pour une démo/présentation client : un volume de tickets à zéro parce que la machine
    est probablement tombée en panne (le bus a bien roulé, confirmé GPS) n'est pas un
    signal de fraude/recette à montrer comme tel. La vue admin (`client_safe=False`,
    défaut) garde tout, avec ces jours marqués plutôt que cachés.
    """
    if anomalies_only:
        days = days[days["anomaly"]]
    if len(days) == 0:
        return []
    # if_score: higher = more normal (score_samples convention) -> most anomalous first
    ex = ticket_anomaly.explain(models, days).sort_values("if_score")

    def _opt(r, col, ndigits=2):
        v = r.get(col)
        return round(float(v), ndigits) if v is not None and pd.notna(v) else None

    rows = []
    for _, r in ex.iterrows():
        nbr_ticket = int(r["nbr_ticket"])
        # Croisement avec les trajets GPS réels de ce bus-jour EXACT (voir
        # ModelManager.get_gps_trip_count) -- distingue "la machine n'a probablement pas
        # fonctionné" (service confirmé, ~0 ticket) de "aucun service ce jour-là" (pas de
        # trajet GPS du tout -- férié/grève/bus hors service), deux causes différentes,
        # ni l'une ni l'autre une vraie anomalie de recette/fraude.
        gps_trip_count = model_manager.get_gps_trip_count(
            r["societe"], r["line"], str(r["bus"]), str(r["day"]))
        # `None` = cross-check GPS indisponible sur ce déploiement (voir
        # ModelManager.get_gps_trip_count) -- reste `None` ici aussi plutôt que de
        # deviner "pas de service"/"pas de panne" : le confondre avec `0` ferait
        # disparaître CHAQUE jour signalé de la vue client (is_no_service=True partout).
        if gps_trip_count is None:
            is_no_service = None
            is_machine_issue = None
        else:
            is_no_service = gps_trip_count == 0
            # Seuil <=2 plutôt que ==0 : tolère un ticket manuel/réédité isolé sans perdre
            # le signal -- un vrai jour normal a des dizaines/centaines de tickets, pas 1 ou 2.
            is_machine_issue = (not is_no_service) and nbr_ticket <= 2
        if client_safe and (is_machine_issue or is_no_service):
            continue
        # Bonne vs mauvaise anomalie : la recette du jour a-t-elle dépassé la normale de
        # CE bus (repli ligne si pas assez d'historique pour ce bus) ? Une recette
        # au-dessus de la normale reste une anomalie statistique (le modèle la signale
        # quand même) mais n'est PAS un problème -- plus d'argent rentré que d'habitude
        # est une bonne nouvelle, pas un signal à traiter comme la recette anormalement
        # BASSE. None quand aucune référence n'est disponible (pas encore de médiane).
        recette = float(r["recette"])
        recette_baseline = r.get("bus_median_recette")
        if recette_baseline is None or pd.isna(recette_baseline):
            recette_baseline = r.get("line_median_recette")
        is_good_anomaly = (bool(recette > recette_baseline)
                           if recette_baseline is not None and pd.notna(recette_baseline)
                           else None)
        rows.append({
            "societe": r["societe"], "line": r["line"], "bus": str(r["bus"]), "day": r["day"],
            "nbr_ticket": nbr_ticket,
            "recette": round(recette, 2),
            "avg_fare": round(float(r["avg_fare"]), 2),
            "if_score": round(float(r["if_score"]), 3),
            "anomaly": bool(r["anomaly"]),
            "reasons": list(r["reasons"]),
            "severity": r.get("severity", "medium"),
            "gps_trip_count": gps_trip_count,
            "is_machine_issue": is_machine_issue,
            "is_no_service": is_no_service,
            "is_good_anomaly": is_good_anomaly,
            # contexte de jugement : la valeur du jour ne dit rien seule -- les médianes de
            # la ligne/du bus + le taux d'anomalie de la ligne permettent de trancher entre
            # "jour réellement louche" et "ligne structurellement atypique"
            "line_median_nbr_ticket": _opt(r, "line_median_nbr_ticket", 0),
            "line_median_recette": _opt(r, "line_median_recette", 0),
            "line_median_avg_fare": _opt(r, "line_median_avg_fare"),
            "bus_median_nbr_ticket": _opt(r, "bus_median_nbr_ticket", 0),
            "bus_median_recette": _opt(r, "bus_median_recette", 0),
            "bus_median_avg_fare": _opt(r, "bus_median_avg_fare"),
            "line_anomaly_rate": _opt(r, "line_anomaly_rate", 3),
            "z_nbr_ticket": _opt(r, "z_nbr_ticket"),
            "z_recette": _opt(r, "z_recette"),
            "z_avg_fare": _opt(r, "z_avg_fare"),
            "model_tier": r.get("model_tier"),
        })
        if limit and len(rows) >= limit:
            break
    return rows


@app.get("/api/ticket-anomaly-history")
# `def`, pas `async def` -- voir la note dans get_anomaly_history. Constaté ici aussi
# 2026-07-20 : "HTTP health check failed (timed out after 5 seconds)" pendant que l'onglet
# "Anomalies billetterie" de l'embed enchaîne DEUX appels synchrones l'un après l'autre
# (ticket-anomaly-patterns puis ticket-anomaly-history, voir embed/assets/app.js) -- sous
# `async def`, leur temps de blocage s'ADDITIONNE sur l'unique event loop au lieu de
# tourner dans le threadpool de Starlette. Même correctif que les 5 handlers GPS déjà
# convertis, appliqué maintenant aux 5 handlers billetterie équivalents.
@_limit_concurrency
def ticket_anomaly_history(
    societe: str,
    line: Optional[str] = None,
    bus: Optional[str] = None,
    limit: int = 30,
    client_safe: bool = False
):
    """Historical ticket/billing anomalies (precomputed at training time)."""
    try:
        models, days = _filter_ticket_days(societe, line, bus)
        anomalies = _ticket_rows_with_reasons(models, days, anomalies_only=True, limit=limit,
                                              client_safe=client_safe)
        return {"anomalies": anomalies, "total": len(anomalies), "total_days": int(len(days))}
    except KeyError:
        raise HTTPException(status_code=503, detail="Ticket anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ticket anomalies: {str(e)}")


@app.get("/api/ticket-anomaly-explain")
@_limit_concurrency
def ticket_anomaly_explain(  # def, pas async def -- voir la note dans ticket_anomaly_history
    societe: str,
    line: Optional[str] = None,
    bus: Optional[str] = None,
    day: Optional[str] = None,
    client_safe: bool = False
):
    """All ticket-days for a scope, with anomalies flagged + explained."""
    try:
        models, days = _filter_ticket_days(societe, line, bus, day)
        rows = _ticket_rows_with_reasons(models, days, anomalies_only=False, client_safe=client_safe)
        anomalous = [r for r in rows if r["anomaly"]]
        return {
            "societe": societe, "line": line, "bus": bus, "day": day,
            "days": rows, "anomalies": anomalous,
            "total_days": len(rows), "anomaly_count": len(anomalous),
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Ticket anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining ticket anomalies: {str(e)}")


@app.get("/api/ticket-anomaly-patterns")
@_limit_concurrency
def ticket_anomaly_patterns(societe: str, line: Optional[str] = None):  # def, pas async def -- voir ticket_anomaly_history
    """Anomaly-RATE aggregations by line and by bus (no hour/direction -- daily grain)."""
    try:
        _models, days = _filter_ticket_days(societe, line)
        if len(days) == 0:
            return {"societe": societe, "line": line, "total_days": 0,
                    "total_anomalies": 0, "overall_rate": 0.0, "by_line": [], "by_bus": []}

        def rate_by(col):
            g = days.groupby(col).agg(
                days=("anomaly", "size"), anomalies=("anomaly", "sum")).reset_index()
            g["anomalies"] = g["anomalies"].astype(int)
            g["rate"] = (g["anomalies"] / g["days"]).round(3)
            return g

        by_line = rate_by("line").sort_values("rate", ascending=False)
        by_bus = rate_by("bus")
        by_bus = by_bus[by_bus["days"] >= 5].sort_values(
            ["rate", "anomalies"], ascending=False).head(15)

        return {
            "societe": societe, "line": line,
            "total_days": int(len(days)),
            "total_anomalies": int(days["anomaly"].sum()),
            "overall_rate": round(float(days["anomaly"].mean()), 3),
            "by_line": by_line.to_dict("records"),
            "by_bus": by_bus.to_dict("records"),
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Ticket anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing ticket patterns: {str(e)}")


def _resolve_station_coords(societe: str, line: str, names: list) -> dict:
    """Best-effort `station_name -> (lat, lon)` via the SAME per-line stop-coordinate
    lookup already used for the GPS trip map (`ModelManager.get_stop_coords`) -- ticket
    station names (`NomFR1`, see `reference_db.populate_tickets_station_daily`) aren't
    guaranteed to match `stops.primary_name` byte-for-byte (case/accents/whitespace), so
    this degrades gracefully : unresolved stations simply get no pin, the list/table view
    still shows them.
    """
    coords = model_manager.get_stop_coords(societe, line)
    norm = {k.strip().upper(): v for k, v in coords.items()}
    out = {}
    for name in names:
        v = norm.get(str(name).strip().upper())
        if v:
            out[name] = v
    return out


def _station_breakdown_rows(societe: str, line: str, bus: str, day: str) -> list:
    """Per-station ticket breakdown for ONE EXACT trip (societe, line, bus, day) -- NOT the
    whole line-day. Constaté (2026-07-11) : avant l'ajout de `bus` au grain de
    `tickets_station_daily`, cet appel sommait les ventes de TOUS les bus de la ligne ce
    jour-là, ce qui ne pouvait pas se recouper avec la recette du bus-jour signalé affiché
    ailleurs -- voir la note dans `reference_db.populate_tickets_station_daily`. Retourne
    une ligne par arrêt d'origine desservi PAR CE BUS ce jour-là (lat/lon best-effort).
    """
    models = model_manager.get("ticket_anomaly")
    station_models = models.get("stations") or {}
    days = station_models.get("days")
    if days is None or len(days) == 0 or "bus" not in days.columns:
        return []
    mask = ((days["societe"] == societe) & (days["line"] == line)
           & (days["bus"].astype(str) == str(bus)) & (days["day"] == day))
    sub = days[mask].copy()
    if len(sub) == 0:
        return []
    ex = ticket_anomaly.explain_stations(station_models, sub)
    coords = _resolve_station_coords(societe, line, ex["station"].tolist())
    rows = []
    for _, r in ex.iterrows():
        lat, lon = coords.get(r["station"], (None, None))
        # Bonne vs mauvaise anomalie -- même logique que _ticket_rows_with_reasons, au
        # grain arrêt : recette de CET arrêt ce jour-là au-dessus de sa normale (repli
        # ligne si pas assez d'historique pour cet arrêt) ? La normale d'un arrêt reste
        # une propriété de L'ARRÊT (pas du bus qui le dessert ce jour-là précis).
        recette = float(r["recette"])
        recette_baseline = r.get("station_median_recette")
        if recette_baseline is None or pd.isna(recette_baseline):
            recette_baseline = r.get("line_median_recette")
        is_good_anomaly = (bool(recette > recette_baseline)
                           if recette_baseline is not None and pd.notna(recette_baseline)
                           else None)
        rows.append({
            "station": r["station"], "nbr_ticket": int(r["nbr_ticket"]),
            "recette": round(recette, 2),
            "avg_fare": round(float(r["avg_fare"]), 2),
            "anomaly": bool(r["anomaly"]), "severity": r.get("severity", "medium"),
            "reasons": list(r["reasons"]), "lat": lat, "lon": lon,
            "is_good_anomaly": is_good_anomaly,
        })
    rows.sort(key=lambda r: (not r["anomaly"], -r["recette"]))
    return rows


def _station_trip_breakdown(societe: str, line: str, bus: str, day: str) -> dict:
    """Répartition ALLER/RETOUR des ventes par arrêt pour CE trajet -- lit
    `tickets_station_trip_daily` (voir reference_db.populate_tickets_station_trip_daily),
    une table SÉPARÉE du modèle d'anomalie (décision utilisateur 2026-07-11 : direction
    reste un raffinement d'AFFICHAGE, pas un nouveau grain de détection). Retourne
    {"ALLER": [...], "RETOUR": [...]} quand la direction est connue pour ce bus-jour
    (voyage suivi par l'appareil), ou {} si tout est 'UNKNOWN' (appareil ne suit pas
    voyage) -- dans ce cas l'appelant doit garder la vue combinée existante plutôt que
    d'afficher un faux ALLER seul.
    """
    conn = rdb.init_db()
    try:
        rows = conn.execute(
            """SELECT t.direction, t.station_name, t.nbr_ticket, t.recette
               FROM tickets_station_trip_daily t
               JOIN companies c ON c.company_id = t.company_id
               JOIN lines l ON l.line_id = t.line_id
               WHERE c.canonical_name = ? AND l.line_code = ? AND t.bus = ? AND t.day = ?""",
            (societe, line, str(bus), day),
        ).fetchall()
    finally:
        conn.close()
    if not rows or all(r[0] == "UNKNOWN" for r in rows):
        return {}
    by_dir: dict = {"ALLER": [], "RETOUR": []}
    all_stations = [r[1] for r in rows if r[0] != "UNKNOWN"]
    coords = _resolve_station_coords(societe, line, all_stations)
    for direction, station, nbr_ticket, recette in rows:
        if direction not in by_dir:
            continue  # UNKNOWN résiduel mélangé à des lignes connues -- ignoré, pas deviné
        lat, lon = coords.get(station, (None, None))
        avg_fare = round(recette / nbr_ticket, 2) if nbr_ticket else 0.0
        by_dir[direction].append({
            "station": station, "nbr_ticket": int(nbr_ticket), "recette": round(recette, 2),
            "avg_fare": avg_fare, "lat": lat, "lon": lon,
        })
    for d in by_dir:
        by_dir[d].sort(key=lambda r: -r["recette"])
    return by_dir


@app.get("/api/ticket-anomaly-stations")
@_limit_concurrency
def ticket_anomaly_stations(societe: str, line: str, bus: str, day: str):  # def, pas async def -- voir ticket_anomaly_history
    """Per-station ticket breakdown for ONE EXACT trip (societe, line, bus, day) -- Phase 2
    drill-down from a flagged bus-day card (see docs/WEBSERVICES_NEEDED.md service 3 and
    the `recursive-bubbling-curry` plan). Returns one row per origin station with lat/lon
    (best-effort) + its own anomaly flag, for the map/table view.
    """
    try:
        rows = _station_breakdown_rows(societe, line, bus, day)
        by_direction = _station_trip_breakdown(societe, line, bus, day)
        return {"societe": societe, "line": line, "bus": bus, "day": day, "stations": rows,
               "by_direction": by_direction}
    except KeyError:
        raise HTTPException(status_code=503, detail="Ticket anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting station breakdown: {str(e)}")


@app.get("/api/ticket-anomaly-reference")
@_limit_concurrency
def ticket_anomaly_reference(societe: str, line: str):  # def, pas async def -- voir ticket_anomaly_history
    """A NORMAL (non-anomalous) bus-day for this line, with its per-station ticket
    breakdown -- companion to `/api/reference-trip` (GPS), so the admin can compare a
    flagged trip's per-station sales against what a normal trip's sales look like, not
    just against an abstract median number. Picked the same way as the GPS reference
    trip: closest to the line's median `recette` among non-anomalous bus-days.
    """
    try:
        models, days = _filter_ticket_days(societe, line)
        if len(days) == 0:
            raise HTTPException(status_code=404, detail="No ticket data for this line")
        ex = ticket_anomaly.explain(models, days)
        normal = ex[~ex["anomaly"]]
        if len(normal) == 0:
            normal = ex
        med = float(ex["recette"].median())
        best = normal.iloc[(normal["recette"] - med).abs().argsort().iloc[0]]
        bus, day = str(best["bus"]), str(best["day"])
        stations = _station_breakdown_rows(societe, line, bus, day)
        by_direction = _station_trip_breakdown(societe, line, bus, day)
        return {
            "societe": societe, "line": line,
            "trip": {
                "bus": bus, "day": day,
                "nbr_ticket": int(best["nbr_ticket"]),
                "recette": round(float(best["recette"]), 2),
                "avg_fare": round(float(best["avg_fare"]), 2),
                "line_median_recette": round(med, 2),
            },
            "stations": stations,
            "by_direction": by_direction,
        }
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=503, detail="Ticket anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting reference trip: {str(e)}")


@app.post("/api/ticket-anomaly/score-live")
async def ticket_anomaly_score_live(request: TicketAnomalyScoreRequest):
    """Score ticket-days that AREN'T in the precomputed history yet.

    Unlike /api/anomaly/score-live, this never touches a database itself -- pass in
    rows already fetched from wherever they came from (a company webservice, or your
    own glue code). See docs/WEBSERVICES_NEEDED.md for the exact source fields.
    """
    try:
        models = model_manager.get("ticket_anomaly")
        if not request.rows:
            return {"days": [], "anomaly_count": 0}
        day_rows = pd.DataFrame([r.model_dump() for r in request.rows])
        scored = ticket_anomaly.score(models, day_rows)
        rows = _ticket_rows_with_reasons(models, scored, anomalies_only=False)
        return {"days": rows, "anomaly_count": sum(1 for r in rows if r["anomaly"])}
    except KeyError:
        raise HTTPException(status_code=503, detail="Ticket anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring live ticket data: {str(e)}")


@app.post("/api/anomaly/score-live")
async def anomaly_score_live(request: AnomalyLiveScoreRequest):
    """Score a trip in REAL TIME from GPS pings -- for an in-progress or just-finished
    trip that isn't in the precomputed `trips_scored.parquet` yet (see /api/anomaly-history
    for previously-scored trips). Reuses the exact same reconstruction + scoring pipeline
    `populate_trips`/`anomaly.train()` already use, just on-demand for one bus-day instead of
    a full offline batch -- no separate model or logic path.

    Like /api/ticket-anomaly/score-live, this never touches a database itself -- pass in
    pings already fetched from wherever they came from (getPingsForDay, see
    docs/WEBSERVICES_NEEDED.md). Fixed 2026-07-13: this used to call MongoDB directly
    (get_db("Historique_pos")), which doesn't work on a deployment that was specifically
    set up to avoid a live Mongo dependency -- see reconstruct_bus_day's `raw_pings` param.
    """
    try:
        models = model_manager.get("anomaly")
        usable = _load_usable_lines()
        key = (request.line, request.societe)
        if key not in usable:
            raise HTTPException(status_code=404, detail=f"Line {request.line} geometry not found")
        stops = usable[key]

        raw_pings = pd.DataFrame([p.model_dump() for p in request.pings])
        if len(raw_pings):
            # format="mixed" : constaté sur données réelles -- certains pings ont des
            # microsecondes ("...T06:00:20.123456"), d'autres non ("...T06:00:20"), un
            # format unique ferait échouer le parsing dès le premier ping non conforme.
            raw_pings["t"] = pd.to_datetime(raw_pings["t"], format="mixed")
            raw_pings = raw_pings.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)

        cfg = fdn.Config()
        fa = fdn.reconstruct_bus_day(None, f"d{request.day}", request.line,
                                     request.societe, request.bus, stops, cfg,
                                     raw_pings=raw_pings)
        if fa.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No usable GPS trip found for bus {request.bus} on {request.day}"
            )

        scored = anomaly.score(models, fa)
        rows = _rows_with_reasons(models, scored, anomalies_only=False)
        return {
            "societe": request.societe, "line": request.line,
            "bus": request.bus, "day": request.day,
            "trips": rows,
            "anomaly_count": sum(1 for r in rows if r["severity"] != "low"),
        }
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=503, detail="Anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring live trip: {str(e)}")


@app.get("/api/anomaly-history")
# `def`, PAS `async def` -- ce handler ne fait QUE du travail pandas/sqlite synchrone (pas
# un seul `await` réel dessous). Sous `async def`, ce travail bloque l'event loop unique de
# l'instance Render (WEB_CONCURRENCY=1) pendant toute sa durée -- y compris /health, qui
# devient alors injoignable -- constaté 2026-07-19 : "HTTP health check failed (timed out
# after 5 seconds)" pendant un appel synchrone de ~20s, PAS un OOM cette fois. `def` simple
# fait tourner ce handler dans le threadpool de Starlette, l'event loop restant libre de
# répondre à /health et aux autres requêtes en parallèle. Appliqué ici aux 5 handlers les
# plus lourds (ceux exercés par le flux qui a crashé) ; le reste de l'API garde `async def`
# par cohérence avec le style existant -- à convertir de la même façon si le même symptôme
# réapparaît ailleurs.
@_limit_concurrency
def get_anomaly_history(
    societe: str,
    line: Optional[str] = None,
    bus: Optional[int] = None,
    limit: int = 30,
    include_data_bugs: bool = False,
    dir: Optional[str] = None
):
    """Historical anomalies for a company/line/bus, with plain-language reasons."""
    try:
        models, trips = _filter_trips(societe, line, bus, include_data_bugs=include_data_bugs, dir=dir)
        anomalies = _rows_with_reasons(models, trips, anomalies_only=True, limit=limit)
        return {
            "anomalies": anomalies,
            "total": len(anomalies),
            "total_trips": int(len(trips)),
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting anomalies: {str(e)}")


@app.get("/api/bus-anomaly-check")
async def check_bus_anomalies(
    societe: str,
    line: str,
    bus: int,
    day: Optional[str] = None,
    include_data_bugs: bool = False
):
    """Check a specific bus for anomalies, optionally on a specific day."""
    try:
        models, trips = _filter_trips(societe, line, bus, day, include_data_bugs=include_data_bugs)
        if len(trips) == 0:
            return {
                "bus": bus, "line": line, "societe": societe, "day": day,
                "has_anomalies": False, "anomalies": [],
                "total_trips": 0, "anomaly_count": 0,
                "message": "No data found for this bus",
            }
        anomalies = _rows_with_reasons(models, trips, anomalies_only=True)
        return {
            "bus": bus, "line": line, "societe": societe, "day": day,
            "has_anomalies": len(anomalies) > 0,
            "anomalies": anomalies,
            "total_trips": int(len(trips)),
            "anomaly_count": len(anomalies),
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking bus anomalies: {str(e)}")


@app.get("/api/current-anomalies")
@_limit_concurrency
def get_current_anomalies(  # def, pas async def -- voir la note dans get_anomaly_history
    societe: str,
    line: Optional[str] = None,
    dir: Optional[str] = None
):
    """Anomalies for "today". Tries the most recent day the live GPS web service has
    actually finished processing (see _score_all_gps_live) -- the precomputed dataset is
    always historical (confirmed 2026-07-13: stuck at 21 juin while the live service
    already has 12 juillet), so without this, "today" silently meant "whenever the last
    training snapshot was taken", not today.

    That "most recent ready day" is YESTERDAY, not the literal calendar day -- confirmed
    2026-07-13 : le traitement de nuit tourne la NUIT SUIVANTE pour la journée qui vient de
    se terminer, donc `isDayReady` pour le jour calendaire courant reste `false` toute la
    journée (les données du jour même sont encore en train d'arriver) ; c'est seulement le
    jour PRÉCÉDENT qui est prêt. Falls back to the historical latest-day view when the web
    service isn't configured, unreachable, or has nothing ready yet (`live` in the response
    says which one happened, so the dashboard/PHP page can show it honestly).
    """
    try:
        models = model_manager.get("anomaly")
        live_day = _live_gps_day(societe)
        today, live = (live_day, True) if live_day else (latest_day_for(societe, line), False)
        _, trips = _filter_trips(societe, line, day=today, dir=dir)
        anomalies = _rows_with_reasons(models, trips, anomalies_only=True)
        return {
            "date": today,
            "live": live,
            "societe": societe,
            "line": line,
            "anomalies": anomalies,
            "total_trips": int(len(trips)),
            "anomaly_count": len(anomalies),
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current anomalies: {str(e)}")


def _trip_sequence(df, societe, line, bus, day, trip_start):
    """Per-stop sequence rows for one trip, with lat/lon for map rendering.

    Matched by `trip_start` (exact timestamp), NOT `trip_id` -- `trip_id` in
    `models["trips"]` (trips_scored.parquet, from anomaly.train()) is a per-bus-day LOCAL
    index (0, 1, 2... reused across every bus-day, confirmed 2026-07-13: only 28 distinct
    values total across 47k trips) left over from the pre-SQL-reference-DB pipeline, while
    `trips`/`trip_stops` here use the DB's own globally-unique trip_id -- the two numbering
    schemes only coincidentally overlap for small values, so matching on trip_id could
    silently return a DIFFERENT trip's sequence, not just an empty one. `trip_start` is a
    real timestamp present in both and reliably identifies one specific trip on a bus-day.

    `coord_suspect` marks stops that are unmatched on (almost) every trip of the
    line — the stop's geocoded coordinates are wrong, so an unmatched flag there
    says nothing about THIS trip.
    """
    ts = pd.Timestamp(trip_start) if trip_start else None
    match = (df["trip_start"] - ts).abs() < pd.Timedelta(seconds=60) if ts is not None else False
    seqdf = df[
        (df["societe"] == societe) & (df["line"] == line) & (df["bus"].astype(str) == str(bus)) &
        (df["day"] == day) & match
    ].sort_values("seq")
    coord_map = model_manager.get_stop_coords(societe, line)
    coord_suspect = model_manager.get_coord_suspect(societe, line)
    seq = []
    for _, s in seqdf.iterrows():
        dwell = float(s["dwell_s"]) if pd.notna(s.get("dwell_s")) else 0.0
        dark  = float(s["dark_s"])  if pd.notna(s.get("dark_s"))  else 0.0
        dist  = float(s["dist_m"])  if pd.notna(s.get("dist_m"))  else 0.0
        stop_name = s["stop"]
        lat, lon = coord_map.get(stop_name, (None, None))
        arrival = pd.Timestamp(s["arrival"]) if pd.notna(s.get("arrival")) else None
        departure = pd.Timestamp(s["departure"]) if pd.notna(s.get("departure")) else None
        seq.append({
            "seq": int(s["seq"]), "stop": stop_name,
            "dwell_min": round(dwell / 60, 1),
            "dark_min":  round(dark  / 60, 1),
            "had_gap":   bool(s.get("had_gap", False)),
            "dist_m": round(dist, 0),
            "matched": bool(s["matched"]),
            "coord_suspect": coord_suspect.get(stop_name, False),
            "lat": lat,
            "lon": lon,
            # horodatages réels du passage (None si arrêt non suivi) -- affichés sur la
            # carte pour que l'admin voie QUAND le bus a réellement atteint chaque arrêt
            "arrival": arrival.isoformat() if arrival is not None else None,
            "departure": departure.isoformat() if departure is not None else None,
        })
    return seq


def _problem_stops(seq):
    """Pinpoint WHERE in the trip things went wrong (stop names)."""
    if not seq:
        return {}
    out = {}
    worst_dwell = max(seq, key=lambda s: s["dwell_min"])
    if worst_dwell["dwell_min"] >= 5:
        out["longest_stop"] = {"stop": worst_dwell["stop"],
                               "dwell_min": worst_dwell["dwell_min"],
                               "arrival": worst_dwell.get("arrival"),
                               "lat": worst_dwell.get("lat"), "lon": worst_dwell.get("lon")}
    # signal-loss stops: stops where a GPS gap interrupted the dwell scan
    gap_stops = [s for s in seq if s.get("had_gap") and s.get("dark_min", 0) >= 5]
    if gap_stops:
        worst_gap = max(gap_stops, key=lambda s: s["dark_min"])
        out["signal_loss_stop"] = {"stop": worst_gap["stop"],
                                   "dark_min": worst_gap["dark_min"]}
        out["signal_loss_count"] = len(gap_stops)
    # Genuinely unserved stops only — chronically-unmatched stops (bad coordinates)
    # would otherwise appear as "non desservi" on every single trip of the line.
    offroute = [s["stop"] for s in seq if not s["matched"] and not s.get("coord_suspect")]
    if offroute:
        out["off_route_stops"] = offroute[:5]
        out["off_route_count"] = len(offroute)
    suspect = [s["stop"] for s in seq if s.get("coord_suspect")]
    if suspect:
        out["suspect_coord_stops"] = suspect[:5]
        out["suspect_coord_count"] = len(suspect)
    # Only flag matched stops that were still far — unmatched stops trivially have large dist_m
    # (they were simply never reached), and that info is already in off_route_stops.
    far = [s for s in seq if s["matched"] and s["dist_m"] >= 800]
    if far:
        worst_far = max(far, key=lambda s: s["dist_m"])
        out["farthest_stop"] = {"stop": worst_far["stop"], "dist_m": worst_far["dist_m"]}
    return out


def _detect_start_detour(g, trip_start, longest_stop, cfg, detour_thresh_m: float = 1000.0):
    """Bus leaves right after `trip_start`, drives well away, then comes back to ~the same
    spot before settling into `longest_stop`'s long dwell -- an unofficial side-trip
    (errand/positioning) bookending the genuine idle time, not GPS noise near the stop.

    Only fires when the bus ENDS UP back close to where it started (`idle_trim_m`) -- if it
    ends up somewhere else, that's just normal driving toward the next stop, not a detour.
    """
    if not longest_stop or not longest_stop.get("arrival"):
        return None
    t_start = pd.Timestamp(trip_start)
    t_end = pd.Timestamp(longest_stop["arrival"])
    if t_end <= t_start:
        return None
    seg = g[(g["t"] >= t_start) & (g["t"] <= t_end)].reset_index(drop=True)
    if len(seg) < 3:
        return None
    s0 = float(seg["s"].iloc[0])
    s_end = float(seg["s"].iloc[-1])
    delta = seg["s"] - s0
    excursion_m = float(delta.abs().max())
    if excursion_m < detour_thresh_m or abs(s_end - s0) > cfg.idle_trim_m:
        return None
    idx_far = int(delta.abs().idxmax())

    def _points(a: int, b: int):
        return [{"lat": round(float(r["lat"]), 6), "lon": round(float(r["lon"]), 6),
                  "t": r["t"].isoformat()} for _, r in seg.iloc[a:b + 1].iterrows()]

    # Scindé en aller (départ -> point le plus éloigné) et retour (point le plus éloigné ->
    # retour) pour que l'admin puisse suivre les deux trajets séparément sur la carte, plutôt
    # qu'un seul tracé qui superpose les deux passages.
    leg_out = _points(0, idx_far)
    leg_back = _points(idx_far, len(seg) - 1)
    return {
        "distance_km": round(excursion_m / 1000, 1),
        "left_at": seg["t"].iloc[0].isoformat(),
        "farthest_at": seg["t"].iloc[idx_far].isoformat(),
        "returned_at": seg["t"].iloc[-1].isoformat(),
        "duration_min": round((seg["t"].iloc[-1] - seg["t"].iloc[0]).total_seconds() / 60, 1),
        "leg_out_duration_min": round((seg["t"].iloc[idx_far] - seg["t"].iloc[0]).total_seconds() / 60, 1),
        "leg_back_duration_min": round((seg["t"].iloc[-1] - seg["t"].iloc[idx_far]).total_seconds() / 60, 1),
        "leg_out": leg_out,
        "leg_back": leg_back,
        "track": leg_out + leg_back[1:],   # conservé pour compat -- chemin complet dans l'ordre chronologique
    }


@app.get("/api/anomaly-explain")
@_limit_concurrency
def anomaly_explain(  # def, pas async def -- voir la note dans get_anomaly_history
    societe: str,
    line: Optional[str] = None,
    bus: Optional[int] = None,
    day: Optional[str] = None,
    include_data_bugs: bool = False,
    dir: Optional[str] = None,
    check_detours: bool = False,
):
    """Per-trip explanations + WHERE (which stops) the anomaly happened.

    `line=None` means "all lines for this operator" (decision 2026-07-15) -- `df`/`trips`
    then span every line, so every per-row lookup below (`_trip_sequence`, the detour
    track fetch) MUST key off that ROW's own `line`, never the outer `line` parameter,
    which may be None. Keying the detour track cache by (line, bus, day) rather than just
    (bus, day) for the same reason: a bus number isn't guaranteed unique across lines.

    `line` is now ALWAYS required in practice (see the guard right below) -- decision
    2026-07-19, stricter than the original "line OR day" rule from two days earlier: even
    "all lines, one specific day" was more than the user wanted to allow through this
    endpoint at all ("prohibit it, only let the user choose one line at a time"). The
    `line=None` code path above is kept working rather than deleted, in case this guard is
    ever relaxed again (e.g. after a move off the 512MB free tier) -- `line: Optional[str]`
    stays in the signature for that reason, not because it's currently reachable.

    `check_detours` is opt-in (default False) -- it needs a LIVE raw-GPS-ping read per
    distinct (line, bus, day) in the list, plus a Kalman filter over each, to reconstruct
    the actual path driven (see `_detect_start_detour`/`_build_gps_track`). Not called by
    the embed widget's bulk analysis anymore (decision 2026-07-19): checking every flagged
    trip in one request was the direct cause of repeated OOM/timeout crashes on the 512MB
    Render instance, a detour is never the ONLY signal a trip is anomalous (something else
    already flags it), and the same detection still runs, cheaply, on a SINGLE trip when
    the client asks for its map (`/api/trip-detail`, no bulk cost). Kept here as an opt-in
    parameter for callers that genuinely want the bulk version (e.g. a scheduled report
    with no request-latency constraint).
    """
    # Garde-fou : ligne TOUJOURS requise (décision utilisateur 2026-07-19, resserré depuis
    # la version précédente qui acceptait aussi "toutes les lignes + un jour précis") --
    # "toutes les lignes" à la fois scanne tout l'historique de la société en une seule
    # requête synchrone, confirmé 2026-07-19 : HTTP 502 (timeout upstream Render) même sans
    # le contrôle de détour (déjà retiré par ailleurs). Refusé explicitement plutôt que
    # laissé échouer après une longue attente.
    if line is None:
        raise HTTPException(status_code=422, detail=(
            "Choisissez une ligne pour analyser -- l'historique de toutes les lignes à la "
            "fois n'est pas proposé, quel que soit le jour."
        ))

    # Tranche fondation chargée au PLUS ÉTROIT possible -- l'ancienne version chargeait
    # toujours TOUT l'historique du périmètre (ligne entière, ou société ENTIÈRE pour
    # "toutes les lignes"), même quand la requête ne portait que sur UN jour : ~200k lignes
    # d'arrêts + 6 colonnes de datetime à parser, ce qui tuait l'instance Render 512MB par
    # OOM (constaté 2026-07-17 : 502 sur anomaly-explain?day=20260716). Ici :
    #   - jour précisé -> tranche de CE jour seulement (minuscule) ;
    #   - ligne précisée sans jour -> tranche de la ligne (modéré) ;
    #   - toutes lignes + tous jours -> rien d'office, tranches chargées PAR JOUR
    #     d'anomalie à la demande (chaque jour = quelques centaines de lignes).
    if day is not None:
        df = model_manager.foundation_slice(societe, line=line, day=day)
    elif line is not None:
        df = model_manager.foundation_slice(societe, line=line)
    else:
        df = None

    day_slices: dict = {}

    def _seq_df_for(day_):
        if df is not None:
            return df
        if day_ not in day_slices:
            day_slices[day_] = model_manager.foundation_slice(societe, day=day_)
        return day_slices[day_]

    try:
        models, trips = _filter_trips(societe, line, bus, day, include_data_bugs=include_data_bugs, dir=dir)
        rows = _rows_with_reasons(models, trips, anomalies_only=False)
        anomalous = [r for r in rows if r["severity"] != "low"]

        # Attach stop-level detail to each anomalous trip
        # Use each row's OWN line/bus (needed when line/bus=None means "all lines"/"all buses")
        for a in anomalous:
            sdf = _seq_df_for(a["day"])
            if sdf is not None:
                seq = _trip_sequence(sdf, societe, a["line"], int(a["bus"]), a["day"], a["trip_start"])
                a["problem_stops"] = _problem_stops(seq)
            else:
                a["problem_stops"] = {}

        if anomalous:
            # Un (ligne, bus, jour) distinct = un aller-retour Mongo (~1s) + un filtre Kalman
            # complet sur les pings bruts de toute la journée. Les fetches sont indépendants
            # (chacun sa propre requête + son propre DataFrame local, pas d'état mutable
            # partagé hors _load_usable_lines() qui n'est que lu), donc parallélisables --
            # MAIS la concurrence et la RÉTENTION mémoire doivent rester bornées : la
            # version précédente (16 workers + cache de TOUTES les traces jusqu'à la fin de
            # la requête) tuait le worker Render 512MB par OOM (constaté 2026-07-17 : le log
            # montre le process redémarrer en boucle, le proxy PHP recevait un 502).
            # Ici : chaque trace est consommée puis relâchée DANS le worker (pic mémoire =
            # DETOUR_WORKERS traces au plus), et DETOUR_MAX_TRACKS borne le nombre total de
            # bus-jours vérifiés (les plus récents d'abord ; au-delà, has_detour reste
            # simplement absent -- affiché comme "non vérifié", jamais comme "pas de détour").
            by_key: dict = {}
            for a in anomalous:
                if (a["problem_stops"].get("longest_stop") or {}).get("arrival"):
                    by_key.setdefault((a["line"], a["bus"], a["day"]), []).append(a)

            keys = sorted(by_key, key=lambda k: str(k[2]), reverse=True)
            max_tracks = int(os.getenv("DETOUR_MAX_TRACKS", "0") or 0)
            if max_tracks > 0:
                keys = keys[:max_tracks]
            workers = max(1, int(os.getenv("DETOUR_WORKERS", "8") or 8))

            def _check(key, ws_groups):
                line_, bus_, day_ = key
                try:
                    built = _build_gps_track(societe, line_, bus_, day_, ws_groups=ws_groups)
                except Exception:
                    built = None
                if built is None:
                    return
                g, _stops, _route_len = built
                for a in by_key[key]:
                    try:
                        detour = _detect_start_detour(g, a["trip_start"],
                                                      a["problem_stops"]["longest_stop"], fdn.Config())
                    except Exception:
                        detour = None
                    a["has_detour"] = bool(detour)
                    if detour:
                        a["detour"] = detour

            if keys:
                # Traitement PAR JOUR : quand les pings viennent du webservice (Mongo
                # absent en production, ou jour plus récent que le dataset), UN appel
                # getPingsForDay couvre tous les bus de la journée -- le pré-chargement
                # par jour évite de refaire ce même appel pour chaque bus-jour vérifié,
                # et une seule journée de pings réside en mémoire à la fois.
                latest_db_day = model_manager.get_latest_day() or ""
                days_order = sorted({k[2] for k in keys}, reverse=True)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for day_ in days_order:
                        day_keys = [k for k in keys if k[2] == day_]
                        ws_groups = None
                        if time.time() < _mongo_gps_down_until or str(day_) > str(latest_db_day):
                            ws_groups = _ws_day_groups(societe, day_)
                        list(pool.map(lambda k: _check(k, ws_groups), day_keys))

        worst_trip, sequence = None, []
        if anomalous:
            worst = max(anomalous, key=lambda r: r["anomaly_strength"])
            worst_trip = worst
            wdf = _seq_df_for(worst["day"])
            if wdf is not None:
                sequence = _trip_sequence(wdf, societe, worst["line"], int(worst["bus"]),
                                          worst["day"], worst["trip_start"])

        # Median trip duration for normal trips on this line (baseline for comparison) --
        # only meaningful PER LINE (different routes have different normal durations), so
        # left None for "all lines" rather than averaging across unrelated routes.
        avg_duration_min = None
        if line:
            try:
                all_scored = models["trips"]
                line_normal = all_scored[
                    (all_scored["societe"] == societe) &
                    (all_scored["line"] == line) &
                    (~all_scored["anomaly"])
                ]
                if len(line_normal) >= 3:
                    avg_duration_min = round(float(line_normal["total_elapsed"].median()), 1)
            except Exception:
                pass

        return {
            "societe": societe, "line": line, "bus": bus, "day": day,
            "trips": rows,
            "anomalies": anomalous,
            "worst_trip": worst_trip,
            "sequence": sequence,
            "total_trips": len(rows),
            "anomaly_count": len(anomalous),
            "avg_duration_min": avg_duration_min,
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining anomalies: {str(e)}")


@app.get("/api/trip-detail")
async def trip_detail(
    societe: str, line: str, bus: int, day: str, trip_start: str
):
    """Sequence + problem stops for one specific trip (used for per-trip map/chart).

    Identified by `trip_start` (exact timestamp), not `trip_id` -- see the note in
    `_trip_sequence` for why `trip_id` from an anomaly row can't be trusted to pick the
    right trip out of the reference DB anymore.

    Also looks for an "unofficial detour" -- the bus leaving right after the trip
    officially starts, driving off, and coming back to ~the same spot before its long
    dwell (e.g. an errand before really departing) -- and returns the raw GPS path for
    it so the admin can see the actual route driven, not just the straight stop-to-stop
    line. Best-effort: needs a live MongoDB read, so any failure here just omits the
    detour rather than failing the whole request.
    """
    df = model_manager.foundation_slice(societe, line=line, bus=bus, day=day)
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    seq = _trip_sequence(df, societe, line, bus, day, trip_start)
    problem_stops = _problem_stops(seq)
    longest_stop = problem_stops.get("longest_stop")
    if longest_stop and longest_stop.get("arrival"):
        try:
            ts = pd.Timestamp(trip_start)
            trip_row = df[
                (df["societe"] == societe) & (df["line"] == line) &
                (df["bus"].astype(str) == str(bus)) & (df["day"] == day) &
                ((df["trip_start"] - ts).abs() < pd.Timedelta(seconds=60))
            ].iloc[0]
            built = _build_gps_track(societe, line, bus, day)
            if built is not None:
                g, _stops, _route_len = built
                detour = _detect_start_detour(g, trip_row["trip_start"], longest_stop, fdn.Config())
                if detour:
                    problem_stops["unofficial_detour"] = detour
        except Exception:
            pass
    return {
        "sequence": seq,
        "problem_stops": problem_stops,
    }


_driver_trip_cache: Optional[pd.DataFrame] = None


def _driver_scored_trips() -> pd.DataFrame:
    """Precomputed anomaly-scored trips (trips_scored.parquet) merged with `driver_code`
    from the reference DB (see reference_db.attach_driver_codes_to_trips). Cached once --
    neither side changes during the app's lifetime (a restart is needed after re-running
    the driver backfill, same as any other model/reference reload).

    Historical only, and only as complete as the ticket backfill window: `driver_code`
    comes from Historique_Tickets 2025+ (see reference_db.populate_driver_services) --
    trips before 2025, or without matching ticket data, simply don't appear here rather
    than being guessed. Live (today's webservice-scored) trips are never driver-attributed
    either, since ticket data lags behind GPS by the same night-processing delay covered
    elsewhere in this file -- a driver view is inherently a look-back tool, not a live one.

    Confirmed data bugs / fragments / partial-coverage trips are EXCLUDED here, same as
    `_filter_trips`'s default (`include_data_bugs=False`) -- these are impossible/outlier
    readings (e.g. corrupted timestamps, <3 stops), never real driver behavior, and were
    never meant to reach a driver's stats (bug found 2026-07-15: they were leaking into
    driver-stats/drivers-ranked before this filter was added here).
    """
    global _driver_trip_cache
    if _driver_trip_cache is not None:
        return _driver_trip_cache
    models = model_manager.get("anomaly")
    scored = models["trips"].copy()
    scored["bus"] = scored["bus"].astype(str)
    scored["trip_start"] = pd.to_datetime(scored["trip_start"])
    scored = _trip_quality_flags(scored, models)
    scored = scored[~(scored["is_data_bug"] | scored["is_fragment"] | scored["is_partial_coverage"])]

    conn = rdb.init_db()
    try:
        rows = conn.execute(
            """SELECT c.canonical_name AS societe, l.line_code AS line, t.bus, t.day,
                      t.trip_start, t.driver_code
               FROM trips t JOIN companies c ON c.company_id = t.company_id
                            JOIN lines l ON l.line_id = t.line_id
               WHERE t.driver_code IS NOT NULL""").fetchall()
    finally:
        conn.close()
    drivers = pd.DataFrame(rows, columns=["societe", "line", "bus", "day", "trip_start", "driver_code"])
    drivers["trip_start"] = pd.to_datetime(drivers["trip_start"], format="mixed")

    merged = scored.merge(drivers, on=["societe", "line", "bus", "day", "trip_start"], how="inner")
    _driver_trip_cache = merged
    return merged


_schedule_cache: Optional[dict] = None


def _scheduled_departures() -> dict:
    """(societe, line) -> {"ALLER": "HH:MM"|None, "RETOUR": "HH:MM"|None, "multi_variant": bool},
    from `winicari.ligne.horaires` (published timetable). Cached once (static reference
    data, doesn't change during the app's lifetime).

    Coverage: 31/68 of our tracked lines (45.6%, confirmed 2026-07-15) publish a schedule at
    all -- the rest just have no entry here, nothing guessed for them.

    Some lines carry MULTIPLE schedule variants in `horaires` (e.g. line 212 has 3 separate
    groups; line 213 has 2 variants in one group) with no field documenting which applies
    when (weekday/weekend? seasonal? genuinely unclear from the data, not worth guessing
    without asking the platform team). Per user decision 2026-07-15: only the FIRST variant
    encountered is used as a single best-effort reference time, and `multi_variant=True`
    flags this so the UI can caveat it as approximate rather than presenting it as certain.

    `Aller`/`Retour` arrays are index-aligned to `stationnames` (same physical route order,
    confirmed by inspecting real documents, not guessed): `Aller[0]` = scheduled departure
    from the line's origin. The RETOUR run's own origin is the LAST station in that same
    list (the line's destination), so its departure time is `Retour[-1]`, NOT `Retour[0]`
    (`Retour[0]` is the return trip's scheduled ARRIVAL back at the outbound origin).
    """
    global _schedule_cache
    if _schedule_cache is not None:
        return _schedule_cache

    # Artefact JSON embarqué d'abord -- les horaires publiés sont des données de référence
    # STATIQUES (comme les modèles entraînés), et MongoDB n'est pas joignable depuis la
    # production (Render n'a que les webservices, décision utilisateur 2026-07-17). Le
    # fichier est (ré)généré ci-dessous à chaque fois que Mongo EST joignable (dev local),
    # committé, et copié dans l'image (voir Dockerfile.render) -- même workflow
    # "réentraîner localement -> redéployer" que le reste des artefacts.
    sched_path = Path("data/reference/schedules.json")
    if sched_path.exists():
        try:
            with open(sched_path, encoding="utf-8") as f:
                rows = json.load(f)
            _schedule_cache = {
                (r["societe"], r["line"]): {
                    "ALLER": r.get("ALLER"), "RETOUR": r.get("RETOUR"),
                    "multi_variant": bool(r.get("multi_variant")),
                } for r in rows
            }
            return _schedule_cache
        except Exception as e:
            print(f"  schedules.json illisible ({e}) -- tentative Mongo")

    cache: dict = {}
    alias_to_canon = {alias: canon for canon, aliases in rdb.CANONICAL_COMPANIES.items() for alias in aliases}
    try:
        wi_db = get_db("winicari")
        for doc in wi_db["ligne"].find({"horaires": {"$exists": True, "$ne": []}},
                                        {"code": 1, "societe": 1, "horaires": 1}):
            canon = alias_to_canon.get(doc.get("societe"))
            if not canon:
                continue
            line_code = str(doc.get("code", "")).strip()
            if not line_code:
                continue
            # Aplatit la structure imbriquée (groupes -> sous-listes -> dicts) pour retenir
            # TOUS les dicts de variante quelle que soit leur position exacte -- la forme
            # varie d'une ligne à l'autre (1 groupe/1 variante, plusieurs groupes, ou
            # plusieurs variantes dans un même groupe, constaté en échantillonnant 15 lignes).
            variants = []
            for grp in (doc.get("horaires") or []):
                if not isinstance(grp, list):
                    continue
                for sub in grp:
                    if isinstance(sub, list):
                        variants.extend(v for v in sub if isinstance(v, dict))
            if not variants:
                continue
            first = variants[0]
            aller_list = first.get("Aller") or []
            retour_list = first.get("Retour") or []
            aller_time = next((v for v in aller_list if v), None)
            retour_time = next((v for v in reversed(retour_list) if v), None)
            if not aller_time and not retour_time:
                continue
            cache[(canon, line_code)] = {
                "ALLER": aller_time, "RETOUR": retour_time,
                "multi_variant": len(variants) > 1,
            }
        # Mongo joignable -> (ré)écrire l'artefact pour le prochain build d'image
        try:
            sched_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sched_path, "w", encoding="utf-8") as f:
                json.dump([
                    {"societe": soc, "line": ln, **vals}
                    for (soc, ln), vals in sorted(cache.items())
                ], f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"  schedules.json non écrit: {e}")
    except Exception as e:
        print(f"  _scheduled_departures unavailable: {e}")
    _schedule_cache = cache
    return cache


_driver_lookup_cache: Optional[dict] = None


def _driver_code_lookup() -> dict:
    """(societe, line, bus, day, trip_start) -> driver_code, for attaching a driver chip to
    any anomaly row without a per-row DB query. Built once from `_driver_scored_trips`."""
    global _driver_lookup_cache
    if _driver_lookup_cache is None:
        df = _driver_scored_trips()
        _driver_lookup_cache = {
            (row.societe, row.line, row.bus, row.day, row.trip_start): row.driver_code
            for row in df.itertuples()
        }
    return _driver_lookup_cache


@app.get("/api/drivers-ranked")
async def drivers_ranked(societe: Optional[str] = None, min_trips: int = 5, limit: int = 50):
    """Drivers ranked by anomaly rate -- lets the dashboard offer a « which driver should I
    look at? » list instead of requiring the admin to already know a driver code.
    `min_trips` filters out drivers with too little history to make their rate meaningful
    (same spirit as the model-tiering thresholds elsewhere in this file).
    """
    df = _driver_scored_trips()
    if societe:
        df = df[df["societe"] == societe]
    if len(df) == 0:
        return {"drivers": []}
    g = (df.groupby(["driver_code", "societe"])
           .agg(n_trips=("anomaly", "size"), n_anomalies=("anomaly", "sum"))
           .reset_index())
    g = g[g["n_trips"] >= min_trips]
    if len(g) == 0:
        return {"drivers": []}
    g["anomaly_rate"] = (100 * g["n_anomalies"] / g["n_trips"]).round(1)
    g = g.sort_values("anomaly_rate", ascending=False).head(limit)
    return {"drivers": g.to_dict("records")}


@app.get("/api/driver-stats")
async def driver_stats(driver_code: str, societe: Optional[str] = None):
    """Trip/anomaly stats for one driver across every line they've driven, plus their
    flagged trips. See `_driver_scored_trips` for coverage caveats (2025+ only).

    `societe` disambiguates -- driver codes (`CodeCh`) are assigned independently by each
    company, NOT globally unique: confirmed 2026-07-15, code "5531" exists at S.T.S,
    S.R.T.K, AND TCV simultaneously with hundreds of trips each, far too many to be one
    person driving for three unrelated companies. Without `societe`, results would
    silently merge different people's stats into one. Kept optional (not required) only
    for backward-compatible lookups where the caller already knows the code is
    company-scoped in practice; the dashboard always passes it.
    """
    df = _driver_scored_trips()
    mask = df["driver_code"] == driver_code
    if societe:
        mask &= df["societe"] == societe
    sub = df[mask]
    if len(sub) == 0:
        raise HTTPException(status_code=404, detail="No scored trips found for this driver code "
                                                     "(unknown code, wrong operator, or outside the 2025+ ticket backfill window)")
    if not societe and sub["societe"].nunique() > 1:
        raise HTTPException(status_code=409, detail=(
            f"Driver code '{driver_code}' exists at multiple operators "
            f"({', '.join(sorted(sub['societe'].unique()))}) -- pass `societe` to disambiguate, "
            "these are independently-assigned codes, not one person."))

    by_line = (sub.groupby(["societe", "line"])
                  .agg(n_trips=("anomaly", "size"), n_anomalies=("anomaly", "sum"))
                  .reset_index())
    by_line["anomaly_rate"] = (100 * by_line["n_anomalies"] / by_line["n_trips"]).round(1)
    by_line = by_line.sort_values("n_anomalies", ascending=False)

    models = model_manager.get("anomaly")
    anomalies = sub[sub["anomaly"]]
    # Non tronqué (pas de `limit`) pour que la distribution des causes porte sur TOUTES les
    # anomalies de ce chauffeur, pas seulement les 50 renvoyées pour l'affichage détaillé --
    # sinon "cause principale" refléterait un artefact de troncature, pas la vraie tendance.
    all_rows = _rows_with_reasons(models, anomalies, anomalies_only=True) if len(anomalies) else []
    cause_counts = Counter(r.get("top_feature") for r in all_rows)
    cause_distribution = [
        {"top_feature": f, "count": c, "pct": round(100 * c / len(all_rows), 1)}
        for f, c in cause_counts.most_common()
    ]
    dominant_cause = cause_distribution[0] if cause_distribution else None
    rows = all_rows[:50]

    n = len(sub)
    n_anom = int(sub["anomaly"].sum())
    return {
        "driver_code": driver_code,
        "societe": societe or (sub["societe"].iloc[0] if len(sub) else None),
        "total_trips": n,
        "total_anomalies": n_anom,
        "anomaly_rate": round(100 * n_anom / n, 1),
        "by_line": by_line.to_dict("records"),
        "dominant_cause": dominant_cause,
        "cause_distribution": cause_distribution,
        "anomalies": rows,
    }


@app.get("/api/reference-trip")
@_limit_concurrency
def reference_trip(societe: str, line: str):  # def, pas async def -- voir get_anomaly_history
    """LE trajet « témoin » d'une ligne, PAR DIRECTION : un trajet réel, non anormal, bien
    suivi, de durée proche de la médiane -- affiché au-dessus des anomalies comme référence
    (« voici à quoi ressemble un trajet normal sur cette ligne »). Renforce la confiance : le
    modèle sait ce qu'est un trajet normal, les anomalies sont des écarts à CETTE norme.

    ALLER et RETOUR PAIRÉS quand possible : même jour, même bus, le RETOUR étant le
    premier à suivre CET aller -- pour que le stationnement terminus affiché illustre la
    VRAIE pause d'UN cycle complet (arrivée -> pause -> redépart), pas deux jours sans
    rapport. Constaté (2026-07-11) : sans ce pairage, l'ALLER de référence pouvait montrer
    ~9 min de pause typique et le RETOUR ~28 min -- des trajets de jours différents, donc
    la différence ne renseigne en rien sur la pause réelle de CE bus entre son arrivée et
    son redépart, ce que le contrôleur cherche justement à savoir. Repli sur le meilleur
    trajet par direction indépendamment (ancien comportement) si aucune paire same-day/
    same-bus n'est disponible. Inclut aussi le stationnement terminus TYPIQUE de cette
    direction (médiane sur les trajets normaux, PAS la valeur du trajet choisi) pour
    donner à l'admin un repère concret : au-delà d'environ 2x cette valeur, un
    stationnement observé est probablement un service non clôturé plutôt qu'une pause
    normale.
    """
    # include_live=False : un trajet de référence est par définition HISTORIQUE (un
    # trajet PASSÉ jugé normal), jamais celui d'aujourd'hui -- voir la note dans
    # foundation_slice pour l'incident que ça évite (double coût : la tranche historique
    # de toute la ligne + une reconstruction live de toute la société, en une requête).
    df = model_manager.foundation_slice(societe, line=line, include_live=False)
    try:
        models, trips = _filter_trips(societe, line, include_live=False)   # déjà nettoyé des bugs/fragments
    except KeyError:
        raise HTTPException(status_code=503, detail="Anomaly models not loaded")
    if df is None or len(trips) == 0:
        raise HTTPException(status_code=404, detail="No trips for this line")

    def _normal(sub: pd.DataFrame) -> pd.DataFrame:
        normal = sub[(~sub["anomaly"]) & (sub["match_rate"] >= 0.7)]
        return normal if len(normal) else sub[~sub["anomaly"]]

    normal_by_dir = {d: _normal(trips[trips["dir"] == d]) for d in ("ALLER", "RETOUR")}
    med_by_dir = {d: float(trips[trips["dir"] == d]["total_elapsed"].median())
                 for d in ("ALLER", "RETOUR")}
    _usable = _load_usable_lines()
    _line_geometry_stops = len(_usable.get((line, societe), []))
    idle_by_dir: dict = {}
    for d in ("ALLER", "RETOUR"):
        idle_col = (normal_by_dir[d]["terminus_idle_min"]
                   if "terminus_idle_min" in normal_by_dir[d].columns else pd.Series(dtype=float))
        idle_by_dir[d] = float(idle_col.median()) if len(idle_col) else None

    def _build(row: pd.Series, direction: str) -> dict:
        typical_idle = idle_by_dir[direction]
        seq = _trip_sequence(df, societe, line, row["bus"], row["day"], row["trip_start"])
        return {
            "trip": {
                "bus": str(row["bus"]), "day": row["day"], "dir": direction,
                "trip_id": int(row["trip_id"]),
                "duration_min": round(float(row["total_elapsed"]), 1),
                "line_median_min": round(med_by_dir[direction], 1),
                "n_stops": int(row["n_stops"]),
                "match_rate": round(float(row["match_rate"]), 3),
                "mean_dwell_min": round(float(row["mean_dwell_s"]) / 60, 1),
                "trip_start": row["trip_start"].isoformat() if pd.notna(row["trip_start"]) else None,
                "trip_end": row["trip_end"].isoformat() if pd.notna(row["trip_end"]) else None,
                "typical_terminus_idle_min": (round(typical_idle, 1) if typical_idle is not None else None),
                "service_not_closed_threshold_min": (round(2 * typical_idle, 1)
                                                     if typical_idle else None),
                # Couverture : un trajet de référence peut être PARTIEL quand la direction
                # n'a (quasi) aucun trajet complet -- constaté 2026-07-18 sur S.R.T.K/202 :
                # 409/498 ALLER complets contre 2/443 RETOUR (le traceur s'arrête
                # systématiquement en route au retour). Sans ce champ, l'écart de nombre
                # d'arrêts/durée entre les deux directions ressemble à un bug d'affichage
                # alors que c'est un fait des données -- l'UI l'explique maintenant.
                "is_full": (bool(row["full"]) if pd.notna(row.get("full")) else None),
                "covered_stops": len(seq),
                "geometry_stops": _line_geometry_stops,
            },
            "sequence": seq,
        }

    directions: dict = {}
    if len(normal_by_dir["ALLER"]) and len(normal_by_dir["RETOUR"]):
        cols = ["day", "bus", "trip_id", "trip_start", "trip_end", "total_elapsed",
               "match_rate", "n_stops", "mean_dwell_s", "full"]
        merged = normal_by_dir["ALLER"][cols].merge(
            normal_by_dir["RETOUR"][cols], on=["day", "bus"], suffixes=("_a", "_r"))
        merged = merged[merged["trip_start_r"] > merged["trip_end_a"]]
        if len(merged):
            merged["gap_min"] = (merged["trip_start_r"] - merged["trip_end_a"]).dt.total_seconds() / 60
            # ne garder QUE le retour le plus proche après chaque aller -- un vrai cycle
            # aller->pause->redépart, pas un retour plus tardif sans rapport le même jour
            merged = (merged.sort_values("gap_min")
                     .drop_duplicates(subset=["day", "bus", "trip_id_a"], keep="first"))
            merged["score"] = ((merged["total_elapsed_a"] - med_by_dir["ALLER"]).abs()
                               + (merged["total_elapsed_r"] - med_by_dir["RETOUR"]).abs())
            best_pair = merged.sort_values("score").iloc[0]
            row_a = pd.Series({c: best_pair.get(f"{c}_a", best_pair.get(c)) for c in cols})
            row_r = pd.Series({c: best_pair.get(f"{c}_r", best_pair.get(c)) for c in cols})
            directions = {"ALLER": _build(row_a, "ALLER"), "RETOUR": _build(row_r, "RETOUR")}

    if not directions:
        for d in ("ALLER", "RETOUR"):
            normal = normal_by_dir[d]
            if len(normal) == 0:
                continue
            # trajets complets d'abord si disponibles, puis durée la plus proche de la médiane
            full = normal[normal["full"] == True]  # noqa: E712 -- `full` est nullable (lignes en boucle)
            pool = full if len(full) >= 3 else normal
            best = pool.iloc[(pool["total_elapsed"] - med_by_dir[d]).abs().argsort().iloc[0]]
            directions[d] = _build(best, d)

    if not directions:
        raise HTTPException(status_code=404, detail="No normal trip found for this line")
    return {"societe": societe, "line": line, "directions": directions}


@app.get("/api/anomaly-patterns")
@_limit_concurrency
def anomaly_patterns(  # def, pas async def -- voir la note dans get_anomaly_history
    societe: str,
    line: Optional[str] = None
):
    """Anomaly-RATE aggregations: by line, direction, hour-of-day, and worst buses."""
    try:
        _models, trips = _filter_trips(societe, line)
        if len(trips) == 0:
            return {"societe": societe, "line": line, "total_trips": 0,
                    "total_anomalies": 0, "overall_rate": 0.0,
                    "by_line": [], "by_dir": [], "by_hour": [], "by_bus": []}

        trips["hour"] = pd.to_datetime(trips["trip_start"]).dt.hour

        def rate_by(col):
            g = trips.groupby(col).agg(
                trips=("anomaly", "size"),
                anomalies=("anomaly", "sum"),
            ).reset_index()
            g["anomalies"] = g["anomalies"].astype(int)
            g["rate"] = (g["anomalies"] / g["trips"]).round(3)
            return g

        by_line = rate_by("line").sort_values("rate", ascending=False)
        by_dir = rate_by("dir")
        by_hour = rate_by("hour").sort_values("hour")
        by_bus = rate_by("bus")
        by_bus = by_bus[by_bus["trips"] >= 5].sort_values(
            ["rate", "anomalies"], ascending=False).head(15)

        return {
            "societe": societe,
            "line": line,
            "total_trips": int(len(trips)),
            "total_anomalies": int(trips["anomaly"].sum()),
            "overall_rate": round(float(trips["anomaly"].mean()), 3),
            "by_line": by_line.to_dict("records"),
            "by_dir": by_dir.to_dict("records"),
            "by_hour": by_hour.to_dict("records"),
            "by_bus": by_bus.to_dict("records"),
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Anomaly models not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing patterns: {str(e)}")

# Delay Prediction - Auto Mode
@app.post("/api/predict/delay/auto")
async def predict_delay_auto(
    societe: str,
    line: str,
    bus: int,
    day: str,
    model_type: str = "hgbm"
):
    """Auto-detect current status and predict ETA."""
    try:
        status = await get_bus_status(societe, line, bus, day)
        
        if status["current_seq"] >= 0:
            models = model_manager.get("delay")
            eta_df = delay.predict_eta(
                models,
                societe=societe,
                line=line,
                direction=status["direction"],
                dep_time=status["trip_start"],
                current_seq=status["current_seq"],
                current_delay_min=status["current_delay_min"],
                model_type=model_type
            )
            
            predictions = eta_df.to_dict(orient="records")
            
            return {
                "societe": societe,
                "line": line,
                "bus": bus,
                "direction": status["direction"],
                "day": day,
                "trip_start": status["trip_start"],
                "current_seq": status["current_seq"],
                "current_stop": status["current_stop"],
                "current_delay_min": status["current_delay_min"],
                "total_stops": status["total_stops"],
                "remaining_stops": status["remaining_stops"],
                "all_stops": status["all_stops"],
                "predictions": predictions,
                "model_used": model_type,
                "mode": "auto"
            }
        else:
            return {
                "societe": societe,
                "line": line,
                "bus": bus,
                "mode": "auto",
                "message": "Trip hasn't started yet",
                "status": status
            }
    except KeyError:
        raise HTTPException(status_code=503, detail="Delay models not loaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Delay Prediction - Manual Mode
@app.post("/api/predict/delay/manual")
async def predict_delay_manual(request: DelayPredictionManualRequest):
    """Manual delay prediction with user-specified parameters."""
    try:
        models = model_manager.get("delay")
        eta_df = delay.predict_eta(
            models,
            societe=request.societe,
            line=request.line,
            direction=request.direction,
            dep_time=request.dep_time,
            current_seq=request.current_seq,
            current_delay_min=request.current_delay_min,
            model_type=request.model_type
        )
        
        predictions = eta_df.to_dict(orient="records")
        
        return {
            "societe": request.societe,
            "line": request.line,
            "direction": request.direction,
            "dep_time": request.dep_time,
            "current_seq": request.current_seq,
            "current_delay_min": request.current_delay_min,
            "predictions": predictions,
            "model_used": request.model_type,
            "mode": "manual"
        }
    except KeyError:
        raise HTTPException(status_code=503, detail="Delay models not loaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# GPS Fallback
@app.post("/api/predict/gps-fallback")
async def predict_gps_fallback(request: GPSFallbackRequest):
    try:
        models = model_manager.get("fallback")
        foundation = model_manager.foundation_slice(request.societe, line=request.line,
                                                    bus=request.bus, day=request.day)

        if foundation is not None:
            if len(foundation) == 0:
                available = model_manager.trip_scopes(request.societe, line=request.line)
                available_buses = (available["bus"].unique().tolist()
                                   if available is not None and len(available) else [])
                available_days = (available["day"].unique().tolist()
                                  if available is not None and len(available) else [])
                raise HTTPException(
                    status_code=404,
                    detail=f"No data for bus {request.bus} on day {request.day}. "
                           f"Available buses: {available_buses[:5]}, Available days: {available_days[:5]}"
                )
        
        gps_db = get_db("Historique_pos")
        cfg = fdn.Config()
        
        mongo_day = f"d{request.day}"
        g = fdn.load_pings(gps_db, mongo_day, request.line, request.bus)
        
        if len(g) == 0:
            # `foundation` est déjà scopé à ce (societe, line, bus, day) exact -- pas besoin
            # de refiltrer, juste se prémunir contre None (BDD de référence injoignable).
            foundation_data = foundation if foundation is not None else pd.DataFrame()

            if len(foundation_data) > 0:
                first_stop = foundation_data.iloc[0]
                usable = _load_usable_lines()
                key = (request.line, request.societe)
                if key in usable:
                    stops = usable[key]
                    stop_row = stops[stops["seq"] == first_stop["route_seq"]]
                    if len(stop_row) > 0:
                        return {
                            "lat": float(stop_row.iloc[0]["lat"]),
                            "lon": float(stop_row.iloc[0]["lon"]),
                            "s_m": float(stop_row.iloc[0]["s_m"]) / 1000,
                            "uncertainty_m": 50.0,
                            "method": "foundation_data_fallback",
                            "ks_m": float(stop_row.iloc[0]["s_m"]) / 1000,
                            "correction_m": 0.0
                        }
            
            raise HTTPException(status_code=404, detail=f"No GPS pings found for bus {request.bus} on {request.day}")
        
        g = fdn.clean_pings(g, cfg)

        usable = _load_usable_lines()
        key = (request.line, request.societe)
        if key not in usable:
            raise HTTPException(status_code=404, detail=f"Line {request.line} geometry not found")
        stops = usable[key]
        
        g, route_len = fdn.project_to_route(g, stops, cfg)
        g_filtered = gps_fallback.run_kalman(g, route_len)
        
        query_time = pd.Timestamp(request.query_time)
        result = gps_fallback.predict_position(
            models, g_filtered, query_time, stops
        )
        
        if result is None:
            raise HTTPException(status_code=404, detail="Could not predict position at query time")
        
        return result
        
    except KeyError:
        raise HTTPException(status_code=503, detail="GPS fallback models not loaded")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Chatbot
@app.post("/api/chatbot/ask")
async def chatbot_ask(request: ChatbotRequest):
    try:
        models = model_manager.get("chatbot")
        result = chatbot.ask(models, request.query, k=request.k)
        
        return {
            "answer": result["answer"],
            "context": result["context"],
            "tokens_used": result.get("tokens_used", 0)
        }
    except KeyError:
        detail = ("Chatbot disabled in this deployment (see ENABLED_MODULES / ENABLE_CHATBOT)"
                  if "chatbot" not in ENABLED_MODULES else "Chatbot models not loaded")
        raise HTTPException(status_code=503, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Forecast
@app.post("/api/predict/delay/forecast")
async def forecast_delay(
    societe: str,
    line: str,
    direction: str,
    periods: int = Query(30, ge=1, le=90)
):
    try:
        models = model_manager.get("delay")
        forecast_df = delay.forecast(
            models,
            societe=societe,
            line=line,
            direction=direction,
            periods=periods
        )
        if forecast_df is None:
            raise HTTPException(status_code=404, detail="No forecast model found")
        return forecast_df.to_dict(orient="records")
    except KeyError:
        raise HTTPException(status_code=503, detail="Delay models not loaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# GPS track / gaps / examples — for the event-driven signal-loss demo
# MongoDB n'existe pas en production (Render) -- seules les webservices de la plateforme
# y sont joignables (WEBSERVICE_URL, décision utilisateur 2026-07-17 : "je n'essaie pas
# d'atteindre MongoDB directement, j'utilise ces webservices"). Ce drapeau évite de payer
# les ~30s de server-selection PyMongo à CHAQUE bus-jour vérifié quand Mongo est absent :
# premier échec -> Mongo considéré indisponible 10 min, tous les appels suivants basculent
# immédiatement sur le webservice.
_mongo_gps_down_until = 0.0


def _ws_day_groups(societe: str, day: str) -> Optional[dict]:
    """Pings pour (societe, day), groupés par (line, bus) -- magasin poussé d'abord (voir
    scripts/push_live_day.py), webservice ensuite (UN seul appel pour tous les bus-jours
    de la journée, le service ne filtre pas par ligne/bus). None si aucune source n'a ce
    jour."""
    pings = _ingest_read(_ingest_path("gps", day, societe))
    if pings is None and ws.WEBSERVICE_URL:
        try:
            pings = ws.get_pings_for_day(day, societe=societe)
        except Exception as e:
            print(f"  webservice pings indisponibles ({societe}/{day}): {e}")
            return None
    return ws.group_pings_by_bus_line(pings) if pings else None


def _build_gps_track(societe: str, line: str, bus: int, day: str, ws_groups: dict | None = None):
    """Raw pings -> cleaned -> projected -> Kalman-filtered track.

    Returns (g_filtered, stops_frame, route_len_m) or None if no usable data.
    Same chain used by /api/predict/gps-fallback; kept in one place.

    Source des pings : MongoDB (Historique_pos) en local ; repli sur le webservice
    getPingsForDay quand Mongo est indisponible (production Render) OU quand Mongo n'a
    pas ce jour (jour "live" pas encore inséré par le traitement de nuit). `ws_groups`
    (sortie de _ws_day_groups) évite de refaire l'appel webservice pour chaque bus d'une
    même journée quand l'appelant vérifie plusieurs bus-jours (boucle détours).
    """
    global _mongo_gps_down_until
    cfg = fdn.Config()
    g = None
    if time.time() >= _mongo_gps_down_until:
        try:
            g = fdn.load_pings(get_db("Historique_pos"), f"d{day}", line, int(bus))
        except Exception as e:
            _mongo_gps_down_until = time.time() + 600
            print(f"  Mongo GPS injoignable ({e.__class__.__name__}) -- repli webservice pendant 10 min")
            g = None
    if g is None or len(g) == 0:
        rows = None
        if ws_groups is not None:
            rows = ws_groups.get((str(line), str(bus)))
        else:
            groups = _ws_day_groups(societe, day)
            if groups:
                rows = groups.get((str(line), str(bus)))
        if not rows:
            return None
        g = pd.DataFrame(ws.pings_to_score_live_rows(rows))
        # Mêmes colonnes que load_pings (t/lat/lon/speed/voyage) ; horodatages du service
        # avec offset -> naïfs, comme partout ailleurs dans le pipeline (voir
        # _score_all_gps_live pour l'erreur réelle qui a motivé ce tz_localize).
        g["t"] = pd.to_datetime(g["t"]).dt.tz_localize(None)
        g = g.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)
        if len(g) == 0:
            return None
    g = fdn.clean_pings(g, cfg)
    usable = _load_usable_lines()
    key = (line, societe)
    if key not in usable:
        return None
    stops = usable[key]
    g, route_len = fdn.project_to_route(g, stops, cfg)
    g_filtered = gps_fallback.run_kalman(g, route_len)
    return g_filtered, stops, route_len


@app.get("/api/gps-track")
async def gps_track(
    societe: str,
    line: str,
    bus: int,
    day: str,
    max_points: int = 2000
):
    """Full projected + Kalman ping trace for a bus-day, plus the route polyline.

    Each point carries both the raw GPS position and the Kalman state (ks/kv/kp),
    so the client can replay normal motion AND propagate an estimate during a gap.
    """
    built = _build_gps_track(societe, line, bus, day)
    if built is None:
        raise HTTPException(status_code=404, detail="No GPS track available for this bus-day")
    g, stops, route_len = built

    from src.data.fallback import s_to_latlon

    if len(g) > max_points:
        step = len(g) // max_points + 1
        g = g.iloc[::step].reset_index(drop=True)

    track = []
    for _, r in g.iterrows():
        klat, klon = s_to_latlon(float(r["ks"]), stops)
        track.append({
            "t": pd.Timestamp(r["t"]).isoformat(),
            "lat": round(float(r["lat"]), 6),
            "lon": round(float(r["lon"]), 6),
            "klat": round(klat, 6),
            "klon": round(klon, 6),
            "s_m": round(float(r["s"]) / 1000, 3),
            "ks_m": round(float(r["ks"]) / 1000, 3),
            "kv": round(float(r["kv"]), 2),
            "uncertainty_m": round(float(r["kp"]), 0),
            "speed": round(float(r.get("speed", 0) or 0), 1),
            "signal_gap": bool(r["signal_gap"]),
            "gap_s": float(r["gap_s"]) if pd.notna(r.get("gap_s")) else 0.0,
        })

    route = [{
        "seq": int(s.seq), "stop": getattr(s, "stop", ""),
        "lat": float(s.lat), "lon": float(s.lon),
        "s_m": round(float(s.s_m) / 1000, 3),
    } for s in stops.itertuples()]

    return {
        "societe": societe, "line": line, "bus": bus, "day": day,
        "route_len_km": round(route_len / 1000, 2),
        "n_points": len(track),
        "track": track,
        "route": route,
    }


@app.get("/api/gps-gaps")
async def gps_gaps(societe: str, line: str, bus: int, day: str):
    """Signal-gap table for a bus-day (start/end, duration, route context)."""
    built = _build_gps_track(societe, line, bus, day)
    if built is None:
        raise HTTPException(status_code=404, detail="No GPS track available for this bus-day")
    g, _stops, _ = built

    from src.data.fallback import gap_table
    gt = gap_table(g)
    gaps = []
    for _, r in gt.iterrows():
        gaps.append({
            "t_start": pd.Timestamp(r["t_start"]).isoformat(),
            "t_end": pd.Timestamp(r["t_end"]).isoformat(),
            "gap_min": float(r["gap_min"]),
            "s_start_km": float(r["s_start_km"]),
            "s_end_km": float(r["s_end_km"]),
            "dist_covered_km": float(r["dist_covered_km"]),
            "speed_before_kph": float(r["speed_before_kph"]),
        })
    return {"societe": societe, "line": line, "bus": bus, "day": day,
            "gaps": gaps, "n_gaps": len(gaps)}


@app.get("/api/gps-gap-examples")
async def gps_gap_examples(
    societe: str,
    line: str,
    limit: int = 5,
    scan_days: int = 6,
    max_scan: int = 18
):
    """Rank recent bus-days on a line by their longest signal gap.

    Powers the 'Replay a real signal-loss event' button — jumps straight to a
    genuine, dramatic historical case.
    """
    df = model_manager.trip_scopes(societe, line=line)
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")

    sub = df[["day", "bus"]].drop_duplicates()
    if len(sub) == 0:
        return {"societe": societe, "line": line, "examples": [], "scanned": 0}

    days = sorted(sub["day"].unique().tolist(), reverse=True)[:scan_days]
    from src.data.fallback import gap_table

    examples, scanned = [], 0
    for day in days:
        if scanned >= max_scan:
            break
        for bus in sub[sub["day"] == day]["bus"].unique().tolist():
            if scanned >= max_scan:
                break
            scanned += 1
            try:
                built = _build_gps_track(societe, line, int(bus), day)
            except Exception:
                continue
            if built is None:
                continue
            g, _stops, _ = built
            gt = gap_table(g)
            if len(gt) == 0:
                continue
            worst = gt.loc[gt["gap_s"].idxmax()]
            examples.append({
                "day": day, "bus": int(bus),
                "max_gap_min": round(float(worst["gap_s"]) / 60, 1),
                "n_gaps": int(len(gt)),
                "s_start_km": float(worst["s_start_km"]),
                "s_end_km": float(worst["s_end_km"]),
            })

    examples.sort(key=lambda e: e["max_gap_min"], reverse=True)
    return {"societe": societe, "line": line, "examples": examples[:limit], "scanned": scanned}


# Live ETA helpers — active buses + ETA to a rider's stop
@app.get("/api/active-buses")
async def active_buses(
    societe: str,
    line: str,
    query_time: Optional[str] = None,
    day: Optional[str] = None
):
    """Buses running on a line at query_time (active / upcoming / completed).

    Prefers the live GPS webservice day (see _live_gps_day) over the static precomputed
    dataset, same as /api/current-anomalies -- without this, "ETA en direct" was silently
    stuck on whatever day the last training snapshot happened to cover (confirmed
    2026-07-15: the reference DB's latest day doesn't move even though live webservices
    have been feeding real data for a while). `foundation_slice` already merges in the
    live per-stop data automatically when `day` equals that live day (see its docstring),
    so passing it through here is enough -- no separate live-scoring call needed.
    """
    live_day = _live_gps_day(societe)
    is_live = day is None and live_day is not None
    day = day or live_day or latest_day_for(societe, line)
    qt = pd.Timestamp(query_time) if query_time else None
    # Re-anchor the wall-clock time-of-day onto the operating day so "active/upcoming"
    # is judged on the same date as the trips.
    if qt is not None:
        day_ts = pd.Timestamp(datetime.strptime(day, "%Y%m%d").date())
        qt = day_ts + (qt - qt.normalize())

    sub = model_manager.foundation_slice(societe, line=line, day=day)
    if sub is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    if len(sub) == 0:
        return {"day": day, "live": is_live,
                "query_time": qt.isoformat() if qt is not None else None, "buses": []}

    trips = sub.groupby(["bus", "trip_id"]).agg(
        trip_start=("trip_start", "first"),
        trip_end=("trip_end", "first"),
        dir=("dir", "first"),
        n_stops=("seq", "count"),
    ).reset_index()

    buses = []
    for _, t in trips.iterrows():
        ts, te = pd.Timestamp(t["trip_start"]), pd.Timestamp(t["trip_end"])
        if qt is not None:
            if ts <= qt <= te:
                status = "active"
            elif ts > qt:
                status = "upcoming"
            else:
                status = "completed"
        else:
            status = "unknown"
        buses.append({
            "bus": int(t["bus"]), "trip_id": int(t["trip_id"]), "dir": t["dir"],
            "trip_start": ts.isoformat(), "trip_end": te.isoformat(),
            "n_stops": int(t["n_stops"]), "status": status,
        })

    order = {"active": 0, "upcoming": 1, "unknown": 2, "completed": 3}
    buses.sort(key=lambda b: (order[b["status"]], b["trip_start"]))
    return {"day": day, "live": is_live, "query_time": qt.isoformat() if qt is not None else None,
            "buses": buses}


def _bus_status_at(societe: str, line: str, bus: int, day: str,
                   query_time: pd.Timestamp) -> Optional[Dict]:
    """Where is a bus AT a given wall-clock time on `day`? (live-ETA aware).

    Unlike get_bus_status (which always uses the latest *completed* trip), this finds
    the trip in progress at query_time and the last stop reached by then, so the delay
    model can predict forward to the rider's stop. Delay is baseline-driven.
    """
    sub = model_manager.foundation_slice(societe, line=line, bus=bus, day=day)
    if sub is None or len(sub) == 0:
        return None

    day_ts = pd.Timestamp(datetime.strptime(day, "%Y%m%d").date())
    qt = day_ts + (query_time - query_time.normalize())
    sub["trip_start"] = pd.to_datetime(sub["trip_start"])
    sub["trip_end"] = pd.to_datetime(sub["trip_end"])
    sub["arrival"] = pd.to_datetime(sub["arrival"])

    trips = sub.groupby("trip_id").agg(ts=("trip_start", "first"),
                                       te=("trip_end", "first")).reset_index()
    active = trips[(trips["ts"] <= qt) & (trips["te"] >= qt)]
    if len(active):
        trip_id, status = active.iloc[0]["trip_id"], "active"
    else:
        upcoming = trips[trips["ts"] > qt].sort_values("ts")
        past = trips[trips["te"] < qt].sort_values("te")
        if len(upcoming):
            trip_id, status = upcoming.iloc[0]["trip_id"], "upcoming"
        elif len(past):
            trip_id, status = past.iloc[-1]["trip_id"], "completed"
        else:
            return None

    trip = sub[sub["trip_id"] == trip_id].sort_values("seq")
    direction = trip.iloc[0]["dir"]
    trip_start = trip.iloc[0]["trip_start"]

    if status == "upcoming":
        cur, current_seq, elapsed = None, 0, 0.0
    else:
        cutoff = qt if status == "active" else trip["arrival"].max()
        arrived = trip[trip["arrival"].notna() & (trip["arrival"] <= cutoff)]
        if len(arrived):
            cur = arrived.iloc[-1]
            current_seq = int(cur["seq"])
            elapsed = (cur["arrival"] - trip_start).total_seconds() / 60
        else:
            cur, current_seq, elapsed = None, 0, 0.0

    # Baseline-driven current delay (consistent with the ETA model)
    current_delay = 0.0
    if cur is not None:
        try:
            baseline = model_manager.get("delay")["baseline"]
            b = baseline[(baseline["societe"] == societe) & (baseline["line"] == line) &
                         (baseline["dir"] == direction) & (baseline["seq"] == current_seq)]
            if len(b):
                current_delay = float(elapsed - b.iloc[0]["expected_min"])
        except Exception:
            pass

    all_stops = [{
        "seq": int(r["seq"]), "stop": r["stop"],
        "arrival": r["arrival"].isoformat() if pd.notna(r["arrival"]) else None,
    } for _, r in trip.iterrows()]

    return {
        "trip_id": int(trip_id), "status": status, "direction": direction,
        "trip_start": trip_start.isoformat(),
        "current_seq": current_seq,
        "current_stop": cur["stop"] if cur is not None else None,
        "current_delay_min": round(current_delay, 1),
        "all_stops": all_stops, "query_time": qt.isoformat(),
    }


@app.get("/api/eta-to-stop")
async def eta_to_stop(
    societe: str,
    line: str,
    bus: int,
    day: str,
    target_seq: int,
    query_time: Optional[str] = None,
    model_type: str = "hgbm"
):
    """Predicted arrival of a bus at the rider's stop + minutes-away from query_time."""
    qt_in = pd.Timestamp(query_time) if query_time else pd.Timestamp(
        datetime.strptime(day, "%Y%m%d").date()) + pd.Timedelta(hours=datetime.now().hour,
                                                                 minutes=datetime.now().minute)
    status = _bus_status_at(societe, line, bus, day, qt_in)
    if status is None:
        raise HTTPException(status_code=404, detail="No trip for this bus on this day")

    qt = pd.Timestamp(status["query_time"])
    try:
        models = model_manager.get("delay")
        eta_df = delay.predict_eta(
            models, societe=societe, line=line, direction=status["direction"],
            dep_time=status["trip_start"], current_seq=status["current_seq"],
            current_delay_min=status["current_delay_min"], model_type=model_type,
        )
    except KeyError:
        raise HTTPException(status_code=503, detail="Delay models not loaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    preds = eta_df.to_dict(orient="records")

    # Canonical stop name per baseline seq for this line+direction (predictions carry
    # baseline seq, which differs from a partial trip's own seq numbering).
    name_map = {}
    df = model_manager.foundation_slice(societe, line=line)
    if df is not None:
        sub = df[df["dir"] == status["direction"]]
        if len(sub):
            name_map = sub.groupby("seq")["stop"].agg(
                lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0]).to_dict()

    out = {
        "bus": bus, "line": line, "day": day, "target_seq": target_seq,
        "status": status["status"],
        "current_seq": status["current_seq"],
        "current_stop": status["current_stop"],
        "current_delay_min": status["current_delay_min"],
        "trip_start": status["trip_start"],
        "direction": status["direction"],
        "query_time": status["query_time"],
        "all_stops": status["all_stops"],
        "predictions": [
            {**p, "stop": name_map.get(int(p["seq"]), f"Stop {int(p['seq'])}"),
             "eta": pd.Timestamp(p["eta"]).isoformat()}
            for p in preds
        ],
    }

    row = next((p for p in preds if int(p["seq"]) == int(target_seq)), None)
    if row is not None:
        eta = pd.Timestamp(row["eta"])
        out["eta"] = eta.isoformat()
        out["pred_delay_min"] = round(float(row["pred_delay_min"]), 1)
        out["minutes_away"] = round((eta - qt).total_seconds() / 60, 1)
    else:
        out["eta"] = None
        out["minutes_away"] = None
        out["note"] = "Stop already passed or not in prediction window"
    return out


# Main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)