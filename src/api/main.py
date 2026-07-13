"""FastAPI serving layer for WiniCari AI"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import model modules
from src.models import delay, gps_fallback, anomaly, chatbot, ticket_anomaly
from src.data import foundation as fdn
from src.data import reference_db as rdb
from src.data import model_version as mv
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
    _foundation_data = None
    _stops_data = {}
    _stop_coords = {}   # f"{societe}_{line}" -> {stop_name: (lat, lon)}
    _stop_coord_suspect = {}   # f"{societe}_{line}" -> {stop_name: bool, chronically wrong coords}
    _gps_trip_counts = None   # (societe, line, str(bus), day) -> n_trips, see get_gps_trip_count
    _latest_day = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_all(self):
        print("Loading models...")
        
        try:
            foundation_path = Path("data/processed/foundation_arrivals_full.parquet")
            if foundation_path.exists():
                self._foundation_data = pd.read_parquet(foundation_path)
                # Mesuré (2026-07-13) : ces colonnes sont des chaînes Python (object) avec
                # très peu de valeurs distinctes sur 637k lignes (ex. `dir` = 2 valeurs mais
                # 39.8 MB en mémoire) -- category ramène ça à quelques centaines de Ko/colonne
                # sans rien changer au comportement (égalité/groupby/tri identiques). C'est
                # ~397 Mo des ~486 Mo de ce DataFrame, la cause principale du dépassement des
                # 512 Mo sur le déploiement Render slim. `day` reste ORDERED (categories
                # triées) car `.max()` est utilisé dessus (horloge démo) -- .max() lève une
                # erreur sur une categorical NON ordonnée.
                _low_card_cols = ["stop", "origin_idle_stop", "end_idle_stop", "dir",
                                  "societe", "line", "full", "trip_dark_before_stop",
                                  "trip_dark_after_stop"]
                for col in _low_card_cols:
                    if col in self._foundation_data.columns:
                        self._foundation_data[col] = self._foundation_data[col].astype("category")
                if "day" in self._foundation_data.columns:
                    day_cat = pd.CategoricalDtype(
                        categories=sorted(self._foundation_data["day"].unique()), ordered=True)
                    self._foundation_data["day"] = self._foundation_data["day"].astype(day_cat)
                print(f"Foundation data loaded ({len(self._foundation_data):,} rows, "
                      f"{round(self._foundation_data.memory_usage(deep=True).sum()/1e6, 1)} MB in memory)")
                
                # Build stops mapping for each line
                for societe in self._foundation_data['societe'].unique():
                    for line in self._foundation_data[self._foundation_data['societe'] == societe]['line'].unique():
                        key = f"{societe}_{line}"
                        stops_data = self._foundation_data[
                            (self._foundation_data['societe'] == societe) &
                            (self._foundation_data['line'] == line)
                        ].sort_values('route_seq')[['route_seq', 'stop']].drop_duplicates()
                        self._stops_data[key] = stops_data.to_dict('records')
                
                print(f"Stops mapping built for {len(self._stops_data)} lines")

                # Build stop lat/lon from the reference DB (used for map view)
                try:
                    usable = _load_usable_lines()
                    for (line_code, soc), sf in usable.items():
                        key = f"{soc}_{line_code}"
                        self._stop_coords[key] = {
                            str(row["name"]): (float(row["lat"]), float(row["lon"]))
                            for _, row in sf.iterrows()
                            if pd.notna(row.get("lat")) and pd.notna(row.get("lon"))
                        }
                    print(f"Stop coordinates loaded for {len(self._stop_coords)} lines")
                except Exception as e:
                    print(f"Stop coordinates not available: {e}")

                # Demo clock: "today" = the most recent day with real data.
                self._latest_day = str(self._foundation_data["day"].max())
                print(f"Demo 'today' set to latest data day: {self._latest_day}")
            else:
                print("Foundation data not found")
        except Exception as e:
            print(f"Failed to load foundation data: {e}")

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

    def get_foundation_data(self) -> Optional[pd.DataFrame]:
        return self._foundation_data

    def get_stops(self, societe: str, line: str) -> List[Dict]:
        key = f"{societe}_{line}"
        return self._stops_data.get(key, [])

    def get_stop_coords(self, societe: str, line: str) -> dict:
        """Maps stop_name -> (lat, lon) for a given line."""
        return self._stop_coords.get(f"{societe}_{line}", {})

    def get_gps_trip_count(self, societe: str, line: str, bus: str, day: str) -> Optional[int]:
        """Nombre de trajets GPS réels pour ce (société, ligne, bus, jour) exact -- ou
        `None` si le cross-check GPS n'est pas disponible DU TOUT sur ce déploiement
        (foundation_arrivals_full.parquet absent -- voir Dockerfile.render, exclu du
        slim anomaly-only pour tenir dans 512 Mo). `None` DOIT rester distinct de `0` :
        `0` veut dire "vérifié, aucun trajet GPS ce jour-là" (vrai signal, voir
        is_no_service dans `_ticket_rows_with_reasons`), alors que `None` veut dire "pas
        vérifiable ici" -- les confondre ferait passer CHAQUE jour-bus signalé pour "pas
        d'anomalie réelle" et les ferait tous disparaître à tort de la vue client.

        Sert à distinguer une panne de machine à tickets (le bus a bien circulé --
        confirmé côté GPS -- mais quasi aucun ticket enregistré) d'une vraie anomalie de
        recette/fraude. Lazy + mise en cache : construit au premier appel à partir de
        `foundation_arrivals_full.parquet` (déjà chargé en mémoire), pas de nouvelle
        lecture disque par requête.
        """
        if self._gps_trip_counts is None:
            fa = self.get_foundation_data()
            if fa is None:
                self._gps_trip_counts = {}
            else:
                counts = (fa.assign(bus=fa["bus"].astype(str))
                          .groupby(["societe", "line", "bus", "day"], observed=True)["trip_id"].nunique())
                self._gps_trip_counts = counts.to_dict()
        if not self._gps_trip_counts and self.get_foundation_data() is None:
            return None
        return int(self._gps_trip_counts.get((societe, line, str(bus), day), 0))

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
        if key not in self._stop_coord_suspect and self._foundation_data is not None:
            sub = self._foundation_data[
                (self._foundation_data["societe"] == societe) &
                (self._foundation_data["line"] == line)
            ]
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
        """Demo 'today' most recent day present in the foundation data."""
        if self._latest_day is None and self._foundation_data is not None:
            self._latest_day = str(self._foundation_data["day"].max())
        return self._latest_day

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
    print("=" * 60)
    print("WiniCari AI API Starting...")
    print("=" * 60)
    model_manager.load_all()
    yield
    print("Shutting down...")

app = FastAPI(
    title="WiniCari AI API",
    description="API for bus delay prediction, GPS fallback, anomaly detection, and RAG chatbot",
    version="2.0.0",
    lifespan=lifespan
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

# Helper Functions
def get_available_options(societe: Optional[str] = None, line: Optional[str] = None) -> Dict:
    df = model_manager.get_foundation_data()
    if df is None or len(df) == 0:
        return {"companies": [], "lines": [], "directions": [], "buses": [], "days": []}
    
    result = {}
    result["companies"] = sorted(df["societe"].unique().tolist())
    
    df_filtered = df.copy()
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
    df = model_manager.get_foundation_data()
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded(),
        "models": model_manager.get_loaded_models(),
        "enabled_modules": sorted(ENABLED_MODULES),
        "foundation_data": df is not None,
        "rows": len(df) if df is not None else 0,
        "latest_day": model_manager.get_latest_day(),
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
    df = model_manager.get_foundation_data()
    if df is None:
        return demo_today()
    mask = df["societe"] == societe
    if line:
        mask &= df["line"] == line
    days = df.loc[mask, "day"]
    return str(days.max()) if len(days) else demo_today()

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
    df = model_manager.get_foundation_data()
    if df is None:
        return {"lines": []}
    sub = df[df["societe"] == societe]
    counts = sub.groupby("line", observed=True)["trip_id"].nunique().sort_values(ascending=False)
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
    df = model_manager.get_foundation_data()
    if df is None:
        return {"societe": societe, "lines": [], "by_line": {}}
    sub = df[df["societe"] == societe][["line", "dir"]].drop_duplicates()
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
    df = model_manager.get_foundation_data()
    if df is None:
        return {"days": []}
    
    df_filtered = df.copy()
    if societe:
        df_filtered = df_filtered[df_filtered["societe"] == societe]
    if line:
        df_filtered = df_filtered[df_filtered["line"] == line]
    
    days = sorted(df_filtered["day"].unique().tolist(), reverse=True)
    return {"days": days[:30]}

@app.get("/api/buses-for-line")
async def get_buses_for_line(societe: str, line: str):
    """All unique buses that have ever run on a given line (across all days in the foundation)."""
    df = model_manager.get_foundation_data()
    if df is None:
        return {"buses": []}
    sub = df[(df["societe"] == societe) & (df["line"] == line)]
    buses = sorted(int(b) for b in sub["bus"].unique())
    return {"buses": buses}


@app.get("/api/buses-for-day")
async def get_buses_for_day(
    societe: str,
    line: str,
    day: str
):
    df = model_manager.get_foundation_data()
    if df is None:
        return {"buses": []}
    
    filtered = df[
        (df["societe"] == societe) &
        (df["line"] == line) &
        (df["day"] == day)
    ]
    
    buses = sorted(filtered["bus"].unique().tolist())
    return {"buses": buses}

@app.get("/api/days-for-line")
async def get_days_for_line(
    societe: str,
    line: str
):
    df = model_manager.get_foundation_data()
    if df is None:
        return {"days": []}
    
    filtered = df[
        (df["societe"] == societe) &
        (df["line"] == line)
    ]
    
    days = sorted(filtered["day"].unique().tolist(), reverse=True)
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
    df = model_manager.get_foundation_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    
    # Get bus data for this day
    bus_data = df[
        (df["societe"] == societe) &
        (df["line"] == line) &
        (df["bus"].astype(str) == str(bus)) &
        (df["day"] == day)
    ].sort_values("trip_start")
    
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
    df = model_manager.get_foundation_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    
    bus_data = df[
        (df["societe"] == societe) &
        (df["line"] == line) &
        (df["bus"].astype(str) == str(bus)) &
        (df["day"] == day)
    ].sort_values("trip_start")
    
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
                  include_data_bugs: bool = False, dir: Optional[str] = None):
    """Filter the precomputed scored-trips table. Returns (models, trips_df)."""
    models = model_manager.get("anomaly")
    trips = models["trips"]
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
    ex = anomaly.explain_trips(models, trips).sort_values(
        "anomaly_strength", ascending=False)
    if limit:
        ex = ex.head(limit)

    # Isolation Forest est entraîné PAR OPÉRATEUR (>=30 trajets, voir models/anomaly.py
    # train()), l'autoencodeur LSTM PAR OPÉRATEUR aussi mais avec un seuil plus haut
    # (>=200, un autoencodeur a plus de paramètres à apprendre) -- EN DESSOUS, repli sur un
    # modèle GLOBAL entraîné sur TOUTES les sociétés/lignes confondues. Dans les deux cas,
    # même un modèle "dédié" à un opérateur reste entraîné sur TOUTES ses lignes poolées
    # ensemble, jamais sur une ligne seule -- il n'y a pas de modèle par ligne. Exposé ici
    # pour que le dashboard puisse avertir : "ce résultat compare au réseau global / à
    # l'opérateur entier, pas à cette ligne spécifiquement, et s'affinera avec plus de
    # données" plutôt que de laisser croire à une comparaison fine qui n'existe pas encore.
    if_models = models.get("if_models", {})
    lstm_models = models.get("lstm_models", {})

    rows = []
    for _, r in ex.iterrows():
        soc = r.get("societe")
        if_dedicated = soc in if_models
        lstm_dedicated = soc in lstm_models
        rows.append({
            "model_if_dedicated": bool(if_dedicated),
            "model_lstm_dedicated": bool(lstm_dedicated),
            "model_low_data": not (if_dedicated and lstm_dedicated),
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
    """Filter the precomputed scored-days table. Returns (models, days_df)."""
    models = model_manager.get("ticket_anomaly")
    days = models["days"]
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
async def ticket_anomaly_history(
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
async def ticket_anomaly_explain(
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
async def ticket_anomaly_patterns(societe: str, line: Optional[str] = None):
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
async def ticket_anomaly_stations(societe: str, line: str, bus: str, day: str):
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
async def ticket_anomaly_reference(societe: str, line: str):
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
async def get_anomaly_history(
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
async def get_current_anomalies(
    societe: str,
    line: Optional[str] = None,
    dir: Optional[str] = None
):
    """Anomalies for the latest day the selected scope actually operated."""
    today = latest_day_for(societe, line)
    try:
        models, trips = _filter_trips(societe, line, day=today, dir=dir)
        anomalies = _rows_with_reasons(models, trips, anomalies_only=True)
        return {
            "date": today,
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


def _trip_sequence(df, societe, line, bus, day, trip_id):
    """Per-stop sequence rows for one trip, with lat/lon for map rendering.

    `coord_suspect` marks stops that are unmatched on (almost) every trip of the
    line — the stop's geocoded coordinates are wrong, so an unmatched flag there
    says nothing about THIS trip.
    """
    seqdf = df[
        (df["societe"] == societe) & (df["line"] == line) & (df["bus"].astype(str) == str(bus)) &
        (df["day"] == day) & (df["trip_id"] == trip_id)
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
async def anomaly_explain(
    societe: str,
    line: str,
    bus: Optional[int] = None,
    day: Optional[str] = None,
    include_data_bugs: bool = False,
    dir: Optional[str] = None
):
    """Per-trip explanations + WHERE (which stops) the anomaly happened."""
    df = model_manager.get_foundation_data()
    try:
        models, trips = _filter_trips(societe, line, bus, day, include_data_bugs=include_data_bugs, dir=dir)
        rows = _rows_with_reasons(models, trips, anomalies_only=False)
        anomalous = [r for r in rows if r["severity"] != "low"]

        # Attach stop-level detail to each anomalous trip
        # Use the bus from each row (needed when bus=None means "all buses")
        if df is not None:
            for a in anomalous:
                seq = _trip_sequence(df, societe, line, int(a["bus"]), a["day"], a["trip_id"])
                a["problem_stops"] = _problem_stops(seq)

        worst_trip, sequence = None, []
        if anomalous and df is not None:
            worst = max(anomalous, key=lambda r: r["anomaly_strength"])
            worst_trip = worst
            sequence = _trip_sequence(df, societe, line, int(worst["bus"]),
                                      worst["day"], worst["trip_id"])

        # Median trip duration for normal trips on this line (baseline for comparison)
        avg_duration_min = None
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
    societe: str, line: str, bus: int, day: str, trip_id: int
):
    """Sequence + problem stops for one specific trip (used for per-trip map/chart).

    Also looks for an "unofficial detour" -- the bus leaving right after the trip
    officially starts, driving off, and coming back to ~the same spot before its long
    dwell (e.g. an errand before really departing) -- and returns the raw GPS path for
    it so the admin can see the actual route driven, not just the straight stop-to-stop
    line. Best-effort: needs a live MongoDB read, so any failure here just omits the
    detour rather than failing the whole request.
    """
    df = model_manager.get_foundation_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    seq = _trip_sequence(df, societe, line, bus, day, trip_id)
    problem_stops = _problem_stops(seq)
    longest_stop = problem_stops.get("longest_stop")
    if longest_stop and longest_stop.get("arrival"):
        try:
            trip_row = df[
                (df["societe"] == societe) & (df["line"] == line) &
                (df["bus"].astype(str) == str(bus)) & (df["day"] == day) &
                (df["trip_id"] == trip_id)
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


@app.get("/api/reference-trip")
async def reference_trip(societe: str, line: str):
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
    df = model_manager.get_foundation_data()
    try:
        models, trips = _filter_trips(societe, line)   # déjà nettoyé des bugs/fragments
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
    idle_by_dir: dict = {}
    for d in ("ALLER", "RETOUR"):
        idle_col = (normal_by_dir[d]["terminus_idle_min"]
                   if "terminus_idle_min" in normal_by_dir[d].columns else pd.Series(dtype=float))
        idle_by_dir[d] = float(idle_col.median()) if len(idle_col) else None

    def _build(row: pd.Series, direction: str) -> dict:
        typical_idle = idle_by_dir[direction]
        seq = _trip_sequence(df, societe, line, row["bus"], row["day"], int(row["trip_id"]))
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
            },
            "sequence": seq,
        }

    directions: dict = {}
    if len(normal_by_dir["ALLER"]) and len(normal_by_dir["RETOUR"]):
        cols = ["day", "bus", "trip_id", "trip_start", "trip_end", "total_elapsed",
               "match_rate", "n_stops", "mean_dwell_s"]
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
async def anomaly_patterns(
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
        foundation = model_manager.get_foundation_data()
        
        if foundation is not None:
            exists = foundation[
                (foundation["day"] == request.day) &
                (foundation["line"] == request.line) &
                (foundation["societe"] == request.societe) &
                (foundation["bus"].astype(str) == str(request.bus))
            ]
            if len(exists) == 0:
                available = foundation[
                    (foundation["line"] == request.line) &
                    (foundation["societe"] == request.societe)
                ]
                available_buses = available["bus"].unique().tolist()
                available_days = available["day"].unique().tolist()
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
            foundation_data = foundation[
                (foundation["day"] == request.day) &
                (foundation["line"] == request.line) &
                (foundation["societe"] == request.societe) &
                (foundation["bus"].astype(str) == str(request.bus))
            ]
            
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
def _build_gps_track(societe: str, line: str, bus: int, day: str):
    """Raw pings -> cleaned -> projected -> Kalman-filtered track.

    Returns (g_filtered, stops_frame, route_len_m) or None if no usable data.
    Same chain used by /api/predict/gps-fallback; kept in one place.
    """
    cfg = fdn.Config()
    g = fdn.load_pings(get_db("Historique_pos"), f"d{day}", line, int(bus))
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
    df = model_manager.get_foundation_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")

    sub = df[(df["societe"] == societe) & (df["line"] == line)][["day", "bus"]].drop_duplicates()
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
    """Buses running on a line at query_time (active / upcoming / completed)."""
    df = model_manager.get_foundation_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    day = day or latest_day_for(societe, line)
    qt = pd.Timestamp(query_time) if query_time else None
    # Re-anchor the wall-clock time-of-day onto the operating day so "active/upcoming"
    # is judged on the same date as the trips (the live demo clock runs on `day`).
    if qt is not None:
        day_ts = pd.Timestamp(datetime.strptime(day, "%Y%m%d").date())
        qt = day_ts + (qt - qt.normalize())

    sub = df[(df["societe"] == societe) & (df["line"] == line) & (df["day"] == day)]
    if len(sub) == 0:
        return {"day": day, "query_time": qt.isoformat() if qt is not None else None, "buses": []}

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
    return {"day": day, "query_time": qt.isoformat() if qt is not None else None,
            "buses": buses}


def _bus_status_at(societe: str, line: str, bus: int, day: str,
                   query_time: pd.Timestamp) -> Optional[Dict]:
    """Where is a bus AT a given wall-clock time on `day`? (live-ETA aware).

    Unlike get_bus_status (which always uses the latest *completed* trip), this finds
    the trip in progress at query_time and the last stop reached by then, so the delay
    model can predict forward to the rider's stop. Delay is baseline-driven.
    """
    df = model_manager.get_foundation_data()
    if df is None:
        return None
    sub = df[(df["societe"] == societe) & (df["line"] == line) &
             (df["bus"].astype(str) == str(bus)) & (df["day"] == day)].copy()
    if len(sub) == 0:
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
    df = model_manager.get_foundation_data()
    if df is not None:
        sub = df[(df["societe"] == societe) & (df["line"] == line) &
                 (df["dir"] == status["direction"])]
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