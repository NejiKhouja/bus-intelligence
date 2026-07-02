"""FastAPI serving layer for WiniCari AI — the HTTP interface the dashboard talks to.

Loads all four trained modules once at startup (`ModelManager.load_all`) and exposes
them as REST endpoints, grouped as:

  Metadata / fleet state  GET /health, /api/options, /api/lines*, /api/directions,
                          /api/buses*, /api/days*, /api/stops, /api/route-info,
                          /api/active-buses
  Delay / ETA             GET /api/prophet-lines, /api/eta-to-stop
                          POST /api/predict/delay/{auto,manual,forecast}
  GPS fallback            GET /api/bus-status, /api/gps-track, /api/gps-gaps,
                          /api/gps-gap-examples
                          POST /api/predict/gps-fallback
  Anomaly detection       GET /api/anomaly-history, /api/bus-anomaly-check,
                          /api/current-anomalies, /api/anomaly-explain,
                          /api/trip-detail, /api/anomaly-patterns
  RAG chatbot             POST /api/chatbot/ask

See `src/models/*.py` for the underlying train/load/predict logic each endpoint wraps.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import model modules
from src.models import delay, gps_fallback, anomaly, chatbot
from src.data import foundation as fdn
from src.data import reference_db as rdb
from src.data.db import get_db

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

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

class ChatbotRequest(BaseModel):
    query: str
    k: int = 5

class ListResponse(BaseModel):
    companies: List[str]
    lines: List[str]
    directions: List[str]
    buses: List[int]
    days: List[str]

# ─────────────────────────────────────────────────────────────────────────────
# Model Manager
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    _instance = None
    _models = {}
    _foundation_data = None
    _stops_data = {}
    _stop_coords = {}   # f"{societe}_{line}" -> {stop_name: (lat, lon)}
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
                print(f"  ✓ Foundation data loaded ({len(self._foundation_data):,} rows)")
                
                # Build stops mapping for each line
                for societe in self._foundation_data['societe'].unique():
                    for line in self._foundation_data[self._foundation_data['societe'] == societe]['line'].unique():
                        key = f"{societe}_{line}"
                        stops_data = self._foundation_data[
                            (self._foundation_data['societe'] == societe) &
                            (self._foundation_data['line'] == line)
                        ].sort_values('route_seq')[['route_seq', 'stop']].drop_duplicates()
                        self._stops_data[key] = stops_data.to_dict('records')
                
                print(f"  ✓ Stops mapping built for {len(self._stops_data)} lines")

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
                    print(f"  ✓ Stop coordinates loaded for {len(self._stop_coords)} lines")
                except Exception as e:
                    print(f"  ! Stop coordinates not available: {e}")

                # Demo clock: "today" = the most recent day with real data.
                self._latest_day = str(self._foundation_data["day"].max())
                print(f"  ✓ Demo 'today' set to latest data day: {self._latest_day}")
            else:
                print("  ✗ Foundation data not found")
        except Exception as e:
            print(f"  ✗ Failed to load foundation data: {e}")

        for model_name in ["delay", "fallback", "anomaly", "chatbot"]:
            try:
                if model_name == "delay":
                    self._models[model_name] = delay.load()
                elif model_name == "fallback":
                    self._models[model_name] = gps_fallback.load()
                elif model_name == "anomaly":
                    self._models[model_name] = anomaly.load()
                elif model_name == "chatbot":
                    self._models[model_name] = chatbot.load()
                print(f"  ✓ {model_name.capitalize()} models loaded")
            except Exception as e:
                print(f"  ✗ Failed to load {model_name} models: {e}")

        return self._models

    def get(self, model_name: str) -> Dict:
        if model_name not in self._models:
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

    def get_latest_day(self) -> Optional[str]:
        """Demo 'today' — most recent day present in the foundation data."""
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

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Health & Options Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
async def health_check():
    df = model_manager.get_foundation_data()
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded(),
        "models": model_manager.get_loaded_models(),
        "foundation_data": df is not None,
        "rows": len(df) if df is not None else 0,
        "latest_day": model_manager.get_latest_day(),
        "timestamp": datetime.now().isoformat()
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
    counts = sub.groupby("line")["trip_id"].nunique().sort_values(ascending=False)
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
        (df["bus"] == bus) &
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
        (df["bus"] == bus) &
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

# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection — served from precomputed trip scores, with explainability
# ─────────────────────────────────────────────────────────────────────────────

def _filter_trips(societe: str, line: Optional[str] = None,
                  bus: Optional[int] = None, day: Optional[str] = None):
    """Filter the precomputed scored-trips table. Returns (models, trips_df)."""
    models = model_manager.get("anomaly")
    trips = models["trips"]
    mask = trips["societe"] == societe
    if line:
        mask &= trips["line"] == line
    if bus is not None:
        mask &= trips["bus"] == bus
    if day:
        mask &= trips["day"] == day
    return models, trips[mask].copy()


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

    rows = []
    for _, r in ex.iterrows():
        rows.append({
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
            "max_dwell_min": round(float(r["max_dwell_s"]) / 60, 1),
            "max_dark_min": round(float(r.get("max_dark_s", 0) or 0) / 60, 1),
            "worst_dwell_stop": r.get("worst_dwell_stop"),
            "trip_duration_min": round(float(r["total_elapsed"]), 1),
            "total_elapsed_min": round(float(r["total_elapsed"]), 1),  # kept for back-compat
            "n_stops": int(r["n_stops"]),
            "trip_start": r["trip_start"].isoformat() if pd.notna(r["trip_start"]) else None,
            "trip_end": r["trip_end"].isoformat() if pd.notna(r["trip_end"]) else None,
        })
    return rows


@app.get("/api/anomaly-history")
async def get_anomaly_history(
    societe: str,
    line: Optional[str] = None,
    bus: Optional[int] = None,
    limit: int = 30
):
    """Historical anomalies for a company/line/bus, with plain-language reasons."""
    try:
        models, trips = _filter_trips(societe, line, bus)
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
    day: Optional[str] = None
):
    """Check a specific bus for anomalies, optionally on a specific day."""
    try:
        models, trips = _filter_trips(societe, line, bus, day)
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
    line: Optional[str] = None
):
    """Anomalies for the latest day the selected scope actually operated."""
    today = latest_day_for(societe, line)
    try:
        models, trips = _filter_trips(societe, line, day=today)
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
    """Per-stop sequence rows for one trip, with lat/lon for map rendering."""
    seqdf = df[
        (df["societe"] == societe) & (df["line"] == line) & (df["bus"] == bus) &
        (df["day"] == day) & (df["trip_id"] == trip_id)
    ].sort_values("seq")
    coord_map = model_manager.get_stop_coords(societe, line)
    seq = []
    for _, s in seqdf.iterrows():
        dwell = float(s["dwell_s"]) if pd.notna(s.get("dwell_s")) else 0.0
        dark  = float(s["dark_s"])  if pd.notna(s.get("dark_s"))  else 0.0
        dist  = float(s["dist_m"])  if pd.notna(s.get("dist_m"))  else 0.0
        stop_name = s["stop"]
        lat, lon = coord_map.get(stop_name, (None, None))
        seq.append({
            "seq": int(s["seq"]), "stop": stop_name,
            "dwell_min": round(dwell / 60, 1),
            "dark_min":  round(dark  / 60, 1),
            "had_gap":   bool(s.get("had_gap", False)),
            "dist_m": round(dist, 0),
            "matched": bool(s["matched"]),
            "lat": lat,
            "lon": lon,
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
                               "dwell_min": worst_dwell["dwell_min"]}
    # signal-loss stops: stops where a GPS gap interrupted the dwell scan
    gap_stops = [s for s in seq if s.get("had_gap") and s.get("dark_min", 0) >= 5]
    if gap_stops:
        worst_gap = max(gap_stops, key=lambda s: s["dark_min"])
        out["signal_loss_stop"] = {"stop": worst_gap["stop"],
                                   "dark_min": worst_gap["dark_min"]}
        out["signal_loss_count"] = len(gap_stops)
    offroute = [s["stop"] for s in seq if not s["matched"]]
    if offroute:
        out["off_route_stops"] = offroute[:5]
        out["off_route_count"] = len(offroute)
    # Only flag matched stops that were still far — unmatched stops trivially have large dist_m
    # (they were simply never reached), and that info is already in off_route_stops.
    far = [s for s in seq if s["matched"] and s["dist_m"] >= 800]
    if far:
        worst_far = max(far, key=lambda s: s["dist_m"])
        out["farthest_stop"] = {"stop": worst_far["stop"], "dist_m": worst_far["dist_m"]}
    return out


@app.get("/api/anomaly-explain")
async def anomaly_explain(
    societe: str,
    line: str,
    bus: Optional[int] = None,
    day: Optional[str] = None
):
    """Per-trip explanations + WHERE (which stops) the anomaly happened."""
    df = model_manager.get_foundation_data()
    try:
        models, trips = _filter_trips(societe, line, bus, day)
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
    """Sequence + problem stops for one specific trip (used for per-trip map/chart)."""
    df = model_manager.get_foundation_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Foundation data not loaded")
    seq = _trip_sequence(df, societe, line, bus, day, trip_id)
    return {
        "sequence": seq,
        "problem_stops": _problem_stops(seq),
    }


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

# ─────────────────────────────────────────────────────────────────────────────
# Delay Prediction - Auto Mode
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Delay Prediction - Manual Mode
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# GPS Fallback
# ─────────────────────────────────────────────────────────────────────────────

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
                (foundation["bus"] == request.bus)
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
                (foundation["bus"] == request.bus)
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

# ─────────────────────────────────────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=503, detail="Chatbot models not loaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# GPS track / gaps / examples — for the event-driven signal-loss demo
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Live ETA helpers — active buses + ETA to a rider's stop
# ─────────────────────────────────────────────────────────────────────────────

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
             (df["bus"] == bus) & (df["day"] == day)].copy()
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)