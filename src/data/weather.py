"""Signal météo journalier pour le modèle de retard — dérivé de `OpenData.historiqueJourMeteo`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = ROOT / "data" / "processed" / "day_weather.parquet"


def build_day_weather_cache(od_db, out_path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Agrège `OpenData.historiqueJourMeteo` (condition météo par station par jour) en UNE
    ligne par jour calendaire : `rain_frac` = fraction des stations ayant rapporté une donnée
    ce jour-là qui ont signalé de la pluie (0.0 à 1.0).
    """
    rows = []
    for doc in od_db["historiqueJourMeteo"].find({}, {"historiqueJours": 1}):
        for d in doc.get("historiqueJours", []):
            rows.append({"date": d.get("date"), "conditionMeteo": d.get("conditionMeteo", "")})

    raw = pd.DataFrame(rows)
    raw["day"] = pd.to_datetime(raw["date"], format="%Y/%m/%d", errors="coerce").dt.strftime("%Y%m%d")
    raw = raw.dropna(subset=["day"])
    raw["is_rain_flag"] = raw["conditionMeteo"].str.contains("Pluie", na=False)

    day_weather = raw.groupby("day")["is_rain_flag"].mean().reset_index()
    day_weather = day_weather.rename(columns={"is_rain_flag": "rain_frac"})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    day_weather.to_parquet(out_path, index=False)
    return day_weather


def load_day_weather(path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame | None:
    """Charge le cache météo journalier, ou None si `build_day_weather_cache` n'a pas encore été lancé."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)
