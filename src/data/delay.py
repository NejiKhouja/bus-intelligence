"""Delay layer — built on top of the reconstructed arrival foundation.

There are NO official per-stop timetables in the data (`ligne.horaires` only stores origin
departure times). So "delay" here is measured against a DATA-DRIVEN baseline: the typical
(median) time each line takes to reach each stop, learned from all reconstructed trips.

    delay = actual elapsed-to-stop  -  expected elapsed-to-stop (baseline)

Interpretation: this is delay *relative to how the line normally performs* (i.e. "this run
is slower/faster than usual / disrupted"), NOT lateness vs a published schedule. If the
company later provides real per-stop timetables, swap `build_baseline` for that schedule and
everything downstream is unchanged.

Pipeline
--------
1. `with_elapsed`   - keep matched arrivals, compute minutes since the trip started, drop
                      physically impossible values.
2. `build_baseline` - expected elapsed-to-stop = median over trips, per
                      (societe, line, dir, seq); keep cells with >= `min_obs` trips.
3. `with_delay`     - delay = elapsed - expected; clip extreme artifacts.
4. `trip_features`  - per-trip table for PREDICTION: state known at `cut_frac` of the route
                      (delay so far, hour, line, ...) -> target = delay at the final stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRIP_KEYS = ["day", "line", "societe", "bus", "trip_id"]


@dataclass(frozen=True)
class DelayConfig:
    min_obs: int = 20            # min trips per (societe,line,dir,seq) to trust a baseline
    max_elapsed_h: float = 24.0  # drop arrivals whose elapsed exceeds this (broken trips)
    max_abs_delay_min: float = 120.0  # clip |delay| beyond this (reconstruction artifacts)
    cut_frac: float = 0.40       # route fraction "known so far" when predicting final delay
    min_trip_stops: int = 4      # trips need at least this many matched stops to be usable


def load_foundation(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trip_start"] = pd.to_datetime(df["trip_start"])
    df["trip_end"] = pd.to_datetime(df["trip_end"])
    df["arrival"] = pd.to_datetime(df["arrival"])
    return df


def with_elapsed(df: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Matched arrivals only, with `elapsed_min` = minutes from trip start to the stop."""
    m = df[df["matched"]].copy()
    m["elapsed_min"] = (m["arrival"] - m["trip_start"]).dt.total_seconds() / 60
    m = m[(m["elapsed_min"] >= 0) & (m["elapsed_min"] < cfg.max_elapsed_h * 60)]
    m["dep_hour"] = m["trip_start"].dt.hour
    m["dow"] = m["trip_start"].dt.dayofweek
    return m.reset_index(drop=True)


def build_baseline(m: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Expected elapsed-to-stop per (societe, line, dir, seq) — the data-driven 'schedule'."""
    g = m.groupby(["societe", "line", "dir", "seq"])["elapsed_min"]
    base = g.agg(expected_min="median", p10=lambda s: s.quantile(0.10),
                 p90=lambda s: s.quantile(0.90), n="count").reset_index()
    return base[base["n"] >= cfg.min_obs].reset_index(drop=True)


def with_delay(m: pd.DataFrame, baseline: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Attach expected time and `delay_min = elapsed - expected` (clipped)."""
    out = m.merge(baseline[["societe", "line", "dir", "seq", "expected_min"]],
                  on=["societe", "line", "dir", "seq"], how="inner")
    out["delay_min"] = (out["elapsed_min"] - out["expected_min"]).clip(
        -cfg.max_abs_delay_min, cfg.max_abs_delay_min)
    return out


def trip_features(d: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """One row per trip: the delay state known at `cut_frac` of the route (the moment we
    'predict from') and the target = delay at the final reached stop.

    This is the table a delay-prediction model trains on: predict how late the bus will be
    at the end of its run, given how it is doing partway through.
    """
    rows = []
    for keys, t in d.sort_values("seq").groupby(TRIP_KEYS):
        if len(t) < cfg.min_trip_stops:
            continue
        smax = t["seq"].max()
        early = t[t["seq"] <= cfg.cut_frac * smax]
        if len(early) == 0:
            continue
        cur, fin = early.iloc[-1], t.iloc[-1]
        rows.append({
            **dict(zip(TRIP_KEYS, keys)),
            "dir": t["dir"].iloc[0],
            "dep_hour": int(cur["dep_hour"]),
            "dow": int(cur["dow"]),
            "cur_seq": int(cur["seq"]),
            "cur_seq_frac": float(cur["seq"] / smax) if smax else 0.0,
            "cur_delay_min": float(cur["delay_min"]),     # delay so far  (key predictor)
            "cur_elapsed_min": float(cur["elapsed_min"]),
            "final_delay_min": float(fin["delay_min"]),   # TARGET
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- day-type + rolling model
# Numeric + categorical features used by the rolling next-stop model. `line`/`dir` are passed
# natively as categoricals (HistGradientBoosting handles them), so training and serving use the
# exact same columns with no manual one-hot bookkeeping.
FEATURES_NUM = ["dep_hour", "dow", "is_weekend", "seq", "seq_frac", "delay_min", "elapsed_min"]
FEATURES_CAT = ["line", "dir"]
TARGET = "next_delay_min"


def add_daytype(m: pd.DataFrame) -> pd.DataFrame:
    """Add cheap calendar features. (Weather would need an external source — not in the DB.)
    Tunisia's weekend is Saturday/Sunday -> dayofweek 5/6."""
    m = m.copy()
    m["is_weekend"] = m["dow"].isin([5, 6]).astype(int)
    m["month"] = m["trip_start"].dt.month
    return m


def rolling_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (trip, stop k): current state + target = delay at the NEXT stop k+1.

    This is the table the rolling model trains on — predict the delay one stop ahead as the
    bus progresses, which chains into a full ETA for the rest of the run.
    """
    d = d.sort_values(TRIP_KEYS + ["seq"]).copy()
    g = d.groupby(TRIP_KEYS)
    d["seq_frac"] = g["seq"].transform(lambda s: s / s.max() if s.max() else 0.0)
    d[TARGET] = g["delay_min"].shift(-1)
    d["next_seq"] = g["seq"].shift(-1)
    return d.dropna(subset=[TARGET]).reset_index(drop=True)


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[FEATURES_NUM + FEATURES_CAT].copy()
    for c in FEATURES_CAT:
        X[c] = X[c].astype("category")
    return X


def train_rolling_model(roll: pd.DataFrame, **kw):
    """Fit the next-stop delay model. `line`/`dir` handled as native categoricals."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        categorical_features=FEATURES_CAT,
        max_iter=kw.get("max_iter", 300),
        learning_rate=kw.get("learning_rate", 0.05),
        max_depth=kw.get("max_depth", 6),
        random_state=0,
    )
    model.fit(_design(roll), roll[TARGET])
    return model


def serve_eta(model, baseline: pd.DataFrame, *, societe, line, direction, dep_time,
              current_seq: int, current_delay_min: float) -> pd.DataFrame:
    """PRODUCTION ETA: given a bus's live state (where it is + how late it is now), roll the
    next-stop model forward to predict delay — and a clock ETA — at every remaining stop.

    Returns one row per downstream stop: seq, expected_min (baseline), pred_delay_min, eta.
    """
    b = baseline[(baseline["societe"] == societe) & (baseline["line"] == line)
                 & (baseline["dir"] == direction)].sort_values("seq")
    if b.empty:
        return pd.DataFrame(columns=["seq", "expected_min", "pred_delay_min", "eta"])
    dep_time = pd.Timestamp(dep_time)
    dep_hour, dow = dep_time.hour, dep_time.dayofweek
    is_weekend = int(dow in (5, 6))
    exp = dict(zip(b["seq"].astype(int), b["expected_min"]))
    smax = int(b["seq"].max())

    cur_seq, cur_delay, rows = int(current_seq), float(current_delay_min), []
    while cur_seq < smax:
        nxt = cur_seq + 1
        if nxt not in exp:
            cur_seq = nxt
            continue
        x = _design(pd.DataFrame([{
            "dep_hour": dep_hour, "dow": dow, "is_weekend": is_weekend,
            "seq": cur_seq, "seq_frac": cur_seq / smax,
            "delay_min": cur_delay, "elapsed_min": exp.get(cur_seq, 0.0) + cur_delay,
            "line": line, "dir": direction,
        }]))
        nd = float(model.predict(x)[0])
        rows.append({"seq": nxt, "expected_min": round(exp[nxt], 1),
                     "pred_delay_min": round(nd, 1),
                     "eta": dep_time + pd.Timedelta(minutes=exp[nxt] + nd)})
        cur_seq, cur_delay = nxt, nd
    return pd.DataFrame(rows)
