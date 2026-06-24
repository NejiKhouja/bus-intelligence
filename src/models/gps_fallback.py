"""GPS Fallback module -- train, save, load, serve.

Complete lifecycle for Module 2:
  train()            -> LSTM correction model -> saved to models/fallback/
  load()             -> loads artefacts from disk
  predict_position() -> best position estimate during a GPS gap

ML engineering notes
--------------------
The Kalman filter has NO learnable parameters -- it is an online estimator
that runs at inference time on each bus's live ping stream. No training needed.

LSTM correction
    The LSTM learns to correct the Kalman s-estimate using the PATTERN of
    recent [ks, kv, kp, speed] values. A bus approaching a terminal dwell,
    or climbing a hill at reduced speed, follows a characteristic profile that
    a linear Kalman model cannot capture.

Train data strategy
    We train on GPS pings from MULTIPLE bus-days pulled directly from MongoDB.
    Using a single trip gives a model that overfits that trip's specific
    route geometry and traffic patterns. More trips = better generalisation
    across different days, times, and buses on the same line.

    Concretely: we load all buses for a given line across several calendar days,
    project them onto the route, run the Kalman filter, and pool all non-gap
    windows into a single training set.

Feature normalisation
    Features [ks, kv, kp, speed] are normalised with mean/std fitted on the
    training pings only (no leakage). The same stats are saved and applied at
    inference.

Train/test split
    Not applied here: the LSTM correction is a regression helper for the Kalman
    filter (it corrects estimates using recent history) and is evaluated via the
    synthetic-gap experiment in the notebook, not a held-out labelled set.
    If labelled gap-error pairs were available, a day-based split would apply.

SMOTE / class balancing
    Not applicable -- regression task, no class labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import fallback as _fb
from src.data import foundation as _fdn

SAVE_DIR = Path("models/fallback")

_N_FEATS = len(_fb._LSTM_CORR_FEATS)   # ["ks", "kv", "kp", "speed"]
_HIDDEN  = 32


def _make_corr_lstm(n_feats: int = _N_FEATS, hidden: int = _HIDDEN):
    """Small LSTM that reads recent Kalman history and outputs a corrected s-value."""
    import torch.nn as nn

    class CorrLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feats, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)

    return CorrLSTM()


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(save_dir: str | Path = SAVE_DIR,
          *,
          line: str = "209",
          societe: str = "S.R.T.K",
          n_days: int = 5,
          window: int = 10,
          epochs: int = 30) -> dict:
    """Train LSTM correction on multiple bus-days for better generalisation.

    Pulls raw GPS pings from MongoDB for the most recent `n_days` available
    calendar days for the given line, projects them, runs Kalman, and trains
    the LSTM on the pooled non-gap windows.

    Parameters
    ----------
    n_days  : number of bus-days to collect training data from
    window  : look-back window fed to the LSTM (number of recent pings)
    epochs  : training epochs (more = better but slower; 30 is a safe default)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.data.db import get_db

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    db_winicari = get_db("winicari")
    db_gps      = get_db("Historique_pos")
    cfg         = _fdn.Config()
    usable      = _fdn.build_usable_lines(db_winicari, cfg)
    stops       = usable[(line, societe)]

    # Discover available bus-days in MongoDB (collection names = 'd{YYYYMMDD}')
    all_day_cols = sorted(db_gps.list_collection_names(), reverse=True)
    day_cols = [d for d in all_day_cols if d.startswith("d")][:n_days]
    print(f"  Collecting pings from {len(day_cols)} bus-days: {day_cols}")

    all_feats, all_targets = [], []

    for day in day_cols:
        # Discover bus codes that ran on this line+day (field names match foundation.load_pings)
        sample = db_gps[day].distinct("bus.code",
                                      {"service.codeLigne": line})
        if not sample:
            continue
        for bus_id in sample[:3]:          # cap at 3 buses per day to stay fast
            try:
                raw = _fdn.load_pings(db_gps, day, line, int(bus_id))
                if len(raw) < 50:
                    continue
                g, route_len = _fdn.project_to_route(
                    _fdn.clean_pings(raw, cfg), stops, cfg)
                g_kf = _fb.kalman_filter_track(g, route_len)

                # Non-gap pings only.
                # Target is the RESIDUAL (s_true - ks), not absolute s.
                # WHY: raw s spans 0..192,000 m; predicting absolute values from
                # normalised features causes a ~10^11 m2 loss (model predicts
                # route midpoint). The Kalman estimate ks is already close to s_true;
                # the LSTM only needs to learn the small correction term (+/-500 m).
                non_gap = g_kf[~g_kf["signal_gap"]].reset_index(drop=True)
                feats   = non_gap[_fb._LSTM_CORR_FEATS].values.astype(np.float32)
                targets = (non_gap["s"] - non_gap["ks"]).values.astype(np.float32)
                all_feats.append(feats)
                all_targets.append(targets)
            except Exception:
                continue

    if not all_feats:
        raise RuntimeError("No usable pings found -- check line/societe or MongoDB.")

    feats   = np.concatenate(all_feats,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    print(f"  Total non-gap pings pooled: {len(feats):,}")
    print(f"  Residual (s_true - ks): mean={targets.mean():.1f} m  "
          f"std={targets.std():.1f} m")

    # Feature normalisation -- fit on ALL collected training pings
    mean = feats.mean(axis=0).astype(np.float32)
    std  = (feats.std(axis=0) + 1e-6).astype(np.float32)
    feats_n = (feats - mean) / std

    # Build sliding-window sequences
    xs, ys = [], []
    for i in range(window, len(feats_n)):
        xs.append(feats_n[i - window:i])
        ys.append(targets[i])
    if not xs:
        raise RuntimeError("Not enough pings to build sequences.")

    X = np.stack(xs).astype(np.float32)
    Y = np.array(ys, dtype=np.float32)
    print(f"  Training sequences: {len(X):,}  window={window}")

    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)),
                        batch_size=128, shuffle=True)

    model   = _make_corr_lstm(_N_FEATS, _HIDDEN)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            total += loss_fn(model(xb), yb).item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs}  loss={total/len(X):.2f} m²")

    model.eval()
    torch.save(model.state_dict(), save_dir / "lstm_corr.pt")
    np.savez(save_dir / "lstm_corr_stats.npz", mean=mean, std=std)
    with open(save_dir / "lstm_corr_config.json", "w") as f:
        json.dump({"window": window, "n_feats": _N_FEATS, "hidden": _HIDDEN,
                   "n_days": n_days, "n_pings": int(len(feats))}, f)

    print(f"  -> Fallback artefacts saved to {save_dir}")
    return {"model": model, "mean": mean, "std": std, "window": window}


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Load trained LSTM correction model.

    Returns dict: model, mean, std, window.
    """
    import torch

    save_dir = Path(save_dir)
    with open(save_dir / "lstm_corr_config.json") as f:
        cfg = json.load(f)

    model = _make_corr_lstm(cfg["n_feats"], cfg["hidden"])
    model.load_state_dict(torch.load(save_dir / "lstm_corr.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    stats = np.load(save_dir / "lstm_corr_stats.npz")
    print(f"GPS Fallback LSTM correction loaded  "
          f"(window={cfg['window']}, trained on {cfg.get('n_pings',0):,} pings)")
    return {"model": model, "mean": stats["mean"], "std": stats["std"],
            "window": cfg["window"]}


# ─────────────────────────────────────────────────────────────────────────────
# Serve
# ─────────────────────────────────────────────────────────────────────────────

def run_kalman(g: pd.DataFrame, route_len: float) -> pd.DataFrame:
    """Apply Kalman filter to a projected ping DataFrame.

    Must be called before predict_position to populate ks/kv/kp columns.
    """
    return _fb.kalman_filter_track(g, route_len)


def predict_position(models: dict,
                     g_filtered: pd.DataFrame,
                     t_query: pd.Timestamp,
                     stops: pd.DataFrame) -> dict | None:
    """Best position estimate (Kalman + LSTM correction) during a GPS gap.

    The LSTM predicts a residual correction (s_true - ks). We add it back to
    the Kalman estimate to get the final position: s_final = ks + lstm_correction.

    g_filtered must be the output of run_kalman().
    Returns dict: lat, lon, s_m (km), uncertainty_m, method -- or None if not in a gap.
    """
    import torch
    from src.data.fallback import s_to_latlon, _LSTM_CORR_FEATS

    t_arr = pd.to_datetime(g_filtered["t"])
    before = g_filtered[t_arr <= t_query]
    if len(before) < models["window"]:
        # Not enough history -- fall back to pure Kalman
        return _fb.kalman_fallback(g_filtered, t_query, stops)

    recent = before.iloc[-models["window"]:][_LSTM_CORR_FEATS].values.astype("float32")
    recent_n = (recent - models["mean"]) / models["std"]

    with torch.no_grad():
        correction = float(models["model"](torch.tensor(recent_n[None]))[0])

    # Kalman estimate at t_query (propagated from last filtered state)
    last_row = before.iloc[-1]
    dt = (t_query - pd.Timestamp(last_row["t"])).total_seconds()
    ks_at_query = float(last_row["ks"] + last_row["kv"] * dt)

    s_final = float(np.clip(ks_at_query + correction, 0.0, g_filtered["ks"].max()))
    lat, lon = s_to_latlon(s_final, stops)

    return {
        "lat": lat, "lon": lon, "s_m": round(s_final / 1000, 2),
        "uncertainty_m": round(float(last_row["kp"]), 0),
        "method": "kalman+lstm",
        "ks_m": round(ks_at_query / 1000, 2),
        "correction_m": round(correction, 1),
    }
