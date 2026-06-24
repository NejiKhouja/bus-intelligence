"""Delay module -- train, save, load, serve.

Complete lifecycle for Module 1:
  train()       -> HistGBM + LSTM + Prophet -> artefacts saved to models/delay/
  load()        -> loads all artefacts from disk
  predict_eta() -> rolling ETA table for a live bus
  forecast()    -> 30-day delay forecast for a line (Prophet)

ML engineering notes
--------------------
Train/test split
    Split is by DAY (chronological), not random. Using a random split would
    leak future patterns into training -- a bus-day in March would appear in both
    train and test. We use 80% of days for training, 20% for testing.

Validation set (LSTM only)
    Within the training portion we hold out the last 10% of sequences as a
    temporal validation set for early stopping. This is the LAST 10%, not a
    random sample, for the same leakage reason.

Feature normalisation (LSTM only)
    Raw features span wildly different scales: delay_min (-120..+120),
    elapsed_min (0..600), dep_hour (0..23). Without normalisation the LSTM
    gradient is dominated by the high-variance feature (elapsed_min) and the
    others are undertrained. We fit a StandardScaler on X_train and apply the
    SAME stats to val, test, and inference.
    Scaler stats are saved to disk alongside the model weights.

HistGBM vs LSTM
    HistGBM doesn't need normalisation (tree splits are scale-invariant).
    It often matches or beats LSTM on tabular data at moderate dataset sizes
    (~100k samples). LSTM adds value when the full trip HISTORY matters more
    than the current-stop state alone.

SMOTE / class balancing
    Not applicable. Both models solve a REGRESSION task (predict minutes of
    delay, a continuous value). SMOTE is a technique for imbalanced
    CLASSIFICATION -- it generates synthetic minority-class samples to rebalance
    label counts. There are no class labels here.

Prophet
    One model per (societe, line, dir) combination fitted on daily mean delay.
    Prophet handles weekly seasonality automatically and produces calibrated
    uncertainty intervals -- useful for timetable planning.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import delay as _dl

SAVE_DIR = Path("models/delay")


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(foundation_path: str | Path,
          save_dir: str | Path = SAVE_DIR,
          *,
          epochs: int = 30,
          hidden: int = 64,
          n_layers: int = 2,
          patience: int = 5) -> dict:
    """Train HistGBM + LSTM + Prophet on the full foundation dataset.

    Saves all artefacts to save_dir. Returns dict with trained models + metrics.
    """
    import joblib
    import torch
    from sklearn.metrics import mean_absolute_error

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "prophet").mkdir(exist_ok=True)

    # ── feature engineering ──────────────────────────────────────────────────
    print("  Loading foundation...")
    cfg = _dl.DelayConfig()
    df = _dl.load_foundation(foundation_path)
    m = _dl.add_daytype(_dl.with_elapsed(df, cfg))
    baseline = _dl.build_baseline(m, cfg)
    d = _dl.add_daytype(_dl.with_delay(m, baseline, cfg))
    roll = _dl.rolling_table(d)

    # Chronological 80/20 split -- no random shuffling to avoid leakage
    days = np.sort(roll["day"].unique())
    cut_day = days[int(0.8 * len(days))]
    tr = roll[roll["day"] < cut_day]
    te = roll[roll["day"] >= cut_day]
    print(f"  Split: train={len(tr):,} rows (days<{cut_day})  "
          f"test={len(te):,} rows (days>={cut_day})")

    # ── HistGBM ──────────────────────────────────────────────────────────────
    # Tree models are scale-invariant -- no normalisation needed.
    # Categoricals (line, dir) handled natively by HistGBM.
    print("  Training HistGBM...")
    hgbm = _dl.train_rolling_model(tr)
    hgbm_mae = mean_absolute_error(te[_dl.TARGET], hgbm.predict(_dl._design(te)))
    print(f"    test MAE: {hgbm_mae:.2f} min")
    joblib.dump(hgbm, save_dir / "hgbm.joblib")

    baseline.to_parquet(save_dir / "baseline.parquet", index=False)

    # ── LSTM ─────────────────────────────────────────────────────────────────
    print(f"  Training LSTM ({epochs} epochs, patience={patience})...")
    X, _, y = _dl.build_lstm_sequences(roll)
    day_arr = roll["day"].values
    X_tr_raw, y_tr = X[day_arr < cut_day], y[day_arr < cut_day]
    X_te_raw, y_te = X[day_arr >= cut_day], y[day_arr >= cut_day]

    # Fit scaler on training data ONLY, then apply to all splits
    feat_mean, feat_std = _dl.fit_lstm_scaler(X_tr_raw)
    X_tr = _dl.scale_sequences(X_tr_raw, feat_mean, feat_std)
    X_te = _dl.scale_sequences(X_te_raw, feat_mean, feat_std)

    lstm = _dl.train_lstm_delay(X_tr, y_tr, hidden=hidden, n_layers=n_layers,
                                epochs=epochs, lr=1e-3, batch=256, patience=patience)
    lstm_mae = mean_absolute_error(y_te, _dl.predict_lstm(lstm, X_te))
    print(f"    test MAE: {lstm_mae:.2f} min")

    torch.save(lstm.state_dict(), save_dir / "lstm_delay.pt")
    # Save scaler stats alongside weights -- inference MUST use the same transform
    np.savez(save_dir / "lstm_scaler.npz", mean=feat_mean, std=feat_std)
    with open(save_dir / "lstm_config.json", "w") as f:
        json.dump({"hidden": hidden, "n_layers": n_layers,
                   "n_feats": X.shape[2], "max_len": 30}, f)

    # ── Prophet (one model per line/dir) ─────────────────────────────────────
    # Trained on the FULL dataset (all dates) -- Prophet is a time-series
    # forecaster, not a supervised model; it captures seasonality from history.
    print("  Fitting Prophet models...")
    combos = d[["societe", "line", "dir"]].drop_duplicates()
    prophet_count = 0
    for _, row in combos.iterrows():
        pm = _dl.fit_prophet(d, line=row["line"], direction=row["dir"],
                             societe=row["societe"])
        if pm is not None:
            fname = (save_dir / "prophet"
                     / f"{row['societe']}_{row['line']}_{row['dir']}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(pm, f)
            prophet_count += 1
    print(f"    {prophet_count} Prophet models saved")

    print(f"  -> Delay artefacts saved to {save_dir}")
    return {
        "hgbm": hgbm, "lstm": lstm, "baseline": baseline,
        "feat_mean": feat_mean, "feat_std": feat_std,
        "hgbm_mae": hgbm_mae, "lstm_mae": lstm_mae,
        "prophet_count": prophet_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Load all trained delay artefacts from save_dir.

    Returns dict: hgbm, lstm, feat_mean, feat_std, baseline,
                  prophet (keyed by societe_line_dir).
    """
    import joblib
    import torch

    save_dir = Path(save_dir)
    hgbm     = joblib.load(save_dir / "hgbm.joblib")
    baseline = pd.read_parquet(save_dir / "baseline.parquet")

    scaler   = np.load(save_dir / "lstm_scaler.npz")
    feat_mean, feat_std = scaler["mean"], scaler["std"]

    with open(save_dir / "lstm_config.json") as f:
        cfg = json.load(f)
    lstm = _dl._make_delay_lstm(cfg["n_feats"], cfg["hidden"], cfg["n_layers"])
    lstm.load_state_dict(torch.load(save_dir / "lstm_delay.pt", map_location="cpu",
                                    weights_only=True))
    lstm.eval()

    prophets: dict = {}
    prophet_dir = save_dir / "prophet"
    if prophet_dir.exists():
        for p in prophet_dir.glob("*.pkl"):
            with open(p, "rb") as f:
                prophets[p.stem] = pickle.load(f)

    print(f"Delay models loaded: HistGBM + LSTM + {len(prophets)} Prophet models")
    return {"hgbm": hgbm, "lstm": lstm, "baseline": baseline,
            "feat_mean": feat_mean, "feat_std": feat_std, "prophet": prophets}


# ─────────────────────────────────────────────────────────────────────────────
# Serve
# ─────────────────────────────────────────────────────────────────────────────

def predict_eta(models: dict, *,
                societe: str, line: str, direction: str,
                dep_time: str,
                current_seq: int,
                current_delay_min: float,
                model_type: str = "hgbm") -> pd.DataFrame:
    """ETA table for all remaining stops given a bus's live state.

    model_type: 'hgbm' (default -- faster, same accuracy at current training size)
                'lstm' (uses full trip history; better with more epochs/GPU)
    Returns DataFrame: seq, expected_min, pred_delay_min, eta.
    """
    if model_type == "lstm":
        return _dl.serve_eta_lstm(
            models["lstm"], models["baseline"],
            societe=societe, line=line, direction=direction,
            dep_time=dep_time, current_seq=current_seq,
            current_delay_min=current_delay_min,
            scaler_mean=models["feat_mean"], scaler_std=models["feat_std"],
        )
    return _dl.serve_eta(
        models["hgbm"], models["baseline"],
        societe=societe, line=line, direction=direction,
        dep_time=dep_time, current_seq=current_seq,
        current_delay_min=current_delay_min,
    )


def forecast(models: dict, *,
             societe: str, line: str, direction: str,
             periods: int = 30) -> pd.DataFrame | None:
    """30-day daily mean delay forecast for one line. Returns None if no model."""
    key = f"{societe}_{line}_{direction}"
    pm  = models["prophet"].get(key)
    if pm is None:
        return None
    return _dl.prophet_forecast(pm, periods=periods)
