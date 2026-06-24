"""Anomaly detection layer — scores trips and stops as normal or anomalous.

Two complementary models
------------------------
Isolation Forest (scikit-learn)
    Trained on per-trip feature vectors. Assigns an anomaly score to each trip
    (-1 = anomalous, 1 = normal). Fast, no labels needed, interpretable features.
    Good for flagging whole trips: "this run was unusual overall."

Autoencoder LSTM (PyTorch)
    Trained on stop-level sequences (dwell, dist_m, matched) to learn what a
    normal trip progression looks like. Reconstruction error = anomaly score.
    Good for pinpointing *where* in a trip something went wrong.

Anomaly signals used
--------------------
- max_dwell_s   : longest stop dwell in the trip (breakdown / incident signal)
- mean_dwell_s  : average stop dwell
- n_stops       : how many stops were matched (low -> GPS or geometry problem)
- match_rate    : fraction of stops with a GPS arrival (low -> bus deviated)
- total_elapsed : total trip time in minutes (far from baseline -> suspect)
- dist_m_max    : worst snap distance across stops (far off route)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRIP_KEYS = ["day", "line", "societe", "bus", "trip_id"]


@dataclass(frozen=True)
class AnomalyConfig:
    if_contamination: float = 0.05   # expected fraction of anomalous trips
    if_n_estimators: int = 200
    lstm_hidden: int = 32
    lstm_epochs: int = 30
    lstm_lr: float = 1e-3
    lstm_batch: int = 64
    seq_pad: int = 30               # pad/truncate sequences to this many stops
    min_trip_stops: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def trip_features(fa: pd.DataFrame, cfg: AnomalyConfig) -> pd.DataFrame:
    """One row per trip: aggregate dwell/match/elapsed into a feature vector."""
    fa = fa.copy()
    fa["elapsed_min"] = (fa["arrival"] - fa["trip_start"]).dt.total_seconds() / 60

    trips = (fa[fa["matched"]].groupby(TRIP_KEYS).agg(
        n_stops=("seq", "count"),
        match_rate=("matched", "mean"),
        max_dwell_s=("dwell_s", "max"),
        mean_dwell_s=("dwell_s", "mean"),
        total_elapsed=("elapsed_min", "max"),
        dist_m_max=("dist_m", "max"),
        dir=("dir", "first"),
        full=("full", "first"),
    ).reset_index())

    trips = trips[trips["n_stops"] >= cfg.min_trip_stops].copy()
    trips["max_dwell_s"] = trips["max_dwell_s"].fillna(0)
    trips["mean_dwell_s"] = trips["mean_dwell_s"].fillna(0)
    trips["dist_m_max"] = trips["dist_m_max"].fillna(0)
    trips["total_elapsed"] = trips["total_elapsed"].fillna(0)
    return trips.reset_index(drop=True)


FEATURES = ["n_stops", "match_rate", "max_dwell_s", "mean_dwell_s",
            "total_elapsed", "dist_m_max"]


def _scale(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """Standard-scale X; return (X_scaled, mean, std)."""
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(trips: pd.DataFrame, cfg: AnomalyConfig):
    """Fit Isolation Forest on trip feature matrix. Returns (model, scaler_mean, scaler_std)."""
    from sklearn.ensemble import IsolationForest
    X = trips[FEATURES].values.astype(float)
    X_s, mean, std = _scale(X)
    model = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        contamination=cfg.if_contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_s)
    return model, mean, std


def score_trips(model, mean: np.ndarray, std: np.ndarray,
                trips: pd.DataFrame) -> pd.DataFrame:
    """Add `if_score` (raw, higher = more normal) and `anomaly` (bool) to trips."""
    X = trips[FEATURES].values.astype(float)
    X_s, _, _ = _scale(X, mean, std)
    trips = trips.copy()
    trips["if_score"] = model.score_samples(X_s)   # negative; more negative = more anomalous
    trips["anomaly"] = model.predict(X_s) == -1
    return trips


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Autoencoder LSTM (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

SEQ_FEATURES = ["dwell_s", "dist_m", "matched"]


def build_sequences(fa: pd.DataFrame, cfg: AnomalyConfig) -> tuple[np.ndarray, list]:
    """Convert per-stop data into fixed-length padded sequences for LSTM.

    Returns (X, trip_ids) where X has shape (n_trips, seq_pad, n_seq_features).
    """
    fa = fa.sort_values(TRIP_KEYS + ["seq"]).copy()
    fa["dwell_s"] = fa["dwell_s"].fillna(0).clip(0, 3600) / 3600   # normalise 0-1h
    fa["dist_m"] = fa["dist_m"].fillna(0).clip(0, 5000) / 5000
    fa["matched"] = fa["matched"].astype(float)

    seqs, ids = [], []
    for keys, grp in fa.groupby(TRIP_KEYS):
        if len(grp) < cfg.min_trip_stops:
            continue
        arr = grp[SEQ_FEATURES].values.astype(np.float32)
        # pad or truncate to seq_pad
        T = cfg.seq_pad
        if len(arr) >= T:
            arr = arr[:T]
        else:
            arr = np.vstack([arr, np.zeros((T - len(arr), arr.shape[1]), dtype=np.float32)])
        seqs.append(arr)
        ids.append(keys)

    return np.stack(seqs), ids


def _make_lstm_autoencoder(seq_len: int, n_feat: int, hidden: int):
    """Build a simple LSTM autoencoder in PyTorch."""
    import torch
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(n_feat, hidden, batch_first=True)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.output  = nn.Linear(hidden, n_feat)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            # repeat the hidden state as decoder input
            dec_in = h.permute(1, 0, 2).expand(-1, seq_len, -1)
            out, _ = self.decoder(dec_in)
            return self.output(out)

    return LSTMAutoencoder()


def train_lstm_autoencoder(X: np.ndarray, cfg: AnomalyConfig):
    """Train LSTM autoencoder; returns (model, per-sample recon errors on training set)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(Xt), batch_size=cfg.lstm_batch, shuffle=True)

    model = _make_lstm_autoencoder(cfg.seq_pad, X.shape[2], cfg.lstm_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for ep in range(cfg.lstm_epochs):
        total = 0
        for (batch,) in loader:
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{cfg.lstm_epochs}  loss={total/len(X):.5f}")

    model.eval()
    with torch.no_grad():
        recon = model(Xt)
        errors = ((recon - Xt) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return model, errors


def lstm_anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """Return per-trip reconstruction errors for a trained LSTM autoencoder."""
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(Xt)
        return ((recon - Xt) ** 2).mean(dim=(1, 2)).cpu().numpy()
