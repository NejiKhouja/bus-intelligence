"""Couche de détection d'anomalies — évalue les trajets et les arrêts comme normaux ou anormaux.

Deux modèles complémentaires
Isolation Forest (scikit-learn)
    Entraîné sur des vecteurs de caractéristiques par trajet. Attribue un score d'anomalie
    à chaque trajet (-1 = anormal, 1 = normal). Rapide, sans labels, caractéristiques
    interprétables. Bon pour signaler des trajets entiers : « cette course était inhabituellement ».

Autoencodeur LSTM (PyTorch)
    Entraîné sur des séquences au niveau des arrêts (dwell, dist_m, matched) pour apprendre
    à quoi ressemble une progression normale d'un trajet. L'erreur de reconstruction = score
    d'anomalie. Bon pour localiser *où* dans un trajet quelque chose s'est mal passé.

Signaux d'anomalie utilisés
- max_dwell_s   : immobilisation maximale à un arrêt dans le trajet (signal de panne / incident)
- mean_dwell_s  : immobilisation moyenne aux arrêts
- n_stops       : nombre d'arrêts correspondants (faible -> problème GPS ou de géométrie)
- match_rate    : fraction des arrêts avec une arrivée GPS (faible -> bus dévié)
- total_elapsed : durée totale du trajet en minutes (loin de la base de référence -> suspect)
- dist_m_max    : pire distance d'accrochage entre tous les arrêts (loin de la route)
- elapsed_vs_bus_z  : durée du trajet vs la moyenne habituelle de CE bus (z-score, bidirectionnel --
                      un service anormalement long OU court par rapport à son propre historique)
- elapsed_vs_line_z : durée du trajet vs la moyenne habituelle de CETTE ligne (même principe,
                      référence = tous les bus de la ligne plutôt qu'un bus spécifique)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRIP_KEYS = ["day", "line", "societe", "bus", "trip_id"]


@dataclass(frozen=True)
class AnomalyConfig:
    if_contamination: str | float = "auto"  # 'auto' = seuil naturel IF à -0.5
    if_n_estimators: int = 200
    lstm_hidden: int = 32
    lstm_epochs: int = 30
    lstm_lr: float = 1e-3
    lstm_batch: int = 64
    seq_pad: int = 30               # rembourrer/tronquer les séquences à ce nombre d'arrêts
    min_trip_stops: int = 3


# Ingénierie des caractéristiques
def trip_features(fa: pd.DataFrame, cfg: AnomalyConfig) -> pd.DataFrame:
    """Une ligne par trajet : agrège immobilisation/correspondance/durée en vecteur de caractéristiques.
    """
    fa = fa.copy()
    fa["elapsed_min"] = (fa["arrival"] - fa["trip_start"]).dt.total_seconds() / 60

    match_rate = fa.groupby(TRIP_KEYS)["matched"].mean().rename("match_rate")

    agg_dict: dict = dict(
        n_stops=("seq", "count"),
        max_dwell_s=("dwell_s", "max"),
        mean_dwell_s=("dwell_s", "mean"),
        total_elapsed=("elapsed_min", "max"),
        dist_m_max=("dist_m", "max"),
        dir=("dir", "first"),
        full=("full", "first"),
        trip_start=("trip_start", "first"),
        trip_end=("trip_end", "first"),
    )
    if "dark_s" in fa.columns:
        agg_dict["max_dark_s"] = ("dark_s", "max")

    matched = fa[fa["matched"]].copy()
    trips = matched.groupby(TRIP_KEYS).agg(**agg_dict).reset_index()
    trips = trips.merge(match_rate, on=TRIP_KEYS, how="left")

    # name of the stop with the worst dwell (for explanation)
    if "dwell_s" in matched.columns and "stop" in matched.columns:
        worst_stop = (matched.sort_values("dwell_s", ascending=False)
                      .groupby(TRIP_KEYS)["stop"].first()
                      .reset_index()
                      .rename(columns={"stop": "worst_dwell_stop"}))
        trips = trips.merge(worst_stop, on=TRIP_KEYS, how="left")

    trips = trips[trips["n_stops"] >= cfg.min_trip_stops].copy()
    trips["max_dwell_s"] = trips["max_dwell_s"].fillna(0)
    trips["mean_dwell_s"] = trips["mean_dwell_s"].fillna(0)
    trips["dist_m_max"] = trips["dist_m_max"].fillna(0)
    trips["total_elapsed"] = trips["total_elapsed"].fillna(0)
    if "max_dark_s" not in trips.columns:
        trips["max_dark_s"] = 0.0
    trips["max_dark_s"] = trips["max_dark_s"].fillna(0)

    for group_cols, col in [(["societe", "bus"], "elapsed_vs_bus_z"),
                            (["societe", "line"], "elapsed_vs_line_z")]:
        g = trips.groupby(group_cols)["total_elapsed"]
        mean = g.transform("mean")
        std = g.transform("std").fillna(0)
        trips[col] = np.where(std > 1e-6, (trips["total_elapsed"] - mean) / std, 0.0)

    return trips.reset_index(drop=True)


FEATURES = ["n_stops", "match_rate", "max_dwell_s", "mean_dwell_s",
            "total_elapsed", "dist_m_max", "max_dark_s",
            "elapsed_vs_bus_z", "elapsed_vs_line_z"]


def _scale(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """Normalise X (z-score) ; retourne (X_normalisé, mean, std)."""
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


# Modèle 1 — Isolation Forest
def train_isolation_forest(trips: pd.DataFrame, cfg: AnomalyConfig):
    """Entraîne l'Isolation Forest sur la matrice de caractéristiques des trajets. Retourne (modèle, scaler_mean, scaler_std)."""
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
    """Ajoute `if_score` (brut, plus élevé = plus normal) et `anomaly` (bool) aux trajets."""
    X = trips[FEATURES].values.astype(float)
    X_s, _, _ = _scale(X, mean, std)
    trips = trips.copy()
    trips["if_score"] = model.score_samples(X_s)   # négatif ; plus négatif = plus anormal
    trips["anomaly"] = model.predict(X_s) == -1
    return trips


# Modèle 2 — Autoencodeur LSTM (PyTorch)
SEQ_FEATURES = ["dwell_s", "dist_m", "matched"]


def build_sequences(fa: pd.DataFrame, cfg: AnomalyConfig) -> tuple[np.ndarray, list]:
    """Convertit les données par arrêt en séquences rembourrées de longueur fixe pour le LSTM.

    Retourne (X, trip_ids) où X a la forme (n_trajets, seq_pad, n_seq_features).
    """
    fa = fa.sort_values(TRIP_KEYS + ["seq"]).copy()
    fa["dwell_s"] = fa["dwell_s"].fillna(0).clip(0, 3600) / 3600   # normaliser 0-1h
    fa["dist_m"] = fa["dist_m"].fillna(0).clip(0, 5000) / 5000
    fa["matched"] = fa["matched"].astype(float)

    seqs, ids = [], []
    for keys, grp in fa.groupby(TRIP_KEYS):
        if len(grp) < cfg.min_trip_stops:
            continue
        arr = grp[SEQ_FEATURES].values.astype(np.float32)
        # rembourrer ou tronquer à seq_pad
        T = cfg.seq_pad
        if len(arr) >= T:
            arr = arr[:T]
        else:
            arr = np.vstack([arr, np.zeros((T - len(arr), arr.shape[1]), dtype=np.float32)])
        seqs.append(arr)
        ids.append(keys)

    return np.stack(seqs), ids


def _make_lstm_autoencoder(seq_len: int, n_feat: int, hidden: int):
    """Construit un autoencodeur LSTM simple en PyTorch."""
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
            # répéter l'état caché comme entrée du décodeur
            dec_in = h.permute(1, 0, 2).expand(-1, seq_len, -1)
            out, _ = self.decoder(dec_in)
            return self.output(out)

    return LSTMAutoencoder()


def train_lstm_autoencoder(X: np.ndarray, cfg: AnomalyConfig):
    """Entraîne l'autoencodeur LSTM ; retourne (modèle, erreurs de reconstruction par échantillon sur l'ensemble d'entraînement)."""
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
            print(f"  époque {ep+1}/{cfg.lstm_epochs}  perte={total/len(X):.5f}")

    model.eval()
    with torch.no_grad():
        recon = model(Xt)
        errors = ((recon - Xt) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return model, errors


def lstm_anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """Retourne les erreurs de reconstruction par trajet pour un autoencodeur LSTM entraîné."""
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(Xt)
        return ((recon - Xt) ** 2).mean(dim=(1, 2)).cpu().numpy()
