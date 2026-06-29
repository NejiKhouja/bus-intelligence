"""Module de détection d'anomalies — entraîner, sauvegarder, charger, scorer.

Cycle de vie complet pour le Module 3 :
  train()  -> Isolation Forest + Autoencodeur LSTM -> sauvegardé dans models/anomaly/
  load()   -> charge les artefacts depuis le disque
  score()  -> signale les trajets anormaux dans de nouvelles données avec les deux modèles
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import anomaly as _an

SAVE_DIR = Path("models/anomaly")


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(foundation_path: str | Path,
          save_dir: str | Path = SAVE_DIR) -> dict:
    """Entraîne l'Isolation Forest + l'Autoencodeur LSTM sur la fondation complète.

    Sauvegarde :
      isolation_forest.joblib   -- modèle IF
      if_scaler.npz             -- moyenne/écart-type des caractéristiques pour la normalisation
      lstm_ae.pt                -- dictionnaire d'état de l'Autoencodeur LSTM
      lstm_ae_config.json       -- paramètres d'architecture
      lstm_ae_threshold.npy     -- erreur de reconstruction au 95e percentile (ensemble d'entraînement)
      trips_scored.parquet      -- tous les trajets avec scores/flags d'anomalie IF
    """
    import joblib
    import torch

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = _an.AnomalyConfig()

    print("  Chargement de la fondation...")
    fa = pd.read_parquet(foundation_path)
    fa["arrival"] = pd.to_datetime(fa["arrival"])
    fa["trip_start"] = pd.to_datetime(fa["trip_start"])
    if "departure" in fa.columns:
        fa["departure"] = pd.to_datetime(fa["departure"])
    fa["dwell_s"] = fa.get("dwell_s", pd.Series(0.0, index=fa.index)).fillna(0)

    # ── Isolation Forest ─────────────────────────────────────────────────────
    print("  Entraînement de l'Isolation Forest...")
    trips = _an.trip_features(fa, cfg)
    print(f"    trajets : {len(trips):,}")

    if_model, if_mean, if_std = _an.train_isolation_forest(trips, cfg)
    trips_scored = _an.score_trips(if_model, if_mean, if_std, trips)
    n_if = int(trips_scored["anomaly"].sum())
    print(f"    signalés : {n_if}/{len(trips)} ({100*n_if/len(trips):.1f}%)")

    joblib.dump(if_model, save_dir / "isolation_forest.joblib")
    np.savez(save_dir / "if_scaler.npz", mean=if_mean, std=if_std)

    # ── Autoencodeur LSTM ────────────────────────────────────────────────────
    print("  Entraînement de l'Autoencodeur LSTM...")
    X, _ = _an.build_sequences(fa, cfg)
    print(f"    séquences : {X.shape}")

    lstm_ae, train_errors = _an.train_lstm_autoencoder(X, cfg)
    threshold = float(np.percentile(train_errors, 95))
    lstm_scores = _an.lstm_anomaly_scores(lstm_ae, X)
    n_lstm = int((lstm_scores > threshold).sum())
    print(f"    signalés : {n_lstm}/{len(X)} (seuil={threshold:.5f})")

    # Attacher les scores LSTM aux trips_scored (alignés par position ; trajets IF et LSTM dans le même ordre)
    n_pad = max(0, len(trips_scored) - len(lstm_scores))
    trips_scored["lstm_score"]   = np.concatenate([lstm_scores, np.zeros(n_pad)])[:len(trips_scored)]
    trips_scored["lstm_anomaly"] = trips_scored["lstm_score"] > threshold
    trips_scored["dual_anomaly"] = trips_scored["anomaly"] & trips_scored["lstm_anomaly"]
    trips_scored.to_parquet(save_dir / "trips_scored.parquet", index=False)

    torch.save(lstm_ae.state_dict(), save_dir / "lstm_ae.pt")
    np.save(save_dir / "lstm_ae_threshold.npy", np.array(threshold))
    with open(save_dir / "lstm_ae_config.json", "w") as f:
        json.dump({"hidden": cfg.lstm_hidden, "seq_pad": cfg.seq_pad,
                   "n_feats": X.shape[2]}, f)

    print(f"  -> Artefacts d'anomalie sauvegardés dans {save_dir}")
    return {
        "if_model": if_model, "lstm_ae": lstm_ae, "trips": trips_scored,
        "n_if": n_if, "n_lstm": n_lstm, "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Charge les modèles d'anomalie entraînés depuis save_dir.

    Retourne dict : if_model, if_mean, if_std, lstm_ae, threshold, trips.
    """
    import joblib
    import torch

    save_dir = Path(save_dir)
    if_model = joblib.load(save_dir / "isolation_forest.joblib")
    scaler = np.load(save_dir / "if_scaler.npz")

    with open(save_dir / "lstm_ae_config.json") as f:
        ae_cfg = json.load(f)

    lstm_ae = _an._make_lstm_autoencoder(
        ae_cfg["seq_pad"], ae_cfg["n_feats"], ae_cfg["hidden"]
    )
    lstm_ae.load_state_dict(torch.load(save_dir / "lstm_ae.pt", map_location="cpu",
                                       weights_only=True))
    lstm_ae.eval()

    threshold = float(np.load(save_dir / "lstm_ae_threshold.npy"))
    trips = pd.read_parquet(save_dir / "trips_scored.parquet")

    print(f"Modèles d'anomalie chargés (IF + LSTM AE, seuil={threshold:.5f})")
    return {
        "if_model": if_model, "if_mean": scaler["mean"], "if_std": scaler["std"],
        "lstm_ae": lstm_ae, "threshold": threshold, "trips": trips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

def score(models: dict, fa: pd.DataFrame) -> pd.DataFrame:
    """Score les trajets dans de nouvelles données avec les deux modèles.

    Retourne le DataFrame de trajets avec les colonnes :
      anomaly        -- flag IF
      if_score       -- score IF brut (plus négatif = plus anormal)
      lstm_score     -- erreur de reconstruction LSTM
      lstm_anomaly   -- flag LSTM (score > seuil)
      dual_anomaly   -- signalé par les deux modèles
    """
    cfg = _an.AnomalyConfig()
    fa = fa.copy()
    fa["dwell_s"] = fa.get("dwell_s", pd.Series(0.0, index=fa.index)).fillna(0)

    trips = _an.trip_features(fa, cfg)
    trips = _an.score_trips(models["if_model"], models["if_mean"],
                            models["if_std"], trips)

    X, _ = _an.build_sequences(fa, cfg)
    if len(X) > 0:
        lstm_scores = _an.lstm_anomaly_scores(models["lstm_ae"], X)
        # aligner : les séquences peuvent couvrir moins de trajets que trip_features (filtre min_trip_stops)
        trips["lstm_score"] = np.pad(lstm_scores, (0, max(0, len(trips) - len(lstm_scores))))[:len(trips)]
        trips["lstm_anomaly"] = trips["lstm_score"] > models["threshold"]
    else:
        trips["lstm_score"] = 0.0
        trips["lstm_anomaly"] = False

    trips["dual_anomaly"] = trips["anomaly"] & trips["lstm_anomaly"]
    return trips
