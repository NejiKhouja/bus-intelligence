"""Module de détection d'anomalies — entraîner, sauvegarder, charger, scorer.

Cycle de vie complet pour le Module 3 :
  train()  -> Isolation Forest par opérateur + Autoencodeur LSTM global -> sauvegardé dans models/anomaly/
  load()   -> charge les artefacts depuis le disque
  score()  -> signale les trajets anormaux dans de nouvelles données avec les deux modèles

Chaque opérateur obtient son propre Isolation Forest afin que la durée normale pour
TCV (trajets urbains courts) ne soit pas utilisée comme référence pour S.R.T.K
(lignes interurbaines longues), et vice versa.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import anomaly as _an

SAVE_DIR = Path("models/anomaly")


def _safe(name: str) -> str:
    """Nom de fichier sûr pour un nom d'opérateur."""
    return "".join(c if c.isalnum() else "_" for c in str(name))


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(foundation_path: str | Path,
          save_dir: str | Path = SAVE_DIR) -> dict:
    """Entraîne un Isolation Forest par opérateur + un Autoencodeur LSTM global.

    Sauvegarde :
      {safe_societe}_isolation_forest.joblib  -- IF par opérateur
      {safe_societe}_if_scaler.npz            -- scaler par opérateur
      company_models.json                      -- index nom_sûr -> nom_original
      isolation_forest.joblib                  -- IF global (repli)
      if_scaler.npz                            -- scaler global (repli)
      lstm_ae.pt                               -- Autoencodeur LSTM global
      lstm_ae_config.json                      -- config architecture
      lstm_ae_threshold.npy                    -- seuil 95e percentile
      trips_scored.parquet                     -- tous les trajets scorés
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

    # ── Isolation Forest par opérateur ───────────────────────────────────────
    print("  Entraînement des Isolation Forests (un par opérateur)...")
    trips = _an.trip_features(fa, cfg)
    print(f"    trajets total : {len(trips):,}")

    company_index: dict[str, str] = {}   # safe_name -> original_name
    all_scored: list[pd.DataFrame] = []
    min_trips_per_company = 30

    for soc in sorted(trips["societe"].unique()):
        soc_trips = trips[trips["societe"] == soc].copy()
        if len(soc_trips) < min_trips_per_company:
            print(f"    {soc}: {len(soc_trips)} trajets — trop peu, repli sur modèle global")
            continue
        m, mean, std = _an.train_isolation_forest(soc_trips, cfg)
        scored = _an.score_trips(m, mean, std, soc_trips)
        n = int(scored["anomaly"].sum())
        print(f"    {soc}: {n}/{len(soc_trips)} signalés ({100*n/len(soc_trips):.1f}%)")
        safe = _safe(soc)
        joblib.dump(m, save_dir / f"{safe}_isolation_forest.joblib")
        np.savez(save_dir / f"{safe}_if_scaler.npz", mean=mean, std=std)
        company_index[safe] = soc
        all_scored.append(scored)

    # Modèle global (repli pour opérateurs avec peu de données)
    m_global, mean_global, std_global = _an.train_isolation_forest(trips, cfg)
    joblib.dump(m_global, save_dir / "isolation_forest.joblib")
    np.savez(save_dir / "if_scaler.npz", mean=mean_global, std=std_global)

    trained_socs = set(company_index.values())
    remaining = trips[~trips["societe"].isin(trained_socs)]
    if len(remaining) > 0:
        all_scored.append(_an.score_trips(m_global, mean_global, std_global, remaining))

    with open(save_dir / "company_models.json", "w") as f:
        json.dump(company_index, f, ensure_ascii=False)

    trips_scored = pd.concat(all_scored, ignore_index=True)
    n_if = int(trips_scored["anomaly"].sum())
    print(f"    total signalés : {n_if}/{len(trips_scored)} ({100*n_if/len(trips_scored):.1f}%)")

    # ── Autoencodeur LSTM (global) ────────────────────────────────────────────
    print("  Entraînement de l'Autoencodeur LSTM...")
    X, _ = _an.build_sequences(fa, cfg)
    print(f"    séquences : {X.shape}")

    lstm_ae, train_errors = _an.train_lstm_autoencoder(X, cfg)
    threshold = float(np.percentile(train_errors, 95))
    lstm_scores = _an.lstm_anomaly_scores(lstm_ae, X)
    n_lstm = int((lstm_scores > threshold).sum())
    print(f"    signalés : {n_lstm}/{len(X)} (seuil={threshold:.5f})")

    # Attacher les scores LSTM par clé de trajet
    lstm_df = pd.DataFrame(
        list(_an.build_sequences(fa, cfg)[1]),
        columns=_an.TRIP_KEYS
    )
    lstm_df["lstm_score"] = lstm_scores
    trips_scored = trips_scored.merge(lstm_df, on=_an.TRIP_KEYS, how="left")
    trips_scored["lstm_score"] = trips_scored["lstm_score"].fillna(0.0)
    trips_scored["lstm_anomaly"] = trips_scored["lstm_score"] > threshold
    trips_scored["dual_anomaly"] = trips_scored["anomaly"] & trips_scored["lstm_anomaly"]
    trips_scored.to_parquet(save_dir / "trips_scored.parquet", index=False)

    torch.save(lstm_ae.state_dict(), save_dir / "lstm_ae.pt")
    np.save(save_dir / "lstm_ae_threshold.npy", np.array(threshold))
    with open(save_dir / "lstm_ae_config.json", "w") as f:
        json.dump({"hidden": cfg.lstm_hidden, "seq_pad": cfg.seq_pad,
                   "n_feats": X.shape[2]}, f)

    print(f"  -> Artefacts sauvegardés dans {save_dir}")
    return {
        "if_models": {soc: None for soc in trained_socs},
        "lstm_ae": lstm_ae, "trips": trips_scored,
        "n_if": n_if, "n_lstm": n_lstm, "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Charge les modèles d'anomalie entraînés depuis save_dir."""
    import joblib
    import torch

    save_dir = Path(save_dir)

    # Per-company IF models
    if_models: dict[str, tuple] = {}
    index_path = save_dir / "company_models.json"
    if index_path.exists():
        with open(index_path) as f:
            company_index = json.load(f)
        for safe, soc in company_index.items():
            m_path = save_dir / f"{safe}_isolation_forest.joblib"
            s_path = save_dir / f"{safe}_if_scaler.npz"
            if m_path.exists() and s_path.exists():
                m = joblib.load(m_path)
                sc = np.load(s_path)
                if_models[soc] = (m, sc["mean"], sc["std"])

    # Global fallback
    global_m = joblib.load(save_dir / "isolation_forest.joblib")
    global_sc = np.load(save_dir / "if_scaler.npz")
    if_models["_global"] = (global_m, global_sc["mean"], global_sc["std"])

    # LSTM AE
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

    n_co = len(if_models) - 1
    print(f"Modèles d'anomalie chargés ({n_co} opérateur(s) + repli global, seuil={threshold:.5f})")
    return {
        "if_models": if_models,
        # backward-compat keys used by explain_trips fallback
        "if_model": global_m, "if_mean": global_sc["mean"], "if_std": global_sc["std"],
        "lstm_ae": lstm_ae, "threshold": threshold, "trips": trips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Service (scoring en ligne)
# ─────────────────────────────────────────────────────────────────────────────

def score(models: dict, fa: pd.DataFrame) -> pd.DataFrame:
    """Score les trajets dans de nouvelles données avec les deux modèles."""
    cfg = _an.AnomalyConfig()
    fa = fa.copy()
    fa["dwell_s"] = fa.get("dwell_s", pd.Series(0.0, index=fa.index)).fillna(0)

    trips = _an.trip_features(fa, cfg)
    if_models = models.get("if_models", {})

    # Score per company using dedicated models
    parts = []
    for soc, grp in trips.groupby("societe"):
        if soc in if_models:
            m, mean, std = if_models[soc]
        else:
            m, mean, std = if_models.get("_global",
                (models["if_model"], models["if_mean"], models["if_std"]))
        parts.append(_an.score_trips(m, mean, std, grp))

    trips = pd.concat(parts, ignore_index=True) if parts else trips

    X, ids = _an.build_sequences(fa, cfg)
    if len(X) > 0:
        lstm_scores = _an.lstm_anomaly_scores(models["lstm_ae"], X)
        lstm_df = pd.DataFrame(ids, columns=_an.TRIP_KEYS)
        lstm_df["lstm_score"] = lstm_scores
        trips = trips.merge(lstm_df, on=_an.TRIP_KEYS, how="left")
        trips["lstm_score"] = trips["lstm_score"].fillna(0.0)
        trips["lstm_anomaly"] = trips["lstm_score"] > models["threshold"]
    else:
        trips["lstm_score"] = 0.0
        trips["lstm_anomaly"] = False

    trips["dual_anomaly"] = trips["anomaly"] & trips["lstm_anomaly"]
    return trips


# ─────────────────────────────────────────────────────────────────────────────
# Explicabilité
# ─────────────────────────────────────────────────────────────────────────────

_REASON_BUILDERS = {
    "max_dwell_s":   ("high", lambda v: f"Immobilisation anormale à un arrêt (~{v/60:.0f} min sans mouvement GPS)"),
    "total_elapsed": ("high", lambda v: f"Trajet anormalement long (~{int(v)//60}h{int(v)%60:02d} au total)"),
    "mean_dwell_s":  ("high", lambda v: f"Durée d'arrêt moyenne élevée (~{v/60:.1f} min/arrêt)"),
    "dist_m_max":    ("high", lambda v: f"Déviation importante de l'itinéraire (~{v:.0f} m hors trajectoire)"),
    "match_rate":    ("low",  lambda v: f"Mauvais suivi GPS / hors itinéraire — seulement {v*100:.0f}% des arrêts détectés"),
    "n_stops":       ("low",  lambda v: f"Nombre d'arrêts desservis anormalement faible ({int(v)} arrêts)"),
    "max_dark_s":    ("high", lambda v: f"Perte de signal GPS prolongée à un arrêt (~{v/60:.0f} min sans ping)"),
}


def explain_trips(models: dict, scored: pd.DataFrame, *,
                  z_thresh: float = 1.5, max_reasons: int = 3) -> pd.DataFrame:
    """Ajoute des colonnes d'explicabilité à un DataFrame issu de `score`.

    Utilise les stats (mean/std) de l'opérateur concerné pour calculer les z-scores,
    afin que la comparaison soit toujours relative à la normale de cet opérateur.
    """
    feats = _an.FEATURES
    if_models = models.get("if_models", {})

    out = scored.copy()
    reasons_col, top_col = [], []

    for _, row in out.iterrows():
        # Use company-specific stats so z-scores compare against that operator's baseline
        soc = row.get("societe", "")
        if soc in if_models:
            _, mean, std = if_models[soc]
        elif "_global" in if_models:
            _, mean, std = if_models["_global"]
        else:
            mean = np.asarray(models["if_mean"], dtype=float)
            std = np.asarray(models["if_std"], dtype=float)

        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)

        vals = row[feats].values.astype(float)
        z = (vals - mean) / std
        scored_feats = []
        for i, f in enumerate(feats):
            direction, _builder = _REASON_BUILDERS.get(f, (None, None))
            if direction is None:
                continue
            signed = z[i] if direction == "high" else -z[i]
            if signed >= z_thresh:
                scored_feats.append((signed, f, row[f]))
        scored_feats.sort(reverse=True)
        reasons = [_REASON_BUILDERS[f][1](v) for _, f, v in scored_feats[:max_reasons]]

        # Informational signal-loss note — only when max_dark_s didn't already trigger
        dark = float(row.get("max_dark_s", 0) or 0)
        already_in_reasons = any(f == "max_dark_s" for _, f, _ in scored_feats)
        if dark > 600 and not already_in_reasons and len(reasons) < max_reasons:
            reasons.append(f"Perte de signal GPS à un arrêt (~{dark/60:.0f} min sans ping — non comptée comme immobilisation)")

        reasons_col.append(reasons)
        top_col.append(scored_feats[0][1] if scored_feats else None)

    out["reasons"] = reasons_col
    out["top_feature"] = top_col
    out["anomaly_strength"] = (-out["if_score"]).round(3)
    out["severity"] = np.where(out.get("dual_anomaly", False), "high",
                       np.where(out.get("anomaly", False), "medium", "low"))
    if "worst_dwell_stop" not in out.columns:
        out["worst_dwell_stop"] = None
    return out
