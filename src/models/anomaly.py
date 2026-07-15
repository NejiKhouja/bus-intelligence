"""Module de détection d'anomalies — entraîner, sauvegarder, charger, scorer.

Cycle de vie complet pour le Module 3 :
  train()  -> Isolation Forest + Autoencodeur LSTM, 3 paliers -> sauvegardé dans models/anomaly/
  load()   -> charge les artefacts depuis le disque
  score()  -> signale les trajets anormaux dans de nouvelles données avec les deux modèles

Trois paliers de spécificité, du plus fin au plus large : (société, ligne) dédié ->
société dédié -> global. Un modèle dédié à une LIGNE est utilisé dès que cette ligne a
assez de trajets ; sinon repli sur le modèle de l'opérateur entier ; sinon repli sur le
modèle global. Sans ça, la durée normale d'une grosse ligne urbaine (ex. S.T.S/219,
9 687 trajets) sert de référence à une petite ligne du même opérateur (ex. S.T.S/304,
588 trajets) et signale son trafic normal comme anormal (constaté : 99.8% des trajets
de la ligne 304 signalés, alors qu'elle a largement assez de données pour son propre
modèle -- 588 trajets >> le seuil de 30 pour l'IF et de 200 pour le LSTM).
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


# Entraînement
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

    # Isolation Forest -- 3 paliers : (société, ligne) dédié -> société dédié -> global
    print("  Entraînement des Isolation Forests (par ligne, puis par opérateur, puis global)...")
    trips = _an.trip_features(fa, cfg)
    print(f"    trajets total : {len(trips):,}")

    min_trips_per_line = 30
    min_trips_per_company = 30

    line_index: dict[str, list] = {}      # safe_name -> [societe, line]
    company_index: dict[str, str] = {}    # safe_name -> societe
    if_line_models: dict[tuple, tuple] = {}
    if_company_models: dict[str, tuple] = {}

    for (soc, line), grp in trips.groupby(["societe", "line"]):
        if len(grp) < min_trips_per_line:
            continue
        m, mean, std = _an.train_isolation_forest(grp, cfg)
        scored = _an.score_trips(m, mean, std, grp)
        n = int(scored["anomaly"].sum())
        print(f"    {soc}/{line}: {n}/{len(grp)} signalés ({100*n/len(grp):.1f}%) -- dédié à la ligne")
        safe = _safe(f"{soc}__{line}")
        joblib.dump(m, save_dir / f"line_{safe}_isolation_forest.joblib")
        np.savez(save_dir / f"line_{safe}_if_scaler.npz", mean=mean, std=std)
        line_index[safe] = [soc, line]
        if_line_models[(soc, line)] = (m, mean, std)

    for soc, grp in trips.groupby("societe"):
        if len(grp) < min_trips_per_company:
            print(f"    {soc}: {len(grp)} trajets — trop peu, repli sur modèle global")
            continue
        m, mean, std = _an.train_isolation_forest(grp, cfg)
        scored = _an.score_trips(m, mean, std, grp)
        n = int(scored["anomaly"].sum())
        print(f"    {soc}: {n}/{len(grp)} signalés ({100*n/len(grp):.1f}%)")
        safe = _safe(soc)
        joblib.dump(m, save_dir / f"{safe}_isolation_forest.joblib")
        np.savez(save_dir / f"{safe}_if_scaler.npz", mean=mean, std=std)
        company_index[safe] = soc
        if_company_models[soc] = (m, mean, std)

    # Modèle global (repli ultime)
    m_global, mean_global, std_global = _an.train_isolation_forest(trips, cfg)
    joblib.dump(m_global, save_dir / "isolation_forest.joblib")
    np.savez(save_dir / "if_scaler.npz", mean=mean_global, std=std_global)

    with open(save_dir / "line_models.json", "w") as f:
        json.dump(line_index, f, ensure_ascii=False)
    with open(save_dir / "company_models.json", "w") as f:
        json.dump(company_index, f, ensure_ascii=False)

    # Score chaque trajet avec le modèle le plus spécifique disponible (ligne > opérateur > global)
    all_scored: list[pd.DataFrame] = []
    for (soc, line), grp in trips.groupby(["societe", "line"]):
        if (soc, line) in if_line_models:
            m, mean, std = if_line_models[(soc, line)]
        elif soc in if_company_models:
            m, mean, std = if_company_models[soc]
        else:
            m, mean, std = m_global, mean_global, std_global
        all_scored.append(_an.score_trips(m, mean, std, grp))

    trips_scored = pd.concat(all_scored, ignore_index=True)
    n_if = int(trips_scored["anomaly"].sum())
    print(f"    total signalés : {n_if}/{len(trips_scored)} ({100*n_if/len(trips_scored):.1f}%)"
          f" -- {len(if_line_models)} ligne(s) dédiée(s), {len(if_company_models)} opérateur(s) dédié(s)")

    #  Autoencodeur LSTM -- 3 paliers : (société, ligne) dédié -> société dédié -> global
    # Même raison que l'IF : un modèle unique dominé par la plus grosse ligne/société
    # apprend "normal" = "ce que fait la plus grosse" et signale le trafic normal des
    # petites lignes/sociétés comme anormal (dual_anomaly=0 constaté pour la dominante,
    # 99.8% signalé pour une petite ligne avec pourtant assez de données pour son propre
    # modèle). `min_trips_lstm_*` est plus élevé que celui de l'IF (30) car un autoencodeur
    # a bien plus de paramètres à apprendre en dessous, repli sur le palier suivant.
    print("  Entraînement des Autoencodeurs LSTM (par ligne, puis par opérateur, puis global)...")
    min_trips_lstm_line = 200
    min_trips_lstm_company = 200

    lstm_line_index: dict[str, list] = {}       # safe_name -> [societe, line]
    lstm_company_index: dict[str, str] = {}     # safe_name -> nom original
    lstm_line_models: dict[tuple, tuple] = {}    # (soc, line) -> (modèle, seuil)
    lstm_company_models: dict[str, tuple] = {}   # soc -> (modèle, seuil)

    for (soc, line), grp_fa in fa.groupby(["societe", "line"]):
        X, _ = _an.build_sequences(grp_fa, cfg)
        if len(X) < min_trips_lstm_line:
            continue
        m, train_errors = _an.train_lstm_autoencoder(X, cfg)
        thr = float(np.percentile(train_errors, 95))
        n = int((train_errors > thr).sum())
        print(f"    {soc}/{line}: {n}/{len(X)} signalés (seuil={thr:.5f}) -- dédié à la ligne")
        safe = _safe(f"{soc}__{line}")
        torch.save(m.state_dict(), save_dir / f"line_{safe}_lstm_ae.pt")
        np.save(save_dir / f"line_{safe}_lstm_threshold.npy", np.array(thr))
        with open(save_dir / f"line_{safe}_lstm_ae_config.json", "w") as f:
            json.dump({"hidden": cfg.lstm_hidden, "seq_pad": cfg.seq_pad, "n_feats": X.shape[2]}, f)
        lstm_line_index[safe] = [soc, line]
        lstm_line_models[(soc, line)] = (m, thr)

    for soc in sorted(fa["societe"].unique()):
        soc_fa = fa[fa["societe"] == soc]
        X_soc, _ = _an.build_sequences(soc_fa, cfg)
        if len(X_soc) < min_trips_lstm_company:
            print(f"    {soc}: {len(X_soc)} trajets -- trop peu pour un LSTM dédié, repli global")
            continue
        m, train_errors = _an.train_lstm_autoencoder(X_soc, cfg)
        thr = float(np.percentile(train_errors, 95))
        n = int((train_errors > thr).sum())
        print(f"    {soc}: {n}/{len(X_soc)} signalés (seuil={thr:.5f})")
        safe = _safe(soc)
        torch.save(m.state_dict(), save_dir / f"{safe}_lstm_ae.pt")
        np.save(save_dir / f"{safe}_lstm_threshold.npy", np.array(thr))
        with open(save_dir / f"{safe}_lstm_ae_config.json", "w") as f:
            json.dump({"hidden": cfg.lstm_hidden, "seq_pad": cfg.seq_pad, "n_feats": X_soc.shape[2]}, f)
        lstm_company_index[safe] = soc
        lstm_company_models[soc] = (m, thr)

    # Repli global entraîné sur TOUS les trajets, utilisé pour les lignes/sociétés sans modèle dédié
    X_all, _ = _an.build_sequences(fa, cfg)
    print(f"    séquences totales : {X_all.shape}")
    lstm_global, train_errors_global = _an.train_lstm_autoencoder(X_all, cfg)
    threshold_global = float(np.percentile(train_errors_global, 95))
    n_global = int((train_errors_global > threshold_global).sum())
    print(f"    Modèle global (repli) : {n_global}/{len(X_all)} signalés (seuil={threshold_global:.5f})")
    torch.save(lstm_global.state_dict(), save_dir / "lstm_ae.pt")
    np.save(save_dir / "lstm_ae_threshold.npy", np.array(threshold_global))
    with open(save_dir / "lstm_ae_config.json", "w") as f:
        json.dump({"hidden": cfg.lstm_hidden, "seq_pad": cfg.seq_pad, "n_feats": X_all.shape[2]}, f)

    with open(save_dir / "lstm_line_models.json", "w") as f:
        json.dump(lstm_line_index, f, ensure_ascii=False)
    with open(save_dir / "lstm_company_models.json", "w") as f:
        json.dump(lstm_company_index, f, ensure_ascii=False)

    # Score chaque trajet avec le modèle le plus spécifique (ligne > opérateur > global)
    lstm_rows = []
    for (soc, line), grp_fa in fa.groupby(["societe", "line"]):
        X, ids = _an.build_sequences(grp_fa, cfg)
        if len(X) == 0:
            continue
        if (soc, line) in lstm_line_models:
            model, thr = lstm_line_models[(soc, line)]
        elif soc in lstm_company_models:
            model, thr = lstm_company_models[soc]
        else:
            model, thr = lstm_global, threshold_global
        scores = _an.lstm_anomaly_scores(model, X)
        df_soc = pd.DataFrame(ids, columns=_an.TRIP_KEYS)
        df_soc["lstm_score"] = scores
        df_soc["lstm_anomaly"] = scores > thr
        lstm_rows.append(df_soc)
    lstm_df = pd.concat(lstm_rows, ignore_index=True)

    trips_scored = trips_scored.merge(lstm_df, on=_an.TRIP_KEYS, how="left")
    trips_scored["lstm_score"] = trips_scored["lstm_score"].fillna(0.0)
    trips_scored["lstm_anomaly"] = trips_scored["lstm_anomaly"].fillna(False)
    trips_scored["dual_anomaly"] = trips_scored["anomaly"] & trips_scored["lstm_anomaly"]
    trips_scored.to_parquet(save_dir / "trips_scored.parquet", index=False)

    n_lstm = int(trips_scored["lstm_anomaly"].sum())
    print(f"    LSTM total signalés : {n_lstm}/{len(trips_scored)} ({100*n_lstm/len(trips_scored):.1f}%) "
          f"-- {len(lstm_line_models)} ligne(s) dédiée(s), {len(lstm_company_models)} opérateur(s) dédié(s) + repli global")

    print(f"  -> Artefacts sauvegardés dans {save_dir}")
    return {
        "if_models": {soc: None for soc in company_index.values()},
        "lstm_ae": lstm_global, "trips": trips_scored,
        "n_if": n_if, "n_lstm": n_lstm, "threshold": threshold_global,
    }


# Chargement
def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Charge les modèles d'anomalie entraînés depuis save_dir.

    Fonctionne aussi SANS torch installé (déploiement slim, voir
    requirements-anomaly.txt) : l'autoencodeur LSTM est alors ignoré et seul
    l'Isolation Forest sert au scoring en ligne. Les scores LSTM PRÉ-CALCULÉS de
    trips_scored.parquet restent disponibles — ils datent de l'entraînement.
    """
    import joblib
    try:
        import torch
    except ImportError:
        torch = None
        print("  torch non installé -- autoencodeur LSTM ignoré (scoring IF uniquement)")

    save_dir = Path(save_dir)

    # IF -- 3 paliers : clé tuple (soc, line) = dédié ligne, clé str soc = dédié opérateur,
    # "_global" = repli. Les 3 types de clé coexistent dans le même dict (pas de collision,
    # tuple != str en tant que clé), donc le lookup `(soc, line) in if_models` puis
    # `soc in if_models` reste sûr.
    if_models: dict = {}
    line_index_path = save_dir / "line_models.json"
    if line_index_path.exists():
        with open(line_index_path) as f:
            line_index = json.load(f)
        for safe, (soc, line) in line_index.items():
            m_path = save_dir / f"line_{safe}_isolation_forest.joblib"
            s_path = save_dir / f"line_{safe}_if_scaler.npz"
            if m_path.exists() and s_path.exists():
                m = joblib.load(m_path)
                sc = np.load(s_path)
                if_models[(soc, line)] = (m, sc["mean"], sc["std"])

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

    # LSTM AE -- même schéma 3 paliers que l'IF ; sauté sans torch
    lstm_models: dict = {}
    lstm_ae, threshold = None, None
    n_line_if = sum(1 for k in if_models if isinstance(k, tuple))
    n_line_lstm = 0
    if torch is not None:
        lstm_line_index_path = save_dir / "lstm_line_models.json"
        if lstm_line_index_path.exists():
            with open(lstm_line_index_path) as f:
                lstm_line_index = json.load(f)
            for safe, (soc, line) in lstm_line_index.items():
                cfg_path = save_dir / f"line_{safe}_lstm_ae_config.json"
                model_path = save_dir / f"line_{safe}_lstm_ae.pt"
                thr_path = save_dir / f"line_{safe}_lstm_threshold.npy"
                if cfg_path.exists() and model_path.exists() and thr_path.exists():
                    with open(cfg_path) as f:
                        ae_cfg_line = json.load(f)
                    m = _an._make_lstm_autoencoder(ae_cfg_line["seq_pad"], ae_cfg_line["n_feats"], ae_cfg_line["hidden"])
                    m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                    m.eval()
                    lstm_models[(soc, line)] = (m, float(np.load(thr_path)))
            n_line_lstm = len(lstm_line_index)

        lstm_index_path = save_dir / "lstm_company_models.json"
        if lstm_index_path.exists():
            with open(lstm_index_path) as f:
                lstm_company_index = json.load(f)
            for safe, soc in lstm_company_index.items():
                cfg_path = save_dir / f"{safe}_lstm_ae_config.json"
                model_path = save_dir / f"{safe}_lstm_ae.pt"
                thr_path = save_dir / f"{safe}_lstm_threshold.npy"
                if cfg_path.exists() and model_path.exists() and thr_path.exists():
                    with open(cfg_path) as f:
                        ae_cfg_soc = json.load(f)
                    m = _an._make_lstm_autoencoder(ae_cfg_soc["seq_pad"], ae_cfg_soc["n_feats"], ae_cfg_soc["hidden"])
                    m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                    m.eval()
                    lstm_models[soc] = (m, float(np.load(thr_path)))

        with open(save_dir / "lstm_ae_config.json") as f:
            ae_cfg = json.load(f)
        lstm_ae = _an._make_lstm_autoencoder(
            ae_cfg["seq_pad"], ae_cfg["n_feats"], ae_cfg["hidden"]
        )
        lstm_ae.load_state_dict(torch.load(save_dir / "lstm_ae.pt", map_location="cpu",
                                           weights_only=True))
        lstm_ae.eval()

        threshold = float(np.load(save_dir / "lstm_ae_threshold.npy"))
        lstm_models["_global"] = (lstm_ae, threshold)

    trips = pd.read_parquet(save_dir / "trips_scored.parquet")

    n_co = sum(1 for k in if_models if isinstance(k, str) and k != "_global")
    n_co_lstm = sum(1 for k in lstm_models if isinstance(k, str) and k != "_global")
    lstm_note = (f"{n_co_lstm} opérateur(s) + {n_line_lstm} ligne(s) LSTM" if torch is not None
                else "LSTM désactivé (pas de torch)")
    print(f"Modèles d'anomalie chargés ({n_co} opérateur(s) + {n_line_if} ligne(s) IF + {lstm_note} + repli global)")
    return {
        "if_models": if_models,
        # backward-compat keys used by explain_trips fallback
        "if_model": global_m, "if_mean": global_sc["mean"], "if_std": global_sc["std"],
        "lstm_models": lstm_models,
        "lstm_ae": lstm_ae, "threshold": threshold, "trips": trips,
    }


# Service (scoring en ligne)
def score(models: dict, fa: pd.DataFrame) -> pd.DataFrame:
    """Score les trajets dans de nouvelles données avec les deux modèles."""
    cfg = _an.AnomalyConfig()
    fa = fa.copy()
    fa["dwell_s"] = fa.get("dwell_s", pd.Series(0.0, index=fa.index)).fillna(0)

    trips = _an.trip_features(fa, cfg)
    if_models = models.get("if_models", {})

    # Score avec le modèle le plus spécifique disponible (ligne > opérateur > global)
    parts = []
    for (soc, line), grp in trips.groupby(["societe", "line"]):
        if (soc, line) in if_models:
            m, mean, std = if_models[(soc, line)]
        elif soc in if_models:
            m, mean, std = if_models[soc]
        else:
            m, mean, std = if_models.get("_global",
                (models["if_model"], models["if_mean"], models["if_std"]))
        parts.append(_an.score_trips(m, mean, std, grp))

    trips = pd.concat(parts, ignore_index=True) if parts else trips

    # Score LSTM avec le modèle le plus spécifique (ligne > opérateur > global).
    # Vide en déploiement slim sans torch -> scoring IF uniquement.
    lstm_models = models.get("lstm_models") or (
        {"_global": (models["lstm_ae"], models["threshold"])}
        if models.get("lstm_ae") is not None else {})
    lstm_rows = []
    for (soc, line), soc_fa in (fa.groupby(["societe", "line"]) if lstm_models else ()):
        X, ids = _an.build_sequences(soc_fa, cfg)
        if len(X) == 0:
            continue
        if (soc, line) in lstm_models:
            model, thr = lstm_models[(soc, line)]
        elif soc in lstm_models:
            model, thr = lstm_models[soc]
        else:
            model, thr = lstm_models["_global"]
        scores = _an.lstm_anomaly_scores(model, X)
        df_soc = pd.DataFrame(ids, columns=_an.TRIP_KEYS)
        df_soc["lstm_score"] = scores
        df_soc["lstm_anomaly"] = scores > thr
        lstm_rows.append(df_soc)

    if lstm_rows:
        lstm_df = pd.concat(lstm_rows, ignore_index=True)
        trips = trips.merge(lstm_df, on=_an.TRIP_KEYS, how="left")
        trips["lstm_score"] = trips["lstm_score"].fillna(0.0)
        trips["lstm_anomaly"] = trips["lstm_anomaly"].fillna(False)
    else:
        trips["lstm_score"] = 0.0
        trips["lstm_anomaly"] = False

    trips["dual_anomaly"] = trips["anomaly"] & trips["lstm_anomaly"]
    return trips


# Explicabilité
_REASON_BUILDERS = {
    "max_dwell_s":   ("high", lambda v: f"Immobilisation anormale à un arrêt (~{v/60:.0f} min sans mouvement GPS)"),
    "total_elapsed": ("high", lambda v: f"Trajet anormalement long (~{int(v)//60}h{int(v)%60:02d} au total)"),
    "mean_dwell_s":  ("high", lambda v: f"Durée d'arrêt moyenne élevée (~{v/60:.1f} min/arrêt)"),
    "dist_m_max":    ("high", lambda v: f"Déviation importante de l'itinéraire (~{v:.0f} m hors trajectoire)"),
    "match_rate":    ("low",  lambda v: f"Mauvais suivi GPS / hors itinéraire — seulement {v*100:.0f}% des arrêts détectés"),
    "n_stops":       ("low",  lambda v: f"Nombre d'arrêts desservis anormalement faible ({int(v)} arrêts)"),
    "max_dark_s":    ("high", lambda v: f"Perte de signal GPS prolongée à un arrêt (~{v/60:.0f} min sans ping)"),
    "elapsed_vs_bus_z":  ("either", lambda v: f"Trajet {'plus long' if v > 0 else 'plus court'} que d'habitude pour CE BUS (z={v:+.1f})"),
    "elapsed_vs_line_z": ("either", lambda v: f"Trajet {'plus long' if v > 0 else 'plus court'} que d'habitude pour CETTE LIGNE (z={v:+.1f})"),
    "terminus_idle_min": ("high", lambda v: f"Stationnement prolongé au terminus (~{v:.0f} min avant départ/après arrivée — service probablement non clôturé)"),
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
    reasons_col, reason_features_col, top_col = [], [], []

    for _, row in out.iterrows():
        # Use the most specific stats available so z-scores compare against the right
        # baseline: this line's own history if it has a dedicated model, else the
        # operator's, else the global network.
        soc = row.get("societe", "")
        line = row.get("line", "")
        if (soc, line) in if_models:
            _, mean, std = if_models[(soc, line)]
        elif soc in if_models:
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
            if direction == "either":
                signed = abs(z[i])
            else:
                signed = z[i] if direction == "high" else -z[i]
            if signed >= z_thresh:
                scored_feats.append((signed, f, row[f]))
        scored_feats.sort(reverse=True)
        reasons = [_REASON_BUILDERS[f][1](v) for _, f, v in scored_feats[:max_reasons]]
        reason_features = [f for _, f, v in scored_feats[:max_reasons]]

        # Informational signal-loss note — only when max_dark_s didn't already trigger
        dark = float(row.get("max_dark_s", 0) or 0)
        already_in_reasons = any(f == "max_dark_s" for _, f, _ in scored_feats)
        if dark > 600 and not already_in_reasons and len(reasons) < max_reasons:
            reasons.append(f"Perte de signal GPS à un arrêt (~{dark/60:.0f} min sans ping — non comptée comme immobilisation)")
            reason_features.append("max_dark_s")

        reasons_col.append(reasons)
        reason_features_col.append(reason_features)
        top_col.append(scored_feats[0][1] if scored_feats else None)

    out["reasons"] = reasons_col
    out["reason_features"] = reason_features_col
    out["top_feature"] = top_col
    out["anomaly_strength"] = (-out["if_score"]).round(3)
    out["severity"] = np.where(out.get("dual_anomaly", False), "high",
                       np.where(out.get("anomaly", False), "medium", "low"))
    if "worst_dwell_stop" not in out.columns:
        out["worst_dwell_stop"] = None
    return out
