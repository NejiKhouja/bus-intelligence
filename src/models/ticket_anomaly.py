"""Module de détection d'anomalies billetterie — entraîner, sauvegarder, charger, scorer.

Signal COMPLÉMENTAIRE à `src/models/anomaly.py` (GPS/trajet), pas fusionné avec lui -- grain
journalier (société, ligne, bus, jour), voir `src/data/ticket_anomaly.py` pour le pourquoi.
Même schéma qu'anomaly.py : un Isolation Forest par société + repli global.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ticket_anomaly as _ta

SAVE_DIR = Path("models/ticket_anomaly")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name))


def train(db_conn, save_dir: str | Path = SAVE_DIR) -> dict:
    """Entraîne un Isolation Forest par société sur `tickets_daily`.

    Sauvegarde :
      {safe_societe}_isolation_forest.joblib / _if_scaler.npz  -- IF par société
      company_models.json                                       -- index nom_sûr -> nom_original
      isolation_forest.joblib / if_scaler.npz                    -- repli global
      days_scored.parquet                                        -- tous les jours scorés
    """
    import joblib

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = _ta.TicketAnomalyConfig()

    df = _ta.build_features(_ta.load_tickets_daily(db_conn))
    print(f"  jours billetterie : {len(df):,}")

    company_index: dict[str, str] = {}
    all_scored: list[pd.DataFrame] = []
    if_mean_by_soc: dict[str, tuple] = {}

    for soc in sorted(df["societe"].unique()):
        soc_days = df[df["societe"] == soc].copy()
        if len(soc_days) < cfg.min_records_per_company:
            print(f"    {soc}: {len(soc_days)} jours -- trop peu, repli sur modèle global")
            continue
        m, mean, std = _ta.train_isolation_forest(soc_days, cfg)
        scored = _ta.score_days(m, mean, std, soc_days)
        n = int(scored["anomaly"].sum())
        print(f"    {soc}: {n}/{len(soc_days)} signalés ({100*n/len(soc_days):.1f}%)")
        safe = _safe(soc)
        joblib.dump(m, save_dir / f"{safe}_isolation_forest.joblib")
        np.savez(save_dir / f"{safe}_if_scaler.npz", mean=mean, std=std)
        company_index[safe] = soc
        if_mean_by_soc[soc] = (mean, std)
        all_scored.append(scored)

    m_global, mean_global, std_global = _ta.train_isolation_forest(df, cfg)
    joblib.dump(m_global, save_dir / "isolation_forest.joblib")
    np.savez(save_dir / "if_scaler.npz", mean=mean_global, std=std_global)

    trained_socs = set(company_index.values())
    remaining = df[~df["societe"].isin(trained_socs)]
    if len(remaining) > 0:
        all_scored.append(_ta.score_days(m_global, mean_global, std_global, remaining))

    with open(save_dir / "company_models.json", "w") as f:
        json.dump(company_index, f, ensure_ascii=False)

    days_scored = pd.concat(all_scored, ignore_index=True)
    n_total = int(days_scored["anomaly"].sum())
    print(f"    total signalés : {n_total}/{len(days_scored)} ({100*n_total/len(days_scored):.1f}%)")

    days_scored.to_parquet(save_dir / "days_scored.parquet", index=False)
    print(f"  -> Artefacts sauvegardés dans {save_dir}")
    return {"days": days_scored, "n_anomaly": n_total, "if_mean_by_soc": if_mean_by_soc}


def load(save_dir: str | Path = SAVE_DIR) -> dict:
    import joblib

    save_dir = Path(save_dir)
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

    global_m = joblib.load(save_dir / "isolation_forest.joblib")
    global_sc = np.load(save_dir / "if_scaler.npz")
    if_models["_global"] = (global_m, global_sc["mean"], global_sc["std"])

    days = pd.read_parquet(save_dir / "days_scored.parquet")
    print(f"Modèles d'anomalie billetterie chargés ({len(if_models)-1} société(s) + repli global)")
    return {"if_models": if_models, "days": days}


def score(models: dict, day_rows: pd.DataFrame) -> pd.DataFrame:
    """Score de nouveaux jours billetterie avec les modèles par société."""
    day_rows = _ta.build_features(day_rows)
    if_models = models.get("if_models", {})
    parts = []
    for soc, grp in day_rows.groupby("societe"):
        m, mean, std = if_models.get(soc, if_models.get("_global"))
        parts.append(_ta.score_days(m, mean, std, grp))
    return pd.concat(parts, ignore_index=True) if parts else day_rows


def explain(models: dict, scored: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des raisons explicables en langage clair (voir `_ta.explain_days`).

    Comme `anomaly.explain_trips` : compare chaque jour à la normale DE SA PROPRE société
    quand un modèle dédié existe, sinon replie sur les stats du modèle global -- une
    société sans historique suffisant pour un IF dédié obtient quand même des raisons.
    """
    if_models = models.get("if_models", {})
    _, g_mean, g_std = if_models.get("_global", (None, None, None))
    if_mean_by_soc: dict[str, tuple] = {}
    for soc in scored["societe"].unique():
        if soc in if_models:
            _, mean, std = if_models[soc]
        else:
            mean, std = g_mean, g_std
        if_mean_by_soc[soc] = (mean, std)
    return _ta.explain_days(scored, if_mean_by_soc)
