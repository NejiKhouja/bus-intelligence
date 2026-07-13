"""Module de détection d'anomalies billetterie — entraîner, sauvegarder, charger, scorer.

Signal COMPLÉMENTAIRE à `src/models/anomaly.py` (GPS/trajet), pas fusionné avec lui -- grain
journalier (société, ligne, bus, jour), voir `src/data/ticket_anomaly.py` pour le pourquoi.

Hiérarchie des modèles (2026-07-07) : un Isolation Forest **par (société, ligne)** d'abord,
repli société, puis repli global. La normale billetterie est une propriété de la LIGNE
(tarification intercity ~20 DT/ticket vs urbaine ~2 DT) -- le modèle par société seul
signalait 100% des jours des lignes tarifairement atypiques (S.R.T.K ligne 220 : 38/38),
un artefact de référence, pas de la fraude. Voir TicketAnomalyConfig.min_records_per_line.
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
    """Entraîne les Isolation Forests billetterie : par (société, ligne) si assez de jours,
    par société en repli, global en dernier recours.

    Sauvegarde :
      line_models.joblib                                        -- {(societe, ligne): (IF, mean, std)}
      {safe_societe}_isolation_forest.joblib / _if_scaler.npz  -- IF par société (repli)
      company_models.json                                       -- index nom_sûr -> nom_original
      isolation_forest.joblib / if_scaler.npz                    -- repli global
      days_scored.parquet                                        -- tous les jours scorés (+model_tier)
    """
    import joblib

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = _ta.TicketAnomalyConfig()

    df = _ta.build_features(_ta.load_tickets_daily(db_conn))
    print(f"  jours billetterie : {len(df):,}")

    # ── Modèles par (société, ligne) -- la vraie référence de normalité ─────────
    line_models: dict[tuple, tuple] = {}
    for (soc, line), grp in df.groupby(["societe", "line"]):
        if len(grp) < cfg.min_records_per_line:
            continue
        m, mean, std = _ta.train_isolation_forest(grp, cfg)
        line_models[(soc, line)] = (m, mean, std)
    n_line_days = int(df.set_index(["societe", "line"]).index.isin(line_models.keys()).sum())
    print(f"    modèles par ligne : {len(line_models)} lignes "
          f"({n_line_days}/{len(df)} jours couverts) ; le reste replie société/global")

    # ── Repli par société (lignes sous le seuil) + repli global ────────────────
    company_index: dict[str, str] = {}
    company_models: dict[str, tuple] = {}
    for soc in sorted(df["societe"].unique()):
        soc_days = df[df["societe"] == soc]
        if len(soc_days) < cfg.min_records_per_company:
            print(f"    {soc}: {len(soc_days)} jours -- trop peu, repli sur modèle global")
            continue
        m, mean, std = _ta.train_isolation_forest(soc_days, cfg)
        safe = _safe(soc)
        joblib.dump(m, save_dir / f"{safe}_isolation_forest.joblib")
        np.savez(save_dir / f"{safe}_if_scaler.npz", mean=mean, std=std)
        company_index[safe] = soc
        company_models[soc] = (m, mean, std)

    m_global, mean_global, std_global = _ta.train_isolation_forest(df, cfg)
    joblib.dump(m_global, save_dir / "isolation_forest.joblib")
    np.savez(save_dir / "if_scaler.npz", mean=mean_global, std=std_global)

    # ── Scorer chaque jour avec le modèle le plus spécifique disponible ────────
    all_scored: list[pd.DataFrame] = []
    for (soc, line), grp in df.groupby(["societe", "line"]):
        if (soc, line) in line_models:
            m, mean, std = line_models[(soc, line)]
            tier = "ligne"
        elif soc in company_models:
            m, mean, std = company_models[soc]
            tier = "societe"
        else:
            m, mean, std = m_global, mean_global, std_global
            tier = "global"
        scored = _ta.score_days(m, mean, std, grp)
        scored["model_tier"] = tier
        all_scored.append(scored)

    days_scored = pd.concat(all_scored, ignore_index=True)
    n_total = int(days_scored["anomaly"].sum())
    print(f"    total signalés : {n_total}/{len(days_scored)} ({100*n_total/len(days_scored):.1f}%)")

    # Contrôle direct de l'artefact corrigé : plus aucune ligne à ~100% signalée
    line_rates = days_scored.groupby(["societe", "line"])["anomaly"].mean()
    n_full = int((line_rates >= 0.9).sum())
    print(f"    lignes signalées à >=90% : {n_full} "
          f"(avant le passage par-ligne : les lignes tarifairement atypiques étaient à 100%)")

    joblib.dump(line_models, save_dir / "line_models.joblib")
    with open(save_dir / "company_models.json", "w") as f:
        json.dump(company_index, f, ensure_ascii=False)
    days_scored.to_parquet(save_dir / "days_scored.parquet", index=False)
    print(f"  -> Artefacts sauvegardés dans {save_dir}")

    if_mean_by_soc = {soc: (mean, std) for soc, (_, mean, std) in company_models.items()}
    return {"days": days_scored, "n_anomaly": int(n_total),
            "n_line_models": len(line_models), "if_mean_by_soc": if_mean_by_soc}


def load(save_dir: str | Path = SAVE_DIR) -> dict:
    import joblib

    save_dir = Path(save_dir)

    # Modèles par ligne (absents si les artefacts datent d'avant le passage par-ligne --
    # le repli société/global couvre alors tout, comme avant)
    line_models: dict = {}
    lm_path = save_dir / "line_models.joblib"
    if lm_path.exists():
        line_models = joblib.load(lm_path)

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
    print(f"Modèles d'anomalie billetterie chargés ({len(line_models)} ligne(s) + "
          f"{len(if_models)-1} société(s) + repli global)")
    # Phase 2, voir train_stations/load_stations -- artefacts SÉPARÉS (sous-dossier
    # "stations"), chargés ici sous la clé "stations" pour rester DANS le même module
    # ("ticket_anomaly") côté API plutôt que d'enregistrer un module ENABLED_MODULES de
    # plus. Absent si train_stations() n'a jamais tourné -- load_stations() gère déjà ce
    # cas (retourne des dicts/DataFrame vides, pas une exception).
    stations = load_stations(save_dir)
    return {"line_models": line_models, "if_models": if_models, "days": days,
            "stations": stations}


def score(models: dict, day_rows: pd.DataFrame) -> pd.DataFrame:
    """Score de nouveaux jours billetterie -- modèle de LA LIGNE si disponible, sinon
    société, sinon global (même hiérarchie qu'à l'entraînement)."""
    day_rows = _ta.build_features(day_rows)
    line_models = models.get("line_models", {})
    if_models = models.get("if_models", {})
    parts = []
    for (soc, line), grp in day_rows.groupby(["societe", "line"]):
        if (soc, line) in line_models:
            m, mean, std = line_models[(soc, line)]
        else:
            m, mean, std = if_models.get(soc, if_models.get("_global"))
        parts.append(_ta.score_days(m, mean, std, grp))
    return pd.concat(parts, ignore_index=True) if parts else day_rows


def explain(models: dict, scored: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des raisons explicables en langage clair (voir `_ta.explain_days`).

    Les z-scores des raisons utilisent les stats du modèle le plus spécifique qui a servi
    au scoring : (société, ligne) d'abord, société sinon, global en dernier -- pour que
    « anormal » dans le texte veuille dire « anormal pour CETTE ligne » dès que possible.
    L'historique complet (`models['days']`) fournit en plus les médianes par ligne/bus et
    le taux d'anomalie de la ligne (contexte de jugement affiché par le tableau de bord).
    """
    line_models = models.get("line_models", {})
    if_models = models.get("if_models", {})
    _, g_mean, g_std = if_models.get("_global", (None, None, None))

    if_mean_by_key: dict = {}
    for (soc, line) in scored.groupby(["societe", "line"]).groups.keys():
        if (soc, line) in line_models:
            _, mean, std = line_models[(soc, line)]
            if_mean_by_key[(soc, line)] = (mean, std)
    for soc in scored["societe"].unique():
        if soc in if_models:
            _, mean, std = if_models[soc]
        else:
            mean, std = g_mean, g_std
        if_mean_by_key[soc] = (mean, std)

    return _ta.explain_days(scored, if_mean_by_key, baseline=models.get("days"))


# ── Phase 2 : billetterie PAR ARRÊT D'ORIGINE ───────────────────────────────────────────
# Même hiérarchie de modèles que train()/load()/score()/explain() ci-dessus -- la normale
# billetterie reste une propriété de la LIGNE, pas de l'arrêt individuel (trop peu
# d'historique par arrêt pour un modèle dédié, voir TicketAnomalyConfig.min_records_per_line)
# -- seule la source de données change (tickets_station_daily au lieu de tickets_daily,
# `station` au lieu de `bus`). Artefacts sauvegardés dans un sous-dossier séparé
# (`{save_dir}/stations/`) pour ne jamais toucher les modèles bus-jour existants.
STATIONS_SUBDIR = "stations"


def train_stations(db_conn, save_dir: str | Path = SAVE_DIR) -> dict:
    """Entraîne les Isolation Forests billetterie PAR ARRÊT D'ORIGINE : par (société,
    ligne) si assez de jours-arrêts, par société en repli, global en dernier recours.

    Sauvegarde (dans `{save_dir}/stations/`) :
      line_models.joblib / company_models.json / isolation_forest.joblib / if_scaler.npz
      -- même schéma que train(), voir cette docstring pour le détail de chaque fichier.
      station_days_scored.parquet -- tous les jours-arrêts scorés (+model_tier)
    """
    import joblib

    save_dir = Path(save_dir) / STATIONS_SUBDIR
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = _ta.TicketAnomalyConfig()

    df = _ta.build_features(_ta.load_tickets_station_daily(db_conn))
    print(f"  jours-arrêts billetterie : {len(df):,}")

    line_models: dict[tuple, tuple] = {}
    for (soc, line), grp in df.groupby(["societe", "line"]):
        if len(grp) < cfg.min_records_per_line:
            continue
        m, mean, std = _ta.train_isolation_forest(grp, cfg)
        line_models[(soc, line)] = (m, mean, std)
    n_line_days = int(df.set_index(["societe", "line"]).index.isin(line_models.keys()).sum())
    print(f"    modèles par ligne : {len(line_models)} lignes "
          f"({n_line_days}/{len(df)} jours-arrêts couverts) ; le reste replie société/global")

    company_index: dict[str, str] = {}
    company_models: dict[str, tuple] = {}
    for soc in sorted(df["societe"].unique()):
        soc_days = df[df["societe"] == soc]
        if len(soc_days) < cfg.min_records_per_company:
            print(f"    {soc}: {len(soc_days)} jours-arrêts -- trop peu, repli sur modèle global")
            continue
        m, mean, std = _ta.train_isolation_forest(soc_days, cfg)
        safe = _safe(soc)
        joblib.dump(m, save_dir / f"{safe}_isolation_forest.joblib")
        np.savez(save_dir / f"{safe}_if_scaler.npz", mean=mean, std=std)
        company_index[safe] = soc
        company_models[soc] = (m, mean, std)

    m_global, mean_global, std_global = _ta.train_isolation_forest(df, cfg)
    joblib.dump(m_global, save_dir / "isolation_forest.joblib")
    np.savez(save_dir / "if_scaler.npz", mean=mean_global, std=std_global)

    all_scored: list[pd.DataFrame] = []
    for (soc, line), grp in df.groupby(["societe", "line"]):
        if (soc, line) in line_models:
            m, mean, std = line_models[(soc, line)]
            tier = "ligne"
        elif soc in company_models:
            m, mean, std = company_models[soc]
            tier = "societe"
        else:
            m, mean, std = m_global, mean_global, std_global
            tier = "global"
        scored = _ta.score_days(m, mean, std, grp)
        scored["model_tier"] = tier
        all_scored.append(scored)

    station_days_scored = pd.concat(all_scored, ignore_index=True)
    n_total = int(station_days_scored["anomaly"].sum())
    print(f"    total signalés : {n_total}/{len(station_days_scored)} "
          f"({100*n_total/len(station_days_scored):.1f}%)")

    joblib.dump(line_models, save_dir / "line_models.joblib")
    with open(save_dir / "company_models.json", "w") as f:
        json.dump(company_index, f, ensure_ascii=False)
    station_days_scored.to_parquet(save_dir / "station_days_scored.parquet", index=False)
    print(f"  -> Artefacts sauvegardés dans {save_dir}")

    return {"days": station_days_scored, "n_anomaly": int(n_total),
            "n_line_models": len(line_models)}


def load_stations(save_dir: str | Path = SAVE_DIR) -> dict:
    """Charge les modèles billetterie PAR ARRÊT (voir train_stations) -- même structure
    que load(), lit depuis `{save_dir}/stations/`."""
    import joblib

    save_dir = Path(save_dir) / STATIONS_SUBDIR
    if not save_dir.exists():
        return {"line_models": {}, "if_models": {}, "days": pd.DataFrame()}

    line_models: dict = {}
    lm_path = save_dir / "line_models.joblib"
    if lm_path.exists():
        line_models = joblib.load(lm_path)

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

    days = pd.read_parquet(save_dir / "station_days_scored.parquet")
    print(f"Modèles d'anomalie billetterie PAR ARRÊT chargés ({len(line_models)} ligne(s) + "
          f"{len(if_models)-1} société(s) + repli global)")
    return {"line_models": line_models, "if_models": if_models, "days": days}


def score_stations(models: dict, station_rows: pd.DataFrame) -> pd.DataFrame:
    """Score de nouveaux jours-arrêts billetterie -- même hiérarchie que score()."""
    station_rows = _ta.build_features(station_rows)
    line_models = models.get("line_models", {})
    if_models = models.get("if_models", {})
    parts = []
    for (soc, line), grp in station_rows.groupby(["societe", "line"]):
        if (soc, line) in line_models:
            m, mean, std = line_models[(soc, line)]
        else:
            m, mean, std = if_models.get(soc, if_models.get("_global"))
        parts.append(_ta.score_days(m, mean, std, grp))
    return pd.concat(parts, ignore_index=True) if parts else station_rows


def explain_stations(models: dict, scored: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des raisons explicables pour les jours-arrêts (voir `_ta.explain_days`,
    `group_key="station"`) -- même logique que explain()."""
    line_models = models.get("line_models", {})
    if_models = models.get("if_models", {})
    _, g_mean, g_std = if_models.get("_global", (None, None, None))

    if_mean_by_key: dict = {}
    for (soc, line) in scored.groupby(["societe", "line"]).groups.keys():
        if (soc, line) in line_models:
            _, mean, std = line_models[(soc, line)]
            if_mean_by_key[(soc, line)] = (mean, std)
    for soc in scored["societe"].unique():
        if soc in if_models:
            _, mean, std = if_models[soc]
        else:
            mean, std = g_mean, g_std
        if_mean_by_key[soc] = (mean, std)

    return _ta.explain_days(scored, if_mean_by_key, baseline=models.get("days"),
                            group_key="station")
