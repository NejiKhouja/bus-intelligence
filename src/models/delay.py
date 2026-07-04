"""Module de retard — entraîner, sauvegarder, charger, servir.

Cycle de vie complet pour le Module 1 :
  train()       -> HistGBM (global + par société) + LSTM + Prophet -> models/delay/
  load()        -> charge tous les artefacts depuis le disque
  predict_eta() -> table ETA glissante pour un bus en direct
  forecast()    -> prévision de retard sur 30 jours pour une ligne (Prophet)

HistGBM par société (2026-07-03)
    Mesuré, pas supposé : un modèle HistGBM DÉDIÉ par société n'aide QUE si cette société a
    >= MIN_TRIPS_COMPANY lignes d'entraînement (voir la constante). En dessous (ex.
    SRT.ELGOUAFEL, 1 228 lignes), un modèle dédié RÉGRESSE par rapport au modèle global
    (5.71->6.59 min de MAE) -- pas assez de données pour apprendre un pattern propre à cette
    société sans surapprendre. Au-dessus (TCV, S.R.T.K, S.T.S), un modèle dédié améliore
    nettement la MAE (TCV 1.62->1.46, S.R.T.K 4.17->4.10, MAE globale 3.12->3.02). D'où le
    repli automatique sur le modèle global pour toute société sous le seuil, plutôt qu'un
    modèle dédié systématique.

Notes d'ingénierie ML
----------------------
Division entraînement/test
    La division est par JOUR (chronologique), pas aléatoire. Utiliser une division aléatoire
    ferait fuir les schémas futurs dans l'entraînement — un bus-jour en mars apparaîtrait
    à la fois dans train et test. On utilise 80% des jours pour l'entraînement, 20% pour le test.

Ensemble de validation (LSTM uniquement)
    Dans la portion d'entraînement, on sépare les 10 derniers % de séquences comme ensemble
    de validation temporel pour l'arrêt anticipé. C'est les DERNIERS 10 %, pas un échantillon
    aléatoire, pour la même raison de fuite.

Normalisation des caractéristiques (LSTM uniquement)
    Les caractéristiques brutes s'étendent sur des échelles très différentes : delay_min (-120..+120),
    elapsed_min (0..600), dep_hour (0..23). Sans normalisation, le gradient du LSTM est dominé
    par la caractéristique à haute variance (elapsed_min) et les autres sont sous-entraînées.
    On ajuste un StandardScaler sur X_train et on applique les MÊMES statistiques à val, test
    et inférence. Les statistiques du scaler sont sauvegardées sur disque avec les poids du modèle.

HistGBM vs LSTM
    HistGBM n'a pas besoin de normalisation (les divisions d'arbre sont invariantes par rapport
    à l'échelle). Il égale ou surpasse souvent le LSTM sur les données tabulaires à taille
    de jeu de données modérée (~100k échantillons). Le LSTM apporte de la valeur quand
    l'HISTORIQUE complet du trajet importe plus que l'état à l'arrêt actuel seul.

SMOTE / équilibrage des classes
    Non applicable. Les deux modèles résolvent une tâche de RÉGRESSION (prédire des minutes
    de retard, une valeur continue). SMOTE est une technique pour la CLASSIFICATION
    déséquilibrée — il génère des échantillons synthétiques de la classe minoritaire pour
    rééquilibrer les comptes d'étiquettes. Il n'y a pas d'étiquettes de classe ici.

Prophet
    Un modèle par combinaison (societe, line, dir) ajusté sur le retard moyen quotidien.
    Prophet gère automatiquement la saisonnalité hebdomadaire et produit des intervalles
    d'incertitude calibrés — utiles pour la planification des horaires.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import delay as _dl
from src.data import weather as _wx

SAVE_DIR = Path("models/delay")

# En dessous de ce nombre de lignes d'entraînement, un HistGBM DÉDIÉ à la société régresse
# par rapport au modèle global (mesuré : SRT.ELGOUAFEL, 1 228 lignes, MAE 5.71->6.59) --
# repli sur le modèle global. Au-dessus (ex. TCV, S.R.T.K, S.T.S), un modèle dédié améliore
# nettement la MAE (TCV 1.62->1.46, S.R.T.K 4.17->4.10) sans dégrader les autres sociétés
# testées. Voir docs/DATA_PIPELINE_REPORT.md pour les chiffres complets avant/après.
MIN_TRIPS_COMPANY = 3000


def _safe(name: str) -> str:
    """Nom de fichier sûr pour un nom d'opérateur."""
    return "".join(c if c.isalnum() else "_" for c in str(name))


# Entraînement
def train(foundation_path: str | Path,
          save_dir: str | Path = SAVE_DIR,
          *,
          epochs: int = 30,
          hidden: int = 64,
          n_layers: int = 2,
          patience: int = 5) -> dict:
    """Entraîne HistGBM + LSTM + Prophet sur le jeu de données de fondation complet.

    Sauvegarde tous les artefacts dans save_dir. Retourne dict avec modèles entraînés + métriques.
    """
    import joblib
    import torch
    from sklearn.metrics import mean_absolute_error

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "prophet").mkdir(exist_ok=True)

    # ingénierie des caractéristiques 
    print("  Chargement de la fondation...")
    cfg = _dl.DelayConfig()
    df = _dl.load_foundation(foundation_path)
    m = _dl.add_daytype(_dl.with_elapsed(df, cfg))

    weather = _wx.load_day_weather()
    m = _dl.add_weather(m, weather)
    if weather is not None:
        coverage = 100 * m["rain_frac"].notna().mean()
        print(f"  Signal météo (rain_frac) : {coverage:.0f}% des lignes couvertes "
              f"(cache: {_wx.DEFAULT_CACHE_PATH})")
    else:
        print(f"  Signal météo (rain_frac) : cache absent ({_wx.DEFAULT_CACHE_PATH}) -- "
              f"rain_frac=NaN partout. Lancer weather.build_day_weather_cache() pour l'activer.")

    baseline = _dl.build_baseline(m, cfg)
    d = _dl.add_daytype(_dl.with_delay(m, baseline, cfg))
    roll = _dl.rolling_table(d)

    # Division chronologique 80/20 — pas de mélange aléatoire pour éviter la fuite
    days = np.sort(roll["day"].unique())
    cut_day = days[int(0.8 * len(days))]
    tr = roll[roll["day"] < cut_day]
    te = roll[roll["day"] >= cut_day]
    print(f"  Division : train={len(tr):,} lignes (jours<{cut_day})  "
          f"test={len(te):,} lignes (jours>={cut_day})")

    # HistGBM (global + par société) 
    # Les modèles à arbres sont invariants par rapport à l'échelle — pas de normalisation nécessaire.
    # Les catégorielles (societe, line, dir) gérées nativement par HistGBM.
    print("  Entraînement de HistGBM (global + par société si assez de données)...")
    hgbm = _dl.train_rolling_model(tr)
    hgbm_mae = mean_absolute_error(te[_dl.TARGET], hgbm.predict(_dl._design(te, hgbm.feature_names_used_)))
    print(f"    MAE test (global) : {hgbm_mae:.2f} min")
    joblib.dump(hgbm, save_dir / "hgbm.joblib")

    # Modèle dédié par société -- seulement si assez de lignes d'entraînement (voir
    # MIN_TRIPS_COMPANY) pour ne pas régresser par rapport au modèle global sur les petites
    # sociétés (mesuré, pas supposé).
    hgbm_company_index: dict[str, str] = {}
    for soc, grp in tr.groupby("societe"):
        if len(grp) < MIN_TRIPS_COMPANY:
            continue
        m_soc = _dl.train_rolling_model(grp)
        safe = _safe(soc)
        joblib.dump(m_soc, save_dir / f"{safe}_hgbm.joblib")
        hgbm_company_index[safe] = soc
    with open(save_dir / "hgbm_company_models.json", "w") as f:
        json.dump(hgbm_company_index, f, ensure_ascii=False)

    # MAE par société en TEST -- dédié si entraîné, repli global sinon -- pour vérifier
    # concrètement qu'aucune société ne régresse (pas juste supposer que ça aide).
    for soc, grp in te.groupby("societe"):
        safe = _safe(soc)
        if safe in hgbm_company_index:
            m_soc = joblib.load(save_dir / f"{safe}_hgbm.joblib")
            pred = m_soc.predict(_dl._design(grp, m_soc.feature_names_used_))
            tag = "dédié"
        else:
            pred = hgbm.predict(_dl._design(grp, hgbm.feature_names_used_))
            tag = "repli global"
        print(f"    {soc}: MAE={mean_absolute_error(grp[_dl.TARGET], pred):.2f} min "
              f"({len(grp)} lignes test, {tag})")

    baseline.to_parquet(save_dir / "baseline.parquet", index=False)

    # LSTM
    print(f"  Entraînement du LSTM ({epochs} époques, patience={patience})...")
    X, _, y = _dl.build_lstm_sequences(roll)
    day_arr = roll["day"].values
    X_tr_raw, y_tr = X[day_arr < cut_day], y[day_arr < cut_day]
    X_te_raw, y_te = X[day_arr >= cut_day], y[day_arr >= cut_day]

    # Ajuster le scaler sur les données d'entraînement UNIQUEMENT, puis appliquer à toutes les divisions
    feat_mean, feat_std = _dl.fit_lstm_scaler(X_tr_raw)
    X_tr = _dl.scale_sequences(X_tr_raw, feat_mean, feat_std)
    X_te = _dl.scale_sequences(X_te_raw, feat_mean, feat_std)

    lstm = _dl.train_lstm_delay(X_tr, y_tr, hidden=hidden, n_layers=n_layers,
                                epochs=epochs, lr=1e-3, batch=256, patience=patience)
    lstm_mae = mean_absolute_error(y_te, _dl.predict_lstm(lstm, X_te))
    print(f"    MAE test : {lstm_mae:.2f} min")

    torch.save(lstm.state_dict(), save_dir / "lstm_delay.pt")
    # Sauvegarder les statistiques du scaler avec les poids — l'inférence DOIT utiliser la même transformation
    np.savez(save_dir / "lstm_scaler.npz", mean=feat_mean, std=feat_std)
    with open(save_dir / "lstm_config.json", "w") as f:
        json.dump({"hidden": hidden, "n_layers": n_layers,
                   "n_feats": X.shape[2], "max_len": 30}, f)

    # Prophet (un modèle par ligne/dir)
    # Entraîné sur le jeu de données COMPLET (toutes les dates) — Prophet est un prédicteur
    # de séries temporelles, pas un modèle supervisé ; il capture la saisonnalité à partir de l'historique.
    print("  Ajustement des modèles Prophet...")
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
    print(f"    {prophet_count} modèles Prophet sauvegardés")

    print(f"  -> Artefacts de retard sauvegardés dans {save_dir}")
    return {
        "hgbm": hgbm, "lstm": lstm, "baseline": baseline,
        "feat_mean": feat_mean, "feat_std": feat_std,
        "hgbm_mae": hgbm_mae, "lstm_mae": lstm_mae,
        "prophet_count": prophet_count,
    }


# Chargement
def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Charge tous les artefacts de retard entraînés depuis save_dir.

    Retourne dict : hgbm (modèle global, rétrocompatible), hgbm_models (dict
                    société -> modèle dédié, + "_global"), lstm, feat_mean, feat_std,
                    baseline, prophet (clé par societe_line_dir).
    """
    import joblib
    import torch

    save_dir = Path(save_dir)
    hgbm     = joblib.load(save_dir / "hgbm.joblib")
    baseline = pd.read_parquet(save_dir / "baseline.parquet")

    hgbm_models: dict = {"_global": hgbm}
    company_index_path = save_dir / "hgbm_company_models.json"
    if company_index_path.exists():
        with open(company_index_path) as f:
            hgbm_company_index = json.load(f)
        for safe, soc in hgbm_company_index.items():
            m_path = save_dir / f"{safe}_hgbm.joblib"
            if m_path.exists():
                hgbm_models[soc] = joblib.load(m_path)

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

    n_dedicated = len(hgbm_models) - 1
    print(f"Modèles de retard chargés : HistGBM ({n_dedicated} société(s) dédiée(s) + repli global) "
          f"+ LSTM + {len(prophets)} modèles Prophet")
    return {"hgbm": hgbm, "hgbm_models": hgbm_models, "lstm": lstm, "baseline": baseline,
            "feat_mean": feat_mean, "feat_std": feat_std, "prophet": prophets}


# Service
def predict_eta(models: dict, *,
                societe: str, line: str, direction: str,
                dep_time: str,
                current_seq: int,
                current_delay_min: float,
                model_type: str = "hgbm") -> pd.DataFrame:
    """Table ETA pour tous les arrêts restants étant donné l'état en direct d'un bus.

    model_type : 'hgbm' (défaut — utilise le modèle DÉDIÉ de la société si disponible
                 (voir MIN_TRIPS_COMPANY dans train()), repli sur le modèle global sinon)
                 'lstm' (utilise l'historique complet du trajet ; meilleur avec plus d'époques/GPU)
    Retourne DataFrame : seq, expected_min, pred_delay_min, eta.
    """
    if model_type == "lstm":
        return _dl.serve_eta_lstm(
            models["lstm"], models["baseline"],
            societe=societe, line=line, direction=direction,
            dep_time=dep_time, current_seq=current_seq,
            current_delay_min=current_delay_min,
            scaler_mean=models["feat_mean"], scaler_std=models["feat_std"],
        )
    hgbm_models = models.get("hgbm_models", {"_global": models["hgbm"]})
    hgbm_model = hgbm_models.get(societe, hgbm_models["_global"])
    return _dl.serve_eta(
        hgbm_model, models["baseline"],
        societe=societe, line=line, direction=direction,
        dep_time=dep_time, current_seq=current_seq,
        current_delay_min=current_delay_min,
    )


def forecast(models: dict, *,
             societe: str, line: str, direction: str,
             periods: int = 30) -> pd.DataFrame | None:
    """Prévision du retard moyen quotidien sur 30 jours pour une ligne. Retourne None si aucun modèle."""
    key = f"{societe}_{line}_{direction}"
    pm  = models["prophet"].get(key)
    if pm is None:
        return None
    return _dl.prophet_forecast(pm, periods=periods)
