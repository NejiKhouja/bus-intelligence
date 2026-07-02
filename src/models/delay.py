"""Module de retard — entraîner, sauvegarder, charger, servir.

Cycle de vie complet pour le Module 1 :
  train()       -> HistGBM + LSTM + Prophet -> artefacts sauvegardés dans models/delay/
  load()        -> charge tous les artefacts depuis le disque
  predict_eta() -> table ETA glissante pour un bus en direct
  forecast()    -> prévision de retard sur 30 jours pour une ligne (Prophet)

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


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── ingénierie des caractéristiques ──────────────────────────────────────
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

    # ── HistGBM ──────────────────────────────────────────────────────────────
    # Les modèles à arbres sont invariants par rapport à l'échelle — pas de normalisation nécessaire.
    # Les catégorielles (line, dir) gérées nativement par HistGBM.
    print("  Entraînement de HistGBM...")
    hgbm = _dl.train_rolling_model(tr)
    hgbm_mae = mean_absolute_error(te[_dl.TARGET], hgbm.predict(_dl._design(te)))
    print(f"    MAE test : {hgbm_mae:.2f} min")
    joblib.dump(hgbm, save_dir / "hgbm.joblib")

    baseline.to_parquet(save_dir / "baseline.parquet", index=False)

    # ── LSTM ─────────────────────────────────────────────────────────────────
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

    # ── Prophet (un modèle par ligne/dir) ────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Charge tous les artefacts de retard entraînés depuis save_dir.

    Retourne dict : hgbm, lstm, feat_mean, feat_std, baseline,
                    prophet (clé par societe_line_dir).
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

    print(f"Modèles de retard chargés : HistGBM + LSTM + {len(prophets)} modèles Prophet")
    return {"hgbm": hgbm, "lstm": lstm, "baseline": baseline,
            "feat_mean": feat_mean, "feat_std": feat_std, "prophet": prophets}


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

def predict_eta(models: dict, *,
                societe: str, line: str, direction: str,
                dep_time: str,
                current_seq: int,
                current_delay_min: float,
                model_type: str = "hgbm") -> pd.DataFrame:
    """Table ETA pour tous les arrêts restants étant donné l'état en direct d'un bus.

    model_type : 'hgbm' (défaut — plus rapide, même précision à la taille d'entraînement actuelle)
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
    return _dl.serve_eta(
        models["hgbm"], models["baseline"],
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
