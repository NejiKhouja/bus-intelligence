"""Module de repli GPS — entraîner, sauvegarder, charger, servir.

Cycle de vie complet pour le Module 2 :
  train()            -> modèle de correction LSTM -> sauvegardé dans models/fallback/
  load()             -> charge les artefacts depuis le disque
  predict_position() -> meilleure estimation de position pendant un écart GPS

Notes d'ingénierie ML
----------------------
Le filtre de Kalman n'a PAS de paramètres apprenables — c'est un estimateur en ligne
qui s'exécute à l'inférence sur le flux de pings en direct de chaque bus. Pas d'entraînement nécessaire.

Correction LSTM
    Le LSTM apprend à corriger l'estimation s de Kalman en utilisant le SCHÉMA des
    valeurs récentes [ks, kv, kp, speed]. Un bus qui s'approche d'un stationnement en terminus,
    ou qui monte une pente à vitesse réduite, suit un profil caractéristique qu'un modèle
    Kalman linéaire ne peut pas capturer.

Stratégie des données d'entraînement
    On s'entraîne sur des pings GPS de PLUSIEURS bus-jours extraits directement de MongoDB.
    Utiliser un seul trajet donne un modèle qui sur-ajuste la géométrie spécifique de ce trajet
    et les schémas de trafic. Plus de trajets = meilleure généralisation à travers différents
    jours, heures et bus sur la même ligne.

    Concrètement : on charge tous les bus pour une ligne donnée sur plusieurs jours du calendrier,
    on les projette sur la route, on exécute le filtre de Kalman, et on regroupe toutes les
    fenêtres sans écart en un seul ensemble d'entraînement.

Normalisation des caractéristiques
    Les caractéristiques [ks, kv, kp, speed] sont normalisées avec moyenne/écart-type ajustés
    sur les pings d'entraînement uniquement (pas de fuite). Les mêmes statistiques sont
    sauvegardées et appliquées à l'inférence.

Division entraînement/test
    Non appliquée ici : la correction LSTM est un assistant de régression pour le filtre de
    Kalman (elle corrige les estimations à partir de l'historique récent) et est évaluée via
    l'expérience de gap synthétique dans le notebook, pas un ensemble étiqueté séparé.
    Si des paires erreur-gap étiquetées étaient disponibles, une division par jour s'appliquerait.

SMOTE / équilibrage des classes
    Non applicable — tâche de régression, pas d'étiquettes de classe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import fallback as _fb
from src.data import foundation as _fdn

SAVE_DIR = Path("models/fallback")

_N_FEATS = len(_fb._LSTM_CORR_FEATS)   # ["ks", "kv", "kp", "speed"]
_HIDDEN  = 32


def _make_corr_lstm(n_feats: int = _N_FEATS, hidden: int = _HIDDEN):
    """Petit LSTM qui lit l'historique Kalman récent et produit une valeur s corrigée."""
    import torch.nn as nn

    class CorrLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feats, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)

    return CorrLSTM()


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train(save_dir: str | Path = SAVE_DIR,
          *,
          line: str = "209",
          societe: str = "S.R.T.K",
          n_days: int = 5,
          window: int = 10,
          epochs: int = 30) -> dict:
    """Entraîne la correction LSTM sur plusieurs bus-jours pour une meilleure généralisation.

    Extrait les pings GPS bruts de MongoDB pour les `n_days` jours du calendrier les plus
    récents disponibles pour la ligne donnée, les projette, exécute Kalman, et entraîne
    le LSTM sur les fenêtres sans écart regroupées.

    Paramètres
    ----------
    n_days  : nombre de bus-jours à partir desquels collecter des données d'entraînement
    window  : fenêtre de regard en arrière fournie au LSTM (nombre de pings récents)
    epochs  : époques d'entraînement (plus = meilleur mais plus lent ; 30 est une valeur sûre)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.data.db import get_db

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    db_winicari = get_db("winicari")
    db_gps      = get_db("Historique_pos")
    cfg         = _fdn.Config()
    usable      = _fdn.build_usable_lines(db_winicari, cfg)
    stops       = usable[(line, societe)]

    # Découvrir les bus-jours disponibles dans MongoDB (noms de collection = 'd{YYYYMMDD}')
    all_day_cols = sorted(db_gps.list_collection_names(), reverse=True)
    day_cols = [d for d in all_day_cols if d.startswith("d")][:n_days]
    print(f"  Collecte de pings depuis {len(day_cols)} bus-jours : {day_cols}")

    all_feats, all_targets = [], []

    for day in day_cols:
        # Découvrir les codes de bus qui ont circulé sur cette ligne+jour
        sample = db_gps[day].distinct("bus.code",
                                      {"service.codeLigne": line})
        if not sample:
            continue
        for bus_id in sample[:3]:          # limiter à 3 bus par jour pour rester rapide
            try:
                raw = _fdn.load_pings(db_gps, day, line, int(bus_id))
                if len(raw) < 50:
                    continue
                g, route_len = _fdn.project_to_route(
                    _fdn.clean_pings(raw, cfg), stops, cfg)
                g_kf = _fb.kalman_filter_track(g, route_len)

                # Pings sans écart uniquement.
                # La cible est le RÉSIDU (s_true - ks), pas s absolu.
                # POURQUOI : s brut s'étend sur 0..192 000 m ; prédire des valeurs absolues depuis
                # des caractéristiques normalisées cause une perte de ~10^11 m2 (le modèle prédit
                # le milieu de route). L'estimation Kalman ks est déjà proche de s_true ;
                # le LSTM n'a qu'à apprendre le petit terme de correction (+/-500 m).
                non_gap = g_kf[~g_kf["signal_gap"]].reset_index(drop=True)
                feats   = non_gap[_fb._LSTM_CORR_FEATS].values.astype(np.float32)
                targets = (non_gap["s"] - non_gap["ks"]).values.astype(np.float32)
                all_feats.append(feats)
                all_targets.append(targets)
            except Exception:
                continue

    if not all_feats:
        raise RuntimeError("Aucun ping utilisable trouvé — vérifier la ligne/societe ou MongoDB.")

    feats   = np.concatenate(all_feats,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    print(f"  Total de pings sans écart regroupés : {len(feats):,}")
    print(f"  Résidu (s_true - ks) : moyenne={targets.mean():.1f} m  "
          f"écart-type={targets.std():.1f} m")

    # Normalisation des caractéristiques — ajuster sur TOUS les pings d'entraînement collectés
    mean = feats.mean(axis=0).astype(np.float32)
    std  = (feats.std(axis=0) + 1e-6).astype(np.float32)
    feats_n = (feats - mean) / std

    # Construire des séquences à fenêtre glissante
    xs, ys = [], []
    for i in range(window, len(feats_n)):
        xs.append(feats_n[i - window:i])
        ys.append(targets[i])
    if not xs:
        raise RuntimeError("Pas assez de pings pour construire des séquences.")

    X = np.stack(xs).astype(np.float32)
    Y = np.array(ys, dtype=np.float32)
    print(f"  Séquences d'entraînement : {len(X):,}  fenêtre={window}")

    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)),
                        batch_size=128, shuffle=True)

    model   = _make_corr_lstm(_N_FEATS, _HIDDEN)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            total += loss_fn(model(xb), yb).item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"    époque {ep+1}/{epochs}  perte={total/len(X):.2f} m²")

    model.eval()
    torch.save(model.state_dict(), save_dir / "lstm_corr.pt")
    np.savez(save_dir / "lstm_corr_stats.npz", mean=mean, std=std)
    with open(save_dir / "lstm_corr_config.json", "w") as f:
        json.dump({"window": window, "n_feats": _N_FEATS, "hidden": _HIDDEN,
                   "n_days": n_days, "n_pings": int(len(feats))}, f)

    print(f"  -> Artefacts de repli sauvegardés dans {save_dir}")
    return {"model": model, "mean": mean, "std": std, "window": window}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load(save_dir: str | Path = SAVE_DIR) -> dict:
    """Charge le modèle de correction LSTM entraîné.

    Retourne dict : model, mean, std, window.
    """
    import torch

    save_dir = Path(save_dir)
    with open(save_dir / "lstm_corr_config.json") as f:
        cfg = json.load(f)

    model = _make_corr_lstm(cfg["n_feats"], cfg["hidden"])
    model.load_state_dict(torch.load(save_dir / "lstm_corr.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    stats = np.load(save_dir / "lstm_corr_stats.npz")
    print(f"Correction LSTM de repli GPS chargée  "
          f"(fenêtre={cfg['window']}, entraîné sur {cfg.get('n_pings',0):,} pings)")
    return {"model": model, "mean": stats["mean"], "std": stats["std"],
            "window": cfg["window"]}


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

def run_kalman(g: pd.DataFrame, route_len: float) -> pd.DataFrame:
    """Applique le filtre de Kalman à un DataFrame de pings projetés.

    Doit être appelé avant predict_position pour remplir les colonnes ks/kv/kp.
    """
    return _fb.kalman_filter_track(g, route_len)


def predict_position(models: dict,
                     g_filtered: pd.DataFrame,
                     t_query: pd.Timestamp,
                     stops: pd.DataFrame) -> dict | None:
    """Meilleure estimation de position (Kalman + correction LSTM) pendant un écart GPS.

    Le LSTM prédit une correction résiduelle (s_true - ks). On l'ajoute à l'estimation
    Kalman pour obtenir la position finale : s_final = ks + correction_lstm.

    g_filtered doit être la sortie de run_kalman().
    Retourne dict : lat, lon, s_m (km), uncertainty_m, method — ou None si pas dans un écart.
    """
    import torch
    from src.data.fallback import s_to_latlon, _LSTM_CORR_FEATS

    t_arr = pd.to_datetime(g_filtered["t"])
    before = g_filtered[t_arr <= t_query]
    if len(before) < models["window"]:
        # Pas assez d'historique — retomber sur le Kalman pur
        return _fb.kalman_fallback(g_filtered, t_query, stops)

    recent = before.iloc[-models["window"]:][_LSTM_CORR_FEATS].values.astype("float32")
    recent_n = (recent - models["mean"]) / models["std"]

    with torch.no_grad():
        correction = float(models["model"](torch.tensor(recent_n[None]))[0])

    # Estimation Kalman à t_query (propagée depuis le dernier état filtré)
    last_row = before.iloc[-1]
    dt = (t_query - pd.Timestamp(last_row["t"])).total_seconds()
    ks_at_query = float(last_row["ks"] + last_row["kv"] * dt)

    s_final = float(np.clip(ks_at_query + correction, 0.0, g_filtered["ks"].max()))
    lat, lon = s_to_latlon(s_final, stops)

    return {
        "lat": lat, "lon": lon, "s_m": round(s_final / 1000, 2),
        "uncertainty_m": round(float(last_row["kp"]), 0),
        "method": "kalman+lstm",
        "ks_m": round(ks_at_query / 1000, 2),
        "correction_m": round(correction, 1),
    }
