"""Module de repli GPS — entraîner, sauvegarder, charger, servir.
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


# Entraînement
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


# Chargement
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


# Service
def _kalman_params() -> dict:
    """Charge les paramètres Kalman réglés (r_std, q_v) si présents, sinon défauts.

    Écrits par la recherche aléatoire dans 02_gps_fallback.ipynb.
    """
    p = SAVE_DIR / "kalman_params.json"
    if p.exists():
        try:
            with open(p) as f:
                cfg = json.load(f)
            return {"r_std": float(cfg.get("r_std", 100.0)),
                    "q_v": float(cfg.get("q_v", 0.5))}
        except Exception:
            pass
    return {"r_std": 100.0, "q_v": 0.5}


def run_kalman(g: pd.DataFrame, route_len: float) -> pd.DataFrame:
    """Applique le filtre de Kalman à un DataFrame de pings projetés.

    Doit être appelé avant predict_position pour remplir les colonnes ks/kv/kp.
    Utilise les paramètres réglés par recherche aléatoire s'ils sont disponibles.
    """
    p = _kalman_params()
    return _fb.kalman_filter_track(g, route_len, r_std=p["r_std"], q_v=p["q_v"])


def predict_position(models: dict,
                     g_filtered: pd.DataFrame,
                     t_query: pd.Timestamp,
                     stops: pd.DataFrame) -> dict | None:
    """Meilleure estimation de position pendant un écart GPS — **Kalman pur**.
    """
    return _fb.kalman_fallback(g_filtered, t_query, stops)
