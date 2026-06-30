"""Couche de repli GPS — estimations de position pendant les interruptions de signal.

Quand un bus perd le GPS pendant plus de `signal_gap_s` secondes, le tableau de bord
opérateur affiche le bus comme « disparu ». Cette couche comble l'écart avec une
estimation de position dérivée de la géométrie de la route.

Méthodes de base
----------------
linear_interp   Interpole s (distance le long de la route, en mètres) linéairement entre
                le dernier ping connu avant l'écart et le premier ping après.

dead_reckoning  Projette en avant depuis le dernier ping en utilisant sa vitesse rapportée.
                Utile quand aucun ping de récupération n'existe encore (bus actuellement dark).

Méthodes améliorées
-------------------
Filtre de Kalman  Suit l'état [s, vitesse] le long de la route. Chaque ping GPS est une
                  mesure bruitée ; pendant un écart seule l'étape de prédiction s'exécute,
                  donnant une estimation d'incertitude rigoureuse (la covariance croît).
                  Implémenté avec filterpy.KalmanFilter.

Correction LSTM   Après la prédiction Kalman, un LSTM entraîné sur des pings historiques
                  corrige l'estimation en utilisant les schémas de trafic appris (profils de
                  vitesse, comportement aux arrêts). Réduit le biais systématique que le
                  modèle Kalman linéaire ne peut pas capturer.

`kalman_filter_track` exécute le filtre de Kalman complet sur une séquence de pings projetés.
`kalman_fallback` interroge la piste filtrée à n'importe quel horodatage dans un écart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Assistants de géométrie de route
# ─────────────────────────────────────────────────────────────────────────────

def s_to_latlon(s_query: float, stops: pd.DataFrame) -> tuple[float, float]:
    """Convertit une distance de route (mètres) en (lat, lon) via la polyligne d'ancrage."""
    s_arr = stops["s_m"].values
    lat_arr = stops["lat"].values
    lon_arr = stops["lon"].values
    if s_query <= s_arr[0]:
        return float(lat_arr[0]), float(lon_arr[0])
    if s_query >= s_arr[-1]:
        return float(lat_arr[-1]), float(lon_arr[-1])
    i = int(np.searchsorted(s_arr, s_query)) - 1
    frac = (s_query - s_arr[i]) / (s_arr[i + 1] - s_arr[i])
    return (float(lat_arr[i] + frac * (lat_arr[i + 1] - lat_arr[i])),
            float(lon_arr[i] + frac * (lon_arr[i + 1] - lon_arr[i])))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance orthodromique en mètres entre deux points (lat, lon)."""
    R = 6_371_000.0
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p)
         * np.sin((lon2 - lon1) * p / 2) ** 2)
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des écarts
# ─────────────────────────────────────────────────────────────────────────────

def gap_table(g: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par écart de signal avec le contexte de route avant/après.

    Entrée : DataFrame de pings projetés (sortie de foundation.project_to_route).
    """
    g = g.reset_index(drop=True)
    rows = []
    for idx in g.index[g["signal_gap"]]:
        if idx == 0:
            continue
        before, after = g.iloc[idx - 1], g.iloc[idx]
        rows.append({
            "gap_idx": int(idx),
            "t_start": before["t"],
            "t_end": after["t"],
            "gap_s": float(after["gap_s"]),
            "gap_min": round(float(after["gap_s"]) / 60, 1),
            "s_start_km": round(float(before["s"]) / 1000, 1),
            "s_end_km": round(float(after["s"]) / 1000, 1),
            "dist_covered_km": round(abs(float(after["s"]) - float(before["s"])) / 1000, 1),
            "speed_before_kph": round(float(before["speed"]), 1),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Méthodes d'estimation
# ─────────────────────────────────────────────────────────────────────────────

def interp_position(t_query: pd.Timestamp, t0: pd.Timestamp, s0: float,
                    t1: pd.Timestamp, s1: float,
                    stops: pd.DataFrame) -> tuple[float, float, float]:
    """Interpolation linéaire de la distance de route pendant un écart → (lat, lon, s_m)."""
    total = (t1 - t0).total_seconds()
    frac = (t_query - t0).total_seconds() / total if total > 0 else 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    s_est = s0 + frac * (s1 - s0)
    lat, lon = s_to_latlon(s_est, stops)
    return lat, lon, s_est


def dead_reckon_position(t_query: pd.Timestamp, t0: pd.Timestamp, s0: float,
                         speed_kph: float, direction: int,
                         stops: pd.DataFrame) -> tuple[float, float, float]:
    """Projette en avant depuis la dernière vitesse connue → (lat, lon, s_m).

    direction : +1 pour ALLER (s croissant), -1 pour RETOUR.
    """
    dt = (t_query - t0).total_seconds()
    s_est = s0 + direction * (speed_kph / 3.6) * dt
    s_max = float(stops["s_m"].max())
    s_est = float(np.clip(s_est, 0.0, s_max))
    lat, lon = s_to_latlon(s_est, stops)
    return lat, lon, s_est


# ─────────────────────────────────────────────────────────────────────────────
# Production : meilleure estimation pour tout horodatage de requête
# ─────────────────────────────────────────────────────────────────────────────

def fallback_position(g: pd.DataFrame, t_query: pd.Timestamp,
                      stops: pd.DataFrame) -> dict | None:
    """Meilleure estimation de position pour un horodatage de requête qui tombe dans un écart.

    Retourne None si t_query n'est pas dans un écart.
    Retourne un dict avec les clés :
        lat_interp, lon_interp, s_interp   — interpolation linéaire (si ping de récupération connu)
        lat_dr, lon_dr, s_dr               — navigation à l'estime depuis la dernière vitesse connue
        gap_s                              — durée de l'écart en secondes
        method                             — 'interp' | 'dead_reckon' (recommandée)
    """
    g = g.reset_index(drop=True)
    t_arr = pd.to_datetime(g["t"])
    before_mask = t_arr <= t_query
    if not before_mask.any():
        return None
    i0 = int(np.where(before_mask)[0][-1])
    if i0 + 1 >= len(g):
        return None

    after = g.iloc[i0 + 1]
    if not bool(after["signal_gap"]):
        return None  # pas dans un écart

    before = g.iloc[i0]
    t0 = pd.Timestamp(before["t"])
    t1 = pd.Timestamp(after["t"])
    s0, s1 = float(before["s"]), float(after["s"])
    speed_kph = float(before["speed"])
    direction = int(np.sign(s1 - s0)) or 1

    lat_i, lon_i, s_i = interp_position(t_query, t0, s0, t1, s1, stops)
    lat_d, lon_d, s_d = dead_reckon_position(t_query, t0, s0, speed_kph, direction, stops)

    return {
        "lat_interp": lat_i, "lon_interp": lon_i, "s_interp": round(s_i / 1000, 2),
        "lat_dr": lat_d, "lon_dr": lon_d, "s_dr": round(s_d / 1000, 2),
        "gap_s": float(after["gap_s"]),
        "method": "interp",  # préférer interp quand le ping de récupération est connu
    }


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation : masquage synthétique
# ─────────────────────────────────────────────────────────────────────────────

def eval_fallback(g: pd.DataFrame, stops: pd.DataFrame,
                  mask_min: float = 3.0, n_samples: int = 200,
                  rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Évalue les deux méthodes en masquant synthétiquement mask_min minutes de pings.

    Pour chacun des n_samples fenêtres aléatoires :
      1. Faire semblant que le bus était dark pendant mask_min minutes à partir d'un ping aléatoire.
      2. Estimer la position au milieu de l'écart avec les deux méthodes.
      3. Mesurer l'erreur (mètres) par rapport à la vraie position GPS.

    Retourne un DataFrame avec les colonnes : err_interp_m, err_dr_m, gap_s, dt_into_gap_s.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mask_s = mask_min * 60
    g = g.reset_index(drop=True)
    t_unix = (pd.to_datetime(g["t"]).astype(np.int64) // 10 ** 9).values
    candidates = np.where(~g["signal_gap"].values)[0]
    candidates = candidates[candidates < len(g) - 5]
    if len(candidates) < 5:
        return pd.DataFrame()

    rows = []
    for _ in range(n_samples):
        i0 = int(rng.choice(candidates))
        t0_u = t_unix[i0]
        future = np.where(t_unix > t0_u + mask_s)[0]
        if len(future) == 0:
            continue
        i1 = int(future[0])
        if i1 <= i0 + 1:
            continue

        inside = g.iloc[i0 + 1:i1]
        if len(inside) == 0:
            continue
        mid = inside.iloc[len(inside) // 2]
        t_q = pd.Timestamp(mid["t"])
        true_lat, true_lon = float(mid["lat"]), float(mid["lon"])

        before, after = g.iloc[i0], g.iloc[i1]
        s0_v, s1_v = float(before["s"]), float(after["s"])
        t0_ts = pd.Timestamp(before["t"])
        t1_ts = pd.Timestamp(after["t"])
        speed_kph = float(before["speed"])
        direction = int(np.sign(s1_v - s0_v)) or 1

        lat_i, lon_i, _ = interp_position(t_q, t0_ts, s0_v, t1_ts, s1_v, stops)
        lat_d, lon_d, _ = dead_reckon_position(t_q, t0_ts, s0_v, speed_kph, direction, stops)

        rows.append({
            "err_interp_m": haversine_m(true_lat, true_lon, lat_i, lon_i),
            "err_dr_m": haversine_m(true_lat, true_lon, lat_d, lon_d),
            "gap_s": (t1_ts - t0_ts).total_seconds(),
            "dt_into_gap_s": (t_q - t0_ts).total_seconds(),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Traceur du filtre de Kalman
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter_track(g: pd.DataFrame, route_len: float,
                        r_std: float = 100.0, q_v: float = 0.5) -> pd.DataFrame:
    """Exécute un filtre de Kalman sur une séquence de pings projetés.

    Vecteur d'état : [s (mètres le long de la route), v (m/s)]
    Processus :      s_{t+1} = s_t + v*dt,  v_{t+1} = v_t   (vitesse constante)
    Mesure :         z = s  (projection GPS, bruit ~ R)

    Paramètres
    ----------
    r_std : écart-type du bruit de mesure GPS (m). Plus petit -> fait davantage confiance
            au GPS ; plus grand -> lisse davantage.
    q_v   : bruit de processus sur la vitesse (m/s par sqrt-seconde). Plus grand -> la vitesse
            peut changer plus vite (suit mieux les accélérations, propage moins loin dans un écart).
    Ces deux valeurs sont réglées par recherche aléatoire dans 02_gps_fallback.ipynb ; les
    valeurs par défaut sont quasi optimales sur la ligne 209.

    Retourne le DataFrame d'entrée avec des colonnes supplémentaires :
        ks  -- distance de route lissée par Kalman (m)
        kv  -- vitesse estimée par Kalman (m/s)
        kp  -- écart-type d'incertitude de position (m)
    """
    from filterpy.kalman import KalmanFilter

    g = g.reset_index(drop=True).copy()
    n = len(g)
    t_sec = (pd.to_datetime(g["t"]).astype(np.int64) // 10 ** 9).values.astype(float)

    R_std = float(r_std)   # bruit de projection GPS (m écart-type)
    Q_v   = float(q_v)     # bruit de processus de vitesse (m/s par sqrt-seconde)

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x  = np.array([[float(g["s"].iloc[0])],
                       [float(g["speed"].iloc[0]) / 3.6]])
    kf.F  = np.eye(2)
    kf.H  = np.array([[1.0, 0.0]])
    kf.R  = np.array([[R_std ** 2]])
    kf.P  = np.diag([R_std ** 2, 10.0 ** 2])
    kf.Q  = np.diag([0.0, Q_v])

    ks, kv, kp = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        if i > 0:
            dt = max(t_sec[i] - t_sec[i - 1], 1.0)
            kf.F = np.array([[1.0, dt], [0.0, 1.0]])
            kf.Q = np.array([[Q_v * dt ** 3 / 3, Q_v * dt ** 2 / 2],
                              [Q_v * dt ** 2 / 2, Q_v * dt]])
            kf.predict()

        if not bool(g["signal_gap"].iloc[i]):
            kf.update(np.array([[float(g["s"].iloc[i])]]))

        ks[i] = float(np.clip(kf.x[0, 0], 0.0, route_len))
        kv[i] = float(kf.x[1, 0])
        kp[i] = float(np.sqrt(max(kf.P[0, 0], 0.0)))

    g["ks"] = ks
    g["kv"] = kv
    g["kp"] = kp
    return g


def kalman_fallback(g_filtered: pd.DataFrame, t_query: pd.Timestamp,
                    stops: pd.DataFrame) -> dict | None:
    """Estimation de position pendant un écart en utilisant la piste filtrée par Kalman.

    Propage le dernier état filtré [s, v] en avant jusqu'à t_query.
    Retourne dict : lat, lon, s_m (km), uncertainty_m, method.
    """
    t_arr = pd.to_datetime(g_filtered["t"])
    before = g_filtered[t_arr <= t_query]
    if before.empty:
        return None

    row = before.iloc[-1]
    dt  = (t_query - pd.Timestamp(row["t"])).total_seconds()
    s_est = float(np.clip(row["ks"] + row["kv"] * dt, 0.0, g_filtered["ks"].max()))
    unc   = float(row["kp"] + abs(row["kv"]) * dt * 0.1)
    lat, lon = s_to_latlon(s_est, stops)

    return {"lat": lat, "lon": lon, "s_m": round(s_est / 1000, 2),
            "uncertainty_m": round(unc, 0), "method": "kalman"}


# ─────────────────────────────────────────────────────────────────────────────
# Correction LSTM des estimations Kalman
# ─────────────────────────────────────────────────────────────────────────────

_LSTM_CORR_FEATS = ["ks", "kv", "kp", "speed"]


def train_lstm_correction(g_filtered: pd.DataFrame, window: int = 10):
    """Entraîne un LSTM qui corrige les estimations s de Kalman en utilisant l'historique récent.

    Entrée :  `window` derniers pas de [ks, kv, kp, speed]
    Cible :   vraie distance de route GPS (projection)
    Retourne (modèle, mean, std) — mean/std utilisés pour la normalisation à l'inférence.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    train = g_filtered[~g_filtered["signal_gap"]].reset_index(drop=True)
    feats   = train[_LSTM_CORR_FEATS].values.astype(np.float32)
    targets = train["s"].values.astype(np.float32)

    mean = feats.mean(axis=0); std = feats.std(axis=0) + 1e-6
    feats_n = (feats - mean) / std

    xs, ys = [], []
    for i in range(window, len(train)):
        xs.append(feats_n[i - window:i])
        ys.append(targets[i])
    if not xs:
        return None, mean, std

    X = np.stack(xs); Y = np.array(ys, dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)),
                        batch_size=128, shuffle=True)

    class CorrLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(_LSTM_CORR_FEATS), 32, batch_first=True)
            self.head = nn.Linear(32, 1)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)

    model = CorrLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(20):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model.eval(), mean, std


def kalman_lstm_fallback(g_filtered: pd.DataFrame, t_query: pd.Timestamp,
                         stops: pd.DataFrame, lstm_model, mean: np.ndarray,
                         std: np.ndarray, window: int = 10) -> dict | None:
    """Kalman + correction LSTM : l'historique filtré récent corrige l'estimation de position."""
    import torch

    t_arr = pd.to_datetime(g_filtered["t"])
    before = g_filtered[t_arr <= t_query]
    if len(before) < window:
        return kalman_fallback(g_filtered, t_query, stops)

    recent = before.iloc[-window:][_LSTM_CORR_FEATS].values.astype(np.float32)
    recent_n = (recent - mean) / std
    with torch.no_grad():
        s_corr = float(lstm_model(torch.tensor(recent_n[None]))[0])

    s_est = float(np.clip(s_corr, 0.0, g_filtered["ks"].max()))
    lat, lon = s_to_latlon(s_est, stops)

    return {"lat": lat, "lon": lon, "s_m": round(s_est / 1000, 2),
            "uncertainty_m": round(float(before.iloc[-1]["kp"]), 0),
            "method": "kalman+lstm"}
