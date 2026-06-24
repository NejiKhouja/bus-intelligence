"""GPS Fallback layer — position estimates during signal gaps.

When a bus loses GPS for > signal_gap_s seconds the operator dashboard shows
the bus as 'disappeared'. This layer fills the gap with a position estimate
derived from the route geometry.

Baseline methods
----------------
linear_interp   Interpolate s (route distance, metres) linearly between the
                last known ping before the gap and the first ping after.

dead_reckoning  Project forward from the last ping using its reported speed.
                Useful when no recovery ping exists yet (bus is currently dark).

Upgraded methods
----------------
Kalman filter   Tracks state [s, velocity] along the route. Each GPS ping is
                a noisy measurement; during a gap only the predict step runs,
                giving a principled uncertainty estimate (covariance grows).
                Implemented with filterpy.KalmanFilter.

LSTM correction After the Kalman predict, an LSTM trained on historical pings
                corrects the estimate using learned traffic patterns (speed
                profiles, stop behaviour). Reduces systematic bias the linear
                Kalman model cannot capture.

`kalman_filter_track` runs the full Kalman filter on a projected ping sequence.
`kalman_fallback` queries the filtered track at any timestamp inside a gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Route geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def s_to_latlon(s_query: float, stops: pd.DataFrame) -> tuple[float, float]:
    """Convert a route distance (metres) back to (lat, lon) via anchor polyline."""
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
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000.0
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p)
         * np.sin((lon2 - lon1) * p / 2) ** 2)
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


# ─────────────────────────────────────────────────────────────────────────────
# Gap extraction
# ─────────────────────────────────────────────────────────────────────────────

def gap_table(g: pd.DataFrame) -> pd.DataFrame:
    """One row per signal gap with before/after route context.

    Input: projected ping DataFrame (output of foundation.project_to_route).
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
# Estimation methods
# ─────────────────────────────────────────────────────────────────────────────

def interp_position(t_query: pd.Timestamp, t0: pd.Timestamp, s0: float,
                    t1: pd.Timestamp, s1: float,
                    stops: pd.DataFrame) -> tuple[float, float, float]:
    """Linear interpolation of route distance during a gap → (lat, lon, s_m)."""
    total = (t1 - t0).total_seconds()
    frac = (t_query - t0).total_seconds() / total if total > 0 else 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    s_est = s0 + frac * (s1 - s0)
    lat, lon = s_to_latlon(s_est, stops)
    return lat, lon, s_est


def dead_reckon_position(t_query: pd.Timestamp, t0: pd.Timestamp, s0: float,
                         speed_kph: float, direction: int,
                         stops: pd.DataFrame) -> tuple[float, float, float]:
    """Project forward from last known speed → (lat, lon, s_m).

    direction: +1 for ALLER (s increasing), -1 for RETOUR.
    """
    dt = (t_query - t0).total_seconds()
    s_est = s0 + direction * (speed_kph / 3.6) * dt
    s_max = float(stops["s_m"].max())
    s_est = float(np.clip(s_est, 0.0, s_max))
    lat, lon = s_to_latlon(s_est, stops)
    return lat, lon, s_est


# ─────────────────────────────────────────────────────────────────────────────
# Production: best estimate for any query time
# ─────────────────────────────────────────────────────────────────────────────

def fallback_position(g: pd.DataFrame, t_query: pd.Timestamp,
                      stops: pd.DataFrame) -> dict | None:
    """Best position estimate for a query timestamp that falls inside a gap.

    Returns None if t_query is not inside any gap.
    Returns a dict with keys:
        lat_interp, lon_interp, s_interp   — linear interpolation (if recovery ping known)
        lat_dr, lon_dr, s_dr               — dead reckoning from last known speed
        gap_s                              — gap duration in seconds
        method                             — 'interp' | 'dead_reckon' (recommended one)
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
        return None  # not in a gap

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
        "method": "interp",  # prefer interp when recovery ping is known
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: synthetic masking
# ─────────────────────────────────────────────────────────────────────────────

def eval_fallback(g: pd.DataFrame, stops: pd.DataFrame,
                  mask_min: float = 3.0, n_samples: int = 200,
                  rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Evaluate both methods by synthetically masking mask_min minutes of pings.

    For each of n_samples random windows:
      1. Pretend the bus was dark for mask_min minutes starting at a random ping.
      2. Estimate position at the gap midpoint with both methods.
      3. Measure error (metres) against true GPS position.

    Returns a DataFrame with columns: err_interp_m, err_dr_m, gap_s, dt_into_gap_s.
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
# Kalman filter tracker
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter_track(g: pd.DataFrame, route_len: float) -> pd.DataFrame:
    """Run a Kalman filter over a projected ping sequence.

    State vector: [s (metres along route), v (m/s)]
    Process:      s_{t+1} = s_t + v*dt,  v_{t+1} = v_t   (constant velocity)
    Measurement:  z = s  (GPS projection, noise ~ R)

    Returns the input DataFrame with extra columns:
        ks  -- Kalman-smoothed route distance (m)
        kv  -- Kalman-estimated velocity (m/s)
        kp  -- position uncertainty std dev (m)
    """
    from filterpy.kalman import KalmanFilter

    g = g.reset_index(drop=True).copy()
    n = len(g)
    t_sec = (pd.to_datetime(g["t"]).astype(np.int64) // 10 ** 9).values.astype(float)

    R_std = 100.0   # GPS projection noise ~100 m std
    Q_v   = 0.5     # velocity process noise (m/s per sqrt-second)

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
    """Position estimate during a gap using the Kalman filtered track.

    Propagates the last filtered [s, v] state forward to t_query.
    Returns dict: lat, lon, s_m (km), uncertainty_m, method.
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
# LSTM correction of Kalman estimates
# ─────────────────────────────────────────────────────────────────────────────

_LSTM_CORR_FEATS = ["ks", "kv", "kp", "speed"]


def train_lstm_correction(g_filtered: pd.DataFrame, window: int = 10):
    """Train an LSTM that corrects Kalman s-estimates using recent history.

    Input:  last `window` steps of [ks, kv, kp, speed]
    Target: true GPS s (route distance from projection)
    Returns (model, mean, std) — mean/std used for normalisation at inference.
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
    """Kalman + LSTM correction: recent filtered history corrects the position estimate."""
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
