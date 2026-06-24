"""Delay layer -- built on top of the reconstructed arrival foundation.

There are NO official per-stop timetables in the data (`ligne.horaires` only stores origin
departure times). So "delay" here is measured against a DATA-DRIVEN baseline: the typical
(median) time each line takes to reach each stop, learned from all reconstructed trips.

    delay = actual elapsed-to-stop  -  expected elapsed-to-stop (baseline)

Interpretation: this is delay *relative to how the line normally performs* (i.e. "this run
is slower/faster than usual / disrupted"), NOT lateness vs a published schedule. If the
company later provides real per-stop timetables, swap `build_baseline` for that schedule and
everything downstream is unchanged.

Pipeline
--------
1. `with_elapsed`   - keep matched arrivals, compute minutes since the trip started, drop
                      physically impossible values.
2. `build_baseline` - expected elapsed-to-stop = median over trips, per
                      (societe, line, dir, seq); keep cells with >= `min_obs` trips.
3. `with_delay`     - delay = elapsed - expected; clip extreme artifacts.
4. `trip_features`  - per-trip table for PREDICTION: state known at `cut_frac` of the route
                      (delay so far, hour, line, ...) -> target = delay at the final stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: torch and prophet are imported lazily inside functions so the module
# loads in environments that don't have them (e.g. py310 kernel).

TRIP_KEYS = ["day", "line", "societe", "bus", "trip_id"]


@dataclass(frozen=True)
class DelayConfig:
    min_obs: int = 20            # min trips per (societe,line,dir,seq) to trust a baseline
    max_elapsed_h: float = 24.0  # drop arrivals whose elapsed exceeds this (broken trips)
    max_abs_delay_min: float = 120.0  # clip |delay| beyond this (reconstruction artifacts)
    cut_frac: float = 0.40       # route fraction "known so far" when predicting final delay
    min_trip_stops: int = 4      # trips need at least this many matched stops to be usable


def load_foundation(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trip_start"] = pd.to_datetime(df["trip_start"])
    df["trip_end"] = pd.to_datetime(df["trip_end"])
    df["arrival"] = pd.to_datetime(df["arrival"])
    return df


def with_elapsed(df: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Matched arrivals only, with `elapsed_min` = minutes from trip start to the stop."""
    m = df[df["matched"]].copy()
    m["elapsed_min"] = (m["arrival"] - m["trip_start"]).dt.total_seconds() / 60
    m = m[(m["elapsed_min"] >= 0) & (m["elapsed_min"] < cfg.max_elapsed_h * 60)]
    m["dep_hour"] = m["trip_start"].dt.hour
    m["dow"] = m["trip_start"].dt.dayofweek
    return m.reset_index(drop=True)


def build_baseline(m: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Expected elapsed-to-stop per (societe, line, dir, seq) -- the data-driven 'schedule'."""
    g = m.groupby(["societe", "line", "dir", "seq"])["elapsed_min"]
    base = g.agg(expected_min="median", p10=lambda s: s.quantile(0.10),
                 p90=lambda s: s.quantile(0.90), n="count").reset_index()
    return base[base["n"] >= cfg.min_obs].reset_index(drop=True)


def with_delay(m: pd.DataFrame, baseline: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Attach expected time and `delay_min = elapsed - expected` (clipped)."""
    out = m.merge(baseline[["societe", "line", "dir", "seq", "expected_min"]],
                  on=["societe", "line", "dir", "seq"], how="inner")
    out["delay_min"] = (out["elapsed_min"] - out["expected_min"]).clip(
        -cfg.max_abs_delay_min, cfg.max_abs_delay_min)
    return out


def trip_features(d: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """One row per trip: the delay state known at `cut_frac` of the route (the moment we
    'predict from') and the target = delay at the final reached stop.

    This is the table a delay-prediction model trains on: predict how late the bus will be
    at the end of its run, given how it is doing partway through.
    """
    rows = []
    for keys, t in d.sort_values("seq").groupby(TRIP_KEYS):
        if len(t) < cfg.min_trip_stops:
            continue
        smax = t["seq"].max()
        early = t[t["seq"] <= cfg.cut_frac * smax]
        if len(early) == 0:
            continue
        cur, fin = early.iloc[-1], t.iloc[-1]
        rows.append({
            **dict(zip(TRIP_KEYS, keys)),
            "dir": t["dir"].iloc[0],
            "dep_hour": int(cur["dep_hour"]),
            "dow": int(cur["dow"]),
            "cur_seq": int(cur["seq"]),
            "cur_seq_frac": float(cur["seq"] / smax) if smax else 0.0,
            "cur_delay_min": float(cur["delay_min"]),     # delay so far  (key predictor)
            "cur_elapsed_min": float(cur["elapsed_min"]),
            "final_delay_min": float(fin["delay_min"]),   # TARGET
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- day-type + rolling model
# Numeric + categorical features used by the rolling next-stop model. `line`/`dir` are passed
# natively as categoricals (HistGradientBoosting handles them), so training and serving use the
# exact same columns with no manual one-hot bookkeeping.
FEATURES_NUM = ["dep_hour", "dow", "is_weekend", "seq", "seq_frac", "delay_min", "elapsed_min"]
FEATURES_CAT = ["line", "dir"]
TARGET = "next_delay_min"


def add_daytype(m: pd.DataFrame) -> pd.DataFrame:
    """Add cheap calendar features. (Weather would need an external source -- not in the DB.)
    Tunisia's weekend is Saturday/Sunday -> dayofweek 5/6."""
    m = m.copy()
    m["is_weekend"] = m["dow"].isin([5, 6]).astype(int)
    m["month"] = m["trip_start"].dt.month
    return m


def rolling_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (trip, stop k): current state + target = delay at the NEXT stop k+1.

    This is the table the rolling model trains on -- predict the delay one stop ahead as the
    bus progresses, which chains into a full ETA for the rest of the run.
    """
    d = d.sort_values(TRIP_KEYS + ["seq"]).copy()
    g = d.groupby(TRIP_KEYS)
    d["seq_frac"] = g["seq"].transform(lambda s: s / s.max() if s.max() else 0.0)
    d[TARGET] = g["delay_min"].shift(-1)
    d["next_seq"] = g["seq"].shift(-1)
    return d.dropna(subset=[TARGET]).reset_index(drop=True)


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[FEATURES_NUM + FEATURES_CAT].copy()
    for c in FEATURES_CAT:
        X[c] = X[c].astype("category")
    return X


def train_rolling_model(roll: pd.DataFrame, **kw):
    """Fit the next-stop delay model. `line`/`dir` handled as native categoricals."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        categorical_features=FEATURES_CAT,
        max_iter=kw.get("max_iter", 300),
        learning_rate=kw.get("learning_rate", 0.05),
        max_depth=kw.get("max_depth", 6),
        random_state=0,
    )
    model.fit(_design(roll), roll[TARGET])
    return model


# ──────────────────────────────────────────────────────────────────────────────
# LSTM rolling delay model (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

# Features fed to the LSTM at each stop along a trip.
# WHY these five:
#   delay_min    -- how late the bus is RIGHT NOW (strongest predictor; delay compounds)
#   elapsed_min  -- absolute time since departure (captures long-haul fatigue effects)
#   seq_frac     -- progress along route 0->1 (later stops see more accumulated delay)
#   is_weekend   -- Tunisia weekend (Sat/Sun) is measurably less congested
#   dep_hour     -- rush-hour vs off-peak behaviour
LSTM_STEP_FEATS = ["delay_min", "elapsed_min", "seq_frac", "is_weekend", "dep_hour"]


def build_lstm_sequences(roll: pd.DataFrame, max_len: int = 30
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded input sequences and targets for LSTM training.

    Sliding-window approach: for each trip and each stop k, the input is the
    full history of stops [0..k] (right-aligned, zero-padded on the left) and
    the target is the delay at the NEXT stop k+1.

    WHY right-alignment: the LSTM reads left-to-right; placing the most recent
    stop at the rightmost position means the hidden state at the last timestep
    always reflects "right now", regardless of trip length.

    Returns
    -------
    X       : (N, max_len, n_feats)  -- padded raw feature sequences
    lengths : (N,)                   -- true sequence length before padding
    y       : (N,)                   -- target next_delay_min

    NOTE: X is returned UN-normalised. Call fit_lstm_scaler(X_train) then
    scale_sequences() before training so stats come from training data only
    (no leakage into val/test).
    """
    seqs, lengths, targets = [], [], []
    roll = roll.sort_values(TRIP_KEYS + ["seq"])
    for _, grp in roll.groupby(TRIP_KEYS):
        grp = grp.reset_index(drop=True)
        feats = grp[LSTM_STEP_FEATS].values.astype(np.float32)
        ys    = grp[TARGET].values.astype(np.float32)
        for k in range(len(grp)):
            seq = feats[: k + 1]
            T   = min(len(seq), max_len)
            pad = np.zeros((max_len, len(LSTM_STEP_FEATS)), dtype=np.float32)
            pad[-T:] = seq[-T:]
            seqs.append(pad)
            lengths.append(T)
            targets.append(float(ys[k]))
    return np.stack(seqs), np.array(lengths, dtype=np.int64), np.array(targets, dtype=np.float32)


def fit_lstm_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from training sequences ONLY.

    WHY: delay_min ranges -120..+120, elapsed_min 0..600, dep_hour 0..23.
    Without standardisation the LSTM gradient is dominated by the largest-scale
    feature and the smaller ones (seq_frac, is_weekend) are effectively ignored.

    MUST be called on X_train only. Apply the returned stats to val and test
    with scale_sequences() to avoid data leakage.

    Returns (mean, std) each shaped (n_feats,).
    """
    # Flatten all timesteps from all training sequences to get global stats
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8   # +eps prevents /0 on binary features (is_weekend)
    return mean.astype(np.float32), std.astype(np.float32)


def scale_sequences(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply standardisation (z-score) to a (N, T, F) sequence array."""
    return ((X - mean) / std).astype(np.float32)


def _make_delay_lstm(n_feats: int, hidden: int = 64, n_layers: int = 2):
    """Stacked LSTM encoder -> linear regression head.

    Architecture choices:
      - 2 layers: first layer learns local stop-to-stop patterns, second learns
        trip-level trends (compounding, recovery).
      - dropout=0.1 between layers: mild regularisation -- the sequences are short
        (<=30 steps) so aggressive dropout hurts more than it helps.
      - Output: scalar (regression, not classification -- we predict minutes, not
        a binary "late/on time" label).

    WHY NOT Transformer: sequences are short (<=30 steps), dataset is ~100k
    samples. LSTMs train faster and perform comparably at this scale.
    """
    import torch.nn as nn

    class DelayLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feats, hidden, num_layers=n_layers,
                                batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)  # last timestep = current bus state

    return DelayLSTM()


def train_lstm_delay(X: np.ndarray, y: np.ndarray, *,
                     hidden: int = 64, n_layers: int = 2,
                     epochs: int = 30, lr: float = 1e-3, batch: int = 256,
                     patience: int = 5) -> object:
    """Train LSTM delay predictor with validation split and early stopping.

    Data split
    ----------
    X/y are already the TRAINING portion (day < cut_day). We split off the
    last 10% of sequences as a temporal validation set. This is intentionally
    the LAST 10% (not random) to simulate future data -- random shuffling would
    leak future trip patterns into the validation set.

    Early stopping
    --------------
    Training halts when validation loss stops improving for `patience` epochs.
    The BEST checkpoint (lowest val loss) is restored before returning, so the
    model is never the overfitted end-of-training state.

    WHY patience=5: each epoch ~200s on CPU; 5 epochs grace = ~17 min max
    overshoot before stopping. On GPU (10x faster) you can raise this.

    Returns trained model (CPU, eval mode).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Temporal val split: keep temporal order, take last 10% as val
    n_val = max(1, int(0.10 * len(X)))
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=batch, shuffle=True,
    )

    model   = _make_delay_lstm(X.shape[2], hidden, n_layers).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        # ── training pass ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)

        # ── validation pass (no gradients) ───────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv))

        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  "
                  f"train={train_loss/len(X_tr):.4f}  val={val_loss:.4f}")

        # ── early stopping: save best, stop if no improvement ────────────────
        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep+1}  "
                      f"(best val={best_val:.4f})")
                break

    # Restore the best checkpoint, not the last epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu().eval()


def predict_lstm(model, X: np.ndarray,
                 scaler_mean: np.ndarray | None = None,
                 scaler_std:  np.ndarray | None = None) -> np.ndarray:
    """Run inference on a batch of sequences. Returns (N,) predictions.

    If scaler_mean/std are provided the input is normalised before inference --
    the same transform applied during training must be applied here.
    """
    import torch
    if scaler_mean is not None:
        X = scale_sequences(X, scaler_mean, scaler_std)
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def serve_eta_lstm(model, baseline: pd.DataFrame, *,
                   societe, line, direction, dep_time,
                   current_seq: int, current_delay_min: float,
                   max_len: int = 30,
                   scaler_mean: np.ndarray | None = None,
                   scaler_std:  np.ndarray | None = None) -> pd.DataFrame:
    """ETA table for all remaining stops using the trained LSTM.

    At each step we build the history up to the current stop, right-align it,
    apply the same normalisation used during training, and predict the next
    stop's delay. We then advance and repeat -- this is autoregressive inference.

    WHY autoregressive (rather than predicting all stops at once): the model
    was trained to predict ONE step ahead; feeding its own output back as input
    lets it extrapolate the full route without requiring a variable-length output.
    """
    b = baseline[(baseline["societe"] == societe) & (baseline["line"] == line)
                 & (baseline["dir"] == direction)].sort_values("seq")
    if b.empty:
        return pd.DataFrame(columns=["seq", "expected_min", "pred_delay_min", "eta"])

    dep_time = pd.Timestamp(dep_time)
    dep_hour = dep_time.hour
    is_wkend = int(dep_time.dayofweek in (5, 6))
    exp  = dict(zip(b["seq"].astype(int), b["expected_min"]))
    smax = int(b["seq"].max())

    # Seed the rolling history with the bus's known current state
    history: list[list[float]] = [
        [current_delay_min, exp.get(current_seq, 0.0) + current_delay_min,
         current_seq / smax if smax else 0.0, is_wkend, dep_hour]
    ]

    cur_seq, cur_delay, rows = int(current_seq), float(current_delay_min), []
    while cur_seq < smax:
        nxt = cur_seq + 1
        if nxt not in exp:
            cur_seq = nxt
            continue

        T   = min(len(history), max_len)
        pad = np.zeros((1, max_len, len(LSTM_STEP_FEATS)), dtype=np.float32)
        pad[0, -T:] = np.array(history[-T:], dtype=np.float32)
        nd  = float(predict_lstm(model, pad, scaler_mean, scaler_std)[0])

        rows.append({
            "seq": nxt, "expected_min": round(exp[nxt], 1),
            "pred_delay_min": round(nd, 1),
            "eta": dep_time + pd.Timedelta(minutes=exp[nxt] + nd),
        })
        history.append([nd, exp[nxt] + nd, nxt / smax, is_wkend, dep_hour])
        cur_seq, cur_delay = nxt, nd

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Prophet delay forecasting
# ──────────────────────────────────────────────────────────────────────────────

def fit_prophet(d: pd.DataFrame, line: str, direction: str, societe: str):
    """Fit a Prophet model on the daily mean delay for one (line, dir).

    Returns the fitted Prophet model. Input `d` must have delay_min and trip_start.
    """
    from prophet import Prophet
    import warnings
    warnings.filterwarnings("ignore")

    sub = d[(d["line"] == line) & (d["dir"] == direction) & (d["societe"] == societe)].copy()
    ts = (sub.groupby(sub["trip_start"].dt.date)["delay_min"]
            .mean()
            .reset_index()
            .rename(columns={"trip_start": "ds", "delay_min": "y"}))
    ts["ds"] = pd.to_datetime(ts["ds"])
    if len(ts) < 10:
        return None
    m = Prophet(weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="additive", interval_width=0.80)
    m.fit(ts)
    return m


def prophet_forecast(m, periods: int = 30) -> pd.DataFrame:
    """Forecast `periods` days ahead. Returns ds, yhat, yhat_lower, yhat_upper."""
    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)


def serve_eta(model, baseline: pd.DataFrame, *, societe, line, direction, dep_time,
              current_seq: int, current_delay_min: float) -> pd.DataFrame:
    """PRODUCTION ETA: given a bus's live state (where it is + how late it is now), roll the
    next-stop model forward to predict delay -- and a clock ETA -- at every remaining stop.

    Returns one row per downstream stop: seq, expected_min (baseline), pred_delay_min, eta.
    """
    b = baseline[(baseline["societe"] == societe) & (baseline["line"] == line)
                 & (baseline["dir"] == direction)].sort_values("seq")
    if b.empty:
        return pd.DataFrame(columns=["seq", "expected_min", "pred_delay_min", "eta"])
    dep_time = pd.Timestamp(dep_time)
    dep_hour, dow = dep_time.hour, dep_time.dayofweek
    is_weekend = int(dow in (5, 6))
    exp = dict(zip(b["seq"].astype(int), b["expected_min"]))
    smax = int(b["seq"].max())

    cur_seq, cur_delay, rows = int(current_seq), float(current_delay_min), []
    while cur_seq < smax:
        nxt = cur_seq + 1
        if nxt not in exp:
            cur_seq = nxt
            continue
        x = _design(pd.DataFrame([{
            "dep_hour": dep_hour, "dow": dow, "is_weekend": is_weekend,
            "seq": cur_seq, "seq_frac": cur_seq / smax,
            "delay_min": cur_delay, "elapsed_min": exp.get(cur_seq, 0.0) + cur_delay,
            "line": line, "dir": direction,
        }]))
        nd = float(model.predict(x)[0])
        rows.append({"seq": nxt, "expected_min": round(exp[nxt], 1),
                     "pred_delay_min": round(nd, 1),
                     "eta": dep_time + pd.Timedelta(minutes=exp[nxt] + nd)})
        cur_seq, cur_delay = nxt, nd
    return pd.DataFrame(rows)
