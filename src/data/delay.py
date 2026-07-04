"""Couche retard — construite sur la fondation d'arrivées reconstruites.

Il n'y a PAS d'horaires officiels par arrêt dans les données (`ligne.horaires` ne stocke
que les heures de départ à l'origine). Donc le « retard » ici est mesuré par rapport à une
base de référence PILOTÉE PAR LES DONNÉES : le temps typique (médiane) que met chaque ligne
pour atteindre chaque arrêt, appris à partir de tous les trajets reconstruits.

    retard = temps écoulé réel jusqu'à l'arrêt  -  temps écoulé attendu jusqu'à l'arrêt (base de référence)

Interprétation : il s'agit du retard *par rapport aux performances habituelles de la ligne*
(c'est-à-dire « cette course est plus lente/rapide que d'habitude / perturbée »), PAS du
retard par rapport à un horaire publié. Si l'entreprise fournit ultérieurement de vrais
horaires par arrêt, remplacer `build_baseline` par cet horaire et tout l'aval reste inchangé.

Pipeline
--------
1. `with_elapsed`   - garder les arrivées correspondantes, calculer les minutes depuis le début
                      du trajet, supprimer les valeurs physiquement impossibles.
2. `build_baseline` - temps écoulé attendu jusqu'à l'arrêt = médiane sur les trajets, par
                      (societe, line, dir, seq) ; garder les cellules avec >= `min_obs` trajets.
3. `with_delay`     - retard = écoulé - attendu ; écrêter les artefacts extrêmes.
4. `trip_features`  - table par trajet pour la PRÉDICTION : état connu à `cut_frac` de la route
                      (retard jusqu'ici, heure, ligne, ...) -> cible = retard au dernier arrêt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRIP_KEYS = ["day", "line", "societe", "bus", "trip_id"]


@dataclass(frozen=True)
class DelayConfig:
    min_obs: int = 20            # trajets min par (societe,line,dir,seq) pour faire confiance à une base de référence
    max_elapsed_h: float = 24.0  # supprimer les arrivées dont le temps écoulé dépasse ceci (trajets cassés)
    max_abs_delay_min: float = 120.0  # écrêter |retard| au-delà de ceci (artefacts de reconstruction)
    cut_frac: float = 0.40       # fraction de la route « connue jusqu'ici » lors de la prédiction du retard final
    min_trip_stops: int = 4      # les trajets ont besoin d'au moins ce nombre d'arrêts correspondants pour être utilisables


def load_foundation(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trip_start"] = pd.to_datetime(df["trip_start"])
    df["trip_end"] = pd.to_datetime(df["trip_end"])
    df["arrival"] = pd.to_datetime(df["arrival"])
    return df


def with_elapsed(df: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Arrivées correspondantes uniquement, avec `elapsed_min` = minutes depuis le début du trajet jusqu'à l'arrêt."""
    m = df[df["matched"]].copy()
    m["elapsed_min"] = (m["arrival"] - m["trip_start"]).dt.total_seconds() / 60
    m = m[(m["elapsed_min"] >= 0) & (m["elapsed_min"] < cfg.max_elapsed_h * 60)]
    m["dep_hour"] = m["trip_start"].dt.hour
    m["dow"] = m["trip_start"].dt.dayofweek
    return m.reset_index(drop=True)


def build_baseline(m: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Temps écoulé attendu jusqu'à l'arrêt par (societe, line, dir, seq) — l'« horaire » piloté par les données."""
    g = m.groupby(["societe", "line", "dir", "seq"])["elapsed_min"]
    base = g.agg(expected_min="median", p10=lambda s: s.quantile(0.10),
                 p90=lambda s: s.quantile(0.90), n="count").reset_index()
    return base[base["n"] >= cfg.min_obs].reset_index(drop=True)


def with_delay(m: pd.DataFrame, baseline: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Attache le temps attendu et `delay_min = écoulé - attendu` (écrêté)."""
    out = m.merge(baseline[["societe", "line", "dir", "seq", "expected_min"]],
                  on=["societe", "line", "dir", "seq"], how="inner")
    out["delay_min"] = (out["elapsed_min"] - out["expected_min"]).clip(
        -cfg.max_abs_delay_min, cfg.max_abs_delay_min)
    return out


def trip_features(d: pd.DataFrame, cfg: DelayConfig) -> pd.DataFrame:
    """Une ligne par trajet : l'état du retard connu à `cut_frac` de la route (le moment où l'on
    « prédit ») et la cible = retard au dernier arrêt atteint.

    C'est la table sur laquelle un modèle de prédiction de retard s'entraîne : prédire le
    retard du bus à la fin de sa course en fonction de ses performances à mi-parcours.
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
            "cur_delay_min": float(cur["delay_min"]),     # retard jusqu'ici  (prédicteur clé)
            "cur_elapsed_min": float(cur["elapsed_min"]),
            "final_delay_min": float(fin["delay_min"]),   # CIBLE
        })
    return pd.DataFrame(rows)


FEATURES_NUM = ["dep_hour", "dow", "is_weekend", "is_rush_hour", "seq", "seq_frac",
                "delay_min", "elapsed_min", "rain_frac"]
FEATURES_CAT = ["societe", "line", "dir"]
TARGET = "next_delay_min"


def add_daytype(m: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des caractéristiques calendaires simples.
    Le week-end tunisien est samedi/dimanche -> dayofweek 5/6.

    `is_rush_hour` : jour de semaine + 7h-9h ou 16h-19h -- une hypothèse calendaire documentée,
    pas ajustée empiriquement. Peu coûteuse et utilisable aussi bien à l'entraînement qu'en
    service en direct, contrairement à la météo/congestion qui nécessitent une source externe.
    """
    m = m.copy()
    m["is_weekend"] = m["dow"].isin([5, 6]).astype(int)
    m["is_rush_hour"] = (
        (~m["is_weekend"].astype(bool)) &
        (((m["dep_hour"] >= 7) & (m["dep_hour"] <= 9)) | ((m["dep_hour"] >= 16) & (m["dep_hour"] <= 19)))
    ).astype(int)
    m["month"] = m["trip_start"].dt.month
    return m


def add_weather(m: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Attache `rain_frac` par jour calendaire, depuis le cache météo journalier précalculé
    (voir `src/data/weather.py::load_day_weather`). Laissé à NaN là où le jour n'est pas
    couvert (~55% des jours, la collection source étant éparse et s'arrêtant ~sept. 2025) --
    HistGBM gère nativement les valeurs manquantes (branche apprise dédiée), donc aucune
    imputation n'est nécessaire. Si `weather` est None (cache pas encore construit), `rain_frac`
    est NaN partout -- comportement inchangé pour qui n'a pas encore lancé la construction du cache.
    """
    m = m.copy()
    if weather is None or len(weather) == 0:
        m["rain_frac"] = np.nan
        return m
    m["day"] = m["trip_start"].dt.strftime("%Y%m%d")
    m = m.merge(weather[["day", "rain_frac"]], on="day", how="left")
    return m


def rolling_table(d: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (trajet, arrêt k) : état actuel + cible = retard à l'arrêt SUIVANT k+1.

    C'est la table sur laquelle le modèle glissant s'entraîne — prédire le retard à un arrêt
    en avance à mesure que le bus progresse, ce qui se chaîne en une ETA complète pour le
    reste de la course.
    """
    d = d.sort_values(TRIP_KEYS + ["seq"]).copy()
    g = d.groupby(TRIP_KEYS)
    d["seq_frac"] = g["seq"].transform(lambda s: s / s.max() if s.max() else 0.0)
    d[TARGET] = g["delay_min"].shift(-1)
    d["next_seq"] = g["seq"].shift(-1)
    return d.dropna(subset=[TARGET]).reset_index(drop=True)


def _design(frame: pd.DataFrame, numeric_features: list[str] | None = None) -> pd.DataFrame:
    numeric_features = FEATURES_NUM if numeric_features is None else numeric_features
    X = frame[numeric_features + FEATURES_CAT].copy()
    for c in FEATURES_CAT:
        X[c] = X[c].astype("category")
    return X


def train_rolling_model(roll: pd.DataFrame, **kw):
    """Entraîne le modèle de retard au prochain arrêt. `line`/`dir` gérées comme catégorielles natives.

    Élimine d'abord toute caractéristique numérique dégénérée (< 2 valeurs non manquantes
    distinctes) dans CE split d'entraînement précis. POURQUOI : `HistGradientBoostingRegressor`
    plante (au lieu de gérer gracieusement) sur une colonne constante lors du binning --
    `rain_frac` y est particulièrement exposé car la couverture météo est figée à ~sept. 2025
    alors que la fenêtre d'entraînement continue de grandir : plus le temps passe, plus la
    fraction de jours couverts rétrécit, et un split futur pourrait ne recouper qu'un seul jour
    météo (donc une seule valeur non manquante) sans qu'aucun code n'ait changé.

    La liste des caractéristiques réellement utilisées est stockée sur le modèle lui-même
    (`model.feature_names_used_`, persistée par joblib) -- `serve_eta()` DOIT la relire plutôt
    que de supposer `FEATURES_NUM` au complet, sinon la matrice de service ne correspondrait
    plus à ce sur quoi le modèle a été entraîné.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    usable_num = [c for c in FEATURES_NUM if roll[c].nunique(dropna=True) >= 2]
    dropped = [c for c in FEATURES_NUM if c not in usable_num]
    if dropped:
        print(f"  ATTENTION : caractéristiques dégénérées ignorées pour cet entraînement : {dropped}")

    model = HistGradientBoostingRegressor(
        categorical_features=FEATURES_CAT,
        max_iter=kw.get("max_iter", 300),
        learning_rate=kw.get("learning_rate", 0.05),
        max_depth=kw.get("max_depth", 6),
        random_state=0,
    )
    model.fit(_design(roll, usable_num), roll[TARGET])
    model.feature_names_used_ = usable_num
    return model


LSTM_STEP_FEATS = ["delay_min", "elapsed_min", "seq_frac", "is_weekend", "dep_hour", "is_rush_hour"]


def build_lstm_sequences(roll: pd.DataFrame, max_len: int = 30
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit des séquences d'entrée rembourrées et des cibles pour l'entraînement LSTM.

    Approche par fenêtre glissante : pour chaque trajet et chaque arrêt k, l'entrée est
    l'historique complet des arrêts [0..k] (aligné à droite, rembourré de zéros à gauche)
    et la cible est le retard à l'arrêt SUIVANT k+1.

    POURQUOI l'alignement à droite : le LSTM lit de gauche à droite ; placer l'arrêt le plus
    récent à la position la plus à droite signifie que l'état caché au dernier pas de temps
    reflète toujours « maintenant », quelle que soit la longueur du trajet.

    Retourne
    --------
    X       : (N, max_len, n_feats)  -- séquences brutes rembourrées
    lengths : (N,)                   -- vraie longueur de séquence avant rembourrage
    y       : (N,)                   -- cible next_delay_min
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
    """Calcule la moyenne et l'écart-type par caractéristique à partir des séquences d'entraînement UNIQUEMENT.

    POURQUOI : delay_min varie de -120..+120, elapsed_min de 0..600, dep_hour de 0..23.
    Sans normalisation, le gradient du LSTM est dominé par la caractéristique à plus grande
    échelle et les plus petites (seq_frac, is_weekend) sont effectivement ignorées.

    DOIT être appelé sur X_train uniquement. Appliquer les statistiques retournées à val et test
    avec scale_sequences() pour éviter la fuite de données.

    Retourne (mean, std) chacun de forme (n_feats,).
    """
    # Aplatir tous les pas de temps de toutes les séquences d'entraînement pour obtenir les statistiques globales
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8   # +eps empêche /0 sur les caractéristiques binaires (is_weekend)
    return mean.astype(np.float32), std.astype(np.float32)


def scale_sequences(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Applique la normalisation (z-score) à un tableau de séquences (N, T, F)."""
    return ((X - mean) / std).astype(np.float32)


def _make_delay_lstm(n_feats: int, hidden: int = 64, n_layers: int = 2):
    """Encodeur LSTM empilé -> tête de régression linéaire.

    Choix d'architecture :
      - 2 couches : la première apprend les schémas locaux arrêt-à-arrêt, la seconde
        apprend les tendances au niveau du trajet (cumul, récupération).
      - dropout=0.1 entre les couches : régularisation légère -- les séquences sont courtes
        (<=30 pas) donc un dropout agressif fait plus de mal que de bien.
      - Sortie : scalaire (régression, pas classification -- on prédit des minutes, pas
        une étiquette binaire « en retard/à l'heure »).

    POURQUOI PAS Transformer : les séquences sont courtes (<=30 pas), le jeu de données
    fait ~100k échantillons. Les LSTM s'entraînent plus vite et performent de façon
    comparable à cette échelle.
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
            return self.head(out[:, -1, :]).squeeze(-1)  # dernier pas de temps = état actuel du bus

    return DelayLSTM()


def train_lstm_delay(X: np.ndarray, y: np.ndarray, *,
                     hidden: int = 64, n_layers: int = 2,
                     epochs: int = 30, lr: float = 1e-3, batch: int = 256,
                     patience: int = 5) -> object:
    """Entraîne le prédicteur de retard LSTM avec division de validation et arrêt anticipé.

    Division des données
    --------------------
    X/y sont déjà la portion d'ENTRAÎNEMENT (jour < cut_day). On sépare les
    10 derniers % de séquences comme ensemble de validation temporel. C'est
    intentionnellement les DERNIERS 10 % (pas aléatoire) pour simuler des données futures —
    un mélange aléatoire fuirait les schémas de trajets futurs dans l'ensemble de validation.

    Arrêt anticipé
    ---------------
    L'entraînement s'arrête quand la perte de validation cesse de s'améliorer pendant
    `patience` époques. Le MEILLEUR point de contrôle (perte val la plus basse) est restauré
    avant le retour, donc le modèle n'est jamais dans l'état de sur-ajustement de fin d'entraînement.

    POURQUOI patience=5 : chaque époque ~200s sur CPU ; 5 époques de grâce = ~17 min max
    de dépassement avant arrêt. Sur GPU (10x plus rapide), on peut augmenter ceci.

    Retourne le modèle entraîné (CPU, mode évaluation).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Division val temporelle : conserver l'ordre temporel, prendre les 10 derniers % comme val
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
        # passe d'entraînement
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)

        # passe de validation (sans gradients)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv))

        if (ep + 1) % 5 == 0:
            print(f"  époque {ep+1:3d}/{epochs}  "
                  f"train={train_loss/len(X_tr):.4f}  val={val_loss:.4f}")

        # arrêt anticipé
        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Arrêt anticipé à l'époque {ep+1}  "
                      f"(meilleure val={best_val:.4f})")
                break

    # Restaurer le meilleur point de contrôle, pas la dernière époque
    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu().eval()


def predict_lstm(model, X: np.ndarray,
                 scaler_mean: np.ndarray | None = None,
                 scaler_std:  np.ndarray | None = None) -> np.ndarray:
    """Exécute l'inférence sur un lot de séquences. Retourne (N,) prédictions.

    Si scaler_mean/std sont fournis, l'entrée est normalisée avant l'inférence —
    la même transformation appliquée pendant l'entraînement doit être appliquée ici.
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
    """Table ETA pour tous les arrêts restants en utilisant le LSTM entraîné.

    À chaque étape, on construit l'historique jusqu'à l'arrêt actuel, on l'aligne à droite,
    on applique la même normalisation utilisée pendant l'entraînement, et on prédit le retard
    au prochain arrêt. On avance ensuite et on répète — c'est une inférence autorégressive.

    POURQUOI autorégressive (plutôt que prédire tous les arrêts à la fois) : le modèle a
    été entraîné à prédire UN pas en avant ; réinjecter sa propre sortie comme entrée
    permet d'extrapoler la route complète sans nécessiter une sortie de longueur variable.
    """
    b = baseline[(baseline["societe"] == societe) & (baseline["line"] == line)
                 & (baseline["dir"] == direction)].sort_values("seq")
    if b.empty:
        return pd.DataFrame(columns=["seq", "expected_min", "pred_delay_min", "eta"])

    dep_time = pd.Timestamp(dep_time)
    dep_hour = dep_time.hour
    is_wkend = int(dep_time.dayofweek in (5, 6))
    is_rush  = int(((7 <= dep_hour <= 9) or (16 <= dep_hour <= 19)) and not is_wkend)
    exp  = dict(zip(b["seq"].astype(int), b["expected_min"]))
    smax = int(b["seq"].max())

    # Initialiser l'historique glissant avec l'état actuel connu du bus
    history: list[list[float]] = [
        [current_delay_min, exp.get(current_seq, 0.0) + current_delay_min,
         current_seq / smax if smax else 0.0, is_wkend, dep_hour, is_rush]
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
        history.append([nd, exp[nxt] + nd, nxt / smax, is_wkend, dep_hour, is_rush])
        cur_seq, cur_delay = nxt, nd

    return pd.DataFrame(rows)


# Prévision de retard avec Prophet

def fit_prophet(d: pd.DataFrame, line: str, direction: str, societe: str):
    """Ajuste un modèle Prophet sur le retard moyen quotidien pour un (ligne, dir).

    Retourne le modèle Prophet ajusté. L'entrée `d` doit avoir delay_min et trip_start.
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
    """Prévoit `periods` jours en avance. Retourne ds, yhat, yhat_lower, yhat_upper."""
    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)


def serve_eta(model, baseline: pd.DataFrame, *, societe, line, direction, dep_time,
              current_seq: int, current_delay_min: float) -> pd.DataFrame:
    """ETA DE PRODUCTION : étant donné l'état en direct d'un bus (où il est + son retard actuel),
    fait avancer le modèle au prochain arrêt pour prédire le retard — et une ETA en heures —
    à chaque arrêt restant.

    Retourne une ligne par arrêt en aval : seq, expected_min (base de référence), pred_delay_min, eta.
    """
    b = baseline[(baseline["societe"] == societe) & (baseline["line"] == line)
                 & (baseline["dir"] == direction)].sort_values("seq")
    if b.empty:
        return pd.DataFrame(columns=["seq", "expected_min", "pred_delay_min", "eta"])
    dep_time = pd.Timestamp(dep_time)
    dep_hour, dow = dep_time.hour, dep_time.dayofweek
    is_weekend = int(dow in (5, 6))
    is_rush_hour = int(((7 <= dep_hour <= 9) or (16 <= dep_hour <= 19)) and not is_weekend)
    exp = dict(zip(b["seq"].astype(int), b["expected_min"]))
    smax = int(b["seq"].max())
    numeric_features = getattr(model, "feature_names_used_", FEATURES_NUM)

    cur_seq, cur_delay, rows = int(current_seq), float(current_delay_min), []
    while cur_seq < smax:
        nxt = cur_seq + 1
        if nxt not in exp:
            cur_seq = nxt
            continue
        x = _design(pd.DataFrame([{
            "dep_hour": dep_hour, "dow": dow, "is_weekend": is_weekend, "is_rush_hour": is_rush_hour,
            "seq": cur_seq, "seq_frac": cur_seq / smax,
            "delay_min": cur_delay, "elapsed_min": exp.get(cur_seq, 0.0) + cur_delay,
            "societe": societe, "line": line, "dir": direction,
            # pas de flux météo en direct branché -- HistGBM gère nativement le NaN
            # (branche apprise dédiée), voir add_weather()
            "rain_frac": np.nan,
        }]), numeric_features)
        nd = float(model.predict(x)[0])
        rows.append({"seq": nxt, "expected_min": round(exp[nxt], 1),
                     "pred_delay_min": round(nd, 1),
                     "eta": dep_time + pd.Timedelta(minutes=exp[nxt] + nd)})
        cur_seq, cur_delay = nxt, nd
    return pd.DataFrame(rows)
