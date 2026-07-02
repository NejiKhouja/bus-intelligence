"""Détection d'anomalies dans les ventes de tickets — signal COMPLÉMENTAIRE au signal GPS
(`src/data/anomaly.py`), pas fusionné avec lui.

Pourquoi un modèle séparé plutôt qu'une fusion dans les caractéristiques de trajet GPS :
`tickets_daily` est agrégé par (société, ligne, bus, JOUR) alors que l'anomalie GPS score au
niveau TRAJET -- un bus peut faire plusieurs trajets par jour, donc répartir le nombre de
tickets d'une journée entre trajets individuels demanderait une hypothèse inventée qu'on ne
peut pas dériver honnêtement des données. Ce module reste donc au grain JOUR : « ce bus a-t-il
vendu un nombre de tickets/une recette anormal ce jour-là sur cette ligne, par rapport à sa
propre normale ? ». Mêmes principes que l'anomalie GPS : un Isolation Forest PAR SOCIÉTÉ
(ce qui est un jour normal pour TCV ne l'est pas forcément pour S.R.T.K), `contamination='auto'`
data-driven, pas de seuil forcé.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY_KEYS = ["societe", "line", "bus", "day"]


@dataclass(frozen=True)
class TicketAnomalyConfig:
    if_contamination: str | float = "auto"
    if_n_estimators: int = 200
    min_records_per_company: int = 30


FEATURES = ["nbr_ticket", "recette", "avg_fare"]


def load_tickets_daily(conn) -> pd.DataFrame:
    """Charge `tickets_daily` depuis le reference DB, avec les noms de société/ligne résolus
    (pas juste les FK) pour rester lisible en aval."""
    return pd.read_sql("""
        SELECT c.canonical_name AS societe, l.line_code AS line, t.bus, t.day,
               t.nbr_ticket, t.recette
        FROM tickets_daily t
        JOIN companies c ON c.company_id = t.company_id
        JOIN lines l ON l.line_id = t.line_id
    """, conn)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `avg_fare` (recette/ticket -- un ticket_daily anormalement bon marché/cher peut
    signaler une fraude ou une erreur de caisse, indépendamment du volume)."""
    df = df.copy()
    df["avg_fare"] = np.where(df["nbr_ticket"] > 0, df["recette"] / df["nbr_ticket"], 0.0)
    return df


def _scale(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """Normalise X (z-score) ; retourne (X_normalisé, mean, std). Identique à
    `anomaly._scale` -- dupliqué plutôt que ré-exporté pour garder ce module autonome
    (grain différent, pas de dépendance croisée avec l'anomalie GPS)."""
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def train_isolation_forest(day_rows: pd.DataFrame, cfg: TicketAnomalyConfig):
    """Entraîne l'Isolation Forest sur les caractéristiques journalières billetterie.
    Retourne (modèle, scaler_mean, scaler_std)."""
    from sklearn.ensemble import IsolationForest
    X = day_rows[FEATURES].values.astype(float)
    X_s, mean, std = _scale(X)
    model = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        contamination=cfg.if_contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_s)
    return model, mean, std


def score_days(model, mean: np.ndarray, std: np.ndarray, day_rows: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `if_score` (brut, plus élevé = plus normal) et `anomaly` (bool)."""
    X = day_rows[FEATURES].values.astype(float)
    X_s, _, _ = _scale(X, mean, std)
    day_rows = day_rows.copy()
    day_rows["if_score"] = model.score_samples(X_s)
    day_rows["anomaly"] = model.predict(X_s) == -1
    return day_rows


_REASON_BUILDERS = {
    "nbr_ticket": ("low", lambda v: f"Volume de tickets anormalement bas ({int(v)} tickets ce jour-là)"),
    "recette":    ("low", lambda v: f"Recette anormalement basse ({v:.0f} DT ce jour-là)"),
    "avg_fare":   ("high", lambda v: f"Prix moyen par ticket anormal ({v:.2f} DT/ticket -- possible erreur de caisse ou fraude)"),
}


def explain_days(scored: pd.DataFrame, if_mean_by_soc: dict, z_thresh: float = 1.5) -> pd.DataFrame:
    """Ajoute des raisons explicables, en z-score contre la normale DE CETTE SOCIÉTÉ
    (même principe que `anomaly.explain_trips`)."""
    out = scored.copy()
    reasons_col = []
    for _, row in out.iterrows():
        soc = row["societe"]
        mean, std = if_mean_by_soc.get(soc, (None, None))
        if mean is None:
            reasons_col.append([])
            continue
        vals = row[FEATURES].values.astype(float)
        z = (vals - np.asarray(mean, dtype=float)) / np.asarray(std, dtype=float)
        scored_feats = []
        for i, f in enumerate(FEATURES):
            direction, _ = _REASON_BUILDERS[f]
            signed = z[i] if direction == "high" else -z[i]
            if signed >= z_thresh:
                scored_feats.append((signed, f, row[f]))
        scored_feats.sort(reverse=True)
        reasons_col.append([_REASON_BUILDERS[f][1](v) for _, f, v in scored_feats[:2]])
    out["reasons"] = reasons_col
    return out
