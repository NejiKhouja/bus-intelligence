"""Détection d'anomalies dans les ventes de tickets — signal COMPLÉMENTAIRE au signal GPS
(`src/data/anomaly.py`), pas fusionné avec lui.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY_KEYS = ["societe", "line", "bus", "day"]
# Phase 2 : billetterie PAR ARRÊT D'ORIGINE, au grain (societe, line, bus, station, day) --
# `bus` inclus depuis le 2026-07-11 pour pouvoir scoper la répartition par arrêt à UN trajet
# précis (celui affiché dans "Jours signalés"), pas à toute la ligne ce jour-là (voir
# reference_db.populate_tickets_station_daily pour l'historique du bug que ça corrige).
STATION_KEYS = ["societe", "line", "bus", "station", "day"]


@dataclass(frozen=True)
class TicketAnomalyConfig:
    if_contamination: str | float = "auto"
    if_n_estimators: int = 200
    min_records_per_company: int = 30
    # En dessous de ce nombre de jours-bus d'historique, une ligne n'a pas son propre modèle
    # et replie sur le modèle société. POURQUOI par ligne d'abord : la normale billetterie
    # est une propriété de LA LIGNE (une intercity à ~20 DT/ticket vs une urbaine à ~2 DT),
    # pas de la société -- un modèle par société pooling toutes les lignes signalait 100% des
    # jours des lignes tarifairement atypiques (constaté : S.R.T.K ligne 220, 38/38 jours),
    # ce qui est un artefact de référence, pas de la fraude quotidienne.
    # NOTE : ce seuil est un PLANCHER D'ÉLIGIBILITÉ, pas la quantité de données utilisée --
    # chaque modèle de ligne s'entraîne toujours sur TOUT l'historique disponible de sa ligne
    # (jusqu'à 1 427 jours-bus pour TCV/3). Monter le seuil ne renforce aucun modèle, il ne
    # fait que disqualifier les petites lignes. Balayage mesuré (2026-07-07, 105 lignes,
    # médiane 7 jours-bus/ligne) : seuil 30 -> 39 lignes couvertes (96% des jours) ; 365 ->
    # 2 lignes (38%) et S.R.T.K/220 retombe à 100% signalée (l'artefact qu'on corrige).
    min_records_per_line: int = 30


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


def load_tickets_station_daily(conn) -> pd.DataFrame:
    """Charge `tickets_station_daily` (Phase 2, voir STATION_KEYS) -- même forme que
    `load_tickets_daily`, plus `station` (nom d'arrêt d'origine) : une ligne par
    (bus, arrêt, jour), pas juste (arrêt, jour) -- voir STATION_KEYS."""
    return pd.read_sql("""
        SELECT c.canonical_name AS societe, l.line_code AS line, t.bus,
               t.station_name AS station, t.day, t.nbr_ticket, t.recette
        FROM tickets_station_daily t
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


# direction + libellé de base ; le texte final est enrichi dans explain_days avec la
# médiane de LA LIGNE quand elle est disponible (bien plus juste pour juger qu'une
# comparaison à la normale société entière -- une ligne intercity à 20 DT/ticket est
# normale pour ELLE même si la société vend surtout de l'urbain à 1.5 DT).
_REASON_BUILDERS = {
    "nbr_ticket": ("low", "Volume de tickets anormalement bas"),
    "recette":    ("low", "Recette anormalement basse"),
    "avg_fare":   ("high", "Prix moyen par ticket anormal"),
}

_FEATURE_FMT = {
    "nbr_ticket": lambda v: f"{int(v)} tickets",
    "recette":    lambda v: f"{v:.0f} DT",
    "avg_fare":   lambda v: f"{v:.2f} DT/ticket",
}


def explain_days(scored: pd.DataFrame, if_mean_by_soc: dict, z_thresh: float = 1.5,
                 baseline: pd.DataFrame | None = None, group_key: str = "bus") -> pd.DataFrame:
    """Ajoute des raisons explicables, en z-score contre la normale DE CETTE SOCIÉTÉ
    (même principe que `anomaly.explain_trips`).

    `baseline` (optionnel) : l'historique complet des jours scorés (days_scored) -- sert à
    calculer les MÉDIANES PAR LIGNE et PAR {group_key} (contexte de jugement : « 20.62
    DT/ticket » ne dit rien seul ; « 20.62 DT vs ~11 DT médian sur cette ligne » permet de
    trancher), plus `line_anomaly_rate` : si ~100% des jours d'une ligne sont signalés,
    c'est presque sûrement une structure tarifaire différente de la normale société (ligne
    intercity vs urbaine), PAS une fraude quotidienne -- signal à afficher tel quel, pas à
    cacher.

    `group_key` : "bus" (défaut, grain historique) ou "station" (Phase 2, billetterie par
    arrêt d'origine -- voir STATION_KEYS) -- seul ce sous-groupe change, la logique
    (médianes/z-scores/raisons) est identique aux deux grains.
    """
    out = scored.copy()

    # Contexte médian par ligne et par {group_key} + taux d'anomalie de la ligne (si historique fourni)
    if baseline is not None and len(baseline):
        line_med = (baseline.groupby(["societe", "line"])[FEATURES].median()
                    .add_prefix("line_median_").reset_index())
        sub_med = (baseline.groupby(["societe", "line", group_key])[FEATURES].median()
                  .add_prefix(f"{group_key}_median_").reset_index())
        out = out.merge(line_med, on=["societe", "line"], how="left")
        out = out.merge(sub_med, on=["societe", "line", group_key], how="left")
        if "anomaly" in baseline.columns:
            line_rate = (baseline.groupby(["societe", "line"])
                         .agg(line_anomaly_rate=("anomaly", "mean"),
                              line_n_days=("anomaly", "size")).reset_index())
            out = out.merge(line_rate, on=["societe", "line"], how="left")

    reasons_col, severity_col = [], []
    z_cols: dict[str, list] = {f: [] for f in FEATURES}
    for _, row in out.iterrows():
        soc = row["societe"]
        # clé (societe, ligne) d'abord (modèles par ligne), repli sur la clé societe seule
        # (anciens artefacts par société / lignes sous le seuil min_records_per_line)
        mean, std = if_mean_by_soc.get((soc, row["line"]),
                                       if_mean_by_soc.get(soc, (None, None)))
        if mean is None:
            reasons_col.append([])
            severity_col.append("low")
            for f in FEATURES:
                z_cols[f].append(None)
            continue
        vals = row[FEATURES].values.astype(float)
        z = (vals - np.asarray(mean, dtype=float)) / np.asarray(std, dtype=float)
        scored_feats = []
        for i, f in enumerate(FEATURES):
            z_cols[f].append(round(float(z[i]), 2))
            direction, _ = _REASON_BUILDERS[f]
            signed = z[i] if direction == "high" else -z[i]
            if signed >= z_thresh:
                scored_feats.append((signed, f, row[f]))
        scored_feats.sort(reverse=True)

        reasons = []
        for signed, f, v in scored_feats[:2]:
            _, label = _REASON_BUILDERS[f]
            txt = f"{label} ({_FEATURE_FMT[f](v)}"
            lm = row.get(f"line_median_{f}")
            if lm is not None and pd.notna(lm):
                txt += f" vs ~{_FEATURE_FMT[f](lm)} médian sur cette ligne"
            txt += f", z={signed:+.1f})"
            reasons.append(txt)
        reasons_col.append(reasons)

        # Gravité : lisible d'un coup d'œil, dérivée des mêmes z-scores que les raisons
        max_signed = scored_feats[0][0] if scored_feats else 0.0
        if bool(row.get("anomaly", False)) and (max_signed >= 3.0 or len(scored_feats) >= 2):
            severity_col.append("high")
        elif bool(row.get("anomaly", False)):
            severity_col.append("medium")
        else:
            severity_col.append("low")

    out["reasons"] = reasons_col
    out["severity"] = severity_col
    for f in FEATURES:
        out[f"z_{f}"] = z_cols[f]
    return out
