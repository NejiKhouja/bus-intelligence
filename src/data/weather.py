"""Signal météo journalier pour le modèle de retard — dérivé de `OpenData.historiqueJourMeteo`.

Cette collection ne fournit qu'une chaîne de condition météo PAR STATION PAR JOUR (pas de
température/précipitation chiffrée), et seulement jusqu'à ~septembre 2025 (alors que les
trajets couvrent jusqu'à mi-2026) -- même dans sa propre plage, seuls ~246 jours sur ~547
(45%) ont au moins une station qui a enregistré quelque chose.

C'est un signal d'ENTRAÎNEMENT réel (explique une partie de la variance historique du retard)
mais qui ne peut PAS améliorer une ETA en direct aujourd'hui : aucune source météo en direct
n'est branchée. `delay.serve_eta()` passe donc toujours `rain_frac=NaN` pour les requêtes en
direct -- ce que HistGBM gère nativement (valeur manquante = branche apprise dédiée), sans
qu'une imputation soit nécessaire.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Ancré sur l'emplacement du module (pas un chemin relatif au répertoire de travail courant) --
# un chemin relatif se résout différemment selon que l'appelant est un script lancé depuis la
# racine du dépôt ou un notebook (dont le cwd d'exécution est `notebooks/`), et se résoudrait
# alors silencieusement vers un fichier inexistant plutôt que de lever une erreur claire.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = ROOT / "data" / "processed" / "day_weather.parquet"


def build_day_weather_cache(od_db, out_path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Agrège `OpenData.historiqueJourMeteo` (condition météo par station par jour) en UNE
    ligne par jour calendaire : `rain_frac` = fraction des stations ayant rapporté une donnée
    ce jour-là qui ont signalé de la pluie (0.0 à 1.0).

    Une fraction plutôt qu'un simple "au moins une station" : avec ~250 stations à travers
    tout le pays, un booléen "any" serait vrai presque tous les jours (une pluie localisée
    quelque part en Tunisie n'implique pas la pluie sur la ligne concernée) -- la fraction
    donne un signal gradué (jour largement pluvieux vs quelques stations isolées) sans
    nécessiter de choisir arbitrairement une station de référence par ligne.
    """
    rows = []
    for doc in od_db["historiqueJourMeteo"].find({}, {"historiqueJours": 1}):
        for d in doc.get("historiqueJours", []):
            rows.append({"date": d.get("date"), "conditionMeteo": d.get("conditionMeteo", "")})

    raw = pd.DataFrame(rows)
    raw["day"] = pd.to_datetime(raw["date"], format="%Y/%m/%d", errors="coerce").dt.strftime("%Y%m%d")
    raw = raw.dropna(subset=["day"])
    raw["is_rain_flag"] = raw["conditionMeteo"].str.contains("Pluie", na=False)

    day_weather = raw.groupby("day")["is_rain_flag"].mean().reset_index()
    day_weather = day_weather.rename(columns={"is_rain_flag": "rain_frac"})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    day_weather.to_parquet(out_path, index=False)
    return day_weather


def load_day_weather(path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame | None:
    """Charge le cache météo journalier, ou None si `build_day_weather_cache` n'a pas encore été lancé."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)
