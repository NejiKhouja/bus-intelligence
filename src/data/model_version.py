"""Versionnement léger des artefacts modèles -- écrit/lit models/models_version.json.

PAS un service de registre de modèles : juste un petit fichier JSON qui enregistre QUAND et
depuis QUEL commit git le lot actuel de models/ a été construit, plus les métriques scalaires
que chaque train() retourne déjà (MAE, nombre d'anomalies, etc.) -- suffisant pour répondre
"quel est cet artefact ?" depuis /health sans nouvelle infrastructure. Le réentraînement
lui-même reste une commande manuelle (voir docs/DEPLOYMENT.md) ; ce module ne fait
qu'enregistrer ce qui vient de se passer.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VERSION_FILE = Path("models") / "models_version.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def scalars_only(d: dict) -> dict:
    """Filtre un dict retourné par train() aux seules valeurs sérialisables en JSON --
    évite de coupler ce module aux clés exactes de chaque module (modèles/dataframes
    inclus dans ces dicts sont silencieusement ignorés, pas une liste blanche à maintenir)."""
    return {k: v for k, v in d.items() if v is None or isinstance(v, (int, float, str, bool))}


def write_version_file(metrics: dict, out_path: Path = VERSION_FILE) -> dict:
    """metrics : {"delay": {...}, "fallback": {...}, "anomaly": {...}} -- un dict de
    scalaires par module (voir scalars_only).

    Fusionne avec le fichier existant plutôt que de l'écraser : quand
    train_pipeline.py est lancé avec certains modules commentés (entraînement
    partiel), les modules absents de `metrics` gardent leur dernière entrée connue
    au lieu de disparaître alors que leurs artefacts restent sur disque et servent."""
    out_path = Path(out_path)
    existing = read_version_file(out_path) or {}
    modules = dict(existing.get("modules") or {})
    modules.update(metrics)
    info = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "modules": modules,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return info


def read_version_file(path: Path = VERSION_FILE) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
