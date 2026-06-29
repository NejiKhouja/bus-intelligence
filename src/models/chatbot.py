"""Module chatbot RAG — construire, charger, interroger.

Cycle de vie complet pour le Module 4 :
  build()  -> encapsule les documents et persiste l'index ChromaDB
  load()   -> charge l'index ChromaDB existant depuis le disque
  ask()    -> pipeline RAG complet : récupérer -> générer avec Llama 3 via Groq
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data import rag as _rag

CHROMA_DIR = Path("data/processed/chroma_kb")


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

def build(foundation_path: str | Path,
          line_distances_path: str | Path,
          anomaly_trips_path: str | Path | None = None,
          chroma_dir: str | Path = CHROMA_DIR) -> dict:
    """Construit la base de connaissances ChromaDB à partir de la fondation + artefacts d'anomalies.

    Documents générés :
      - Un par (societe, line) : géométrie, couverture, distance
      - Un par entreprise : trajets, jours de service, taux de correspondance
      - Un par trajet anormal (si anomaly_trips_path est fourni)

    Écrase toute collection existante. Relancer pour rafraîchir après une reconstruction des données.
    """
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print("  Chargement des données pour la base de connaissances...")
    fa = pd.read_parquet(foundation_path)
    fa["arrival"] = pd.to_datetime(fa["arrival"])
    fa["trip_start"] = pd.to_datetime(fa["trip_start"])

    ld = pd.read_parquet(line_distances_path)

    anomalies = None
    if anomaly_trips_path is not None:
        p = Path(anomaly_trips_path)
        if p.exists():
            anomalies = pd.read_parquet(p)
            if "anomaly" in anomalies.columns:
                anomalies = anomalies[anomalies["anomaly"]].copy()

    docs, ids, metas = _rag.build_knowledge_base(fa, ld, anomalies)
    col, embed_model = _rag.build_chroma(docs, ids, metas,
                                         str(chroma_dir), _rag.DEFAULT_EMBED_MODEL)

    n_anomaly = sum(1 for m in metas if m["type"] == "anomaly")
    print(f"  -> Base de connaissances construite : {col.count()} docs "
          f"({len(ids)-n_anomaly} factuels + {n_anomaly} anomalies)")
    return {"col": col, "embed_model": embed_model}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load(chroma_dir: str | Path = CHROMA_DIR) -> dict:
    """Charge une base de connaissances ChromaDB existante depuis le disque.

    Retourne dict : col, embed_model.
    """
    col, embed_model = _rag.load_chroma(str(chroma_dir), _rag.DEFAULT_EMBED_MODEL)
    print(f"Base de connaissances chargée : {col.count()} documents")
    return {"col": col, "embed_model": embed_model}


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

def ask(models: dict, query: str,
        *,
        api_key: str | None = None,
        k: int = 5) -> dict:
    """Pipeline RAG complet : récupérer le contexte -> générer la réponse avec Llama 3.

    api_key : clé API Groq. Retombe sur la variable d'environnement GROQ_API_KEY.
    Retourne dict : answer (str), context (list[str]), tokens_used (int).
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    return _rag.ask(
        query, models["col"], models["embed_model"],
        k=k, llm_model=_rag.DEFAULT_LLM_MODEL, api_key=api_key,
    )
