"""RAG Chatbot module -- build, load, ask.

Complete lifecycle for Module 4:
  build()  -> embeds documents and persists ChromaDB index
  load()   -> loads existing ChromaDB index from disk
  ask()    -> full RAG pipeline: retrieve -> generate with Llama 3 via Groq
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data import rag as _rag

CHROMA_DIR = Path("data/processed/chroma_kb")


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build(foundation_path: str | Path,
          line_distances_path: str | Path,
          anomaly_trips_path: str | Path | None = None,
          chroma_dir: str | Path = CHROMA_DIR) -> dict:
    """Build the ChromaDB knowledge base from foundation + anomaly artefacts.

    Documents generated:
      - One per (societe, line): geometry, coverage, distance
      - One per company: trips, service days, match rate
      - One per anomalous trip (if anomaly_trips_path provided)

    Overwrites any existing collection. Call again to refresh after a data rebuild.
    """
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading data for knowledge base...")
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
    print(f"  -> Knowledge base built: {col.count()} docs "
          f"({len(ids)-n_anomaly} factual + {n_anomaly} anomaly)")
    return {"col": col, "embed_model": embed_model}


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load(chroma_dir: str | Path = CHROMA_DIR) -> dict:
    """Load an existing ChromaDB knowledge base from disk.

    Returns dict: col, embed_model.
    """
    col, embed_model = _rag.load_chroma(str(chroma_dir), _rag.DEFAULT_EMBED_MODEL)
    print(f"Knowledge base loaded: {col.count()} documents")
    return {"col": col, "embed_model": embed_model}


# ─────────────────────────────────────────────────────────────────────────────
# Serve
# ─────────────────────────────────────────────────────────────────────────────

def ask(models: dict, query: str,
        *,
        api_key: str | None = None,
        k: int = 5) -> dict:
    """Full RAG pipeline: retrieve context -> generate answer with Llama 3.

    api_key: Groq API key. Falls back to GROQ_API_KEY env var.
    Returns dict: answer (str), context (list[str]), tokens_used (int).
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    return _rag.ask(
        query, models["col"], models["embed_model"],
        k=k, llm_model=_rag.DEFAULT_LLM_MODEL, api_key=api_key,
    )
