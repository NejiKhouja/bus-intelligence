"""RAG Chatbot layer — operators query the network in plain language.

Architecture
------------
1. Knowledge base (ChromaDB)
   Text documents are generated from the foundation + anomaly results:
   - One document per line: geometry, coverage, distance, typical delay
   - One document per flagged anomaly: date, company, what happened
   - Company summaries: service days, match rate, coverage quality

2. Retrieval (sentence-transformers)
   Query is embedded with the same model used to embed the knowledge base.
   ChromaDB returns the top-k most relevant documents.

3. Generation (Groq / Llama 3)
   The retrieved context + the user's question are sent to Llama 3.
   The model answers in plain language, grounded in the retrieved facts.

Key design decision: keep the knowledge base small and factual (no raw GPS
rows). Documents are aggregated summaries — this keeps retrieval fast and
prevents hallucination from irrelevant context.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL   = "llama-3.1-8b-instant"
COLLECTION_NAME     = "winicari_kb"


# ─────────────────────────────────────────────────────────────────────────────
# Document generation — turn structured data into searchable text
# ─────────────────────────────────────────────────────────────────────────────

def _line_doc(row: pd.Series) -> str:
    cov = row.get("coverage_type", "unknown")
    km  = row.get("route_km", 0)
    geo = row.get("geocoded_stops", 0)
    tot = row.get("total_stops", 0)
    return (
        f"Line {row['line']} operated by {row['societe']}. "
        f"Route length: {km:.1f} km. "
        f"Stop coverage: {cov} ({geo}/{tot} stops geocoded). "
        f"Country: {row.get('country', 'Tunisia')}."
    )


def _trip_doc(row: pd.Series) -> str:
    dwell = row.get("max_dwell_s", 0)
    return (
        f"Anomalous trip detected: line {row['line']} ({row['societe']}), "
        f"direction {row['dir']}, date {row['day']}, bus {row['bus']}. "
        f"Max stop dwell: {dwell:.0f} s ({dwell/60:.1f} min). "
        f"Match rate: {100*row['match_rate']:.0f}%. "
        f"Trip duration: {row['total_elapsed']:.0f} min."
    )


def _company_doc(societe: str, fa: pd.DataFrame) -> str:
    sub = fa[fa["societe"] == societe]
    lines = sub["line"].nunique()
    days  = sub["day"].nunique()
    match = 100 * sub["matched"].mean()
    trips = sub.groupby(["day", "line", "bus", "trip_id"]).ngroups
    return (
        f"Company {societe}: operates {lines} lines in the foundation dataset. "
        f"{days} service days, {trips} trips reconstructed. "
        f"Average GPS stop match rate: {match:.0f}%."
    )


def build_knowledge_base(fa: pd.DataFrame,
                         line_dist: pd.DataFrame,
                         anomalies: pd.DataFrame | None = None
                         ) -> tuple[list[str], list[str], list[dict]]:
    """Return (documents, ids, metadatas) ready for ChromaDB ingestion."""
    docs, ids, metas = [], [], []

    # Line geometry documents (deduplicate on (societe, line) first)
    seen_lines = set()
    for _, row in line_dist.iterrows():
        key = (str(row["societe"]), str(row["line"]))
        if key in seen_lines:
            continue
        seen_lines.add(key)
        doc = _line_doc(row)
        did = f"line_{row['societe']}_{row['line']}"
        docs.append(doc); ids.append(did)
        metas.append({"type": "line", "societe": str(row["societe"]),
                      "line": str(row["line"])})

    # Company summary documents
    for soc in fa["societe"].unique():
        doc = _company_doc(soc, fa)
        did = f"company_{soc}"
        docs.append(doc); ids.append(did)
        metas.append({"type": "company", "societe": str(soc)})

    # Anomaly documents (if supplied)
    if anomalies is not None:
        for i, row in anomalies.iterrows():
            doc = _trip_doc(row)
            did = f"anomaly_{i}"
            docs.append(doc); ids.append(did)
            metas.append({"type": "anomaly", "societe": str(row["societe"]),
                          "line": str(row["line"]), "day": str(row["day"])})

    return docs, ids, metas


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB — build / load the vector store
# ─────────────────────────────────────────────────────────────────────────────

def build_chroma(docs: list[str], ids: list[str], metas: list[dict],
                 persist_dir: str | Path,
                 embed_model: str = DEFAULT_EMBED_MODEL):
    """Embed documents and persist to ChromaDB. Overwrites any existing collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    persist_dir = str(persist_dir)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME)

    model = SentenceTransformer(embed_model)
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=64).tolist()
    col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    print(f"ChromaDB: {col.count()} documents indexed at {persist_dir}")
    return col, model


def load_chroma(persist_dir: str | Path,
                embed_model: str = DEFAULT_EMBED_MODEL):
    """Load an existing ChromaDB collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(embed_model)
    return col, model


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str, col, model, k: int = 5) -> list[str]:
    """Return the top-k most relevant document texts for the query."""
    q_emb = model.encode([query]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    return res["documents"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Generation — Groq / Llama 3
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent assistant for WiniCari, a Tunisian public transport \
management platform. You answer questions from bus company operators and managers about \
their network — delays, anomalies, GPS coverage, line distances, company performance, etc.

Answer ONLY from the provided context. If the context does not contain enough information, \
say so clearly. Be concise and factual. Use numbers when available."""


def ask(query: str, col, embed_model, *,
        k: int = 5,
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None) -> dict:
    """Full RAG pipeline: retrieve context, generate answer.

    Returns dict with keys: answer, context (list of retrieved docs).
    """
    from groq import Groq

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set — add it to .env or pass api_key=")

    context_docs = retrieve(query, col, embed_model, k=k)
    context_text = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(context_docs))

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "context": context_docs,
        "tokens_used": response.usage.total_tokens,
    }
