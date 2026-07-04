"""Couche chatbot RAG — les opérateurs interrogent le réseau en langage naturel.

Architecture
------------
1. Base de connaissances (ChromaDB)
   Des documents texte sont générés à partir des résultats de fondation + anomalies :
   - Un document par ligne : géométrie, couverture, distance, retard typique
   - Un document par anomalie signalée : date, entreprise, ce qui s'est passé
   - Résumés par entreprise : jours de service, taux de correspondance, qualité de couverture

2. Récupération (sentence-transformers)
   La requête est encapsulée avec le même modèle utilisé pour encapsuler la base de connaissances.
   ChromaDB retourne les k documents les plus pertinents.

3. Génération (Groq / Llama 3)
   Le contexte récupéré + la question de l'utilisateur sont envoyés à Llama 3.
   Le modèle répond en langage naturel, fondé sur les faits récupérés.

Décision de conception clé : garder la base de connaissances petite et factuelle (pas de lignes
GPS brutes). Les documents sont des résumés agrégés — cela maintient la récupération rapide
et empêche les hallucinations à partir d'un contexte non pertinent.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL   = "llama-3.1-8b-instant"
COLLECTION_NAME     = "winicari_kb"


# Génération de documents — transformer les données structurées en texte consultable
def _line_doc(row: pd.Series) -> str:
    cov = row.get("coverage_type", "unknown")
    km  = row.get("route_km", 0)
    geo = row.get("geocoded_stops", 0)
    tot = row.get("total_stops", 0)
    return (
        f"Ligne {row['line']} exploitée par {row['societe']}. "
        f"Longueur de route : {km:.1f} km. "
        f"Couverture des arrêts : {cov} ({geo}/{tot} arrêts géocodés). "
        f"Pays : {row.get('country', 'Tunisie')}."
    )


def _trip_doc(row: pd.Series) -> str:
    dwell = row.get("max_dwell_s", 0)
    return (
        f"Trajet anormal détecté : ligne {row['line']} ({row['societe']}), "
        f"direction {row['dir']}, date {row['day']}, bus {row['bus']}. "
        f"Immobilisation max à un arrêt : {dwell:.0f} s ({dwell/60:.1f} min). "
        f"Taux de correspondance : {100*row['match_rate']:.0f}%. "
        f"Durée du trajet : {row['total_elapsed']:.0f} min."
    )


def _company_doc(societe: str, fa: pd.DataFrame) -> str:
    sub = fa[fa["societe"] == societe]
    lines = sub["line"].nunique()
    days  = sub["day"].nunique()
    match = 100 * sub["matched"].mean()
    trips = sub.groupby(["day", "line", "bus", "trip_id"]).ngroups
    return (
        f"Entreprise {societe} : exploite {lines} lignes dans le jeu de données de fondation. "
        f"{days} jours de service, {trips} trajets reconstruits. "
        f"Taux de correspondance GPS moyen : {match:.0f}%."
    )


def build_knowledge_base(fa: pd.DataFrame,
                         line_dist: pd.DataFrame,
                         anomalies: pd.DataFrame | None = None
                         ) -> tuple[list[str], list[str], list[dict]]:
    """Retourne (documents, ids, métadonnées) prêts pour l'ingestion dans ChromaDB."""
    docs, ids, metas = [], [], []

    # Documents de géométrie de ligne (dédupliquer sur (societe, line) d'abord)
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

    # Documents de résumé par entreprise
    for soc in fa["societe"].unique():
        doc = _company_doc(soc, fa)
        did = f"company_{soc}"
        docs.append(doc); ids.append(did)
        metas.append({"type": "company", "societe": str(soc)})

    # Documents d'anomalies (si fournis)
    if anomalies is not None:
        for i, row in anomalies.iterrows():
            doc = _trip_doc(row)
            did = f"anomaly_{i}"
            docs.append(doc); ids.append(did)
            metas.append({"type": "anomaly", "societe": str(row["societe"]),
                          "line": str(row["line"]), "day": str(row["day"])})

    return docs, ids, metas


# ChromaDB — construire / charger le magasin de vecteurs
def build_chroma(docs: list[str], ids: list[str], metas: list[dict],
                 persist_dir: str | Path,
                 embed_model: str = DEFAULT_EMBED_MODEL):
    """Encapsule les documents et persiste dans ChromaDB. Écrase toute collection existante."""
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
    print(f"ChromaDB : {col.count()} documents indexés dans {persist_dir}")
    return col, model


def load_chroma(persist_dir: str | Path,
                embed_model: str = DEFAULT_EMBED_MODEL):
    """Charge une collection ChromaDB existante."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(embed_model)
    return col, model


# Récupération
def retrieve(query: str, col, model, k: int = 5) -> list[str]:
    """Retourne les textes des k documents les plus pertinents pour la requête."""
    q_emb = model.encode([query]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    return res["documents"][0]


# Génération — Groq / Llama 3
SYSTEM_PROMPT = """Vous êtes un assistant intelligent pour WiniCari, une plateforme tunisienne \
de gestion des transports publics. Vous répondez aux questions des opérateurs et gestionnaires \
de compagnies de bus sur leur réseau — retards, anomalies, couverture GPS, distances de lignes, \
performances des entreprises, etc.

Répondez UNIQUEMENT à partir du contexte fourni. Si le contexte ne contient pas assez \
d'informations, dites-le clairement. Soyez concis et factuel. Utilisez des chiffres quand \
ils sont disponibles."""


def ask(query: str, col, embed_model, *,
        k: int = 5,
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None) -> dict:
    """Pipeline RAG complet : récupérer le contexte, générer la réponse.

    Retourne un dict avec les clés : answer, context (liste de docs récupérés).
    """
    from groq import Groq

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY non défini — ajoutez-le à .env ou passez api_key=")

    context_docs = retrieve(query, col, embed_model, k=k)
    context_text = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(context_docs))

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Contexte :\n{context_text}\n\nQuestion : {query}"},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "context": context_docs,
        "tokens_used": response.usage.total_tokens,
    }
