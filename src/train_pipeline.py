"""Pipeline d'entraînement IA WiniCari — entraîne les 4 modules de bout en bout.

Utilisation
-----------
    conda activate bus-intelligence
    python -m src.train_pipeline

Ce que ça fait
--------------
  1. Retard         -- HistGBM + LSTM + Prophet  -> models/delay/
  2. Repli GPS      -- Kalman + correction LSTM  -> models/fallback/
  3. Anomalies      -- Isolation Forest + LSTM AE -> models/anomaly/
  4. Chatbot RAG    -- Index ChromaDB            -> data/processed/chroma_kb/

Prérequis
---------
  - data/processed/foundation_arrivals_full.parquet  (exécuter build_foundation d'abord)
  - data/processed/line_distances.parquet            (construit par 02_preprocessing)
  - GROQ_API_KEY dans .env ou l'environnement        (pour le RAG)
  - environnement conda bus-intelligence             (PyTorch, prophet, filterpy, ...)

Chargement des modèles après l'entraînement
--------------------------------------------
    from src.models import delay, gps_fallback, anomaly, chatbot

    delay_models    = delay.load()
    fallback_models = gps_fallback.load()
    anomaly_models  = anomaly.load()
    chat_models     = chatbot.load()

    # Servir une ETA
    eta = delay.predict_eta(delay_models, societe="S.R.T.K", line="209",
                            direction="ALLER", dep_time="2026-06-15 06:00",
                            current_seq=5, current_delay_min=8.0)

    # Interroger le chatbot
    result = chatbot.ask(chat_models, "Quelle ligne a le plus d'anomalies ?")
    print(result["answer"])
"""
from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FOUNDATION = Path("data/processed/foundation_arrivals_full.parquet")
LINE_DISTANCES = Path("data/processed/line_distances.parquet")
MODELS_DIR = Path("models")


def _check_prereqs():
    missing = []
    if not FOUNDATION.exists():
        missing.append(str(FOUNDATION))
    if not LINE_DISTANCES.exists():
        missing.append(str(LINE_DISTANCES))
    if missing:
        raise FileNotFoundError(
            "Fichiers requis introuvables :\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nExécuter build_foundation d'abord : python -m src.data.build_foundation --since 202501"
        )


def main():
    _check_prereqs()
    t0 = time.time()

    print("=" * 60)
    print("Pipeline d'entraînement IA WiniCari")
    print("=" * 60)

    # ── Module 1 : Retard ────────────────────────────────────────────────────
    print("\n[1/4] Prédiction de retard  (HistGBM + LSTM + Prophet)")
    print("-" * 50)
    from src.models import delay
    delay.train(FOUNDATION, MODELS_DIR / "delay", epochs=30)

    # ── Module 2 : Repli GPS ─────────────────────────────────────────────────
    print("\n[2/4] Repli GPS  (Kalman + correction LSTM)")
    print("-" * 50)
    from src.models import gps_fallback
    gps_fallback.train(MODELS_DIR / "fallback")

    # ── Module 3 : Détection d'anomalies ─────────────────────────────────────
    print("\n[3/4] Détection d'anomalies  (Isolation Forest + Autoencodeur LSTM)")
    print("-" * 50)
    from src.models import anomaly
    anomaly.train(FOUNDATION, MODELS_DIR / "anomaly")

    # ── Module 4 : Chatbot RAG ───────────────────────────────────────────────
    print("\n[4/4] Chatbot RAG  (ChromaDB + Llama 3 via Groq)")
    print("-" * 50)
    from src.models import chatbot
    anomaly_trips = MODELS_DIR / "anomaly" / "trips_scored.parquet"
    chatbot.build(FOUNDATION, LINE_DISTANCES, anomaly_trips)

    # ── Terminé ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Tous les modèles entraînés en {elapsed/60:.1f} minutes")
    print(f"Artefacts sauvegardés dans :")
    print(f"  {(MODELS_DIR / 'delay').resolve()}")
    print(f"  {(MODELS_DIR / 'fallback').resolve()}")
    print(f"  {(MODELS_DIR / 'anomaly').resolve()}")
    print(f"  {Path('data/processed/chroma_kb').resolve()}")
    print("\nCharger les modèles pour le service :")
    print("  from src.models import delay; models = delay.load()")


if __name__ == "__main__":
    main()
