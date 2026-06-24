"""WiniCari AI Training Pipeline -- trains all 4 modules end-to-end.

Usage
-----
    conda activate bus-intelligence
    python -m src.train_pipeline

What it does
------------
  1. Delay         -- HistGBM + LSTM + Prophet  -> models/delay/
  2. GPS Fallback  -- Kalman + LSTM correction  -> models/fallback/
  3. Anomaly       -- Isolation Forest + LSTM AE -> models/anomaly/
  4. RAG Chatbot   -- ChromaDB index            -> data/processed/chroma_kb/

Prerequisites
-------------
  - data/processed/foundation_arrivals_full.parquet  (run build_foundation first)
  - data/processed/line_distances.parquet            (built by 02_preprocessing)
  - GROQ_API_KEY in .env or environment              (for RAG)
  - bus-intelligence conda environment               (PyTorch, prophet, filterpy, ...)

Loading models after training
------------------------------
    from src.models import delay, gps_fallback, anomaly, chatbot

    delay_models    = delay.load()
    fallback_models = gps_fallback.load()
    anomaly_models  = anomaly.load()
    chat_models     = chatbot.load()

    # Serve an ETA
    eta = delay.predict_eta(delay_models, societe="S.R.T.K", line="209",
                            direction="ALLER", dep_time="2026-06-15 06:00",
                            current_seq=5, current_delay_min=8.0)

    # Ask the chatbot
    result = chatbot.ask(chat_models, "Which line has the most anomalies?")
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
            "Required files not found:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nRun build_foundation first: python -m src.data.build_foundation --since 202501"
        )


def main():
    _check_prereqs()
    t0 = time.time()

    print("=" * 60)
    print("WiniCari AI Training Pipeline")
    print("=" * 60)

    # ── Module 1: Delay ──────────────────────────────────────────────────────
    print("\n[1/4] Delay Prediction  (HistGBM + LSTM + Prophet)")
    print("-" * 50)
    from src.models import delay
    delay.train(FOUNDATION, MODELS_DIR / "delay", epochs=30)

    # ── Module 2: GPS Fallback ───────────────────────────────────────────────
    print("\n[2/4] GPS Fallback  (Kalman + LSTM correction)")
    print("-" * 50)
    from src.models import gps_fallback
    gps_fallback.train(MODELS_DIR / "fallback")

    # ── Module 3: Anomaly Detection ──────────────────────────────────────────
    print("\n[3/4] Anomaly Detection  (Isolation Forest + LSTM Autoencoder)")
    print("-" * 50)
    from src.models import anomaly
    anomaly.train(FOUNDATION, MODELS_DIR / "anomaly")

    # ── Module 4: RAG Chatbot ────────────────────────────────────────────────
    print("\n[4/4] RAG Chatbot  (ChromaDB + Llama 3 via Groq)")
    print("-" * 50)
    from src.models import chatbot
    anomaly_trips = MODELS_DIR / "anomaly" / "trips_scored.parquet"
    chatbot.build(FOUNDATION, LINE_DISTANCES, anomaly_trips)

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All models trained in {elapsed/60:.1f} minutes")
    print(f"Artefacts saved to:")
    print(f"  {(MODELS_DIR / 'delay').resolve()}")
    print(f"  {(MODELS_DIR / 'fallback').resolve()}")
    print(f"  {(MODELS_DIR / 'anomaly').resolve()}")
    print(f"  {Path('data/processed/chroma_kb').resolve()}")
    print("\nLoad models for serving:")
    print("  from src.models import delay; models = delay.load()")


if __name__ == "__main__":
    main()
