# WiniCari — AI Layer for Tunisian Bus Operations

An AI layer built on top of the WiniCari bus-management platform (MongoDB). It never
modifies that source system — it reads from it, builds a clean **reference database**,
and trains four models on top: delay/ETA prediction, GPS gap fallback, anomaly detection,
and a RAG chatbot.

For the full story (raw-MongoDB audit, methodology, before/after numbers, known
limitations) see **[docs/DATA_PIPELINE_REPORT.md](docs/DATA_PIPELINE_REPORT.md)**. This
file is intentionally just the operational quick-start.

## Pipeline order

```
MongoDB (raw: winicari, OpenData, Historique_Tickets, Historique_pos)
        │
        ▼
1. src/build_reference_db.py        → data/reference/winicari_reference.db
        │                             (companies, stops, lines, line_stops, trips,
        │                              trip_stops, tickets_daily)
        ▼
2. export_foundation_parquet()       → data/processed/foundation_arrivals_full.parquet
        │                             (the single flat table every model reads)
        ▼
3. src/train_pipeline.py             → models/delay/, models/fallback/, models/anomaly/,
   (or notebooks 03-06 individually)   data/processed/chroma_kb/ (chatbot)
        │
        ▼
4. src/api/main.py + src/dashboard/  → serves predictions, reads the reference DB +
                                        foundation parquet + trained models
```

Step 1 is the only step that talks to MongoDB. Everything after it reads local files —
regenerate step 1 whenever the source MongoDB data changes.

## Setup

```bash
conda create -n bus-intelligence python=3.11
conda activate bus-intelligence
pip install -r requirements.txt

cp .env.example .env   # fill in MONGO_URL and GROQ_API_KEY
```

## 1. Seed / refresh the reference database

```bash
conda activate bus-intelligence
python -m src.build_reference_db                 # reference tables only, <1 minute
python -m src.build_reference_db --with-trips     # + full GPS trip reconstruction, ~40 min
python -m src.build_reference_db --with-trips --since 20250101 --until 20250630
python -m src.build_reference_db --with-trips --company S.R.T.BIZERTE --company TUS
```
`--with-trips` is only needed when the GPS history itself changed (new company connected,
longer window wanted) — an admin who just fixed a station name doesn't need the 40-minute
rebuild. This command is always admin-triggered; nothing here runs on a schedule.

## 2. Train the models

Two equivalent paths — pick whichever fits:

```bash
python -m src.train_pipeline        # all 4 modules end-to-end
```
or open any of `notebooks/03_delay.ipynb` .. `06_rag_chatbot.ipynb` and run it top to
bottom — each notebook's final cell calls the same `src.models.X.train()`/`build()`
function the pipeline calls, saving to the same `models/X/` location.

## Repository layout

```
winicari/
├── data/
│   ├── reference/winicari_reference.db        # the reference DB (step 1's output)
│   └── processed/
│       ├── foundation_arrivals_full.parquet   # flat trip/stop table every model reads
│       ├── day_weather.parquet                # rain_frac cache for the delay model
│       └── chroma_kb/                         # RAG chatbot vector store
│
├── models/{delay,fallback,anomaly,ticket_anomaly}/   # trained artifacts per module
│
├── notebooks/
│   ├── 01_eda.ipynb                    # original raw-MongoDB EDA
│   ├── 02_preprocessing.ipynb          # foundation build walkthrough
│   ├── 03_delay.ipynb                  # Module 1 — train/explore
│   ├── 04_gps_fallback.ipynb           # Module 2 — train/explore
│   ├── 05_anomaly.ipynb                # Module 3 — train/explore
│   ├── 06_rag_chatbot.ipynb            # Module 4 — train/explore
│   ├── 07_evaluation.ipynb             # cross-module evaluation
│   ├── 08_reference_db_eda.ipynb       # EDA of the reference DB + before/after
│   └── archive/                        # earlier station-enrichment investigation notebooks
│
├── src/
│   ├── build_reference_db.py     # admin entrypoint: seed/refresh the reference DB
│   ├── train_pipeline.py         # admin entrypoint: train all 4 modules
│   │
│   ├── data/                     # feature engineering + training logic (no artifacts saved)
│   │   ├── db.py                 #   MongoDB connection helper
│   │   ├── reference_db.py       #   reference DB schema + populate_*/export_* functions
│   │   ├── stations.py           #   stop-name resolution (OpenData + ticket-order matching)
│   │   ├── foundation.py         #   GPS ping -> trip reconstruction (used by reference_db.py)
│   │   ├── build_foundation.py   #   legacy direct-MongoDB foundation build (superseded by
│   │   │                         #   reference_db.export_foundation_parquet, kept for reference)
│   │   ├── delay.py              #   delay features (incl. rush-hour, weather join) + models
│   │   ├── weather.py            #   day-level rain_frac cache for delay.py
│   │   ├── fallback.py           #   Kalman filter + GPS fallback logic
│   │   ├── anomaly.py            #   per-trip anomaly feature engineering
│   │   ├── ticket_anomaly.py     #   per-day ticket-sale anomaly features
│   │   └── rag.py                #   ChromaDB knowledge-base build/query
│   │
│   ├── models/                   # train() / load() / predict() wrappers — these persist to models/
│   │   ├── delay.py
│   │   ├── gps_fallback.py
│   │   ├── anomaly.py
│   │   ├── ticket_anomaly.py
│   │   └── chatbot.py
│   │
│   ├── api/main.py               # FastAPI serving layer
│   └── dashboard/                # Streamlit dashboard
│
├── docs/
│   ├── DATA_PIPELINE_REPORT.md   # full methodology, MongoDB audit, before/after, limitations
│   ├── GLOSSARY.md
│   └── notes/                    # raw working notes (superseded by DATA_PIPELINE_REPORT.md)
├── requirements.txt
└── docker-compose.yml
```

## Using trained models

```python
from src.models import delay, gps_fallback, anomaly, chatbot

delay_models    = delay.load()
fallback_models = gps_fallback.load()
anomaly_models  = anomaly.load()
chat_models     = chatbot.load()

eta = delay.predict_eta(delay_models, societe="S.R.T.K", line="209", direction="ALLER",
                         dep_time="2026-06-15 06:00", current_seq=5, current_delay_min=8.0)
answer = chatbot.ask(chat_models, "Which line has the most anomalous trips?")
```

Run the API with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m uvicorn src.api.main:app`,
and the dashboard with `streamlit run src/dashboard/app.py`.
