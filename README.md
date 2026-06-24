# WiniCari — AI Layer for Tunisian Bus Operations

An end-to-end machine learning pipeline built on top of the WiniCari bus-management
platform (MongoDB). Four AI modules cover delay prediction, GPS gap recovery,
anomaly detection, and a natural-language chatbot.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Sources](#2-data-sources)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Pipeline — Step by Step](#5-pipeline--step-by-step)
6. [Notebooks](#6-notebooks)
7. [Model Artifacts](#7-model-artifacts)
8. [AI Modules](#8-ai-modules)
9. [Evaluation Results](#9-evaluation-results)
10. [Conclusions & Limitations](#10-conclusions--limitations)

---

## 1. Project Overview

WiniCari is a Tunisian bus-software company. Their platform stores ticketing data,
route definitions, and GPS telemetry in MongoDB. This project adds an AI layer with
four modules that can run on top of that data without modifying the existing system.

| Module | Task | Models |
|--------|------|--------|
| 1 — Delay Prediction | Predict bus delay at every remaining stop in real time | HistGBM, LSTM, Prophet |
| 2 — GPS Fallback | Estimate bus position during a GPS signal gap | Kalman filter + LSTM correction |
| 3 — Anomaly Detection | Flag suspicious or broken trips automatically | Isolation Forest + LSTM Autoencoder |
| 4 — RAG Chatbot | Answer natural-language questions about the fleet | ChromaDB + sentence-transformers + Llama 3 |

---

## 2. Data Sources

Two MongoDB databases are used. Neither is modified by this project — all writes go
to local files only.

### `winicari` — Service / Ticketing Database
Contains route definitions, stop lists, company information, and ticketing records.
Key collections used:
- `services` — route metadata (line code, direction, company)
- `arrets` — geocoded stop coordinates per line
- `tickets` / `billetDeCommandes` — passenger records used to reconstruct trips

### `Historique_pos` — GPS Telemetry Database
One collection per calendar day, named `d{YYYYMMDD}` (e.g. `d20260615`).
Each document is one GPS ping from a bus transponder with fields including:
- `bus.code` — bus identifier
- `service.codeLigne` — line code
- `position.lat`, `position.lng` — WGS-84 coordinates
- `position.vitesse` — speed (km/h)
- `createdAt` — timestamp

**Important data caveats discovered during EDA:**
- GPS ping interval changed mid-dataset (from ~60 s to ~30 s). Average pings per day
  also increased as the fleet adopted newer transponders that report speed directly.
- No actual arrival times exist in the database. "Arrival" is approximated from the
  last GPS ping near a stop before the next stop sequence begins.
- Ticket reservation history covers only ~1 week of real demand data; most entries
  show empty buses, likely because the companies serve rural/low-demand routes.
- Company name strings are inconsistent across collections (e.g. `"S.R.T.K"` vs
  `"SRTK"`). All joins use normalized forms.
- Lines with 0 geocoded stops exist — these run terminal-to-terminal with no
  intermediate stops and are excluded from per-stop models.

---

## 3. Project Structure

```
winicari/
├── data/
│   ├── processed/
│   │   ├── foundation_arrivals_full.parquet   # 168k rows, main dataset
│   │   ├── line_distances.parquet             # route lengths per line
│   │   ├── shards/                            # monthly foundation shards
│   │   └── chroma_kb/                         # ChromaDB knowledge base
│   └── raw/                                   # raw exports (not tracked)
│
├── models/
│   ├── delay/
│   │   ├── hgbm.joblib                        # HistGradientBoosting model
│   │   ├── lstm_delay.pt                      # LSTM weights
│   │   ├── lstm_scaler.npz                    # feature mean/std (train-only)
│   │   ├── lstm_config.json                   # architecture params
│   │   ├── baseline.parquet                   # median elapsed time per stop
│   │   └── prophet/                           # 33 Prophet .pkl files
│   ├── fallback/
│   │   ├── lstm_corr.pt                       # Kalman correction LSTM weights
│   │   ├── lstm_corr_stats.npz               # feature normalisation stats
│   │   └── lstm_corr_config.json
│   └── anomaly/
│       ├── isolation_forest.joblib
│       ├── if_scaler.npz
│       ├── lstm_ae.pt                         # LSTM Autoencoder weights
│       ├── lstm_ae_config.json
│       ├── lstm_ae_threshold.npy              # 95th-percentile threshold
│       └── trips_scored.parquet              # all trips with anomaly flags
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_delay.ipynb
│   ├── 04_gps_fallback.ipynb
│   ├── 05_anomaly.ipynb
│   ├── 06_rag_chatbot.ipynb
│   └── 07_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── db.py               # MongoDB connection helpers
│   │   ├── foundation.py       # builds the foundation dataset
│   │   ├── build_foundation.py # CLI to run foundation build
│   │   ├── delay.py            # delay feature engineering + model training
│   │   ├── fallback.py         # Kalman filter + GPS fallback logic
│   │   ├── anomaly.py          # anomaly feature engineering + models
│   │   └── rag.py              # ChromaDB build/load/query
│   ├── models/
│   │   ├── delay.py            # train / load / predict_eta / forecast
│   │   ├── gps_fallback.py     # train / load / predict_position
│   │   ├── anomaly.py          # train / load / score
│   │   └── chatbot.py          # build / load / ask
│   ├── api/main.py             # FastAPI serving layer (stub)
│   ├── dashboard/app.py        # Streamlit dashboard (stub)
│   └── train_pipeline.py       # single entry point to train all 4 modules
│
├── docs/GLOSSARY.md
├── requirements.txt
└── docker-compose.yml
```

---

## 4. Environment Setup

The project requires a dedicated conda environment (`bus-intelligence`) with PyTorch,
Prophet, filterpy, sentence-transformers, and ChromaDB. A second lighter environment
(`py310`) is used for the foundation build step only.

```bash
# Create the main environment (first time only)
conda create -n bus-intelligence python=3.11
conda activate bus-intelligence
pip install -r requirements.txt

# MongoDB connection
# Copy .env.example to .env and fill in MONGO_URI and GROQ_API_KEY
cp .env.example .env
```

**.env** must contain:
```
MONGO_URI=mongodb://localhost:27017
GROQ_API_KEY=gsk_...
```

---

## 5. Pipeline — Step by Step

### Step 1 — Build the foundation dataset

The foundation is a flat parquet file joining ticketing events with GPS positions and
stop geometries. It is the single input consumed by all four AI modules.

```bash
conda activate py310
python -m src.data.build_foundation --since 202501
```

Output: `data/processed/foundation_arrivals_full.parquet`

**What it does:**
- Iterates over every usable (line, company) pair found in the `winicari` database
- For each pair, loads the stop list and projects GPS pings onto the route as a
  1-D distance along the route (`s_m`)
- Applies a Kalman filter to smooth noisy GPS positions
- Matches each GPS ping to the nearest stop within a configurable tolerance
- Reconstructs individual trips by detecting terminal dwells
- Outputs one row per (trip, stop) arrival event

**Key schema of the foundation:**

| Column | Description |
|--------|-------------|
| `day` | Date string YYYYMMDD |
| `societe` | Bus company name |
| `line` | Line code (e.g. "209") |
| `dir` | Direction: ALLER / RETOUR |
| `bus` | Bus transponder ID |
| `trip_id` | Within-day trip counter |
| `seq` | Stop sequence number |
| `arrival` | Reconstructed arrival timestamp |
| `trip_start` | First ping timestamp of the trip |
| `matched` | Whether the stop was within GPS tolerance |
| `dist_m` | Distance from GPS ping to stop centroid (m) |
| `dwell_s` | Seconds the bus spent at this stop |
| `full` | True if trip started and ended at the correct terminals |

### Step 2 — Train all four AI modules

```bash
conda activate bus-intelligence
python -m src.train_pipeline
```

This runs all four modules in sequence and saves every artifact to `models/`.
Total runtime: ~10 minutes on CPU.

### Step 3 — Explore and evaluate in notebooks

Open JupyterLab and run the notebooks in order (see Section 6).

---

## 6. Notebooks

Each notebook is self-contained and can be run independently after the foundation
dataset and trained models exist.

### `01_eda.ipynb` — Exploratory Data Analysis
- Fleet overview: companies, lines, buses, service days
- GPS quality metrics: ping frequency, gap distribution, signal loss patterns
- Stop geocoding coverage: how many lines have complete stop lists
- Discovery of GPS schema drift (transponder upgrade mid-dataset)
- Peak hour analysis (5:00, 6:00, 10:00 are highest activity)
- Identification of lines with 0 geocoded stops (terminal-to-terminal only)

### `02_preprocessing.ipynb` — Foundation Build & Preprocessing
- Demonstrates the projection of GPS pings onto route geometry
- Kalman filter smoothing visualization
- Trip segmentation logic: detecting terminal layovers to split trips
- Baseline schedule computation: median elapsed time per (line, dir, stop)
- Data quality summary: match rates, partial vs full trips, gap statistics

### `03_delay.ipynb` — Delay Prediction
- Feature engineering: `delay_min`, `elapsed_min`, `seq_frac`, `dep_hour`, `is_weekend`
- Rolling next-stop model formulation: predict delay at stop k+1 given state at stop k
- HistGBM training and feature importance
- LSTM training with temporal validation split and early stopping
- Prophet time-series model for daily mean delay forecasting per line/direction
- ETA simulation: chaining next-stop predictions to forecast the full remaining journey

### `04_gps_fallback.ipynb` — GPS Gap Recovery
- Kalman filter design: state = [s (route distance), v (speed)], observation = GPS s
- Filter tuning: process noise Q, observation noise R calibration
- LSTM correction model: learns residual (s_true - ks) from recent Kalman history
- Multi-bus-day training strategy for better generalisation
- Synthetic gap evaluation: masking real pings and measuring recovery error

### `05_anomaly.ipynb` — Anomaly Detection
- Trip-level feature extraction: stop count, match rate, max/mean dwell time,
  total elapsed time, max stop distance
- Isolation Forest training with `contamination=0.05`
- LSTM Autoencoder: sequence-to-sequence reconstruction of per-stop stop profiles
- Threshold calibration: 95th percentile of training reconstruction error
- Score distribution analysis and top anomalous trip inspection

### `06_rag_chatbot.ipynb` — RAG Chatbot
- Document generation: one document per (company, line) with statistics, one per
  anomalous trip
- ChromaDB indexing with `all-MiniLM-L6-v2` sentence-transformer embeddings
- Retrieval-augmented generation: top-k document retrieval + Llama 3.1-8b-instant
  via Groq API
- Interactive Q&A demonstration

### `07_evaluation.ipynb` — Full Evaluation
- Rebuilds the exact 80/20 day-based train/test split for Modules 1 and 3
- Module 1: MAE/RMSE/R² table, predicted-vs-actual scatter, residuals by hour/line/
  day-type, live ETA demo, Prophet forecast chart
- Module 2: 400-sample synthetic gap experiment comparing interpolation, Kalman, and
  Kalman+LSTM; error distribution plots; Kalman uncertainty band visualization
- Module 3: IF and LSTM AE score distributions with threshold lines, model agreement
  scatter and pie chart, feature separation plots, top-10 anomalous trips table
- Module 4: 6 live RAG queries with retrieved context and generated answers
- Final summary scorecard across all modules

---

## 7. Model Artifacts

All artifacts are saved to `models/` after running `train_pipeline.py`.

### Delay (`models/delay/`)

| File | Contents |
|------|----------|
| `hgbm.joblib` | Trained `HistGradientBoostingRegressor`; features: dep_hour, dow, is_weekend, seq, seq_frac, delay_min, elapsed_min, line (cat), dir (cat) |
| `lstm_delay.pt` | 2-layer LSTM (hidden=64) PyTorch state dict |
| `lstm_scaler.npz` | Per-feature mean and std fitted on **training data only** |
| `lstm_config.json` | Architecture params (hidden, n_layers, window) |
| `baseline.parquet` | Median/P10/P90 elapsed time per (societe, line, dir, seq) — the data-derived schedule |
| `prophet/*.pkl` | One Prophet model per (societe, line, dir) key — 33 models total |

**Design decisions:**
- Train/test split is **chronological by day** (80/20), not random. Random splitting
  would leak future delay patterns into training and produce over-optimistic metrics.
- The LSTM validation set is the temporal last 10% of training sequences. Early
  stopping (patience=5) restores the best checkpoint.
- Feature normalisation (`lstm_scaler.npz`) is fitted on training data only and
  applied identically to validation, test, and inference. This prevents data leakage
  through the scaler.
- HistGBM handles categorical features (`line`, `dir`) natively and does not require
  normalisation.

### GPS Fallback (`models/fallback/`)

| File | Contents |
|------|----------|
| `lstm_corr.pt` | Small LSTM (hidden=32) that predicts the residual correction (s_true - ks) |
| `lstm_corr_stats.npz` | Feature mean/std for [ks, kv, kp, speed] |
| `lstm_corr_config.json` | Window size, n_feats, hidden, training ping count |

**Design decisions:**
- The Kalman filter has no learnable parameters and runs online at inference time.
  It only requires the route length.
- The LSTM target is the **residual** `s_true - ks`, not absolute `s`. This is
  critical: raw route distance spans 0–192,000 m; predicting it from normalised
  features produces a 10^11 m² loss because the model defaults to predicting the
  route midpoint. The Kalman estimate is already close to ground truth; the LSTM
  only needs to learn small corrections (~±40 m std).
- Training pools GPS pings from **multiple bus-days and multiple buses** on the
  same line to prevent overfitting to one trip's geometry and traffic pattern.

### Anomaly (`models/anomaly/`)

| File | Contents |
|------|----------|
| `isolation_forest.joblib` | Trained `IsolationForest(contamination=0.05)` |
| `if_scaler.npz` | Feature mean/std for the 6 trip-level features |
| `trips_scored.parquet` | All 17,565 trips with `if_score`, `anomaly`, `lstm_score`, `lstm_anomaly`, `dual_anomaly` |
| `lstm_ae.pt` | LSTM Autoencoder state dict |
| `lstm_ae_config.json` | seq_pad, n_feats, hidden |
| `lstm_ae_threshold.npy` | 95th-percentile reconstruction error from training set |

**Trip-level features used by Isolation Forest:**

| Feature | Description |
|---------|-------------|
| `n_stops` | Number of matched stops in the trip |
| `match_rate` | Fraction of stops within GPS tolerance |
| `max_dwell_s` | Longest single-stop dwell time (seconds) |
| `mean_dwell_s` | Average dwell time per stop |
| `total_elapsed` | Total trip duration (minutes) |
| `dist_m_max` | Largest GPS-to-stop distance observed |

### RAG Chatbot (`data/processed/chroma_kb/`)

ChromaDB persistent store containing 1,276 documents:
- One document per (company, line) with route statistics, coverage quality,
  match rates, and anomaly counts
- One document per flagged anomalous trip with full feature values
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- LLM: `llama-3.1-8b-instant` via Groq API

---

## 8. AI Modules

### Loading and using models

```python
from src.models import delay, gps_fallback, anomaly, chatbot

# Load all models from disk
delay_models    = delay.load()          # loads from models/delay/
fallback_models = gps_fallback.load()   # loads from models/fallback/
anomaly_models  = anomaly.load()        # loads from models/anomaly/
chat_models     = chatbot.load()        # loads from data/processed/chroma_kb/
```

### Module 1 — Delay / ETA

```python
# Rolling stop-by-stop ETA (HistGBM, default)
eta = delay.predict_eta(
    delay_models,
    societe="S.R.T.K", line="209", direction="ALLER",
    dep_time="2026-06-15 06:00",
    current_seq=5,           # bus is at stop 5
    current_delay_min=8.0,   # currently 8 minutes late
    model_type="hgbm",       # or "lstm"
)
# Returns DataFrame: seq, expected_min, pred_delay_min, eta

# Prophet 30-day forecast for a line
fc = delay.forecast(delay_models, societe="S.R.T.K", line="209", direction="ALLER")
```

### Module 2 — GPS Fallback

```python
import pandas as pd
from src.models import gps_fallback as fb_mod

# g_filtered must be the output of run_kalman()
g_filtered = fb_mod.run_kalman(g, route_len)

# Estimate position at any query time (during or outside a gap)
pos = fb_mod.predict_position(fallback_models, g_filtered, pd.Timestamp("2026-06-15 08:35"), stops)
# Returns: {"lat": ..., "lon": ..., "s_m": ..., "uncertainty_m": ..., "method": "kalman+lstm"}
```

### Module 3 — Anomaly Detection

```python
import pandas as pd

fa = pd.read_parquet("data/processed/foundation_arrivals_full.parquet")
scored_trips = anomaly.score(anomaly_models, fa)
# Returns trips DataFrame with: anomaly, if_score, lstm_score, lstm_anomaly, dual_anomaly

# Or use the pre-scored trips from training
trips = anomaly_models["trips"]
dual_anomalies = trips[trips["dual_anomaly"]]
```

### Module 4 — RAG Chatbot

```python
result = chatbot.ask(chat_models, "Which line has the most anomalous trips?")
print(result["answer"])
# result also contains: context (list of retrieved docs), tokens_used
```

---

## 9. Evaluation Results

Evaluated on **held-out test data** (last 20% of days chronologically).
Full diagnostics in `notebooks/07_evaluation.ipynb`.

### Module 1 — Delay Prediction

| Model | MAE (min) | RMSE (min) | R² |
|-------|-----------|------------|-----|
| Naive (always 0) | 13.76 | 23.07 | -0.003 |
| Persistence (next ≈ current) | 3.06 | 7.66 | 0.889 |
| **HistGBM** | **2.75** | **6.82** | **0.912** |
| LSTM | 3.03 | 7.12 | 0.904 |

Test set: 12,009 rows across 104 held-out days.

**Interpretation:** HistGBM is the clear winner — 10% better MAE than the naive
persistence baseline, R²=0.912. The LSTM marginally beats persistence (3.03 vs 3.06)
but does not yet add meaningful value over HistGBM at this data volume.

### Module 2 — GPS Fallback (synthetic 3-min gaps, n=400)

| Method | Median error | P90 error |
|--------|-------------|-----------|
| Linear interpolation | 371 m | 828 m |
| Kalman filter | 159 m | 692 m |
| Kalman + LSTM correction | 159 m | 692 m |

**Interpretation:** The Kalman filter cuts interpolation error by more than half.
The LSTM correction currently adds no improvement — both have identical median and
near-identical P90. The model learned near-zero corrections because the Kalman is
already accurate on the windowed history it was trained on (non-gap segments).
The Kalman alone is the production-ready component.

### Module 3 — Anomaly Detection

| Model | Flagged trips | Rate |
|-------|--------------|------|
| Isolation Forest | 879 / 17,565 | 5.0% |
| LSTM Autoencoder | 793 / 17,565 | 4.5% |
| Dual-flagged (both) | 54 | 0.3% |

**Interpretation:** The 5% IF rate is by design (`contamination=0.05`). Inspection
of the top-10 most anomalous trips confirms IF is detecting real events: dwell times
of 1.7–3.5 hours at a single stop, likely breakdowns or driver incidents. The low
overlap between IF and LSTM (54 dual-flagged) means the two models capture different
signal — IF is driven by extreme dwell times, while LSTM AE is driven by unusual
stop-sequence patterns.

### Module 4 — RAG Chatbot

- 1,276 documents indexed, covering all (company, line) pairs and anomalous trips
- 6 test queries answered correctly with factual fleet statistics
- Average token consumption: 420 tokens/query

---

## 10. Conclusions & Limitations

### What works well

- **HistGBM delay model** is production-ready. R²=0.912 on unseen days, trains in
  under 2 minutes, and requires no GPU. It generalises across 33 distinct line/
  direction combinations without per-line training.
- **Kalman filter** is the backbone of GPS fallback. Cuts positioning error by >50%
  vs linear interpolation, with no learning required.
- **Isolation Forest** reliably surfaces genuine operational anomalies (extreme
  dwell times, broken trips) with a simple 6-feature representation.
- **RAG chatbot** provides a usable natural-language interface to fleet statistics
  without any fine-tuning.

### What needs more work

| Component | Issue | Suggested fix |
|-----------|-------|---------------|
| LSTM (delay) | Barely beats persistence baseline | Train 100+ epochs with a GPU; or gather 12+ months of data |
| LSTM (GPS correction) | Zero improvement over Kalman | Retrain on windows that include actual gap segments, not just normal driving |
| IF / LSTM AE agreement | Only 6% overlap between model flags | Collect operator labels to determine which model is right; use ensemble with voting |
| RAG answer quality | Not formally evaluated | Manual spot-check of 50+ queries; consider retrieval precision@k metric |
| No live retraining | Models go stale as new data arrives | Schedule monthly `python -m src.train_pipeline` via cron or Airflow |
| No serving layer | No API or dashboard deployed | `src/api/main.py` (FastAPI stub) and `src/dashboard/app.py` (Streamlit stub) exist but are not yet implemented |

### Key data insights discovered during EDA

- **Peak hours** are 05:00, 06:00, and 10:00 — matching early-morning commuter
  patterns for rural Tunisian routes.
- **GPS transponder upgrade** partway through the dataset changed the ping interval
  and added direct speed reporting. Any model using raw ping count as a feature
  must account for this structural break.
- **Most buses run near-empty** in the ticket reservation data. This is expected:
  the dataset covers rural routes where walk-up ridership dominates over advance
  booking, and the booking history available was only ~1 week.
- **Several lines have zero geocoded stops.** These run directly terminal-to-terminal
  and cannot be evaluated at a per-stop granularity. They are excluded from Modules
  1 and 2 but are included in the anomaly and RAG modules.
- **If full reservation history were available**, stop locations could potentially be
  derived from boarding/alighting clustering without needing an official stop list.
  This remains an open research question.

---

## Reproducibility

```bash
# Full pipeline from scratch (requires MongoDB access)
conda activate py310
python -m src.data.build_foundation --since 202501

conda activate bus-intelligence
python -m src.train_pipeline

# Evaluation
jupyter lab notebooks/07_evaluation.ipynb
```

All random seeds are fixed (`random_state=42`, `np.random.default_rng(42)`,
`torch.manual_seed` in training loops). Results are deterministic given the same
foundation dataset.
