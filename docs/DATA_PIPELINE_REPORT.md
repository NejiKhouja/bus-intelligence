# WiniCari Data Pipeline — From Raw MongoDB to the AI Modules

This report documents what the WiniCari AI layer's data pipeline does, why it exists in
its current form, and what changed between the original ad-hoc pipeline and the current
reference-database-backed one. It supersedes the informal working notes in
`docs/notes/project_summary.txt` and `docs/notes/reference_db_findings.txt` (kept in place
as raw source material, not deleted).

---

## 1. Executive summary

WiniCari's bus-management platform stores GPS telemetry, ticketing records, and route
definitions across several MongoDB databases (`winicari`, `Historique_pos`,
`Historique_Tickets`, `OpenData`). That system is never modified by this project — it is
read from, and a separate AI layer is built on top: delay prediction, GPS fallback during
signal loss, unsupervised anomaly detection, and a RAG chatbot for plain-language fleet
queries.

Until this redesign, every AI module re-derived its own notion of "which lines exist,
which stops belong to them, in what order" directly from raw MongoDB documents each time
it ran, with no persistent, auditable record of how any of it was resolved. This report
covers the work that replaced that ad-hoc approach with a **versioned SQLite reference
database** (`data/reference/winicari_reference.db`, 8 tables, built by
`src/build_reference_db.py`), plus the concrete, measured improvements that came out of
building it: more usable line geometry, a higher GPS-match rate, a real bug fix in the
already-deployed anomaly model, and per-company anomaly models that can now actually see
the dominant operator's own outliers.

Net effect, in one line: usable line geometry went from 143 to 217 lines (+52%), the
GPS-ping match rate went from 70.1% to 80.1%, and the anomaly model's training data went
from a dead, constant feature to a real, variable one — all traceable to specific,
documented causes rather than incidental noise.

---

## 2. Original MongoDB audit — 14 structural anomalies

These are structural or schema-design problems found in the raw source data, not routine
data-quality noise. Numbered for reference; re-verified against the live database rather
than pulled from memory.

1. **No canonical company table used as a foreign key.** `societe` is a free-text string
   copy-pasted into every collection (`ligne`, tickets, GPS pings, `STOP<societe>`,
   `station`, `sav`, `details`, ...) instead of a foreign key into one authoritative table.
   The same real company ends up spelled differently in different places (`S.R.T.K` vs
   `S.R.T.K0`, `Winicari` vs `winicari`) with nothing detecting the drift.
2. **The one collection that looks like it should be that authoritative table**
   (`winicari.societe`, 10 documents, has `nomComplet`/`Gouvernorat`/`active`) is itself
   incomplete — `SORETRAS` and `S.T.C.I` are entirely missing from it despite being real,
   active operators with real lines, tickets, and SAV records elsewhere in the same
   database.
3. **Station/stop identity has six incompatible ID schemes** for the same real-world
   place, none of which reliably cross-reference each other: `OpenData.*.code_station`,
   `winicari.station.stop_id`, `STOP<societe>.NAMENRnew`, `STOP<societe>.NAMENR`,
   `winicari.Names.Code`, and ticket `origine`/`Distination` codes (a per-line local
   sequence that resets to 01 on every route). Nothing in the schema documents which is
   authoritative or how they relate — each pairing had to be empirically tested.
4. **The same field name means different things per company.** `STOP<societe>.NAMENR`
   equals an OpenData `code_station` for S.R.T.BIZERTE/S.T.S/TCV (20/20 tested), but is
   just a meaningless sequential integer for SRT.ELGOUAFEL/EPE-TVE (0/20). Trusting the
   field without per-company verification would have produced silently wrong joins.
5. **Station names collide across physically distant places** under the exact same
   string, with no deduplication ever enforced — 210 confirmed conflicts more than 1km
   apart among 2,109 duplicate names across the OpenData collections (e.g. "STADE
   MUNICIPALE", "MEUBLATEX").
6. **Four parallel, overlapping OpenData station collections** (`Station`, `Station2`,
   `Station_new`, `Station_sts`) look like sequential migrations layered on top of each
   other rather than one maintained collection — the same station can appear 2–4 times
   with slightly different coordinates depending on which one recorded it.
7. **`ligne.stationnames` contains a confirmed data-entry error baked into a route
   definition**: S.R.T.K line 201's Tunis-side terminal is labeled "KASSERINE" — the
   departure city's name, not the actual stop name. Undetectable without an independent
   name source to cross-reference against.
8. **Line codes are not unique across companies** — the same numeric code means a
   different route depending on company, with no compound key enforced except by
   convention.
9. **No referential integrity anywhere** (MongoDB has no FK constraints). A
   `STOP<societe>.NAMENRnew` can point at a `winicari.station` id or `Names.Code` that
   doesn't exist; `ligne.societe` can contain a spelling that matches no canonical
   company. Every join built for this project had to be manually verified with real
   hit-rate tests, because nothing in the database guarantees it holds.
10. **Dead/test collections sitting in what looks like production data**: `stop` (7 empty
    docs, literally a Java class-name placeholder), `historiqueStation` (mostly junk
    entries), `demande_ligne` (literal test strings `"aaa"`/`"aaaa"` stored as real field
    data), `klm` (an incomplete, superseded predecessor to `STOP<societe>`). None are
    documented as deprecated — each had to be sampled and tested to discover it was dead.
11. **GPS ping schema drifted over ~4.5 years with zero versioning.** The `speed` field
    moved from nested `bus.vitesse` to top-level `speed` around 2025; `service.codeLigne`
    (the line link) only exists on pings from ~mid-2022 onward; ping density per bus
    jumped roughly 12x with no field marking when or why.
12. **GPS equipment status is not an explicit field anywhere** — whether a company
    currently has working trackers must be inferred by scanning raw ping history, and even
    a "recent window" check is misleading: S.R.T.BIZERTE, TUS, and EPE-TVE all had real,
    now-discontinued GPS windows that a naive 30-day check completely misses.
13. **`winicari.station`** (the coordinate table the most reliable stop-resolution tier
    depends on) only has entries for 5 of the 12 real companies.
14. **Duplicate near-identical `(societe, code)` line documents exist inside `ligne`
    itself** — 6 found (mostly S.R.T.SELIANA), empty stub documents with zero
    `stationnames`, with no uniqueness constraint preventing their creation.

---

## 3. Enrichment methodology

### 3.1 Company canonicalization
Twelve canonical companies were established by manually auditing every raw `societe`
string variant across all collections. Two merges were made: `Winicari`/`winicari` (user
confirmed same entity despite very different data volumes), and `S.R.T.K0` as an alias of
`S.R.T.K` (same device serial numbers, same lines — a ticketing-system artifact, not a
distinct company).

### 3.2 Station clustering (not name-matching)
Roughly 8,683 raw geocoded points, pulled from every source collection, were clustered by
**physical proximity** (DBSCAN, haversine distance, `eps=150m`) into 2,991 canonical
stops — solving the name-collision problem (anomaly #5 above) structurally, rather than by
a name-matching heuristic that would inherit the same collisions. Each cluster's
`primary_name` is chosen by majority vote **across distinct source collections**, not raw
row count — otherwise a source like `ligne` (which can contribute 30+ near-duplicate rows
for one lazy name) would wrongly outvote OpenData's fewer but more accurate names.
Confidence tiers record how reliable each stop's identity is: `verifie` (≥2 independent
sources agree), `inferee` (1 source only), `non_nomme` (no real name known, placeholder
only), plus three `triangule_*` tiers for stops recovered via ticket/GPS triangulation
(see 3.4).

### 3.3 Line-stop geometry: a 6-tier resolver
`line_stops` (which stops belong to which line, in what order) is resolved per line by
trying six methods in order of reliability, stopping at the first that resolves ≥4 stops:

| Tier | Method | Lines resolved |
|---|---|---|
| 1 | `STOP<societe>` + `winicari.station` (direct coordinate join) | 59 |
| 2 | `STOP<societe>.NAMENR` → `OpenData.code_station` (direct code join) | 20 |
| 3 | `STOP<societe>.NAMENRnew` → `winicari.Names.Code` → name → stops | 81 |
| 4 | `ligne.stations` → `winicari.station.stop_id` | 3 |
| 5 | `ligne.array_lat_opendata` raw anchors → nearest canonical stop | 38 |
| 6 | Ticket boarding-order, disambiguated (reused, not reimplemented, from `src/data/stations.py`) | 16 |
| — | Unresolved | 185 |

**217/402 lines (54%) resolved**, and critically, **160 of those 217 (74%) came from
direct structural ID joins (tiers 1–3)** discovered this session — not from probabilistic
name-matching. Only 57 came from the weaker anchor/ticket-fallback tiers that an earlier,
lighter notebook-based pass (`08_station_eda.ipynb`/`09_station_matching.ipynb`, now
archived in `notebooks/archive/`) already used. That earlier pass recovered 143→187 lines
(+44) using name-matching plus ticket triangulation alone — a fast, useful first pass that
de-risked the concept before the heavier, more rigorous schema-building effort in this
report.

### 3.4 What triangulation attempts worked, and what didn't
Ticket-time × GPS-ping triangulation (matching a ticket's boarding timestamp to the bus's
GPS position at that moment) was used both to recover otherwise-unresolvable stop names
and, separately, to try to close the last 185 unresolved `line_stops` lines.

- **It worked, with verification, in specific cases**: `HOP.SAHLOUL` was recovered and its
  coordinate manually confirmed correct by the user after the algorithm's initial pick was
  proven wrong. `SBIKHA` (S.R.T.SELIANA) and `BEN GARDANE` (S.R.T.K) were kept as
  `triangule_non_verifie` — not confirmed, but their ticket sequences show plausible linear
  intercity progression.
- **It failed in a way worth documenting — the S.T.S "gravity well" bug**: an early,
  naive version of the algorithm systematically pulled multiple genuinely different S.T.S
  stops toward the same physical point (a depot location, per `winicari.centre`), because
  it favored the largest ping cluster near a rough time window without checking whether
  that cluster made physical sense as a distinct stop. Twelve S.T.S entries are kept as
  `triangule_nom_incertain` — the GPS coordinate is a real place, but the name is probably
  wrong — specifically so this caveat isn't silently hidden.
- **A batch attempt to close the remaining 185 unresolved lines was tried and rejected**:
  0 real recoveries out of 34 attempted, because even buses with excellent (5-second
  median) ping density have genuine, unpredictable multi-hour GPS "dark periods" that make
  ticket-to-ping matching structurally unreliable at batch scale. The 185 unresolved
  lines are a genuine data ceiling (52 lines have zero structured stop data anywhere ever;
  132 have resolved names but no coordinate in any source), dominated by S.R.T.SELIANA
  (village-level stops in Siliana governorate with no OpenData coverage) and Winicari —
  not a resolver bug. Decision: stop investing further in this lever rather than risk
  another undetected systematic bias like the S.T.S case.

---

## 4. Reference database schema

`data/reference/winicari_reference.db` (SQLite, built/refreshed by
`src/build_reference_db.py`, library code in `src/data/reference_db.py`):

| Table | Rows (last build) | Purpose |
|---|---|---|
| `companies` | 12 | Canonical operator identity, GPS activity window, admin enrichment |
| `stops` | 2,991 | Canonical physical stops, clustered by proximity, confidence-tiered |
| `lines` | 402 | Union of every line code seen in `ligne`/ticket/GPS sources |
| `line_stops` | 2,446 | Ordered stop sequence per line (6-tier resolved) |
| `trips` | 26,132 | Reconstructed real GPS trips on the new geometry |
| `trip_stops` | 198,430 | Per-stop arrival/departure/dwell/match detail per trip |
| `tickets_daily` | 7,069 | Daily ticket-sale aggregates per (company, line, bus, day) |
| `anomaly_scores` | 0 (reserved) | Intended for persisting per-trip anomaly scores; currently unused — scores live in `models/anomaly/trips_scored.parquet` instead |

All tables use real foreign keys (`company_id`, `line_id`, `stop_id`, `trip_id`) and carry
`confidence`/`source`/`notes` columns wherever a value's provenance matters, so any future
consumer can ask "how was this resolved" without re-deriving the answer from raw MongoDB.

`src/data/reference_db.py::export_foundation_parquet()` reconstructs
`data/processed/foundation_arrivals_full.parquet` — with exactly the same columns the
original ad-hoc pipeline produced — from these tables. This is the bridge that lets every
existing AI module (`src/data/delay.py`, `fallback.py`, `anomaly.py`, `src/train_pipeline.py`)
benefit from the enriched data **with zero code changes to those modules**; only the data
source underneath changed.

---

## 5. Model training changes

- **`match_rate` dead-feature bug, fixed.** `src/data/anomaly.py::trip_features()`
  computed `match_rate` on a frame already filtered to `matched == True` rows, so its mean
  was mathematically always `1.0` — a zero-variance feature that had been silently dead in
  the already-deployed anomaly model. Fixed by computing it on the full per-trip window
  before filtering. Verified: real variance now (min 0.136, mean 0.807, max 1.0).
- **Loop-route ("full trip") mislabeling, fixed via nullable classification.**
  `foundation.py`'s trip segmentation assumed every route is linear (position goes up then
  down for aller/retour). TCV line 3 is a real urban loop (the same physical stop appears
  twice in its stop sequence), which the linear assumption misclassified. Rather than guess
  a new threshold, `detect_loop_route()` was added and loop routes now get `is_full = NULL`
  ("unreliable to classify") instead of a wrong `True`/`False` — a decision made explicitly
  with the user (see `foundation.py::segment_trips`/`reconstruct_bus_day`). This matters at
  scale: TCV/3 alone is ~47% of all trips in the full-history rebuild, so a wrong label here
  would have meaningfully biased any model trained on "is this trip complete."
- **Per-company LSTM autoencoder split for anomaly detection.** A single global LSTM
  autoencoder had been trained on all companies pooled together; since TCV is ~75% of all
  trips, the model learned "normal = what TCV does" and could never flag TCV's own
  outliers (0 dual-flagged TCV trips). Companies with ≥200 trips (`S.R.T.BIZERTE`,
  `S.R.T.K`, `S.T.S`, `SRT.ELGOUAFEL`, `TCV`) now get their own dedicated model; the two
  smaller companies (`S.R.T.SELIANA`: 106 trips, `TUS`: 37 trips) correctly fall back to a
  global model rather than getting an unstable dedicated one. Isolation Forest was already
  per-company before this change and needed no split.
- **Per-company HistGBM delay models, added only where measured to help.** The pooled
  (all-companies) delay model showed the same "dominant company defines normal" pattern as
  the anomaly module: TCV (largest, 56.7k training rows) MAE 1.62 min vs. S.R.T.SELIANA MAE
  10.05 min. Adding `societe` as a plain feature made no measurable difference (the model
  already captured most of that signal via `line`). A dedicated per-company HistGBM model
  was tested at several minimum-training-row thresholds — below ~3,000 rows a dedicated
  model actively *regresses* (SRT.ELGOUAFEL, 1,228 rows: 5.71→6.59 min), while above it,
  dedicated models clearly help (TCV 1.62→1.46, S.R.T.K 4.17→4.10, overall 3.12→3.02) with
  no regression for any company that has real test data. `MIN_TRIPS_COMPANY = 3000` in
  `src/models/delay.py` — companies below that threshold automatically fall back to the
  global model rather than getting an unstable dedicated one.
- **Service-duration anomaly features** (`elapsed_vs_bus_z`, `elapsed_vs_line_z` in
  `src/data/anomaly.py::trip_features()`): bidirectional z-scores comparing a trip's total
  duration to that specific bus's own historical average and to that line's average
  (distinct from `total_elapsed`, which only compares against the whole company's pooled
  distribution). Added directly to the existing per-company Isolation Forest / LSTM
  autoencoder rather than as a separate model.
- **New, separate ticket-sale anomaly signal** (`src/data/ticket_anomaly.py` /
  `src/models/ticket_anomaly.py`), deliberately not merged with GPS trip anomaly — it
  scores at the (company, line, bus, **day**) grain from `tickets_daily`, while GPS anomaly
  scores per **trip**; a day's ticket total can't be honestly split across multiple trips.
  **Not yet wired into `src/train_pipeline.py`** — it needs the SQLite reference-DB
  connection rather than the foundation parquet; see §7.

---

## 6. Before / after metrics

All figures below are the actual, measured full-history numbers (same date range,
2025-01 to 2026-06 plus each company's own extended historical GPS window), not
projections. See `notebooks/08_reference_db_eda.ipynb` for the live-computed, re-runnable
version of this comparison.

| Metric | Before | After | Delta |
|---|---|---|---|
| Usable line geometry | 143 lines | 217 lines | +74 (+52%) |
| Stop-visit rows | 177,334 | 198,430 | +21,096 |
| Reconstructed trips | 21,669 | 26,132 | +4,463 |
| Distinct lines with trips | 35 | 60 | +25 |
| Distinct companies with trips | 3 | 7 | +4 |
| GPS-ping match rate | 70.1% | 80.1% | +10.0 pts |
| Companies with a dedicated anomaly LSTM | 0 (1 global model for all) | 5 (+ global fallback for 2 small companies) | — |
| Anomaly `match_rate` feature variance | 0 (constant 1.0, dead) | real (0.136–1.0) | fixed |

Companies previously invisible to trip reconstruction at all (`S.R.T.BIZERTE`, `TUS`,
`EPE-TVE`) now contribute real historical trips, because their genuine (but currently
discontinued) GPS windows were discovered via full-history scanning instead of a
recent-30-day check (anomaly #12).

---

## 7. Known limitations

- **185/402 lines still have no resolved stop geometry** (§3.3/§3.4) — a genuine data
  ceiling in the source MongoDB (missing OpenData coverage for smaller villages), not a
  resolver bug. Not pursued further this session; would require new geocoding data to
  close, not new code.
- **EPE-TVE lines 976 and 992 have impossible route lengths** (11,494 km and 8,167 km) in
  both the old and new geometry — a source-data bug (almost certainly one bad anchor
  coordinate), not introduced by this redesign. Flagged, not yet isolated/fixed.
- **`line_stops` does not persist which resolver tier produced each line's geometry.** The
  6-tier breakdown in §3.3 and in `08_reference_db_eda.ipynb` reflects the last full
  `populate_line_stops` build's printed output, not a queryable column. Adding a
  `tier` column to `line_stops` would make this auditable without needing to re-run the
  build or consult logs — a reasonable follow-up, not done here to avoid an unnecessary
  schema/behavior change mid-redesign.
- **`ticket_anomaly` is not wired into `src/train_pipeline.py`** — it depends on the SQLite
  reference DB connection (via `src/build_reference_db.py`) rather than the foundation
  parquet that the other three modules share, so it currently has to be trained
  separately (see `src/models/ticket_anomaly.py::train(conn)`).
- **No automated test suite or CI.** This is a real gap, not a design choice — explicitly
  out of scope for this redesign per current project scope (no autonomous/scheduled
  operation is wanted either; every step described in §8 is admin-triggered).
- **`populate_trips` has no shard/resume capability** — it always does a full
  delete-and-rebuild of `trips`/`trip_stops` (or an explicit `additive=True` merge), unlike
  `build_foundation.py`'s monthly-shard pattern. Fine for the current scale (~40 minutes
  full rebuild); would need resumability added before a much longer multi-year window
  became routine.

---

## 8. How to operate this pipeline

Two independent things can be rebuilt, on purpose kept separate since they change at very
different frequencies and costs:

### Rebuilding the reference database
```
conda activate bus-intelligence
python -m src.build_reference_db                 # reference tables only, <1 minute
python -m src.build_reference_db --with-trips     # + full GPS trip reconstruction, ~40 minutes
python -m src.build_reference_db --with-trips --since 20250101 --until 20250630
python -m src.build_reference_db --with-trips --company S.R.T.BIZERTE --company TUS
```
Run this whenever underlying MongoDB data changes (new stations, corrected line
definitions, a newly connected company). `--with-trips` is only needed when the GPS
history itself needs reprocessing — an admin who just fixed a station name doesn't need
to wait 40 minutes for a full trip rebuild. This is a manually-triggered command, run by
an admin when needed — **there is no autonomous or scheduled execution**, by explicit
design choice.

### Training the AI models
Two equivalent paths, by design, so training doesn't require being an ML expert:
```
conda activate bus-intelligence
python -m src.train_pipeline        # all 4 modules end-to-end, from the current parquet/reference DB
```
or open any of `notebooks/03_delay.ipynb` .. `notebooks/06_rag_chatbot.ipynb` and run it
end-to-end — each notebook's final cell now calls the exact same `src.models.X.train()`
function the pipeline calls, saving to the exact same `models/X/` location, so either path
produces an equivalent, usable production artifact.

### Exploring / auditing the data
- `notebooks/01_eda.ipynb` — the **original raw MongoDB** data, unchanged by this redesign.
- `notebooks/08_reference_db_eda.ipynb` — the **current reference database**, including the
  before/after comparison in §6, fully re-runnable.
- `notebooks/archive/` — the earlier station-enrichment investigation notebooks (08–10
  before this redesign), kept for their investigative value but no longer in the main
  numbered sequence.
