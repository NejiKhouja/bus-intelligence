"""Batch-build the GPS arrival foundation dataset over a range of days.

What this does ON TOP OF foundation.py
--------------------------------------
`foundation.py` knows how to reconstruct ONE bus-day (one `(day, line, societe, bus)`).
This file is the **orchestrator** that runs that logic across the whole database and turns
it into a single dataset on disk. It does four things foundation.py does not:

1. ENUMERATE work — for every day in range it asks `candidates_for_day` which
   `(line, societe, bus)` actually ran on a usable-geometry line that day.
2. LOOP + ISOLATE — calls `reconstruct_bus_day` for each candidate inside a try/except, so
   one bad bus-day logs a warning and is skipped instead of killing the whole run.
3. SHARD + RESUME — writes one parquet per month to data/processed/shards/ and SKIPS months
   whose shard already exists, so a 40-min run can be stopped and restarted freely
   (that is why a re-run prints e.g. "[202606] shard exists, skip").
4. COMBINE — concatenates all monthly shards into the one file everything downstream reads:
   data/processed/foundation_arrivals_full.parquet.

Reading the progress lines
--------------------------
    [202502] days=28 cand=219 busdays_with_trips=215 rows=17240 match=83% (89s)
      days                = day-collections processed that month
      cand                = candidate (line,societe,bus) bus-days found
      busdays_with_trips  = how many yielded >=1 trip (the rest had no clean trip)
      rows                = stop-arrival rows written (one per covered stop)
      match               = % of covered stops with a GPS arrival within arrival_thresh_m

Usage:
    python -m src.data.build_foundation                # full usable range (>= 2022-06)
    python -m src.data.build_foundation --since 202501 # only months >= 202501
    python -m src.data.build_foundation --since 202606 --until 202606
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

warnings.filterwarnings("ignore", category=FutureWarning)

from src.data import foundation as fdn

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "data" / "processed" / "shards"
OUT = ROOT / "data" / "processed" / "foundation_arrivals_full.parquet"


def build(since: str = None, until: str = None, mongo_url: str = "mongodb://localhost:27017"):
    cfg = fdn.Config()
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=8000)
    win, gps = client["winicari"], client["Historique_pos"]

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    usable = fdn.build_usable_lines(win, cfg)
    days = fdn.gps_days(gps, cfg)
    months = sorted({d[1:7] for d in days})
    if since:
        months = [m for m in months if m >= since]
    if until:
        months = [m for m in months if m <= until]
    print(f"usable lines={len(usable)} | days={len(days)} | months={len(months)} "
          f"({months[0]}..{months[-1]})", flush=True)

    t_all = time.time()
    for m in months:
        shard = SHARD_DIR / f"foundation_{m}.parquet"
        if shard.exists():
            print(f"[{m}] shard exists, skip", flush=True)
            continue
        mdays = [d for d in days if d[1:7] == m]
        t0 = time.time()
        frames, n_cand, n_ok = [], 0, 0
        for day in mdays:
            for (dy, line, soc, bus) in fdn.candidates_for_day(gps, day, usable, cfg):
                n_cand += 1
                try:
                    f = fdn.reconstruct_bus_day(gps, dy, line, soc, bus, usable[(line, soc)], cfg)
                except Exception as e:                      # never let one bus-day kill the run
                    print(f"   !! {dy} {line}/{soc}/{bus}: {e}", flush=True)
                    continue
                if len(f):
                    frames.append(f); n_ok += 1
        out = (pd.concat(frames, ignore_index=True) if frames
               else pd.DataFrame(columns=cfg.out_columns))
        out.to_parquet(shard, index=False)
        mr = (100 * out["matched"].mean()) if len(out) else 0
        print(f"[{m}] days={len(mdays)} cand={n_cand} busdays_with_trips={n_ok} "
              f"rows={len(out)} match={mr:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(SHARD_DIR.glob("foundation_*.parquet"))]
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cfg.out_columns)
    full.to_parquet(OUT, index=False)
    bd = full.groupby(["day", "line", "societe", "bus"]).ngroups if len(full) else 0
    tr = full.groupby(["day", "line", "societe", "bus", "trip_id"]).ngroups if len(full) else 0
    print(f"\nDONE in {time.time()-t_all:.0f}s -> {OUT}", flush=True)
    if len(full):
        print(f"rows={len(full)} bus-days={bd} trips={tr} lines={full['line'].nunique()} "
              f"overall match={100*full['matched'].mean():.0f}%", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="first month YYYYMM (inclusive)")
    ap.add_argument("--until", help="last month YYYYMM (inclusive)")
    ap.add_argument("--mongo-url", default="mongodb://localhost:27017")
    args = ap.parse_args()
    build(since=args.since, until=args.until, mongo_url=args.mongo_url)
