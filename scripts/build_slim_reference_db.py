"""Regenerate data/reference/winicari_reference_slim.db from the full reference DB.

Slim = same schema as the full DB, minus the per-station ticket detail
(tickets_station_daily/tickets_station_trip_daily, ~600k rows) that only the excluded
196MB per-station drill-down model reads (see Dockerfile.render.dockerignore). Emptied
rather than dropped: every query written against the full schema still works unchanged
on the slim DB, it just returns nothing for those two tables.

Re-run this whenever the full reference DB's schema changes (new table/column) -- the
slim DB is a committed git artifact (~59MB, under GitHub's 100MB limit), not derived at
build/deploy time, so it silently goes stale otherwise. Confirmed 2026-07-20: it was
missing the driver_code column and the whole driver_services table (added to the full DB
after the slim DB's last regeneration) -- /api/drivers-ranked and the driver chip on
anomaly cards were silently empty in production, no error, no crash, just nothing.

Usage: python scripts/build_slim_reference_db.py
"""
import shutil
import sqlite3
from pathlib import Path

FULL = Path("data/reference/winicari_reference.db")
SLIM = Path("data/reference/winicari_reference_slim.db")
EMPTY_TABLES = ["tickets_station_daily", "tickets_station_trip_daily"]


def main():
    if not FULL.exists():
        raise SystemExit(f"{FULL} not found -- run this from the repo root.")
    if SLIM.exists():
        SLIM.unlink()
    shutil.copyfile(FULL, SLIM)

    conn = sqlite3.connect(SLIM)
    try:
        for t in EMPTY_TABLES:
            conn.execute(f"DELETE FROM {t}")
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()

    print(f"{SLIM} regenerated: {SLIM.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
