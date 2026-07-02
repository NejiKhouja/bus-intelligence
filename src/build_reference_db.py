"""Pipeline de reconstruction de la base de référence WiniCari -- point d'entrée admin.

Utilisation
-----------
    conda activate bus-intelligence
    python -m src.build_reference_db                          # tables de référence seulement (<1 min)
    python -m src.build_reference_db --with-trips              # + reconstruction complète des trajets GPS (~40 min)
    python -m src.build_reference_db --with-trips --since 20250101 --until 20250630
    python -m src.build_reference_db --with-trips --company S.R.T.BIZERTE --company TUS

Ce que ça fait
--------------
  1. companies       -- regroupement canonique + fenêtre GPS + enrichissement `societe`
  2. lines           -- union ligne/tickets/GPS
  3. stops           -- clustering géographique (DBSCAN) de tous les points géocodés bruts
  4. line_stops      -- résolveur à 6 niveaux (jointures directes -> repli billetterie)
  5. tickets_daily   -- agrégats journaliers billetterie (depuis winicari.details)
  6. [optionnel] trips/trip_stops -- reconstruction GPS complète (`--with-trips`)
  7. export_foundation_parquet -- réécrit data/processed/foundation_arrivals_full.parquet
                                   depuis la base SQLite, pour que tous les modules IA
                                   existants (delay/fallback/anomaly/RAG) bénéficient des
                                   données enrichies SANS aucune modification de leur code.

Un admin qui a simplement ajouté/corrigé des données MongoDB (nouvelles stations, nouvelles
lignes, etc.) peut relancer cette commande SANS `--with-trips` pour rafraîchir rapidement
tout sauf les trajets. `--with-trips` n'est nécessaire que si l'historique GPS lui-même a
changé (nouvelle fenêtre de données, nouvelle société connectée).

Prérequis
---------
  - Accès MongoDB (MONGO_URL dans .env) -- bases winicari, OpenData, Historique_Tickets,
    et Historique_pos (uniquement si --with-trips)
  - environnement conda bus-intelligence (pandas, scikit-learn, pymongo)

Après reconstruction
--------------------
    from src.data import reference_db as rdb
    conn = rdb.init_db()
    usable = rdb._usable_lines_from_line_stops(conn)   # géométrie ligne/arrêt pour l'API

    # Ou charger le parquet de fondation régénéré, comme avant :
    import pandas as pd
    fa = pd.read_parquet("data/processed/foundation_arrivals_full.parquet")
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FOUNDATION_OUT = Path("data/processed/foundation_arrivals_full.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-trips", action="store_true",
                         help="Reconstruit aussi trips/trip_stops depuis les pings GPS (~40 min)")
    parser.add_argument("--since", default=None,
                         help="Jour de début (YYYYMMDD) pour --with-trips, ex. 20250101")
    parser.add_argument("--until", default=None,
                         help="Jour de fin (YYYYMMDD) pour --with-trips")
    parser.add_argument("--company", action="append", default=None, dest="companies",
                         help="Restreindre --with-trips à une société (répétable, ex. --company TCV --company TUS)")
    args = parser.parse_args()

    from src.data import reference_db as rdb
    from src.data.db import get_db

    t0 = time.time()
    print("=" * 60)
    print("Reconstruction de la base de référence WiniCari")
    print("=" * 60)

    conn = rdb.init_db()
    wi_db = get_db("winicari")
    od_db = get_db("OpenData")
    tk_db = get_db("Historique_Tickets")

    print("\n[1/7] Sociétés (regroupement canonique + fenêtre GPS + enrichissement)")
    print("-" * 50)
    gps_db = get_db("Historique_pos")
    rdb.populate_companies(conn, gps_db=gps_db)
    rdb.enrich_companies_from_societe(conn, wi_db)
    n_companies = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    print(f"  -> {n_companies} sociétés")

    print("\n[2/7] Lignes (union ligne/tickets/GPS)")
    print("-" * 50)
    rdb.populate_lines(conn, wi_db, tk_db=tk_db, gps_db=gps_db)
    n_lines = conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
    print(f"  -> {n_lines} lignes")

    print("\n[3/7] Arrêts (clustering géographique DBSCAN)")
    print("-" * 50)
    rdb.populate_stops(conn, wi_db, od_db)
    n_stops = conn.execute("SELECT COUNT(*) FROM stops").fetchone()[0]
    print(f"  -> {n_stops} arrêts canoniques")

    print("\n[4/7] Géométrie ligne/arrêt (résolveur à 6 niveaux)")
    print("-" * 50)
    tier_counts = rdb.populate_line_stops(conn, wi_db, od_db, tk_db=tk_db)
    n_resolved = conn.execute(
        "SELECT COUNT(DISTINCT line_id) FROM line_stops").fetchone()[0]
    print(f"  -> {n_resolved}/{n_lines} lignes résolues  (détail par niveau : {tier_counts})")

    print("\n[5/7] Billetterie journalière (winicari.details)")
    print("-" * 50)
    rdb.populate_tickets_daily(conn, wi_db)
    n_days = conn.execute("SELECT COUNT(*) FROM tickets_daily").fetchone()[0]
    print(f"  -> {n_days} jours-lignes-bus")

    if args.with_trips:
        print("\n[6/7] Trajets GPS (reconstruction complète -- peut prendre ~40 min)")
        print("-" * 50)
        stats = rdb.populate_trips(conn, gps_db, since_day=args.since, until_day=args.until,
                                    companies=args.companies)
        print(f"  -> {stats['n_trips']} trajets, {stats.get('n_loop_unknown_full', 0)} en boucle "
              f"(full=NULL), {stats['n_stop_rows']} arrêts-trajets")
    else:
        print("\n[6/7] Trajets GPS -- SKIP (passer --with-trips pour reconstruire, ~40 min)")
        print("-" * 50)

    print("\n[7/7] Export du parquet de fondation (compatibilité modules IA existants)")
    print("-" * 50)
    n_trips_db = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    if n_trips_db == 0:
        print(f"  -> SKIP : table `trips` vide (relancer avec --with-trips au moins une fois)")
    else:
        exp = rdb.export_foundation_parquet(conn, FOUNDATION_OUT)
        print(f"  -> {exp['rows']:,} lignes, {exp['trips']:,} trajets, {exp['lines']} lignes, "
              f"{exp['companies']} sociétés -> {exp['out_path']}")

    conn.close()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Base de référence reconstruite en {elapsed/60:.1f} minutes")
    print(f"  {Path('data/reference/winicari_reference.db').resolve()}")
    if args.with_trips:
        print(f"  {FOUNDATION_OUT.resolve()}")
    print("\nEntraîner les modèles à partir des données rafraîchies :")
    print("  python -m src.train_pipeline")


if __name__ == "__main__":
    main()
