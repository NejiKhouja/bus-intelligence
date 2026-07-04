"""Construction par lots du jeu de données de fondation GPS sur une plage de jours.

Ce que ce fichier fait EN PLUS DE foundation.py
`foundation.py` sait reconstruire UN seul bus-jour (un `(jour, ligne, societe, bus)`).
Ce fichier est l'**orchestrateur** qui exécute cette logique sur toute la base de données
et la transforme en un seul jeu de données sur disque. Il fait quatre choses que foundation.py
ne fait pas :

1. ÉNUMÉRER le travail — pour chaque jour de la plage, il appelle `candidates_for_day`
   pour savoir quels `(ligne, societe, bus)` ont réellement circulé sur une ligne
   avec une géométrie utilisable ce jour-là.
2. BOUCLE + ISOLATION — appelle `reconstruct_bus_day` pour chaque candidat dans un try/except,
   de sorte qu'un mauvais bus-jour enregistre un avertissement et est ignoré au lieu de
   faire échouer toute l'exécution.
3. FRAGMENT + REPRISE — écrit un parquet par mois dans data/processed/shards/ et IGNORE les mois
   dont le fragment existe déjà, de sorte qu'une exécution de 40 min peut être arrêtée
   et redémarrée librement
   (c'est pourquoi une ré-exécution affiche par exemple « [202606] shard exists, skip »).
4. COMBINER — concatène tous les fragments mensuels en un seul fichier lu par tout l'aval :
   data/processed/foundation_arrivals_full.parquet.

Lecture des lignes de progression
    [202502] days=28 cand=219 busdays_with_trips=215 rows=17240 match=83% (89s)
      days                = collections de jours traitées ce mois
      cand                = bus-jours candidats (ligne,societe,bus) trouvés
      busdays_with_trips  = combien ont produit >=1 trajet (le reste n'avait pas de trajet propre)
      rows                = lignes d'arrivée aux arrêts écrites (une par arrêt couvert)
      match               = % d'arrêts couverts avec une arrivée GPS dans arrival_thresh_m

Utilisation :
    python -m src.data.build_foundation                # plage complète utilisable (>= 2022-06)
    python -m src.data.build_foundation --since 202501 # uniquement les mois >= 202501
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
    print(f"lignes utilisables={len(usable)} | jours={len(days)} | mois={len(months)} "
          f"({months[0]}..{months[-1]})", flush=True)

    t_all = time.time()
    for m in months:
        shard = SHARD_DIR / f"foundation_{m}.parquet"
        if shard.exists():
            print(f"[{m}] fragment existant, ignoré", flush=True)
            continue
        mdays = [d for d in days if d[1:7] == m]
        t0 = time.time()
        frames, n_cand, n_ok = [], 0, 0
        for day in mdays:
            for (dy, line, soc, bus) in fdn.candidates_for_day(gps, day, usable, cfg):
                n_cand += 1
                try:
                    f = fdn.reconstruct_bus_day(gps, dy, line, soc, bus, usable[(line, soc)], cfg)
                except Exception as e:                      # ne jamais laisser un bus-jour faire échouer l'exécution
                    print(f"   !! {dy} {line}/{soc}/{bus}: {e}", flush=True)
                    continue
                if len(f):
                    frames.append(f); n_ok += 1
        out = (pd.concat(frames, ignore_index=True) if frames
               else pd.DataFrame(columns=cfg.out_columns))
        out.to_parquet(shard, index=False)
        mr = (100 * out["matched"].mean()) if len(out) else 0
        print(f"[{m}] jours={len(mdays)} cand={n_cand} busdays_avec_trajets={n_ok} "
              f"lignes={len(out)} correspondance={mr:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(SHARD_DIR.glob("foundation_*.parquet"))]
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cfg.out_columns)
    full.to_parquet(OUT, index=False)
    bd = full.groupby(["day", "line", "societe", "bus"]).ngroups if len(full) else 0
    tr = full.groupby(["day", "line", "societe", "bus", "trip_id"]).ngroups if len(full) else 0
    print(f"\nTERMINÉ en {time.time()-t_all:.0f}s -> {OUT}", flush=True)
    if len(full):
        print(f"lignes={len(full)} bus-jours={bd} trajets={tr} lignes={full['line'].nunique()} "
              f"correspondance globale={100*full['matched'].mean():.0f}%", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="premier mois YYYYMM (inclus)")
    ap.add_argument("--until", help="dernier mois YYYYMM (inclus)")
    ap.add_argument("--mongo-url", default="mongodb://localhost:27017")
    args = ap.parse_args()
    build(since=args.since, until=args.until, mongo_url=args.mongo_url)
