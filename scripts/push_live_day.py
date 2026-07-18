"""Relais de données en direct : réseau local de la plateforme -> API cloud (Render).

Pourquoi ce script existe
-------------------------
Les webservices de la plateforme (WEBSERVICE_URL) tournent sur un réseau local SANS URL
publique : le serveur cloud (Render, Allemagne) ne peut pas les atteindre -- chaque
tentative time out (constaté 2026-07-17), et personne n'a la main sur le serveur des
webservices pour l'exposer. Ce script inverse le sens : il tourne sur une machine DU
réseau local (n'importe laquelle -- votre PC convient), tire la journée d'hier depuis
les webservices (accessibles d'ici), et la POUSSE vers l'API publique via
POST /api/ingest/gps-day et /api/ingest/ticket-day (sortie HTTPS simple, autorisée sur
à peu près n'importe quel réseau). Côté serveur, les scoreurs "live" lisent ce magasin
poussé en priorité (voir main.py::_score_all_gps_live).

Utilisation
-----------
    conda activate bus-intelligence
    python -m scripts.push_live_day                    # pousse HIER (jour "live" normal)
    python -m scripts.push_live_day --day 20260716     # pousse un jour précis
    python -m scripts.push_live_day --societes S.T.S S.R.T.K   # sociétés précises

Variables d'environnement (.env du dépôt, déjà en place pour la plupart) :
    WEBSERVICE_URL   -- webservices de la plateforme (réseau local)
    RENDER_API_URL   -- API cloud, ex. https://bus-intelligence.onrender.com
    API_KEY          -- même clé que celle configurée sur Render (en-tête X-API-Key)

À planifier (Windows, Planificateur de tâches / cron) chaque matin après le traitement
de nuit de la plateforme, ex. 07:00 :
    schtasks /Create /SC DAILY /ST 07:00 /TN "WiniCari push live" /TR ^
      "C:\\Users\\<vous>\\anaconda3\\envs\\bus-intelligence\\python.exe -m scripts.push_live_day"
(exécuté depuis la racine du dépôt ; ajouter /V1 et le répertoire de travail au besoin).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from src.data import webservices as ws  # noqa: E402


def _api() -> tuple[str, dict]:
    base = (os.getenv("RENDER_API_URL") or "").rstrip("/")
    if not base:
        raise SystemExit("RENDER_API_URL non défini (ex. https://bus-intelligence.onrender.com)")
    headers = {}
    key = os.getenv("API_KEY")
    if key:
        headers["X-API-Key"] = key
    return base, headers


def push_day(day: str, societes: list[str] | None) -> None:
    base, headers = _api()

    if not ws.is_day_ready(day):
        print(f"{day}: traitement de nuit pas prêt côté plateforme (isDayReady=false) -- rien à pousser.")
        return

    if not societes:
        r = requests.get(f"{base}/api/options", headers=headers, timeout=60)
        r.raise_for_status()
        societes = r.json().get("companies", [])
        print(f"Sociétés (via {base}/api/options) : {societes}")

    for soc in societes:
        try:
            pings = ws.get_pings_for_day(day, societe=soc)
        except Exception as e:
            print(f"  {soc}: échec webservice ({e}) -- ignorée")
            continue
        if not pings:
            print(f"  {soc}: aucun ping ce jour -- ignorée")
            continue
        r = requests.post(f"{base}/api/ingest/gps-day", headers=headers, timeout=300,
                          json={"day": day, "societe": soc, "pings": pings})
        r.raise_for_status()
        print(f"  {soc}: {r.json()}")

    day_dashed = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    try:
        rows = ws.get_ticket_totals_for_day(day_dashed)
    except Exception as e:
        print(f"  billetterie: échec webservice ({e}) -- ignorée")
        rows = None
    if rows:
        r = requests.post(f"{base}/api/ingest/ticket-day", headers=headers, timeout=120,
                          json={"day": day, "rows": rows})
        r.raise_for_status()
        print(f"  billetterie: {r.json()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", help="YYYYMMDD (défaut : hier)")
    ap.add_argument("--societes", nargs="*", help="sociétés à pousser (défaut : toutes via /api/options)")
    args = ap.parse_args()
    day = args.day or (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    print(f"Push du jour {day} vers {os.getenv('RENDER_API_URL')}")
    push_day(day, args.societes)


if __name__ == "__main__":
    main()
