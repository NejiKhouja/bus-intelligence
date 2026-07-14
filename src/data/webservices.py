"""Client léger pour les web services de la plateforme (GPS + billetterie), voir
docs/WEBSERVICES_NEEDED.md et docs/webservice_fields.txt pour les champs demandés/reçus.

Vérifié directement contre les services réels le 2026-07-13 (voir conversation) -- champs
déjà réduits au strict nécessaire côté plateforme (ex. getPingsForDay : 121 Mo -> 9.6 Mo
pour la même société/jour, confirmé). Ce module fait juste l'appel HTTP + la mise en forme
vers ce que /api/anomaly/score-live et /api/ticket-anomaly/score-live attendent -- aucun
appel MongoDB direct ici, cohérent avec le principe déjà établi pour ces deux endpoints.
"""
from __future__ import annotations

import os
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()

WEBSERVICE_URL = os.getenv("WEBSERVICE_URL", "").rstrip("/")


def _base_url() -> str:
    if not WEBSERVICE_URL:
        raise RuntimeError("WEBSERVICE_URL non défini (voir .env) -- impossible d'appeler "
                          "les web services de la plateforme.")
    return WEBSERVICE_URL


def is_day_ready(day: str) -> bool:
    """`day` au format YYYYMMDD. Le traitement de nuit peut exister mais être vide --
    l'API renvoie déjà `ready=false` dans ce cas côté plateforme (voir la précision
    donnée : collection absente OU countDocuments()==0 -> ready=false)."""
    r = requests.get(f"{_base_url()}/Service/isDayReady", params={"day": day}, timeout=30)
    r.raise_for_status()
    return bool(r.json().get("ready", False))


def get_pings_for_day(day: str, societe: str | None = None) -> list[dict]:
    """`day` au format YYYYMMDD. Pings bruts (forme réduite : _id, localisation, date,
    service.codeLigne, service.voyage, bus.code, bus.vitesse) -- PAS filtrés par ligne/bus,
    une société peut couvrir plusieurs lignes/bus en un seul appel."""
    params = {"day": day}
    if societe:
        params["societe"] = societe
    r = requests.get(f"{_base_url()}/Service/getPingsForDay", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def get_ticket_totals_for_day(day: str) -> list[dict]:
    """`day` au format YYYY-MM-DD (PAS YYYYMMDD -- confirmé sur le service réel)."""
    r = requests.get(f"{_base_url()}/ServiceDetais/getTicketTotalsForDay",
                     params={"day": day}, timeout=60)
    r.raise_for_status()
    return r.json()


def get_ticket_details_for_day(day: str) -> list[dict]:
    """`day` au format YYYY-MM-DD. Tickets individuels (pour le détail par arrêt)."""
    r = requests.get(f"{_base_url()}/ticketsHorsLigne/getTicketDetailsForDay",
                     params={"day": day}, timeout=60)
    r.raise_for_status()
    return r.json()


def group_pings_by_bus_line(pings: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Regroupe les pings bruts par (codeLigne, bus.code) -- un jour/société peut couvrir
    plusieurs lignes/bus, alors que /api/anomaly/score-live score UN bus-jour exact à la
    fois. Retourne {(line, bus_code_str): [pings...]}."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pings:
        svc = p.get("service") or {}
        bus = p.get("bus") or {}
        line = str(svc.get("codeLigne", ""))
        bus_code = str(bus.get("code", ""))
        if not line or not bus_code:
            continue
        groups[(line, bus_code)].append(p)
    return dict(groups)


def pings_to_score_live_rows(pings: list[dict]) -> list[dict]:
    """Un groupe de pings BRUTS (déjà filtré à un seul bus/ligne, voir
    group_pings_by_bus_line) -> liste de dicts au format GpsPingRow attendu par
    /api/anomaly/score-live (t/lat/lon/speed/voyage).

    `speed` : le service ne renvoie plus le `speed` racine (voir docs/webservice_fields.txt
    -- seul bus.vitesse est demandé), donc systématiquement replié sur bus.vitesse ici --
    à savoir : `bus.vitesse` est souvent 0/obsolète sur les données récentes (voir
    foundation.load_pings), donc ce champ sera moins fiable qu'avant tant que `speed`
    racine n'est pas redemandé à la plateforme.
    """
    rows = []
    for p in pings:
        loc = p.get("localisation") or {}
        bus = p.get("bus") or {}
        svc = p.get("service") or {}
        if loc.get("x") is None or loc.get("y") is None or not p.get("date"):
            continue
        rows.append({
            "t": p["date"],
            "lat": loc["x"],
            "lon": loc["y"],
            "speed": bus.get("vitesse"),
            "voyage": svc.get("voyage"),
        })
    return rows


def ticket_totals_to_rows(raw_rows: list[dict], day_yyyymmdd: str) -> list[dict]:
    """Lignes brutes de getTicketTotalsForDay -> liste de dicts au format TicketDayRow
    attendu par /api/ticket-anomaly/score-live (societe/line/bus/day/nbr_ticket/recette).
    `day_yyyymmdd` est réinjecté tel quel (le format interne du modèle, YYYYMMDD) plutôt
    que reconverti depuis le "date" du service -- évite toute ambiguïté de fuseau/format.
    """
    rows = []
    for r in raw_rows:
        if not r.get("societe") or r.get("CodeLigne") is None or r.get("codeBus") is None:
            continue
        rows.append({
            "societe": r["societe"],
            "line": str(r["CodeLigne"]),
            "bus": str(r["codeBus"]),
            "day": day_yyyymmdd,
            "nbr_ticket": int(r.get("nbrTicket") or 0),
            "recette": float(r.get("recette") or 0.0),
        })
    return rows
