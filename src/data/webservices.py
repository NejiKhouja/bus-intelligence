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
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()

WEBSERVICE_URL = os.getenv("WEBSERVICE_URL", "").rstrip("/")

# Disjoncteur : les webservices tournent sur le RÉSEAU LOCAL de la plateforme (pas d'URL
# publique) -- depuis un serveur cloud (Render, Allemagne) chaque appel TIME OUT au lieu
# d'être refusé, et un timeout de 30s dans un endpoint async bloque l'event loop assez
# longtemps pour faire échouer le health check de Render (5s) et tuer l'instance
# (constaté 2026-07-17). Ici : connexion limitée à ~3s, et après UN échec réseau, le
# service est marqué injoignable 10 min -- tous les appels suivants échouent
# instantanément (les appelants attrapent déjà l'exception et retombent sur
# l'historique / le magasin ingéré, voir main.py).
_down_until = 0.0
_CONNECT_TIMEOUT_S = 3.05


def _guard() -> None:
    if time.time() < _down_until:
        raise RuntimeError("webservice marqué injoignable (circuit ouvert, nouvel essai plus tard)")


def _mark_down(e: Exception) -> None:
    global _down_until
    _down_until = time.time() + 600
    # Attendu depuis Render (réseau privé, voir le commentaire du disjoncteur ci-dessus) --
    # pas une panne à traiter, juste un rappel que ce chemin restera silencieux 10 min.
    print(f"  webservice injoignable depuis ce serveur (normal, réseau privé) : "
          f"{e.__class__.__name__} -- nouvel essai dans 10 min")


def _get(path: str, params: dict, read_timeout: float, stream: bool = False) -> requests.Response:
    _guard()
    try:
        r = requests.get(f"{_base_url()}{path}", params=params,
                         timeout=(_CONNECT_TIMEOUT_S, read_timeout), stream=stream)
    except (requests.ConnectionError, requests.Timeout) as e:
        _mark_down(e)
        raise
    r.raise_for_status()
    return r


def _base_url() -> str:
    if not WEBSERVICE_URL:
        raise RuntimeError("WEBSERVICE_URL non défini (voir .env) -- impossible d'appeler "
                          "les web services de la plateforme.")
    return WEBSERVICE_URL


def is_day_ready(day: str) -> bool:
    """`day` au format YYYYMMDD. Le traitement de nuit peut exister mais être vide --
    l'API renvoie déjà `ready=false` dans ce cas côté plateforme (voir la précision
    donnée : collection absente OU countDocuments()==0 -> ready=false)."""
    r = _get("/Service/isDayReady", {"day": day}, read_timeout=30)
    return bool(r.json().get("ready", False))


def _slim_ping(p: dict) -> dict | None:
    """Ping brut du service -> dict compact à 7 champs (line/bus/t/lat/lon/speed/voyage).
    Retourne None si le ping est inutilisable (pas de position/date/ligne/bus)."""
    loc = p.get("localisation") or {}
    bus = p.get("bus") or {}
    svc = p.get("service") or {}
    line = str(svc.get("codeLigne", "") or "")
    bus_code = str(bus.get("code", "") or "")
    if not line or not bus_code or loc.get("x") is None or loc.get("y") is None or not p.get("date"):
        return None
    speed = bus.get("vitesse")
    return {
        "line": line,
        "bus": bus_code,
        "t": p["date"],
        "lat": float(loc["x"]),
        "lon": float(loc["y"]),
        "speed": float(speed) if speed is not None else None,
        "voyage": svc.get("voyage"),
    }


def get_pings_for_day(day: str, societe: str | None = None) -> list[dict]:
    """`day` au format YYYYMMDD. Retourne des pings COMPACTS (voir _slim_ping) -- PAS
    filtrés par ligne/bus, une société peut couvrir plusieurs lignes/bus en un seul appel.

    Parsé en STREAMING (ijson) plutôt que r.json() : une journée complète d'une grosse
    société fait ~10 Mo de JSON, soit 100-200 Mo une fois matérialisée en dicts Python --
    c'est précisément ce qui tuait le worker Render 512MB par OOM à chaque tentative de
    scoring en direct (constaté 2026-07-17 : crash juste après current-anomalies?societe=
    S.T.S, la plus grosse société). En streaming, seuls les dicts compacts à 7 champs
    survivent au parcours ; le JSON complet ne réside jamais en mémoire. Les floats ijson
    (Decimal) sont convertis dans _slim_ping. Repli sur r.json() + compaction immédiate si
    ijson n'est pas installé (env de dev pas encore à jour) -- même sortie, pic mémoire
    supérieur."""
    params = {"day": day}
    if societe:
        params["societe"] = societe
    r = _get("/Service/getPingsForDay", params, read_timeout=120, stream=True)
    try:
        import ijson
    except ImportError:
        rows = [s for p in r.json() if (s := _slim_ping(p)) is not None]
        return rows
    # r.raw ne décompresse pas gzip/deflate par défaut -- sans ça ijson lirait des octets
    # compressés et échouerait dès le premier token.
    r.raw.decode_content = True
    rows = []
    try:
        for p in ijson.items(r.raw, "item"):
            s = _slim_ping(p)
            if s is not None:
                rows.append(s)
    finally:
        r.close()
    return rows


def get_ticket_totals_for_day(day: str) -> list[dict]:
    """`day` au format YYYY-MM-DD (PAS YYYYMMDD -- confirmé sur le service réel)."""
    r = _get("/ServiceDetais/getTicketTotalsForDay", {"day": day}, read_timeout=60)
    return r.json()


def get_ticket_details_for_day(day: str) -> list[dict]:
    """`day` au format YYYY-MM-DD. Tickets individuels (pour le détail par arrêt)."""
    r = _get("/ticketsHorsLigne/getTicketDetailsForDay", {"day": day}, read_timeout=60)
    return r.json()


def group_pings_by_bus_line(pings: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Regroupe les pings COMPACTS (sortie de get_pings_for_day, voir _slim_ping) par
    (line, bus) -- un jour/société peut couvrir plusieurs lignes/bus, alors que
    /api/anomaly/score-live score UN bus-jour exact à la fois.
    Retourne {(line, bus_code_str): [pings...]}."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pings:
        groups[(p["line"], p["bus"])].append(p)
    return dict(groups)


def pings_to_score_live_rows(pings: list[dict]) -> list[dict]:
    """Un groupe de pings COMPACTS (déjà filtré à un seul bus/ligne, voir
    group_pings_by_bus_line) -> liste de dicts au format GpsPingRow attendu par
    /api/anomaly/score-live (t/lat/lon/speed/voyage). Le filtrage des pings inutilisables
    est déjà fait au parsing (_slim_ping) ; ne reste qu'à projeter les champs.

    `speed` : le service ne renvoie plus le `speed` racine (voir docs/webservice_fields.txt
    -- seul bus.vitesse est demandé), donc systématiquement replié sur bus.vitesse en
    amont -- à savoir : `bus.vitesse` est souvent 0/obsolète sur les données récentes
    (voir foundation.load_pings), donc ce champ sera moins fiable qu'avant tant que
    `speed` racine n'est pas redemandé à la plateforme.
    """
    return [{"t": p["t"], "lat": p["lat"], "lon": p["lon"],
             "speed": p["speed"], "voyage": p["voyage"]} for p in pings]


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
