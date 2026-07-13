"""Démo « Créer une ligne » -- page Streamlit autonome, RIEN n'est persisté en base : le
contrôleur dessine un itinéraire sur Google Maps (clic pour ajouter des arrêts, en partant
du terminus), et on affiche un horaire estimé en comparant DEUX méthodes : le temps de
trajet réel Google (Directions API, route routière) et une estimation basée sur la vitesse
commerciale moyenne DE NOTRE FLOTTE (mesurée sur données réelles, voir FLEET_AVG_SPEED_KMH)
-- utile en particulier quand la ligne dessinée est nouvelle et n'a aucun historique propre.

Le résultat de l'appel Directions (payant au-delà du quota gratuit Google, voir
docs) est mis en cache localement par séquence d'arrêts (décision utilisateur 2026-07-11 :
« garder les coûts bas ») -- PAS une sauvegarde de ligne visible/rechargeable, juste un
cache technique pour ne pas refacturer un itinéraire déjà demandé.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.dashboard.i18n import t

COMPONENT_DIR = (Path(__file__).parent / "components" / "route_map").resolve()
CACHE_PATH = Path("data/cache/demo_route_directions_cache.json")

# Vitesse commerciale moyenne mesurée sur données réelles (2026-07-11) : 29 746 trajets
# arrêt-à-arrêt, trajets bien suivis (match_rate >= 0.9), distance = haversine entre arrêts
# consécutifs (PAS trip_stops.dist_m, qui est une distance de calage GPS-arrêt, pas une
# distance de trajet -- confondre les deux donnait ~0.6 km/h, absurde). Médiane retenue
# (plus robuste que la moyenne aux trajets très courts/longs) : 33.8 km/h -> arrondi 34.
FLEET_AVG_SPEED_KMH = 34.0
# Immobilisation médiane à un arrêt intermédiaire (hors terminus), mêmes trajets : 30s.
FLEET_AVG_DWELL_S = 30

_component_func = components.declare_component("route_map", path=str(COMPONENT_DIR))


def _get_api_key() -> str | None:
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GOOGLE_MAPS_API_KEY")
    except Exception:
        return None


def _cache_key(stops: list) -> str:
    rounded = [(round(s["lat"], 5), round(s["lng"], 5)) for s in stops]
    return hashlib.sha1(json.dumps(rounded).encode()).hexdigest()


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _fmt_duration(seconds: float) -> str:
    m = round(seconds / 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}" if h else f"{m} min"


def render_route_demo_page():
    st.title("🗺️ " + t("route_demo_title"))
    st.caption(t("route_demo_caption"))
    st.markdown("---")

    api_key = _get_api_key()
    if not api_key:
        st.warning(t("route_demo_no_key_warning"))
        with st.expander(t("route_demo_no_key_howto"), expanded=True):
            st.markdown(t("route_demo_no_key_steps"))
        api_key = st.text_input(t("route_demo_paste_key"), type="password",
                                key="route_demo_session_key")
        if not api_key:
            st.stop()

    for k, default in (("route_demo_stops", []), ("route_demo_clear_sig", 0),
                       ("route_demo_undo_sig", 0), ("route_demo_compute", False),
                       ("route_demo_legs", None)):
        if k not in st.session_state:
            st.session_state[k] = default

    col_map, col_side = st.columns([3, 2])

    with col_side:
        st.subheader(t("route_demo_stops_header"))
        stops = st.session_state["route_demo_stops"]
        if not stops:
            st.info(t("route_demo_no_stops_yet"))
        else:
            for i, s in enumerate(stops):
                label = t("route_demo_terminal_label") if i == 0 else f"{t('route_demo_stop_label')} {i}"
                st.caption(f"**{label}** — {s['lat']:.5f}, {s['lng']:.5f}")

        bcols = st.columns(2)
        if bcols[0].button(t("route_demo_undo_btn"), disabled=not stops, width='stretch'):
            st.session_state["route_demo_undo_sig"] += 1
            st.session_state["route_demo_stops"] = stops[:-1]
            st.session_state["route_demo_legs"] = None
            st.rerun()
        if bcols[1].button(t("route_demo_clear_btn"), disabled=not stops, width='stretch'):
            st.session_state["route_demo_clear_sig"] += 1
            st.session_state["route_demo_stops"] = []
            st.session_state["route_demo_legs"] = None
            st.rerun()

        st.markdown("---")
        depart_time = st.time_input(t("route_demo_departure_time"), value=None,
                                    key="route_demo_depart")

        finish_disabled = len(stops) < 2
        if st.button(t("route_demo_finish_btn"), type="primary", disabled=finish_disabled,
                    width='stretch'):
            cache = _load_cache()
            ck = _cache_key(stops)
            if ck in cache:
                st.session_state["route_demo_legs"] = cache[ck]
                st.session_state["route_demo_compute"] = False
                st.toast(t("route_demo_cache_hit"))
            else:
                st.session_state["route_demo_compute"] = True
            st.rerun()

    with col_map:
        center_lat, center_lon = 35.0, 9.5  # Tunisie centrale, repère par défaut
        if stops:
            center_lat = sum(s["lat"] for s in stops) / len(stops)
            center_lon = sum(s["lng"] for s in stops) / len(stops)
        result = _component_func(
            api_key=api_key, center_lat=center_lat, center_lon=center_lon,
            existing_stops=stops, clear_signal=st.session_state["route_demo_clear_sig"],
            undo_signal=st.session_state["route_demo_undo_sig"],
            compute_directions=st.session_state["route_demo_compute"],
            key="route_map_component", default=None,
        )
        st.caption(t("route_demo_map_help"))

    if result and isinstance(result, dict):
        new_stops = result.get("stops") or []
        if new_stops != st.session_state["route_demo_stops"]:
            st.session_state["route_demo_stops"] = new_stops
            st.session_state["route_demo_legs"] = None
            st.rerun()
        legs = result.get("legs")
        if legs and st.session_state["route_demo_compute"]:
            st.session_state["route_demo_legs"] = legs
            st.session_state["route_demo_compute"] = False
            cache = _load_cache()
            cache[_cache_key(st.session_state["route_demo_stops"])] = legs
            _save_cache(cache)
            st.rerun()

    legs = st.session_state["route_demo_legs"]
    if legs:
        _render_timetable(stops, legs, depart_time)


def _render_timetable(stops: list, legs: list, depart_time):
    st.markdown("---")
    st.subheader(t("route_demo_timetable_header"))

    total_google_s = sum(l["duration_s"] or 0 for l in legs)
    total_dist_km = sum((l["distance_m"] or 0) for l in legs) / 1000
    st.caption(t("route_demo_timetable_caption", dist=round(total_dist_km, 1),
               n_stops=len(stops)))

    base = depart_time or time.strptime("06:00", "%H:%M")
    base_seconds = base.hour * 3600 + base.minute * 60 if hasattr(base, "hour") else 6 * 3600

    def _fmt_clock(seconds_from_midnight):
        seconds_from_midnight %= 24 * 3600
        h = int(seconds_from_midnight // 3600)
        m = int((seconds_from_midnight % 3600) // 60)
        return f"{h:02d}:{m:02d}"

    rows = []
    cum_google = base_seconds
    cum_fleet = base_seconds
    rows.append({
        "Arrêt": t("route_demo_terminal_label"),
        t("route_demo_col_google"): _fmt_clock(cum_google),
        t("route_demo_col_fleet"): _fmt_clock(cum_fleet),
    })
    for i, leg in enumerate(legs):
        dist_km = (leg["distance_m"] or 0) / 1000
        google_s = leg["duration_s"] or 0
        fleet_s = (dist_km / FLEET_AVG_SPEED_KMH) * 3600
        cum_google += google_s
        cum_fleet += fleet_s
        is_last = i == len(legs) - 1
        if not is_last:
            cum_fleet += FLEET_AVG_DWELL_S
            cum_google += FLEET_AVG_DWELL_S  # même hypothèse d'arrêt pour les deux colonnes
        label = t("route_demo_terminal_label") if is_last else f"{t('route_demo_stop_label')} {i + 1}"
        rows.append({
            "Arrêt": label,
            t("route_demo_col_google"): _fmt_clock(cum_google),
            t("route_demo_col_fleet"): _fmt_clock(cum_fleet),
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    mcols = st.columns(3)
    mcols[0].metric(t("route_demo_metric_distance"), f"{total_dist_km:.1f} km")
    mcols[1].metric(t("route_demo_metric_google_eta"), _fmt_duration(total_google_s))
    fleet_total_s = (total_dist_km / FLEET_AVG_SPEED_KMH) * 3600 + FLEET_AVG_DWELL_S * max(0, len(legs) - 1)
    mcols[2].metric(t("route_demo_metric_fleet_eta"), _fmt_duration(fleet_total_s))

    st.caption(t("route_demo_new_line_note", speed=int(FLEET_AVG_SPEED_KMH), dwell=FLEET_AVG_DWELL_S))
