"""WiniCari AI — Operations & Rider Dashboard (demo build).

Demo clock: the dataset ends 2026-06-21, so "now" = the current wall-clock time-of-day
on the latest day each line actually operated. This keeps every live view populated
with real data while behaving like a live system.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# `streamlit run src/dashboard/app.py` puts this file's own directory on sys.path, not the
# repo root -- so `from src...` imports (here and transitively, e.g. realtime.py's own
# `from src.data.fallback import ...`) fail unless the root is added explicitly first.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

from src.dashboard import realtime as rt
from src.dashboard.i18n import t, get_lang, set_lang, LANGS

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="WiniCari AI", page_icon="🚌", layout="wide")

st.markdown(
    """
    <style>
    .stProgress > div > div > div > div { background-color: #2563eb !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(path, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def _post(path, json=None, **params):
    try:
        r = requests.post(f"{API_URL}{path}", params=params, json=json, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=10)
def get_health():
    return _get("/health") or {}

@st.cache_data(ttl=60)
def get_companies():
    d = _get("/api/options")
    return d.get("companies", []) if d else []

@st.cache_data(ttl=60)
def get_lines(company):
    d = _get("/api/lines-ranked", societe=company) or _get("/api/lines", societe=company)
    return d.get("lines", []) if d else []

@st.cache_data(ttl=60)
def get_days_for_line(company, line):
    d = _get("/api/days-for-line", societe=company, line=line)
    return d.get("days", []) if d else []

@st.cache_data(ttl=60)
def get_buses_for_day(company, line, day):
    d = _get("/api/buses-for-day", societe=company, line=line, day=day)
    return d.get("buses", []) if d else []

@st.cache_data(ttl=60)
def get_directions(company, line):
    d = _get("/api/directions", societe=company, line=line)
    return d.get("directions", []) if d else []

@st.cache_data(ttl=60)
def get_prophet_lines(company):
    return _get("/api/prophet-lines", societe=company) or {"lines": [], "by_line": {}}

@st.cache_data(ttl=300)
def get_gps_track(company, line, bus, day):
    return _get("/api/gps-track", societe=company, line=line, bus=bus, day=day)

@st.cache_data(ttl=300)
def get_gps_gaps(company, line, bus, day):
    return _get("/api/gps-gaps", societe=company, line=line, bus=bus, day=day)

@st.cache_data(ttl=300)
def get_gap_examples(company, line):
    return _get("/api/gps-gap-examples", societe=company, line=line)

@st.cache_data(ttl=30)
def get_active_buses(company, line, query_time):
    return _get("/api/active-buses", societe=company, line=line, query_time=query_time)

@st.cache_data(ttl=30)
def get_eta_to_stop(company, line, bus, day, target_seq, query_time, model_type):
    return _get("/api/eta-to-stop", societe=company, line=line, bus=bus, day=day,
                target_seq=target_seq, query_time=query_time, model_type=model_type)

@st.cache_data(ttl=300)
def get_buses_for_line(company, line):
    d = _get("/api/buses-for-line", societe=company, line=line)
    return d.get("buses", []) if d else []

@st.cache_data(ttl=300)
def get_trip_detail(company, line, bus, day, trip_start):
    return _get("/api/trip-detail", societe=company, line=line, bus=bus, day=day, trip_start=trip_start)

@st.cache_data(ttl=60)
def get_anomaly_history(company, line=None, limit=50, include_bugs=False, direction=None):
    return _get("/api/anomaly-history", societe=company, line=line, limit=limit,
                include_data_bugs=include_bugs, dir=direction)

@st.cache_data(ttl=60)
def get_anomaly_explain(company, line, bus, day=None, include_bugs=False, direction=None):
    return _get("/api/anomaly-explain", societe=company, line=line, bus=bus, day=day,
                include_data_bugs=include_bugs, dir=direction)

@st.cache_data(ttl=60)
def get_anomaly_patterns(company, line=None):
    return _get("/api/anomaly-patterns", societe=company, line=line)

@st.cache_data(ttl=60)
def get_current_anomalies(company, line=None, direction=None):
    return _get("/api/current-anomalies", societe=company, line=line, dir=direction)

@st.cache_data(ttl=60)
def get_ticket_anomaly_history(company, line=None, bus=None, limit=50, client_safe=False):
    return _get("/api/ticket-anomaly-history", societe=company, line=line, bus=bus, limit=limit,
                client_safe=client_safe)

@st.cache_data(ttl=60)
def get_ticket_anomaly_patterns(company, line=None):
    return _get("/api/ticket-anomaly-patterns", societe=company, line=line)

@st.cache_data(ttl=300)
def get_ticket_anomaly_explain(company, line=None, bus=None, client_safe=False):
    return _get("/api/ticket-anomaly-explain", societe=company, line=line, bus=bus,
               client_safe=client_safe)

@st.cache_data(ttl=300)
def get_ticket_anomaly_stations(company, line, bus, day):
    return _get("/api/ticket-anomaly-stations", societe=company, line=line, bus=bus, day=day)

@st.cache_data(ttl=300)
def get_ticket_anomaly_reference(company, line):
    return _get("/api/ticket-anomaly-reference", societe=company, line=line)

@st.cache_data(ttl=300)
def get_reference_trip(company, line):
    return _get("/api/reference-trip", societe=company, line=line)

def chat_with_bot(query, k=5):
    return _post("/api/chatbot/ask", json={"query": query, "k": k})

def get_forecast(company, line, direction, periods=30):
    return _post("/api/predict/delay/forecast", societe=company, line=line,
                 direction=direction, periods=periods)

# ─────────────────────────────────────────────────────────────────────────────
# Demo clock
# ─────────────────────────────────────────────────────────────────────────────

HEALTH = get_health()
LATEST_DAY = HEALTH.get("latest_day") or datetime.now().strftime("%Y%m%d")

def demo_now(day=None):
    """Current wall-clock time-of-day applied to the demo operating day."""
    d = day or LATEST_DAY
    base = datetime.strptime(d, "%Y%m%d").date()
    return datetime.combine(base, datetime.now().time())

_MOIS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
            "juil.", "août", "sept.", "oct.", "nov.", "déc."]

def fmt_day(d):
    try:
        dt = datetime.strptime(str(d), "%Y%m%d")
        return f"{dt.day} {_MOIS_FR[dt.month - 1]} {dt.year}"
    except Exception:
        return d

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🚌 WiniCari AI")
st.sidebar.caption("Plateforme d'intelligence du transport")

# Sélecteur de langue -- couvre pour l'instant la page « Détection d'anomalies » (voir
# src/dashboard/i18n.py) ; les autres pages restent en français, à étendre au même schéma.
_lang_codes = list(LANGS.keys())
_lang_idx = _lang_codes.index(get_lang()) if get_lang() in _lang_codes else 0
_picked_lang = st.sidebar.selectbox(
    t("lang_label"), _lang_codes, index=_lang_idx,
    format_func=lambda code: LANGS[code], key="lang_selector",
)
if _picked_lang != get_lang():
    set_lang(_picked_lang)
    st.rerun()

if HEALTH.get("status") == "healthy":
    st.sidebar.success("● Système en ligne")
    st.sidebar.caption(f"{HEALTH.get('rows', 0):,} trajets · {len(HEALTH.get('models', []))} modèles")
else:
    st.sidebar.error("● API hors ligne — démarrez le serveur FastAPI")

st.sidebar.info(
    f"🟢 **Mode démo**\n\nSimulation du **{fmt_day(LATEST_DAY)}**, "
    f"{datetime.now().strftime('%H:%M:%S')}"
)

# Moteur de prédiction : on garde le meilleur modèle (HGBM, MAE ~2.7 min vs LSTM ~3.3).
model_type = "hgbm"
st.sidebar.markdown("---")

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Tableau de bord", "ETA en direct", "Repli GPS",
                 "Détection d'anomalies", "Démo — Créer une ligne", "Assistant", "Prévisions"],
        icons=["speedometer2", "geo-alt", "broadcast-pin", "shield-exclamation", "map",
               "chat-left-text", "graph-up-arrow"],
        menu_icon="layers", default_index=0,
        styles={
            "container": {"padding": "6px!important", "background-color": "#0f172a", "border-radius": "8px"},
            "menu-title": {"color": "#e2e8f0", "font-size": "13px"},
            "icon": {"color": "#cbd5e1", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "color": "#e2e8f0", "text-align": "left",
                         "margin": "4px", "border-radius": "6px", "--hover-color": "#1e293b"},
            "nav-link-selected": {"background-color": "#2563eb", "color": "white", "font-weight": "600"},
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

if selected == "Tableau de bord":
    st.title("Tableau de bord des opérations")
    st.caption("Démo en direct des fonctions IA de WiniCari sur des données de flotte réelles.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trajets dans les données", f"{HEALTH.get('rows', 0):,}")
    c2.metric("Opérateurs suivis", len(get_companies()))
    c3.metric("Modèles IA actifs", len(HEALTH.get("models", [])))
    c4.metric("Dernier jour de données", fmt_day(LATEST_DAY))

    st.markdown("### Ce que vous pouvez faire ici")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🛰️ ETA en direct")
        st.write("Choisissez votre ligne et votre arrêt, voyez dans combien de minutes votre bus arrive, et regardez-le avancer vers vous sur la carte.")
    with b:
        st.markdown("#### 📡 Repli GPS")
        st.write("Rejouez un vrai moment où un bus a perdu le GPS. Le système le détecte et le filtre de Kalman estime sa position en continu.")
    with c:
        st.markdown("#### 🛡️ Détection d'anomalies")
        st.write("Trajets signalés comme anormaux — *avec les raisons* (arrêt trop long, anormalement en retard, hors itinéraire) — et les tendances de la flotte.")

    st.info("Utilisez la barre latérale pour naviguer. Chaque vue tourne sur des données réelles, rejouées comme si c'était maintenant.")

# ─────────────────────────────────────────────────────────────────────────────
# Live ETA
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "ETA en direct":
    st.title("🛰️ ETA en direct")
    st.caption("Quand mon bus arrive-t-il ? — choisissez votre ligne et votre arrêt, puis regardez-le approcher.")
    st.markdown("---")

    companies = get_companies()
    if not companies:
        st.warning("Aucune donnée disponible depuis l'API.")
        st.stop()

    c1, c2 = st.columns(2)
    company = c1.selectbox("Opérateur", companies, key="eta_co")
    lines = get_lines(company)
    if not lines:
        st.warning("Aucune ligne pour cet opérateur.")
        st.stop()
    line = c2.selectbox("Ligne", lines, key="eta_line")

    # Trouver les bus en service « maintenant » sur le dernier jour d'exploitation
    qt_iso = demo_now().isoformat()
    active = get_active_buses(company, line, qt_iso)
    if not active or not active.get("buses"):
        st.warning(f"La ligne {line} n'a aucun trajet enregistré à simuler.")
        st.stop()

    op_day = active["day"]
    buses = active["buses"]
    live = [b for b in buses if b["status"] in ("active", "upcoming")] or buses
    _STATUS_FR = {"active": "en service", "upcoming": "à venir", "completed": "terminé", "unknown": "—"}
    st.caption(f"📅 Jour d'exploitation **{fmt_day(op_day)}** · heure simulée "
               f"**{pd.Timestamp(active['query_time']).strftime('%H:%M')}** · "
               f"{sum(b['status']=='active' for b in buses)} bus en circulation")

    labels = {f"Bus {b['bus']} · {b['dir']} · {_STATUS_FR.get(b['status'], b['status'])} "
              f"({pd.Timestamp(b['trip_start']).strftime('%H:%M')}→{pd.Timestamp(b['trip_end']).strftime('%H:%M')})": b
              for b in live}
    pick = st.selectbox("Choisissez un bus", list(labels.keys()), key="eta_bus")
    bus_info = labels[pick]
    bus = bus_info["bus"]

    # Effective clock: use 'now' if the bus is live; otherwise drop into its trip
    # (25% in) so there's always a moving bus with road ahead to demo.
    if bus_info["status"] == "active":
        eff_qt = qt_iso
        live_now = True
    else:
        ts, te = pd.Timestamp(bus_info["trip_start"]), pd.Timestamp(bus_info["trip_end"])
        eff_qt = (ts + (te - ts) * 0.25).isoformat()
        live_now = False

    eta = get_eta_to_stop(company, line, bus, op_day, 0, eff_qt, model_type)
    track = get_gps_track(company, line, bus, op_day)
    if not eta or not eta.get("predictions"):
        st.info("Ce bus n'a aucun arrêt à venir à prédire — choisissez un autre bus.")
        st.stop()

    preds = eta["predictions"]
    qt = pd.Timestamp(eta["query_time"])
    if not live_now:
        st.caption(f"⏱️ Aucun bus en circulation à {pd.Timestamp(qt_iso).strftime('%H:%M')} ; "
                   f"rejeu du **Bus {bus}** en cours de trajet à **{qt.strftime('%H:%M')}**.")

    # Arrêt du voyageur = l'un des prochains arrêts que le bus va atteindre
    stop_rows = []
    for p in preds:
        eta_ts = pd.Timestamp(p["eta"])
        stop_rows.append({
            "seq": int(p["seq"]),
            "Arrêt": p.get("stop") or f"Arrêt {p['seq']}",
            "Arrivée (ETA)": eta_ts.strftime("%H:%M"),
            "Dans (min)": round((eta_ts - qt).total_seconds() / 60, 1),
            "Retard prévu (min)": round(float(p["pred_delay_min"]), 1),
        })
    stops_df = pd.DataFrame(stop_rows)

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Où attendez-vous ?")
        st.caption("Cliquez votre arrêt dans le tableau.")
        ev = st.dataframe(
            stops_df.drop(columns=["seq"]), hide_index=True, width='stretch',
            on_select="rerun", selection_mode="single-row", key="eta_stop_table",
        )
        sel_rows = ev.selection.rows if ev and ev.selection else []
        rider = stops_df.iloc[sel_rows[0]] if sel_rows else stops_df.iloc[min(3, len(stops_df) - 1)]
        rider_seq = int(rider["seq"])

        st.markdown("---")
        st.metric(f"🚌 Arrive à **{rider['Arrêt']}** dans", f"{rider['Dans (min)']:.0f} min",
                  delta=f"{rider['Retard prévu (min)']:+.0f} min vs habituel")
        st.caption(f"Bus actuellement près de **{eta.get('current_stop') or 'départ'}** "
                   f"· {eta.get('current_delay_min', 0):+.0f} min vs habituel "
                   f"· statut : {_STATUS_FR.get(eta.get('status'), eta.get('status'))}")

        # ── Mettre en valeur le modèle : COMMENT l'ETA est produite ───────────
        with st.container(border=True):
            st.markdown(f"##### 🧠 Comment le modèle **{model_type.upper()}** obtient ce résultat")
            cur_delay = eta.get("current_delay_min", 0) or 0
            base_min = max(0.0, rider["Dans (min)"] - rider["Retard prévu (min)"])
            b1, b2, b3 = st.columns(3)
            b1.metric("Trajet habituel", f"{base_min:.0f} min", help="Temps de parcours de référence appris de l'historique")
            b2.metric("Retard du modèle", f"{rider['Retard prévu (min)']:+.0f} min",
                      help="Retard supplémentaire prédit à partir de l'état actuel du bus")
            b3.metric("→ ETA", f"{rider['Dans (min)']:.0f} min")
            st.caption(f"Le bus a actuellement **{cur_delay:+.0f} min** d'écart vs son rythme habituel ; "
                       f"le modèle propage cet écart arrêt par arrêt pour projeter l'arrivée à chaque arrêt.")
            trend = pd.DataFrame({
                "Arrêt": [p.get("stop") or f"#{p['seq']}" for p in preds],
                "Retard prévu (min)": [round(float(p["pred_delay_min"]), 1) for p in preds],
            })
            fig_d = px.line(trend, x="Arrêt", y="Retard prévu (min)", markers=True)
            fig_d.update_traces(line=dict(color="#2563eb", width=3))
            fig_d.update_layout(template="plotly_white", height=240,
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title=None, yaxis_title="Retard (min)")
            fig_d.add_vline(x=rider["Arrêt"], line_dash="dot", line_color="#16a34a")
            st.plotly_chart(fig_d, width='stretch', key="eta_delay_trend")
            st.caption("Le retard prévu s'accumule le long de l'itinéraire — la ligne verte marque votre arrêt.")

    with right:
        st.subheader("Carte en direct")
        if not track:
            st.info("Géométrie de carte indisponible pour cette ligne.")
        else:
            route = track["route"]
            P = rt.prep_track(track["track"])
            rider_route_seq = next(
                (r["seq"] for r in route if r.get("stop") == rider["Arrêt"]), rider_seq)
            rider_eta_ts = next((pd.Timestamp(p["eta"]) for p in preds
                                 if int(p["seq"]) == rider_seq), None)
            rider_eta_unix = rider_eta_ts.timestamp() if rider_eta_ts is not None else None
            fig = rt.build_eta_animation(route, P, qt.timestamp(), rider_route_seq,
                                         rider_eta_unix, rider["Arrêt"])
            st.plotly_chart(fig, width='stretch', key="eta_anim")
            st.caption("▶ Appuyez sur **Lecture** (× 0.5 à × 4 pour la vitesse) pour voir tout le trajet, "
                       "ou faites glisser le curseur.")

# ─────────────────────────────────────────────────────────────────────────────
# GPS Fallback — event-driven signal-loss demo
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "Repli GPS":
    st.title("📡 Repli en cas de perte GPS")
    st.caption("Quand un bus perd le signal, le système le détecte et le filtre de Kalman estime sa position en continu.")
    st.markdown("---")

    companies = get_companies()
    if not companies:
        st.warning("Aucune donnée disponible depuis l'API.")
        st.stop()

    ss = st.session_state
    c1, c2, c3 = st.columns([1, 1, 1.4])
    company = c1.selectbox("Opérateur", companies, key="gps_co")
    lines = get_lines(company)
    line = c2.selectbox("Ligne", lines, key="gps_line") if lines else None

    with c3:
        st.write("")
        st.write("")
        replay = st.button("▶ Rejouer une vraie perte de signal", type="primary",
                           width='stretch')

    manual = st.expander("…ou choisir un bus-jour précis")
    with manual:
        days = get_days_for_line(company, line) if line else []
        m1, m2, m3 = st.columns([1, 1, 1])
        m_day = m1.selectbox("Jour", days, key="gps_day") if days else None
        m_buses = get_buses_for_day(company, line, m_day) if m_day else []
        m_bus = m2.selectbox("Bus", m_buses, key="gps_bus_pick") if m_buses else None
        m3.write("")
        m3.write("")
        load_manual = m3.button("Charger le trajet", width='stretch') if m_bus else False

    target = None
    if replay and line:
        with st.spinner("Recherche d'une vraie perte de signal dans l'historique…"):
            ex = get_gap_examples(company, line)
        if ex and ex.get("examples"):
            e = ex["examples"][0]
            target = (e["bus"], e["day"], e["max_gap_min"])
        else:
            st.warning("Aucune perte de signal trouvée sur cette ligne.")
    elif load_manual and line and m_bus and m_day:
        target = (m_bus, m_day, None)

    if target:
        bus, day, gap_hint = target
        track = get_gps_track(company, line, bus, day)
        gaps = get_gps_gaps(company, line, bus, day)
        if not track or not track.get("track"):
            st.error("Aucun trajet GPS pour ce bus-jour.")
            st.stop()
        ss.gps_ctx = {"company": company, "line": line, "bus": bus, "day": day}
        ss.gps_track = track
        ss.gps_gaps = gaps
        ss.gps_P = rt.prep_track(track["track"])
        ss.gps_sim_off = 0.0
        ss.gps_playing = True
        ss.gps_prev_dark = False
        ss.gps_last_est = None
        if gap_hint:
            st.success(f"Cas réel chargé : **Bus {bus}** le **{fmt_day(day)}** — "
                       f"plus longue perte de signal **{gap_hint:.0f} min**.")

    if "gps_track" in ss:
        ctx = ss.gps_ctx
        track = ss.gps_track
        gaps = ss.gps_gaps
        P = ss.gps_P
        route = track["route"]

        st.markdown(f"#### Bus {ctx['bus']} · Ligne {ctx['line']} · {fmt_day(ctx['day'])}")
        left, right = st.columns([1.4, 1], gap="large")

        with right:
            # Preuve de précision : à quel point l'estimation Kalman colle pendant chaque coupure
            errs = []
            for i, p in enumerate(track["track"]):
                if p["signal_gap"] and i > 0:
                    est = rt.position_at(P, route, float(P["t_unix"][i]) - 1)
                    if est["dark"]:
                        errs.append((p["gap_s"] / 60, rt.haversine_m(
                            est["lat"], est["lon"], p["lat"], p["lon"])))
            st.markdown("##### Précision du repli IA")
            if errs:
                worst = max(errs, key=lambda e: e[0])
                med_err = float(pd.Series([e[1] for e in errs]).median())
                a1, a2 = st.columns(2)
                a1.metric("Coupures gérées", len(errs))
                a2.metric("Erreur médiane d'estimation", f"~{med_err:.0f} m")
                st.caption(f"Pire coupure : **{worst[0]:.0f} min** sans signal → estimation à "
                           f"**~{worst[1]:.0f} m** de l'endroit où le bus est réapparu.")
            else:
                st.caption("Aucune coupure au-dessus du seuil de détection dans ce trajet.")

            st.markdown("##### Pertes de signal")
            if gaps and gaps.get("gaps"):
                gdf = pd.DataFrame(gaps["gaps"])
                show = pd.DataFrame({
                    "De": pd.to_datetime(gdf["t_start"]).dt.strftime("%H:%M"),
                    "À": pd.to_datetime(gdf["t_end"]).dt.strftime("%H:%M"),
                    "Sans signal (min)": gdf["gap_min"].round(0),
                    "Parcouru (km)": gdf["dist_covered_km"],
                })
                st.dataframe(show, hide_index=True, width='stretch', height=200)
            else:
                st.caption("Aucune perte au-dessus du seuil de détection.")

        with left:
            fig = rt.build_gps_animation(route, P)
            st.plotly_chart(fig, width='stretch', key="gps_anim")
            st.caption("▶ Appuyez sur **Lecture** (× 0.5 à × 4 pour la vitesse). Quand le bus perd le signal, "
                       "l'itinéraire passe en **rouge** et le filtre de Kalman estime sa position "
                       "(gris = dernier point connu) jusqu'au retour du signal.")

        st.caption("ℹ️ La détection est une fonction pure (`detect_signal_loss`) sur le flux de pings — "
                   "ici pilotée par une horloge de rejeu, mais prête à se brancher sur des pings en direct.")
    else:
        st.info("Appuyez sur **▶ Rejouer une vraie perte de signal** pour aller directement à un cas réel, "
                "ou dépliez le panneau ci-dessus pour choisir un bus-jour précis.")

# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection — explainable + patterns
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "Détection d'anomalies":
    st.title(f":blue[:material/shield:] {t('page_title_anomaly')}")
    st.caption(t("page_subtitle_anomaly"))

    # Mode admin : seuls les bugs de données prouvés (>24h, horodatages corrompus) et les
    # fragments trop courts pour être jugés sont masqués par défaut. Les trajets longs
    # « gonflés » (service non clôturé, trous de signal) restent visibles avec leur
    # explication -- décision délibérée : sans explication l'admin croirait le modèle cassé.
    show_bugs = st.checkbox(t("show_data_bugs"), value=False, key="an_show_bugs")
    st.markdown("---")

    companies = get_companies()
    if not companies:
        st.warning("Aucune donnée disponible depuis l'API.")
        st.stop()

    tab_live, tab_explain, tab_patterns, tab_tickets = st.tabs([
        f":material/warning: {t('tab_live')}", f":material/search: {t('tab_explain')}",
        f":material/bar_chart: {t('tab_patterns')}", f":material/confirmation_number: {t('tab_tickets')}",
    ])

    # Icônes Material Symbols. `icon=` est un paramètre DÉDIÉ sur st.error/warning/info/
    # badge/expander/button -- il n'accepte QU'un seul emoji ou un shortcode `:material/x:`
    # brut ; lui passer du balisage `:color[...]` lève une StreamlitAPIException à l'exécution
    # (confirmé en testant en direct : "The value ':orange[...]' is not a valid emoji").
    # `ICON` reste donc plat, pour ces call sites. Quand une icône est plutôt insérée dans du
    # TEXTE affiché (st.caption/markdown, ex. les puces de `render_alert_cards`), CICON(key)
    # l'enveloppe dans la syntaxe couleur native de Streamlit -- sinon toutes les puces
    # ressortent dans le même gris terne du texte de légende, indiscernables au premier
    # coup d'oeil. Couleur choisie par SÉMANTIQUE (rouge = franchement anormal, orange = à
    # surveiller, bleu = informatif, violet = position/déviation, gris = note sur la qualité
    # des données plutôt qu'une anomalie).
    ICON = {
        "parking": ":material/local_parking:", "stopped": ":material/pan_tool:",
        "signal_lost": ":material/signal_disconnected:", "location": ":material/location_on:",
        "off_route": ":material/directions_off:", "suspect": ":material/help:",
        "detour": ":material/sync_alt:", "formula": ":material/calculate:",
        "duration": ":material/schedule:", "bug": ":material/bug_report:",
        "fragment": ":material/broken_image:", "partial": ":material/straighten:",
        "map": ":material/map:", "hide": ":material/visibility_off:", "info": ":material/info:",
    }
    ICON_COLOR = {
        "parking": "orange", "stopped": "blue", "signal_lost": "orange", "location": "violet",
        "off_route": "red", "suspect": "gray", "detour": "violet", "formula": "blue",
        "duration": "blue", "bug": "red", "fragment": "gray", "partial": "gray",
        "map": "blue", "hide": "gray", "info": "blue",
    }

    def CICON(key: str) -> str:
        """Icône ICON[key] teintée pour insertion dans du texte (voir commentaire ci-dessus).
        NE PAS passer le résultat à un paramètre `icon=` dédié -- seulement dans du texte."""
        return f":{ICON_COLOR[key]}[{ICON[key]}]"

    SEV_META = {
        "high":   {"label": t("sev_high"),   "icon": ":material/priority_high:", "color": "red"},
        "medium": {"label": t("sev_medium"), "icon": ":material/warning:",       "color": "orange"},
        "low":    {"label": t("sev_low"),    "icon": ":material/info:",          "color": "gray"},
    }
    # Explication au survol de chaque raison générée par le modèle (clé = feature déclencheuse,
    # voir reason_features renvoyé par anomaly.explain_trips) -- rappelle explicitement quand une
    # durée exclut DÉJÀ le stationnement terminus rogné, pour ne pas laisser croire à un artefact.
    REASON_HELP = {
        "max_dwell_s": t("exp_max_dwell_s"), "total_elapsed": t("exp_total_elapsed"),
        "mean_dwell_s": t("exp_mean_dwell_s"), "dist_m_max": t("exp_dist_m_max"),
        "match_rate": t("exp_match_rate"), "n_stops": t("exp_n_stops"),
        "max_dark_s": t("exp_max_dark_s"), "terminus_idle_min": t("exp_terminus_idle_min"),
        "elapsed_vs_bus_z": t("exp_elapsed_vs_bus_z"), "elapsed_vs_line_z": t("exp_elapsed_vs_line_z"),
    }

    # Catégorie dominante par trajet (`top_feature`, calculée par anomaly.explain_trips) --
    # même vocabulaire que les puces de raison, pour filtrer une longue liste de trajets
    # signalés par TYPE de problème plutôt que de les parcourir un par un. Options de
    # `st.multiselect` = texte brut (pas d'icône rendue) -- libellés sans emoji ni shortcode.
    TOP_FEATURE_FR = {
        "max_dwell_s": t("topfeat_max_dwell_s"),
        "total_elapsed": t("topfeat_total_elapsed"),
        "mean_dwell_s": t("topfeat_mean_dwell_s"),
        "dist_m_max": t("topfeat_dist_m_max"),
        "match_rate": t("topfeat_match_rate"),
        "n_stops": t("topfeat_n_stops"),
        "max_dark_s": t("topfeat_max_dark_s"),
        "terminus_idle_min": t("topfeat_terminus_idle_min"),
        "elapsed_vs_bus_z": t("topfeat_elapsed_vs_bus_z"),
        "elapsed_vs_line_z": t("topfeat_elapsed_vs_line_z"),
        None: t("other_uncategorized"),
    }
    SEV_RANK = {"high": 3, "medium": 2, "low": 1}

    def category_filter(anomalies, key):
        """Multiselect par type de problème dominant -- retourne la liste filtrée.
        N'affiche le widget que s'il y a au moins 2 catégories parmi lesquelles choisir."""
        if not anomalies:
            return anomalies
        present = sorted({a.get("top_feature") for a in anomalies},
                         key=lambda f: TOP_FEATURE_FR.get(f, str(f)))
        if len(present) < 2:
            return anomalies
        options = [TOP_FEATURE_FR.get(f, str(f)) for f in present]
        picked = st.multiselect(t("filter_anomaly_type"), options,
                               default=options, key=key)
        picked_features = {f for f in present if TOP_FEATURE_FR.get(f, str(f)) in picked}
        return [a for a in anomalies if a.get("top_feature") in picked_features]

    def _sort_anomalies(anomalies, sort_by="date_desc"):
        """Trie selon le critère choisi par l'admin (widget `sort_by`, voir tab_explain).

        Défaut `date_desc` (plus récent d'abord, gravité décroissante à date égale) --
        c'est ce qui s'est produit le plus récemment qui a le plus de chances de nécessiter
        une action, et à date égale, la gravité départage."""
        day = lambda a: a.get("day") or ""
        sev = lambda a: SEV_RANK.get(a.get("severity"), 0)
        if sort_by == "date_asc":
            return sorted(anomalies, key=lambda a: (day(a), sev(a)))
        if sort_by == "severity_desc":
            return sorted(anomalies, key=lambda a: (sev(a), day(a)), reverse=True)
        if sort_by == "severity_asc":
            return sorted(anomalies, key=lambda a: (sev(a), day(a)))
        if sort_by == "duration_desc":
            return sorted(anomalies, key=lambda a: a.get("trip_duration_min") or 0, reverse=True)
        return sorted(anomalies, key=lambda a: (day(a), sev(a)), reverse=True)  # date_desc

    def _fmt_duration(minutes):
        h, m = int(minutes) // 60, int(minutes) % 60
        return f"{h}h{m:02d}" if h else f"{m} min"

    def render_trip_map(seq_list, key=None, direction=None, detour=None):
        """Carte des arrêts d'un trajet — cercles colorés selon le problème détecté.

        `direction` (ALLER/RETOUR) oriente la carte : les arrêts sont numérotés dans
        l'ordre de passage réel du bus (RETOUR = seq décroissant), avec départ/terminus.
        """
        map_rows = [s for s in seq_list if s.get("lat") and s.get("lon")]
        if not map_rows:
            st.info(t("map_no_coords"))
            return
        def _dot(color):
            return (f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                    f'background:{color};margin:0 3px 0 8px;vertical-align:middle;"></span>')
        st.caption(
            t("map_legend")
            + _dot("#22c55e") + t("map_normal")
            + _dot("#2563eb") + t("map_long_stop")
            + _dot("#f59e0b") + t("map_signal_loss")
            + _dot("#ef4444") + t("map_unserved")
            + _dot("#9ca3af") + t("map_suspect"),
            unsafe_allow_html=True,
        )
        mdf = pd.DataFrame(map_rows)
        if "dark_min" not in mdf.columns:
            mdf["dark_min"] = 0.0
        if "coord_suspect" not in mdf.columns:
            mdf["coord_suspect"] = False

        # Ordre de passage réel : les seq suivent la géométrie ALLER, donc un trajet
        # RETOUR visite les arrêts en seq décroissant.
        mdf = mdf.sort_values("seq", ascending=(direction != "RETOUR")).reset_index(drop=True)
        mdf["visit_order"] = range(1, len(mdf) + 1)

        def _scolor(row):
            if row["coord_suspect"]: return "#9ca3af"
            if not row["matched"]: return "#ef4444"
            if row.get("dark_min", 0) >= 5: return "#f59e0b"
            if row.get("dwell_min", 0) >= 10: return "#2563eb"
            return "#22c55e"

        mdf["color"] = mdf.apply(_scolor, axis=1)
        mdf["msize"] = (mdf["dwell_min"] + mdf["dark_min"]).clip(8, 28)
        # heure de passage réelle à chaque arrêt suivi (None si non suivi)
        if "arrival" not in mdf.columns:
            mdf["arrival"] = None
        mdf["heure"] = mdf["arrival"].apply(
            lambda v: pd.Timestamp(v).strftime("%H:%M") if v else "—")
        mdf["hover"] = (
            "<b>" + mdf["visit_order"].astype(str) + " : " + mdf["stop"] + "</b><br>" +
            mdf["heure"].apply(lambda v: t("map_hover_time", v=v)) + "<br>" +
            mdf["dwell_min"].apply(lambda v: t("map_hover_dwell", v=f"{v:.1f} min")) + "<br>" +
            mdf["dark_min"].apply(lambda v: t("map_hover_dark", v=f"{v:.1f} min")) + "<br>" +
            mdf["dist_m"].apply(lambda v: t("map_hover_dist", v=f"{v:.0f} m")) + "<br>" +
            mdf["matched"].map({True: t("map_hover_tracked", v=t("map_hover_yes")),
                                False: t("map_hover_tracked", v=t("map_hover_no"))}) +
            mdf["coord_suspect"].map({True: t("map_hover_suspect"), False: ""})
        )
        # départ réel (premier arrêt suivi) affiché sous la carte
        first_tracked = mdf[mdf["arrival"].notna()]
        if len(first_tracked):
            t0 = pd.Timestamp(first_tracked["arrival"].iloc[0])
            t1 = pd.Timestamp(first_tracked["arrival"].iloc[-1])
            st.caption(f"{CICON('location')} " + t("map_first_last_tracked",
                       t0=t0.strftime('%H:%M'), stop0=first_tracked['stop'].iloc[0],
                       t1=t1.strftime('%H:%M'), stop1=first_tracked['stop'].iloc[-1]))

        if detour and detour.get("track"):
            left = pd.Timestamp(detour["left_at"]).strftime("%H:%M")
            far = pd.Timestamp(detour["farthest_at"]).strftime("%H:%M")
            back = pd.Timestamp(detour["returned_at"]).strftime("%H:%M")
            out_min = detour.get("leg_out_duration_min")
            back_min = detour.get("leg_back_duration_min")
            legs = (t("detour_legs", left=left, far=far, out=out_min, back=back, back_min=back_min)
                   if out_min is not None and back_min is not None else "")
            st.warning(
                t("detour_warning", left=left, km=detour["distance_km"], far=far, back=back,
                  total=detour["duration_min"], legs=legs),
                icon=ICON["detour"],
            )

        fig_map = go.Figure()
        if detour and detour.get("track"):
            # Aller (départ -> point le plus éloigné) et retour (point le plus éloigné ->
            # retour au point de départ) tracés séparément -- un seul tracé superposerait les
            # deux passages et rendrait le sens du détour illisible.
            leg_out = detour.get("leg_out") or detour["track"]
            leg_back = detour.get("leg_back") or []
            for leg, color, label in [(leg_out, "#f97316", t("detour_leg_out_label")),
                                       (leg_back, "#a855f7", t("detour_leg_back_label"))]:
                if not leg:
                    continue
                dtr = pd.DataFrame(leg)
                fig_map.add_trace(go.Scattermapbox(
                    lat=dtr["lat"].tolist(), lon=dtr["lon"].tolist(),
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                    marker=dict(size=5, color=color),
                    name=label,
                    hovertext=dtr["t"].apply(lambda v: pd.Timestamp(v).strftime("%H:%M:%S")).tolist(),
                    hovertemplate=f"{label} · " + "%{hovertext}<extra></extra>",
                ))
        fig_map.add_trace(go.Scattermapbox(
            lat=mdf["lat"].tolist(), lon=mdf["lon"].tolist(),
            mode="lines", line=dict(width=3, color="#94a3b8"),
            name=t("map_planned_route"), hoverinfo="skip",
        ))
        for color, label in [
            ("#22c55e", t("legend_normal_stop")),
            ("#2563eb", t("legend_long_standstill")),
            ("#f59e0b", t("legend_signal_loss")),
            ("#ef4444", t("legend_unserved")),
            ("#9ca3af", t("legend_suspect_coords")),
        ]:
            sub = mdf[mdf["color"] == color]
            if sub.empty:
                continue
            fig_map.add_trace(go.Scattermapbox(
                lat=sub["lat"].tolist(), lon=sub["lon"].tolist(),
                mode="markers",
                marker=dict(size=sub["msize"].tolist(), color=color),
                name=label,
                text=sub["hover"].tolist(),
                hovertemplate="%{text}<extra></extra>",
            ))
        # Numéros d'ordre de passage superposés aux cercles → sens de circulation lisible
        fig_map.add_trace(go.Scattermapbox(
            lat=mdf["lat"].tolist(), lon=mdf["lon"].tolist(),
            mode="text", text=mdf["visit_order"].astype(str).tolist(),
            textfont=dict(size=10, color="#111827"),
            hoverinfo="skip", showlegend=False,
        ))
        # Départ / terminus du trajet dans le sens de circulation -- anneaux distincts +
        # étiquette texte plutôt que des pictogrammes emoji (rendu Plotly/Mapbox direct,
        # pas de police d'icônes disponible à ce niveau).
        dir_label = f" ({direction})" if direction else ""
        fig_map.add_trace(go.Scattermapbox(
            lat=[mdf["lat"].iloc[0]], lon=[mdf["lon"].iloc[0]],
            mode="markers+text", marker=dict(size=18, color="#16a34a"),
            text=[t("map_departure")], textposition="top center", textfont=dict(size=11, color="#111827"),
            name=f"{t('map_departure')}{dir_label}", hoverinfo="skip",
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=[mdf["lat"].iloc[-1]], lon=[mdf["lon"].iloc[-1]],
            mode="markers+text", marker=dict(size=18, color="#dc2626"),
            text=[t("map_terminus")], textposition="top center", textfont=dict(size=11, color="#111827"),
            name=t("map_terminus"), hoverinfo="skip",
        ))
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=float(mdf["lat"].mean()), lon=float(mdf["lon"].mean())),
                zoom=10,
            ),
            margin=dict(l=0, r=0, t=0, b=0), height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, width='stretch', key=key)

    def render_ticket_station_map(stations, key=None):
        """Carte par ARRÊT D'ORIGINE pour un jour de billetterie anormal (Phase 2, voir
        /api/ticket-anomaly-stations) -- vert = normal, bleu = anomalie "bonne" (recette
        au-dessus de la normale de cet arrêt), rouge = anomalie "mauvaise" (en dessous) --
        voir `is_good_anomaly` côté API. Contrairement au trajet GPS il n'y a pas de route/
        ordre de passage ici, juste des points -- certains arrêts peuvent manquer de
        lat/lon (résolution par nom best-effort côté API) et sont listés à part sous la
        carte plutôt qu'invisibles sans explication.
        """
        with_coords = [s for s in stations if s.get("lat") and s.get("lon")]
        without_coords = [s for s in stations if not (s.get("lat") and s.get("lon"))]
        if not with_coords:
            st.info("Coordonnées non disponibles pour les arrêts de cette ligne.")
        else:
            sdf = pd.DataFrame(with_coords)
            def _color(row):
                if not row["anomaly"]:
                    return "#22c55e"
                return "#3b82f6" if row.get("is_good_anomaly") else "#ef4444"
            sdf["color"] = sdf.apply(_color, axis=1)
            sdf["hover"] = (
                "<b>" + sdf["station"] + "</b><br>"
                + sdf["nbr_ticket"].astype(str) + " tickets · "
                + sdf["recette"].round(0).astype(str) + " DT · "
                + sdf["avg_fare"].round(2).astype(str) + " DT/ticket"
            )
            fig = go.Figure()
            for color, label in [("#22c55e", "Normal"), ("#3b82f6", "Bonne anomalie (recette ↑)"),
                                 ("#ef4444", "Anomalie à surveiller (recette ↓)")]:
                sub = sdf[sdf["color"] == color]
                if sub.empty:
                    continue
                fig.add_trace(go.Scattermapbox(
                    lat=sub["lat"].tolist(), lon=sub["lon"].tolist(), mode="markers",
                    marker=dict(size=14, color=color), name=label,
                    text=sub["hover"].tolist(), hovertemplate="%{text}<extra></extra>",
                ))
            fig.update_layout(
                mapbox=dict(style="open-street-map",
                           center=dict(lat=float(sdf["lat"].mean()), lon=float(sdf["lon"].mean())),
                           zoom=8),
                margin=dict(l=0, r=0, t=0, b=0), height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, width='stretch', key=key)
        if without_coords:
            names = ", ".join(s["station"] for s in without_coords[:10])
            suffix = f" (+{len(without_coords)-10} autres)" if len(without_coords) > 10 else ""
            st.caption(f"Sans coordonnées résolues : {names}{suffix}")

    def render_alert_cards(anomalies, detail_company=None, sort_by="date_desc", key_prefix=""):
        """detail_company : si renseigné, chaque carte reçoit un bouton pour charger à la
        demande la carte des arrêts du trajet concerné.

        Design : les raisons et les puces (arrêts concernés) restent visibles directement sur
        la carte -- c'est le contenu que l'admin scanne réellement -- mais chacune porte une
        explication au survol de l'icône (i) plutôt qu'un paragraphe complet imprimé en dur,
        pour rester lisible sur une longue liste. Les 4 classes qualité (bug/fragment/durée
        gonflée/couverture partielle) gardent leur paragraphe complet car elles changent le
        jugement global du trajet, pas juste un détail.

        `sort_by` : critère choisi par l'admin, voir `_sort_anomalies` -- défaut inchangé
        (date d'occurrence, plus récent d'abord, puis gravité à date égale).

        `key_prefix` : distingue les clés de widget (bouton carte) entre les DIFFÉRENTS
        appels de cette fonction sur la même page (ex. "Trajets ce jour" et "Historique
        récent" dans tab_live peuvent tous deux lister le même trajet) -- sans ça, deux
        cartes du même trajet à des appels différents produisent la même clé et Streamlit
        lève une DuplicateWidgetID."""
        # Avertissement modèle à faible historique -- une seule fois pour la liste, pas par
        # carte (un appel donné de render_alert_cards correspond toujours à UN seul opérateur,
        # puisque tab_live/tab_explain filtrent déjà par société avant d'appeler cette
        # fonction ; répéter le même avertissement sur chaque carte serait du bruit). Isolation
        # Forest ET l'autoencodeur LSTM sont entraînés PAR OPÉRATEUR (voir models/anomaly.py) --
        # même "dédié", un modèle reste entraîné sur TOUTES les lignes de l'opérateur poolées
        # ensemble, jamais sur une ligne seule. En dessous du seuil de trajets requis
        # (30 pour l'IF, 200 pour le LSTM), repli sur un modèle GLOBAL entraîné sur TOUTES les
        # sociétés confondues -- confirmé en base (2026-07) : TUS a un IF dédié mais PAS de LSTM
        # dédié (repli global), S.R.T.SELIANA n'a ni l'un ni l'autre.
        if anomalies:
            a0 = anomalies[0]
            if a0.get("model_low_data"):
                if_ded, lstm_ded = a0.get("model_if_dedicated"), a0.get("model_lstm_dedicated")
                key = ("model_warning_neither" if not if_ded and not lstm_ded
                      else "model_warning_lstm_only")
                st.warning(t(key), icon="⚠️")
        for i, a in enumerate(_sort_anomalies(anomalies, sort_by)):
            sev = SEV_META.get(a["severity"], SEV_META["medium"])
            with st.container(border=True):
                dur = a.get("trip_duration_min") or a.get("total_elapsed_min", 0)
                # « Activité GPS vérifiable » = total − immobilisations − trous de signal.
                # Affichée SEULEMENT quand elle est interprétable : durée GONFLÉE (au-dessus
                # de la médiane de la ligne), blocs significatifs, ET trajet correctement
                # suivi (>= 50% des arrêts) -- sur un trajet à peine suivi (ex. 29% des
                # arrêts), « 53 min » est vrai mais se lit comme une absurdité (245 km en
                # 53 min) ; le mauvais suivi est déjà signalé par ses propres raisons.
                _dwell = a.get("max_dwell_min", 0) or 0
                _dark = a.get("max_dark_min", 0) or 0
                _est = a.get("driving_time_est_min")
                _med = a.get("line_median_elapsed_min")
                _match = a.get("match_rate") or 0
                _show_est = bool(_est is not None and (_dwell + _dark) >= 15
                                 and _med and dur > _med and _match >= 0.5)
                _dep = pd.Timestamp(a["trip_start"]).strftime("%H:%M") if a.get("trip_start") else "—"
                _arr = pd.Timestamp(a["trip_end"]).strftime("%H:%M") if a.get("trip_end") else "—"
                _idle = a.get("terminus_idle_min") or 0
                # Détail par terminus (nom + horodatage réel), voir foundation.segment_trips --
                # absent sur les anciens trajets pas encore reconstruits avec le champ ; dans ce
                # cas on retombe sur le chip générique `_idle` ci-dessous (sans nom ni horaire).
                _origin_idle = a.get("origin_idle_min") or 0
                _end_idle = a.get("end_idle_min") or 0
                _origin_stop = a.get("origin_idle_stop")
                _end_stop = a.get("end_idle_stop")
                _origin_from = (pd.Timestamp(a["origin_idle_from"]).strftime("%H:%M")
                                if a.get("origin_idle_from") else "—")
                _end_to = (pd.Timestamp(a["end_idle_to"]).strftime("%H:%M")
                          if a.get("end_idle_to") else "—")
                _has_named_idle = bool(_origin_stop or _end_stop)

                hdr = st.columns([0.16, 0.84], vertical_alignment="center")
                hdr[0].badge(sev["label"], icon=sev["icon"], color=sev["color"])
                hdr[1].markdown(f"**Bus {a['bus']} · Ligne {a['line']} · {a['dir']}** — {fmt_day(a['day'])}")

                top = st.columns([1, 1, 1] if _show_est else [1, 1])
                # La durée exclut DÉJÀ le stationnement terminus rogné -- rappelé explicitement
                # au survol pour qu'on ne pense jamais que le modèle "invente" une durée plus
                # courte que ce que montrent les pings bruts.
                top[0].metric(t("metric_trip_duration"), _fmt_duration(dur), help=t("formula_help"))
                top[1].metric(t("metric_departure_arrival"), f"{_dep} → {_arr}")
                if _show_est:
                    top[2].metric(t("metric_verifiable_activity"), f"≈ {_fmt_duration(_est)}",
                                 help=t("verifiable_activity_caption", est=_fmt_duration(_est),
                                        dur=_fmt_duration(dur), dwell=_dwell, dark=_dark, match=_match*100))

                top_label = TOP_FEATURE_FR.get(a.get("top_feature"), t("flagged_no_reason"))
                st.caption(t("main_cause", label=top_label))

                # Classes qualité (paragraphe complet -- change le jugement global du trajet).
                # Bug/fragment/couverture partielle : visibles seulement en mode admin (déjà
                # filtrés en amont sinon). Durées gonflées : visibles par défaut AVEC
                # explication — sans elle, l'admin conclut à tort que le modèle est incorrect.
                if a.get("is_data_bug"):
                    st.error(t("q_data_bug"), icon=ICON["bug"])
                elif a.get("is_fragment"):
                    st.warning(t("q_fragment"), icon=ICON["fragment"])
                elif a.get("is_dark_inflated"):
                    st.info(t("q_dark_inflated"), icon=ICON["signal_lost"])
                elif a.get("is_implausible"):
                    st.info(t("q_implausible"), icon=ICON["duration"])
                elif a.get("is_partial_coverage"):
                    st.info(t("q_partial_coverage", ns=a.get("n_stops"),
                             mns=a.get("line_median_n_stops")), icon=ICON["partial"])

                # Raisons du modèle -- toujours visibles (pas de clic requis), chacune avec une
                # explication au survol basée sur la feature qui l'a déclenchée. Rappelle
                # explicitement, pour les raisons liées à la durée, que le stationnement
                # terminus est déjà retiré -- ce n'est pas un artefact de calcul.
                reasons = a.get("reasons") or []
                reason_feats = a.get("reason_features") or []
                if reasons:
                    for j, reason in enumerate(reasons):
                        feat = reason_feats[j] if j < len(reason_feats) else None
                        help_txt = REASON_HELP.get(feat, t("exp_default"))
                        st.caption(f"• {reason}", help=help_txt)
                else:
                    st.caption(t("flagged_no_reason"))

                # Puces (arrêts/segments concernés) -- toujours visibles, chacune avec une
                # explication au survol de son propre (i).
                ps = a.get("problem_stops") or {}
                # Stationnement terminus rogné de la durée mais conservé comme signal -- c'est
                # l'anomalie « bus resté trop longtemps au terminus » que l'admin veut voir, et
                # on redit explicitement ici qu'il est DÉJÀ RETIRÉ de la durée ci-dessus. Nommé
                # (arrêt + horodatage réel) quand disponible -- répond à « stationné OÙ, et à
                # quelle heure le bus a-t-il vraiment démarré ? » au lieu d'un chiffre nu.
                if _has_named_idle:
                    if _origin_idle >= 30 and _origin_stop:
                        st.caption(t("chip_origin_idle", icon=CICON("parking"), stop=_origin_stop,
                                    min=_origin_idle, from_t=_origin_from, to_t=_dep),
                                  help=t("chip_origin_idle_help"))
                    if _end_idle >= 30 and _end_stop:
                        st.caption(t("chip_end_idle", icon=CICON("parking"), stop=_end_stop,
                                    min=_end_idle, from_t=_arr, to_t=_end_to),
                                  help=t("chip_end_idle_help"))
                elif _idle >= 30:
                    st.caption(t("chip_terminus_idle", icon=CICON("parking"), min=_idle),
                              help=t("chip_terminus_idle_help"))
                # Genuine dwell anomaly at a named stop
                if ps.get("longest_stop") and ps["longest_stop"]["dwell_min"] >= 5:
                    st.caption(t("chip_real_stop", icon=CICON("stopped"), stop=ps["longest_stop"]["stop"],
                                min=ps["longest_stop"]["dwell_min"]), help=t("chip_real_stop_help"))
                    _dwell_stop = ps["longest_stop"]["stop"]
                    _same_idle = (_origin_idle if _dwell_stop == _origin_stop else
                                 _end_idle if _dwell_stop == _end_stop else 0)
                    if _has_named_idle and _same_idle >= 30:
                        # On CONNAÎT le nom des deux termini -- si l'arrêt de cette immobilisation
                        # est LE MÊME que le terminus stationné ci-dessus (chip affichée, donc
                        # >=30 min), ce n'est presque certainement pas une coïncidence : un bref
                        # sursaut GPS isolé (voir `bridge_geometry_outliers`/idle-trim dans
                        # foundation.segment_trips) a pu couper une seule longue immobilisation en
                        # deux morceaux comptés à part.
                        st.caption(t("chip_same_terminus_hint", stop=_dwell_stop,
                                    total=_same_idle + ps["longest_stop"]["dwell_min"]))
                    elif _idle >= 30 and not _has_named_idle:
                        # Anciens trajets sans nom de terminus -- ambiguïté générique inchangée.
                        st.caption(t("chip_detour_hint"))
                # Signal loss at a named stop (separate from dwell)
                if ps.get("signal_loss_stop"):
                    sl = ps["signal_loss_stop"]
                    st.caption(t("chip_signal_loss", icon=CICON("signal_lost"), stop=sl["stop"], min=sl["dark_min"]),
                              help=t("chip_signal_loss_help"))
                # Trou de signal EN ROUTE (entre deux arrêts, jamais rattaché à l'attente d'un
                # arrêt matché -- invisible au scan par-arrêt ci-dessus, voir
                # foundation.derive_arrivals). Distinct de signal_loss_stop : celui-ci peut
                # expliquer À LUI SEUL un mauvais taux de suivi + une durée gonflée.
                _gap_before = a.get("dark_gap_before_stop")
                _gap_after = a.get("dark_gap_after_stop")
                _dark_min = a.get("max_dark_min") or 0
                if _gap_before and _dark_min >= 15:
                    key = "chip_dark_gap_between" if _gap_after else "chip_dark_gap_after_only"
                    st.caption(t(key, icon=CICON("signal_lost"), before=_gap_before,
                                after=_gap_after, min=_dark_min), help=t("chip_dark_gap_help"))
                # Matched stops that were still far off expected position (GPS drift / detour)
                if ps.get("farthest_stop"):
                    st.caption(t("chip_farthest", icon=CICON("location"), stop=ps["farthest_stop"]["stop"],
                                dist=ps["farthest_stop"]["dist_m"]), help=t("chip_farthest_help"))
                # Unmatched stops — bus never passed within range
                if ps.get("off_route_stops"):
                    others = ps.get('off_route_count', len(ps['off_route_stops'])) - len(ps['off_route_stops'])
                    suffix = t("and_others", n=others) if others > 0 else ""
                    st.caption(t("chip_off_route", icon=CICON("off_route"),
                                stops=', '.join(ps['off_route_stops']), suffix=suffix),
                              help=t("chip_off_route_help"))
                # Stops never matched on ANY trip of the line = bad geocoding, not an anomaly
                if ps.get("suspect_coord_count"):
                    st.caption(t("chip_suspect_coord", icon=CICON("suspect"), count=ps["suspect_coord_count"]),
                              help=t("chip_suspect_coord_help"))

                # Bouton carte : charge la carte du trajet à la demande.
                if detail_company and a.get("trip_id") is not None:
                    k = f"an_card_{key_prefix}_{i}_{a['bus']}_{a['day']}_{a['trip_id']}"
                    shown = st.session_state.get(k, False)
                    if st.button(t("btn_hide_map") if shown else t("btn_show_map"),
                                 icon=ICON["hide"] if shown else ICON["map"], key=k + "_btn"):
                        st.session_state[k] = not shown
                        st.rerun()
                    if shown:
                        detail = get_trip_detail(detail_company, a["line"], a["bus"],
                                                 a["day"], a["trip_start"])
                        render_trip_map((detail or {}).get("sequence", []),
                                        key=k + "_map", direction=a.get("dir"),
                                        detour=(detail or {}).get("problem_stops", {}).get("unofficial_detour"))

    with tab_live:
        col = st.columns([1, 1, 1])
        company = col[0].selectbox(t("filter_operator"), companies, key="an_live_co")
        lines = [t("filter_all_lines")] + get_lines(company)
        line_sel = col[1].selectbox(t("filter_line"), lines, key="an_live_line")
        line_param = None if line_sel == t("filter_all_lines") else line_sel
        dir_sel = col[2].selectbox(t("filter_direction"),
                                   [t("filter_both_directions"), "ALLER", "RETOUR"], key="an_live_dir")
        dir_param = None if dir_sel == t("filter_both_directions") else dir_sel

        data = get_current_anomalies(company, line_param, direction=dir_param)
        if data:
            m = st.columns(3)
            m[0].metric(t("metric_operating_day"), fmt_day(data["date"]))
            m[1].metric(t("metric_trips_today"), data["total_trips"])
            pct = 100 * data["anomaly_count"] / data["total_trips"] if data["total_trips"] else 0
            m[2].metric(t("metric_flagged"), f"{data['anomaly_count']}  ({pct:.1f} %)")
            if data["anomalies"]:
                st.markdown(f"#### {t('section_flagged_today')}")
                filtered = category_filter(data["anomalies"], key="an_live_cat")
                render_alert_cards(filtered, detail_company=company, key_prefix="live_today")
            else:
                st.success(t("no_anomaly_today"))
                st.caption(t("try_specific_line"))

        st.markdown("---")
        st.markdown(f"#### {t('section_recent_history')}")
        hist = get_anomaly_history(company, line_param, limit=40, include_bugs=show_bugs, direction=dir_param)
        if hist and hist.get("anomalies"):
            filtered_hist = category_filter(hist["anomalies"][:12], key="an_hist_cat")
            render_alert_cards(filtered_hist, detail_company=company, key_prefix="live_hist")
        else:
            st.info(t("no_history"))

    with tab_explain:
        # ── Filtres ──────────────────────────────────────────────────────────
        col = st.columns(6)
        company = col[0].selectbox(t("filter_operator"), companies, key="an_ex_co")
        lines = get_lines(company)
        line = col[1].selectbox(t("filter_line"), lines, key="an_ex_line") if lines else None

        # ── Verdict global de la ligne — visible dès qu'une ligne est choisie ──
        if line:
            pat = get_anomaly_patterns(company, line)
            if pat and pat["total_trips"] >= 5:
                rate = pat["overall_rate"]
                n, k = pat["total_trips"], pat["total_anomalies"]
                # Repère : l'Isolation Forest est calibrée pour ~5% d'anomalies par
                # construction (contamination=0.05) — c'est le taux "normal" attendu.
                if rate <= 0.07:
                    v_icon, verdict, delta_color, badge_color = (":material/check_circle:",
                        t("line_good"), "normal", "green")
                elif rate <= 0.15:
                    v_icon, verdict, delta_color, badge_color = (":material/warning:",
                        t("line_watch"), "off", "orange")
                else:
                    v_icon, verdict, delta_color, badge_color = (":material/error:",
                        t("line_risk"), "inverse", "red")
                with st.container(border=True):
                    vc = st.columns([2, 1, 1])
                    with vc[0]:
                        st.badge(verdict, icon=v_icon, color=badge_color)
                        st.caption(t("line_verdict_caption", line=line, company=company, n=n))
                    vc[1].metric(t("metric_anomaly_rate"), f"{rate*100:.1f} %",
                                delta=f"{(rate-0.05)*100:+.1f} pts",
                                delta_color=delta_color)
                    vc[2].metric(t("metric_flagged_trips"), f"{k} / {n}")
            elif pat:
                st.info(t("line_not_enough_data", line=line, n=pat['total_trips']))

            # ── Trajet de référence : « voici un trajet NORMAL sur cette ligne » ──
            # Ancre de confiance pour l'admin : le modèle sait ce qu'est un trajet normal ;
            # les anomalies ci-dessous sont des écarts à CETTE norme, pas du bruit. Affiché
            # PAR DIRECTION (ALLER et RETOUR peuvent différer en durée/stationnement typique).
            with st.expander(t("ref_trip_expander", line=line), icon=":material/verified:"):
                ref = get_reference_trip(company, line)
                dirs = (ref or {}).get("directions") or {}
                if not dirs:
                    st.info(t("ref_trip_none"))
                else:
                    def _render_ref_direction(d, entry):
                        rt = entry["trip"]
                        st.caption(t("ref_trip_caption", match=rt['match_rate']*100))
                        # Bus et Jour sur des métriques séparées : combinées dans une seule
                        # ("6024 · 5 nov. 2024"), la date se faisait tronquer dans la colonne
                        # étroite -- chacune a maintenant assez de place pour s'afficher entière.
                        rm = st.columns([0.8, 1.2, 1, 0.9, 1.1, 1.1])
                        rm[0].metric(t("ref_trip_bus"), str(rt["bus"]))
                        rm[1].metric(t("filter_day"), fmt_day(rt["day"]))
                        rm[2].metric(t("ref_trip_duration"), _fmt_duration(rt["duration_min"]),
                                     delta=t("ref_trip_line_median", med=_fmt_duration(rt['line_median_min'])),
                                     delta_color="off")
                        rm[3].metric(t("ref_trip_stops_tracked"), rt["n_stops"])
                        rm[4].metric(t("ref_trip_avg_dwell"), f"{rt['mean_dwell_min']:.1f} min/arrêt")
                        dep = pd.Timestamp(rt["trip_start"]).strftime("%H:%M") if rt.get("trip_start") else "—"
                        arr = pd.Timestamp(rt["trip_end"]).strftime("%H:%M") if rt.get("trip_end") else "—"
                        rm[5].metric(t("metric_departure_arrival"), f"{dep} → {arr}")

                        # Stationnement terminus TYPIQUE de cette direction (médiane sur les
                        # trajets jugés normaux) -- donne à l'admin un repère concret pour
                        # juger un stationnement observé ailleurs, au lieu de deviner.
                        typ = rt.get("typical_terminus_idle_min")
                        if typ is not None:
                            thr = rt.get("service_not_closed_threshold_min")
                            st.caption(t("ref_typical_idle", icon=CICON("parking"), typ=typ, thr=thr))
                        render_trip_map(entry["sequence"], key=f"ref_map_{company}_{line}_{d}",
                                        direction=d)

                    if len(dirs) == 2:
                        # Si ALLER et RETOUR viennent du même bus/jour, c'est un vrai cycle
                        # (arrivée -> pause -> redépart) : le dire explicitement, sinon rien ne
                        # distingue ce cas visuellement d'un pairage par repli (jours différents).
                        aller_t, retour_t = dirs["ALLER"]["trip"], dirs["RETOUR"]["trip"]
                        if aller_t["bus"] == retour_t["bus"] and aller_t["day"] == retour_t["day"]:
                            st.caption(t("ref_trip_same_cycle", bus=aller_t["bus"], day=fmt_day(aller_t["day"])))
                        dtabs = st.tabs(list(dirs.keys()))
                        for dtab, (d, entry) in zip(dtabs, dirs.items()):
                            with dtab:
                                _render_ref_direction(d, entry)
                    else:
                        d, entry = next(iter(dirs.items()))
                        st.caption(t("ref_trip_missing_other_dir", dir=d))
                        _render_ref_direction(d, entry)

        # Bus is optional — "Tous les bus" analyses the whole line
        bus_opts = [t("filter_all_buses")] + [str(b) for b in (get_buses_for_line(company, line) if line else [])]
        bus_label = col[2].selectbox(t("filter_bus"), bus_opts, key="an_ex_bus")
        bus = None if bus_label == t("filter_all_buses") else int(bus_label)

        days = [t("filter_all_days")] + (get_days_for_line(company, line) if line else [])
        day_sel = col[3].selectbox(t("filter_day"), days, key="an_ex_day")
        day_param = None if day_sel == t("filter_all_days") else day_sel

        dir_sel = col[4].selectbox(t("filter_direction"),
                                   [t("filter_both_directions"), "ALLER", "RETOUR"], key="an_ex_dir")
        dir_param = None if dir_sel == t("filter_both_directions") else dir_sel

        if line and col[5].button(t("btn_analyze"), type="primary", width='stretch'):
            res = get_anomaly_explain(company, line, bus, day_param, include_bugs=show_bugs, direction=dir_param)
            st.session_state["an_ex_res"] = res
            st.session_state["an_ex_ctx"] = (company, line, bus, day_param, dir_param)

        res = st.session_state.get("an_ex_res")
        ctx = st.session_state.get("an_ex_ctx")

        if res and ctx == (company, line, bus, day_param, dir_param):
            scope = t("scope_bus_line", bus=bus, line=line) if bus else t("scope_line", line=line)
            if not res:
                st.error(t("btn_analyze_fail"))
            elif res["anomaly_count"] == 0:
                st.success(t("no_anomaly_found", scope=scope))
            else:
                st.markdown(f"#### {t('analysis_header', scope=scope)}")

                # ── Métriques générales (avec explications) ──────────────────
                mc = st.columns(3)
                ex_pct = 100 * res["anomaly_count"] / res["total_trips"] if res["total_trips"] else 0
                mc[0].metric(t("metric_trips_analyzed"), f"{res['total_trips']}")
                mc[0].caption(t("metric_trips_analyzed_help"))
                mc[1].metric(t("metric_abnormal_trips"), f"{res['anomaly_count']}  ({ex_pct:.1f} %)")
                mc[1].caption(t("metric_abnormal_trips_help"))
                if res.get("avg_duration_min"):
                    mc[2].metric(t("metric_normal_duration"), _fmt_duration(res["avg_duration_min"]))
                    mc[2].caption(t("metric_normal_duration_help"))

                st.markdown("---")
                # ── Cartes d'alerte ──────────────────────────────────────────
                st.markdown(f"##### {t('section_flagged_trips')}")
                filtered_ex = category_filter(res["anomalies"], key="an_ex_cat")
                # Tri choisi par l'admin -- appliqué à `filtered_ex` lui-même (pas seulement
                # à l'affichage des cartes) pour que le sélecteur "Trajet à analyser" plus bas
                # utilise le MÊME ordre que ce qui est montré ici ; sinon la carte #1 affichée
                # ne correspondrait pas à trip_labels[0] dans le sélecteur.
                SORT_OPTIONS = {
                    "date_desc": t("sort_date_desc"), "date_asc": t("sort_date_asc"),
                    "severity_desc": t("sort_severity_desc"), "severity_asc": t("sort_severity_asc"),
                    "duration_desc": t("sort_duration_desc"),
                }
                sort_key = st.selectbox(t("sort_by"), list(SORT_OPTIONS.keys()),
                                        format_func=lambda k: SORT_OPTIONS[k], key="an_ex_sort")
                filtered_ex = _sort_anomalies(filtered_ex, sort_key)
                render_alert_cards(filtered_ex, detail_company=company, sort_by=sort_key, key_prefix="explain")

                # ── Détail par trajet ─────────────────────────────────────────
                st.markdown("---")
                st.markdown(f"##### {t('section_trip_detail')}")

                if not filtered_ex:
                    st.info(t("no_trip_matches_filter"))
                    sel, seq_list = None, []
                else:
                    st.caption(t("select_trip_prompt"))

                    trip_labels = [
                        f"Bus {a['bus']} · {fmt_day(a['day'])} · {a['dir']} · score {a['anomaly_strength']:.2f}"
                        for a in filtered_ex
                    ]
                    sel_idx = st.selectbox(
                        t("trip_to_analyze"), range(len(trip_labels)),
                        format_func=lambda i: trip_labels[i],
                        key="an_ex_trip_sel",
                    )
                    sel = filtered_ex[sel_idx]

                    # Fetch per-trip sequence on demand (cached)
                    detail = get_trip_detail(company, line, sel["bus"], sel["day"], sel["trip_start"])
                    seq_list = (detail or {}).get("sequence", [])

                if seq_list:
                    seq = pd.DataFrame(seq_list)
                    if "dark_min" not in seq.columns:
                        seq["dark_min"] = 0.0

                    # ── Métriques du trajet sélectionné ──────────────────────
                    tc = st.columns(4)
                    tc[0].metric(t("metric_trip_duration"), _fmt_duration(sel["trip_duration_min"]),
                                help=t("formula_help"))
                    if res.get("avg_duration_min"):
                        delta_min = sel["trip_duration_min"] - res["avg_duration_min"]
                        sign = "+" if delta_min > 0 else ""
                        tc[1].metric(t("metric_delta_vs_normal"), f"{sign}{_fmt_duration(abs(delta_min))}",
                                    help=t("metric_delta_vs_normal_help"))
                    tc[2].metric(t("metric_stops_served"), f"{int(seq['matched'].sum())} / {len(seq)}",
                                help=t("metric_stops_served_help"))
                    tc[3].metric(t("metric_anomaly_score"), f"{sel['anomaly_strength']:.2f}",
                                help=t("metric_anomaly_score_help"))

                    # ── Carte des arrêts ──────────────────────────────────────
                    st.markdown(f"###### {t('section_trip_map')}")
                    render_trip_map(seq_list, key="an_ex_detail_map", direction=sel.get("dir"),
                                    detour=(detail or {}).get("problem_stops", {}).get("unofficial_detour"))

                    # ── Graphique immobilisation par arrêt ────────────────────
                    st.markdown(f"###### {t('section_dwell_chart')}")
                    st.caption(t("dwell_chart_caption"))
                    seq["label"] = seq["seq"].astype(str) + " · " + seq["stop"]
                    bar_colors = ["#2563eb" if m else "#dc2626" for m in seq["matched"]]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=seq["label"], y=seq["dwell_min"], name=t("series_real_standstill"),
                        marker_color=bar_colors,
                        hovertemplate=t("hover_real_standstill"),
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=seq["label"], y=seq["dark_min"], name=t("series_signal_lost"),
                        marker_color="rgba(234,179,8,0.85)",
                        hovertemplate=t("hover_signal_lost"),
                    ))
                    fig_bar.update_layout(
                        barmode="stack", template="plotly_white", height=360,
                        margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-40,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        yaxis_title=t("axis_minutes"),
                    )
                    st.plotly_chart(fig_bar, width='stretch')

                    # ── Tableau détaillé arrêt par arrêt ─────────────────────
                    st.markdown(f"###### {t('section_stop_table')}")
                    st.caption(t("stop_table_caption"))
                    tbl = seq[["stop", "matched", "dwell_min", "dark_min", "dist_m"]].copy()
                    tbl["matched"] = tbl["matched"].map({True: t("val_tracked"), False: t("val_unserved")})
                    tbl["dist_m"] = tbl["dist_m"].apply(lambda v: f"{v:.0f} m" if v > 0 else "—")
                    tbl["dwell_min"] = tbl["dwell_min"].apply(lambda v: f"{v:.1f} min" if v > 0 else "—")
                    tbl["dark_min"] = tbl["dark_min"].apply(lambda v: f"{v:.1f} min" if v > 0 else "—")
                    tbl.columns = [t("col_stop"), t("col_gps_tracked"), t("col_real_standstill"),
                                   t("col_signal_lost"), t("col_stop_distance")]
                    st.dataframe(tbl, hide_index=True, width='stretch')
                elif sel is not None:
                    st.info(t("no_sequence_data"))

    with tab_patterns:
        # ── Comparaison inter-opérateurs ──────────────────────────────────────
        st.markdown("##### Taux d'anomalie par opérateur (toutes lignes)")
        all_pats = {co: get_anomaly_patterns(co) for co in companies}
        cmp_rows = [
            {"Opérateur": co, "Trajets": p["total_trips"],
             "Anomalies": p["total_anomalies"], "Taux (%)": round(p["overall_rate"]*100, 1)}
            for co, p in all_pats.items() if p and p["total_trips"] > 0
        ]
        if cmp_rows:
            cmp_df = pd.DataFrame(cmp_rows).sort_values("Taux (%)", ascending=False)
            fig_cmp = px.bar(cmp_df, x="Opérateur", y="Taux (%)",
                             color="Taux (%)", color_continuous_scale="Reds",
                             text="Taux (%)", labels={"Taux (%)": "Taux d'anomalie (%)"})
            fig_cmp.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
            fig_cmp.update_layout(template="plotly_white", height=280,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  coloraxis_showscale=False, yaxis_title="Taux d'anomalie (%)")
            st.plotly_chart(fig_cmp, width='stretch')
            st.dataframe(cmp_df, hide_index=True, width='stretch')
            st.caption(
                "Taux calculé par le modèle Isolation Forest **par opérateur** (chaque opérateur est comparé à lui-même, "
                "pas aux autres). Un taux élevé ne signifie pas que cet opérateur est pire — cela signifie que ses "
                "trajets varient davantage par rapport à sa propre normale."
            )
        st.markdown("---")

        col = st.columns([1, 3])
        company = col[0].selectbox("Opérateur", companies, key="an_pat_co")
        pat = get_anomaly_patterns(company)
        if not pat or pat["total_trips"] == 0:
            st.info("Aucune donnée de tendance pour cet opérateur.")
        else:
            m = st.columns(3)
            m[0].metric("Trajets au total", f"{pat['total_trips']:,}")
            m[1].metric("Anomalies signalées", pat["total_anomalies"])
            m[2].metric("Taux d'anomalie global", f"{pat['overall_rate']*100:.1f} %")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Taux d'anomalie par ligne")
                bl = pd.DataFrame(pat["by_line"])
                if len(bl):
                    bl["rate_pct"] = (bl["rate"] * 100).round(1)
                    fig = px.bar(bl.sort_values("rate_pct", ascending=True),
                                 x="rate_pct", y="line", orientation="h",
                                 labels={"rate_pct": "Taux d'anomalie (%)", "line": "Ligne"},
                                 color="rate_pct", color_continuous_scale="Reds")
                    fig.update_layout(template="plotly_white", height=380,
                                      margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')
                    st.caption(f"Ligne la plus sujette aux anomalies : **{bl.sort_values('rate', ascending=False).iloc[0]['line']}**.")
            with c2:
                st.markdown("##### Taux d'anomalie par heure de la journée")
                bh = pd.DataFrame(pat["by_hour"])
                if len(bh):
                    bh["rate_pct"] = (bh["rate"] * 100).round(1)
                    fig = px.bar(bh, x="hour", y="rate_pct",
                                 labels={"hour": "Heure de départ", "rate_pct": "Taux d'anomalie (%)"},
                                 color="rate_pct", color_continuous_scale="Oranges")
                    fig.update_layout(template="plotly_white", height=380,
                                      margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')

            st.markdown("##### Bus les plus problématiques")
            bb = pd.DataFrame(pat["by_bus"])
            if len(bb):
                bb["rate_pct"] = (bb["rate"] * 100).round(1)
                st.dataframe(
                    bb.rename(columns={"bus": "Bus", "trips": "Trajets", "anomalies": "Anomalies",
                                       "rate_pct": "Taux (%)"})[["Bus", "Trajets", "Anomalies", "Taux (%)"]],
                    hide_index=True, width='stretch')

    with tab_tickets:
        st.caption(
            "Signal **complémentaire** aux trajets GPS ci-dessus, pas fusionné avec eux -- grain "
            "(société, ligne, bus, **jour**), pas trajet. Isolation Forest **par ligne** "
            "(repli société/global si historique insuffisant) sur `nbr_ticket` / `recette` / "
            "`avg_fare` (voir src/data/ticket_anomaly.py)."
        )

        with st.expander("Comment lire ces anomalies (avant de juger)", icon=":material/menu_book:"):
            st.markdown(
                """
**Les 3 signaux surveillés, et leurs causes plausibles :**

| Signal | Ce qu'il mesure | Causes bénignes possibles | Causes à investiguer |
|---|---|---|---|
| **Volume de tickets bas** | Nombre de tickets vendus ce jour-là | Jour férié, grève, service partiel, panne du bus (croiser avec l'onglet GPS) | Tickets non émis (encaissement sans ticket) |
| **Recette basse** | Total encaissé ce jour-là | Tarifs réduits massifs (scolaires), faible affluence réelle | Recette non déclarée |
| **Prix moyen anormal** | Recette ÷ tickets, comparé à la normale de la ligne | Mélange de catégories tarifaires, changement de barème | Erreur de caisse, tickets à tarif manipulé |

**Comment juger :** comparez toujours la valeur du jour à la **médiane de la ligne** (affichée
sur chaque carte) — un prix moyen de 20 DT est normal pour une ligne intercity et aberrant
pour une ligne urbaine. Le z-score (`z=+3.2`) indique à combien d'écarts-types de la normale
le jour se situe : au-delà de ±3, l'écart est fort.

**:orange[:material/warning:] Ligne signalée à ~100% :** si presque tous les jours d'une ligne sont marqués anormaux,
ce n'est PAS de la fraude quotidienne — c'est que la ligne diffère structurellement de la
référence utilisée (artefact de l'ancien modèle par société ; corrigé par le passage aux
modèles par ligne après réentraînement). Chaque carte concernée porte un avertissement.
                """
            )

        colt = st.columns([1, 2, 2])
        t_company = colt[0].selectbox("Opérateur", companies, key="tk_an_co")
        # Vue admin (tout, y compris nos propres pannes de machine) vs vue client (fiable
        # uniquement) -- voir _ticket_rows_with_reasons/is_machine_issue/is_no_service côté
        # API. Un volume de tickets à ~0 alors que le GPS confirme que le bus a bien roulé
        # n'est presque certainement pas une fraude/anomalie de recette mais une panne de
        # notre machine -- pas quelque chose à montrer comme une anomalie business face à
        # un client. `help=` porte l'explication du croisement GPS -- la petite icône "?"
        # que l'admin (et le client, la question se pose des deux côtés) peut survoler.
        t_view = colt[1].radio(t("tk_view_label"), [t("tk_view_admin"), t("tk_view_client")],
                               horizontal=True, key="tk_an_view",
                               help=t("tk_machine_detection_explain"))
        t_client_safe = (t_view == t("tk_view_client"))
        # Filtre bonne/mauvaise anomalie -- PRIORITÉ par défaut sur "à surveiller" (recette
        # en dessous de la normale), voir is_good_anomaly côté API : c'est ce qui a besoin
        # d'attention, une recette au-dessus de la normale n'est pas un problème à traiter.
        # Filtre CLIENT (après is_good_anomaly déjà calculé sur les lignes déjà reçues) --
        # pas besoin d'un aller-retour API, la donnée est déjà là.
        RECETTE_FILTERS = {
            "bad": t("tk_filter_bad"), "good": t("tk_filter_good"), "all": t("tk_filter_all"),
        }
        t_recette_filter = colt[2].selectbox(
            t("tk_filter_label"), list(RECETTE_FILTERS.keys()),
            format_func=lambda k: RECETTE_FILTERS[k], index=0, key="tk_an_recette_filter")
        t_pat = get_ticket_anomaly_patterns(t_company)

        if not t_pat or t_pat["total_days"] == 0:
            st.info("Aucune donnée de billetterie anormale pour cet opérateur.")
        else:
            m = st.columns(3)
            m[0].metric("Jours-bus au total", f"{t_pat['total_days']:,}")
            m[1].metric("Anomalies signalées", t_pat["total_anomalies"])
            m[2].metric("Taux d'anomalie global", f"{t_pat['overall_rate']*100:.1f} %")

            bl = pd.DataFrame(t_pat["by_line"])
            if len(bl):
                st.markdown("##### Taux d'anomalie par ligne")
                bl["rate_pct"] = (bl["rate"] * 100).round(1)
                fig = px.bar(bl.sort_values("rate_pct", ascending=True),
                             x="rate_pct", y="line", orientation="h",
                             labels={"rate_pct": "Taux d'anomalie (%)", "line": "Ligne"},
                             color="rate_pct", color_continuous_scale="Reds")
                fig.update_layout(template="plotly_white", height=380,
                                  margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, width='stretch')
                n_sat = int((bl["rate"] >= 0.9).sum())
                if n_sat:
                    st.warning(
                        f"{n_sat} ligne(s) signalée(s) à ≥90% de leurs jours — presque sûrement "
                        f"un artefact de référence (tarification atypique vs la normale utilisée), "
                        f"pas de la fraude quotidienne. Réentraîner le modèle billetterie "
                        f"(modèles par ligne) fait disparaître cet artefact.",
                        icon=":material/warning:",
                    )

            st.markdown("---")
            st.markdown("##### Jours signalés")
            t_line = st.selectbox("Filtrer par ligne (optionnel)",
                                  ["Toutes"] + sorted(bl["line"].tolist()) if len(bl) else ["Toutes"],
                                  key="tk_an_line")
            line_filter = None if t_line == "Toutes" else t_line
            # limit relevé (150 au lieu de 30) : le filtre bonne/mauvaise anomalie ci-dessous
            # s'applique APRÈS coup sur ce qui a été récupéré -- avec seulement les 30 plus
            # fortes anomalies, un filtre "à surveiller" pourrait ne presque rien montrer si
            # les plus fortes se trouvaient être de bonnes anomalies, alors qu'il y en a
            # peut-être beaucoup au-delà de ce top 30.
            t_hist = get_ticket_anomaly_history(t_company, line=line_filter, limit=150,
                                               client_safe=t_client_safe)
            anomalies = (t_hist or {}).get("anomalies", [])
            if t_recette_filter == "bad":
                anomalies = [a for a in anomalies if a.get("is_good_anomaly") is False]
            elif t_recette_filter == "good":
                anomalies = [a for a in anomalies if a.get("is_good_anomaly") is True]
            if not anomalies:
                st.info("Aucun jour anormal trouvé pour ce filtre.")
            for i, a in enumerate(anomalies):
                with st.container(border=True):
                    sev = SEV_META.get(a.get("severity", "medium"), SEV_META["medium"])
                    hdr = st.columns([0.16, 0.84], vertical_alignment="center")
                    hdr[0].badge(sev["label"], icon=sev["icon"], color=sev["color"])
                    hdr[1].markdown(f"**Bus {a['bus']} · Ligne {a['line']}** — {fmt_day(a['day'])}")
                    # Croisement GPS (voir main.py::_ticket_rows_with_reasons) -- ne peut
                    # apparaître qu'en vue admin, la vue client filtre déjà ces jours côté API.
                    # Bonne/mauvaise anomalie seulement si NI l'un NI l'autre n'explique déjà
                    # le jour -- sinon redondant (une panne machine est de toute façon toujours
                    # "recette en dessous de la normale", pas la peine de le redire).
                    if a.get("is_machine_issue"):
                        st.warning(t("tk_machine_issue", trips=a.get("gps_trip_count", 0)),
                                  icon="⚠️")
                    elif a.get("is_no_service"):
                        st.info(t("tk_no_service"), icon="ℹ️")
                    elif a.get("anomaly") and a.get("is_good_anomaly") is not None:
                        if a["is_good_anomaly"]:
                            st.success(t("tk_good_anomaly"), icon="📈")
                        else:
                            st.warning(t("tk_bad_anomaly"), icon="📉")
                    top = st.columns([1, 1, 1])
                    top[0].metric("Tickets", a["nbr_ticket"])
                    top[1].metric("Recette", f"{a['recette']:.0f} DT")
                    top[2].metric("Prix moyen", f"{a['avg_fare']:.1f} DT")

                    for reason in a["reasons"]:
                        st.markdown(f"&nbsp;&nbsp;• {reason}")

                    # Contexte de jugement : ce jour vs la normale de la ligne et de ce bus.
                    # Toutes les cellules en str : une colonne Arrow doit être homogène
                    # (mélanger int et "261 DT" fait échouer la sérialisation Streamlit).
                    if a.get("line_median_avg_fare") is not None:
                        def _cell(v, fmt):
                            return fmt.format(v) if v is not None else "—"
                        ctx = pd.DataFrame({
                            "Ce jour": [str(a["nbr_ticket"]), f"{a['recette']:.0f} DT", f"{a['avg_fare']:.2f} DT"],
                            "Médiane ligne": [
                                _cell(a.get("line_median_nbr_ticket"), "{:.0f}"),
                                _cell(a.get("line_median_recette"), "{:.0f} DT"),
                                _cell(a.get("line_median_avg_fare"), "{:.2f} DT"),
                            ],
                            "Médiane de ce bus": [
                                _cell(a.get("bus_median_nbr_ticket"), "{:.0f}"),
                                _cell(a.get("bus_median_recette"), "{:.0f} DT"),
                                _cell(a.get("bus_median_avg_fare"), "{:.2f} DT"),
                            ],
                        }, index=["Tickets", "Recette", "Prix moyen"])
                        st.dataframe(ctx, width='stretch')

                    rate = a.get("line_anomaly_rate")
                    if rate is not None and rate >= 0.9:
                        st.warning(
                            f"⚠️ {rate*100:.0f}% des jours de cette ligne sont signalés — écart "
                            f"structurel de la ligne (tarification), pas un incident de CE jour. "
                            f"À réévaluer après réentraînement par ligne."
                        )

                    if st.button("📈 Historique de ce bus sur cette ligne", key=f"tk_hist_{i}"):
                        detail = get_ticket_anomaly_explain(t_company, line=a["line"], bus=a["bus"],
                                                            client_safe=t_client_safe)
                        days_rows = (detail or {}).get("days", [])
                        if days_rows:
                            hdf = pd.DataFrame(days_rows)
                            hdf["date"] = pd.to_datetime(hdf["day"], format="%Y%m%d")
                            hdf = hdf.sort_values("date")
                            hdf["état"] = hdf["anomaly"].map({True: "Anormal", False: "Normal"})
                            c1, c2 = st.columns(2)
                            with c1:
                                figh = px.bar(hdf, x="date", y="nbr_ticket", color="état",
                                              color_discrete_map={"Normal": "#22c55e", "Anormal": "#ef4444"},
                                              labels={"nbr_ticket": "Tickets/jour", "date": ""})
                                figh.update_layout(template="plotly_white", height=260,
                                                   margin=dict(l=10, r=10, t=30, b=10),
                                                   title="Volume de tickets — chaque jour connu de ce bus")
                                st.plotly_chart(figh, width='stretch')
                            with c2:
                                figf = px.scatter(hdf, x="date", y="avg_fare", color="état",
                                                  color_discrete_map={"Normal": "#22c55e", "Anormal": "#ef4444"},
                                                  labels={"avg_fare": "Prix moyen (DT)", "date": ""})
                                if a.get("line_median_avg_fare") is not None:
                                    figf.add_hline(y=a["line_median_avg_fare"], line_dash="dash",
                                                   annotation_text="médiane ligne")
                                figf.update_layout(template="plotly_white", height=260,
                                                   margin=dict(l=10, r=10, t=30, b=10),
                                                   title="Prix moyen par ticket — vs médiane de la ligne")
                                st.plotly_chart(figf, width='stretch')
                        else:
                            st.info("Pas d'historique disponible pour ce bus.")

                    # Phase 2 -- détail par arrêt d'origine pour CE TRAJET PRÉCIS (bus + jour,
                    # voir /api/ticket-anomaly-stations) -- scopé au bus depuis le 2026-07-11 :
                    # sans lui, la répartition sommait TOUS les bus de la ligne ce jour-là et
                    # ne se recoupait pas avec la recette du bus-jour affiché ci-dessus.
                    sk = f"tk_stations_{i}"
                    shown = st.session_state.get(sk, False)
                    if st.button("🗺️ Voir le détail par arrêt" if not shown else "🗺️ Masquer le détail par arrêt",
                                key=sk + "_btn"):
                        st.session_state[sk] = not shown
                        st.rerun()
                    if shown:
                        stations_res = get_ticket_anomaly_stations(
                            t_company, line=a["line"], bus=a["bus"], day=a["day"])
                        stations = (stations_res or {}).get("stations", [])
                        if not stations:
                            st.info("Aucune donnée par arrêt pour ce trajet "
                                   "(modèle par arrêt pas encore entraîné, ou trop peu de données).")
                        else:
                            sum_ticket = sum(s["nbr_ticket"] for s in stations)
                            sum_recette = sum(s["recette"] for s in stations)
                            st.caption(f"{len(stations)} arrêt(s) desservi(s) par le bus {a['bus']} ce "
                                      f"jour-là — {sum_ticket} tickets / {sum_recette:.0f} DT au total "
                                      f"(vs {a['nbr_ticket']} tickets / {a['recette']:.0f} DT affiché "
                                      f"ci-dessus pour ce bus-jour).")
                            render_ticket_station_map(stations, key=sk + "_map")

                            def _station_table(rows, with_type=True):
                                tdf = pd.DataFrame(rows)
                                if with_type and "is_good_anomaly" in tdf.columns:
                                    tdf["Type"] = tdf["is_good_anomaly"].map(
                                        {True: "📈 Bonne", False: "📉 À surveiller"}).fillna("—")
                                    tdf.loc[~tdf["anomaly"], "Type"] = "—"
                                    return tdf[["station", "nbr_ticket", "recette", "avg_fare", "Type"]]
                                return tdf[["station", "nbr_ticket", "recette", "avg_fare"]]

                            def _render_trip_breakdown(label, combined_rows, by_direction):
                                """Sous-tables ALLER/RETOUR quand la direction est connue pour ce
                                bus-jour (voir /api/ticket-anomaly-stations `by_direction`, dérivée
                                de la parité de `voyage`) -- sinon repli sur la vue combinée
                                existante. Séparer par direction évite de mélanger dans un seul
                                total un aller et un retour qui n'ont pas la même pause terminus/
                                normale (voir la discussion sur bus 6030/ligne 209)."""
                                aller = (by_direction or {}).get("ALLER") or []
                                retour = (by_direction or {}).get("RETOUR") or []
                                if not aller and not retour:
                                    st.markdown(f"**{label}**")
                                    st.dataframe(_station_table(combined_rows), width='stretch', hide_index=True)
                                    return
                                st.markdown(f"**{label}** — direction connue pour ce trajet "
                                           "(ALLER/RETOUR séparés)")
                                dcols = st.columns(2)
                                for col, dname, rows_d in ((dcols[0], "ALLER", aller), (dcols[1], "RETOUR", retour)):
                                    with col:
                                        if rows_d:
                                            n_t = sum(r["nbr_ticket"] for r in rows_d)
                                            n_r = sum(r["recette"] for r in rows_d)
                                            st.caption(f"{dname} — {n_t} tickets / {n_r:.0f} DT")
                                            st.dataframe(_station_table(rows_d, with_type=False),
                                                        width='stretch', hide_index=True)
                                        else:
                                            st.caption(f"{dname} — pas de vente enregistrée")

                            _render_trip_breakdown(f"Ce trajet — bus {a['bus']} · {fmt_day(a['day'])}",
                                                   stations, (stations_res or {}).get("by_direction"))

                            # Trajet de référence billetterie (bus-jour NORMAL de cette ligne,
                            # voir /api/ticket-anomaly-reference) -- pour comparer la répartition
                            # par arrêt de CE trajet signalé à quoi ressemble un trajet normal,
                            # pas juste à un chiffre médian abstrait.
                            ref_res = get_ticket_anomaly_reference(t_company, a["line"])
                            ref_trip = (ref_res or {}).get("trip")
                            ref_stations = (ref_res or {}).get("stations") or []
                            if ref_trip and ref_stations:
                                _render_trip_breakdown(
                                    f"Trajet de référence (normal) — bus {ref_trip['bus']} · "
                                    f"{fmt_day(ref_trip['day'])} — {ref_trip['nbr_ticket']} tickets / "
                                    f"{ref_trip['recette']:.0f} DT au total",
                                    ref_stations, (ref_res or {}).get("by_direction"))
                            elif ref_trip:
                                st.caption("Trajet de référence trouvé mais pas de détail par arrêt "
                                          "disponible pour lui (bus/jour probablement en dehors de "
                                          "l'historique billetterie par arrêt).")

# ─────────────────────────────────────────────────────────────────────────────
# Démo -- Créer une ligne (Google Maps, rien n'est persisté -- voir route_demo.py)
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "Démo — Créer une ligne":
    from src.dashboard.route_demo import render_route_demo_page
    render_route_demo_page()

# ─────────────────────────────────────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "Assistant":
    st.title("💬 Copilote des opérations")
    st.caption("Posez vos questions sur l'exploitation des bus, les retards ou les anomalies.")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour ! Posez-moi vos questions sur l'exploitation des bus WiniCari, les retards ou les anomalies."}
        ]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Posez une question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Réflexion…"):
                resp = chat_with_bot(prompt)
                if resp:
                    answer = resp.get("answer", "Je n'ai pas trouvé de réponse.")
                    st.markdown(answer)
                    ctx = resp.get("context", [])
                    if ctx:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(ctx, 1):
                                st.write(f"{i}. {doc[:200]}…")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Échec de la réponse du service d'assistant.")

# ─────────────────────────────────────────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "Prévisions":
    st.title("📈 Prévision des retards")
    st.caption("Projections du retard quotidien par ligne (Prophet).")
    st.markdown("---")

    companies = get_companies()
    if companies:
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            company = st.selectbox("Opérateur", companies, key="fc_co")
            prophet = get_prophet_lines(company)
            lines = prophet.get("lines", [])
            by_line = prophet.get("by_line", {})
            if not lines:
                st.warning("Aucun modèle Prophet entraîné pour cet opérateur.")
                line = direction = None
            else:
                st.caption(f"Affichage des {len(lines)} ligne(s) avec un modèle Prophet entraîné.")
                line = st.selectbox("Ligne", lines, key="fc_line")
                directions = by_line.get(line, [])
                direction = st.selectbox("Direction", directions, key="fc_dir") if directions else None
            periods = st.slider("Horizon (jours)", 7, 90, 30, 7)
            if direction and st.button("Générer la prévision", type="primary", width='stretch'):
                res = get_forecast(company, line, direction, periods)
                if res:
                    st.session_state.fc_res = res
                else:
                    st.error("Échec de la génération de prévision pour cette ligne/direction.")
        with c2:
            if st.session_state.get("fc_res"):
                df = pd.DataFrame(st.session_state.fc_res)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pd.to_datetime(df["ds"]), y=df["yhat"],
                                         mode="lines", name="Prévision", line=dict(color="#0f172a", width=2)))
                if "yhat_lower" in df.columns:
                    fig.add_trace(go.Scatter(x=pd.to_datetime(df["ds"]), y=df["yhat_upper"],
                                             mode="lines", line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=pd.to_datetime(df["ds"]), y=df["yhat_lower"],
                                             mode="lines", line=dict(width=0), fill="tonexty",
                                             fillcolor="rgba(15,23,42,0.15)", showlegend=False))
                fig.update_layout(title="Retard quotidien projeté", xaxis_title="Date",
                                  yaxis_title="Retard (min)", template="plotly_white",
                                  margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width='stretch')

st.divider()
st.caption(f"WiniCari AI · Horloge démo {fmt_day(LATEST_DAY)} {datetime.now().strftime('%H:%M:%S')} · "
           f"FastAPI + Streamlit")
