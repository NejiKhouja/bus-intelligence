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
def get_trip_detail(company, line, bus, day, trip_id):
    return _get("/api/trip-detail", societe=company, line=line, bus=bus, day=day, trip_id=trip_id)

@st.cache_data(ttl=60)
def get_anomaly_history(company, line=None, limit=50):
    return _get("/api/anomaly-history", societe=company, line=line, limit=limit)

@st.cache_data(ttl=60)
def get_anomaly_explain(company, line, bus, day=None):
    return _get("/api/anomaly-explain", societe=company, line=line, bus=bus, day=day)

@st.cache_data(ttl=60)
def get_anomaly_patterns(company, line=None):
    return _get("/api/anomaly-patterns", societe=company, line=line)

@st.cache_data(ttl=60)
def get_current_anomalies(company, line=None):
    return _get("/api/current-anomalies", societe=company, line=line)

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
                 "Détection d'anomalies", "Assistant", "Prévisions"],
        icons=["speedometer2", "geo-alt", "broadcast-pin", "shield-exclamation", "chat-left-text", "graph-up-arrow"],
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
            stops_df.drop(columns=["seq"]), hide_index=True, use_container_width=True,
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
            st.plotly_chart(fig_d, use_container_width=True, key="eta_delay_trend")
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
            st.plotly_chart(fig, use_container_width=True, key="eta_anim")
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
                           use_container_width=True)

    manual = st.expander("…ou choisir un bus-jour précis")
    with manual:
        days = get_days_for_line(company, line) if line else []
        m1, m2, m3 = st.columns([1, 1, 1])
        m_day = m1.selectbox("Jour", days, key="gps_day") if days else None
        m_buses = get_buses_for_day(company, line, m_day) if m_day else []
        m_bus = m2.selectbox("Bus", m_buses, key="gps_bus_pick") if m_buses else None
        m3.write("")
        m3.write("")
        load_manual = m3.button("Charger le trajet", use_container_width=True) if m_bus else False

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
                st.dataframe(show, hide_index=True, use_container_width=True, height=200)
            else:
                st.caption("Aucune perte au-dessus du seuil de détection.")

        with left:
            fig = rt.build_gps_animation(route, P)
            st.plotly_chart(fig, use_container_width=True, key="gps_anim")
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
    st.title("🛡️ Détection d'anomalies")
    st.caption("Trajets signalés par Isolation Forest + autoencodeur LSTM — expliqués en langage clair.")
    st.markdown("---")

    companies = get_companies()
    if not companies:
        st.warning("Aucune donnée disponible depuis l'API.")
        st.stop()

    tab_live, tab_explain, tab_patterns = st.tabs(
        ["🚨 Trajets signalés", "🔍 Expliquer un bus", "📊 Tendances"])

    SEV_COLOR = {"high": "🔴", "medium": "🟠", "low": "🟡"}
    SEV_FR = {"high": "Élevée", "medium": "Moyenne", "low": "Faible"}

    def _fmt_duration(minutes):
        h, m = int(minutes) // 60, int(minutes) % 60
        return f"{h}h{m:02d}" if h else f"{m} min"

    def render_alert_cards(anomalies):
        for a in anomalies:
            icon = SEV_COLOR.get(a["severity"], "🟡")
            with st.container(border=True):
                top = st.columns([3, 1, 1])
                top[0].markdown(f"**{icon} Bus {a['bus']} · Ligne {a['line']} · {a['dir']}** — {fmt_day(a['day'])}")
                top[1].metric("Gravité", SEV_FR.get(a["severity"], a["severity"]))
                dur = a.get("trip_duration_min") or a.get("total_elapsed_min", 0)
                top[2].metric("Durée trajet", _fmt_duration(dur))
                if a["reasons"]:
                    for reason in a["reasons"]:
                        st.markdown(f"&nbsp;&nbsp;• {reason}")
                else:
                    st.caption("Signalé par le score du modèle (pas de cause unique dominante).")
                ps = a.get("problem_stops") or {}
                chips = []
                # Genuine dwell anomaly at a named stop
                if ps.get("longest_stop") and ps["longest_stop"]["dwell_min"] >= 5:
                    chips.append(f"🛑 Immobilisation réelle : **{ps['longest_stop']['stop']}** "
                                 f"({ps['longest_stop']['dwell_min']:.0f} min sans mouvement GPS)")
                # Signal loss at a named stop (separate from dwell)
                if ps.get("signal_loss_stop"):
                    sl = ps["signal_loss_stop"]
                    chips.append(f"📡 Perte de signal : **{sl['stop']}** (~{sl['dark_min']:.0f} min sans ping)")
                # Matched stops that were still far off expected position (GPS drift / detour)
                if ps.get("farthest_stop"):
                    chips.append(f"📍 Arrêt suivi mais décalé : **{ps['farthest_stop']['stop']}** "
                                 f"(~{ps['farthest_stop']['dist_m']:.0f} m de la position attendue)")
                # Unmatched stops — bus never passed within range
                if ps.get("off_route_stops"):
                    others = ps.get('off_route_count', len(ps['off_route_stops'])) - len(ps['off_route_stops'])
                    suffix = f" (+{others} autres)" if others > 0 else ""
                    chips.append(f"🚧 Arrêts non desservis : {', '.join(ps['off_route_stops'])}{suffix}")
                for c in chips:
                    st.caption(c)

    with tab_live:
        col = st.columns([1, 1, 2])
        company = col[0].selectbox("Opérateur", companies, key="an_live_co")
        lines = ["Toutes les lignes"] + get_lines(company)
        line_sel = col[1].selectbox("Ligne", lines, key="an_live_line")
        line_param = None if line_sel == "Toutes les lignes" else line_sel

        data = get_current_anomalies(company, line_param)
        if data:
            m = st.columns(3)
            m[0].metric("Jour d'exploitation", fmt_day(data["date"]))
            m[1].metric("Trajets ce jour", data["total_trips"])
            pct = 100 * data["anomaly_count"] / data["total_trips"] if data["total_trips"] else 0
            m[2].metric("Signalés", f"{data['anomaly_count']}  ({pct:.1f} %)")
            if data["anomalies"]:
                st.markdown("#### Trajets signalés ce jour")
                render_alert_cards(data["anomalies"])
            else:
                st.success("Aucune anomalie le dernier jour d'exploitation pour ce périmètre.")
                st.caption("Essayez une ligne précise, ou consultez l'historique ci-dessous.")

        st.markdown("---")
        st.markdown("#### Historique récent")
        hist = get_anomaly_history(company, line_param, limit=40)
        if hist and hist.get("anomalies"):
            render_alert_cards(hist["anomalies"][:12])
        else:
            st.info("Aucune anomalie historique pour ce périmètre.")

    with tab_explain:
        # ── Filtres ──────────────────────────────────────────────────────────
        col = st.columns(5)
        company = col[0].selectbox("Opérateur", companies, key="an_ex_co")
        lines = get_lines(company)
        line = col[1].selectbox("Ligne", lines, key="an_ex_line") if lines else None

        # Bus is optional — "Tous les bus" analyses the whole line
        bus_opts = ["Tous les bus"] + [str(b) for b in (get_buses_for_line(company, line) if line else [])]
        bus_label = col[2].selectbox("Bus", bus_opts, key="an_ex_bus")
        bus = None if bus_label == "Tous les bus" else int(bus_label)

        days = ["Tous les jours"] + (get_days_for_line(company, line) if line else [])
        day_sel = col[3].selectbox("Jour", days, key="an_ex_day")
        day_param = None if day_sel == "Tous les jours" else day_sel

        if line and col[4].button("Analyser", type="primary", use_container_width=True):
            res = get_anomaly_explain(company, line, bus, day_param)
            st.session_state["an_ex_res"] = res
            st.session_state["an_ex_ctx"] = (company, line, bus, day_param)

        res = st.session_state.get("an_ex_res")
        ctx = st.session_state.get("an_ex_ctx")

        if res and ctx == (company, line, bus, day_param):
            if not res:
                st.error("Impossible de récupérer l'explication.")
            elif res["anomaly_count"] == 0:
                scope = f"Bus {bus} · Ligne {line}" if bus else f"Ligne {line}"
                st.success(f"{scope} : aucun trajet anormal détecté — tout est dans la normale.")
            else:
                scope = f"Bus {bus} · Ligne {line}" if bus else f"Ligne {line}"
                st.markdown(f"#### {scope} — analyse des anomalies")

                # ── Métriques générales (avec explications) ──────────────────
                mc = st.columns(3)
                ex_pct = 100 * res["anomaly_count"] / res["total_trips"] if res["total_trips"] else 0
                mc[0].metric("Trajets analysés", f"{res['total_trips']}")
                mc[0].caption("Nombre total de trajets dans la période sélectionnée pour ce périmètre.")
                mc[1].metric("Trajets anormaux", f"{res['anomaly_count']}  ({ex_pct:.1f} %)")
                mc[1].caption("Trajets signalés comme anormaux par le modèle de détection (Isolation Forest + LSTM).")
                if res.get("avg_duration_min"):
                    mc[2].metric("Durée normale (médiane)", _fmt_duration(res["avg_duration_min"]))
                    mc[2].caption("Durée médiane d'un trajet non anormal sur cette ligne — sert de référence pour juger si un trajet est trop long ou trop court.")

                st.markdown("---")
                # ── Cartes d'alerte ──────────────────────────────────────────
                st.markdown("##### Trajets signalés")
                render_alert_cards(res["anomalies"])

                # ── Détail par trajet ─────────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Analyse détaillée d'un trajet")
                st.caption("Sélectionnez un trajet ci-dessous pour voir la carte de ses arrêts, le graphique d'immobilisation et le tableau complet arrêt par arrêt.")

                trip_labels = [
                    f"Bus {a['bus']} · {fmt_day(a['day'])} · {a['dir']} · score {a['anomaly_strength']:.2f}"
                    for a in res["anomalies"]
                ]
                sel_idx = st.selectbox(
                    "Trajet à analyser :", range(len(trip_labels)),
                    format_func=lambda i: trip_labels[i],
                    key="an_ex_trip_sel",
                )
                sel = res["anomalies"][sel_idx]

                # Fetch per-trip sequence on demand (cached)
                detail = get_trip_detail(company, line, sel["bus"], sel["day"], sel["trip_id"])
                seq_list = (detail or {}).get("sequence", [])

                if seq_list:
                    seq = pd.DataFrame(seq_list)
                    if "dark_min" not in seq.columns:
                        seq["dark_min"] = 0.0

                    # ── Métriques du trajet sélectionné ──────────────────────
                    tc = st.columns(4)
                    tc[0].metric("Durée trajet", _fmt_duration(sel["trip_duration_min"]))
                    tc[0].caption("Durée réelle du trajet du premier au dernier arrêt enregistré.")
                    if res.get("avg_duration_min"):
                        delta_min = sel["trip_duration_min"] - res["avg_duration_min"]
                        sign = "+" if delta_min > 0 else ""
                        tc[1].metric("Écart vs normal", f"{sign}{_fmt_duration(abs(delta_min))}")
                        tc[1].caption("Différence entre ce trajet et la durée normale de la ligne. Un écart positif signifie un trajet plus long que d'habitude.")
                    tc[2].metric("Arrêts desservis", f"{int(seq['matched'].sum())} / {len(seq)}")
                    tc[2].caption("Nombre d'arrêts où le bus a été détecté dans la zone GPS (suivi) sur le total d'arrêts prévus sur la ligne.")
                    tc[3].metric("Score anomalie", f"{sel['anomaly_strength']:.2f}")
                    tc[3].caption("Score calculé par le modèle Isolation Forest. Plus il est élevé, plus le comportement du bus s'écarte de la normale.")

                    # ── Carte des arrêts ──────────────────────────────────────
                    map_rows = [s for s in seq_list if s.get("lat") and s.get("lon")]
                    if map_rows:
                        st.markdown("###### Carte du trajet")
                        st.caption(
                            "Chaque cercle représente un arrêt. La taille reflète la durée d'immobilisation + perte de signal. "
                            "🟢 Normal · 🔵 Immobilisation longue (≥10 min) · 🟡 Perte de signal (≥5 min) · 🔴 Arrêt non desservi (bus jamais passé)"
                        )
                        mdf = pd.DataFrame(map_rows)

                        def _scolor(row):
                            if not row["matched"]: return "#ef4444"
                            if row.get("dark_min", 0) >= 5: return "#f59e0b"
                            if row.get("dwell_min", 0) >= 10: return "#2563eb"
                            return "#22c55e"

                        mdf["color"] = mdf.apply(_scolor, axis=1)
                        mdf["msize"] = (mdf["dwell_min"] + mdf.get("dark_min", 0)).clip(8, 28)
                        mdf["hover"] = (
                            "<b>" + mdf["seq"].astype(str) + " : " + mdf["stop"] + "</b><br>" +
                            "Immobilisation réelle : " + mdf["dwell_min"].apply(lambda v: f"{v:.1f} min") + "<br>" +
                            "Signal perdu : " + mdf["dark_min"].apply(lambda v: f"{v:.1f} min") + "<br>" +
                            "Distance de l'arrêt attendu : " + mdf["dist_m"].apply(lambda v: f"{v:.0f} m") + "<br>" +
                            "Suivi GPS : " + mdf["matched"].map({True: "Oui", False: "Non"})
                        )

                        fig_map = go.Figure()
                        fig_map.add_trace(go.Scattermapbox(
                            lat=mdf["lat"].tolist(), lon=mdf["lon"].tolist(),
                            mode="lines", line=dict(width=3, color="#94a3b8"),
                            name="Itinéraire prévu", hoverinfo="skip",
                        ))
                        for color, label in [
                            ("#22c55e", "Arrêt normal"),
                            ("#2563eb", "Immobilisation longue"),
                            ("#f59e0b", "Perte de signal GPS"),
                            ("#ef4444", "Arrêt non desservi"),
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
                        fig_map.update_layout(
                            mapbox=dict(
                                style="open-street-map",
                                center=dict(lat=float(mdf["lat"].mean()), lon=float(mdf["lon"].mean())),
                                zoom=10,
                            ),
                            margin=dict(l=0, r=0, t=0, b=0), height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.info("Coordonnées GPS non disponibles pour les arrêts de cette ligne.")

                    # ── Graphique immobilisation par arrêt ────────────────────
                    st.markdown("###### Immobilisation et perte de signal par arrêt")
                    st.caption(
                        "Les barres bleues montrent le temps réel où le bus était à l'arrêt avec GPS actif. "
                        "Les barres jaunes montrent les périodes sans ping GPS à cet arrêt (signal perdu). "
                        "Les barres rouges correspondent aux arrêts non desservis."
                    )
                    seq["label"] = seq["seq"].astype(str) + " · " + seq["stop"]
                    bar_colors = ["#2563eb" if m else "#dc2626" for m in seq["matched"]]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=seq["label"], y=seq["dwell_min"], name="Immobilisation réelle (GPS actif)",
                        marker_color=bar_colors,
                        hovertemplate="<b>%{x}</b><br>Immob. réelle : %{y:.1f} min<extra></extra>",
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=seq["label"], y=seq["dark_min"], name="Signal perdu (sans ping)",
                        marker_color="rgba(234,179,8,0.85)",
                        hovertemplate="<b>%{x}</b><br>Signal perdu : %{y:.1f} min<extra></extra>",
                    ))
                    fig_bar.update_layout(
                        barmode="stack", template="plotly_white", height=360,
                        margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-40,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        yaxis_title="Minutes",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # ── Tableau détaillé arrêt par arrêt ─────────────────────
                    st.markdown("###### Tableau arrêt par arrêt")
                    st.caption(
                        "**Suivi GPS** : le bus est passé dans la zone de l'arrêt et a été détecté. "
                        "**Immob. réelle** : temps d'arrêt avec GPS actif. "
                        "**Signal perdu** : durée sans ping GPS à cet arrêt (non comptée comme arrêt). "
                        "**Distance arrêt** : écart entre la position GPS du bus et la position théorique de l'arrêt."
                    )
                    tbl = seq[["stop", "matched", "dwell_min", "dark_min", "dist_m"]].copy()
                    tbl["matched"] = tbl["matched"].map({True: "✅ Suivi", False: "❌ Non desservi"})
                    tbl["dist_m"] = tbl["dist_m"].apply(lambda v: f"{v:.0f} m" if v > 0 else "—")
                    tbl["dwell_min"] = tbl["dwell_min"].apply(lambda v: f"{v:.1f} min" if v > 0 else "—")
                    tbl["dark_min"] = tbl["dark_min"].apply(lambda v: f"{v:.1f} min" if v > 0 else "—")
                    tbl.columns = ["Arrêt", "Suivi GPS", "Immob. réelle", "Signal perdu", "Distance arrêt"]
                    st.dataframe(tbl, hide_index=True, use_container_width=True)
                else:
                    st.info("Aucune donnée de séquence disponible pour ce trajet.")

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
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.dataframe(cmp_df, hide_index=True, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Bus les plus problématiques")
            bb = pd.DataFrame(pat["by_bus"])
            if len(bb):
                bb["rate_pct"] = (bb["rate"] * 100).round(1)
                st.dataframe(
                    bb.rename(columns={"bus": "Bus", "trips": "Trajets", "anomalies": "Anomalies",
                                       "rate_pct": "Taux (%)"})[["Bus", "Trajets", "Anomalies", "Taux (%)"]],
                    hide_index=True, use_container_width=True)

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
            if direction and st.button("Générer la prévision", type="primary", use_container_width=True):
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
                st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(f"WiniCari AI · Horloge démo {fmt_day(LATEST_DAY)} {datetime.now().strftime('%H:%M:%S')} · "
           f"FastAPI + Streamlit")
