
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date

from src.data import webservices as ws


def render_webservice_demo_page(_post):
    st.title("📡 Démo — Web services en direct")
    st.caption("Récupère les données d'un jour directement depuis les web services de la "
              "plateforme (pas MongoDB), puis les envoie aux endpoints de scoring "
              "`/api/anomaly/score-live` et `/api/ticket-anomaly/score-live` -- exactement "
              "le pipeline que la future page PHP suivra. Rien n'est enregistré ici.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    day_val = col1.date_input("Jour", value=date(2026, 7, 12))
    societe = col2.text_input("Société", value="SRT.ELGOUAFEL")
    day_compact = day_val.strftime("%Y%m%d")
    day_dashed = day_val.strftime("%Y-%m-%d")

    if st.button("Vérifier + charger", type="primary"):
        try:
            ready = ws.is_day_ready(day_compact)
        except Exception as e:
            st.error(f"Erreur en contactant le web service (isDayReady) : {e}")
            st.stop()
        if not ready:
            st.warning(f"Le traitement de nuit n'est pas encore prêt pour {day_compact} "
                      "(collection absente ou vide côté plateforme) -- réessayez plus tard.")
            st.stop()

        with st.spinner("Récupération des pings GPS (getPingsForDay)..."):
            try:
                pings = ws.get_pings_for_day(day_compact, societe=societe)
            except Exception as e:
                st.error(f"Erreur getPingsForDay : {e}")
                pings = []

        with st.spinner("Récupération des totaux billetterie (getTicketTotalsForDay)..."):
            try:
                totals = ws.get_ticket_totals_for_day(day_dashed)
            except Exception as e:
                st.error(f"Erreur getTicketTotalsForDay : {e}")
                totals = []

        st.session_state["ws_pings"] = pings
        st.session_state["ws_ticket_totals"] = totals
        st.session_state["ws_day_compact"] = day_compact
        st.session_state["ws_societe"] = societe

    pings = st.session_state.get("ws_pings")
    if pings is not None:
        st.markdown("### Trajets GPS")
        groups = ws.group_pings_by_bus_line(pings)
        st.success(f"{len(pings):,} pings reçus -- {len(groups)} paire(s) ligne/bus "
                  f"disponibles pour {st.session_state.get('ws_societe')} ce jour-là.")

        if groups:
            options = sorted(groups.keys())
            labels = [f"Ligne {l} · Bus {b} ({len(groups[(l, b)])} pings)" for l, b in options]
            idx = st.selectbox("Choisir un trajet à scorer", range(len(options)),
                              format_func=lambda i: labels[i], key="ws_trip_pick")
            line, bus = options[idx]

            if st.button("Scorer ce trajet (GPS, en direct)"):
                rows = ws.pings_to_score_live_rows(groups[(line, bus)])
                payload = {"day": st.session_state["ws_day_compact"], "line": line,
                          "societe": st.session_state["ws_societe"], "bus": int(bus),
                          "pings": rows}
                with st.spinner("Reconstruction du trajet + scoring en direct..."):
                    result = _post("/api/anomaly/score-live", json=payload)
                if not result:
                    st.error("Le scoring a échoué (voir les logs de l'API pour le détail).")
                else:
                    trips = result.get("trips", [])
                    st.write(f"{len(trips)} trajet(s) reconstruit(s) pour ce bus-jour, "
                            f"{result.get('anomaly_count', 0)} signalé(s).")
                    for t in trips:
                        badge = "🔴" if t.get("severity") == "high" else (
                                "🟠" if t.get("severity") == "medium" else "🟢")
                        st.write(f"{badge} **{t.get('dir')}** · {t.get('trip_duration_min')} min "
                                f"· {t.get('n_stops')} arrêts")
                        for reason in t.get("reasons", []):
                            st.caption(f"• {reason}")

    totals = st.session_state.get("ws_ticket_totals")
    if totals is not None:
        st.markdown("### Billetterie")
        soc = st.session_state.get("ws_societe")
        sub = [r for r in totals if r.get("societe") == soc]
        st.info(f"{len(totals)} ligne(s) de billetterie reçues au total ce jour-là, "
               f"{len(sub)} pour {soc}.")

        if sub and st.button("Scorer la billetterie (toutes les lignes de cette société, en direct)"):
            rows = ws.ticket_totals_to_rows(sub, st.session_state["ws_day_compact"])
            with st.spinner("Scoring billetterie en direct..."):
                result = _post("/api/ticket-anomaly/score-live", json={"rows": rows})
            if not result:
                st.error("Le scoring billetterie a échoué (voir les logs de l'API).")
            else:
                days = result.get("days", [])
                st.write(f"{len(days)} bus-jour scoré(s), "
                        f"{result.get('anomaly_count', 0)} signalé(s).")
                if days:
                    df = pd.DataFrame(days)
                    cols = [c for c in ["line", "bus", "nbr_ticket", "recette", "avg_fare",
                                       "anomaly", "severity", "is_good_anomaly"] if c in df.columns]
                    st.dataframe(df[cols], width="stretch", hide_index=True)
