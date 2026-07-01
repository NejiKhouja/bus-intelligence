"""Real-time map / replay / animation helpers for the WiniCari dashboard.

Pure presentation logic kept out of app.py: GPS-track replay math (including the
signal-gap estimate that mirrors the backend Kalman fallback), and the Plotly
open-street-map figure builders for the GPS-fallback and Live-ETA views.

Production note
---------------
`detect_signal_loss` is the same event predicate a live system would run on each
incoming ping: "no fix for longer than `threshold_s` => the bus is dark". In the
demo it is driven by an accelerated replay clock; plugged into a real ping stream
the notify/estimate path is identical — only the time source changes.
"""
from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data.fallback import haversine_m


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def s_to_latlon(s_km: float, route: List[Dict]) -> tuple[float, float]:
    """Distance along route (km) -> (lat, lon) via the stop polyline. Mirrors backend."""
    s = np.array([r["s_m"] for r in route], dtype=float)        # route s_m is in km
    lat = np.array([r["lat"] for r in route], dtype=float)
    lon = np.array([r["lon"] for r in route], dtype=float)
    if len(s) == 0:
        return 0.0, 0.0
    if s_km <= s[0]:
        return float(lat[0]), float(lon[0])
    if s_km >= s[-1]:
        return float(lat[-1]), float(lon[-1])
    i = int(np.searchsorted(s, s_km)) - 1
    denom = (s[i + 1] - s[i]) or 1e-9
    f = (s_km - s[i]) / denom
    return float(lat[i] + f * (lat[i + 1] - lat[i])), float(lon[i] + f * (lon[i + 1] - lon[i]))


def circle_lonlat(lat: float, lon: float, radius_m: float, n: int = 48):
    """Points approximating a circle of given radius (m) around (lat, lon)."""
    radius_m = max(radius_m, 30.0)
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(np.cos(np.radians(lat)), 0.01))
    ang = np.linspace(0, 2 * np.pi, n)
    return (lat + dlat * np.sin(ang)).tolist(), (lon + dlon * np.cos(ang)).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Replay state
# ─────────────────────────────────────────────────────────────────────────────

def prep_track(track: List[Dict]) -> Dict:
    """Vectorize a /api/gps-track payload for fast per-tick lookup."""
    t = pd.to_datetime([p["t"] for p in track], format="ISO8601")
    return {
        "t_unix": (t.astype("int64") // 10 ** 9).to_numpy().astype(float),
        "t0": t[0], "t1": t[-1],
        "lat": np.array([p["lat"] for p in track], float),
        "lon": np.array([p["lon"] for p in track], float),
        "ks_m": np.array([p["ks_m"] for p in track], float),     # km
        "kv": np.array([p["kv"] for p in track], float),          # m/s
        "unc": np.array([p["uncertainty_m"] for p in track], float),
        "gap": np.array([bool(p["signal_gap"]) for p in track]),
        "speed": np.array([p.get("speed", 0) for p in track], float),
    }


def detect_signal_loss(dt_since_last_fix_s: float, threshold_s: float = 180.0) -> bool:
    """Live-ready predicate: bus is 'dark' if no fix for longer than threshold."""
    return dt_since_last_fix_s > threshold_s


def position_at(P: Dict, route: List[Dict], sim_unix: float,
                threshold_s: float = 180.0) -> Dict:
    """Best-estimate bus state at sim time `sim_unix` along the replay.

    Returns dict: lat, lon, dark(bool), uncertainty_m, dt_dark_s, last_idx,
    last_fix_t (unix), recovery (bool, just came back this step is handled by caller).
    During a gap we propagate the last Kalman state forward — exactly the backend
    fallback — instead of revealing the (unknown-in-real-life) recovery ping.
    """
    tu = P["t_unix"]
    if sim_unix <= tu[0]:
        return {"lat": P["lat"][0], "lon": P["lon"][0], "dark": False,
                "uncertainty_m": P["unc"][0], "dt_dark_s": 0.0, "last_idx": 0,
                "last_fix_unix": tu[0], "s_est_km": P["ks_m"][0], "s_last_km": P["ks_m"][0]}
    if sim_unix >= tu[-1]:
        return {"lat": P["lat"][-1], "lon": P["lon"][-1], "dark": False,
                "uncertainty_m": P["unc"][-1], "dt_dark_s": 0.0,
                "last_idx": len(tu) - 1, "last_fix_unix": tu[-1],
                "s_est_km": P["ks_m"][-1], "s_last_km": P["ks_m"][-1]}

    i = int(np.searchsorted(tu, sim_unix)) - 1      # last point with t <= sim
    i = max(0, min(i, len(tu) - 2))
    nxt_is_gap = bool(P["gap"][i + 1])
    dt = sim_unix - tu[i]

    if nxt_is_gap and detect_signal_loss(dt, threshold_s):
        # Dark: propagate last Kalman state (s_km + kv*dt), grow uncertainty.
        s_est_km = P["ks_m"][i] + (P["kv"][i] * dt) / 1000.0
        s_max = route[-1]["s_m"] if route else s_est_km
        s_est_km = float(np.clip(s_est_km, 0.0, s_max))
        lat, lon = s_to_latlon(s_est_km, route)
        unc = float(P["unc"][i] + abs(P["kv"][i]) * dt * 0.1)
        return {"lat": lat, "lon": lon, "dark": True, "uncertainty_m": unc,
                "dt_dark_s": dt, "last_idx": i, "last_fix_unix": tu[i],
                "s_est_km": s_est_km, "s_last_km": float(P["ks_m"][i])}

    # Normal: linear interpolation between consecutive fixes.
    span = (tu[i + 1] - tu[i]) or 1.0
    f = np.clip(dt / span, 0.0, 1.0)
    lat = float(P["lat"][i] + f * (P["lat"][i + 1] - P["lat"][i]))
    lon = float(P["lon"][i] + f * (P["lon"][i + 1] - P["lon"][i]))
    unc = float(P["unc"][i] + f * (P["unc"][i + 1] - P["unc"][i]))
    s_now = float(P["ks_m"][i] + f * (P["ks_m"][i + 1] - P["ks_m"][i]))
    return {"lat": lat, "lon": lon, "dark": False, "uncertainty_m": unc,
            "dt_dark_s": 0.0, "last_idx": i, "last_fix_unix": tu[i],
            "s_est_km": s_now, "s_last_km": s_now}


# ─────────────────────────────────────────────────────────────────────────────
# Map builders (Plotly open-street-map — no token required)
# ─────────────────────────────────────────────────────────────────────────────

def _route_traces(route: List[Dict], color="#94a3b8") -> list:
    return [go.Scattermapbox(
        lat=[r["lat"] for r in route], lon=[r["lon"] for r in route], mode="lines",
        line=dict(width=3, color=color), name="Itinéraire", hoverinfo="skip")]


def _stops_trace(route: List[Dict], name="Arrêts") -> go.Scattermapbox:
    """Clear, numbered stop markers with names on hover (and labels on short lines)."""
    show_labels = len(route) <= 16
    return go.Scattermapbox(
        lat=[r["lat"] for r in route], lon=[r["lon"] for r in route],
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=11, color="#0f172a"),
        text=[r.get("stop", "") for r in route] if show_labels else None,
        textposition="top right", textfont=dict(size=10, color="#334155"),
        hovertext=[f"#{int(r['seq'])+1} · {r.get('stop','')}" for r in route],
        hoverinfo="text", name=name)


def _center(route: List[Dict], extra_lat=None, extra_lon=None):
    lats = [r["lat"] for r in route] + ([extra_lat] if extra_lat else [])
    lons = [r["lon"] for r in route] + ([extra_lon] if extra_lon else [])
    return (float(np.mean(lats)), float(np.mean(lons))) if lats else (35.0, 9.0)


def route_segment(route: List[Dict], s_lo: float, s_hi: float):
    """Polyline lat/lon following the route between two distances (km)."""
    s_lo, s_hi = sorted([float(s_lo), float(s_hi)])
    la0, lo0 = s_to_latlon(s_lo, route)
    lats, lons = [la0], [lo0]
    for r in route:
        if s_lo < r["s_m"] < s_hi:
            lats.append(r["lat"]); lons.append(r["lon"])
    la1, lo1 = s_to_latlon(s_hi, route)
    lats.append(la1); lons.append(lo1)
    return lats, lons


def _bus_marker(lat, lon, name, color="#2563eb") -> go.Scattermapbox:
    """A clear, professional bus icon (emoji glyph + halo) anchored at the position."""
    return go.Scattermapbox(
        lat=[lat], lon=[lon], mode="markers+text",
        marker=dict(size=26, color=color),
        text=["🚌"], textfont=dict(size=17), textposition="middle center",
        name=name, hoverinfo="name")


def _layout(fig, route, extra_lat=None, extra_lon=None, zoom=8):
    clat, clon = _center(route, extra_lat, extra_lon)
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=clat, lon=clon), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01,
                    bgcolor="rgba(255,255,255,0.75)"),
        showlegend=True,
        # uirevision keeps zoom/pan & avoids re-mounting the map each tick (no flashing)
        uirevision="keep")
    return fig


def build_replay_map(route: List[Dict], P: Dict, upto_idx: int, pos: Dict,
                     last_known: Optional[Dict] = None) -> go.Figure:
    """GPS-fallback replay map: route + tracked path + dark segment (red) + bus."""
    fig = go.Figure(_route_traces(route))
    fig.add_trace(_stops_trace(route))

    j = max(1, upto_idx + 1)
    fig.add_trace(go.Scattermapbox(
        lat=P["lat"][:j].tolist(), lon=P["lon"][:j].tolist(), mode="lines",
        line=dict(width=5, color="#2563eb"), name="Tracked (live GPS)", hoverinfo="skip"))

    if pos["dark"]:
        # The stretch of route the bus covered WHILE DARK, drawn in red.
        rlat, rlon = route_segment(route, pos["s_last_km"], pos["s_est_km"])
        fig.add_trace(go.Scattermapbox(
            lat=rlat, lon=rlon, mode="lines",
            line=dict(width=6, color="#dc2626"), name="Lost signal (estimated)",
            hoverinfo="skip"))
        clat, clon = circle_lonlat(pos["lat"], pos["lon"], pos["uncertainty_m"])
        fig.add_trace(go.Scattermapbox(
            lat=clat, lon=clon, mode="lines", fill="toself",
            fillcolor="rgba(220,38,38,0.15)", line=dict(width=1, color="rgba(220,38,38,0.6)"),
            name="Position uncertainty", hoverinfo="skip"))
        if last_known:
            fig.add_trace(go.Scattermapbox(
                lat=[last_known["lat"]], lon=[last_known["lon"]], mode="markers",
                marker=dict(size=13, color="#64748b"), name="Last known fix",
                hoverinfo="name"))
        fig.add_trace(_bus_marker(pos["lat"], pos["lon"], "Bus (AI estimate)", "#dc2626"))
    else:
        fig.add_trace(_bus_marker(pos["lat"], pos["lon"], "Bus (live GPS)", "#2563eb"))

    return _layout(fig, route)


def build_eta_map(route: List[Dict], bus_pos: Dict, rider_seq: int,
                  upto_lat=None, upto_lon=None) -> go.Figure:
    """Live-ETA map: route + clear stops + rider's stop pin + moving bus icon."""
    fig = go.Figure(_route_traces(route))

    # Highlight the leg between the bus and the rider, if both known.
    rider = next((r for r in route if int(r["seq"]) == int(rider_seq)), None)
    if bus_pos and rider:
        rlat, rlon = route_segment(route, bus_pos.get("s_est_km", 0), rider["s_m"])
        fig.add_trace(go.Scattermapbox(
            lat=rlat, lon=rlon, mode="lines",
            line=dict(width=6, color="#2563eb"), name="Bus → you", hoverinfo="skip"))

    fig.add_trace(_stops_trace(route))

    if rider:
        fig.add_trace(go.Scattermapbox(
            lat=[rider["lat"]], lon=[rider["lon"]], mode="markers+text",
            marker=dict(size=20, color="#16a34a"),
            text=["🧍"], textfont=dict(size=15), textposition="middle center",
            hovertext=[f"Your stop · {rider.get('stop','')}"], hoverinfo="text",
            name="Your stop"))

    if bus_pos:
        fig.add_trace(_bus_marker(bus_pos["lat"], bus_pos["lon"], "Bus"))

    return _layout(fig, route, bus_pos.get("lat") if bus_pos else None,
                   bus_pos.get("lon") if bus_pos else None)


# ─────────────────────────────────────────────────────────────────────────────
# Client-side animation (Plotly frames) — renders ONCE, no Streamlit reruns,
# so the map never flashes. Play/pause + scrub are native Plotly controls.
# ─────────────────────────────────────────────────────────────────────────────

def _ds(arr, n=220):
    """Downsample a 1-D array to at most n points (keep last)."""
    if len(arr) <= n:
        return list(arr)
    step = len(arr) // n + 1
    out = list(arr[::step])
    if out[-1] != arr[-1]:
        out.append(arr[-1])
    return out


def _annotation(text: str, color: str) -> dict:
    return dict(text=text, x=0.5, y=0.98, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="white"),
                bgcolor=color, borderpad=6, opacity=0.92)


def _anim_layout(fig, route, frames, extra=(None, None), frame_ms=160, zoom=8):
    """Mise en page animée : boutons de VITESSE visibles (foncés) + curseur de défilement."""
    clat, clon = _center(route, *extra)

    def play(label, dur):
        return dict(label=label, method="animate",
                    args=[None, {"frame": {"duration": int(dur), "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}])
    pause = dict(label="⏸", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}])
    # Contrôle de la vitesse : ralentir (×0.5) … accélérer (×4)
    speed_buttons = [pause,
                     play("× 0.5", frame_ms * 2),
                     play("× 1", frame_ms),
                     play("× 2", frame_ms / 2),
                     play("× 4", frame_ms / 4)]
    steps = [dict(method="animate", label="",
                  args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}]) for f in frames]
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=clat, lon=clon), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0), height=560, uirevision="keep",
        legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.01,
                    bgcolor="rgba(15,23,42,0.75)", font=dict(color="white", size=11)),
        # Boutons FONCÉS pour rester visibles même quand les tuiles de carte sont claires
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.01, y=0.13, xanchor="left", yanchor="bottom",
                          bgcolor="#0f172a", bordercolor="#334155", borderwidth=1,
                          font=dict(color="white", size=13),
                          pad=dict(l=6, r=6, t=4, b=4), buttons=speed_buttons)],
        sliders=[dict(active=0, x=0.0, len=1.0, y=0, pad=dict(b=8, t=4),
                      steps=steps, currentvalue=dict(visible=False),
                      bgcolor="#cbd5e1", activebgcolor="#2563eb",
                      bordercolor="#334155", tickcolor="#334155")],
    )
    return fig


def build_gps_animation(route: List[Dict], P: Dict, n_frames: int = 90,
                        threshold_s: float = 180.0) -> go.Figure:
    """Animated GPS-fallback replay: tracked path + red dark segment + AI estimate."""
    t0, t1 = float(P["t_unix"][0]), float(P["t_unix"][-1])
    times = np.linspace(t0, t1, n_frames)

    def traces_at(sim):
        pos = position_at(P, route, sim, threshold_s)
        i = pos["last_idx"]
        path = go.Scattermapbox(
            lat=_ds(P["lat"][:i + 1]), lon=_ds(P["lon"][:i + 1]), mode="lines",
            line=dict(width=5, color="#2563eb"), name="Trajet suivi (GPS)", hoverinfo="skip")
        if pos["dark"]:
            rlat, rlon = route_segment(route, pos["s_last_km"], pos["s_est_km"])
            red = go.Scattermapbox(lat=rlat, lon=rlon, mode="lines",
                                   line=dict(width=7, color="#dc2626"),
                                   name="Signal perdu (estimé)", hoverinfo="skip")
            clat, clon = circle_lonlat(pos["lat"], pos["lon"], pos["uncertainty_m"])
            circ = go.Scattermapbox(lat=clat, lon=clon, mode="lines", fill="toself",
                                    fillcolor="rgba(220,38,38,0.15)",
                                    line=dict(width=1, color="rgba(220,38,38,0.5)"),
                                    name="Incertitude", hoverinfo="skip")
            lastk = go.Scattermapbox(lat=[float(P["lat"][i])], lon=[float(P["lon"][i])],
                                     mode="markers", marker=dict(size=12, color="#64748b"),
                                     name="Dernier point connu", hoverinfo="name")
            bus = _bus_marker(pos["lat"], pos["lon"], "Bus (estimation IA)", "#dc2626")
        else:
            red = go.Scattermapbox(lat=[None], lon=[None], mode="lines",
                                   name="Signal perdu (estimé)", hoverinfo="skip")
            circ = go.Scattermapbox(lat=[None], lon=[None], mode="lines",
                                    name="Incertitude", hoverinfo="skip")
            lastk = go.Scattermapbox(lat=[None], lon=[None], mode="markers",
                                     name="Dernier point connu", hoverinfo="skip")
            bus = _bus_marker(pos["lat"], pos["lon"], "Bus (GPS direct)", "#2563eb")
        return [_route_traces(route)[0], _stops_trace(route), path, red, circ, lastk, bus], pos

    def annot(pos, sim):
        ts = pd.Timestamp(sim, unit="s").strftime("%H:%M:%S")
        if pos["dark"]:
            return _annotation(
                f"⚠️ SIGNAL PERDU — {pos['dt_dark_s']/60:.0f} min sans signal · "
                f"est. ±{pos['uncertainty_m']:.0f} m · {ts}", "#dc2626")
        return _annotation(f"● GPS DIRECT · {ts}", "#16a34a")

    base, pos0 = traces_at(times[0])
    frames = []
    for k, sim in enumerate(times):
        trs, pos = traces_at(sim)
        frames.append(go.Frame(data=trs, name=str(k),
                               layout=go.Layout(annotations=[annot(pos, sim)])))
    fig = go.Figure(data=base, frames=frames)
    fig.update_layout(annotations=[annot(pos0, times[0])])
    return _anim_layout(fig, route, frames)


def build_eta_animation(route: List[Dict], P: Dict, start_unix: float,
                        rider_route_seq: int, rider_eta_unix: Optional[float],
                        rider_name: str = "", n_frames: int = 110) -> go.Figure:
    """ETA en direct animée : le bus parcourt TOUT le trajet (diffusion complète)."""
    t0 = float(P["t_unix"][0]); t1 = float(P["t_unix"][-1])
    # Diffuser la course entière : du tout début jusqu'à la fin du trajet.
    times = np.linspace(t0, t1, n_frames)
    rider = next((r for r in route if int(r["seq"]) == int(rider_route_seq)), None)

    def traces_at(sim):
        pos = position_at(P, route, sim)
        seg = go.Scattermapbox(lat=[None], lon=[None], mode="lines",
                               line=dict(width=6, color="#2563eb"), name="Bus → vous",
                               hoverinfo="skip")
        if rider:
            rlat, rlon = route_segment(route, pos["s_est_km"], rider["s_m"])
            seg = go.Scattermapbox(lat=rlat, lon=rlon, mode="lines",
                                   line=dict(width=6, color="#2563eb"),
                                   name="Bus → vous", hoverinfo="skip")
        rider_tr = go.Scattermapbox(
            lat=[rider["lat"]] if rider else [None], lon=[rider["lon"]] if rider else [None],
            mode="markers+text", marker=dict(size=20, color="#16a34a"),
            text=["🧍"], textfont=dict(size=15), textposition="middle center",
            hovertext=[f"Votre arrêt · {rider_name}"], hoverinfo="text", name="Votre arrêt")
        bus = _bus_marker(pos["lat"], pos["lon"], "Bus")
        return [_route_traces(route)[0], seg, _stops_trace(route), rider_tr, bus], pos

    def annot(sim):
        ts = pd.Timestamp(sim, unit="s").strftime("%H:%M:%S")
        if rider_eta_unix:
            mins = (rider_eta_unix - sim) / 60
            if abs(mins) <= 0.3:
                return _annotation(f"🎉 Bus à {rider_name} · {ts}", "#16a34a")
            if mins < 0:
                return _annotation(f"🚌 Bus déjà passé à {rider_name} · {ts}", "#64748b")
            return _annotation(f"🚌 Arrive à {rider_name} dans {mins:.0f} min · {ts}", "#2563eb")
        return _annotation(f"🚌 {ts}", "#2563eb")

    base, _ = traces_at(times[0])
    frames = [go.Frame(data=traces_at(s)[0], name=str(k),
                       layout=go.Layout(annotations=[annot(s)]))
              for k, s in enumerate(times)]
    fig = go.Figure(data=base, frames=frames)
    fig.update_layout(annotations=[annot(times[0])])
    extra = (rider["lat"], rider["lon"]) if rider else (None, None)
    return _anim_layout(fig, route, frames, extra=extra)
