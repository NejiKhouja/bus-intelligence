"""GPS trip reconstruction — the shared foundation layer.

Turns raw GPS pings (`Historique_pos`) into reconstructed trips with a derived actual
arrival time per stop. Used by Delay prediction (labels), Anomaly detection and GPS
Fallback. This module is the single source of truth; the notebook and the batch CLI
(`build_foundation.py`) both import from here.

How the segmentation is made (the 4-step chain)
------------------------------------------------
1. CLEAN  (`clean_pings`)
   Drop consecutive identical-coordinate pings (a parked bus still pings every ~5 s, ~10%
   of rows), keeping the FIRST contact so arrival timing is preserved. Annotate the time
   gap between pings and flag `signal_gap` where the bus went quiet.

2. MAP-MATCH  (`project_to_route`)
   Project every ping onto the line's anchor polyline to get `s` = distance along the route
   in metres (then smoothed). The match is *windowed-sequential* — each ping is matched only
   near the previous ping's segment — which stops `s` from jumping backward when anchors are
   sparse. This turns a messy lat/lon track into ONE clean number that rises as the bus
   heads toward the far terminus and falls on the way back.

3. SEGMENT  (`segment_trips`) — watch `s` over time:
   - A TURNAROUND = `s` reverses by more than a hysteresis threshold
     (`reversal_frac * route_len`, so it scales: large for a 192 km line, small for a 6 km
     loop). Each stretch between turnarounds is a trip; direction = ALLER if `s` rises,
     RETOUR if it falls.
   - A run is split ONLY at a *parked layover* (a long time-gap where `s` barely changed),
     so a mid-route SIGNAL gap does not fake a new trip.
   - Trips shorter than `min_span` / `min_trip_min` are dropped; each is tagged `full`
     (spans both route ends) or PARTIAL (bus turned back early, or the day ended mid-run).

4. ARRIVALS  (`derive_arrivals`)
   For each trip, snap the stops in its covered range to the nearest ping, in travel order
   with monotonically increasing times; `matched` flags whether the bus passed within
   `arrival_thresh_m` (350 m). Match rate per line is the headline data-quality signal.

What this layer does and does NOT compute yet
---------------------------------------------
DONE   - actual ARRIVAL time per stop (`arrival`), trip structure, signal gaps.
DONE   - STOPPAGE / DWELL per stop (`departure`, `dwell_s`): arrival = first ping within
         range, departure = last consecutive ping still within range before the bus moves
         on, dwell_s = their gap. A long dwell is a strong anomaly signal (breakdown /
         incident / unscheduled stop). NOTE: requires re-running build_foundation to appear
         in the persisted dataset.
TODO   - DELAY: delay = actual - scheduled arrival. We have `actual` but there are no
         per-stop scheduled times — `ligne.horaires` only stores DEPARTURE times at the
         origin, not a per-stop timetable. Built in 03_delay against a DATA-DRIVEN baseline
         (median observed elapsed-to-stop) rather than an official schedule.
This module is the prerequisite LAYER; delay (03_delay) is the next layer on top of it.

Assumptions / limits
---------------------
- Line geometry is the ordered `array_lat/lng_opendata` anchors with `0.0` placeholders
  dropped; routes need >= `min_anchors` real anchors. With sparse anchors the polyline is
  a coarse approximation of the road, which is why segmentation uses distance *swings*
  with hysteresis rather than exact map-matching.
- `code` is not unique across companies; always key by `(code, societe)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pymongo.database import Database


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    # geometry
    min_anchors: int = 4              # min real anchor stops for a line to be usable
    # candidate selection
    min_pings: int = 300              # min pings for a (day, line, bus) to be worth running
    first_usable_day: str = "d20220601"   # route-link (service.codeLigne) starts ~here
    # cleaning
    dedup_round: int = 6              # coord rounding for stationary-duplicate removal
    signal_gap_s: int = 600           # annotate gaps larger than this
    # projection (map-matching)
    proj_window: int = 3              # sequential search window (segments) around last match
    proj_gap_reset_s: int = 900       # after a gap larger than this, re-search globally
    smooth_window: int = 15           # rolling-median window on distance-along-route
    # segmentation (scale-invariant in route length)
    reversal_frac: float = 0.15       # turnaround hysteresis as fraction of route length
    reversal_floor_m: float = 2000.0
    min_span_frac: float = 0.06       # min trip length as fraction of route length
    min_span_floor_m: float = 1500.0
    min_trip_min: float = 8.0         # min trip duration
    layover_gap_s: int = 2400         # split a run only at gaps >= this ...
    park_frac: float = 0.05           # ... and only if the bus barely moved across the gap
    full_frac: float = 0.10           # trip is "full" if it spans both route ends within this band
    # arrival snapping
    arrival_thresh_m: float = 350.0   # max ping-to-stop distance to count as an arrival

    @property
    def out_columns(self) -> list:
        return ["day", "line", "societe", "bus", "trip_id", "dir", "full",
                "trip_start", "trip_end", "seq", "route_seq", "stop",
                "arrival", "departure", "dwell_s", "dist_m", "matched"]


# --------------------------------------------------------------------------- geometry
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (scalars or numpy arrays)."""
    R = 6371000.0
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def real_anchor_stops(lg: dict) -> list:
    """Geocoded stops only, in route order, keeping each stop's original route position."""
    la = lg.get("array_lat_opendata") or []
    lo = lg.get("array_lng_opendata") or []
    names = lg.get("stationnames") or []
    rows = []
    for i in range(min(len(la), len(lo))):
        try:
            lat, lon = float(la[i]), float(lo[i])
        except (TypeError, ValueError):
            continue
        if abs(lat) > 1 and abs(lon) > 1:                 # drop 0.0 placeholders
            rows.append({"route_seq": i,
                         "name": names[i] if i < len(names) else f"stop{i}",
                         "lat": lat, "lon": lon})
    return rows


def stops_frame(lg: dict) -> pd.DataFrame:
    """Anchor stops with a compact `seq` and cumulative distance-along-route `s_m`."""
    rows = real_anchor_stops(lg)
    st = pd.DataFrame(rows)
    seg = haversine(st["lat"].values[:-1], st["lon"].values[:-1],
                    st["lat"].values[1:], st["lon"].values[1:])
    st["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
    st.insert(0, "seq", range(len(st)))
    return st


def usable_geometry(lg: dict, cfg: Config) -> bool:
    return len(real_anchor_stops(lg)) >= cfg.min_anchors


def build_usable_lines(db: Database, cfg: Config) -> dict:
    """Map {(code, societe) -> stops_frame} for every usable line (queried once)."""
    out = {}
    for lg in db["ligne"].find({}):
        if usable_geometry(lg, cfg):
            out[(str(lg["code"]), lg.get("societe"))] = stops_frame(lg)
    return out


# --------------------------------------------------------------------------- pings
def load_pings(gps_db: Database, day: str, line: str, bus) -> pd.DataFrame:
    """One bus-day for one line. Filtered + projected query = memory-safe.

    Speed: prefer top-level `speed`; fall back to `bus.vitesse` only when absent (in 2025+
    data `bus.vitesse` is often a stale 0, so `speed` is authoritative when present).
    """
    cur = gps_db[day].find(
        {"service.codeLigne": line, "bus.code": bus},
        {"date": 1, "localisation": 1, "speed": 1, "bus.vitesse": 1, "_id": 0},
    )
    rows = []
    for d in cur:
        loc = d.get("localisation") or {}
        b = d.get("bus") or {}
        speed = d.get("speed")
        if speed is None:
            speed = b.get("vitesse")
        rows.append({"t": d.get("date"), "lat": loc.get("x"),
                     "lon": loc.get("y"), "speed": speed})
    g = pd.DataFrame(rows)
    if len(g) == 0 or "t" not in g:
        return pd.DataFrame(columns=["t", "lat", "lon", "speed", "gap_s", "signal_gap"])
    return g.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)


def clean_pings(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop consecutive identical-coordinate pings (stationary spam), keeping the first
    contact so arrival timing is preserved; annotate `gap_s` and `signal_gap`."""
    if len(g) == 0:
        return g
    r = cfg.dedup_round
    same = (g["lat"].round(r).eq(g["lat"].round(r).shift())
            & g["lon"].round(r).eq(g["lon"].round(r).shift()))
    g = g[~same].reset_index(drop=True)
    g["gap_s"] = g["t"].diff().dt.total_seconds()
    g["signal_gap"] = g["gap_s"] > cfg.signal_gap_s
    return g


# --------------------------------------------------------------------------- projection
def project_to_route(g: pd.DataFrame, stops: pd.DataFrame, cfg: Config):
    """Sequential (windowed) map-match: distance-along-route `s` and off-route `off`.

    Constraining each ping's matched segment to a small window around the previous match
    keeps `s` physically smooth despite sparse anchors. Returns (g_with_s_off, route_len_m).
    """
    slat, slon = stops["lat"].values, stops["lon"].values
    cum = stops["s_m"].values
    mlat = 111320.0
    mlon = 111320.0 * np.cos(np.radians(slat.mean()))
    ax, ay = slon[:-1] * mlon, slat[:-1] * mlat
    bx, by = slon[1:] * mlon, slat[1:] * mlat
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    L2[L2 == 0] = 1e-9
    nseg = len(ax)

    px, py = g["lon"].values * mlon, g["lat"].values * mlat
    gap = g["gap_s"].values
    s = np.empty(len(g))
    off = np.empty(len(g))
    s_raw = np.empty(len(g))
    prevk: Optional[int] = None
    for i in range(len(g)):
        if prevk is None or (not np.isnan(gap[i]) and gap[i] > cfg.proj_gap_reset_s):
            cand = range(nseg)
        else:
            cand = range(max(0, prevk - cfg.proj_window), min(nseg, prevk + cfg.proj_window + 1))
        best_k, best_d, best_t = 0, np.inf, 0.0
        for k in cand:
            t = min(1.0, max(0.0, ((px[i] - ax[k]) * vx[k] + (py[i] - ay[k]) * vy[k]) / L2[k]))
            dx = px[i] - (ax[k] + t * vx[k])
            dy = py[i] - (ay[k] + t * vy[k])
            d = np.hypot(dx, dy)
            if d < best_d:
                best_d, best_k, best_t = d, k, t
        prevk = best_k
        s_raw[i] = cum[best_k] + best_t * (cum[best_k + 1] - cum[best_k])
        off[i] = best_d
    g = g.copy()
    g["s_raw"] = s_raw
    g["s"] = pd.Series(s_raw).rolling(cfg.smooth_window, center=True, min_periods=1).median()
    g["off_m"] = off
    return g, float(cum[-1])


# --------------------------------------------------------------------------- segmentation
def segment_trips(g: pd.DataFrame, route_len: float, cfg: Config) -> pd.DataFrame:
    """Direction-swing segmentation. Captures full and partial trips; splits a run at a
    gap only when the bus was parked across it (a real between-run layover)."""
    if len(g) < 5 or route_len <= 0:
        return pd.DataFrame()
    rev = max(cfg.reversal_floor_m, cfg.reversal_frac * route_len)
    min_span = max(cfg.min_span_floor_m, cfg.min_span_frac * route_len)
    park = cfg.park_frac * route_len
    s = g["s"].values
    s_raw = g["s_raw"].values
    tm = g["t"].values
    gp = g["gap_s"].values
    N = len(g)

    # zigzag pivots with hysteresis on distance-along-route
    piv = [0]
    direction = 0
    ext = 0
    for i in range(1, N):
        if direction >= 0 and s[i] > s[ext]:
            ext = i
        elif direction <= 0 and s[i] < s[ext]:
            ext = i
        if direction >= 0 and s[i] <= s[ext] - rev:
            piv.append(ext); direction = -1; ext = i
        elif direction <= 0 and s[i] >= s[ext] + rev:
            piv.append(ext); direction = 1; ext = i
        elif direction == 0:
            if s[i] >= s[0] + rev:
                direction = 1; ext = i
            elif s[i] <= s[0] - rev:
                direction = -1; ext = i
    piv.append(N - 1)
    piv = sorted(set(piv))

    trips = []
    for a, b in zip(piv[:-1], piv[1:]):
        # split a same-direction run only at parked layover gaps
        cuts = [a] + [i for i in range(a + 1, b + 1)
                      if gp[i] > cfg.layover_gap_s and abs(s_raw[i] - s_raw[i - 1]) < park] + [b]
        for sa, se in zip(sorted(set(cuts))[:-1], sorted(set(cuts))[1:]):
            if se <= sa:
                continue
            span = abs(s[se] - s[sa])
            dur = (pd.Timestamp(tm[se]) - pd.Timestamp(tm[sa])).total_seconds() / 60
            if span < min_span or dur < cfg.min_trip_min:
                continue
            lo, hi = float(min(s[sa], s[se])), float(max(s[sa], s[se]))
            trips.append({
                "dir": "ALLER" if s[se] > s[sa] else "RETOUR",
                "start": pd.Timestamp(tm[sa]), "end": pd.Timestamp(tm[se]),
                "s_lo": lo, "s_hi": hi,
                "full": lo <= cfg.full_frac * route_len and hi >= (1 - cfg.full_frac) * route_len,
            })
    return pd.DataFrame(trips).reset_index(drop=True)


# --------------------------------------------------------------------------- arrivals
def derive_arrivals(g: pd.DataFrame, trip: pd.Series, stops: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Snap stops within the trip's covered range to pings, in travel order, enforcing
    monotonic arrival times. Returns one row per covered stop (matched or not).

    Also derives STOPPAGE/DWELL: `arrival` = first ping within range, `departure` = last
    consecutive ping still within range before the bus moves on, `dwell_s` = their gap. A
    long dwell is a strong anomaly signal (breakdown / incident / unscheduled stop).
    """
    seg = g[(g["t"] >= trip["start"]) & (g["t"] <= trip["end"])]
    if len(seg) == 0:
        return pd.DataFrame()
    lat = seg["lat"].values
    lon = seg["lon"].values
    t = seg["t"].values
    margin = cfg.arrival_thresh_m
    covered = stops[(stops["s_m"] >= trip["s_lo"] - margin) & (stops["s_m"] <= trip["s_hi"] + margin)]
    order = covered.sort_values("s_m", ascending=(trip["dir"] == "ALLER"))

    out = []
    ptr = 0                                   # enforce monotonic arrivals along the trip
    for _, st in order.iterrows():
        if ptr >= len(seg):
            d_arr, j_local, matched = np.inf, None, False
        else:
            d = haversine(lat[ptr:], lon[ptr:], st["lat"], st["lon"])
            j_local = int(np.argmin(d)) + ptr
            d_arr = float(d.min())
            matched = d_arr <= cfg.arrival_thresh_m

        departure, dwell_s = pd.NaT, None
        if matched:
            # departure = last *consecutive* ping still within range of this stop
            d_fwd = haversine(lat[j_local:], lon[j_local:], st["lat"], st["lon"])
            within = d_fwd <= cfg.arrival_thresh_m
            last = 0
            for k in range(len(within)):
                if within[k]:
                    last = k
                else:
                    break
            dep_idx = j_local + last
            departure = pd.Timestamp(t[dep_idx])
            dwell_s = round(float((t[dep_idx] - t[j_local]) / np.timedelta64(1, "s")), 1)
            ptr = dep_idx + 1
        out.append({
            "seq": int(st["seq"]),
            "route_seq": int(st["route_seq"]),
            "stop": str(st["name"])[:24],
            "arrival": pd.Timestamp(t[j_local]) if matched else pd.NaT,
            "departure": departure,
            "dwell_s": dwell_s,
            "dist_m": int(d_arr) if np.isfinite(d_arr) else None,
            "matched": bool(matched),
        })
    return pd.DataFrame(out)


def reconstruct_bus_day(gps_db: Database, day: str, line: str, societe, bus,
                        stops: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Full pipeline for one (day, line, societe, bus). Returns stop-arrival rows."""
    g = clean_pings(load_pings(gps_db, day, line, bus), cfg)
    if len(g) < 20:
        return pd.DataFrame()
    g, route_len = project_to_route(g, stops, cfg)
    trips = segment_trips(g, route_len, cfg)
    frames = []
    for tid, tr in trips.iterrows():
        a = derive_arrivals(g, tr, stops, cfg)
        if a.empty:
            continue
        a.insert(0, "day", day[1:]); a.insert(1, "line", line)
        a.insert(2, "societe", societe); a.insert(3, "bus", bus)
        a.insert(4, "trip_id", tid); a.insert(5, "dir", tr["dir"])
        a.insert(6, "full", bool(tr["full"]))
        a.insert(7, "trip_start", tr["start"]); a.insert(8, "trip_end", tr["end"])
        frames.append(a)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- candidates
def candidates_for_day(gps_db: Database, day: str, usable: dict, cfg: Config) -> list:
    """Distinct (day, line, societe, bus) active that day on usable-geometry lines."""
    pipe = [{"$group": {"_id": {"l": "$service.codeLigne", "s": "$service.societe",
                                "b": "$bus.code"}, "n": {"$sum": 1}}}]
    out = []
    for a in gps_db[day].aggregate(pipe):
        l, s, b = a["_id"].get("l"), a["_id"].get("s"), a["_id"].get("b")
        if b is not None and a["n"] >= cfg.min_pings and (str(l), s) in usable:
            out.append((day, str(l), s, b))
    return out


def gps_days(gps_db: Database, cfg: Config) -> list:
    """Sorted daily GPS collections from the first route-linked day onward."""
    return sorted(x for x in gps_db.list_collection_names()
                  if re.fullmatch(r"d\d{8}", x) and x >= cfg.first_usable_day)
