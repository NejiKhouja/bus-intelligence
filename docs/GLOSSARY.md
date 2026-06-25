# WiniCari AI — Glossary

Plain-language meaning of every term used in the notebooks and in `src/data/*.py`.

## Raw data
- **ping** — one GPS report from a bus: timestamp + lat/lon (+ speed). Sent every ~5 s while a bus is in service. Stored in `Historique_pos` (one collection per day, `dYYYYMMDD`).
- **bus-day** — all the pings of one bus on one line on one day, i.e. one `(day, line, societe, bus)`. The unit of work the pipeline reconstructs.
- **service / run** — a driver "opening" a line on a bus (logged in `service`). It only records the *start*; it does NOT record turnarounds or the return, which is why we rebuild trips from GPS.
- **societe** — the bus company (S.R.T.K, TCV, …). A line `code` is reused across companies, so we always identify a line by **(code, societe)**.

## Geometry (stops & route)
- **anchor stop** — a stop that actually has real coordinates. Many stops are stored as `0.0` (not geocoded) = **placeholders**, which we drop.
- **geocoded** — has a real lat/lon (not `0.0`).
- **usable line** — a line with ≥ 4 anchor stops (enough to trace the route). ~135 of 402 lines qualify.
- **seq** — the stop's position among the *kept anchors*, numbered 0,1,2,… (compact).
- **route_seq** — the stop's *original* position on the full line (before placeholders were dropped). Lets us map back to the real timetable later.
- **route_len** — total length of the line in metres, measured along the anchor polyline.

## Trip reconstruction (02_preprocessing / foundation.py)
- **map-matching / projection** — turning a messy lat/lon track into one number per ping: **`s` = distance along the route (metres)**. `s` rises as the bus heads to the far end, falls on the way back.
- **trip** — one traversal of the route in one direction. Found by watching `s` rise then fall.
- **ALLER / RETOUR** — direction. ALLER = toward the last stop (`s` rising); RETOUR = back toward the first (`s` falling).
- **turnaround / swing** — the moment `s` reverses → the boundary between two trips.
- **hysteresis / reversal threshold** — how big a reversal must be to count as a real turnaround (not GPS wobble). Scaled to route length so it works for a 6 km loop and a 200 km line.
- **full vs partial trip** — *full* spans both ends of the route; *partial* = the bus turned back early or the day ended mid-run (`full=False`).
- **signal gap** — a stretch where the bus stopped sending pings (lost GPS). Flagged, but does NOT split a trip.
- **parked layover** — a long time-gap where the bus barely moved (`s` flat) = a real break between runs; this DOES split trips.

## Arrivals & stoppage
- **matched** — the bus passed within `arrival_thresh_m` (350 m) of the stop, so we trust the arrival.
- **arrival_thresh_m** — the 350 m radius used to decide a stop was reached.
- **match rate** — % of a line's stops that got a matched arrival = the headline **data-quality** signal per line. Low = bad stop coordinates.
- **arrival** — derived actual time the bus reached the stop (the closest ping).
- **departure** — last consecutive ping still within range of the stop before the bus moves on.
- **dwell_s** — seconds the bus *sat* at the stop = departure − arrival. Long dwell = possible breakdown/incident (used by Anomaly).

## Delay (03_delay / delay.py)
- **elapsed-to-stop (`elapsed_min`)** — minutes from the **trip start** to the **arrival** at a stop = `(arrival − trip_start)/60`. It's the *cumulative travel time* from the line's origin to that stop on this run.
- **baseline / expected_min** — the *typical* elapsed-to-stop for a `(societe, line, dir, seq)`, computed as the **median** across all reconstructed trips. This is our stand-in for a timetable, since no official one exists.
- **p10 / p90** — the 10th/90th percentile band around the baseline = the normal spread.
- **delay_min** — `elapsed_min − expected_min`. Positive = slower than usual. ⚠️ This is delay vs the line's *own typical* performance, **not** vs an official schedule.
- **rolling (next-stop) prediction** — predict the delay one stop ahead from the bus's current state; chain it forward for a full ETA.
- **persistence baseline** — the dumb guess "next delay = current delay"; our model must beat it.
- **naive baseline** — the dumb guess "the bus is on time (delay 0)".
- **MAE** — mean absolute error in minutes; lower = better predictions.
- **serve_eta** — the production function: given where the bus is now and how late it is, returns a predicted clock **ETA** for every remaining stop.
- **is_weekend / day-type** — calendar features (Tunisia weekend = Sat/Sun). Weekends run closer to baseline.

## Misc
- **candidate** — a `(day, line, societe, bus)` worth reconstructing (usable line + enough pings).
- **shard** — one monthly parquet file written by the batch build; combined into `foundation_arrivals_full.parquet`.


Geometry / Candidate Filtering

min_anchors: int = 4
A line is only usable if it has at least 4 real geocoded stops in the database. Lines with fewer than 4 stops (usually terminal-to-terminal routes with no intermediate data) are skipped entirely — you can't map-match a bus to a route you can barely describe geometrically.


min_pings: int = 300
A (day, line, bus) combination is skipped if it has fewer than 300 GPS pings. A full day's operation on line 209 produces ~3,900 pings. 300 is the floor below which the data is too sparse to reconstruct meaningful trips — it means the bus was only active for maybe 20–30 minutes total.


first_usable_day: str = "d20220601"
MongoDB collections before this date used an older GPS schema where service.codeLigne didn't exist yet. Everything before June 2022 is excluded.

Cleaning

dedup_round: int = 6
When a bus parks at a terminal, the transponder keeps firing pings at the same coordinate every 30–60 seconds. These are noise — they would create fake "arrivals" at every nearby stop. Rounding to 6 decimal places (~0.1 m precision) and dropping consecutive identical-coordinate pings removes stationary spam while keeping the first contact (which tells you when the bus arrived at that location).


signal_gap_s: int = 600
If two consecutive pings are more than 10 minutes apart, it's flagged as a signal_gap = True. This is just a label — it doesn't drop the data. It tells downstream code "something interrupted the signal here." 10 minutes is the threshold because anything shorter is normal urban traffic (tunnels, bridges) and anything longer is a genuine transponder loss.

Projection (Map-Matching)
The projection step converts raw (lat, lon) GPS coordinates into s_m — a single number representing distance along the route (0 at the first terminal, 192,000 at the last for line 209). This makes every calculation 1-dimensional.


proj_window: int = 3
For each new ping, the algorithm only searches segments ±3 segments around where the previous ping matched. This is critical for efficiency and correctness: without a window, a ping near the middle of a 192km route might match a geometrically similar segment 80km away. The window enforces physical continuity — a bus can't teleport.


proj_gap_reset_s: int = 900
After a signal gap longer than 15 minutes, the bus could be anywhere — the window constraint is no longer valid. So the algorithm resets and searches the entire route globally for that one ping, then resumes windowed matching from there.


smooth_window: int = 15
After projection, the raw s_raw values are noisy (GPS jitter moves the projected position back and forth by 50–200m). A rolling median over 15 pings smooths this into s — the clean distance signal used everywhere else. Median (not mean) is used because it's robust to the occasional wildly mis-matched ping.

Segmentation (Splitting Into Trips)
This is the most complex part. The bus drives ALLER (outbound), reaches the far terminal, waits, then drives RETOUR (return). The segmentation code has to detect direction changes and split the continuous day of pings into individual trip segments.


reversal_frac: float = 0.15
reversal_floor_m: float = 2000.0
These two together form a hysteresis band for detecting direction reversals. A reversal is only declared when the bus has moved at least max(2000 m, 15% × route_length) in the opposite direction from its last peak.

Why hysteresis? Because GPS jitter and stop dwells make the s signal wiggle constantly. Without a minimum reversal distance, the code would split every small wobble into a new "trip." With reversal_frac=0.15 on a 192km route, that minimum is max(2000, 28800) = 28,800 m — a bus must travel at least 28.8 km backwards before a genuine turnaround is declared.


min_span_frac: float = 0.06
min_span_floor_m: float = 1500.0
A segment is only kept as a trip if it spans at least max(1500 m, 6% × route_length) of the route. On a 192km route that's max(1500, 11520) = 11,520 m. This filters out micro-segments caused by a bus making a U-turn in a depot yard or reversing slightly at a stop.


min_trip_min: float = 8.0
Even if a segment passes the distance check, it must also last at least 8 minutes in wall-clock time. A 12km segment done in 3 minutes is physically impossible and is a data artifact.


layover_gap_s: int = 2400
park_frac: float = 0.05
These two work together to decide whether a signal gap should split a trip.

Not every gap splits a trip — the bus might drive through a tunnel for 15 minutes and come out the other side still on the same run. A gap only causes a split when both conditions are true:

The gap is longer than 40 minutes (layover_gap_s=2400)
The bus barely moved across the gap — less than 5% × route_length (on 192km that's 9.6km)
This correctly handles: a bus that disappears for 2 hours at the terminal between runs (split → two trips) vs. a bus that loses signal for 30 minutes mid-route but travels 40km while dark (not split → one trip).


full_frac: float = 0.10
A trip is labeled full=True if both its start and end are within 10% of route length from the respective terminals. On a 192km route, the bus must start within 19.2km of one terminal and end within 19.2km of the other. Partial trips (bus started mid-route, or turned back early) get full=False and are kept in the data but flagged so models can choose to exclude them.

Arrival Snapping

arrival_thresh_m: float = 350.0
After projecting all pings onto the route, each ping is compared to every stop's projected s_m position. A ping is counted as an "arrival" at a stop only if the raw GPS coordinate is within 350 metres of the stop's coordinates in geographic space (not route distance).

350m is a deliberate compromise:

Too tight (e.g. 50m): many real arrivals are missed because GPS drifts 20–100m even when the bus is directly at the stop
Too loose (e.g. 1000m): pings from the next street over start matching stops they didn't visit
On the real data, this setting determines the match_rate column in the foundation — the fraction of stops where the bus came within 350m. Lines in dense urban areas have high match rates (0.85+); rural lines with imprecise stop coordinates have lower rates.