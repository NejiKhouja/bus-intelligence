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
