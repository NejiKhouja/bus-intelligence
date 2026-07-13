# Web services needed from the WiniCari platform team

This is the reverse of `docs/PHP_INTEGRATION.md` (which documents how the PHP platform
calls the AI API). This doc specifies what **you need from them** — three narrow,
purpose-built web services, in their own existing style (`/Service/getXxx?param=value`,
same shape as `getServiceSoc`) — so the AI layer can detect anomalies on data newer than
its last offline training run, without needing standing access to their MongoDB.

Every field listed below was verified directly against this codebase's own query code
(`src/data/foundation.py::load_pings`, `src/data/reference_db.py::populate_tickets_daily`,
`src/data/reference_db.py::populate_tickets_station_daily`) — not guessed, and for service
3 specifically checked against real documents pulled from the collection, not just field
names. If their schema has since changed, that's the first thing to check against these
exact functions.

**Important constraint on freshness**: `Historique_pos` is written to once nightly, not
continuously through the day — a given day's pings only exist once that night's batch
job has run. That means there is no version of this feature that alerts on a trip as
it's happening; the realistic cadence is **once a day, reviewing yesterday's trips each
morning**, not a live/instant popup. This shapes all three web services below: they only
need to be checked once a day, not polled frequently.

## Why three separate services, not one

The AI layer runs three independent anomaly detectors on three different data sources —
they don't share fields, so they need three separate feeds:

| | GPS/trip anomaly | Ticket/billing anomaly (per bus-day) | Ticket/billing anomaly (per station) |
|---|---|---|---|
| Grain | one trip | one (societe, line, bus, day) | one (societe, line, station, day) |
| Detects | signal loss, stuck at a stop, off-route, trip too long/short | ticket volume/revenue that doesn't match normal for that line, suspicious average fare | same, but pinpoints WHICH stop along the line is behaving oddly, with a map |
| Needs data from | `Historique_pos` | `winicari.details` | `Historique_Tickets.Ticket{year}` (individual tickets — this is the one collection service 2 explicitly does NOT need, see below) |

Services 1 and 2 need nothing beyond what's listed under them — in particular, **no
ticket/passenger detail beyond a daily count and a total revenue figure** for service 2,
and **no other collections** (not `winicari.station`, not individual `Ticket{year}`
records, nothing route/company metadata — the AI layer already has that, refreshed
periodically offline, not live). Service 3 is the one exception — it specifically needs
the individual ticket records, because that's the only place the ORIGIN STOP of each
ticket is recorded; explained in its own section below.

---

## Web service 1 — recent GPS pings

**Source collection**: `Historique_pos`, one collection per calendar day named
`d{YYYYMMDD}` (e.g. `d20260706`).

**Exact fields needed**, per ping (verified against `load_pings()`):

| Field (their schema) | Type | Notes |
|---|---|---|
| `date` | datetime | ping timestamp |
| `localisation.x` | float | **latitude** (their field is literally named `x`, not `lat`) |
| `localisation.y` | float | **longitude** (named `y`, not `lon`) |
| `speed` | float, nullable | preferred speed source |
| `bus.vitesse` | float, nullable | fallback only — recent (2025+) data often has this stuck at 0, `speed` wins when both are present |
| `service.voyage` | int, nullable | even = ALLER, odd = RETOUR — used to split each day into individual trips, same as the offline pipeline already does |
| `service.codeLigne` | string | line code |
| `service.societe` | string | company name |
| `bus.code` | string/int | bus identifier |

**Suggested endpoint**, matching their existing convention — a whole day at a time,
since that's the actual grain the data exists at:
```
GET /Service/getPingsForDay?day=20260706
```
Optionally scoped further if they'd rather limit exposure to a pilot company/line first:
```
GET /Service/getPingsForDay?day=20260706&societe=S.R.T.K&line=217
```
A lighter companion worth asking for alongside it — checking every day for a batch that
usually only finishes once a night is wasted effort on both sides:
```
GET /Service/isDayReady?day=20260706   -> {"ready": true}
```
(or simply ask what time the nightly job usually finishes, and call `getPingsForDay`
once shortly after — either works, `isDayReady` is just cleaner).

**Suggested response shape** — an array, one object per ping, same fields as above:
```json
[
  {"date": "2026-07-06T14:03:12", "lat": 36.123, "lon": 10.456, "speed": 42.0,
   "voyage": 7, "codeLigne": "217", "societe": "S.R.T.K", "bus": "6037"}
]
```

**How this gets used**: once a full day's pings are available, they go through the exact
same reconstruction pipeline `build_foundation.py` already uses offline (trip
segmentation, stop matching, dwell/gap detection — nothing new to build) then get scored
by the already-trained anomaly models. No retraining happens, only inference, once a
day. This is simpler than a streaming design would have been, precisely because the
underlying data isn't a stream.

**What this does NOT need**: no historical backfill beyond the requested day, no other
GPS fields (device battery, heading, etc. if present — harmless if included, just
unused), no write access, no other collections.

---

## Web service 2 — recent ticket/billing totals

**Source collection**: `winicari.details` (daily aggregates — **not** the ~5.4M
individual `Ticket{year}` records, which are a completely different, much larger
collection this does NOT need).

**Exact fields needed** (verified against `reference_db.py::populate_tickets_daily`):

| Field (their schema) | Type | Notes |
|---|---|---|
| `societe` | string | company name |
| `CodeLigne` | string | line code (note capitalization — differs from the GPS side's `codeLigne`) |
| `codeBus` | string/int | bus identifier |
| `date` | string, `"YYYY/MM/DD"` | day this total covers |
| `nbrTicket` | int | tickets sold that bus-day |
| `recette` | float | total revenue collected that bus-day |

One real quirk worth telling them about: their own `details` collection has raw
duplicate documents on `(societe, CodeLigne, codeBus, date)` — our offline pipeline sums
them. Either they can hand over pre-summed totals per bus-day, or hand over the raw
matching rows and let the AI layer sum them (already has the logic) — either is fine,
just needs to be clear which one they're giving so it isn't double-counted.

**Suggested endpoint** — same daily-grain framing as the GPS side, one day at a time:
```
GET /Service/getTicketTotalsForDay?day=2026-07-05
```

**Suggested response shape**:
```json
[
  {"societe": "S.R.T.K", "CodeLigne": "220", "codeBus": "6041",
   "date": "2026/07/05", "nbrTicket": 154, "recette": 3175.8}
]
```

**Why this is a smaller, easier ask than it might sound**: this is exactly the kind of
reassurance worth leading with when you ask — it's a daily total, not individual
passenger/seat/payment records. Nothing here identifies a rider or a transaction.

**A framing note on "real-time"**: this is a *daily* aggregate, and a bus-day's totals
are only meaningful once the day is actually over — so it was never going to support
mid-day alerting even on its own. That said, it turns out to match the GPS side exactly:
`Historique_pos` (web service 1) is also only written to once nightly, so both signals
land on the same cadence. See the freshness note at the top of this document — the whole
feature is a once-a-day, next-morning review, not a live popup either way.

---

## Web service 3 — recent individual ticket records (per-station detection)

**Why this one is different from service 2**: `winicari.details` (service 2) is already
aggregated per bus-day, with no stop information at all — there is no way to tell WHICH
station along a line sold an unusual number of tickets from that collection alone. Only
the individual ticket record carries the origin stop, so this service asks for those
directly, one day at a time, same framing as the other two.

**Source collection**: `Historique_Tickets.Ticket{year}` (e.g. `Ticket2026`) — individual
ticket records, ~5.5M total across 2019-2026 in our historical copy, but only a single
day's worth (a few thousand records) is needed per call here.

**Exact fields we use** (verified against real documents pulled from this collection, not
just field names — see `reference_db.py::populate_tickets_station_daily`):

| Field (their schema) | Type | Notes |
|---|---|---|
| `Societe` | string | company name |
| `CodeRoute` | string | **the real line code** — confirmed by inspecting real documents. `Codeligne` (lowercase `l`) also exists on the same document but is a category code (values seen: `"00"`, `"99"`), **not** the line — please don't send that one by mistake |
| `origine` | string | origin stop code for this ticket |
| `NomFR1` | string | origin stop name (human-readable, e.g. `"KASSERINE"`) — this is what we actually group by, `origine` is just context |
| `Prix` | float | fare for this individual ticket |
| `jour_service` | string, `"YYYY/MM/DD"` | **service day** — same format as service 2's `date`. There are several other date-like fields on the same document (`date`, `date_ticket`, `date_service`, `date_debut_service`) that mean subtly different things (ticket issuance timestamp vs. service day) — `jour_service` is the one we use and the one worth confirming stays stable on their end |
| `requisition` | string, `"O"`/`"N"` | **excluded from revenue, but not from the ticket count** — see the reconciliation note below, this one actually matters for getting numbers that match `winicari.details` |

We're fine receiving the **whole raw document** for the day, same as the other two
services — no need to filter down to just these fields on your end, this list is just
what we currently read from it.

**A reconciliation quirk worth flagging to them directly**: summing `Prix` naively across
all tickets for a given (societe, line, bus, day) does **not** match that same bus-day's
`recette` in `winicari.details` (service 2) — verified concretely, not assumed: for one
real bus-day (S.R.T.K/217/bus 6028/2026-06-21), both collections agree on the ticket count
(94 = 94), but summed `Prix` comes to 704.36 DT against a `winicari.details.recette` of
561.26 DT. Isolated the cause: tickets with `requisition = "O"` (15 of the 94, summing to
exactly 143.10 DT) are **counted** in `nbrTicket` on both sides but **excluded** from
`recette` — excluding them from our sum lands on 561.26 DT exactly. We now do this
exclusion in `populate_tickets_station_daily`. This worked perfectly on some bus-days
checked but left a smaller unexplained gap on others (ticket *counts* didn't even match
between the two collections before `requisition` entered into it, on those particular
days) — worth asking them directly what `requisition` represents (a government/
institutional-mandated fare that isn't collected as normal revenue, is our guess) and
whether there's a cleaner way to compute per-ticket "real revenue" than `Prix` alone.

**Suggested endpoint** — same daily-grain framing as the other two:
```
GET /Service/getTicketDetailsForDay?day=2026-07-05
```

**Suggested response shape**: the raw collection for that day, unfiltered.

**Why this is still a reasonable ask despite being individual-ticket-level**: it's still
just fare + origin stop + line/bus/day — no passenger identity, no seat, no payment
method. The same "nothing here identifies a rider" reassurance from service 2 applies
here too.

**Suggested rollout**: same pilot-one-company-first approach as the other two — this one
in particular is worth piloting narrow first since it's a materially larger daily volume
than service 2 (a handful of tickets per stop vs. one row per bus-day).

---

## What to ask for, and how to frame it

Given they've been cautious about data access generally, two things make this an easier
ask than "give us access to your database":

1. **It's a small, explicit field list**, not a schema or connection string — copy the
   three tables above directly into the request to their dev team. (We're fine receiving
   the whole raw document per day rather than a filtered projection — see the "full
   collection" notes under each service — so this is about naming what we read, not
   asking them to build custom filtering.)
2. **They control it entirely** — a narrow, purpose-built endpoint they can rate-limit,
   revoke, or restrict to specific companies/lines at any time, unlike a standing DB
   credential.
3. **It's called once a day, not polled** — all three feeds are worth checking once,
   shortly after their nightly batch finishes, not repeatedly through the day. Worth
   saying explicitly, since "how often will this be hit" is a reasonable thing for them
   to ask.

Suggested rollout: pilot with one company first (`S.R.T.K` has the most complete
anomaly models on all three sides) rather than asking for all companies/lines at once.

## Once you have a URL for any of these

**Ticket anomalies (bus-day) are ready today.** `/api/ticket-anomaly/score-live` already
accepts pre-fetched rows directly in its request body (`{"rows": [{societe, line, bus,
day, nbr_ticket, recette}, ...]}`) — it never talks to MongoDB itself. Whatever calls
their new web service (your own glue code, or a small script) just reshapes the response
into that shape and POSTs it. Verified working end-to-end with real trained models.

**Per-station ticket anomalies (service 3) are trained, but the live-scoring endpoint
doesn't exist yet.** The model itself is real and served today
(`ticket_anomaly.score_stations`/`explain_stations` in `src/models/ticket_anomaly.py`,
plus `/api/ticket-anomaly-stations` for drilling into ALREADY-scored historical days from
the dashboard) — same situation as the GPS side below: `/api/ticket-anomaly/score-live`
only accepts bus-day rows right now, not station rows. It needs the same kind of small
addition (accept a `station` field, route to `score_stations`/`explain_stations` instead
of `score`/`explain`) before service 3 can feed fresh data the same way. Flagging this now
so it isn't assumed done — ask for this change when service 3 has a URL.

**GPS anomalies need one more code change first.** `/api/anomaly/score-live` currently
still calls MongoDB internally (`get_db("Historique_pos")`) rather than accepting pings
in the request body — it was written for the earlier "your server has direct Mongo
access" deployment shape, before this pull-a-web-service plan existed. Once web service 1
above is real, that endpoint needs a small rework to accept `pings: [...]` directly
instead of querying Mongo itself, mirroring what `/api/ticket-anomaly/score-live` already
does. Flagging this now so it isn't assumed done — ask for this change when you're ready
to wire up the GPS side.
