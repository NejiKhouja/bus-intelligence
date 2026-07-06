# Web services needed from the WiniCari platform team

This is the reverse of `docs/PHP_INTEGRATION.md` (which documents how the PHP platform
calls the AI API). This doc specifies what **you need from them** — two narrow,
purpose-built web services, in their own existing style (`/Service/getXxx?param=value`,
same shape as `getServiceSoc`) — so the AI layer can detect anomalies on data newer than
its last offline training run, without needing standing access to their MongoDB.

Every field listed below was verified directly against this codebase's own query code
(`src/data/foundation.py::load_pings`, `src/data/reference_db.py::populate_tickets_daily`)
— not guessed. If their schema has since changed, that's the first thing to check against
these exact functions.

## Why two separate services, not one

The AI layer runs two independent anomaly detectors on two different data sources —
they don't share fields, so they need two separate feeds:

| | GPS/trip anomaly | Ticket/billing anomaly |
|---|---|---|
| Grain | one trip | one (societe, line, bus, day) |
| Detects | signal loss, stuck at a stop, off-route, trip too long/short | ticket volume/revenue that doesn't match normal for that line, suspicious average fare |
| Needs data from | `Historique_pos` | `winicari.details` |

Neither needs anything beyond what's listed below — in particular, **no ticket/passenger
detail beyond a daily count and a total revenue figure**, and **no other collections**
(not `winicari.station`, not individual `Ticket{year}` records, nothing route/company
metadata — the AI layer already has that, refreshed periodically offline, not live).

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
| `service.voyage` | int, nullable | increments per trip leg; even = ALLER, odd = RETOUR — **this is the trip-boundary signal**, see below |
| `service.codeLigne` | string | line code |
| `service.societe` | string | company name |
| `bus.code` | string/int | bus identifier |

**Suggested endpoint**, matching their existing convention:
```
GET /Service/getRecentPings?since=2026-07-06T14:00:00
```
Optionally scoped further if they'd rather limit exposure to a pilot company/line first:
```
GET /Service/getRecentPings?societe=S.R.T.K&line=217&minutes=60
```

**Suggested response shape** — an array, one object per ping, same fields as above:
```json
[
  {"date": "2026-07-06T14:03:12", "lat": 36.123, "lon": 10.456, "speed": 42.0,
   "voyage": 7, "codeLigne": "217", "societe": "S.R.T.K", "bus": "6037"}
]
```

**How this gets used**: for each `bus`, the AI layer remembers the last `voyage` number
it's seen. When a poll returns a higher `voyage` for that bus, the *previous* voyage's
pings are a finished trip — they get run through the same reconstruction pipeline
`build_foundation.py` already uses offline, then scored by the trained anomaly models.
No retraining happens on each poll, only inference.

**What this does NOT need**: no historical backfill (only pings from `since` onward),
no other GPS fields (device battery, heading, etc. if present — harmless if included,
just unused), no write access, no other collections.

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

**Suggested endpoint**:
```
GET /Service/getRecentTicketTotals?since=2026-07-05
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

**A framing note on "real-time"**: because this is a *daily* aggregate, it doesn't
support instant, mid-day alerting the way the GPS feed does — a bus-day's totals are
only meaningful once enough of that day has actually happened. This one fits an
end-of-day or next-morning check better than a live popup.

---

## What to ask for, and how to frame it

Given they've been cautious about data access generally, two things make this an easier
ask than "give us access to your database":

1. **It's a small, explicit field list**, not a schema or connection string — copy the
   two tables above directly into the request to their dev team.
2. **They control it entirely** — a narrow, purpose-built endpoint they can rate-limit,
   revoke, or restrict to specific companies/lines at any time, unlike a standing DB
   credential.

Suggested rollout: pilot with one company first (`S.R.T.K` has the most complete
anomaly models on both sides) rather than asking for all companies/lines at once.

## Once you have a URL for either of these

**Ticket anomalies are ready today.** `/api/ticket-anomaly/score-live` already accepts
pre-fetched rows directly in its request body (`{"rows": [{societe, line, bus, day,
nbr_ticket, recette}, ...]}`) — it never talks to MongoDB itself. Whatever calls their
new web service (your own glue code, or a small script) just reshapes the response into
that shape and POSTs it. Verified working end-to-end with real trained models.

**GPS anomalies need one more code change first.** `/api/anomaly/score-live` currently
still calls MongoDB internally (`get_db("Historique_pos")`) rather than accepting pings
in the request body — it was written for the earlier "your server has direct Mongo
access" deployment shape, before this pull-a-web-service plan existed. Once web service 1
above is real, that endpoint needs a small rework to accept `pings: [...]` directly
instead of querying Mongo itself, mirroring what `/api/ticket-anomaly/score-live` already
does. Flagging this now so it isn't assumed done — ask for this change when you're ready
to wire up the GPS side.
