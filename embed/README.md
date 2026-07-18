# WiniCari AI — embeddable anomaly-detection widget

Plain PHP + HTML + CSS + JS (no build step, no framework) meant to be dropped into the
company's own dashboard. Scoped to the anomaly-detection features only — that's all
that's hosted on the Render API (`https://bus-intelligence.onrender.com`).

## Setup

1. `cp config.example.php config.php` and fill in the real `WINICARI_API_KEY` (must match
   the `API_KEY` env var set on the Render service). `config.php` is gitignored — never
   commit the real key.
2. Serve `index.php` behind any PHP-capable web server (Apache, Nginx+PHP-FPM, etc.).
   **Do not** use PHP's built-in dev server (`php -S`) in production — it's single-threaded
   and will stall under concurrent requests; that's fine for local testing only.
3. Set `$_SESSION['winicari_company']` from your own login flow (see `session.php`) so
   each logged-in user only ever sees their own operator's data. Until that's wired in,
   or for testing, the picker screen in `index.php` lets you choose a company manually.

## Architecture

```
Browser (JS)  →  proxy.php (same-origin, holds the API key)  →  Render API
```

The JS never talks to Render directly — `proxy.php` holds `WINICARI_API_KEY` server-side
and forwards whitelisted endpoints only (see `ALLOWED_ENDPOINTS` in `proxy.php`). Two
reasons: the key would otherwise be visible in the browser's page source/devtools, and
this also means there's no CORS setup needed against Render — the browser only ever
talks to its own origin.

Every proxied request is pinned server-side to `$_SESSION['winicari_company']` — a
`societe` query param sent from the client is ignored on purpose, so one company's
session can never read another's data by editing the URL.

## Views (mirrors the Streamlit dashboard's anomaly page, tab for tab)

- **Trajets signalés** — today's live-scored flagged trips (green pulsing badge when the
  webservice day is live, falls back to the last historical day otherwise) + the full
  anomaly history below it (sort selector, category filter pills, client-side
  pagination). Every card has a "Voir la carte du trajet" button.
- **Expliquer un bus** — filters (line incl. "Toutes les lignes", bus, known-day
  dropdown, free manual date, direction). Picking a line immediately shows the **line
  verdict** (bon état / à surveiller / à risque, same thresholds as Streamlit) and the
  **reference trip** block (per-direction metrics + map — "what a NORMAL trip looks
  like"). **Analyser** runs `/api/anomaly-explain` with `check_detours=true` (can take
  30-40s+ on a heavily-flagged line or "all lines"; inline brain animation keeps the
  rest of the page usable). Results carry the low-data model warnings, metric help
  tooltips (hover the ⓘ), sort, category pills, and per-trip maps.
- **Tendances** — anomaly rate by line / by hour, via Chart.js (loaded lazily from CDN).
- **Anomalies billetterie** — ticket-anomaly patterns + admin/client view toggle.
- **Chauffeurs** — driver leaderboard + code lookup, with the same "for reference only,
  not a verdict" disclaimer as the main dashboard.

## Trip map

Leaflet + OpenStreetMap tiles (free, no API key), lazy-loaded from CDN like Chart.js.
Faithful to the Streamlit map: numbered stops in real visit order (RETOUR = reverse),
color code (green normal / blue long dwell ≥10min / amber signal loss ≥5min / red
unserved / gray suspect coords), size = dwell+signal-loss, Départ/Terminus labels,
planned-route line, detour overlay (orange = out leg, purple = back leg) with the
detour warning banner, and the first/last-tracked-passage caption.

## Live data relay (relay.php)

The platform webservices have no public URL — the Render API can't reach them. Instead,
`relay.php` runs **on the company's server** (which IS on the right network), pulls
yesterday's GPS pings + ticket totals from the webservices, and pushes them to Render
(`POST /api/ingest/*`, same X-API-Key). Once pushed, every visitor of this page sees
live data — visitors never run anything themselves.

Setup: fill `WINICARI_WEBSERVICE_URL` in `config.php`, then schedule ONE daily run after
the platform's night processing (e.g. 07:00):

```
# crontab (shell access)
0 7 * * * php /path/to/embed/relay.php >> /var/log/winicari_relay.log 2>&1

# or cPanel "Cron Jobs" -> command:
php /home/ACCOUNT/public_html/embed/relay.php

# or pure-HTTP cron (no shell): protect with the API key
0 7 * * * wget -qO- "https://their-domain/embed/relay.php?key=API_KEY"
```

Manual test: `php relay.php` (pushes yesterday) or `php relay.php 20260716`. If the run
is missed one morning, the page simply keeps showing the previous day until the next
successful run — nothing breaks.

## Known gaps / things to revisit

- Chart.js and Leaflet are loaded from public CDNs (`cdn.jsdelivr.net`, `unpkg.com`) —
  if the host environment blocks external script tags, charts/maps degrade gracefully
  (metrics/lists still show, with an "unavailable" notice) rather than breaking the page.
- Driver/Chauffeurs data depends on the `attach_driver_codes_to_trips` backfill having
  been run against whichever reference DB is deployed — confirm it's present in
  production before relying on it (it may be empty if the production DB predates that
  migration).
