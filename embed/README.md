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

## Views

- **Trajets & analyse** — merges the old "today's flagged trips" quick view with a full
  filter/drill-down panel (line, bus, day, manual date, "all lines", direction). The
  *default* landing view uses `/api/current-anomalies` (fast, today only); clicking
  **Analyser** runs the full `/api/anomaly-explain` with `check_detours=true`, which can
  take 30-40s on a heavily-flagged line or "all lines" — that's why it's not the default.
- **Tendances** — anomaly rate by line / by hour, via Chart.js (loaded lazily from CDN).
- **Anomalies billetterie** — ticket-anomaly patterns + admin/client view toggle.
- **Chauffeurs** — driver leaderboard + code lookup, with the same "for reference only,
  not a verdict" disclaimer as the main dashboard.

## Known gaps / things to revisit

- No trip map (the Streamlit dashboard renders one via Plotly) — this build shows
  stop-level chips (dwell, signal loss, detour, etc.) as text instead. Add one later if
  needed; no mapping API key was in scope for this pass.
- Chart.js is loaded from a public CDN (`cdn.jsdelivr.net`) — if the host environment
  blocks external script tags, charts degrade gracefully (metrics/lists still show, with
  a "graphs unavailable" notice) rather than breaking the page.
- Driver/Chauffeurs data depends on the `attach_driver_codes_to_trips` backfill having
  been run against whichever reference DB is deployed — confirm it's present in
  production before relying on it (it may be empty if the production DB predates that
  migration).
