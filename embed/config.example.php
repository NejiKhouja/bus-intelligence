<?php
// Copy this file to config.php and fill in the real values. config.php is gitignored --
// it holds the actual API key, never committed. This example file IS committed so the
// next person knows what to fill in.

// Render-hosted FastAPI backend (anomaly + ticket-anomaly modules only -- that's all
// that's deployed there, see docs/DEPLOYMENT.md in the main repo).
define('WINICARI_API_BASE', 'https://bus-intelligence.onrender.com');

// Must match the API_KEY env var set on the Render service. Sent as the X-API-Key header
// by proxy.php on every request -- NEVER expose this to the browser/JS directly, that
// defeats the point of having it. See src/api/main.py's `require_api_key` middleware.
define('WINICARI_API_KEY', 'REPLACE_ME');

// Companies the picker offers when no session company is set yet (see session.php).
// Kept as a static fallback list so the picker works even if /api/options is briefly
// unreachable -- the live list from /api/options is still preferred when available.
define('WINICARI_FALLBACK_COMPANIES', [
    'EPE-TVE', 'S.R.T.BIZERTE', 'S.R.T.K', 'S.R.T.M', 'S.R.T.SELIANA', 'S.T.C.I',
    'S.T.S', 'SORETRAS', 'SRT.ELGOUAFEL', 'TCV', 'TUS', 'Winicari',
]);
