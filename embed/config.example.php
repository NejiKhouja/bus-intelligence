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

// Web services de la plateforme (réseau local -- joignables depuis CE serveur, pas
// depuis Render). Utilisé uniquement par relay.php (le pont quotidien qui pousse la
// journée de la veille vers l'API Render, voir relay.php). Laisser vide si le relais
// tourne ailleurs.
define('WINICARI_WEBSERVICE_URL', 'http://102.128.57.59:8123');

// Clé Google Maps JavaScript API -- onglet "Tracer une nouvelle ligne" (voir
// assets/app.js::renderTraceLineView). ATTENTION, différent de WINICARI_API_KEY
// au-dessus : celle-ci N'A PAS BESOIN de rester secrète -- le SDK Google Maps JS
// s'exécute dans le navigateur, donc cette clé apparaît forcément dans le code source
// de la page (visible via view-source). C'est le modèle de sécurité normal de Google :
// on la restreint côté Google Cloud Console (API Google Maps Platform > Identifiants)
// par référent HTTP (le domaine de ce site) et par API activée (Maps JavaScript API +
// Directions API uniquement), pas en la cachant. Stockée ici uniquement pour que
// l'admin n'ait pas à la recoller à chaque session (retour utilisateur 2026-07-24).
define('WINICARI_GOOGLE_MAPS_KEY', 'REPLACE_ME');
