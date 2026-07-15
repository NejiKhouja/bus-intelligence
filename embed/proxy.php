<?php
/**
 * Same-origin proxy to the Render-hosted WiniCari API.
 *
 * Why this exists instead of calling the API directly from JS: the API is protected by
 * an X-API-Key header (see src/api/main.py's require_api_key middleware in the main
 * repo). If the JS called Render directly, that key would have to live in client-side
 * JS -- visible to anyone via view-source/devtools on the company's own dashboard. This
 * proxy keeps the key server-side (config.php, gitignored) and forwards on the browser's
 * behalf, so the key never reaches the browser. It also means the browser only ever
 * talks to its own origin, so there's no CORS setup needed against Render at all.
 *
 * Endpoint whitelist: only the anomaly/ticket-anomaly/driver endpoints this widget uses
 * are allowed through -- deliberately scoped down (per project decision: this embed is
 * for the anomaly-detection features only, nothing else is hosted on Render anyway).
 */

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/session.php';

header('Content-Type: application/json; charset=utf-8');

const ALLOWED_ENDPOINTS = [
    '/health',
    '/api/options',
    '/api/lines-ranked',
    '/api/lines',
    '/api/days-for-line',
    '/api/buses-for-line',
    '/api/buses-for-day',
    '/api/directions',
    '/api/anomaly-explain',
    '/api/anomaly-patterns',
    '/api/anomaly-history',
    '/api/current-anomalies',
    '/api/reference-trip',
    '/api/trip-detail',
    '/api/driver-stats',
    '/api/drivers-ranked',
    '/api/ticket-anomaly-history',
    '/api/ticket-anomaly-patterns',
    '/api/ticket-anomaly-explain',
    '/api/ticket-anomaly-stations',
    '/api/ticket-anomaly-reference',
];

$endpoint = $_GET['endpoint'] ?? '';
if (!in_array($endpoint, ALLOWED_ENDPOINTS, true)) {
    http_response_code(400);
    echo json_encode(['detail' => 'Unknown or disallowed endpoint']);
    exit;
}

// Every request is pinned to the session's company -- a company embed must never be able
// to read another company's data just by editing the querystring in devtools. `societe`
// from the client is IGNORED on purpose; the session value always wins.
$company = winicari_current_company();
if ($company === null && $endpoint !== '/health' && $endpoint !== '/api/options') {
    http_response_code(403);
    echo json_encode(['detail' => 'No company selected for this session']);
    exit;
}

$params = $_GET;
unset($params['endpoint']);
if ($company !== null) {
    $params['societe'] = $company;
}

$url = WINICARI_API_BASE . $endpoint;
if ($params) {
    $url .= '?' . http_build_query($params);
}

$ch = curl_init($url);
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['X-API-Key: ' . WINICARI_API_KEY],
    CURLOPT_TIMEOUT => 90, // le contrôle de détour côté API peut prendre jusqu'à ~40s
    CURLOPT_CONNECTTIMEOUT => 15,
]);
$body = curl_exec($ch);
$status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$err = curl_error($ch);
curl_close($ch);

if ($body === false) {
    http_response_code(502);
    echo json_encode(['detail' => 'Upstream request failed: ' . $err]);
    exit;
}

http_response_code($status ?: 502);
echo $body;
