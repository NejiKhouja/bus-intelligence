# PHP Integration Guide — WiniCari AI API

This is the REST contract for the native PHP/HTML platform to call the WiniCari AI layer.
The API is a standalone service (FastAPI) — the PHP app never runs Python directly, it just
makes HTTP requests. See `docs/DEPLOYMENT.md` for how the service itself is hosted.

This deployment phase covers **3 of the 4 modules**: delay/ETA, GPS fallback, anomaly
detection. The RAG chatbot is excluded for now (`ENABLE_CHATBOT=false`) — calling
`/api/chatbot/ask` returns `503`.

## Base URL and authentication

```
https://<your-api-domain>            (set up in docs/DEPLOYMENT.md, via Caddy)
```

Every `/api/*` request must include the shared secret header:

```
X-API-Key: <the API_KEY value set in the server's .env>
```

`GET /health` is the one exception — always open, no key needed, useful for your own
uptime checks. Missing/wrong key on any other endpoint returns `401`.

## PHP request pattern

Mirrors the pattern already used internally by `src/dashboard/app.py` (plain HTTP, JSON
body or query params, no session state) — just implemented with PHP's curl instead of
Python's `requests`:

```php
<?php
function winicari_get(string $path, array $query = []): ?array {
    $url = getenv('WINICARI_API_URL') . $path;
    if ($query) { $url .= '?' . http_build_query($query); }
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_HTTPHEADER => ['X-API-Key: ' . getenv('WINICARI_API_KEY')],
    ]);
    $body = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    return $status === 200 ? json_decode($body, true) : null;
}

function winicari_post(string $path, array $payload): ?array {
    $ch = curl_init(getenv('WINICARI_API_URL') . $path);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => json_encode($payload),
        CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'X-API-Key: ' . getenv('WINICARI_API_KEY'),
        ],
    ]);
    $body = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    return $status === 200 ? json_decode($body, true) : null;
}

// A few endpoints (e.g. /api/predict/delay/auto) are POST but take query-string params,
// not a JSON body -- FastAPI treats undecorated function params that way when no request
// body model is declared. Same idea as winicari_post, just no JSON payload.
function winicari_post_query(string $path, array $query): ?array {
    $ch = curl_init(getenv('WINICARI_API_URL') . $path . '?' . http_build_query($query));
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_POST => true,
        CURLOPT_HTTPHEADER => ['X-API-Key: ' . getenv('WINICARI_API_KEY')],
    ]);
    $body = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    return $status === 200 ? json_decode($body, true) : null;
}
```

Set `WINICARI_API_URL` and `WINICARI_API_KEY` in the PHP app's own environment config —
never hardcode the key in source.

## Module 1 — Delay / ETA

**Auto mode** (server figures out the bus's current position from its live GPS data) —
query params, not a JSON body, despite being `POST`:

```php
$eta = winicari_post_query('/api/predict/delay/auto', [
    'societe' => 'S.R.T.K', 'line' => '202', 'bus' => 6028, 'day' => '20250120',
]);
// -> {"societe":"S.R.T.K", "line":"202", "bus":6028, "direction":"RETOUR",
//     "current_stop":"EL MENZEL", "current_delay_min":140.1, "total_stops":13,
//     "predictions":[{"seq":..., "expected_min":..., "pred_delay_min":..., "eta":"..."}, ...],
//     "model_used":"hgbm", "mode":"auto"}
```

**ETA to a specific rider's stop**:
```php
$eta = winicari_get('/api/eta-to-stop', [
    'societe' => 'S.R.T.K', 'line' => '202', 'bus' => 6028,
    'day' => '20250120', 'target_seq' => 5,
]);
```

**Manual mode** (you already know the bus's current stop/delay — JSON body):
```php
$eta = winicari_post('/api/predict/delay/manual', [
    'societe' => 'S.R.T.K', 'line' => '202', 'direction' => 'RETOUR',
    'dep_time' => '2025-01-20 10:28', 'current_seq' => 2, 'current_delay_min' => 5.0,
    'model_type' => 'hgbm',  // or 'lstm'
]);
```

## Module 2 — GPS Fallback

Estimates bus position during a GPS signal gap — requires live MongoDB reachable from the
API server (the one module in this deployment that does):

```php
$pos = winicari_post('/api/predict/gps-fallback', [
    'day' => '20250120', 'line' => '202', 'societe' => 'S.R.T.K',
    'bus' => 6028, 'query_time' => '2025-01-20 13:00:00',
]);
// -> {"lat":..., "lon":..., "s_m":..., "uncertainty_m":..., "method":"kalman"}
```

## Module 3 — Anomaly Detection

**Previously-scored trips** (fast, static file, no live Mongo needed):
```php
$history = winicari_get('/api/anomaly-history', [
    'societe' => 'S.R.T.K', 'line' => '202', 'limit' => 30,
]);
// -> {"anomalies":[{"day":"20260208", "trip_id":20839, "bus":6029, "severity":"high",
//        "if_score":..., "lstm_score":..., "reasons":[...]}], "total":..., "total_trips":...}
```

**Real-time scoring** (new in this deployment) — score an in-progress/just-finished trip
that isn't in the precomputed history yet. Needs live MongoDB (same GPS-ping requirement
as GPS fallback):
```php
$live = winicari_post('/api/anomaly/score-live', [
    'day' => '20250120', 'line' => '202', 'societe' => 'S.R.T.K', 'bus' => 6028,
]);
// -> {"societe":"S.R.T.K", "trips":[{"trip_id":0, "severity":"low"|"medium"|"high",
//        "reasons":[...], "if_score":..., "lstm_score":...}], "anomaly_count":0}
```

## Error handling

- `401` — missing/wrong `X-API-Key`.
- `404` — no data found for the given societe/line/bus/day (check spelling — line codes
  are company-specific, e.g. line `"202"` under `S.R.T.K` is unrelated to line `"202"`
  under another company).
- `503` — the relevant model group failed to load at startup (check `/health`'s `models`
  list) or the chatbot is disabled.
- `500` — unexpected server error (check `logs/api.log` on the server, or ask the AI team
  — this shouldn't happen in normal operation).

## Reference

Full endpoint inventory, hosting, and operational details: `docs/DEPLOYMENT.md`. Data
provenance and model methodology: `docs/DATA_PIPELINE_REPORT.md`.
