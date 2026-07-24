<?php
require_once __DIR__ . '/config.php';
require_once __DIR__ . '/session.php';
require_once __DIR__ . '/http.php';

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
    '/api/line-coverage',
];

const CACHE_RULES = [
    '/health'                       => [10, 30],
    '/api/options'                  => [3600, 86400],
    '/api/lines-ranked'             => [1800, 21600],
    '/api/lines'                    => [1800, 21600],
    '/api/days-for-line'            => [900, 21600],
    '/api/buses-for-line'           => [1800, 21600],
    '/api/buses-for-day'            => [900, 21600],
    '/api/directions'               => [1800, 21600],
    '/api/reference-trip'           => [1800, 21600],
    '/api/trip-detail'              => [3600, 86400],
    '/api/current-anomalies'        => [90, 1800],
    '/api/anomaly-history'          => [300, 3600],
    '/api/anomaly-patterns'         => [600, 3600],
    '/api/anomaly-explain'          => [900, 3600],
    '/api/driver-stats'             => [600, 3600],
    '/api/drivers-ranked'           => [600, 3600],
    '/api/ticket-anomaly-history'   => [600, 3600],
    '/api/ticket-anomaly-patterns'  => [600, 3600],
    '/api/ticket-anomaly-explain'   => [600, 3600],
    '/api/ticket-anomaly-stations'  => [1800, 21600],
    '/api/ticket-anomaly-reference' => [1800, 21600],
    // Ne change que quand quelqu'un peuple manuellement line_stops côté reference DB, puis
    // redémarre l'API -- aussi statique que /api/options, même TTL large.
    '/api/line-coverage'            => [3600, 86400],
];

const NO_BG_REFRESH = [
    '/api/anomaly-explain',
];

$endpoint = $_GET['endpoint'] ?? '';
if (!in_array($endpoint, ALLOWED_ENDPOINTS, true)) {
    http_response_code(400);
    echo json_encode(['detail' => 'Unknown or disallowed endpoint']);
    exit;
}
$is_bg = isset($_GET['_bg']) && isset($_GET['_bg_key']) && hash_equals(WINICARI_API_KEY, (string)$_GET['_bg_key']);
if ($is_bg) {
    $company = $_GET['_company'] ?? null;
    $company = ($company === '' ) ? null : $company;
} else {
    $company = winicari_current_company();
}
if ($company === null && $endpoint !== '/health' && $endpoint !== '/api/options') {
    http_response_code(403);
    echo json_encode(['detail' => 'No company selected for this session']);
    exit;
}

$params = $_GET;
unset($params['endpoint'], $params['_bg'], $params['_bg_key'], $params['_company']);
if ($company !== null) {
    $params['societe'] = $company;
}

function winicari_fetch_upstream(string $endpoint, array $params): array {
    $url = WINICARI_API_BASE . $endpoint;
    if ($params) {
        $url .= '?' . http_build_query($params);
    }
    // le contrôle de détour côté API peut prendre jusqu'à ~40s -- timeout large
    return winicari_http_request($url, ['X-API-Key: ' . WINICARI_API_KEY], 'GET', null, 90.0, true);
}

// ── Clé de cache : endpoint + paramètres (dont societe) triés, insensible à l'ordre ────
ksort($params);
$cache_key = md5($endpoint . '?' . http_build_query($params));
$cache_dir = __DIR__ . '/var/cache';
if (!is_dir($cache_dir)) {
    @mkdir($cache_dir, 0775, true);
}
$cache_file = "$cache_dir/$cache_key.json";
$status_file = "$cache_dir/$cache_key.status";

// ── Mode revalidation en arrière-plan : toujours aller chercher, jamais servir le cache ─
if ($is_bg) {
    ignore_user_abort(true);
    set_time_limit(120);
    $lock = fopen("$cache_dir/$cache_key.lock", 'c');
    if ($lock === false || !flock($lock, LOCK_EX | LOCK_NB)) {
        exit; // une revalidation de CETTE clé tourne déjà
    }
    [$body, $status, $err] = winicari_fetch_upstream($endpoint, $params);
    if ($body !== false && $status < 500) {
        file_put_contents($cache_file, $body, LOCK_EX);
        file_put_contents($status_file, (string)$status, LOCK_EX);
    }
    flock($lock, LOCK_UN);
    exit;
}

set_time_limit(120);
[$ttl, $stale_max] = CACHE_RULES[$endpoint] ?? [60, 600];
$age = file_exists($cache_file) ? (time() - filemtime($cache_file)) : null;
$cached_status = file_exists($status_file) ? (int)file_get_contents($status_file) : 200;

if ($age !== null && $cached_status < 400) {
    if ($age <= $ttl) {
        header('X-Cache: HIT');
        header("X-Cache-Age: $age");
        http_response_code($cached_status);
        readfile($cache_file);
        exit;
    }
    if ($age <= $stale_max) {
        header('X-Cache: STALE');
        header("X-Cache-Age: $age");
        http_response_code($cached_status);
        readfile($cache_file);
        if (function_exists('fastcgi_finish_request')) {
            fastcgi_finish_request();
        } else {
            @ob_end_flush();
            @flush();
        }
        if (!in_array($endpoint, NO_BG_REFRESH, true)) {
            $self = sprintf('%s://%s%s', (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off') ? 'https' : 'http',
                            $_SERVER['HTTP_HOST'] ?? 'localhost', $_SERVER['SCRIPT_NAME'] ?? '/proxy.php');
            $bg_params = $params;
            unset($bg_params['societe']);
            $bg_params['endpoint'] = $endpoint;
            $bg_params['_bg'] = 1;
            $bg_params['_bg_key'] = WINICARI_API_KEY;
            if ($company !== null) {
                $bg_params['_company'] = $company;
            }
            winicari_http_request($self . '?' . http_build_query($bg_params), [], 'GET', null, 0.4, false);
        }
        exit;
    }
}
$miss_lock = fopen("$cache_dir/$cache_key.miss.lock", 'c');
$have_miss_lock = false;
if ($miss_lock !== false) {
    $waited = 0.0;
    while (!($have_miss_lock = flock($miss_lock, LOCK_EX | LOCK_NB))) {
        if ($waited >= 4.0) {
            break; // filet de sécurité -- ne jamais bloquer un worker près de sa propre limite d'exécution
        }
        usleep(200000); // 200ms
        $waited += 0.2;
    }
    if (!$have_miss_lock) {
        @fclose($miss_lock);
        $miss_lock = null;
    } elseif ($waited > 0) {
        clearstatcache(true, $cache_file);
        clearstatcache(true, $status_file);
        $age2 = file_exists($cache_file) ? (time() - filemtime($cache_file)) : null;
        if ($age2 !== null && $age2 <= $stale_max) {
            $cached_status2 = file_exists($status_file) ? (int)file_get_contents($status_file) : 200;
            if ($cached_status2 < 400) {
                flock($miss_lock, LOCK_UN);
                fclose($miss_lock);
                header('X-Cache: HIT-AFTER-WAIT');
                http_response_code($cached_status2);
                readfile($cache_file);
                exit;
            }
        }
    }
}

[$body, $status, $err] = winicari_fetch_upstream($endpoint, $params);
if ($body === false) {
    if ($have_miss_lock) { flock($miss_lock, LOCK_UN); fclose($miss_lock); }
    http_response_code(502);
    echo json_encode(['detail' => 'Upstream request failed: ' . $err]);
    exit;
}
if ($status < 500) {
    file_put_contents($cache_file, $body, LOCK_EX);
    file_put_contents($status_file, (string)$status, LOCK_EX);
}
if ($have_miss_lock) { flock($miss_lock, LOCK_UN); fclose($miss_lock); }
header('X-Cache: MISS');
http_response_code($status ?: 502);
echo $body;
