<?php
/**
 * Same-origin proxy to the Render-hosted WiniCari API -- WITH file-based caching.
 *
 * Why this exists instead of calling the API directly from JS: the API is protected by
 * an X-API-Key header (see src/api/main.py's require_api_key middleware in the main
 * repo). If the JS called Render directly, that key would have to live in client-side
 * JS -- visible to anyone via view-source/devtools on the company's own dashboard. This
 * proxy keeps the key server-side (config.php, gitignored) and forwards on the browser's
 * behalf, so the key never reaches the browser. It also means the browser only ever
 * talks to its own origin, so there's no CORS setup needed against Render at all.
 *
 * CACHING (added 2026-07-19, user request: "exhaustive web caching... show results that
 * are available at first if possible"). Three states per (endpoint, params, company):
 *
 *   FRESH (age <= ttl)       -> serve the cached file immediately.        X-Cache: HIT
 *   STALE (ttl < age <= max) -> serve the cached file immediately AND     X-Cache: STALE
 *                                fire a background self-request that revalidates it for
 *                                the NEXT caller (this caller never waits for that).
 *   MISS  (no file / too old)-> block, fetch from Render, cache, serve.   X-Cache: MISS
 *
 * This is what makes "first visitor pulls, everyone else is instant" apply not just to
 * the daily live-data push (see relay.php) but to every read this widget makes,
 * including the 30-40s "Analyser" detour-checked query -- a second admin re-running the
 * same line/day gets it instantly, and a background refresh keeps it from going stale
 * for long. Endpoint whitelist unchanged from before; only the serving logic changed.
 */

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
];

// [ttl_frais_secondes, age_max_avant_blocage_secondes] par endpoint. Choisi large plutôt
// que strict : un badge "peut-être un peu daté" coûte moins qu'un visiteur qui attend --
// surtout pour les listes qui ne bougent presque jamais (lignes, trajet de référence).
// current-anomalies/anomaly-explain restent courts car ce SONT les données en direct que
// l'admin veut fraîches ; passé leur max, on rebloque plutôt que de servir un résultat
// trop vieux pour être utile.
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
];

// Endpoints trop coûteux côté serveur Render pour être revalidés SILENCIEUSEMENT --
// servis périmés sans déclencher de rafraîchissement en arrière-plan (voir la note dans
// la branche STALE plus bas). Seul anomaly-explain (détours = filtre de Kalman par
// bus-jour signalé) l'est aujourd'hui ; ajouter ici tout futur endpoint du même acabit.
const NO_BG_REFRESH = [
    '/api/anomaly-explain',
];

$endpoint = $_GET['endpoint'] ?? '';
if (!in_array($endpoint, ALLOWED_ENDPOINTS, true)) {
    http_response_code(400);
    echo json_encode(['detail' => 'Unknown or disallowed endpoint']);
    exit;
}

// ── Résolution de la société ──────────────────────────────────────────────────────────
// Cas normal (visiteur réel) : la session -- `societe` envoyé par le client est IGNORÉ,
// la session gagne toujours, pour qu'un embed ne puisse jamais lire les données d'une
// autre société en éditant l'URL.
// Cas revalidation en arrière-plan (voir plus bas, _bg=1) : cet appel EST fait par ce
// serveur lui-même, sans le cookie du visiteur -- la société est alors passée
// explicitement et validée par une clé partagée (WINICARI_API_KEY, jamais exposée au
// navigateur) plutôt que par la session, qui n'existe pas pour cet appel interne.
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

// ── Chemin normal : servir depuis le cache si possible ──────────────────────────────────
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
        // Rendre la réponse au visiteur MAINTENANT -- le déclenchement de revalidation
        // ci-dessous est un appel HTTP vers ce serveur lui-même (hairpin via l'IPLB) dont
        // le timeout n'est plus garanti aussi strict qu'avec curl (voir http.php) ; sans
        // ce flush, le visiteur restait bloqué jusqu'à la fin du script entier, pas
        // seulement jusqu'à readfile() -- constaté 2026-07-24 (Render répondait vite,
        // mais l'onglet restait en "loading").
        if (function_exists('fastcgi_finish_request')) {
            fastcgi_finish_request();
        } else {
            @ob_end_flush();
            @flush();
        }
        // PAS de revalidation en arrière-plan pour les endpoints coûteux (voir
        // NO_BG_REFRESH) -- anomaly-explain?check_detours=true fait tourner un filtre de
        // Kalman par bus-jour signalé sur le serveur Render 512MB, déjà responsable de
        // plusieurs OOM avant même l'ajout du cache. Le déclencher SILENCIEUSEMENT en
        // arrière-plan (sans qu'aucun visiteur n'ait cliqué "Analyser") pouvait faire
        // tourner cette analyse en parallèle d'une requête réellement en cours, cumulant
        // la mémoire des deux -- constaté 2026-07-19 juste après l'ajout du cache. Un
        // résultat périmé reste servi instantanément (juste sans rafraîchissement
        // automatique) ; il ne redevient frais que sur un vrai clic "Analyser" du
        // visiteur, ou après STALE_MAX quand le cache tombe en MISS.
        if (!in_array($endpoint, NO_BG_REFRESH, true)) {
            // Déclenchement tir-et-oublie de la revalidation -- CE visiteur ne l'attend pas
            // (timeout très court côté déclencheur) ; le script continue seul côté serveur
            // (ignore_user_abort dans la branche _bg=1 ci-dessus), même mécanisme que
            // autorun.php pour le relais quotidien.
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

// MISS ou trop vieux pour être servi tel quel : on bloque, comme avant caching -- MAIS
// avec un verrou anti-troupeau (stampede lock), ajouté 2026-07-24 suite aux 502 constatés
// en production dès que 2 visiteurs arrivaient en même temps.
//
// Cause : SANS ce verrou, 2 visiteurs qui tombent sur la même clé manquante en même temps
// déclenchaient chacun leur propre appel bloquant vers Render (jusqu'à 90s) -- double
// charge mémoire sur un service Render à 512MB déjà responsable de plusieurs OOM (voir
// NO_BG_REFRESH plus haut), ET chacun occupe un worker PHP-FPM pendant tout ce temps. Une
// seule page ouvre pourtant ~8-10 appels proxy.php en parallèle (options, lignes,
// current-anomalies, historique, tickets, chauffeurs...) : 2 visiteurs suffisent à
// épuiser un pool de workers PHP-FPM mutualisé de taille modeste. Une fois le pool à sec,
// TOUTE requête suivante (y compris d'autres visiteurs, y compris des endpoints déjà en
// cache) reçoit un 502 Bad Gateway du serveur web -- alors même que Render, dans ses
// propres logs, répond normalement : le goulot est ici, côté PHP, pas côté Render.
//
// Fix : un seul visiteur par clé de cache va réellement chercher la donnée chez Render ;
// les autres attendent (verrou fichier, borné) puis servent le résultat que le premier
// vient d'écrire -- une seule requête Render au lieu de N, et une attente bornée au lieu
// d'un doublement du temps de réponse.
$miss_lock = fopen("$cache_dir/$cache_key.miss.lock", 'c');
$have_miss_lock = false;
if ($miss_lock !== false) {
    $waited = 0.0;
    // Borne large (proche du timeout upstream de 90s, voir winicari_fetch_upstream) --
    // le but est que l'attente se résolve presque toujours par un cache tout frais
    // écrit par le premier visiteur, pas par un abandon suivi d'un 2e appel Render.
    while (!($have_miss_lock = flock($miss_lock, LOCK_EX | LOCK_NB))) {
        if ($waited >= 95.0) {
            break; // filet de sécurité -- ne jamais bloquer un worker indéfiniment
        }
        usleep(200000); // 200ms
        $waited += 0.2;
    }
    if (!$have_miss_lock) {
        @fclose($miss_lock);
        $miss_lock = null;
    } elseif ($waited > 0) {
        // On a attendu : un autre visiteur vient peut-être de rafraîchir cette même clé
        // pendant notre attente -- si oui, on sert son résultat au lieu de resolliciter
        // Render pour rien.
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
