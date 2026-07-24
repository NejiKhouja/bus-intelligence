<?php
/**
 * Relais quotidien : web services (réseau local) -> API Render.
 *
 * Pourquoi : les web services de la plateforme n'ont pas d'URL publique -- le serveur
 * Render (Allemagne) ne peut pas les atteindre. CE serveur-ci, lui, est sur le bon
 * réseau : ce script tire la journée de la VEILLE depuis les web services et la pousse
 * vers l'API Render (POST /api/ingest/*, protégé par la même X-API-Key que le reste).
 * Une fois poussée, la page d'anomalies affiche ces données "en direct" pour tous les
 * visiteurs, où qu'ils soient -- aucun visiteur n'exécute quoi que ce soit.
 *
 * Planification (une fois par jour, après le traitement de nuit de la plateforme) :
 *
 *   Linux/cPanel (crontab -e) :
 *     0 7 * * * php /chemin/vers/embed/relay.php >> /var/log/winicari_relay.log 2>&1
 *
 *   cPanel "Cron Jobs" (si pas d'accès shell) -- commande :
 *     php /home/COMPTE/public_html/embed/relay.php
 *
 *   Ou via HTTP (wget/curl planifié, ou hébergeur sans cron PHP direct) :
 *     0 7 * * * wget -qO- "https://votre-domaine/embed/relay.php?key=VOTRE_API_KEY&day="
 *   (l'invocation web exige ?key= égal à WINICARI_API_KEY -- jamais exposée aux
 *    visiteurs : ne mettez ce lien nulle part dans une page.)
 *
 * Test manuel :  php relay.php            (pousse hier)
 *                php relay.php 20260716   (pousse un jour précis)
 *
 * MODE AUTO (?auto=1, déclenché par index.php à la première visite -- voir autorun.php) :
 * pas de planificateur à configurer du tout. La requête de déclenchement coupe après
 * ~300 ms (le visiteur n'attend RIEN) ; ce script continue seul en arrière-plan
 * (ignore_user_abort), vérifie d'abord auprès de l'API si la journée d'hier y est déjà
 * (GET /api/ingest/status -- l'API est la seule source de vérité, son magasin étant
 * éphémère), et ne pousse que si elle manque. Un verrou fichier garantit qu'une seule
 * exécution tourne à la fois même si plusieurs visiteurs arrivent ensemble.
 */

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/http.php';

// ── Garde d'accès : CLI librement ; via web, uniquement avec la clé API ──────────────
$is_cli = (php_sapi_name() === 'cli');
$is_auto = !$is_cli && isset($_GET['auto']);
if (!$is_cli) {
    if (!isset($_GET['key']) || !hash_equals(WINICARI_API_KEY, $_GET['key'])) {
        http_response_code(403);
        exit("Forbidden\n");
    }
    header('Content-Type: text/plain; charset=utf-8');
    // Une journée complète peut prendre 1-2 min à tirer + pousser ; en mode auto le
    // déclencheur coupe la connexion après ~300 ms et on continue sans lui.
    ignore_user_abort(true);
    set_time_limit(600);
}

// ── Verrou : une seule exécution à la fois (2 visiteurs simultanés = 1 seul push) ────
$var_dir = __DIR__ . '/var';
if (!is_dir($var_dir)) {
    @mkdir($var_dir, 0775, true);
}
$lock = fopen($var_dir . '/relay.lock', 'c');
if ($lock === false || !flock($lock, LOCK_EX | LOCK_NB)) {
    exit("Un relais tourne déjà -- rien à faire.\n");
}
// json_decode d'une journée complète (~10 Mo de JSON) coûte bien plus en RAM que la
// limite PHP par défaut (souvent 128M) -- relevé explicitement ici, uniquement pour ce
// script (ini_set est local au process, pas au serveur entier).
ini_set('memory_limit', '512M');

if (!defined('WINICARI_WEBSERVICE_URL') || !WINICARI_WEBSERVICE_URL) {
    exit("WINICARI_WEBSERVICE_URL non défini dans config.php -- voir config.example.php\n");
}

$day = $is_cli ? ($argv[1] ?? '') : ($_GET['day'] ?? '');
if (!preg_match('/^\d{8}$/', $day)) {
    $day = date('Ymd', strtotime('-1 day'));
}
$day_dashed = substr($day, 0, 4) . '-' . substr($day, 4, 2) . '-' . substr($day, 6, 2);

echo "Relais WiniCari -- jour $day -> " . WINICARI_API_BASE . "\n";
if ($is_auto) {
    // La connexion du déclencheur peut être coupée d'un instant à l'autre -- vider ce
    // qu'on a déjà écrit et continuer en autonome.
    if (function_exists('fastcgi_finish_request')) {
        fastcgi_finish_request();
    } else {
        @ob_end_flush();
        @flush();
    }
}

// ── Helpers HTTP ─────────────────────────────────────────────────────────────────────
function ws_get(string $path, array $params, int $timeout = 180) {
    $url = rtrim(WINICARI_WEBSERVICE_URL, '/') . $path . '?' . http_build_query($params);
    [$body, $status, $err] = winicari_http_request($url, [], 'GET', null, (float)$timeout, true);
    if ($body === false || $status >= 400) {
        throw new RuntimeException("web service $path: " . ($err ?: "HTTP $status"));
    }
    return json_decode($body, true);
}

function render_req(string $method, string $path, $json_body = null, int $timeout = 300) {
    $headers = ['X-API-Key: ' . WINICARI_API_KEY];
    $body = null;
    if ($json_body !== null) {
        $headers[] = 'Content-Type: application/json';
        $body = json_encode($json_body);
    }
    $url = rtrim(WINICARI_API_BASE, '/') . $path;
    [$respBody, $status, $err] = winicari_http_request($url, $headers, $method, $body, (float)$timeout, true);
    if ($respBody === false || $status >= 400) {
        throw new RuntimeException("API $path: " . ($err ?: "HTTP $status -- $respBody"));
    }
    return json_decode($respBody, true);
}

// Miroir de src/data/webservices.py::_slim_ping -- mêmes 7 champs, même filtrage.
function slim_ping(array $p): ?array {
    $loc = $p['localisation'] ?? [];
    $bus = $p['bus'] ?? [];
    $svc = $p['service'] ?? [];
    $line = (string)($svc['codeLigne'] ?? '');
    $bus_code = (string)($bus['code'] ?? '');
    if ($line === '' || $bus_code === '' || !isset($loc['x'], $loc['y']) || empty($p['date'])) {
        return null;
    }
    return [
        'line' => $line,
        'bus' => $bus_code,
        't' => $p['date'],
        'lat' => (float)$loc['x'],
        'lon' => (float)$loc['y'],
        'speed' => isset($bus['vitesse']) ? (float)$bus['vitesse'] : null,
        'voyage' => $svc['voyage'] ?? null,
    ];
}

// ── 0. Mode auto : l'API a-t-elle déjà cette journée ? ───────────────────────────────
// L'API est la SEULE source de vérité (son magasin est éphémère : instance redémarrée =
// magasin vide) -- un marqueur local dirait "déjà poussé" à tort. Si au moins un fichier
// GPS de ce jour y est présent, rien à faire.
if ($is_auto) {
    try {
        $status = render_req('GET', '/api/ingest/status', null, 60);
        foreach (($status['files'] ?? []) as $f) {
            if (($f['kind'] ?? '') === 'gps' && ($f['day'] ?? '') === $day) {
                exit("$day : déjà présent sur l'API -- rien à pousser.\n");
            }
        }
    } catch (Exception $e) {
        // API injoignable (instance froide qui démarre ?) -- on tente le push quand
        // même, les POST réveilleront/attendront l'instance.
        echo 'status indisponible (' . $e->getMessage() . ") -- push tenté quand même\n";
    }
}

// ── 1. Le traitement de nuit est-il prêt ? ───────────────────────────────────────────
try {
    $ready = ws_get('/Service/isDayReady', ['day' => $day], 30);
    if (empty($ready['ready'])) {
        exit("$day : traitement de nuit pas prêt côté plateforme -- rien à pousser.\n");
    }
} catch (Exception $e) {
    exit('Échec isDayReady : ' . $e->getMessage() . "\n");
}

// ── 2. Sociétés à pousser (liste vivante depuis l'API) ───────────────────────────────
try {
    $options = render_req('GET', '/api/options', null, 60);
    $societes = $options['companies'] ?? [];
} catch (Exception $e) {
    $societes = WINICARI_FALLBACK_COMPANIES;
    echo 'options indisponibles (' . $e->getMessage() . ") -- liste de secours utilisée\n";
}

// ── 3. GPS par société ───────────────────────────────────────────────────────────────
foreach ($societes as $soc) {
    try {
        $raw = ws_get('/Service/getPingsForDay', ['day' => $day, 'societe' => $soc]);
    } catch (Exception $e) {
        echo "  $soc : échec web service (" . $e->getMessage() . ") -- ignorée\n";
        continue;
    }
    $pings = [];
    foreach (($raw ?: []) as $p) {
        $s = slim_ping($p);
        if ($s !== null) {
            $pings[] = $s;
        }
    }
    unset($raw);
    if (!$pings) {
        echo "  $soc : aucun ping ce jour -- ignorée\n";
        continue;
    }
    try {
        $res = render_req('POST', '/api/ingest/gps-day',
                          ['day' => $day, 'societe' => $soc, 'pings' => $pings]);
        echo "  $soc : " . json_encode($res) . "\n";
    } catch (Exception $e) {
        echo "  $soc : échec push (" . $e->getMessage() . ")\n";
    }
    unset($pings);
}

// ── 4. Billetterie (toutes sociétés en un appel) ─────────────────────────────────────
try {
    $rows = ws_get('/ServiceDetais/getTicketTotalsForDay', ['day' => $day_dashed], 120);
    if ($rows) {
        $res = render_req('POST', '/api/ingest/ticket-day', ['day' => $day, 'rows' => $rows]);
        echo '  billetterie : ' . json_encode($res) . "\n";
    } else {
        echo "  billetterie : aucune ligne ce jour\n";
    }
} catch (Exception $e) {
    echo '  billetterie : échec (' . $e->getMessage() . ")\n";
}

echo "Terminé.\n";
