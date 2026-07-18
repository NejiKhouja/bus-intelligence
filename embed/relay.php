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
 */

require_once __DIR__ . '/config.php';

// ── Garde d'accès : CLI librement ; via web, uniquement avec la clé API ──────────────
$is_cli = (php_sapi_name() === 'cli');
if (!$is_cli) {
    if (!isset($_GET['key']) || !hash_equals(WINICARI_API_KEY, $_GET['key'])) {
        http_response_code(403);
        exit("Forbidden\n");
    }
    header('Content-Type: text/plain; charset=utf-8');
    // Une journée complète peut prendre 1-2 min à tirer + pousser
    set_time_limit(600);
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

// ── Helpers HTTP ─────────────────────────────────────────────────────────────────────
function ws_get(string $path, array $params, int $timeout = 180) {
    $url = rtrim(WINICARI_WEBSERVICE_URL, '/') . $path . '?' . http_build_query($params);
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => $timeout,
        CURLOPT_CONNECTTIMEOUT => 10,
    ]);
    $body = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err = curl_error($ch);
    curl_close($ch);
    if ($body === false || $status >= 400) {
        throw new RuntimeException("web service $path: " . ($err ?: "HTTP $status"));
    }
    return json_decode($body, true);
}

function render_req(string $method, string $path, $json_body = null, int $timeout = 300) {
    $ch = curl_init(rtrim(WINICARI_API_BASE, '/') . $path);
    $headers = ['X-API-Key: ' . WINICARI_API_KEY];
    $opts = [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => $timeout,
        CURLOPT_CONNECTTIMEOUT => 15,
        CURLOPT_CUSTOMREQUEST => $method,
    ];
    if ($json_body !== null) {
        $headers[] = 'Content-Type: application/json';
        $opts[CURLOPT_POSTFIELDS] = json_encode($json_body);
    }
    $opts[CURLOPT_HTTPHEADER] = $headers;
    curl_setopt_array($ch, $opts);
    $body = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err = curl_error($ch);
    curl_close($ch);
    if ($body === false || $status >= 400) {
        throw new RuntimeException("API $path: " . ($err ?: "HTTP $status -- $body"));
    }
    return json_decode($body, true);
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
