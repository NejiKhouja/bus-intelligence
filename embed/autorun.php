<?php
require_once __DIR__ . '/http.php';
register_shutdown_function('winicari_maybe_trigger_relay');

function winicari_maybe_trigger_relay(): void
{

    if (function_exists('fastcgi_finish_request')) {
        @fastcgi_finish_request();
    }

    if (!defined('WINICARI_WEBSERVICE_URL') || !WINICARI_WEBSERVICE_URL) {
        return; // pas de webservices configurés sur ce serveur -> relais impossible ici
    }

    $var_dir = __DIR__ . '/var';
    if (!is_dir($var_dir) && !@mkdir($var_dir, 0775, true)) {
        return;
    }
    $marker = $var_dir . '/relay_trigger.last';
    if (file_exists($marker) && (time() - filemtime($marker)) < 600) {
        return; // déjà tenté il y a moins de 10 min
    }
    @touch($marker);

    if (defined('WINICARI_RELAY_URL') && WINICARI_RELAY_URL) {
        $url = WINICARI_RELAY_URL;
    } else {
        $scheme = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off') ? 'https' : 'http';
        $host = $_SERVER['HTTP_HOST'] ?? 'localhost';
        $dir = rtrim(dirname($_SERVER['SCRIPT_NAME'] ?? '/'), '/\\');
        $url = "$scheme://$host$dir/relay.php";
    }
    $url .= '?' . http_build_query(['auto' => 1, 'key' => WINICARI_API_KEY]);

    winicari_http_request($url, [], 'GET', null, 0.3, false); // auto-requête locale, éventuel certif interne
}
