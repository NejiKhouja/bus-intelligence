<?php
/**
 * Déclencheur du relais À LA VISITE (inclus par index.php) -- décision utilisateur
 * 2026-07-18 : pas de planificateur à configurer ; le premier visiteur de la journée
 * déclenche le push des données d'hier vers l'API, les visiteurs suivants trouvent les
 * données déjà en place.
 *
 * Coût pour le visiteur : ~0. On lance une requête HTTP vers relay.php?auto=1 avec un
 * timeout de ~400 ms puis on l'abandonne -- relay.php continue seul en arrière-plan
 * (ignore_user_abort). Le visiteur reçoit sa page immédiatement, avec les données
 * actuelles (la veille apparaît à la visite/au rafraîchissement suivant, une fois le
 * push terminé -- typiquement ~1-2 min).
 *
 * Anti-rafale : au plus UNE tentative de déclenchement toutes les 10 minutes (marqueur
 * fichier local) -- relay.php a en plus son propre verrou (une seule exécution à la
 * fois) et sa propre vérification "l'API a-t-elle déjà ce jour ?" (voir relay.php,
 * mode auto). Trois couches, donc N visiteurs simultanés = 1 seul push réel.
 */

function winicari_maybe_trigger_relay(): void
{
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

    // URL de relay.php sur CE même serveur. Détection standard scheme/host/chemin --
    // définissez WINICARI_RELAY_URL dans config.php si votre hébergement est derrière
    // un proxy qui fausse cette détection.
    if (defined('WINICARI_RELAY_URL') && WINICARI_RELAY_URL) {
        $url = WINICARI_RELAY_URL;
    } else {
        $scheme = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off') ? 'https' : 'http';
        $host = $_SERVER['HTTP_HOST'] ?? 'localhost';
        $dir = rtrim(dirname($_SERVER['SCRIPT_NAME'] ?? '/'), '/\\');
        $url = "$scheme://$host$dir/relay.php";
    }
    $url .= '?' . http_build_query(['auto' => 1, 'key' => WINICARI_API_KEY]);

    // Tir et oubli : timeout court, résultat ignoré -- relay.php survit à la coupure.
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT_MS => 400,
        CURLOPT_CONNECTTIMEOUT_MS => 300,
        CURLOPT_NOSIGNAL => 1, // requis pour les timeouts < 1s (comportement libcurl)
        CURLOPT_SSL_VERIFYPEER => false, // auto-requête locale, éventuel certif interne
    ]);
    curl_exec($ch);
    curl_close($ch);
}

winicari_maybe_trigger_relay();
