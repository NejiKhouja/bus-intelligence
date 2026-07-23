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
    // Ce déclencheur DOIT tourner aussi sur le serveur de dev `php -S` : c'est PRÉCISÉMENT
    // la machine locale (seule à joindre le webservice 102.128.57.59) qui pousse les données
    // de la veille vers l'API Render (qui, elle, ne peut PAS joindre le webservice). Le
    // sauter couperait l'alimentation en données live -> la veille afficherait 0 trajet pour
    // toutes les sociétés (régression 2026-07-23). Le déclenchement lui-même est désormais
    // « tir et oubli » sur socket (voir plus bas) : ~1 ms, il ne retarde plus la page.
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

    // Tir et oubli VRAIMENT non bloquant (retour utilisateur 2026-07-23 : "ne bloque pas
    // l'appli, affiche la page tout de suite et fais l'envoi vers Render en arrière-plan").
    // On ENVOIE la requête sur une socket brute puis on ferme SANS attendre la réponse --
    // relay.php tourne seul (ignore_user_abort, verrou, vérif "déjà poussé ?"). L'ancien
    // curl, lui, ATTENDAIT la réponse jusqu'à 400 ms : sur le serveur de dev `php -S`
    // (mono-thread) relay.php ne démarre même pas tant qu'index.php n'a pas fini, donc ces
    // 400 ms étaient du temps mort pur devant la page. Ici : le noyau accepte la connexion
    // dans son backlog, on pousse ~120 octets de requête, on ferme -> ~1 ms, aucune latence
    // ajoutée ; relay.php est traité juste après, sans le visiteur.
    $parts = parse_url($url);
    if ($parts === false || empty($parts['host'])) {
        return;
    }
    $https  = (($parts['scheme'] ?? 'http') === 'https');
    $host   = $parts['host'];
    $port   = $parts['port'] ?? ($https ? 443 : 80);
    $target = ($parts['path'] ?? '/relay.php') . (isset($parts['query']) ? '?' . $parts['query'] : '');
    $remote = ($https ? 'ssl://' : '') . $host;

    $errno = 0;
    $errstr = '';
    // 1 s de timeout UNIQUEMENT pour établir la connexion (instantané en local) ; on n'attend
    // jamais la réponse. @ + garde : si ça échoue (backlog plein, refus...), on abandonne
    // silencieusement -- le cron/relais manuel reste le filet de sécurité.
    $fp = @fsockopen($remote, $port, $errno, $errstr, 1.0);
    if ($fp) {
        $req  = "GET $target HTTP/1.1\r\n";
        $req .= "Host: $host\r\n";
        $req .= "Connection: Close\r\n";
        $req .= "\r\n";
        fwrite($fp, $req);
        fclose($fp);
    }
}

winicari_maybe_trigger_relay();
