<?php
require_once __DIR__ . '/config.php';
require_once __DIR__ . '/session.php';
require_once __DIR__ . '/i18n.php';
// Déclenche (au plus 1x/10 min, en arrière-plan, coût ~0 pour le visiteur) le push des
// données de la veille vers l'API si elles n'y sont pas encore -- voir autorun.php.
require_once __DIR__ . '/autorun.php';

// Dev/testing convenience: the picker form below posts here to set the session company.
// In the company's real integration, their own login flow sets $_SESSION['winicari_company']
// directly and this form is simply never reached (see the `if ($company === null)` branch).
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['winicari_company'])) {
    winicari_set_company($_POST['winicari_company']);
    header('Location: ' . strtok($_SERVER['REQUEST_URI'], '?'));
    exit;
}
if (isset($_GET['change_company'])) {
    winicari_clear_company();
    header('Location: ' . strtok($_SERVER['REQUEST_URI'], '?'));
    exit;
}
// Sélecteur de langue (dropdown dans l'en-tête, voir plus bas) -- POST vers cette même
// page, comme le picker d'opérateur ci-dessus ; stocké en session (winicari_set_lang),
// donc persiste tant que la session dure, pas besoin de le repasser dans chaque lien.
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['winicari_lang'])) {
    winicari_set_lang($_POST['winicari_lang']);
    header('Location: ' . strtok($_SERVER['REQUEST_URI'], '?'));
    exit;
}

$company = winicari_current_company();
$lang = winicari_current_lang();
$dir = $lang === 'ar' ? 'rtl' : 'ltr';
?>
<!DOCTYPE html>
<html lang="<?= htmlspecialchars($lang) ?>" dir="<?= $dir ?>">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title><?= htmlspecialchars(wt('page_title')) ?></title>
<!-- Ouvre tôt la connexion (DNS + TLS) vers les CDN utilisés à la demande (cartes Leaflet,
     tuiles OpenStreetMap, graphiques Chart.js) -- la 1re carte/graphique ouvert ne paie
     alors plus la poignée de main réseau (retour utilisateur 2026-07-23 : chargement lent). -->
<link rel="preconnect" href="https://unpkg.com" crossorigin>
<link rel="preconnect" href="https://tile.openstreetmap.org" crossorigin>
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="preconnect" href="https://maps.googleapis.com" crossorigin>
<link rel="preconnect" href="https://maps.gstatic.com" crossorigin>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="wc-embed" dir="<?= $dir ?>">

<?php
// Sélecteur de langue -- même `<select>` réutilisé sur les deux écrans (picker et app
// principale), soumis vers cette page (voir le handler POST winicari_lang plus haut).
$langSwitchHtml = '
    <form method="post" class="wc-lang-switch">
        <select name="winicari_lang" onchange="this.form.submit()" aria-label="' . htmlspecialchars(wt('lang_label')) . '">
            <option value="fr"' . ($lang === 'fr' ? ' selected' : '') . '>Français</option>
            <option value="ar"' . ($lang === 'ar' ? ' selected' : '') . '>العربية</option>
        </select>
    </form>';
?>

<?php if ($company === null): ?>

    <!-- ── Company picker (dev/testing fallback — see session.php) ───────────────── -->
    <div class="wc-card wc-picker">
        <?= $langSwitchHtml ?>
        <div class="wc-brand">
            <span class="wc-brand-icon"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="3" width="16" height="14" rx="2"/><line x1="4" y1="11" x2="20" y2="11"/><circle cx="8" cy="19" r="1.6"/><circle cx="16" cy="19" r="1.6"/></svg></span>
            <span>WiniCari AI</span>
        </div>
        <h2><?= htmlspecialchars(wt('picker_heading')) ?></h2>
        <p class="wc-muted"><?= wt('picker_body') ?></p>
        <form method="post" class="wc-picker-form">
            <select name="winicari_company" required>
                <option value="" disabled selected><?= htmlspecialchars(wt('picker_select')) ?></option>
                <?php foreach (WINICARI_FALLBACK_COMPANIES as $c): ?>
                    <option value="<?= htmlspecialchars($c) ?>"><?= htmlspecialchars($c) ?></option>
                <?php endforeach; ?>
            </select>
            <button type="submit"><?= htmlspecialchars(wt('picker_continue')) ?></button>
        </form>
    </div>

<?php else: ?>

    <header class="wc-header">
        <div class="wc-brand">
            <span class="wc-brand-icon"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></span>
            <div>
                <div class="wc-title"><?= htmlspecialchars(wt('header_title')) ?></div>
                <div class="wc-subtitle"><?= htmlspecialchars(wt('header_subtitle')) ?></div>
            </div>
        </div>
        <div class="wc-header-right">
            <?= $langSwitchHtml ?>
            <span class="wc-company-badge"><?= htmlspecialchars($company) ?></span>
            <a href="?change_company=1" class="wc-link-muted" title="Développement/test uniquement"><?= htmlspecialchars(wt('change_company')) ?></a>
        </div>
    </header>

    <nav class="wc-tabs" role="tablist">
        <button class="wc-tab active" data-view="trips" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg><?= htmlspecialchars(wt('tab_trips')) ?></button>
        <button class="wc-tab" data-view="trends" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="20" x2="6" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="18" y1="20" x2="18" y2="14"/></svg><?= htmlspecialchars(wt('tab_trends')) ?></button>
        <button class="wc-tab" data-view="tickets" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 9a3 3 0 0 1 0 6v4a1 1 0 0 0 1 1h18a1 1 0 0 0 1-1v-4a3 3 0 0 1 0-6V5a1 1 0 0 0-1-1H3a1 1 0 0 0-1 1z"/><line x1="13" y1="5" x2="13" y2="19" stroke-dasharray="2 3"/></svg><?= htmlspecialchars(wt('tab_tickets')) ?></button>
        <button class="wc-tab" data-view="drivers" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><circle cx="8" cy="11" r="2"/><path d="M5.5 17c.5-2 4.5-2 5 0"/><line x1="14" y1="9" x2="19" y2="9"/><line x1="14" y1="13" x2="18" y2="13"/></svg><?= htmlspecialchars(wt('tab_drivers')) ?></button>
        <button class="wc-tab" data-view="traceline" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg><?= htmlspecialchars(wt('tab_traceline')) ?></button>
    </nav>

    <main id="wc-view-root" class="wc-view-root">
        <!-- populated by app.js -->
    </main>

    <p class="wc-footer-disclaimer"><?= htmlspecialchars(wt('footer_disclaimer')) ?></p>

<?php endif; ?>

</div>

<script>
    const WINICARI_COMPANY = <?= json_encode($company) ?>;
    const WINICARI_PROXY = 'proxy.php';
    const WINICARI_LANG = <?= json_encode($lang) ?>;
    // Voir config.php -- clé publique par design (SDK Google Maps JS), restreinte par
    // référent HTTP côté Google Cloud Console, pas cachée comme WINICARI_API_KEY (qui,
    // lui, ne quitte jamais proxy.php -- voir sa note en tête de ce fichier).
    const WINICARI_GMAPS_KEY = <?= json_encode(
        (defined('WINICARI_GOOGLE_MAPS_KEY') && WINICARI_GOOGLE_MAPS_KEY !== 'REPLACE_ME') ? WINICARI_GOOGLE_MAPS_KEY : null
    ) ?>;
</script>
<script src="assets/i18n.js"></script>
<script src="assets/app.js"></script>
</body>
</html>
