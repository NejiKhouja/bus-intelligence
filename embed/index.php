<?php
require_once __DIR__ . '/config.php';
require_once __DIR__ . '/session.php';
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

$company = winicari_current_company();
?>
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WiniCari AI — Détection d'anomalies</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="wc-embed">

<?php if ($company === null): ?>

    <!-- ── Company picker (dev/testing fallback — see session.php) ───────────────── -->
    <div class="wc-card wc-picker">
        <div class="wc-brand">
            <span class="wc-brand-icon"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="3" width="16" height="14" rx="2"/><line x1="4" y1="11" x2="20" y2="11"/><circle cx="8" cy="19" r="1.6"/><circle cx="16" cy="19" r="1.6"/></svg></span>
            <span>WiniCari AI</span>
        </div>
        <h2>Choisir un opérateur</h2>
        <p class="wc-muted">
            Aucun opérateur n'est associé à cette session. Dans l'intégration finale,
            votre système de connexion définira <code>$_SESSION['winicari_company']</code>
            automatiquement — ce choix manuel n'est là que pour le développement/test.
        </p>
        <form method="post" class="wc-picker-form">
            <select name="winicari_company" required>
                <option value="" disabled selected>Sélectionner…</option>
                <?php foreach (WINICARI_FALLBACK_COMPANIES as $c): ?>
                    <option value="<?= htmlspecialchars($c) ?>"><?= htmlspecialchars($c) ?></option>
                <?php endforeach; ?>
            </select>
            <button type="submit">Continuer</button>
        </form>
    </div>

<?php else: ?>

    <!-- ── Main app ─────────────────────────────────────────────────────────────── -->
    <header class="wc-header">
        <div class="wc-brand">
            <span class="wc-brand-icon"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></span>
            <div>
                <div class="wc-title">Détection d'anomalies</div>
                <div class="wc-subtitle">Suivi automatique des trajets et alertes en cas de comportement inhabituel</div>
            </div>
        </div>
        <div class="wc-header-right">
            <span class="wc-company-badge"><?= htmlspecialchars($company) ?></span>
            <a href="?change_company=1" class="wc-link-muted" title="Développement/test uniquement">changer</a>
        </div>
    </header>

    <nav class="wc-tabs" role="tablist">
        <button class="wc-tab active" data-view="trips" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>Trajets signalés</button>
        <button class="wc-tab" data-view="trends" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="20" x2="6" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="18" y1="20" x2="18" y2="14"/></svg>Tendances</button>
        <button class="wc-tab" data-view="tickets" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 9a3 3 0 0 1 0 6v4a1 1 0 0 0 1 1h18a1 1 0 0 0 1-1v-4a3 3 0 0 1 0-6V5a1 1 0 0 0-1-1H3a1 1 0 0 0-1 1z"/><line x1="13" y1="5" x2="13" y2="19" stroke-dasharray="2 3"/></svg>Anomalies billetterie</button>
        <button class="wc-tab" data-view="drivers" role="tab"><svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><circle cx="8" cy="11" r="2"/><path d="M5.5 17c.5-2 4.5-2 5 0"/><line x1="14" y1="9" x2="19" y2="9"/><line x1="14" y1="13" x2="18" y2="13"/></svg>Chauffeurs</button>
    </nav>

    <main id="wc-view-root" class="wc-view-root">
        <!-- populated by app.js -->
    </main>

<?php endif; ?>

</div>

<script>
    const WINICARI_COMPANY = <?= json_encode($company) ?>;
    const WINICARI_PROXY = 'proxy.php';
</script>
<script src="assets/app.js"></script>
</body>
</html>
