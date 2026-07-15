<?php
require_once __DIR__ . '/config.php';
require_once __DIR__ . '/session.php';

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
            <span class="wc-brand-icon">🚌</span>
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
            <span class="wc-brand-icon">🛡️</span>
            <div>
                <div class="wc-title">Détection d'anomalies</div>
                <div class="wc-subtitle">Trajets signalés par Isolation Forest + autoencodeur LSTM</div>
            </div>
        </div>
        <div class="wc-header-right">
            <span class="wc-company-badge"><?= htmlspecialchars($company) ?></span>
            <a href="?change_company=1" class="wc-link-muted" title="Développement/test uniquement">changer</a>
        </div>
    </header>

    <nav class="wc-tabs" role="tablist">
        <button class="wc-tab active" data-view="trips" role="tab">📋 Trajets &amp; analyse</button>
        <button class="wc-tab" data-view="trends" role="tab">📊 Tendances</button>
        <button class="wc-tab" data-view="tickets" role="tab">🎫 Anomalies billetterie</button>
        <button class="wc-tab" data-view="drivers" role="tab">🪪 Chauffeurs</button>
    </nav>

    <main id="wc-view-root" class="wc-view-root">
        <!-- populated by app.js -->
    </main>

    <div id="wc-loading" class="wc-loading-overlay" hidden>
        <div class="wc-brain">
            <svg viewBox="0 0 100 80" class="wc-brain-svg" aria-hidden="true">
                <path class="wc-brain-path wc-brain-left"
                      d="M45 10 C30 8, 15 18, 15 32 C15 40, 20 44, 18 50 C14 58, 20 66, 30 66 C34 70, 42 72, 46 66 L46 14 Z"/>
                <path class="wc-brain-path wc-brain-right"
                      d="M55 10 C70 8, 85 18, 85 32 C85 40, 80 44, 82 50 C86 58, 80 66, 70 66 C66 70, 58 72, 54 66 L54 14 Z"/>
                <circle class="wc-synapse s1" cx="28" cy="28" r="2.4"/>
                <circle class="wc-synapse s2" cx="38" cy="45" r="2"/>
                <circle class="wc-synapse s3" cx="24" cy="52" r="2.2"/>
                <circle class="wc-synapse s4" cx="72" cy="28" r="2.2"/>
                <circle class="wc-synapse s5" cx="62" cy="45" r="2"/>
                <circle class="wc-synapse s6" cx="76" cy="52" r="2.4"/>
            </svg>
        </div>
        <div class="wc-loading-text" id="wc-loading-text">Analyse en cours…</div>
        <div class="wc-loading-bar"><div class="wc-loading-bar-fill"></div></div>
    </div>

<?php endif; ?>

</div>

<script>
    const WINICARI_COMPANY = <?= json_encode($company) ?>;
    const WINICARI_PROXY = 'proxy.php';
</script>
<script src="assets/app.js"></script>
</body>
</html>
