<?php
/**
 * Per-company session scoping.
 *
 * `winicari_company` is a PHP session variable, not a hardcoded constant, precisely so
 * the company's own login/dashboard code can set it (e.g. right after their user logs
 * in: `$_SESSION['winicari_company'] = $theirLoggedInOperatorName;`) without touching
 * this file at all. Until that's wired up, or for local testing with several operators,
 * leaving it unset falls back to the picker screen in index.php.
 */

if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

function winicari_current_company(): ?string {
    $c = $_SESSION['winicari_company'] ?? null;
    return ($c === null || $c === '') ? null : $c;
}

function winicari_set_company(string $company): void {
    $_SESSION['winicari_company'] = $company;
}

function winicari_clear_company(): void {
    unset($_SESSION['winicari_company']);
}

// ── Langue de l'interface (fr/ar) ────────────────────────────────────────────────────
// Session, pas cookie/query-string permanent : même pattern que winicari_company --
// l'intégration finale pourrait aussi la fixer directement depuis le compte de
// l'opérateur (langue préférée du client) sans toucher ce fichier.
const WINICARI_SUPPORTED_LANGS = ['fr', 'ar'];

function winicari_current_lang(): string {
    $l = $_SESSION['winicari_lang'] ?? 'fr';
    return in_array($l, WINICARI_SUPPORTED_LANGS, true) ? $l : 'fr';
}

function winicari_set_lang(string $lang): void {
    if (in_array($lang, WINICARI_SUPPORTED_LANGS, true)) {
        $_SESSION['winicari_lang'] = $lang;
    }
}
