<?php


if (session_status() === PHP_SESSION_NONE) {
    session_start();
}
// $_SESSION['nom']
function winicari_current_company(): ?string {
    $c = $_SESSION['nom'] ?? null;
    return ($c === null || $c === '') ? null : $c;
}

function winicari_set_company(string $company): void {
    $_SESSION['nom'] = $company;
}

function winicari_clear_company(): void {
    unset($_SESSION['nom']);
}

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
