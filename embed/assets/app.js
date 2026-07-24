/**
 * WiniCari AI — embeddable anomaly-detection widget.
 * Vanilla JS, no build step. Talks only to proxy.php (same-origin) -- see proxy.php's
 * header comment for why the Render API is never called directly from here.
 *
 * Charts use Chart.js via CDN (no bundler needed, just a <script> tag) -- loaded lazily
 * the first time a view that needs a chart is opened, not on initial page load.
 */
(() => {
"use strict";

// ── Labels (traduits via assets/i18n.js -- clés topfeat_*/sev_*) ───────────────────────
const TOP_FEATURE_LABELS = {
    max_dwell_s: t("topfeat_max_dwell_s"),
    total_elapsed: t("topfeat_total_elapsed"),
    mean_dwell_s: t("topfeat_mean_dwell_s"),
    dist_m_max: t("topfeat_dist_m_max"),
    match_rate: t("topfeat_match_rate"),
    n_stops: t("topfeat_n_stops"),
    max_dark_s: t("topfeat_max_dark_s"),
    terminus_idle_min: t("topfeat_terminus_idle_min"),
    elapsed_vs_bus_z: t("topfeat_elapsed_vs_bus_z"),
    elapsed_vs_line_z: t("topfeat_elapsed_vs_line_z"),
    unofficial_detour: t("topfeat_unofficial_detour"),
    null: t("topfeat_other"),
};
const SEV_META = {
    high: { label: t("sev_high"), cls: "high" },
    medium: { label: t("sev_medium"), cls: "medium" },
    low: { label: t("sev_low"), cls: "low" },
};

// ── Explications au survol des raisons (mirrors src/dashboard/i18n.py's REASON_HELP /
// exp_* keys) -- portées ici pour donner la même explicabilité que le dashboard
// Streamlit (retour utilisateur 2026-07-19 : "expliquer tout avec le survol comme dans
// app.py"). Affichées via l'attribut title (tooltip natif), même convention que
// .wc-metric-label[title] déjà utilisée pour "Taux d'anomalie".
const REASON_HELP = {
    max_dwell_s: t("exp_max_dwell_s"),
    total_elapsed: t("exp_total_elapsed"),
    mean_dwell_s: t("exp_mean_dwell_s"),
    dist_m_max: t("exp_dist_m_max"),
    match_rate: t("exp_match_rate"),
    n_stops: t("exp_n_stops"),
    max_dark_s: t("exp_max_dark_s"),
    terminus_idle_min: t("exp_terminus_idle_min"),
    elapsed_vs_bus_z: t("exp_elapsed_vs_bus_z"),
    elapsed_vs_line_z: t("exp_elapsed_vs_line_z"),
    unofficial_detour: t("exp_unofficial_detour"),
    // Note informationnelle de perte de signal (voir models/anomaly.py::explain_trips) --
    // même explication que la raison "max_dark_s" normale, seul le phrasing affiché change.
    max_dark_s_info: t("exp_max_dark_s"),
};
const REASON_HELP_DEFAULT = t("exp_default");
const FORMULA_HELP = t("formula_help");
// Aide au survol de la 3e métrique « Activité GPS vérifiable » -- texte de
// verifiable_activity_caption dans src/dashboard/i18n.py (paramètres remplis à l'affichage).
function verifiableActivityHelp(est, dur, dwell, dark, match) {
    return t("verifiable_activity_help", {
        est, dur, dwell: dwell.toFixed(0), dark: dark.toFixed(0), match: (match * 100).toFixed(0),
    });
}

// ── Raisons du modèle (mirrors src/models/anomaly.py's _REASON_BUILDERS) ──────────────
// L'API renvoie `reason_features` (clé) + `reason_values` (valeur brute, MÊMES unités
// que les lambdas Python -- secondes pour *_s, minutes déjà pour terminus_idle_min,
// z-score brut pour elapsed_vs_*_z) plutôt qu'un texte français déjà composé, pour que
// CE widget puisse reconstruire la phrase dans la langue active au lieu d'afficher le
// français en dur (retour utilisateur 2026-07-22).
function formatReasonValue(feat, v) {
    switch (feat) {
        case "max_dwell_s": case "mean_dwell_s":
            return (v / 60).toFixed(feat === "mean_dwell_s" ? 1 : 0);
        case "total_elapsed": {
            const iv = Math.trunc(v);
            return `${Math.trunc(iv / 60)}h${String(iv % 60).padStart(2, "0")}`;
        }
        case "dist_m_max": return v.toFixed(0);
        case "match_rate": return (v * 100).toFixed(0);
        case "n_stops": return String(Math.trunc(v));
        case "max_dark_s": case "max_dark_s_info": return (v / 60).toFixed(0);
        case "elapsed_vs_bus_z": case "elapsed_vs_line_z": return `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;
        case "terminus_idle_min": return v.toFixed(0);
        default: return String(v);
    }
}
function reasonText(feat, v) {
    const val = formatReasonValue(feat, v);
    if (feat === "elapsed_vs_bus_z" || feat === "elapsed_vs_line_z") {
        return t(`reason_${feat}`, { val, dir: t(v > 0 ? "reason_longer" : "reason_shorter") });
    }
    return t(`reason_${feat}`, { val });
}

// ── Explications au survol des puces (mirrors src/dashboard/i18n.py's chip_*_help keys) ──
const CHIP_HELP = {
    origin_idle: t("chip_help_origin_idle"),
    end_idle: t("chip_help_end_idle"),
    real_stop: t("chip_help_real_stop"),
    signal_loss: t("chip_help_signal_loss"),
    dark_gap: t("chip_help_dark_gap"),
    farthest: t("chip_help_farthest"),
    off_route: t("chip_help_off_route"),
    suspect_coord: t("chip_help_suspect_coord"),
    detour: t("chip_help_detour"),
};

// ── Small utilities ─────────────────────────────────────────────────────────────────────
function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}
// Repli client si l'API ne renvoie pas encore `yesterday_date` (backend pas encore
// redéployé avec ce champ, ou réponse mise en cache par proxy.php AVANT le déploiement)
// -- sans ça, fmtDay(undefined) affichait juste "—" à la place de la date (retour
// utilisateur 2026-07-22). Même fuseau que le serveur visé (Tunisie), calcul local au
// navigateur donc potentiellement décalé de quelques heures autour de minuit, mais
// nettement mieux qu'un tiret vide.
function ymdYesterday() {
    const d = new Date();
    d.setDate(d.getDate() - 1);
    return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
}
function fmtDay(d) {
    if (!d) return "—";
    const s = String(d);
    const y = s.slice(0, 4), m = s.slice(4, 6), day = s.slice(6, 8);
    const MOIS = WC_MONTHS[WC_LANG] || WC_MONTHS.fr;
    const mi = parseInt(m, 10) - 1;
    return (mi >= 0 && mi < 12) ? `${parseInt(day, 10)} ${MOIS[mi]} ${y}` : s;
}
function fmtTime(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (isNaN(d)) return "—";
    return d.toLocaleTimeString(WC_LOCALE, { hour: "2-digit", minute: "2-digit" });
}
function fmtDuration(min) {
    min = Math.round(min || 0);
    const h = Math.floor(min / 60), m = min % 60;
    return h > 0 ? `${h}h${String(m).padStart(2, "0")}` : `${m} min`;
}
function el(html) {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
}

// ── Inline SVG icons (replaces emojis -- decision 2026-07-17) ──────────────────────────
// stroke: currentColor => each icon inherits its container's text color (banner/chip tints).
const _ICON_PATHS = {
    alert: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    fragment: '<circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/>',
    signal: '<circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49M7.76 16.24a6 6 0 0 1 0-8.49M19.07 4.93a10 10 0 0 1 0 14.14M4.93 19.07a10 10 0 0 1 0-14.14"/>',
    clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    pin: '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
    parking: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 17V7h4a3 3 0 0 1 0 6H9"/>',
    ban: '<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>',
    detour: '<polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/>',
    driver: '<rect x="2" y="4" width="20" height="16" rx="2"/><circle cx="8" cy="11" r="2"/><path d="M5.5 17c.5-2 4.5-2 5 0"/><line x1="14" y1="9" x2="19" y2="9"/><line x1="14" y1="13" x2="18" y2="13"/>',
    chart: '<line x1="6" y1="20" x2="6" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="18" y1="20" x2="18" y2="14"/>',
    check: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    up: '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    down: '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    search: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
};
function icon(name) {
    const p = _ICON_PATHS[name];
    if (!p) return "";
    return `<svg class="wc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg> `;
}
// Pulsing green dot for "live data" -- pure CSS, no SVG needed.
const LIVE_DOT = '<span class="wc-live-dot" aria-hidden="true"></span> ';

// ── API + inline loading block ──────────────────────────────────────────────────────────
const LOADING_MESSAGES = [
    t("loading_1"), t("loading_2"), t("loading_3"), t("loading_4"), t("loading_5"),
];
let _loadingTimer = null;

// Rendered INSIDE the given container (not a full-page overlay -- decision 2026-07-17:
// the fixed overlay blocked the whole page during a 30-40s analysis; inline, the rest of
// the widget and the host dashboard stay usable while only the results zone shows work).
function showLoadingIn(container) {
    clearInterval(_loadingTimer);
    container.innerHTML = `
    <div class="wc-loading-inline">
        <div class="wc-loader" aria-hidden="true">
            <div class="wc-loader-glow"></div>
            <div class="wc-loader-ring"></div>
        </div>
        <div class="wc-loading-text">${LOADING_MESSAGES[0]}</div>
        <div class="wc-loading-bar"><div class="wc-loading-bar-fill"></div></div>
    </div>`;
    const textEl = container.querySelector(".wc-loading-text");
    let i = 0;
    _loadingTimer = setInterval(() => {
        i = (i + 1) % LOADING_MESSAGES.length;
        // The block may have been replaced by results/error markup already -- the guard
        // avoids touching a detached node; the interval itself is cleared by the caller.
        if (textEl.isConnected) textEl.textContent = LOADING_MESSAGES[i];
    }, 2200);
}
function stopLoading() {
    clearInterval(_loadingTimer);
}

// Squelette de chargement (voir la note CSS .wc-skel) -- des cartes vides de la même
// forme que renderAlertCard, avec un balayage lumineux. `n` cartes, largeurs de lignes
// légèrement variées pour ne pas avoir l'air d'un pur pavé répété.
function skeletonCards(n = 4) {
    const widths = ["w70", "w55", "w40"];
    let html = "";
    for (let i = 0; i < n; i++) {
        html += `
        <div class="wc-skel-card" aria-hidden="true">
            <div class="wc-skel-head">
                <span class="wc-skel wc-skel-badge"></span>
                <span class="wc-skel wc-skel-title"></span>
                <span class="wc-skel wc-skel-date"></span>
            </div>
            <div class="wc-skel-metrics">
                <span class="wc-skel wc-skel-metric"></span>
                <span class="wc-skel wc-skel-metric"></span>
            </div>
            <span class="wc-skel wc-skel-line ${widths[i % 3]}"></span>
            <span class="wc-skel wc-skel-line ${widths[(i + 1) % 3]}"></span>
        </div>`;
    }
    return html;
}

// Squelette "carte avec graphique" (métriques + grand bloc) -- pour Tendances/Billetterie,
// dont le contenu principal est un graphique, pas une liste de cartes d'alerte.
function skeletonChartCard(height = 240) {
    return `
    <div class="wc-skel-card" aria-hidden="true">
        <div class="wc-skel-metrics">
            <span class="wc-skel wc-skel-metric"></span>
            <span class="wc-skel wc-skel-metric"></span>
            <span class="wc-skel wc-skel-metric"></span>
        </div>
        ${height > 0 ? `<span class="wc-skel" style="display:block;height:${height}px;border-radius:8px"></span>` : ""}
    </div>`;
}

// Squelette "tableau" (lignes horizontales) -- pour le classement des chauffeurs.
function skeletonTable(n = 6) {
    let html = `<div aria-hidden="true">`;
    for (let i = 0; i < n; i++) {
        html += `<span class="wc-skel wc-skel-line" style="width:${92 - (i % 3) * 9}%;height:13px;margin:11px 0"></span>`;
    }
    return html + `</div>`;
}

// Cache mémoire des réponses (2026-07-20, retour utilisateur : "ping le serveur une fois
// puis stocke la donnée -- en réalité on ne regarde que la journée d'hier, rien de neuf
// n'arrive en cours de session"). Une même requête (endpoint + paramètres identiques)
// n'est refaite qu'après 5 min -- entre-temps, ré-ouvrir un panneau, relancer la même
// analyse ou revenir sur un onglet répond instantanément depuis la mémoire. S'ajoute au
// cache fichier de proxy.php (qui, lui, sert TOUS les visiteurs) ; celui-ci évite même
// l'aller-retour HTTP local. Un rechargement de page vide ce cache -- c'est le geste
// naturel pour forcer du frais.
const _apiMemCache = new Map(); // url -> { t: Date.now(), data }
const _API_MEM_TTL_MS = 5 * 60 * 1000;

async function api(endpoint, params = {}, { fresh = false } = {}) {
    const qs = new URLSearchParams({ endpoint, ...cleanParams(params) });
    const url = `${WINICARI_PROXY}?${qs.toString()}`;
    // `fresh` (interrogation répétée de la veille, voir loadCurrent) SAUTE le cache mémoire
    // JS pour vraiment re-solliciter proxy.php -- mais SANS casser le cache fichier du proxy
    // (même URL, donc même clé) : on profite de son stale-while-revalidate au lieu de
    // marteler Render. Une réponse fraîche met quand même à jour le cache mémoire ci-dessous.
    const hit = _apiMemCache.get(url);
    if (!fresh && hit && (Date.now() - hit.t) < _API_MEM_TTL_MS) return hit.data;
    const res = await fetch(url);
    const cacheStatus = res.headers.get("X-Cache"); // HIT / STALE / MISS -- voir proxy.php
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    // Étiquette non énumérable-friendly : un simple champ suffit, aucun endpoint utilisé
    // ici ne retourne un tableau nu au premier niveau, donc jamais en conflit avec les
    // données réelles. Sert uniquement à afficher une note "résultat en cache" (voir
    // cacheNote ci-dessous) -- ignoré partout ailleurs.
    if (data && typeof data === "object") data.__cache = cacheStatus;
    if (endpoint !== "/health") _apiMemCache.set(url, { t: Date.now(), data });
    return data;
}

// ── Indicateur "serveur en veille" ──────────────────────────────────────────────────────
// L'API (Render, hébergement gratuit) se met en pause après une période sans trafic et met
// généralement 30 à 50s à redémarrer (voir CURLOPT_TIMEOUT => 90 dans proxy.php). Sans
// indication, la toute première requête après une pause ressemble à une page cassée plutôt
// qu'à une attente normale. Si l'appel n'a pas répondu après COLD_START_HINT_MS, on
// remplace le contenu de `container` (bandeau ou squelette, peu importe -- l'appelant
// réécrit ce même conteneur au succès/à l'échec de toute façon) par un message explicite +
// un chrono qui tourne. Cas normal (donnée déjà en cache) : le message n'a même pas le
// temps d'apparaître.
const COLD_START_HINT_MS = 3000;

function withColdStartHint(promise, container) {
    let settled = false;
    let interval = null;
    const start = Date.now();
    const timer = setTimeout(() => {
        if (settled) return;
        const tick = () => {
            if (!container.isConnected) return;
            const s = Math.round((Date.now() - start) / 1000);
            container.innerHTML = `
            <div class="wc-banner info wc-coldstart">
                <span class="wc-spin"></span>
                <div>
                    <strong>${t("coldstart_title")}</strong>
                    <div class="wc-muted">${t("coldstart_body")}</div>
                    <div class="wc-coldstart-bar"><span></span></div>
                    <div class="wc-muted">${t("coldstart_elapsed", { s })}</div>
                </div>
            </div>`;
        };
        tick();
        interval = setInterval(tick, 1000);
    }, COLD_START_HINT_MS);
    return promise.finally(() => {
        settled = true;
        clearTimeout(timer);
        if (interval) clearInterval(interval);
    });
}

// Note "résultat en cache" retirée de l'affichage (décision utilisateur 2026-07-19 :
// "je n'aime pas que ce message apparaisse") -- le cache stale-while-revalidate lui-même
// reste actif côté proxy.php (la donnée EST réellement rafraîchie en arrière-plan), on
// arrête juste de le dire à l'écran. `data.__cache` reste disponible sur chaque réponse
// si besoin de le réintroduire (ex. dans les devtools) sans toucher à api().
function cacheNote(_data) {
    return "";
}
function cleanParams(params) {
    const out = {};
    for (const [k, v] of Object.entries(params)) {
        if (v !== null && v !== undefined && v !== "") out[k] = v;
    }
    return out;
}

// Chart.js is loaded lazily -- only the Tendances/Chauffeurs views need it.
let _chartJsPromise = null;
function ensureChartJs() {
    if (window.Chart) return Promise.resolve();
    if (_chartJsPromise) return _chartJsPromise;
    _chartJsPromise = new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js";
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
    });
    return _chartJsPromise;
}
function horizontalBarChart(canvas, labels, values, color) {
    return new Chart(canvas, {
        type: "bar",
        data: { labels, datasets: [{ data: values, backgroundColor: color, borderRadius: 4 }] },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: "#8ea0bd" }, grid: { color: "#1a2c47" }, beginAtZero: true },
                y: { ticks: { color: "#c8d4e6" }, grid: { display: false } },
            },
            animation: { duration: 500 },
        },
    });
}
function verticalBarChart(canvas, labels, values, color) {
    return new Chart(canvas, {
        type: "bar",
        data: { labels, datasets: [{ data: values, backgroundColor: color, borderRadius: 4 }] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { ticks: { color: "#8ea0bd" }, grid: { color: "#1a2c47" }, beginAtZero: true },
                x: { ticks: { color: "#c8d4e6" }, grid: { display: false } },
            },
            animation: { duration: 500 },
        },
    });
}

// Volume de tickets par jour, coloré Normal/Anormal -- mirrors app.py's `figh` (px.bar,
// color="état"). `rows` = jours triés chronologiquement (voir /api/ticket-anomaly-explain).
function ticketVolumeChart(canvas, rows) {
    const colors = rows.map((r) => (r.anomaly ? "#ef4444" : "#22c55e"));
    return new Chart(canvas, {
        type: "bar",
        data: { labels: rows.map((r) => fmtDay(r.day)), datasets: [{ data: rows.map((r) => r.nbr_ticket), backgroundColor: colors, borderRadius: 3 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, title: { display: true, text: t("chart_tickets_per_day"), color: "#c8d4e6", font: { size: 12 } } },
            scales: {
                y: { ticks: { color: "#8ea0bd" }, grid: { color: "#1a2c47" }, beginAtZero: true },
                x: { ticks: { color: "#8ea0bd", maxRotation: 60, minRotation: 60, font: { size: 10 } }, grid: { display: false } },
            },
            animation: { duration: 400 },
        },
    });
}
// Prix moyen par jour (points colorés Normal/Anormal) vs médiane ligne (ligne pointillée) --
// mirrors app.py's `figf` (px.scatter + add_hline).
function ticketFareChart(canvas, rows, lineMedian) {
    const colors = rows.map((r) => (r.anomaly ? "#ef4444" : "#22c55e"));
    const datasets = [{
        label: t("legend_avg_price"), data: rows.map((r) => r.avg_fare), showLine: false,
        pointBackgroundColor: colors, pointBorderColor: colors, pointRadius: 4,
    }];
    if (lineMedian !== null && lineMedian !== undefined) {
        datasets.push({
            label: t("legend_line_median"), data: rows.map(() => lineMedian),
            borderColor: "#8ea0bd", borderDash: [5, 5], pointRadius: 0, fill: false,
        });
    }
    return new Chart(canvas, {
        type: "line",
        data: { labels: rows.map((r) => fmtDay(r.day)), datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: "#c8d4e6", font: { size: 10 } } },
                title: { display: true, text: t("chart_avg_price_vs_median"), color: "#c8d4e6", font: { size: 12 } },
            },
            scales: {
                y: { ticks: { color: "#8ea0bd" }, grid: { color: "#1a2c47" }, beginAtZero: true },
                x: { ticks: { color: "#8ea0bd", maxRotation: 60, minRotation: 60, font: { size: 10 } }, grid: { display: false } },
            },
            animation: { duration: 400 },
        },
    });
}

// ── Leaflet (trip map) -- lazy CDN load, same pattern as Chart.js ───────────────────────
// Free OSM tiles, no API key -- mirrors the Streamlit map (plotly "open-street-map" style).
let _leafletPromise = null;
function ensureLeaflet() {
    if (window.L) return Promise.resolve();
    if (_leafletPromise) return _leafletPromise;
    _leafletPromise = new Promise((resolve, reject) => {
        const css = document.createElement("link");
        css.rel = "stylesheet";
        css.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(css);
        const s = document.createElement("script");
        s.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
    });
    return _leafletPromise;
}

// Mirrors src/dashboard/app.py::render_trip_map -- same colors, same thresholds, same
// visit-order semantics (RETOUR = seq décroissant), same legend, same detour overlay.
function stopColor(s) {
    if (s.coord_suspect) return "#9ca3af";
    if (!s.matched) return "#ef4444";
    if ((s.dark_min || 0) >= 5) return "#f59e0b";
    if ((s.dwell_min || 0) >= 10) return "#2563eb";
    return "#22c55e";
}
function renderTripMap(container, seqList, direction, detour) {
    const rows = (seqList || []).filter((s) => s.lat && s.lon);
    if (!rows.length) {
        container.innerHTML = `<div class="wc-banner info">${icon("info")}${t("map_no_coords")}</div>`;
        return;
    }
    const dot = (c) => `<span class="wc-legend-dot" style="background:${c}"></span>`;
    let html = `<p class="wc-muted wc-map-legend">${t("map_legend_caption")}<br>
        ${dot("#22c55e")}${t("map_dot_normal")} ${dot("#2563eb")}${t("map_dot_long_stop")} ${dot("#f59e0b")}${t("map_dot_signal_loss")} ${dot("#ef4444")}${t("map_dot_unserved")} ${dot("#9ca3af")}${t("map_dot_suspect")}</p>`;

    if (detour && detour.track) {
        const tt = (v) => v ? fmtTime(v) : "—";
        html += `<div class="wc-banner warn">${icon("detour")}${t("map_detour_banner", {
            left: `<strong>${tt(detour.left_at)}</strong>`, km: detour.distance_km, far: tt(detour.farthest_at),
            back: `<strong>${tt(detour.returned_at)}</strong>`, min: (detour.duration_min || 0).toFixed(0),
        })}</div>`;
    }
    html += `<div class="wc-map"></div>`;

    // Ordre de passage réel : les seq suivent la géométrie ALLER, donc un RETOUR visite
    // les arrêts en seq décroissant (même logique que le dashboard Streamlit).
    const sorted = [...rows].sort((a, b) => direction === "RETOUR" ? b.seq - a.seq : a.seq - b.seq);
    const tracked = sorted.filter((s) => s.arrival);
    if (tracked.length) {
        const first = tracked[0], last = tracked[tracked.length - 1];
        html += `<p class="wc-muted">${icon("pin")}${t("map_first_last_tracked", {
            t0: `<strong>${fmtTime(first.arrival)}</strong>`, stop0: esc(first.stop),
            t1: `<strong>${fmtTime(last.arrival)}</strong>`, stop1: esc(last.stop),
        })}</p>`;
    }
    container.innerHTML = html;

    const mapEl = container.querySelector(".wc-map");
    const map = L.map(mapEl, { scrollWheelZoom: false });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        maxZoom: 19,
    }).addTo(map);

    // Itinéraire prévu (ligne grise reliant les arrêts dans l'ordre de passage)
    const latlngs = sorted.map((s) => [s.lat, s.lon]);
    L.polyline(latlngs, { color: "#94a3b8", weight: 3, opacity: .8 }).addTo(map);

    // Détour : aller (orange) et retour (violet) tracés séparément, comme dans Streamlit --
    // un seul tracé superposerait les deux passages et rendrait le sens illisible.
    if (detour && detour.track) {
        const legs = [[detour.leg_out || detour.track, "#f97316"], [detour.leg_back || [], "#a855f7"]];
        for (const [leg, color] of legs) {
            if (leg && leg.length) {
                L.polyline(leg.map((p) => [p.lat, p.lon]), { color, weight: 3 }).addTo(map);
            }
        }
    }

    sorted.forEach((s, i) => {
        const color = stopColor(s);
        const size = Math.min(28, Math.max(16, (s.dwell_min || 0) + (s.dark_min || 0)));
        const iconEl = L.divIcon({
            className: "",
            html: `<div class="wc-map-stop" style="background:${color};width:${size}px;height:${size}px;line-height:${size}px">${i + 1}</div>`,
            iconSize: [size, size],
            iconAnchor: [size / 2, size / 2],
        });
        const popup = `<b>${i + 1} : ${esc(s.stop)}</b><br>
            ${t("map_popup_pass", { v: s.arrival ? fmtTime(s.arrival) : "—" })}<br>
            ${t("map_popup_dwell", { v: (s.dwell_min || 0).toFixed(1) })}<br>
            ${t("map_popup_dark", { v: (s.dark_min || 0).toFixed(1) })}<br>
            ${t("map_popup_dist", { v: (s.dist_m || 0).toFixed(0) })}<br>
            ${t("map_popup_tracked", { v: s.matched ? t("map_popup_yes") : t("map_popup_no") })}${s.coord_suspect ? `<br><i>${t("map_popup_suspect")}</i>` : ""}`;
        L.marker([s.lat, s.lon], { icon: iconEl }).addTo(map).bindPopup(popup);
    });

    // Départ / terminus dans le sens de circulation (mêmes couleurs que Streamlit)
    const mkEnd = (s, color, label) => {
        const ic = L.divIcon({
            className: "",
            html: `<div class="wc-map-end" style="border-color:${color}"><span style="color:${color}">${label}</span></div>`,
            iconSize: [70, 30], iconAnchor: [35, 34],
        });
        L.marker([s.lat, s.lon], { icon: ic, interactive: false }).addTo(map);
    };
    mkEnd(sorted[0], "#16a34a", `${t("map_departure")}${direction ? " (" + direction + ")" : ""}`);
    mkEnd(sorted[sorted.length - 1], "#dc2626", t("map_terminus"));

    map.fitBounds(L.latLngBounds(latlngs).pad(0.15));
}

// ── Trajet de référence « à quoi ressemble un trajet NORMAL sur cette ligne » ───────────
// Extrait du panneau "Expliquer un bus" (2026-07-23) pour être RÉUTILISÉ tel quel par le
// bouton du même nom sur les cartes de la vue d'ensemble -- même rendu (onglets ALLER/
// RETOUR, métriques, carte par direction). `container` reçoit le HTML ; `card:false` évite
// d'imbriquer une seconde .wc-card quand l'appelant fournit déjà un conteneur (bouton de
// carte). Peut lever (erreur API) -- à l'appelant de décider comment l'afficher.
async function renderReferenceTrip(container, line, { card = true, open = true, withStats = false } = {}) {
    container.innerHTML = `<div class="wc-card"><p class="wc-muted"><span class="wc-spin"></span> ${t("finding_reference_trip")}</p></div>`;
    // `withStats` (bouton "trajet normal" d'une carte) : on récupère AUSSI, en parallèle, le
    // bilan de la ligne (nb de trajets analysés + nb signalés) pour le montrer au-dessus du
    // trajet de référence -- l'utilisateur voit d'un coup sur combien de trajets repose le
    // jugement de la ligne (retour utilisateur 2026-07-23). Inutile dans le panneau
    // "Expliquer un bus", qui affiche déjà ce bilan séparément (loadLineVerdict).
    const [ref, pat] = await Promise.all([
        api("/api/reference-trip", { line }),
        withStats ? api("/api/anomaly-patterns", { line }).catch(() => null) : Promise.resolve(null),
    ]);
    const dirs = (ref || {}).directions || {};
    const dirNames = Object.keys(dirs);
    if (!dirNames.length) {
        container.innerHTML = `<div class="wc-banner info">${icon("info")}${t("no_reference_trip")}</div>`;
        return;
    }
    let inner = `<details class="wc-ref"${open ? " open" : ""}><summary>${icon("check")}${t("ref_trip_summary", { line: esc(line) })}</summary>`;
    if (withStats && pat && pat.total_trips) {
        inner += `<p class="wc-muted wc-ref-stats">${t("ref_line_stats", {
            trips: pat.total_trips.toLocaleString(WC_LOCALE),
            flagged: pat.total_anomalies,
            rate: (pat.overall_rate * 100).toFixed(1),
        })}</p>`;
    }
    if (dirNames.length === 2) {
        const at = dirs["ALLER"] && dirs["ALLER"].trip, rt = dirs["RETOUR"] && dirs["RETOUR"].trip;
        if (at && rt && at.bus === rt.bus && at.day === rt.day) {
            inner += `<p class="wc-muted">${t("ref_same_cycle", { bus: `<strong>${esc(at.bus)}</strong>`, day: `<strong>${fmtDay(at.day)}</strong>` })}</p>`;
        }
    } else {
        inner += `<p class="wc-muted">${t("ref_missing_other_dir", { dir: `<strong>${esc(dirNames[0])}</strong>` })}</p>`;
    }
    // Onglets ALLER/RETOUR côte à côte (mirrors Streamlit's st.tabs) -- un seul bouton = pas
    // de barre d'onglets, rien à sélectionner.
    if (dirNames.length > 1) {
        inner += `<div class="wc-ref-tabs">${dirNames.map((d, i) =>
            `<button type="button" class="wc-ref-tab${i === 0 ? " active" : ""}" data-dir="${esc(d)}">${esc(d)}</button>`
        ).join("")}</div>`;
    }
    for (const d of dirNames) {
        const rt = dirs[d].trip;
        // Un trajet de référence peut être PARTIEL quand la direction n'a (quasi) aucun
        // trajet complet -- fait des DONNÉES, pas un bug d'affichage.
        const partialNote = (rt.is_full === false && rt.geometry_stops)
            ? `<div class="wc-banner info">${icon("info")}${t("ref_partial_coverage", { covered: `<strong>${rt.covered_stops}</strong>`, total: rt.geometry_stops })}</div>`
            : "";
        inner += `
        <div class="wc-ref-dir" data-dir="${esc(d)}"${d === dirNames[0] ? "" : " hidden"}>
            ${dirNames.length > 1 ? "" : `<h4>${esc(d)}</h4>`}
            <p class="wc-muted">${t("ref_trip_caption", { match: (rt.match_rate * 100).toFixed(0) })}</p>
            ${partialNote}
            <div class="wc-metrics">
                <div class="wc-metric"><div class="wc-metric-label">${t("ref_bus")}</div><div class="wc-metric-value">${esc(rt.bus)}</div></div>
                <div class="wc-metric"><div class="wc-metric-label">${t("ref_day")}</div><div class="wc-metric-value" style="font-size:15px">${fmtDay(rt.day)}</div></div>
                <div class="wc-metric"><div class="wc-metric-label">${t("ref_duration")}</div><div class="wc-metric-value">${fmtDuration(rt.duration_min)}</div>
                    <div class="wc-metric-sub">${t("ref_line_median", { med: fmtDuration(rt.line_median_min) })}</div></div>
                <div class="wc-metric"><div class="wc-metric-label">${t("ref_stops_tracked")}</div><div class="wc-metric-value">${rt.n_stops}</div></div>
                <div class="wc-metric"><div class="wc-metric-label">${t("ref_avg_dwell")}</div><div class="wc-metric-value">${(rt.mean_dwell_min || 0).toFixed(1)} min</div></div>
                <div class="wc-metric"><div class="wc-metric-label">${t("metric_departure_arrival")}</div><div class="wc-metric-value" style="font-size:15px">${fmtTime(rt.trip_start)} → ${fmtTime(rt.trip_end)}</div></div>
            </div>
            ${rt.typical_terminus_idle_min !== null && rt.typical_terminus_idle_min !== undefined ? `
            <div class="wc-chip">${icon("parking")}${t("ref_typical_idle", {
                typ: rt.typical_terminus_idle_min.toFixed(0),
                extra: rt.service_not_closed_threshold_min ? t("ref_typical_idle_extra", { thr: rt.service_not_closed_threshold_min.toFixed(0) }) : "",
            })}</div>` : ""}
            <div class="wc-ref-map" data-dir="${esc(d)}"></div>
        </div>`;
    }
    inner += `</details>`;
    container.innerHTML = card ? `<div class="wc-card">${inner}</div>` : inner;

    const tabBtns = [...container.querySelectorAll(".wc-ref-tab")];
    tabBtns.forEach((btn) => btn.addEventListener("click", () => {
        const d = btn.dataset.dir;
        tabBtns.forEach((b) => b.classList.toggle("active", b === btn));
        for (const dd of dirNames) {
            container.querySelector(`.wc-ref-dir[data-dir="${CSS.escape(dd)}"]`).hidden = dd !== d;
        }
        renderRefMap(d);
    }));

    // Cartes chargées à l'OUVERTURE du bloc (Leaflet dans un <details> fermé ou un onglet
    // caché mesure une taille nulle et rend une carte grise) -- une seule fois chacune.
    const details = container.querySelector("details");
    const mapsBuilt = new Set();
    async function renderRefMap(d) {
        if (!details.open || mapsBuilt.has(d) || !dirs[d].sequence) return;
        mapsBuilt.add(d);
        try { await ensureLeaflet(); } catch { return; }
        const holder = container.querySelector(`.wc-ref-map[data-dir="${CSS.escape(d)}"]`);
        if (holder) renderTripMap(holder, dirs[d].sequence, d, null);
    }
    details.addEventListener("toggle", () => { if (details.open) renderRefMap(dirNames[0]); });
    // Ouvert par défaut (<details open>) -> rend la carte de la 1re direction tout de suite.
    renderRefMap(dirNames[0]);
}

// ── Shared: alert card rendering (used by Trips view, Chauffeurs view) ─────────────────
function rowCategories(a) {
    const cats = [a.top_feature ?? null];
    if (a.has_detour) cats.push("unofficial_detour");
    return cats;
}
function renderAlertCard(a, { showDriverStatsHint = true, withMap = false, overview = false } = {}) {
    const sev = SEV_META[a.severity] || SEV_META.medium;
    const dep = fmtTime(a.trip_start), arr = fmtTime(a.trip_end);
    const dur = a.trip_duration_min || a.total_elapsed_min || 0;
    const reasons = a.reasons || [];
    const ps = a.problem_stops || {};

    // 3e métrique « Activité GPS vérifiable » (≈ temps de conduite réel), affichée seulement
    // quand la durée est GONFLÉE et interprétable -- mêmes conditions que src/dashboard/app.py
    // (_show_est) : estimation dispo, immobilisation+trou de signal >= 15 min, durée au-dessus
    // de la médiane de la ligne, et trajet correctement suivi (>= 50% des arrêts). Sur un
    // trajet à peine suivi, « X min » est vrai mais se lit comme une absurdité -- on l'omet.
    const est = a.driving_time_est_min;
    const dwellM = a.max_dwell_min || 0;
    const darkM = a.max_dark_min || 0;
    const medM = a.line_median_elapsed_min;
    const matchR = a.match_rate || 0;
    const showEst = est !== undefined && est !== null && (dwellM + darkM) >= 15
                    && medM && dur > medM && matchR >= 0.5;

    // Confiance à nuancer quand CETTE LIGNE n'a pas son propre modèle dédié : pas assez de
    // trajets sur la ligne, donc elle s'appuie sur le modèle de l'opérateur (voire du réseau
    // entier). Le déclencheur est `model_line_dedicated` (niveau LIGNE) et NON `model_low_data`
    // (niveau OPÉRATEUR) -- ce dernier manquait les lignes peu fournies d'un opérateur par
    // ailleurs bien doté, ex. ligne 84 qui n'affichait donc aucun "!" (retour utilisateur
    // 2026-07-23). `=== false` (pas juste falsy) : ne s'affiche que si le champ est bien
    // présent dans la réponse, pas sur une vieille réponse en cache qui l'ignore.
    // Affichée sur TOUTES les cartes -- vue d'ensemble ET "Expliquer un bus" (l'utilisateur
    // veut le rappel partout). Icône "!" discrète au survol, jamais un bandeau ; formulée
    // comme un manque de recul DE DONNÉES, jamais comme un défaut du système de détection.
    const lowDataTip = (a.model_line_dedicated === false)
        ? t(a.model_if_dedicated ? "low_data_tip_partial" : "low_data_tip_full")
        : null;

    let html = `<div class="wc-alert-card">
        <div class="wc-alert-head">
            <span class="wc-badge ${sev.cls}">${sev.label}</span>
            ${lowDataTip ? `<span class="wc-low-data-flag" data-tip="${esc(lowDataTip)}" tabindex="0" aria-label="${esc(t("low_data_flag_label"))}">${icon("alert")}</span>` : ""}
            <span class="wc-alert-title">${t("alert_bus_line_dir", { bus: esc(a.bus), line: esc(a.line), dir: esc(a.dir) })}</span>
            <span class="wc-alert-date">${fmtDay(a.day)}</span>
        </div>
        <div class="wc-metrics">
            <div class="wc-metric"><div class="wc-metric-label" data-tip="${esc(FORMULA_HELP)}">${t("metric_trip_duration")}</div><div class="wc-metric-value">${fmtDuration(dur)}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">${t("metric_departure_arrival")}</div><div class="wc-metric-value" style="font-size:16px">${dep} → ${arr}</div></div>
            ${showEst ? `<div class="wc-metric"><div class="wc-metric-label" data-tip="${esc(verifiableActivityHelp(fmtDuration(est), fmtDuration(dur), dwellM, darkM, matchR))}">${t("metric_verifiable_activity")}</div><div class="wc-metric-value">≈ ${fmtDuration(est)}</div></div>` : ""}
        </div>`;

    if (a.top_feature !== undefined) {
        html += `<p class="wc-muted">${t("main_cause", { label: `<strong>${esc(TOP_FEATURE_LABELS[a.top_feature] || t("uncategorized"))}</strong>` })}</p>`;
    }
    // Boîtes qualité DÉTAILLÉES -- texte complet des q_* de src/dashboard/i18n.py (retour
    // utilisateur 2026-07-20 : les one-liners précédents ("Durée improbable — à vérifier")
    // n'expliquaient pas la CAUSE probable, alors que Streamlit dit explicitement
    // "chauffeur n'ayant pas clôturé son service", "trou de signal GPS", etc.).
    if (a.is_data_bug) html += `<div class="wc-banner error">${icon("alert")}${t("q_data_bug")}</div>`;
    else if (a.is_fragment) html += `<div class="wc-banner warn">${icon("fragment")}${t("q_fragment")}</div>`;
    else if (a.is_dark_inflated) html += `<div class="wc-banner info">${icon("signal")}${t("q_dark_inflated")}</div>`;
    else if (a.is_implausible) html += `<div class="wc-banner info">${icon("clock")}${t("q_implausible")}</div>`;
    else if (a.is_partial_coverage) html += `<div class="wc-banner info">${icon("pin")}${t("q_partial_coverage", { ns: a.n_stops ?? "?", mns: a.line_median_n_stops ?? "?" })}</div>`;

    const reasonFeats = a.reason_features || [];
    const reasonVals = a.reason_values || [];
    reasons.forEach((r, i) => {
        const feat = reasonFeats[i];
        const help = REASON_HELP[feat] || REASON_HELP_DEFAULT;
        // Texte reconstruit dans LA LANGUE ACTIVE à partir de (feature, valeur brute) --
        // `r` (texte déjà rendu en français par l'API, voir models/anomaly.py) sert
        // seulement de repli si `reason_values` manque (vieille réponse en cache) ou si
        // la feature est inconnue du frontend.
        const v = reasonVals[i];
        const text = (feat && v !== undefined && v !== null) ? reasonText(feat, v) : r;
        html += `<div class="wc-reason" data-tip="${esc(help)}">${esc(text)}</div>`;
    });

    // Puces (arrêts/segments concernés) -- chacune avec sa propre explication au survol
    // (mirrors src/dashboard/app.py's per-chip help=... calls). Stationnement terminus
    // DÉTAILLÉ (quel terminus, de quand à quand) : le chiffre "~N min avant départ/après
    // arrivée" de la raison modèle est mesuré sur les pings GPS (bus immobile au terminus,
    // temps DÉJÀ RETIRÉ de la durée du trajet affichée), et méritait d'être nommé +
    // horodaté au lieu d'un chiffre nu (retour utilisateur 2026-07-18).
    let originIdleShown = false, endIdleShown = false;
    if ((a.origin_idle_min || 0) >= 30 && a.origin_idle_stop) {
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.origin_idle)}">${icon("parking")}${t("chip_origin_idle", {
            stop: `<strong>${esc(a.origin_idle_stop)}</strong>`, min: `<strong>${a.origin_idle_min.toFixed(0)}</strong>`,
            from: fmtTime(a.origin_idle_from), to: dep,
        })}</div>`;
        originIdleShown = true;
    }
    if ((a.end_idle_min || 0) >= 30 && a.end_idle_stop) {
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.end_idle)}">${icon("parking")}${t("chip_end_idle", {
            stop: `<strong>${esc(a.end_idle_stop)}</strong>`, min: `<strong>${a.end_idle_min.toFixed(0)}</strong>`,
            from: arr, to: fmtTime(a.end_idle_to),
        })}</div>`;
        endIdleShown = true;
    }
    if (ps.longest_stop && ps.longest_stop.dwell_min >= 5) {
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.real_stop)}">${icon("parking")}${t("chip_longest_stop", {
            stop: `<strong>${esc(ps.longest_stop.stop)}</strong>`, min: ps.longest_stop.dwell_min.toFixed(0),
        })}</div>`;
        // Même arrêt que le stationnement terminus déjà affiché ci-dessus ? Très probablement
        // UNE seule immobilisation continue coupée en deux par un sursaut GPS isolé, pas deux
        // événements distincts -- sinon, hypothèse plus générique (détour possible).
        const sameOrigin = originIdleShown && ps.longest_stop.stop === a.origin_idle_stop;
        const sameEnd = endIdleShown && ps.longest_stop.stop === a.end_idle_stop;
        if (sameOrigin || sameEnd) {
            const sameIdle = sameOrigin ? a.origin_idle_min : a.end_idle_min;
            const total = sameIdle + ps.longest_stop.dwell_min;
            html += `<div class="wc-chip wc-chip-hint">&nbsp;&nbsp;↳ <em>${t("chip_same_terminus_hint", {
                stop: `<strong>${esc(ps.longest_stop.stop)}</strong>`, total: total.toFixed(0),
            })}</em></div>`;
        } else if (!originIdleShown && !endIdleShown) {
            html += `<div class="wc-chip wc-chip-hint">&nbsp;&nbsp;↳ <em>${t("chip_detour_hint")}</em></div>`;
        }
    }
    if (ps.signal_loss_stop) {
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.signal_loss)}">${icon("signal")}${t("chip_signal_loss_at", {
            stop: `<strong>${esc(ps.signal_loss_stop.stop)}</strong>`, min: ps.signal_loss_stop.dark_min.toFixed(0),
        })}</div>`;
    }
    // Trou de signal EN ROUTE (entre deux arrêts, jamais rattaché à l'attente d'un arrêt
    // matché) -- invisible au scan arrêt-par-arrêt ci-dessus, mais peut expliquer à lui
    // seul un mauvais taux de suivi + une durée gonflée.
    if (a.dark_gap_before_stop && (a.max_dark_min || 0) >= 15) {
        const after = a.dark_gap_after_stop;
        const gapKey = after ? "chip_dark_gap_between" : "chip_dark_gap_after_only";
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.dark_gap)}">${icon("signal")}${t(gapKey, {
            before: `<strong>${esc(a.dark_gap_before_stop)}</strong>`, after: `<strong>${esc(after)}</strong>`, min: a.max_dark_min.toFixed(0),
        })}</div>`;
    }
    if (ps.farthest_stop) {
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.farthest)}">${icon("pin")}${t("chip_farthest_at", {
            stop: `<strong>${esc(ps.farthest_stop.stop)}</strong>`, dist: ps.farthest_stop.dist_m.toFixed(0),
        })}</div>`;
    }
    if (ps.off_route_stops && ps.off_route_stops.length) {
        const others = (ps.off_route_count || ps.off_route_stops.length) - ps.off_route_stops.length;
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.off_route)}">${icon("ban")}${t("chip_off_route", {
            stops: esc(ps.off_route_stops.join(", ")), suffix: others > 0 ? t("and_others", { n: others }) : "",
        })}</div>`;
    }
    if (ps.suspect_coord_count) {
        html += `<div class="wc-chip" data-tip="${esc(CHIP_HELP.suspect_coord)}">${icon("info")}${t("chip_suspect_coord", { n: ps.suspect_coord_count })}</div>`;
    }
    if (a.has_detour && a.detour) {
        html += `<div class="wc-chip detour" data-tip="${esc(CHIP_HELP.detour)}">${icon("detour")}${t("chip_detour_confirmed", {
            km: a.detour.distance_km, min: a.detour.duration_min.toFixed(0),
        })}</div>`;
    }
    if (a.scheduled_departure && a.departure_delay_min !== null && a.departure_delay_min !== undefined && Math.abs(a.departure_delay_min) >= 3) {
        const late = a.departure_delay_min > 0;
        const key = late ? "chip_departure_late" : "chip_departure_early";
        html += `<div class="wc-chip">${icon("clock")}${t(key, {
            sched: `<strong>${esc(a.scheduled_departure)}</strong>`, dep: `<strong>${dep}</strong>`,
            min: Math.abs(a.departure_delay_min).toFixed(0),
            variant: a.schedule_multi_variant ? t("chip_departure_multi_variant") : "",
        })}</div>`;
    }
    if (a.driver_code) {
        html += `<div class="wc-chip driver">${icon("driver")}${t("chip_driver", { code: `<strong>${esc(a.driver_code)}</strong>` })}</div>
        <div class="wc-disclaimer">${t("chip_driver_disclaimer")}</div>`;
    }

    html += `</div>`;
    const card = el(html);

    // Rangée de boutons (carte + éventuellement "trajet normal de cette ligne").
    const actions = el(`<div class="wc-card-actions"></div>`);

    // Bouton carte -- même comportement que le dashboard Streamlit : la séquence par arrêt
    // est chargée À LA DEMANDE (/api/trip-detail), pas avec la liste (une carte Leaflet par
    // carte d'alerte chargée d'office serait ruineux en DOM et en appels API).
    if (withMap && a.trip_start) {
        const btn = el(`<button class="wc-btn-secondary wc-btn-map">${icon("pin")}${t("btn_show_map")}</button>`);
        const mapBox = el(`<div class="wc-map-holder"></div>`);
        let loaded = false;
        btn.addEventListener("click", async () => {
            if (loaded) { mapBox.hidden = !mapBox.hidden; return; }
            btn.disabled = true;
            mapBox.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> ${t("map_loading")}</p>`;
            try {
                // Leaflet (CDN) et le détail du trajet (API) chargés EN PARALLÈLE plutôt
                // qu'en série -- les deux latences se recouvrent au lieu de s'additionner
                // (retour utilisateur 2026-07-23 : "la carte met trop de temps à charger").
                const [, d] = await Promise.all([
                    ensureLeaflet(),
                    api("/api/trip-detail", { line: a.line, bus: a.bus, day: a.day, trip_start: a.trip_start }),
                ]);
                renderTripMap(mapBox, d.sequence, a.dir, (d.problem_stops || {}).unofficial_detour || a.detour);
                loaded = true;
            } catch (e) {
                mapBox.innerHTML = `<div class="wc-banner error">${icon("alert")}${t("map_unavailable", { msg: esc(e.message) })}</div>`;
            } finally {
                btn.disabled = false;
            }
        });
        actions.appendChild(btn);
        card.appendChild(actions);
        card.appendChild(mapBox);
    }

    // Bouton "trajet normal de cette ligne" -- RÉSERVÉ aux cartes de la vue d'ensemble
    // (chargement initial) : le panneau "Expliquer un bus" a déjà son propre bloc trajet de
    // référence au-dessus de la liste, l'y répéter par carte serait redondant. Ancre de
    // confiance : montre à quoi ressemble un trajet JUGÉ NORMAL sur la même ligne, à comparer
    // à l'anomalie (retour utilisateur 2026-07-23).
    if (overview && a.line) {
        const refBtn = el(`<button class="wc-btn-secondary">${icon("check")}${t("btn_show_reference")}</button>`);
        const refBox = el(`<div class="wc-map-holder"></div>`);
        let refLoaded = false;
        refBtn.addEventListener("click", async () => {
            if (refLoaded) { refBox.hidden = !refBox.hidden; return; }
            refBtn.disabled = true;
            refBox.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> ${t("finding_reference_trip")}</p>`;
            try {
                await renderReferenceTrip(refBox, a.line, { card: false, withStats: true });
                refLoaded = true;
            } catch (e) {
                refBox.innerHTML = `<div class="wc-banner error">${icon("alert")}${t("error_generic", { msg: esc(e.message) })}</div>`;
            } finally {
                refBtn.disabled = false;
            }
        });
        if (!actions.parentNode) card.appendChild(actions);
        actions.appendChild(refBtn);
        card.appendChild(refBox);
    }
    return card;
}

// "Filtrer par type d'anomalie" -- mirrors src/dashboard/app.py's category_filter(), a real
// st.multiselect (label above, selected options as removable tags, all selected by
// default, a dropdown to re-add a removed one) rather than a plain pill-toggle row
// (redesign 2026-07-19, retour utilisateur : "je veux que ça ressemble à celui de
// Streamlit"). Une ligne reste affichée si AU MOINS UNE de ses catégories est cochée.
function categoryFilterPills(container, anomalies, onChange) {
    const present = [...new Set(anomalies.flatMap(rowCategories))]
        .sort((a, b) => (TOP_FEATURE_LABELS[a] || "").localeCompare(TOP_FEATURE_LABELS[b] || ""));
    if (present.length < 2) { container.innerHTML = ""; return () => anomalies; }
    const selected = new Set(present);

    function filtered() { return anomalies.filter((a) => rowCategories(a).some((c) => selected.has(c))); }

    function draw() {
        const tagsHtml = present.filter((c) => selected.has(c)).map((c) => `
            <span class="wc-ms-tag" data-cat="${esc(c)}">${esc(TOP_FEATURE_LABELS[c] || c)}
                <button type="button" aria-label="${t("remove_tag")}">&times;</button>
            </span>`).join("");
        const remaining = present.filter((c) => !selected.has(c));
        const addHtml = remaining.length
            ? `<select class="wc-ms-add"><option value="" selected disabled>${t("add_category")}</option>${
                remaining.map((c) => `<option value="${esc(c)}">${esc(TOP_FEATURE_LABELS[c] || c)}</option>`).join("")
              }</select>`
            : "";
        container.innerHTML = `
        <div class="wc-multiselect">
            <label>${t("filter_by_category")}</label>
            <div class="wc-ms-box">${tagsHtml}${addHtml}</div>
        </div>`;
        container.querySelectorAll(".wc-ms-tag button").forEach((btn) => {
            btn.addEventListener("click", () => {
                selected.delete(btn.closest(".wc-ms-tag").dataset.cat);
                draw();
                onChange(filtered());
            });
        });
        const addSel = container.querySelector(".wc-ms-add");
        if (addSel) {
            addSel.addEventListener("change", () => {
                selected.add(addSel.value);
                draw();
                onChange(filtered());
            });
        }
    }
    draw();
    return filtered;
}

// ── Tri des anomalies (mêmes options que le dashboard Streamlit) ────────────────────────
const SORT_OPTIONS = {
    date_desc: t("sort_date_desc"),
    date_asc: t("sort_date_asc"),
    severity_desc: t("sort_severity_desc"),
    severity_asc: t("sort_severity_asc"),
    duration_desc: t("sort_duration_desc"),
};
const SEV_RANK = { high: 3, medium: 2, low: 1 };
function sortAnomalies(list, key) {
    const dur = (a) => a.trip_duration_min || a.total_elapsed_min || 0;
    const day = (a) => a.day || "";
    const sev = (a) => SEV_RANK[a.severity] || 0;
    const sorted = [...list];
    if (key === "date_asc") sorted.sort((a, b) => day(a).localeCompare(day(b)) || sev(b) - sev(a));
    else if (key === "severity_desc") sorted.sort((a, b) => sev(b) - sev(a) || day(b).localeCompare(day(a)));
    else if (key === "severity_asc") sorted.sort((a, b) => sev(a) - sev(b) || day(b).localeCompare(day(a)));
    else if (key === "duration_desc") sorted.sort((a, b) => dur(b) - dur(a));
    else sorted.sort((a, b) => day(b).localeCompare(day(a)) || sev(b) - sev(a));
    return sorted;
}

// ── View: Trajets signalés (fusion "vue d'ensemble" + "Expliquer un bus") ───────────────
// Redesign 2026-07-19 (retour utilisateur : la 1ère fusion "n'était que les deux anciens
// onglets empilés, pas vraiment fusionnés"). Une seule liste de trajets signalés (tri +
// catégories + cartes + pagination), UNE fois -- pas une carte "aujourd'hui" séparée
// au-dessus d'un "historique" qui la recontient déjà : /api/anomaly-history sans `day`
// fusionne DÉJÀ les données en direct dans l'historique côté API (voir _filter_trips,
// day=None -> merge live), donc les afficher deux fois était la vraie source de
// duplication perçue. Le panneau "Expliquer un bus" reste pour le CONTEXTE (verdict de
// ligne, trajet de référence, filtres) mais ne produit plus sa propre liste parallèle --
// il RECADRE cette même liste partagée via setScope(), avec un lien pour revenir à la
// vue d'ensemble par défaut.
async function renderTripsView(root) {
    root.innerHTML = `
    <div id="wc-t-freshness"><div class="wc-banner info"><span class="wc-spin"></span> ${t("checking_live_data")}</div></div>
    <button id="wc-t-explain-toggle" class="wc-btn-secondary wc-explain-toggle">
        ${icon("search")}<span>${t("explain_toggle_open")}</span>
    </button>
    <div id="wc-t-explain-panel" class="wc-explain-panel" hidden></div>
    <div class="wc-card">
        <div class="wc-list-head">
            <h4 id="wc-t-list-title">${t("flagged_trips_title")}</h4>
            <div class="wc-sort-row"><label>${t("sort_by")}</label><select id="wc-t-sort">${
                Object.entries(SORT_OPTIONS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")
            }</select></div>
        </div>
        <button id="wc-t-back" class="wc-link-muted" hidden>&larr; ${t("back_to_overview")}</button>
        <div id="wc-t-pills"></div>
        <div id="wc-t-cards">${skeletonCards(4)}</div>
        <button id="wc-t-more" class="wc-btn-secondary" style="margin-top:10px" hidden>${t("show_more", { shown: 0, total: 0 })}</button>
    </div>
    `;

    const freshBox = root.querySelector("#wc-t-freshness");
    const toggleBtn = root.querySelector("#wc-t-explain-toggle");
    const explainPanel = root.querySelector("#wc-t-explain-panel");
    const listTitle = root.querySelector("#wc-t-list-title");
    const backBtn = root.querySelector("#wc-t-back");
    const cardsBox = root.querySelector("#wc-t-cards");
    const pillsBox = root.querySelector("#wc-t-pills");
    const moreBtn = root.querySelector("#wc-t-more");
    const sortSel = root.querySelector("#wc-t-sort");

    // Préchauffe Leaflet pendant que l'utilisateur lit la liste -- la 1re carte (ou 1er
    // "trajet normal") ouverte n'attend alors plus le téléchargement du CDN (retour
    // utilisateur 2026-07-23 : "la carte met trop de temps à charger"). Fait pendant un
    // temps mort du navigateur, sans jamais bloquer l'affichage de la liste.
    const warmLeaflet = () => ensureLeaflet().catch(() => {});
    if ("requestIdleCallback" in window) requestIdleCallback(warmLeaflet, { timeout: 4000 });
    else setTimeout(warmLeaflet, 1500);

    const PAGE = 15;
    let shown = PAGE;
    let currentList = [];
    let baseList = [];
    // `overview` = liste non recadrée (chargement initial / retour à la vue d'ensemble),
    // par opposition aux résultats du panneau "Expliquer un bus" (scoped). Détermine
    // quelles cartes reçoivent l'icône "!" confiance réduite ET le bouton "trajet normal".
    let overview = true;
    // Jour (YYYY-MM-DD) des données "en direct" une fois confirmées par
    // /api/current-anomalies (voir loadCurrent) -- sert à séparer visuellement ces
    // anomalies-là du reste de l'historique dans drawList() ci-dessous. `null` tant que
    // la veille n'est pas encore confirmée : pas de séparation prématurée.
    let liveDate = null;

    function drawList() {
        cardsBox.innerHTML = "";
        const sorted = sortAnomalies(currentList, sortSel.value);
        if (!sorted.length) {
            cardsBox.innerHTML = `<div class="wc-banner success">${icon("check")}${t("no_anomaly_for_scope")}</div>`;
            moreBtn.hidden = true;
            return;
        }
        const page = sorted.slice(0, shown);
        // Séparation visuelle "hier / en direct" vs "historique" (retour utilisateur
        // 2026-07-24 : les deux étaient mélangées dans une seule liste, pas de distinction
        // claire) -- seulement en vue d'ensemble (pas dans les résultats recadrés du
        // panneau "Expliquer un bus"), et seulement une fois la veille confirmée. Le tri
        // choisi par l'utilisateur continue de s'appliquer normalement, à l'intérieur de
        // chaque groupe.
        if (overview && liveDate) {
            const recent = page.filter((a) => a.day === liveDate);
            const older = page.filter((a) => a.day !== liveDate);
            if (recent.length) {
                cardsBox.appendChild(el(`<div class="wc-anomaly-group-label">${LIVE_DOT}${t("recent_anomalies_label")}</div>`));
                for (const a of recent) cardsBox.appendChild(renderAlertCard(a, { withMap: true, overview }));
            }
            if (older.length) {
                cardsBox.appendChild(el(`<div class="wc-anomaly-divider"><span>${t("historique_anomalies_label")}</span></div>`));
                for (const a of older) cardsBox.appendChild(renderAlertCard(a, { withMap: true, overview }));
            }
        } else {
            for (const a of page) cardsBox.appendChild(renderAlertCard(a, { withMap: true, overview }));
        }
        moreBtn.hidden = shown >= sorted.length;
        moreBtn.textContent = t("show_more", { shown: Math.min(shown, sorted.length), total: sorted.length });
    }
    moreBtn.addEventListener("click", () => { shown += PAGE; drawList(); });
    sortSel.addEventListener("change", () => { shown = PAGE; drawList(); });

    // Un seul point d'entrée pour peupler la liste partagée -- utilisé par le chargement
    // initial (vue d'ensemble) ET par le panneau "Expliquer un bus" (vue recadrée),
    // jamais deux instances séparées de tri/catégories/cartes.
    function setScope(list, { title, scoped = false } = {}) {
        currentList = list;
        shown = PAGE;
        overview = !scoped;
        listTitle.textContent = title;
        backBtn.hidden = !scoped;
        const getFiltered = categoryFilterPills(pillsBox, list, (filtered) => { currentList = filtered; shown = PAGE; drawList(); });
        currentList = getFiltered();
        drawList();
    }
    backBtn.addEventListener("click", () => {
        setScope(baseList, { title: t("flagged_trips_title") });
        root.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    let explainLoaded = false;
    toggleBtn.addEventListener("click", () => {
        const opening = explainPanel.hidden;
        explainPanel.hidden = !opening;
        toggleBtn.classList.toggle("active", opening);
        toggleBtn.querySelector("span").textContent = opening ? t("explain_toggle_close") : t("explain_toggle_open");
        if (opening && !explainLoaded) {
            explainLoaded = true;
            renderExplainPanel(explainPanel, {
                onResults: (anomalies, { title }) => {
                    setScope(anomalies, { title, scoped: true });
                    root.querySelector(".wc-card").scrollIntoView({ behavior: "smooth", block: "start" });
                },
            });
        }
        if (opening) explainPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    // DÉCOUPLÉ en deux temps (2026-07-20, retour utilisateur : "l'utilisateur ne voit
    // d'abord que les anciennes anomalies, puis il doit recharger la page pour voir celles
    // de la veille") -- avant, les deux appels étaient attendus ENSEMBLE (Promise.all) :
    // current-anomalies peut être long (à froid, il déclenche la reconstruction + le
    // scoring de toute la société sur les pings de la veille) et l'historique, souvent
    // servi instantanément par le cache du proxy, restait bloqué derrière lui -- ou
    // l'inverse, un historique en cache SANS la veille s'affichait à côté d'un bandeau qui
    // comptait pourtant une anomalie de la veille introuvable dans la liste.
    // Ici : (1) l'historique s'affiche dès qu'il arrive ; (2) le bandeau garde son
    // animation de chargement pendant que la veille se calcule ; (3) à l'arrivée, les
    // anomalies de la veille manquantes sont FUSIONNÉES en tête de liste, sans rechargement.
    try {
        const hist = await withColdStartHint(api("/api/anomaly-history", { limit: 300 }), freshBox);
        baseList = (hist || {}).anomalies || [];
        setScope(baseList, { title: t("flagged_trips_title") });
    } catch (e) {
        freshBox.innerHTML = `<div class="wc-banner error">${t("error_generic", { msg: esc(e.message) })}</div>`;
        cardsBox.innerHTML = "";
        return;
    }

    // ── Récupération de la veille + INTERROGATION RÉPÉTÉE (2026-07-23) ────────────────────
    // Au 1er chargement, la veille n'est peut-être pas ENCORE sur Render : le relais vient
    // d'être déclenché en arrière-plan (autorun.php) et met ~1-2 min à tirer les pings puis
    // à les pousser. Plutôt qu'un sec "aucune donnée" exigeant un rafraîchissement manuel, on
    // montre une animation "données en cours de préparation" expliquant ce qui se passe, et
    // on RE-INTERROGE l'API à intervalle régulier : dès que la veille arrive, elle s'affiche
    // toute seule (bandeau + fusion des anomalies). Borné (LIVE_POLL_MAX) pour ne jamais
    // tourner indéfiniment ni solliciter le serveur sans fin ; passé ce délai, message calme
    // invitant à revenir plus tard. `fresh:true` saute le cache mémoire JS mais s'appuie sur
    // le stale-while-revalidate de proxy.php -> pas de martèlement de Render.
    const LIVE_POLL_INTERVAL_MS = 30000;
    const LIVE_POLL_MAX = 8; // ~4 min de fenêtre : couvre largement le push du relais
    let livePollTimer = null;
    let livePollTries = 0;

    function mergeLiveAnoms(today) {
        // Insère les anomalies de la veille manquantes en tête (l'historique servi par le
        // cache proxy peut être antérieur au push) -- clef (jour, bus, trip_start), jamais
        // de doublon. Ne re-rend la liste que si l'utilisateur est encore sur la vue
        // d'ensemble (pas s'il a recadré via "Filtrer / analyser un bus").
        const liveAnoms = today.anomalies || [];
        if (!liveAnoms.length) return;
        const have = new Set(baseList.map((a) => `${a.day}|${a.bus}|${a.trip_start}`));
        const missing = liveAnoms.filter((a) => !have.has(`${a.day}|${a.bus}|${a.trip_start}`));
        if (missing.length) {
            baseList = [...missing, ...baseList];
            if (backBtn.hidden) setScope(baseList, { title: t("flagged_trips_title") });
        }
    }
    function histFallbackLine(today) {
        const n = today.total_trips ?? 0;
        const k = today.anomaly_count ?? 0;
        const anom = k === 0 ? t("no_anomaly_detected") : tPlural("anomaly_detected", k);
        return `<div class="wc-muted" style="margin-top:8px">${t("historical_data_badge", { date: fmtDay(today.date) })} · ${tPlural("trips_analyzed", n)} · ${anom}</div>`;
    }
    function renderLiveBanner(today) {
        const n = today.total_trips ?? 0;
        const k = today.anomaly_count ?? 0;
        const anom = k === 0 ? t("no_anomaly_detected") : tPlural("anomaly_detected", k);
        const live = `<div class="wc-banner success">${LIVE_DOT}${t("live_data_badge", { date: fmtDay(today.date) })} · ${tPlural("trips_analyzed", n)} · ${anom}</div>`;
        const hist = today.historical_date
            ? `<div class="wc-muted" style="margin-top:4px">${t("historical_data_badge", { date: fmtDay(today.historical_date) })}</div>`
            : "";
        freshBox.innerHTML = live + hist;
    }
    function renderIncomingBanner(today, exhausted) {
        const yday = fmtDay(today.yesterday_date || ymdYesterday());
        if (exhausted) {
            // Assez attendu : on n'affirme rien de faux, on invite à revenir. L'historique
            // reste consultable ci-dessous.
            freshBox.innerHTML = `<div class="wc-banner info">${icon("chart")}${t("live_incoming_timeout", { date: yday })}</div>${histFallbackLine(today)}`;
            return;
        }
        // Animation "en préparation" : point vert pulsé + barre indéterminée + explication.
        freshBox.innerHTML = `
        <div class="wc-banner info wc-live-incoming">
            <span class="wc-live-dot" aria-hidden="true"></span>
            <div class="wc-live-incoming-body">
                <div class="wc-live-incoming-title">${t("live_incoming_title", { date: yday })}</div>
                <div class="wc-live-incoming-sub">${t("live_incoming_sub")}</div>
                <div class="wc-live-incoming-bar" aria-hidden="true"><span></span></div>
            </div>
        </div>${histFallbackLine(today)}`;
    }
    function scheduleNextPoll() {
        clearTimeout(livePollTimer);
        livePollTimer = setTimeout(() => { livePollTries += 1; loadCurrent(true); }, LIVE_POLL_INTERVAL_MS);
    }
    async function loadCurrent(isPoll) {
        let today;
        try {
            const req = api("/api/current-anomalies", {}, { fresh: isPoll });
            // Indice "serveur en veille" seulement au 1er appel -- en polling, l'animation
            // "en préparation" est déjà à l'écran, pas besoin de la remplacer.
            today = isPoll ? await req : await withColdStartHint(req, freshBox);
        } catch {
            if (!isPoll) freshBox.innerHTML = `<div class="wc-banner warn">${icon("alert")}${t("yesterday_unavailable")}</div>`;
            if (livePollTries < LIVE_POLL_MAX) scheduleNextPoll(); // erreur réseau transitoire -> on retente (borné)
            return;
        }
        if (today.live) {
            // Veille confirmée par le web service GPS (éventuellement 0 trajet un jour férié,
            // mais c'est une vraie réponse). On l'affiche et on ARRÊTE d'interroger.
            renderLiveBanner(today);
            liveDate = today.date;
            mergeLiveAnoms(today);
            // mergeLiveAnoms() ne redessine QUE s'il avait de nouvelles anomalies à
            // fusionner -- si l'historique les contenait déjà (voir commentaire plus haut
            // sur /api/anomaly-history qui fusionne déjà le direct), rien ne redessinait
            // et la séparation "hier / historique" ci-dessus n'apparaissait jamais tant
            // que l'utilisateur ne changeait pas le tri. On force donc un redessin ici,
            // maintenant que liveDate est connu, dans tous les cas.
            if (backBtn.hidden) drawList();
            return;
        }
        // Pas encore là : animation "en préparation" + prochaine tentative (tant qu'il en reste).
        const exhausted = livePollTries >= LIVE_POLL_MAX;
        renderIncomingBanner(today, exhausted);
        if (!exhausted) scheduleNextPoll();
    }

    freshBox.innerHTML = `<div class="wc-banner info"><span class="wc-spin"></span> ${t("fetching_yesterday")}</div>`;
    loadCurrent(false);
}

// ── Panneau "Expliquer un bus" (filtres + verdict de ligne + référence) ─────────────────
// Repliable DANS l'onglet "Trajets signalés" (fusion 2026-07-19). Fournit le CONTEXTE
// (verdict de ligne, trajet de référence, filtres, métriques de la requête) ; les trajets
// signalés eux-mêmes ne sont plus rendus ici -- `onResults(anomalies, {title})` les
// pousse vers la liste PARTAGÉE de renderTripsView au lieu d'en dessiner une seconde.
async function renderExplainPanel(root, { onResults }) {
    root.innerHTML = `
    <div class="wc-card">
        <div class="wc-filters">
            <div class="wc-field">
                <label>${t("field_line")}</label>
                <select id="wc-e-line"><option value="" disabled selected>${t("choose_line_placeholder")}</option></select>
            </div>
            <div class="wc-field">
                <label>${t("field_bus")}</label>
                <select id="wc-e-bus"><option value="">${t("all_buses")}</option></select>
            </div>
            <div class="wc-field">
                <label>${t("field_day")}</label>
                <select id="wc-e-day"><option value="">${t("all_days")}</option></select>
            </div>
            <div class="wc-field">
                <label data-tip="${t("field_manual_date_help")}">${t("field_manual_date")}</label>
                <input type="date" id="wc-e-manual-date">
            </div>
            <div class="wc-field">
                <label>${t("field_direction")}</label>
                <select id="wc-e-dir"><option value="">${t("dir_both")}</option><option value="ALLER">ALLER</option><option value="RETOUR">RETOUR</option></select>
            </div>
        </div>
        <div class="wc-field-analyze"><button id="wc-e-analyze">${t("analyze_btn")}</button></div>
        <p class="wc-muted" id="wc-e-hint">${t("loading_lines")}</p>
    </div>
    <div id="wc-e-verdict"></div>
    <div id="wc-e-reference"></div>
    <div id="wc-e-results"></div>
    `;

    const lineSel = root.querySelector("#wc-e-line");
    const busSel = root.querySelector("#wc-e-bus");
    const daySel = root.querySelector("#wc-e-day");
    const manualDate = root.querySelector("#wc-e-manual-date");
    const dirSel = root.querySelector("#wc-e-dir");
    const hint = root.querySelector("#wc-e-hint");
    const verdictBox = root.querySelector("#wc-e-verdict");
    const refBox = root.querySelector("#wc-e-reference");
    const resBox = root.querySelector("#wc-e-results");

    api("/api/lines-ranked").then((d) => {
        for (const line of (d.lines || [])) {
            lineSel.appendChild(el(`<option value="${esc(line)}">${esc(line)}</option>`));
        }
        // "Toutes les lignes" retiré (décision utilisateur 2026-07-19) : analyser
        // l'historique de toutes les lignes d'une société à la fois causait un HTTP 502
        // (requête trop lourde pour une réponse synchrone) -- une ligne est maintenant
        // toujours obligatoire, ce qui garde chaque analyse dans un périmètre rapide.
        hint.textContent = t("choose_line_and_analyze");
    }).catch(() => { hint.textContent = t("lines_load_failed"); });

    lineSel.addEventListener("change", async () => {
        busSel.innerHTML = `<option value="">${t("all_buses")}</option>`;
        daySel.innerHTML = `<option value="">${t("all_days")}</option>`;
        verdictBox.innerHTML = "";
        refBox.innerHTML = "";
        if (!lineSel.value) return;
        loadLineVerdict(lineSel.value);
        // Bloc replié par défaut dans le panneau "Expliquer un bus" (open:false) --
        // l'utilisateur déplie via le résumé ; l'échec API vide simplement la boîte.
        renderReferenceTrip(refBox, lineSel.value, { open: false }).catch(() => { refBox.innerHTML = ""; });
        try {
            const [busesD, daysD] = await Promise.all([
                api("/api/buses-for-line", { line: lineSel.value }),
                api("/api/days-for-line", { line: lineSel.value }),
            ]);
            for (const b of (busesD.buses || [])) busSel.appendChild(el(`<option value="${esc(b)}">${esc(b)}</option>`));
            for (const d of (daysD.days || [])) daySel.appendChild(el(`<option value="${esc(d)}">${esc(fmtDay(d))}</option>`));
        } catch { /* les listes restent réduites, l'analyse marche quand même */ }
    });

    // Verdict global de la ligne -- mêmes seuils que Streamlit : l'Isolation Forest est
    // calibrée pour ~5% d'anomalies par construction, donc <=7% = bon état, <=15% = à
    // surveiller, au-delà = risque élevé. Delta affiché vs cette base de 5%.
    async function loadLineVerdict(line) {
        verdictBox.innerHTML = `<div class="wc-card"><p class="wc-muted"><span class="wc-spin"></span> ${t("evaluating_line")}</p></div>`;
        let pat;
        try { pat = await api("/api/anomaly-patterns", { line }); }
        catch { verdictBox.innerHTML = ""; return; }
        if (!pat || pat.total_trips < 5) {
            verdictBox.innerHTML = pat && pat.total_trips
                ? `<div class="wc-banner info">${icon("info")}${t("line_not_enough_data", { line: esc(line), n: pat.total_trips })}</div>`
                : "";
            return;
        }
        const rate = pat.overall_rate;
        let cls = "success", label = t("line_good"), ic = "check";
        if (rate > 0.15) { cls = "error"; label = t("line_risk"); ic = "alert"; }
        else if (rate > 0.07) { cls = "warn"; label = t("line_watch"); ic = "alert"; }
        const delta = (rate - 0.05) * 100;
        verdictBox.innerHTML = `
        <div class="wc-card">
            <div class="wc-banner ${cls}" style="margin-bottom:10px">${icon(ic)}<strong>${label}</strong> — ${t("line_verdict_caption", { line: esc(line), n: pat.total_trips.toLocaleString(WC_LOCALE) })}</div>
            <div class="wc-metrics">
                <div class="wc-metric"><div class="wc-metric-label" data-tip="${t("anomaly_rate_tip")}">${t("metric_anomaly_rate")}</div>
                    <div class="wc-metric-value">${(rate * 100).toFixed(1)} %</div>
                    <div class="wc-metric-sub">${t("delta_vs_base", { sign: delta >= 0 ? "+" : "", delta: delta.toFixed(1) })}</div></div>
                <div class="wc-metric"><div class="wc-metric-label">${t("metric_flagged_trips")}</div>
                    <div class="wc-metric-value">${pat.total_anomalies} / ${pat.total_trips}</div></div>
            </div>
        </div>`;
    }


    async function runAnalysis() {
        const line = lineSel.value || null;
        const bus = busSel.value || null;
        const dir = dirSel.value || null;
        let day = daySel.value || null;
        if (manualDate.value) day = manualDate.value.replace(/-/g, "");

        // Ligne obligatoire (décision utilisateur 2026-07-19, en plus stricte que le
        // garde-fou précédent) : "Toutes les lignes" a été retiré du menu ci-dessus, donc
        // ce cas ne devrait plus se produire depuis l'UI -- gardé quand même en dernier
        // recours (ex. le <select> pourrait être manipulé). "Toutes les lignes" à la fois
        // scannait tout l'historique de la société en une seule requête synchrone,
        // confirmé 2026-07-19 : HTTP 502 (timeout upstream Render).
        if (!line) {
            resBox.innerHTML = `<div class="wc-banner warn">${icon("alert")}${t("choose_line_to_analyze")}</div>`;
            return;
        }

        showLoadingIn(resBox);
        try {
            // check_detours RETIRÉ (décision utilisateur 2026-07-19) : le contrôle en
            // masse sur tous les trajets signalés d'une ligne était la cause directe de
            // plusieurs pannes mémoire côté serveur (Kalman filter par bus-jour signalé,
            // jusqu'à des dizaines par requête). Un détour n'a de toute façon jamais été
            // le SEUL signal d'un trajet -- il est déjà repéré par ailleurs (trajet trop
            // long, perte de signal...) -- et reste visible EN DÉTAIL sur la carte d'un
            // trajet précis (bouton "Voir la carte du trajet", /api/trip-detail fait sa
            // propre vérification ciblée sur CE seul trajet, à la demande, sans risque).
            const res = await api("/api/anomaly-explain", { line, bus, day, dir });
            renderExplainResults(res, { line, bus });
        } catch (e) {
            resBox.innerHTML = `<div class="wc-banner error">${icon("alert")}${t("analysis_error", { msg: esc(e.message) })}</div>`;
        } finally {
            stopLoading();
        }
    }
    root.querySelector("#wc-e-analyze").addEventListener("click", runAnalysis);

    // Ne rend plus la liste des trajets elle-même -- seulement le résumé de la requête
    // (métriques + avertissement) -- et pousse les trajets vers la liste PARTAGÉE de
    // renderTripsView via onResults, au lieu d'en dessiner une seconde copie séparée.
    function renderExplainResults(res, scope) {
        const label = scope.bus ? t("scope_bus_line", { bus: scope.bus, line: scope.line }) : (scope.line ? t("scope_line", { line: scope.line }) : t("scope_all_lines"));
        if (!res || res.anomaly_count === 0) {
            const n = res ? (res.total_trips ?? 0) : 0;
            const tripsLabel = tPlural("trips_analyzed", n);
            resBox.innerHTML = `<div class="wc-banner success">${icon("check")}${t("no_anomaly_scope", { label, trips: tripsLabel })}</div>`;
            onResults([], { title: t("flagged_trips_for_scope", { label }) });
            return;
        }

        // Avertissement historique insuffisant -- une seule fois pour la liste, en langage
        // SIMPLE (retour utilisateur 2026-07-18 : la version citant Isolation Forest /
        // autoencodeur LSTM / modèle global était trop technique pour un client). Le fond
        // reste exact : comparaison au réseau entier plutôt qu'à ce périmètre précis.
        // model_low_data est fiable depuis le correctif côté API (un LSTM désactivé au
        // niveau du déploiement n'est plus présenté comme un manque de données).
        const a0 = res.anomalies[0];
        let warning = "";
        if (a0 && a0.model_low_data) {
            const txt = t(a0.model_if_dedicated ? "model_low_data_partial" : "model_low_data_line");
            warning = `<div class="wc-banner warn">${icon("alert")}${txt}</div>`;
        }

        const pct = res.total_trips ? (100 * res.anomaly_count / res.total_trips) : 0;
        resBox.innerHTML = `
        <div class="wc-card">
            ${warning}${cacheNote(res)}
            <div class="wc-metrics">
                <div class="wc-metric"><div class="wc-metric-label" data-tip="${t("metric_trips_analyzed_tip")}">${t("metric_trips_analyzed")}</div><div class="wc-metric-value">${res.total_trips}</div></div>
                <div class="wc-metric"><div class="wc-metric-label" data-tip="${t("metric_abnormal_trips_tip")}">${t("metric_abnormal_trips")}</div><div class="wc-metric-value">${res.anomaly_count} (${pct.toFixed(1)}%)</div></div>
                ${res.avg_duration_min ? `<div class="wc-metric"><div class="wc-metric-label" data-tip="${t("metric_normal_duration_tip")}">${t("metric_normal_duration")}</div><div class="wc-metric-value">${fmtDuration(res.avg_duration_min)}</div></div>` : ""}
            </div>
            <p class="wc-muted">${t("scope_trips_shown_below")}</p>
        </div>`;
        onResults(res.anomalies, { title: t("flagged_trips_for_scope", { label }) });
    }
}

// ── View: Tendances ──────────────────────────────────────────────────────────────────────
async function renderTrendsView(root) {
    root.innerHTML = skeletonChartCard(0) + `<div class="wc-charts-grid">${skeletonChartCard(240)}${skeletonChartCard(240)}</div>`;
    let chartsOk = true;
    try { await ensureChartJs(); } catch { chartsOk = false; }

    let pat;
    try {
        pat = await withColdStartHint(api("/api/anomaly-patterns", {}), root);
    } catch (e) {
        root.innerHTML = `<div class="wc-banner error">${t("error_generic", { msg: esc(e.message) })}</div>`;
        return;
    }
    if (!pat || pat.total_trips === 0) {
        root.innerHTML = `<div class="wc-banner info">${t("no_trend_data")}</div>`;
        return;
    }
    root.innerHTML = `
    <div class="wc-card">
        <div class="wc-metrics">
            <div class="wc-metric"><div class="wc-metric-label">${t("metric_total_trips")}</div><div class="wc-metric-value">${pat.total_trips.toLocaleString(WC_LOCALE)}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">${t("metric_flagged_anomalies")}</div><div class="wc-metric-value">${pat.total_anomalies}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">${t("metric_overall_rate")}</div><div class="wc-metric-value">${(pat.overall_rate * 100).toFixed(1)}%</div></div>
        </div>
    </div>
    ${!chartsOk ? `<div class="wc-banner warn">${t("charts_unavailable")}</div>` : `
    <div class="wc-charts-grid">
        <div class="wc-card"><h4>${t("chart_rate_by_line")}</h4><div class="wc-chart-wrap"><canvas id="wc-chart-byline"></canvas></div></div>
        <div class="wc-card"><h4>${t("chart_rate_by_hour")}</h4><div class="wc-chart-wrap"><canvas id="wc-chart-byhour"></canvas></div></div>
    </div>`}`;
    if (!chartsOk) return;

    const byLine = (pat.by_line || []).slice().sort((a, b) => a.rate - b.rate);
    horizontalBarChart(
        root.querySelector("#wc-chart-byline"),
        byLine.map((r) => r.line),
        byLine.map((r) => +(r.rate * 100).toFixed(1)),
        "#ef4a5f",
    );
    const byHour = pat.by_hour || [];
    verticalBarChart(
        root.querySelector("#wc-chart-byhour"),
        byHour.map((r) => r.hour),
        byHour.map((r) => +(r.rate * 100).toFixed(1)),
        "#ff8a3d",
    );
}

// ── View: Anomalies billetterie ─────────────────────────────────────────────────────────
async function renderTicketsView(root) {
    root.innerHTML = `
    <div class="wc-card">
        <div class="wc-filters">
            <div class="wc-field">
                <label>${t("field_view")}</label>
                <select id="wc-tk-view" data-tip="${t("view_tip")}">
                    <option value="client" selected>${t("view_client")}</option>
                    <option value="admin">${t("view_admin")}</option>
                </select>
            </div>
            <div class="wc-field">
                <label>${t("field_line_optional")}</label>
                <select id="wc-tk-line"><option value="">${t("all_fem")}</option></select>
            </div>
            <div class="wc-field">
                <label>${t("field_anomaly_type")}</label>
                <select id="wc-tk-recette" data-tip="${t("anomaly_type_tip")}">
                    <option value="" selected>${t("all_fem")}</option>
                    <option value="bad">${t("anomaly_type_bad")}</option>
                    <option value="good">${t("anomaly_type_good")}</option>
                </select>
            </div>
        </div>
    </div>
    <div id="wc-tk-body"></div>`;

    const viewSel = root.querySelector("#wc-tk-view");
    const lineSel = root.querySelector("#wc-tk-line");
    const recetteSel = root.querySelector("#wc-tk-recette");
    const body = root.querySelector("#wc-tk-body");

    async function load() {
        body.innerHTML = skeletonChartCard(200) + skeletonCards(3);
        try {
            const clientSafe = viewSel.value === "client";
            const pat = await withColdStartHint(api("/api/ticket-anomaly-patterns", {}), body);
            if (!pat || pat.total_days === 0) {
                body.innerHTML = `<div class="wc-banner info">${t("no_ticket_anomaly_data")}</div>`;
                return;
            }
            if (!lineSel.dataset.loaded) {
                for (const r of (pat.by_line || [])) lineSel.appendChild(el(`<option value="${esc(r.line)}">${esc(r.line)}</option>`));
                lineSel.dataset.loaded = "1";
            }
            let chartsOk = true;
            try { await ensureChartJs(); } catch { chartsOk = false; }
            body.innerHTML = `
            <div class="wc-card">
                <div class="wc-metrics">
                    <div class="wc-metric"><div class="wc-metric-label">${t("metric_bus_days")}</div><div class="wc-metric-value">${pat.total_days.toLocaleString(WC_LOCALE)}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">${t("metric_flagged_anomalies")}</div><div class="wc-metric-value">${pat.total_anomalies}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">${t("metric_overall_rate_simple")}</div><div class="wc-metric-value">${(pat.overall_rate * 100).toFixed(1)}%</div></div>
                </div>
                ${chartsOk ? `<div class="wc-chart-wrap" style="max-width:100%"><canvas id="wc-tk-chart"></canvas></div>`
                           : `<div class="wc-banner warn">${t("chart_unavailable_simple")}</div>`}
            </div>
            <div id="wc-tk-cards"></div>`;

            if (chartsOk) {
                const byLine = (pat.by_line || []).slice().sort((a, b) => a.rate - b.rate);
                horizontalBarChart(body.querySelector("#wc-tk-chart"), byLine.map((r) => r.line),
                    byLine.map((r) => +(r.rate * 100).toFixed(1)), "#2f6fed");
            }

            const hist = await api("/api/ticket-anomaly-history", {
                line: lineSel.value || null, limit: 150, client_safe: clientSafe,
            });
            const cardsBox = body.querySelector("#wc-tk-cards");
            let anomalies = (hist || {}).anomalies || [];
            // Filtre bonne/mauvaise anomalie APRÈS coup sur les 150 récupérées (même
            // logique que Streamlit, voir app.py t_recette_filter) -- c'est un critère
            // d'affichage, pas un paramètre du modèle ni de l'API.
            if (recetteSel.value === "bad") anomalies = anomalies.filter((a) => a.is_good_anomaly === false);
            else if (recetteSel.value === "good") anomalies = anomalies.filter((a) => a.is_good_anomaly === true);
            if (!anomalies.length) {
                cardsBox.innerHTML = `<div class="wc-banner info">${t("no_anomaly_for_filter")}</div>`;
                return;
            }
            for (const a of anomalies) cardsBox.appendChild(renderTicketCard(a, clientSafe));
        } catch (e) {
            body.innerHTML = `<div class="wc-banner error">${t("error_generic", { msg: esc(e.message) })}</div>`;
        }
    }
    viewSel.addEventListener("change", load);
    lineSel.addEventListener("change", load);
    recetteSel.addEventListener("change", load);
    load();
}
// ── Puces de table utilitaires (billetterie) ─────────────────────────────────────────────
function _tcell(v, fmt) { return (v === null || v === undefined) ? "—" : fmt(v); }

function stationTableHtml(rows, withType) {
    let html = `<table class="wc-table"><thead><tr><th>${t("col_stop")}</th><th>${t("col_tickets")}</th><th>${t("col_revenue")}</th><th>${t("col_avg_price")}</th>${withType ? `<th>${t("col_type")}</th>` : ""}</tr></thead><tbody>`;
    for (const r of rows) {
        const type = r.anomaly ? (r.is_good_anomaly ? t("type_good") : t("type_watch")) : "—";
        html += `<tr><td>${esc(r.station)}</td><td>${r.nbr_ticket}</td><td>${r.recette.toFixed(0)} DT</td><td>${r.avg_fare.toFixed(2)} DT</td>${withType ? `<td>${esc(type)}</td>` : ""}</tr>`;
    }
    html += `</tbody></table>`;
    return html;
}
// ALLER/RETOUR séparés quand la direction est connue pour ce bus-jour (voir
// /api/ticket-anomaly-stations `by_direction`), sinon repli sur la vue combinée -- mirrors
// src/dashboard/app.py::_render_trip_breakdown.
function tripBreakdownHtml(label, combinedRows, byDirection) {
    const aller = (byDirection || {}).ALLER || [];
    const retour = (byDirection || {}).RETOUR || [];
    if (!aller.length && !retour.length) {
        return `<p class="wc-muted"><strong>${esc(label)}</strong></p>${stationTableHtml(combinedRows, true)}`;
    }
    let html = `<p class="wc-muted"><strong>${esc(label)}</strong> — ${t("direction_known_note")}</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">`;
    for (const [dname, rows] of [["ALLER", aller], ["RETOUR", retour]]) {
        const nT = rows.reduce((s, r) => s + r.nbr_ticket, 0);
        const nR = rows.reduce((s, r) => s + r.recette, 0);
        html += `<div><p class="wc-muted">${dname} — ${rows.length ? t("tickets_revenue_summary", { n: nT, dt: nR.toFixed(0) }) : t("no_sale_recorded")}</p>${rows.length ? stationTableHtml(rows, false) : ""}</div>`;
    }
    html += `</div>`;
    return html;
}

function renderTicketCard(a, clientSafe) {
    const sev = SEV_META[a.severity] || SEV_META.medium;
    let html = `<div class="wc-alert-card">
        <div class="wc-alert-head">
            <span class="wc-badge ${sev.cls}">${sev.label}</span>
            <span class="wc-alert-title">Bus ${esc(a.bus)} · Ligne ${esc(a.line)}</span>
            <span class="wc-alert-date">${fmtDay(a.day)}</span>
        </div>`;
    if (a.is_machine_issue) html += `<div class="wc-banner warn">${icon("alert")}${t("machine_issue", { n: a.gps_trip_count || 0 })}</div>`;
    else if (a.is_no_service) html += `<div class="wc-banner info">${icon("info")}${t("no_service_day")}</div>`;
    else if (a.anomaly && a.is_good_anomaly !== null && a.is_good_anomaly !== undefined) {
        html += a.is_good_anomaly
            ? `<div class="wc-banner success">${icon("up")}${t("revenue_above_normal")}</div>`
            : `<div class="wc-banner warn">${icon("down")}${t("revenue_below_normal")}</div>`;
    }
    html += `<div class="wc-metrics">
        <div class="wc-metric"><div class="wc-metric-label">${t("metric_tickets")}</div><div class="wc-metric-value">${a.nbr_ticket}</div></div>
        <div class="wc-metric"><div class="wc-metric-label">${t("metric_revenue")}</div><div class="wc-metric-value">${a.recette.toFixed(0)} DT</div></div>
        <div class="wc-metric"><div class="wc-metric-label">${t("metric_avg_price")}</div><div class="wc-metric-value">${a.avg_fare.toFixed(1)} DT</div></div>
    </div>`;
    for (const r of (a.reasons || [])) html += `<div class="wc-reason">${esc(r)}</div>`;

    // Contexte de jugement -- tableau, pas une ligne de texte (retour utilisateur 2026-07-20
    // : "je veux plus de détails comme dans app.py, un tableau etc etc") -- mirrors
    // src/dashboard/app.py's `ctx` DataFrame (Ce jour / Médiane ligne / Médiane de ce bus).
    if (a.line_median_avg_fare !== undefined && a.line_median_avg_fare !== null) {
        html += `<table class="wc-table" style="margin-top:8px">
            <thead><tr><th></th><th>${t("ctx_table_this_day")}</th><th>${t("ctx_table_line_median")}</th><th>${t("ctx_table_bus_median")}</th></tr></thead>
            <tbody>
                <tr><td>${t("metric_tickets")}</td><td>${a.nbr_ticket}</td>
                    <td>${_tcell(a.line_median_nbr_ticket, (v) => v.toFixed(0))}</td>
                    <td>${_tcell(a.bus_median_nbr_ticket, (v) => v.toFixed(0))}</td></tr>
                <tr><td>${t("metric_revenue")}</td><td>${a.recette.toFixed(0)} DT</td>
                    <td>${_tcell(a.line_median_recette, (v) => v.toFixed(0) + " DT")}</td>
                    <td>${_tcell(a.bus_median_recette, (v) => v.toFixed(0) + " DT")}</td></tr>
                <tr><td>${t("metric_avg_price")}</td><td>${a.avg_fare.toFixed(2)} DT</td>
                    <td>${_tcell(a.line_median_avg_fare, (v) => v.toFixed(2) + " DT")}</td>
                    <td>${_tcell(a.bus_median_avg_fare, (v) => v.toFixed(2) + " DT")}</td></tr>
            </tbody>
        </table>`;
    }

    // Ligne structurellement atypique (quasi tous ses jours signalés) -- pas un incident de
    // CE jour, à réévaluer après un réentraînement par ligne plutôt qu'à traiter au cas par
    // cas (mirrors app.py's `rate >= 0.9` warning).
    if (a.line_anomaly_rate !== undefined && a.line_anomaly_rate !== null && a.line_anomaly_rate >= 0.9) {
        html += `<div class="wc-banner warn">${icon("alert")}${t("line_structural_warning", { pct: (a.line_anomaly_rate * 100).toFixed(0) })}</div>`;
    }

    html += `</div>`;
    const card = el(html);

    // ── "Historique de ce bus sur cette ligne" -- volume + prix moyen dans le temps ───────
    const histBtn = el(`<button class="wc-btn-secondary" style="margin-top:8px">${icon("chart")}${t("bus_history_btn")}</button>`);
    const histBox = el(`<div class="wc-map-holder" hidden></div>`);
    let histLoaded = false;
    histBtn.addEventListener("click", async () => {
        if (histLoaded) { histBox.hidden = !histBox.hidden; return; }
        histBtn.disabled = true;
        histBox.hidden = false;
        histBox.innerHTML = `<div class="wc-charts-grid">${skeletonChartCard(200)}${skeletonChartCard(200)}</div>`;
        try {
            const detail = await withColdStartHint(api("/api/ticket-anomaly-explain", { line: a.line, bus: a.bus, client_safe: clientSafe }), histBox);
            const rows = ((detail || {}).days || []).slice().sort((x, y) => String(x.day).localeCompare(String(y.day)));
            if (!rows.length) {
                histBox.innerHTML = `<div class="wc-banner info">${icon("info")}${t("no_bus_history")}</div>`;
            } else {
                histBox.innerHTML = `<div class="wc-charts-grid">
                    <div class="wc-chart-wrap"><canvas class="wc-tk-vol"></canvas></div>
                    <div class="wc-chart-wrap"><canvas class="wc-tk-fare"></canvas></div>
                </div>`;
                try {
                    await ensureChartJs();
                    ticketVolumeChart(histBox.querySelector(".wc-tk-vol"), rows);
                    ticketFareChart(histBox.querySelector(".wc-tk-fare"), rows, a.line_median_avg_fare);
                } catch {
                    histBox.innerHTML = `<div class="wc-banner warn">${icon("alert")}${t("charts_unavailable_short")}</div>`;
                }
            }
            histLoaded = true;
        } catch (e) {
            histBox.innerHTML = `<div class="wc-banner error">${icon("alert")}${t("error_generic", { msg: esc(e.message) })}</div>`;
        } finally {
            histBtn.disabled = false;
        }
    });
    card.appendChild(histBtn);
    card.appendChild(histBox);

    // ── "Voir le détail par arrêt" -- répartition par arrêt pour CE trajet + comparaison ──
    // au trajet de référence (bus-jour normal de cette ligne), voir /api/ticket-anomaly-
    // stations et /api/ticket-anomaly-reference.
    const stBtn = el(`<button class="wc-btn-secondary" style="margin-top:8px;margin-left:8px">${icon("pin")}${t("stop_detail_btn")}</button>`);
    const stBox = el(`<div class="wc-map-holder" hidden></div>`);
    let stLoaded = false;
    stBtn.addEventListener("click", async () => {
        if (stLoaded) { stBox.hidden = !stBox.hidden; return; }
        stBtn.disabled = true;
        stBox.hidden = false;
        stBox.innerHTML = `<div class="wc-skel-card" aria-hidden="true"><span class="wc-skel wc-skel-line w70"></span>${skeletonTable(8)}</div>`;
        try {
            const [stRes, refRes] = await withColdStartHint(Promise.all([
                api("/api/ticket-anomaly-stations", { line: a.line, bus: a.bus, day: a.day }),
                api("/api/ticket-anomaly-reference", { line: a.line }).catch(() => null),
            ]), stBox);
            const stations = (stRes || {}).stations || [];
            if (!stations.length) {
                stBox.innerHTML = `<div class="wc-banner info">${icon("info")}${t("no_stop_data")}</div>`;
            } else {
                const sumTicket = stations.reduce((s, r) => s + r.nbr_ticket, 0);
                const sumRecette = stations.reduce((s, r) => s + r.recette, 0);
                let inner = `<p class="wc-muted">${t("stops_summary", {
                    n: stations.length, bus: esc(a.bus), sumT: sumTicket, sumR: sumRecette.toFixed(0),
                    t: a.nbr_ticket, r: a.recette.toFixed(0),
                })}</p>`;
                inner += tripBreakdownHtml(t("this_trip_label", { bus: a.bus, day: fmtDay(a.day) }), stations, (stRes || {}).by_direction);
                const refTrip = (refRes || {}).trip;
                const refStations = (refRes || {}).stations || [];
                if (refTrip && refStations.length) {
                    inner += tripBreakdownHtml(
                        t("reference_trip_label", { bus: esc(refTrip.bus), day: fmtDay(refTrip.day), n: refTrip.nbr_ticket, dt: refTrip.recette.toFixed(0) }),
                        refStations, (refRes || {}).by_direction);
                }
                stBox.innerHTML = inner;
            }
            stLoaded = true;
        } catch (e) {
            stBox.innerHTML = `<div class="wc-banner error">${icon("alert")}${t("error_generic", { msg: esc(e.message) })}</div>`;
        } finally {
            stBtn.disabled = false;
        }
    });
    card.appendChild(stBtn);
    card.appendChild(stBox);

    return card;
}

// ── View: Chauffeurs ─────────────────────────────────────────────────────────────────────
async function renderDriversView(root) {
    root.innerHTML = `
    <div class="wc-banner info">${t("drivers_disclaimer")}</div>
    <div class="wc-card">
        <div class="wc-filters">
            <div class="wc-field">
                <label>${t("field_min_trips")}</label>
                <input type="number" id="wc-dr-min" value="5" min="1" max="50" style="width:80px">
            </div>
            <div class="wc-field"><label>&nbsp;</label><button id="wc-dr-refresh" class="wc-btn-secondary">${t("refresh_btn")}</button></div>
        </div>
        <h4>${t("drivers_leaderboard_title")}</h4>
        <div id="wc-dr-leaderboard">${skeletonTable(6)}</div>
    </div>
    <div class="wc-card">
        <h4>${t("driver_lookup_title")}</h4>
        <div class="wc-filters">
            <div class="wc-field"><label>${t("field_driver_code")}</label><input type="text" id="wc-dr-code"></div>
            <div class="wc-field"><label>&nbsp;</label><button id="wc-dr-lookup">${t("search_btn")}</button></div>
        </div>
        <div id="wc-dr-detail"></div>
    </div>`;

    async function loadLeaderboard() {
        const minTrips = root.querySelector("#wc-dr-min").value || 5;
        const box = root.querySelector("#wc-dr-leaderboard");
        box.innerHTML = skeletonTable(6);
        try {
            const d = await withColdStartHint(api("/api/drivers-ranked", { min_trips: minTrips, limit: 50 }), box);
            const drivers = (d || {}).drivers || [];
            if (!drivers.length) { box.innerHTML = `<div class="wc-banner info">${t("no_driver_enough_trips")}</div>`; return; }
            let rows = drivers.map((r) => `<tr>
                <td>${esc(r.driver_code)}</td><td>${r.n_trips}</td><td>${r.n_anomalies}</td>
                <td>${r.anomaly_rate.toFixed(1)}%</td></tr>`).join("");
            box.innerHTML = `<table class="wc-table"><thead><tr><th>${t("col_code")}</th><th>${t("col_trips")}</th><th>${t("col_anomalies")}</th><th>${t("col_rate")}</th></tr></thead><tbody>${rows}</tbody></table>`;
        } catch (e) {
            box.innerHTML = `<div class="wc-banner error">${t("error_generic", { msg: esc(e.message) })}</div>`;
        }
    }
    root.querySelector("#wc-dr-refresh").addEventListener("click", loadLeaderboard);
    loadLeaderboard();

    async function lookup() {
        const code = root.querySelector("#wc-dr-code").value.trim();
        const detail = root.querySelector("#wc-dr-detail");
        if (!code) return;
        detail.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> ${t("searching")}</p>`;
        let ds;
        try {
            ds = await api("/api/driver-stats", { driver_code: code });
        } catch (e) {
            detail.innerHTML = `<div class="wc-banner error">${esc(e.message)}</div>`;
            return;
        }
        // Chart.js is a nice-to-have here -- if the CDN is unreachable (offline dev, strict
        // network policy, etc.) the numbers/list below are still useful, so a failure to
        // load it must not abort the whole render (confirmed bug 2026-07-15: an uncaught
        // rejection here left the "Recherche…" spinner stuck forever).
        let chartsOk = true;
        try { await ensureChartJs(); } catch { chartsOk = false; }

        const dom = ds.dominant_cause;
        detail.innerHTML = `
        <div class="wc-metrics">
            <div class="wc-metric"><div class="wc-metric-label">${t("driver_metric_trips")}</div><div class="wc-metric-value">${ds.total_trips}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">${t("driver_metric_anomalies")}</div><div class="wc-metric-value">${ds.total_anomalies}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">${t("driver_metric_rate")}</div><div class="wc-metric-value">${ds.anomaly_rate.toFixed(1)}%</div></div>
        </div>
        ${dom ? `<p class="wc-muted">${t("driver_dominant_cause", { cause: `<strong>${esc(TOP_FEATURE_LABELS[dom.top_feature] || dom.top_feature)}</strong>`, pct: dom.pct.toFixed(1) })}</p>` : ""}
        ${chartsOk ? `<div class="wc-charts-grid">
            <div class="wc-chart-wrap"><canvas id="wc-dr-chart-cause"></canvas></div>
            <div class="wc-chart-wrap"><canvas id="wc-dr-chart-line"></canvas></div>
        </div>` : `<div class="wc-banner warn">${t("charts_unavailable_drivers")}</div>`}
        <div id="wc-dr-trips"></div>`;

        if (!chartsOk) {
            const tripsBox = detail.querySelector("#wc-dr-trips");
            for (const a of (ds.anomalies || [])) tripsBox.appendChild(renderAlertCard(a));
            return;
        }

        const dist = (ds.cause_distribution || []).slice().sort((a, b) => a.pct - b.pct);
        horizontalBarChart(detail.querySelector("#wc-dr-chart-cause"),
            dist.map((d) => TOP_FEATURE_LABELS[d.top_feature] || String(d.top_feature)),
            dist.map((d) => d.pct), "#ff8a3d");

        const byLine = (ds.by_line || []).slice().sort((a, b) => a.anomaly_rate - b.anomaly_rate);
        horizontalBarChart(detail.querySelector("#wc-dr-chart-line"),
            byLine.map((r) => r.line), byLine.map((r) => r.anomaly_rate), "#ef4a5f");

        const tripsBox = detail.querySelector("#wc-dr-trips");
        for (const a of (ds.anomalies || [])) tripsBox.appendChild(renderAlertCard(a));
    }
    root.querySelector("#wc-dr-lookup").addEventListener("click", lookup);
    root.querySelector("#wc-dr-code").addEventListener("keydown", (e) => { if (e.key === "Enter") lookup(); });
}

// ── Tab wiring ───────────────────────────────────────────────────────────────────────────
const VIEWS = { trips: renderTripsView, trends: renderTrendsView, tickets: renderTicketsView, drivers: renderDriversView };

function init() {
    const root = document.getElementById("wc-view-root");
    if (!root) return; // company picker screen, nothing to wire up
    const tabs = document.querySelectorAll(".wc-tab");
    // Panneaux PERSISTANTS par onglet (redesign 2026-07-20, retour utilisateur : "naviguer
    // entre les onglets perd les données et je dois ré-attendre") -- chaque vue est rendue
    // UNE fois dans son propre <div>, puis simplement masquée/affichée (attribut hidden,
    // couvert par le filet [hidden] global du CSS) au lieu d'être détruite et re-rendue
    // avec re-fetch à chaque clic. Cartes déroulées, graphiques, résultats d'analyse et
    // position de défilement survivent tous à un aller-retour d'onglet. Les données ne
    // changent de toute façon qu'une fois par jour (la veille poussée par le relais) --
    // un rechargement de page suffit pour forcer du frais.
    const panels = {};
    function show(view) {
        for (const [name, p] of Object.entries(panels)) p.hidden = name !== view;
        if (!panels[view]) {
            const p = document.createElement("div");
            panels[view] = p;
            root.appendChild(p);
            VIEWS[view](p);
        }
    }
    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            tabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            show(tab.dataset.view);
        });
    });
    show("trips"); // default view on load
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();

})();
