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

// ── Labels (mirrors src/dashboard/app.py's TOP_FEATURE_FR / i18n) ──────────────────────
const TOP_FEATURE_LABELS = {
    max_dwell_s: "Immobilisation anormale",
    total_elapsed: "Trajet anormalement long",
    mean_dwell_s: "Durée d'arrêt moyenne élevée",
    dist_m_max: "Déviation de l'itinéraire",
    match_rate: "Mauvais suivi GPS / hors itinéraire",
    n_stops: "Arrêts desservis anormalement faible",
    max_dark_s: "Perte de signal GPS",
    terminus_idle_min: "Stationnement terminus (service non clôturé)",
    elapsed_vs_bus_z: "Durée inhabituelle pour ce bus",
    elapsed_vs_line_z: "Durée inhabituelle pour cette ligne",
    unofficial_detour: "Détour non-officiel",
    null: "Autre / non catégorisé",
};
const SEV_META = {
    high: { label: "Élevée", cls: "high" },
    medium: { label: "Moyenne", cls: "medium" },
    low: { label: "Faible", cls: "low" },
};

// ── Explications au survol des raisons (mirrors src/dashboard/i18n.py's REASON_HELP /
// exp_* keys, FR) -- portées ici pour donner la même explicabilité que le dashboard
// Streamlit (retour utilisateur 2026-07-19 : "expliquer tout avec le survol comme dans
// app.py"). Affichées via l'attribut title (tooltip natif), même convention que
// .wc-metric-label[title] déjà utilisée pour "Taux d'anomalie".
const REASON_HELP = {
    max_dwell_s: "Le plus long arrêt immobile détecté sur ce trajet dépasse nettement la normale. Mesuré pings-GPS-présents (bus vraiment immobile), pas un trou de signal.",
    total_elapsed: "Durée totale du trajet (arrivée − départ), après retrait du stationnement en bordure de trajet (voir Formule) -- donc déjà nettoyée du service-non-clôturé aux deux bouts, mais PAS des arrêts survenus en cours de route.",
    mean_dwell_s: "Durée moyenne d'immobilisation par arrêt sur ce trajet, plus élevée que la normale de la ligne -- suggère un ralentissement généralisé (trafic, retards en cascade) plutôt qu'un incident ponctuel à un seul arrêt.",
    dist_m_max: "Écart maximal observé entre la position GPS du bus et la position théorique d'un arrêt qu'il a quand même été compté comme ayant desservi -- déviation d'itinéraire ou dérive GPS.",
    match_rate: "Part des arrêts de la ligne effectivement détectés par le GPS sur ce trajet. Un taux bas peut venir d'un trajet partiel, d'un mauvais suivi GPS, ou d'arrêts aux coordonnées douteuses.",
    n_stops: "Nombre d'arrêts couverts par ce trajet, nettement inférieur à la normale de la ligne/direction -- indique un trajet partiel plutôt qu'un problème de vitesse.",
    max_dark_s: "Le plus long trou de signal GPS détecté (aucun ping reçu) sur ce trajet. Le bus a pu continuer à rouler normalement pendant ce silence -- ce temps N'EST PAS compté comme immobilisation, seulement comme incertitude.",
    terminus_idle_min: "Temps où le traceur GPS a continué à pinger alors que le bus était garé au terminus -- avant le vrai départ ou après la vraie arrivée. Ce temps est DÉJÀ RETIRÉ de la durée du trajet affichée ; ce chiffre le montre séparément.",
    elapsed_vs_bus_z: "Écart-type (z-score) de la durée de CE trajet par rapport à l'historique de CE BUS précis sur cette ligne. La durée comparée exclut déjà le stationnement en bordure de trajet -- ce n'est pas un artefact de stationnement non retiré.",
    elapsed_vs_line_z: "Écart-type (z-score) de la durée de CE trajet par rapport à la médiane de TOUS les bus sur cette ligne/direction. La durée comparée exclut déjà le stationnement en bordure de trajet -- ce n'est pas un artefact de stationnement non retiré.",
    unofficial_detour: "Le trajet réel suivi par GPS s'écarte de l'itinéraire officiel de la ligne sur une portion significative, avant de revenir dans le circuit normal.",
};
const REASON_HELP_DEFAULT = "Signal utilisé par le modèle de détection pour juger ce trajet anormal.";
const FORMULA_HELP = "Durée trajet = dernier ping − premier ping, APRÈS avoir retiré le stationnement immobile en bordure du trajet (bus garé avant le vrai départ / après la vraie arrivée). Un arrêt ou une immobilisation EN COURS de trajet, elle, reste comptée dans cette durée -- ce n'est pas la même chose.";

// ── Explications au survol des puces (mirrors src/dashboard/i18n.py's chip_*_help keys) ──
const CHIP_HELP = {
    origin_idle: "Ce temps est RETIRÉ de la « Durée trajet » affichée en haut de la carte -- le bus pingait déjà au terminus DE DÉPART mais n'avait pas encore démarré son service. Signal opérationnel à part entière (chauffeur n'ayant probablement pas coupé le traceur), pas une erreur de mesure.",
    end_idle: "Ce temps est RETIRÉ de la « Durée trajet » affichée en haut de la carte -- le bus pingait encore au terminus D'ARRIVÉE mais n'avait pas vraiment terminé son service. Signal opérationnel à part entière (chauffeur n'ayant probablement pas coupé le traceur), pas une erreur de mesure.",
    real_stop: "Immobilisation détectée EN COURS de trajet (pings GPS présents, bus vraiment immobile) -- reste comptée dans la « Durée trajet » ci-dessus, contrairement au stationnement terminus qui lui en est retiré.",
    signal_loss: "Aucun ping GPS reçu pendant cette durée à cet arrêt -- le bus a pu continuer à rouler normalement pendant ce silence, ce N'EST PAS une immobilisation confirmée.",
    dark_gap: "Trou de signal survenu ENTRE deux arrêts (pas pendant l'attente à un arrêt déjà repéré) -- invisible au scan arrêt-par-arrêt classique, mais bien réel : le traceur n'a envoyé AUCUN ping pendant cette durée. Explique généralement à lui seul le mauvais taux de suivi et la durée gonflée du reste du trajet -- le bus a très probablement continué de rouler normalement pendant ce silence.",
    farthest: "Le bus a bien été détecté à cet arrêt, mais sa position GPS était nettement plus loin que la position officielle de l'arrêt -- déviation d'itinéraire ou dérive GPS.",
    off_route: "Ces arrêts sont dans l'étendue du trajet mais le bus n'y a jamais été détecté à portée GPS -- trajet partiel, itinéraire différent, ou desserte réellement sautée.",
    suspect_coord: "Ces arrêts ne sont JAMAIS détectés sur AUCUN trajet de cette ligne -- leurs coordonnées géographiques sont probablement fausses dans la base de référence, pas un problème de CE trajet précis. Exclus du diagnostic pour cette raison.",
    detour: "Détecté sur les positions GPS brutes : le bus a quitté son point de départ, s'est éloigné significativement, puis est revenu quasiment au même endroit avant sa longue immobilisation — probablement une course annexe (dépôt, ravitaillement...) plutôt qu'un simple bruit GPS.",
};

// ── Small utilities ─────────────────────────────────────────────────────────────────────
function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}
function fmtDay(d) {
    if (!d) return "—";
    const s = String(d);
    const y = s.slice(0, 4), m = s.slice(4, 6), day = s.slice(6, 8);
    const MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."];
    const mi = parseInt(m, 10) - 1;
    return (mi >= 0 && mi < 12) ? `${parseInt(day, 10)} ${MOIS[mi]} ${y}` : s;
}
function fmtTime(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (isNaN(d)) return "—";
    return d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
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
    "Analyse des trajets…", "Détection des anomalies…", "Vérification des détours…",
    "Recoupement des horaires…", "Presque prêt…",
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

async function api(endpoint, params = {}) {
    const qs = new URLSearchParams({ endpoint, ...cleanParams(params) });
    const res = await fetch(`${WINICARI_PROXY}?${qs.toString()}`);
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
    return data;
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
            plugins: { legend: { display: false }, title: { display: true, text: "Tickets/jour", color: "#c8d4e6", font: { size: 12 } } },
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
        label: "Prix moyen", data: rows.map((r) => r.avg_fare), showLine: false,
        pointBackgroundColor: colors, pointBorderColor: colors, pointRadius: 4,
    }];
    if (lineMedian !== null && lineMedian !== undefined) {
        datasets.push({
            label: "Médiane ligne", data: rows.map(() => lineMedian),
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
                title: { display: true, text: "Prix moyen (DT) vs médiane ligne", color: "#c8d4e6", font: { size: 12 } },
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
        container.innerHTML = `<div class="wc-banner info">${icon("info")}Coordonnées GPS non disponibles pour les arrêts de cette ligne.</div>`;
        return;
    }
    const dot = (c) => `<span class="wc-legend-dot" style="background:${c}"></span>`;
    let html = `<p class="wc-muted wc-map-legend">Chaque cercle représente un arrêt, numéroté dans l'ordre de passage (départ → terminus). La taille reflète la durée d'immobilisation + perte de signal.<br>
        ${dot("#22c55e")}Normal ${dot("#2563eb")}Immobilisation longue (≥10 min) ${dot("#f59e0b")}Perte de signal (≥5 min) ${dot("#ef4444")}Arrêt non desservi ${dot("#9ca3af")}Coordonnées douteuses</p>`;

    if (detour && detour.track) {
        const t = (v) => v ? fmtTime(v) : "—";
        html += `<div class="wc-banner warn">${icon("detour")}Détour non-officiel détecté : le bus a quitté son point de départ à <strong>${t(detour.left_at)}</strong>, s'est éloigné de ~${detour.distance_km} km (point le plus éloigné à ${t(detour.farthest_at)}), puis est revenu à <strong>${t(detour.returned_at)}</strong> — ~${(detour.duration_min || 0).toFixed(0)} min au total. Tracé orange = aller, violet = retour.</div>`;
    }
    html += `<div class="wc-map"></div>`;

    // Ordre de passage réel : les seq suivent la géométrie ALLER, donc un RETOUR visite
    // les arrêts en seq décroissant (même logique que le dashboard Streamlit).
    const sorted = [...rows].sort((a, b) => direction === "RETOUR" ? b.seq - a.seq : a.seq - b.seq);
    const tracked = sorted.filter((s) => s.arrival);
    if (tracked.length) {
        const first = tracked[0], last = tracked[tracked.length - 1];
        html += `<p class="wc-muted">${icon("pin")}Premier passage suivi : <strong>${fmtTime(first.arrival)}</strong> (${esc(first.stop)}) → dernier passage suivi : <strong>${fmtTime(last.arrival)}</strong> (${esc(last.stop)})</p>`;
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
            Passage : ${s.arrival ? fmtTime(s.arrival) : "—"}<br>
            Immobilisation : ${(s.dwell_min || 0).toFixed(1)} min<br>
            Perte de signal : ${(s.dark_min || 0).toFixed(1)} min<br>
            Écart de position : ${(s.dist_m || 0).toFixed(0)} m<br>
            Suivi : ${s.matched ? "oui" : "non"}${s.coord_suspect ? "<br><i>Coordonnées d'arrêt douteuses</i>" : ""}`;
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
    mkEnd(sorted[0], "#16a34a", `Départ${direction ? " (" + direction + ")" : ""}`);
    mkEnd(sorted[sorted.length - 1], "#dc2626", "Terminus");

    map.fitBounds(L.latLngBounds(latlngs).pad(0.15));
}

// ── Shared: alert card rendering (used by Trips view, Chauffeurs view) ─────────────────
function rowCategories(a) {
    const cats = [a.top_feature ?? null];
    if (a.has_detour) cats.push("unofficial_detour");
    return cats;
}
function renderAlertCard(a, { showDriverStatsHint = true, withMap = false } = {}) {
    const sev = SEV_META[a.severity] || SEV_META.medium;
    const dep = fmtTime(a.trip_start), arr = fmtTime(a.trip_end);
    const dur = a.trip_duration_min || a.total_elapsed_min || 0;
    const reasons = a.reasons || [];
    const ps = a.problem_stops || {};

    let html = `<div class="wc-alert-card">
        <div class="wc-alert-head">
            <span class="wc-badge ${sev.cls}">${sev.label}</span>
            <span class="wc-alert-title">Bus ${esc(a.bus)} · Ligne ${esc(a.line)} · ${esc(a.dir)}</span>
            <span class="wc-alert-date">${fmtDay(a.day)}</span>
        </div>
        <div class="wc-metrics">
            <div class="wc-metric"><div class="wc-metric-label" title="${esc(FORMULA_HELP)}">Durée du trajet ⓘ</div><div class="wc-metric-value">${fmtDuration(dur)}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">Départ → Arrivée</div><div class="wc-metric-value" style="font-size:16px">${dep} → ${arr}</div></div>
        </div>`;

    if (a.top_feature !== undefined) {
        html += `<p class="wc-muted">Cause principale : <strong>${esc(TOP_FEATURE_LABELS[a.top_feature] || "Non catégorisé")}</strong></p>`;
    }
    if (a.is_data_bug) html += `<div class="wc-banner error">${icon("alert")}Probable bug de données (durée/horodatage incohérent) — à vérifier avant d'agir.</div>`;
    else if (a.is_fragment) html += `<div class="wc-banner warn">${icon("fragment")}Trajet trop court/fragmentaire pour être jugé fiable.</div>`;
    else if (a.is_dark_inflated) html += `<div class="wc-banner info">${icon("signal")}Durée gonflée par une perte de signal GPS prolongée, pas une vraie immobilisation.</div>`;
    else if (a.is_implausible) html += `<div class="wc-banner info">${icon("clock")}Durée improbable — à vérifier.</div>`;
    else if (a.is_partial_coverage) html += `<div class="wc-banner info">${icon("pin")}Couverture partielle (${a.n_stops ?? "?"} arrêts vs ${a.line_median_n_stops ?? "?"} normalement).</div>`;

    const reasonFeats = a.reason_features || [];
    reasons.forEach((r, i) => {
        const help = REASON_HELP[reasonFeats[i]] || REASON_HELP_DEFAULT;
        html += `<div class="wc-reason" title="${esc(help)}">${esc(r)}</div>`;
    });

    // Puces (arrêts/segments concernés) -- chacune avec sa propre explication au survol
    // (mirrors src/dashboard/app.py's per-chip help=... calls). Stationnement terminus
    // DÉTAILLÉ (quel terminus, de quand à quand) : le chiffre "~N min avant départ/après
    // arrivée" de la raison modèle est mesuré sur les pings GPS (bus immobile au terminus,
    // temps DÉJÀ RETIRÉ de la durée du trajet affichée), et méritait d'être nommé +
    // horodaté au lieu d'un chiffre nu (retour utilisateur 2026-07-18).
    let originIdleShown = false, endIdleShown = false;
    if ((a.origin_idle_min || 0) >= 30 && a.origin_idle_stop) {
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.origin_idle)}">${icon("parking")}Stationné au terminus <strong>${esc(a.origin_idle_stop)}</strong> avant le départ : <strong>${a.origin_idle_min.toFixed(0)} min</strong> — le traceur pingait sur place de ${fmtTime(a.origin_idle_from)} à ${dep} (départ réel). Temps non compté dans la durée du trajet ci-dessus.</div>`;
        originIdleShown = true;
    }
    if ((a.end_idle_min || 0) >= 30 && a.end_idle_stop) {
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.end_idle)}">${icon("parking")}Stationné au terminus <strong>${esc(a.end_idle_stop)}</strong> après l'arrivée : <strong>${a.end_idle_min.toFixed(0)} min</strong> — immobile de ${arr} (arrivée réelle) à ${fmtTime(a.end_idle_to)}. Temps non compté dans la durée du trajet ci-dessus.</div>`;
        endIdleShown = true;
    }
    if (ps.longest_stop && ps.longest_stop.dwell_min >= 5) {
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.real_stop)}">${icon("parking")}Immobilisation la plus longue : <strong>${esc(ps.longest_stop.stop)}</strong> (${ps.longest_stop.dwell_min.toFixed(0)} min)</div>`;
        // Même arrêt que le stationnement terminus déjà affiché ci-dessus ? Très probablement
        // UNE seule immobilisation continue coupée en deux par un sursaut GPS isolé, pas deux
        // événements distincts -- sinon, hypothèse plus générique (détour possible).
        const sameOrigin = originIdleShown && ps.longest_stop.stop === a.origin_idle_stop;
        const sameEnd = endIdleShown && ps.longest_stop.stop === a.end_idle_stop;
        if (sameOrigin || sameEnd) {
            const sameIdle = sameOrigin ? a.origin_idle_min : a.end_idle_min;
            const total = sameIdle + ps.longest_stop.dwell_min;
            html += `<div class="wc-chip wc-chip-hint">&nbsp;&nbsp;↳ <em>C'est le MÊME arrêt que le stationnement terminus ci-dessus (<strong>${esc(ps.longest_stop.stop)}</strong>) -- très probablement UNE seule immobilisation continue coupée en deux par un bref sursaut GPS isolé, pas deux événements distincts. Durée réelle probable : ~${total.toFixed(0)} min.</em></div>`;
        } else if (!originIdleShown && !endIdleShown) {
            html += `<div class="wc-chip wc-chip-hint">&nbsp;&nbsp;↳ <em>Si ce délai suit le stationnement terminus, ce n'est pas forcément le même arrêt : le bus a pu repartir puis revenir avant de s'immobiliser à nouveau (détour non officiel / course non planifiée). « Voir la carte du trajet » vérifie sur les pings GPS réels et affiche le trajet emprunté si c'est le cas.</em></div>`;
        }
    }
    if (ps.signal_loss_stop) {
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.signal_loss)}">${icon("signal")}Perte de signal à <strong>${esc(ps.signal_loss_stop.stop)}</strong> (~${ps.signal_loss_stop.dark_min.toFixed(0)} min)</div>`;
    }
    // Trou de signal EN ROUTE (entre deux arrêts, jamais rattaché à l'attente d'un arrêt
    // matché) -- invisible au scan arrêt-par-arrêt ci-dessus, mais peut expliquer à lui
    // seul un mauvais taux de suivi + une durée gonflée.
    if (a.dark_gap_before_stop && (a.max_dark_min || 0) >= 15) {
        const after = a.dark_gap_after_stop;
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.dark_gap)}">${icon("signal")}Perte de signal en route : <strong>${after
            ? `entre ${esc(a.dark_gap_before_stop)} et ${esc(after)}`
            : `après ${esc(a.dark_gap_before_stop)}, plus aucun arrêt suivi jusqu'à la fin du trajet`
        }</strong> (~${a.max_dark_min.toFixed(0)} min sans aucun ping).</div>`;
    }
    if (ps.farthest_stop) {
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.farthest)}">${icon("pin")}Écart de position à <strong>${esc(ps.farthest_stop.stop)}</strong> (~${ps.farthest_stop.dist_m.toFixed(0)} m)</div>`;
    }
    if (ps.off_route_stops && ps.off_route_stops.length) {
        const others = (ps.off_route_count || ps.off_route_stops.length) - ps.off_route_stops.length;
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.off_route)}">${icon("ban")}Arrêts non desservis : ${esc(ps.off_route_stops.join(", "))}${others > 0 ? ` (+${others} autres)` : ""}</div>`;
    }
    if (ps.suspect_coord_count) {
        html += `<div class="wc-chip" title="${esc(CHIP_HELP.suspect_coord)}">${icon("info")}${ps.suspect_coord_count} arrêt(s) aux coordonnées douteuses (jamais suivis sur cette ligne — exclus du diagnostic).</div>`;
    }
    if (a.has_detour && a.detour) {
        html += `<div class="wc-chip detour" title="${esc(CHIP_HELP.detour)}">${icon("detour")}Détour non-officiel confirmé — ~${a.detour.distance_km} km pendant ~${a.detour.duration_min.toFixed(0)} min avant de revenir.</div>`;
    }
    if (a.scheduled_departure && a.departure_delay_min !== null && a.departure_delay_min !== undefined && Math.abs(a.departure_delay_min) >= 3) {
        const late = a.departure_delay_min > 0;
        html += `<div class="wc-chip">${icon("clock")}Départ prévu <strong>${esc(a.scheduled_departure)}</strong>, départ réel <strong>${dep}</strong> — ${Math.abs(a.departure_delay_min).toFixed(0)} min ${late ? "de retard" : "d'avance"}${a.schedule_multi_variant ? " (horaire multiple, à titre indicatif)" : ""}.</div>`;
    }
    if (a.driver_code) {
        html += `<div class="wc-chip driver">${icon("driver")}Chauffeur : <strong>${esc(a.driver_code)}</strong></div>
        <div class="wc-disclaimer">Information fournie à titre indicatif — une corrélation entre un chauffeur et des trajets signalés n'est pas un verdict automatique. À interpréter selon le contexte (trafic, panne, météo…).</div>`;
    }

    html += `</div>`;
    const card = el(html);

    // Bouton carte -- même comportement que le dashboard Streamlit : la séquence par arrêt
    // est chargée À LA DEMANDE (/api/trip-detail), pas avec la liste (une carte Leaflet par
    // carte d'alerte chargée d'office serait ruineux en DOM et en appels API).
    if (withMap && a.trip_start) {
        const btn = el(`<button class="wc-btn-secondary wc-btn-map">${icon("pin")}Voir la carte du trajet</button>`);
        const mapBox = el(`<div class="wc-map-holder"></div>`);
        let loaded = false;
        btn.addEventListener("click", async () => {
            if (loaded) { mapBox.hidden = !mapBox.hidden; return; }
            btn.disabled = true;
            mapBox.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Chargement de la carte…</p>`;
            try {
                await ensureLeaflet();
                const d = await api("/api/trip-detail", {
                    line: a.line, bus: a.bus, day: a.day, trip_start: a.trip_start,
                });
                renderTripMap(mapBox, d.sequence, a.dir, (d.problem_stops || {}).unofficial_detour || a.detour);
                loaded = true;
            } catch (e) {
                mapBox.innerHTML = `<div class="wc-banner error">${icon("alert")}Carte indisponible : ${esc(e.message)}</div>`;
            } finally {
                btn.disabled = false;
            }
        });
        card.appendChild(btn);
        card.appendChild(mapBox);
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
                <button type="button" aria-label="Retirer">&times;</button>
            </span>`).join("");
        const remaining = present.filter((c) => !selected.has(c));
        const addHtml = remaining.length
            ? `<select class="wc-ms-add"><option value="" selected disabled>+ Ajouter…</option>${
                remaining.map((c) => `<option value="${esc(c)}">${esc(TOP_FEATURE_LABELS[c] || c)}</option>`).join("")
              }</select>`
            : "";
        container.innerHTML = `
        <div class="wc-multiselect">
            <label>Filtrer par type d'anomalie</label>
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
    date_desc: "Plus récent d'abord",
    date_asc: "Plus ancien d'abord",
    severity_desc: "Gravité décroissante",
    severity_asc: "Gravité croissante",
    duration_desc: "Durée décroissante",
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
    <div id="wc-t-freshness"><div class="wc-banner info"><span class="wc-spin"></span> Vérification des données en direct…</div></div>
    <button id="wc-t-explain-toggle" class="wc-btn-secondary wc-explain-toggle">
        ${icon("search")}<span>Filtrer / analyser un bus précis</span>
    </button>
    <div id="wc-t-explain-panel" class="wc-explain-panel" hidden></div>
    <div class="wc-card">
        <div class="wc-list-head">
            <h4 id="wc-t-list-title">Trajets signalés</h4>
            <div class="wc-sort-row"><label>Trier par</label><select id="wc-t-sort">${
                Object.entries(SORT_OPTIONS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")
            }</select></div>
        </div>
        <button id="wc-t-back" class="wc-link-muted" hidden>&larr; Revenir à la vue d'ensemble</button>
        <div id="wc-t-pills"></div>
        <div id="wc-t-cards">${skeletonCards(4)}</div>
        <button id="wc-t-more" class="wc-btn-secondary" style="margin-top:10px" hidden>Afficher plus</button>
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

    const PAGE = 15;
    let shown = PAGE;
    let currentList = [];
    let baseList = [];

    function drawList() {
        cardsBox.innerHTML = "";
        const sorted = sortAnomalies(currentList, sortSel.value);
        if (!sorted.length) {
            cardsBox.innerHTML = `<div class="wc-banner success">${icon("check")}Aucune anomalie trouvée pour ce périmètre.</div>`;
            moreBtn.hidden = true;
            return;
        }
        for (const a of sorted.slice(0, shown)) cardsBox.appendChild(renderAlertCard(a, { withMap: true }));
        moreBtn.hidden = shown >= sorted.length;
        moreBtn.textContent = `Afficher plus (${Math.min(shown, sorted.length)}/${sorted.length})`;
    }
    moreBtn.addEventListener("click", () => { shown += PAGE; drawList(); });
    sortSel.addEventListener("change", () => { shown = PAGE; drawList(); });

    // Un seul point d'entrée pour peupler la liste partagée -- utilisé par le chargement
    // initial (vue d'ensemble) ET par le panneau "Expliquer un bus" (vue recadrée),
    // jamais deux instances séparées de tri/catégories/cartes.
    function setScope(list, { title, scoped = false } = {}) {
        currentList = list;
        shown = PAGE;
        listTitle.textContent = title;
        backBtn.hidden = !scoped;
        const getFiltered = categoryFilterPills(pillsBox, list, (filtered) => { currentList = filtered; shown = PAGE; drawList(); });
        currentList = getFiltered();
        drawList();
    }
    backBtn.addEventListener("click", () => {
        setScope(baseList, { title: "Trajets signalés" });
        root.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    let explainLoaded = false;
    toggleBtn.addEventListener("click", () => {
        const opening = explainPanel.hidden;
        explainPanel.hidden = !opening;
        toggleBtn.classList.toggle("active", opening);
        toggleBtn.querySelector("span").textContent = opening ? "Masquer les filtres" : "Filtrer / analyser un bus précis";
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

    // Fraîcheur (léger) + liste unifiée (déjà fusionnée en direct côté API) en parallèle --
    // plus de second appel dupliqué à current-anomalies pour peupler une carte séparée.
    try {
        const [today, hist] = await Promise.all([
            api("/api/current-anomalies", {}),
            api("/api/anomaly-history", { limit: 300 }),
        ]);
        // Toujours dire COMBIEN de trajets ont été analysés, et le dire explicitement quand
        // AUCUN n'est anormal (retour utilisateur 2026-07-20) -- sans ça, "pas d'anomalie
        // hier" et "les données d'hier ne sont pas encore arrivées" se ressemblaient trop.
        const dayLabel = today.live ? `Données en direct — ${fmtDay(today.date)}` : `Dernier jour historique disponible — ${fmtDay(today.date)}`;
        const nTrips = today.total_trips ?? 0;
        const tripsLabel = `${nTrips} trajet${nTrips === 1 ? "" : "s"} analysé${nTrips === 1 ? "" : "s"}`;
        const nAnom = today.anomaly_count ?? 0;
        const anomLabel = nAnom === 0 ? "aucune anomalie détectée" : `${nAnom} anomalie${nAnom === 1 ? "" : "s"} détectée${nAnom === 1 ? "" : "s"}`;
        freshBox.innerHTML = `<div class="wc-banner ${today.live ? "success" : "info"}">${today.live ? LIVE_DOT : icon("chart")}${dayLabel} · ${tripsLabel} · ${anomLabel}</div>`
            + cacheNote(today) + cacheNote(hist);
        baseList = (hist || {}).anomalies || [];
        setScope(baseList, { title: "Trajets signalés" });
    } catch (e) {
        freshBox.innerHTML = `<div class="wc-banner error">Erreur : ${esc(e.message)}</div>`;
        cardsBox.innerHTML = "";
    }
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
                <label>Ligne</label>
                <select id="wc-e-line"><option value="" disabled selected>— Choisir une ligne —</option></select>
            </div>
            <div class="wc-field">
                <label>Bus</label>
                <select id="wc-e-bus"><option value="">Tous les bus</option></select>
            </div>
            <div class="wc-field">
                <label>Jour</label>
                <select id="wc-e-day"><option value="">Tous les jours</option></select>
            </div>
            <div class="wc-field">
                <label title="Prioritaire sur le menu « Jour » si renseignée — utile pour un jour tout juste arrivé via les webservices en direct et pas encore dans la liste précalculée.">Ou date précise</label>
                <input type="date" id="wc-e-manual-date">
            </div>
            <div class="wc-field">
                <label>Direction</label>
                <select id="wc-e-dir"><option value="">Les deux</option><option value="ALLER">ALLER</option><option value="RETOUR">RETOUR</option></select>
            </div>
        </div>
        <div class="wc-field-analyze"><button id="wc-e-analyze">Analyser</button></div>
        <p class="wc-muted" id="wc-e-hint">Chargement des lignes…</p>
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
        hint.textContent = "Choisissez une ligne puis cliquez Analyser.";
    }).catch(() => { hint.textContent = "Impossible de charger la liste des lignes."; });

    lineSel.addEventListener("change", async () => {
        busSel.innerHTML = `<option value="">Tous les bus</option>`;
        daySel.innerHTML = `<option value="">Tous les jours</option>`;
        verdictBox.innerHTML = "";
        refBox.innerHTML = "";
        if (!lineSel.value) return;
        loadLineVerdict(lineSel.value);
        loadReferenceTrip(lineSel.value);
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
        verdictBox.innerHTML = `<div class="wc-card"><p class="wc-muted"><span class="wc-spin"></span> Évaluation de la ligne…</p></div>`;
        let pat;
        try { pat = await api("/api/anomaly-patterns", { line }); }
        catch { verdictBox.innerHTML = ""; return; }
        if (!pat || pat.total_trips < 5) {
            verdictBox.innerHTML = pat && pat.total_trips
                ? `<div class="wc-banner info">${icon("info")}Ligne ${esc(line)} : seulement ${pat.total_trips} trajet(s) enregistré(s) — pas assez de données pour juger l'état général de la ligne.</div>`
                : "";
            return;
        }
        const rate = pat.overall_rate;
        let cls = "success", label = "Ligne en bon état", ic = "check";
        if (rate > 0.15) { cls = "error"; label = "Ligne à risque élevé"; ic = "alert"; }
        else if (rate > 0.07) { cls = "warn"; label = "Ligne à surveiller"; ic = "alert"; }
        const delta = (rate - 0.05) * 100;
        verdictBox.innerHTML = `
        <div class="wc-card">
            <div class="wc-banner ${cls}" style="margin-bottom:10px">${icon(ic)}<strong>${label}</strong> — Ligne ${esc(line)} · basé sur ${pat.total_trips.toLocaleString("fr-FR")} trajets analysés.</div>
            <div class="wc-metrics">
                <div class="wc-metric"><div class="wc-metric-label" title="Part des trajets signalés comme anormaux. Le modèle est calibré pour ~5% d'anomalies « naturelles » — le delta est mesuré par rapport à cette base.">Taux d'anomalie ⓘ</div>
                    <div class="wc-metric-value">${(rate * 100).toFixed(1)} %</div>
                    <div class="wc-metric-sub">${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts vs base 5%</div></div>
                <div class="wc-metric"><div class="wc-metric-label">Trajets signalés</div>
                    <div class="wc-metric-value">${pat.total_anomalies} / ${pat.total_trips}</div></div>
            </div>
        </div>`;
    }

    // Trajet de référence -- « voici à quoi ressemble un trajet NORMAL sur cette ligne » :
    // ancre de confiance, comme l'expander Streamlit, avec carte par direction.
    async function loadReferenceTrip(line) {
        refBox.innerHTML = `<div class="wc-card"><p class="wc-muted"><span class="wc-spin"></span> Recherche d'un trajet de référence…</p></div>`;
        let ref;
        try { ref = await api("/api/reference-trip", { line }); }
        catch { refBox.innerHTML = ""; return; }
        const dirs = (ref || {}).directions || {};
        const dirNames = Object.keys(dirs);
        if (!dirNames.length) {
            refBox.innerHTML = `<div class="wc-banner info">${icon("info")}Pas de trajet de référence disponible pour cette ligne.</div>`;
            return;
        }
        let inner = `<details class="wc-ref"><summary>${icon("check")}Trajet de référence — à quoi ressemble un trajet NORMAL sur la ligne ${esc(line)} ?</summary>`;
        if (dirNames.length === 2) {
            const at = dirs["ALLER"] && dirs["ALLER"].trip, rt = dirs["RETOUR"] && dirs["RETOUR"].trip;
            if (at && rt && at.bus === rt.bus && at.day === rt.day) {
                inner += `<p class="wc-muted">Cycle complet : bus <strong>${esc(at.bus)}</strong>, le <strong>${fmtDay(at.day)}</strong> — l'ALLER et le RETOUR ci-dessous sont le même bus le même jour, donc la pause au terminus reflète une vraie pause entre l'arrivée et le redépart.</p>`;
            }
        } else {
            inner += `<p class="wc-muted">Direction : <strong>${esc(dirNames[0])}</strong> (aucun trajet normal exploitable trouvé pour l'autre direction sur cette ligne).</p>`;
        }
        // Onglets ALLER/RETOUR côte à côte (redesign 2026-07-19, retour utilisateur : "je
        // veux y accéder à travers les tabs comme dans Streamlit" -- st.tabs(dirs.keys())).
        // Un seul bouton = pas de barre d'onglets, rien à sélectionner.
        if (dirNames.length > 1) {
            inner += `<div class="wc-ref-tabs">${dirNames.map((d, i) =>
                `<button type="button" class="wc-ref-tab${i === 0 ? " active" : ""}" data-dir="${esc(d)}">${esc(d)}</button>`
            ).join("")}</div>`;
        }
        for (const d of dirNames) {
            const entry = dirs[d];
            const rt = entry.trip;
            // Un trajet de référence peut être PARTIEL quand la direction n'a (quasi)
            // aucun trajet complet -- fait des DONNÉES, pas un bug d'affichage (constaté
            // 2026-07-18, S.R.T.K/202 : 409/498 ALLER complets contre 2/443 RETOUR, le
            // traceur s'arrête systématiquement en route au retour). Sans cette note,
            // l'écart de nombre d'arrêts/durée entre les deux directions est illisible.
            const partialNote = (rt.is_full === false && rt.geometry_stops)
                ? `<div class="wc-banner info">${icon("info")}Couverture GPS partielle : le traceur ne couvre que <strong>${rt.covered_stops} des ${rt.geometry_stops} arrêts</strong> de la ligne sur ce trajet — aucun trajet entièrement suivi n'était disponible dans cette direction (fréquent quand le traceur est coupé en cours de route). Durée et arrêts affichés ne portent que sur la partie couverte.</div>`
                : "";
            inner += `
            <div class="wc-ref-dir" data-dir="${esc(d)}"${d === dirNames[0] ? "" : " hidden"}>
                ${dirNames.length > 1 ? "" : `<h4>${esc(d)}</h4>`}
                <p class="wc-muted">Trajet réel, jugé normal par le modèle, choisi parmi les mieux suivis (${(rt.match_rate * 100).toFixed(0)}% des arrêts) avec une durée proche de la médiane de la ligne pour cette direction — comparez les anomalies ci-dessous à cette référence.</p>
                ${partialNote}
                <div class="wc-metrics">
                    <div class="wc-metric"><div class="wc-metric-label">Bus</div><div class="wc-metric-value">${esc(rt.bus)}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">Jour</div><div class="wc-metric-value" style="font-size:15px">${fmtDay(rt.day)}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">Durée</div><div class="wc-metric-value">${fmtDuration(rt.duration_min)}</div>
                        <div class="wc-metric-sub">médiane ligne : ${fmtDuration(rt.line_median_min)}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">Arrêts suivis</div><div class="wc-metric-value">${rt.n_stops}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">Immobilisation moy.</div><div class="wc-metric-value">${(rt.mean_dwell_min || 0).toFixed(1)} min</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">Départ → Arrivée</div><div class="wc-metric-value" style="font-size:15px">${fmtTime(rt.trip_start)} → ${fmtTime(rt.trip_end)}</div></div>
                </div>
                ${rt.typical_terminus_idle_min !== null && rt.typical_terminus_idle_min !== undefined ? `
                <div class="wc-chip">${icon("parking")}Stationnement terminus typique pour cette direction : <strong>~${rt.typical_terminus_idle_min.toFixed(0)} min</strong> avant le vrai départ / après la vraie arrivée (médiane sur les trajets normaux).${rt.service_not_closed_threshold_min ? ` Au-delà d'environ <strong>${rt.service_not_closed_threshold_min.toFixed(0)} min</strong> (~2x la normale), un stationnement observé est probablement un <strong>service non clôturé</strong> (chauffeur n'ayant pas coupé le traceur) plutôt qu'une pause normale.` : ""}</div>` : ""}
                <div class="wc-ref-map" data-dir="${esc(d)}"></div>
            </div>`;
        }
        inner += `</details>`;
        refBox.innerHTML = `<div class="wc-card">${inner}</div>`;

        // Bascule d'onglet : montre/cache le bloc de direction correspondant. La carte
        // Leaflet de la direction nouvellement affichée est instanciée à la demande (voir
        // plus bas) -- pas les deux à l'ouverture -- puisque seule une est visible à la fois.
        const tabBtns = [...refBox.querySelectorAll(".wc-ref-tab")];
        tabBtns.forEach((btn) => btn.addEventListener("click", () => {
            const d = btn.dataset.dir;
            tabBtns.forEach((b) => b.classList.toggle("active", b === btn));
            for (const dd of dirNames) {
                refBox.querySelector(`.wc-ref-dir[data-dir="${CSS.escape(dd)}"]`).hidden = dd !== d;
            }
            renderRefMap(d);
        }));

        // Cartes chargées à l'OUVERTURE du bloc (pas d'office) -- Leaflet dans un <details>
        // fermé, ou dans un onglet caché (display:none), mesure une taille nulle et rend
        // une carte grise ; on attend le premier affichage RÉEL de chaque direction pour
        // instancier la sienne, une seule fois chacune.
        const details = refBox.querySelector("details");
        const mapsBuilt = new Set();
        async function renderRefMap(d) {
            if (!details.open || mapsBuilt.has(d) || !dirs[d].sequence) return;
            mapsBuilt.add(d);
            try { await ensureLeaflet(); } catch { return; }
            const holder = refBox.querySelector(`.wc-ref-map[data-dir="${CSS.escape(d)}"]`);
            if (holder) renderTripMap(holder, dirs[d].sequence, d, null);
        }
        details.addEventListener("toggle", () => { if (details.open) renderRefMap(dirNames[0]); });
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
            resBox.innerHTML = `<div class="wc-banner warn">${icon("alert")}Choisissez une ligne pour analyser.</div>`;
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
            resBox.innerHTML = `<div class="wc-banner error">${icon("alert")}Erreur d'analyse : ${esc(e.message)}</div>`;
        } finally {
            stopLoading();
        }
    }
    root.querySelector("#wc-e-analyze").addEventListener("click", runAnalysis);

    // Ne rend plus la liste des trajets elle-même -- seulement le résumé de la requête
    // (métriques + avertissement) -- et pousse les trajets vers la liste PARTAGÉE de
    // renderTripsView via onResults, au lieu d'en dessiner une seconde copie séparée.
    function renderExplainResults(res, scope) {
        const label = scope.bus ? `Bus ${scope.bus} · Ligne ${scope.line}` : (scope.line ? `Ligne ${scope.line}` : "Toutes les lignes");
        if (!res || res.anomaly_count === 0) {
            const n = res ? (res.total_trips ?? 0) : 0;
            const tripsLabel = `${n} trajet${n === 1 ? "" : "s"} analysé${n === 1 ? "" : "s"}`;
            resBox.innerHTML = `<div class="wc-banner success">${icon("check")}${label} : ${tripsLabel} · aucune anomalie détectée — tout est dans la normale.</div>`;
            onResults([], { title: `Trajets signalés — ${label}` });
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
            const txt = (!a0.model_if_dedicated)
                ? "Le système a encore peu d'historique pour cet opérateur : ces trajets sont comparés à l'ensemble du réseau, pas à cette ligne précisément. Résultats à prendre avec un peu de recul — la précision s'affinera d'elle-même à mesure que l'historique grandit."
                : "L'historique de cet opérateur est encore partiel : une partie de l'analyse le compare à l'ensemble du réseau plutôt qu'à cette ligne précisément. Résultats à prendre avec un peu de recul — la précision s'affinera d'elle-même à mesure que l'historique grandit.";
            warning = `<div class="wc-banner warn">${icon("alert")}${txt}</div>`;
        }

        const pct = res.total_trips ? (100 * res.anomaly_count / res.total_trips) : 0;
        resBox.innerHTML = `
        <div class="wc-card">
            ${warning}${cacheNote(res)}
            <div class="wc-metrics">
                <div class="wc-metric"><div class="wc-metric-label" title="Nombre total de trajets dans la période sélectionnée pour ce périmètre.">Trajets analysés ⓘ</div><div class="wc-metric-value">${res.total_trips}</div></div>
                <div class="wc-metric"><div class="wc-metric-label" title="Trajets signalés comme anormaux par le système de détection automatique.">Trajets anormaux ⓘ</div><div class="wc-metric-value">${res.anomaly_count} (${pct.toFixed(1)}%)</div></div>
                ${res.avg_duration_min ? `<div class="wc-metric"><div class="wc-metric-label" title="Durée médiane d'un trajet non anormal sur cette ligne — sert de référence pour juger si un trajet est trop long ou trop court.">Durée normale (médiane) ⓘ</div><div class="wc-metric-value">${fmtDuration(res.avg_duration_min)}</div></div>` : ""}
            </div>
            <p class="wc-muted">Les trajets signalés pour ce périmètre sont affichés dans la liste ci-dessous.</p>
        </div>`;
        onResults(res.anomalies, { title: `Trajets signalés — ${label}` });
    }
}

// ── View: Tendances ──────────────────────────────────────────────────────────────────────
async function renderTrendsView(root) {
    root.innerHTML = `<div class="wc-card"><p class="wc-muted"><span class="wc-spin"></span> Chargement des tendances…</p></div>`;
    let chartsOk = true;
    try { await ensureChartJs(); } catch { chartsOk = false; }

    let pat;
    try {
        pat = await api("/api/anomaly-patterns", {});
    } catch (e) {
        root.innerHTML = `<div class="wc-banner error">Erreur : ${esc(e.message)}</div>`;
        return;
    }
    if (!pat || pat.total_trips === 0) {
        root.innerHTML = `<div class="wc-banner info">Aucune donnée de tendance disponible pour cet opérateur.</div>`;
        return;
    }
    root.innerHTML = `
    <div class="wc-card">
        <div class="wc-metrics">
            <div class="wc-metric"><div class="wc-metric-label">Trajets au total</div><div class="wc-metric-value">${pat.total_trips.toLocaleString("fr-FR")}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">Anomalies signalées</div><div class="wc-metric-value">${pat.total_anomalies}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">Taux d'anomalie global</div><div class="wc-metric-value">${(pat.overall_rate * 100).toFixed(1)}%</div></div>
        </div>
    </div>
    ${!chartsOk ? `<div class="wc-banner warn">Graphiques indisponibles (bibliothèque de graphiques injoignable).</div>` : `
    <div class="wc-charts-grid">
        <div class="wc-card"><h4>Taux d'anomalie par ligne</h4><div class="wc-chart-wrap"><canvas id="wc-chart-byline"></canvas></div></div>
        <div class="wc-card"><h4>Taux d'anomalie par heure de départ</h4><div class="wc-chart-wrap"><canvas id="wc-chart-byhour"></canvas></div></div>
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
                <label>Vue</label>
                <select id="wc-tk-view">
                    <option value="admin">Admin (tout)</option>
                    <option value="client">Client (fiable uniquement)</option>
                </select>
            </div>
            <div class="wc-field">
                <label>Ligne (optionnel)</label>
                <select id="wc-tk-line"><option value="">Toutes</option></select>
            </div>
        </div>
    </div>
    <div id="wc-tk-body"><p class="wc-muted">Chargement…</p></div>`;

    const viewSel = root.querySelector("#wc-tk-view");
    const lineSel = root.querySelector("#wc-tk-line");
    const body = root.querySelector("#wc-tk-body");

    async function load() {
        body.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Chargement…</p>`;
        try {
            const clientSafe = viewSel.value === "client";
            const pat = await api("/api/ticket-anomaly-patterns", {});
            if (!pat || pat.total_days === 0) {
                body.innerHTML = `<div class="wc-banner info">Aucune donnée de billetterie anormale pour cet opérateur.</div>`;
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
                    <div class="wc-metric"><div class="wc-metric-label">Jours-bus au total</div><div class="wc-metric-value">${pat.total_days.toLocaleString("fr-FR")}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">Anomalies signalées</div><div class="wc-metric-value">${pat.total_anomalies}</div></div>
                    <div class="wc-metric"><div class="wc-metric-label">Taux global</div><div class="wc-metric-value">${(pat.overall_rate * 100).toFixed(1)}%</div></div>
                </div>
                ${chartsOk ? `<div class="wc-chart-wrap" style="max-width:100%"><canvas id="wc-tk-chart"></canvas></div>`
                           : `<div class="wc-banner warn">Graphique indisponible (bibliothèque de graphiques injoignable).</div>`}
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
            const anomalies = (hist || {}).anomalies || [];
            if (!anomalies.length) {
                cardsBox.innerHTML = `<div class="wc-banner info">Aucun jour anormal trouvé pour ce filtre.</div>`;
                return;
            }
            for (const a of anomalies) cardsBox.appendChild(renderTicketCard(a, clientSafe));
        } catch (e) {
            body.innerHTML = `<div class="wc-banner error">Erreur : ${esc(e.message)}</div>`;
        }
    }
    viewSel.addEventListener("change", load);
    lineSel.addEventListener("change", load);
    load();
}
// ── Puces de table utilitaires (billetterie) ─────────────────────────────────────────────
function _tcell(v, fmt) { return (v === null || v === undefined) ? "—" : fmt(v); }

function stationTableHtml(rows, withType) {
    let html = `<table class="wc-table"><thead><tr><th>Arrêt</th><th>Tickets</th><th>Recette</th><th>Prix moyen</th>${withType ? "<th>Type</th>" : ""}</tr></thead><tbody>`;
    for (const r of rows) {
        const type = r.anomaly ? (r.is_good_anomaly ? "Bonne" : "À surveiller") : "—";
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
    let html = `<p class="wc-muted"><strong>${esc(label)}</strong> — direction connue pour ce trajet (ALLER/RETOUR séparés)</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">`;
    for (const [dname, rows] of [["ALLER", aller], ["RETOUR", retour]]) {
        const nT = rows.reduce((s, r) => s + r.nbr_ticket, 0);
        const nR = rows.reduce((s, r) => s + r.recette, 0);
        html += `<div><p class="wc-muted">${dname} — ${rows.length ? `${nT} tickets / ${nR.toFixed(0)} DT` : "pas de vente enregistrée"}</p>${rows.length ? stationTableHtml(rows, false) : ""}</div>`;
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
    if (a.is_machine_issue) html += `<div class="wc-banner warn">${icon("alert")}Probable panne de la machine à tickets (${a.gps_trip_count || 0} trajets GPS confirmés ce jour).</div>`;
    else if (a.is_no_service) html += `<div class="wc-banner info">${icon("info")}Aucun trajet GPS ce jour — jour sans service, pas une anomalie de recette.</div>`;
    else if (a.anomaly && a.is_good_anomaly !== null && a.is_good_anomaly !== undefined) {
        html += a.is_good_anomaly
            ? `<div class="wc-banner success">${icon("up")}Recette au-dessus de la normale.</div>`
            : `<div class="wc-banner warn">${icon("down")}Recette en dessous de la normale — à surveiller.</div>`;
    }
    html += `<div class="wc-metrics">
        <div class="wc-metric"><div class="wc-metric-label">Tickets</div><div class="wc-metric-value">${a.nbr_ticket}</div></div>
        <div class="wc-metric"><div class="wc-metric-label">Recette</div><div class="wc-metric-value">${a.recette.toFixed(0)} DT</div></div>
        <div class="wc-metric"><div class="wc-metric-label">Prix moyen</div><div class="wc-metric-value">${a.avg_fare.toFixed(1)} DT</div></div>
    </div>`;
    for (const r of (a.reasons || [])) html += `<div class="wc-reason">${esc(r)}</div>`;

    // Contexte de jugement -- tableau, pas une ligne de texte (retour utilisateur 2026-07-20
    // : "je veux plus de détails comme dans app.py, un tableau etc etc") -- mirrors
    // src/dashboard/app.py's `ctx` DataFrame (Ce jour / Médiane ligne / Médiane de ce bus).
    if (a.line_median_avg_fare !== undefined && a.line_median_avg_fare !== null) {
        html += `<table class="wc-table" style="margin-top:8px">
            <thead><tr><th></th><th>Ce jour</th><th>Médiane ligne</th><th>Médiane de ce bus</th></tr></thead>
            <tbody>
                <tr><td>Tickets</td><td>${a.nbr_ticket}</td>
                    <td>${_tcell(a.line_median_nbr_ticket, (v) => v.toFixed(0))}</td>
                    <td>${_tcell(a.bus_median_nbr_ticket, (v) => v.toFixed(0))}</td></tr>
                <tr><td>Recette</td><td>${a.recette.toFixed(0)} DT</td>
                    <td>${_tcell(a.line_median_recette, (v) => v.toFixed(0) + " DT")}</td>
                    <td>${_tcell(a.bus_median_recette, (v) => v.toFixed(0) + " DT")}</td></tr>
                <tr><td>Prix moyen</td><td>${a.avg_fare.toFixed(2)} DT</td>
                    <td>${_tcell(a.line_median_avg_fare, (v) => v.toFixed(2) + " DT")}</td>
                    <td>${_tcell(a.bus_median_avg_fare, (v) => v.toFixed(2) + " DT")}</td></tr>
            </tbody>
        </table>`;
    }

    // Ligne structurellement atypique (quasi tous ses jours signalés) -- pas un incident de
    // CE jour, à réévaluer après un réentraînement par ligne plutôt qu'à traiter au cas par
    // cas (mirrors app.py's `rate >= 0.9` warning).
    if (a.line_anomaly_rate !== undefined && a.line_anomaly_rate !== null && a.line_anomaly_rate >= 0.9) {
        html += `<div class="wc-banner warn">${icon("alert")}${(a.line_anomaly_rate * 100).toFixed(0)}% des jours de cette ligne sont signalés — écart structurel de la ligne (tarification), pas un incident de ce jour précis.</div>`;
    }

    html += `</div>`;
    const card = el(html);

    // ── "Historique de ce bus sur cette ligne" -- volume + prix moyen dans le temps ───────
    const histBtn = el(`<button class="wc-btn-secondary" style="margin-top:8px">${icon("chart")}Historique de ce bus sur cette ligne</button>`);
    const histBox = el(`<div class="wc-map-holder" hidden></div>`);
    let histLoaded = false;
    histBtn.addEventListener("click", async () => {
        if (histLoaded) { histBox.hidden = !histBox.hidden; return; }
        histBtn.disabled = true;
        histBox.hidden = false;
        histBox.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Chargement de l'historique…</p>`;
        try {
            const detail = await api("/api/ticket-anomaly-explain", { line: a.line, bus: a.bus, client_safe: clientSafe });
            const rows = ((detail || {}).days || []).slice().sort((x, y) => String(x.day).localeCompare(String(y.day)));
            if (!rows.length) {
                histBox.innerHTML = `<div class="wc-banner info">${icon("info")}Pas d'historique disponible pour ce bus.</div>`;
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
                    histBox.innerHTML = `<div class="wc-banner warn">${icon("alert")}Graphiques indisponibles (bibliothèque injoignable).</div>`;
                }
            }
            histLoaded = true;
        } catch (e) {
            histBox.innerHTML = `<div class="wc-banner error">${icon("alert")}Erreur : ${esc(e.message)}</div>`;
        } finally {
            histBtn.disabled = false;
        }
    });
    card.appendChild(histBtn);
    card.appendChild(histBox);

    // ── "Voir le détail par arrêt" -- répartition par arrêt pour CE trajet + comparaison ──
    // au trajet de référence (bus-jour normal de cette ligne), voir /api/ticket-anomaly-
    // stations et /api/ticket-anomaly-reference.
    const stBtn = el(`<button class="wc-btn-secondary" style="margin-top:8px;margin-left:8px">${icon("pin")}Voir le détail par arrêt</button>`);
    const stBox = el(`<div class="wc-map-holder" hidden></div>`);
    let stLoaded = false;
    stBtn.addEventListener("click", async () => {
        if (stLoaded) { stBox.hidden = !stBox.hidden; return; }
        stBtn.disabled = true;
        stBox.hidden = false;
        stBox.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Chargement du détail par arrêt…</p>`;
        try {
            const [stRes, refRes] = await Promise.all([
                api("/api/ticket-anomaly-stations", { line: a.line, bus: a.bus, day: a.day }),
                api("/api/ticket-anomaly-reference", { line: a.line }).catch(() => null),
            ]);
            const stations = (stRes || {}).stations || [];
            if (!stations.length) {
                stBox.innerHTML = `<div class="wc-banner info">${icon("info")}Aucune donnée par arrêt pour ce trajet (modèle par arrêt pas encore entraîné, ou trop peu de données).</div>`;
            } else {
                const sumTicket = stations.reduce((s, r) => s + r.nbr_ticket, 0);
                const sumRecette = stations.reduce((s, r) => s + r.recette, 0);
                let inner = `<p class="wc-muted">${stations.length} arrêt(s) desservi(s) par le bus ${esc(a.bus)} ce jour-là — ${sumTicket} tickets / ${sumRecette.toFixed(0)} DT au total (vs ${a.nbr_ticket} tickets / ${a.recette.toFixed(0)} DT affiché ci-dessus pour ce bus-jour).</p>`;
                inner += tripBreakdownHtml(`Ce trajet — bus ${a.bus} · ${fmtDay(a.day)}`, stations, (stRes || {}).by_direction);
                const refTrip = (refRes || {}).trip;
                const refStations = (refRes || {}).stations || [];
                if (refTrip && refStations.length) {
                    inner += tripBreakdownHtml(
                        `Trajet de référence (normal) — bus ${esc(refTrip.bus)} · ${fmtDay(refTrip.day)} — ${refTrip.nbr_ticket} tickets / ${refTrip.recette.toFixed(0)} DT`,
                        refStations, (refRes || {}).by_direction);
                }
                stBox.innerHTML = inner;
            }
            stLoaded = true;
        } catch (e) {
            stBox.innerHTML = `<div class="wc-banner error">${icon("alert")}Erreur : ${esc(e.message)}</div>`;
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
    <div class="wc-banner info">Ces informations sur les chauffeurs sont fournies à titre indicatif — une corrélation avec des trajets signalés n'est pas un verdict automatique. Facteurs externes (trafic, panne, météo…) peuvent expliquer une anomalie sans faute du chauffeur.</div>
    <div class="wc-card">
        <div class="wc-filters">
            <div class="wc-field">
                <label>Trajets minimum</label>
                <input type="number" id="wc-dr-min" value="5" min="1" max="50" style="width:80px">
            </div>
            <div class="wc-field"><label>&nbsp;</label><button id="wc-dr-refresh" class="wc-btn-secondary">Actualiser</button></div>
        </div>
        <h4>Chauffeurs classés par taux d'anomalie</h4>
        <div id="wc-dr-leaderboard"><p class="wc-muted">Chargement…</p></div>
    </div>
    <div class="wc-card">
        <h4>Rechercher un chauffeur par code</h4>
        <div class="wc-filters">
            <div class="wc-field"><label>Code chauffeur</label><input type="text" id="wc-dr-code"></div>
            <div class="wc-field"><label>&nbsp;</label><button id="wc-dr-lookup">Rechercher</button></div>
        </div>
        <div id="wc-dr-detail"></div>
    </div>`;

    async function loadLeaderboard() {
        const minTrips = root.querySelector("#wc-dr-min").value || 5;
        const box = root.querySelector("#wc-dr-leaderboard");
        box.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Chargement…</p>`;
        try {
            const d = await api("/api/drivers-ranked", { min_trips: minTrips, limit: 50 });
            const drivers = (d || {}).drivers || [];
            if (!drivers.length) { box.innerHTML = `<div class="wc-banner info">Aucun chauffeur avec assez de trajets.</div>`; return; }
            let rows = drivers.map((r) => `<tr>
                <td>${esc(r.driver_code)}</td><td>${r.n_trips}</td><td>${r.n_anomalies}</td>
                <td>${r.anomaly_rate.toFixed(1)}%</td></tr>`).join("");
            box.innerHTML = `<table class="wc-table"><thead><tr><th>Code</th><th>Trajets</th><th>Anomalies</th><th>Taux</th></tr></thead><tbody>${rows}</tbody></table>`;
        } catch (e) {
            box.innerHTML = `<div class="wc-banner error">Erreur : ${esc(e.message)}</div>`;
        }
    }
    root.querySelector("#wc-dr-refresh").addEventListener("click", loadLeaderboard);
    loadLeaderboard();

    async function lookup() {
        const code = root.querySelector("#wc-dr-code").value.trim();
        const detail = root.querySelector("#wc-dr-detail");
        if (!code) return;
        detail.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Recherche…</p>`;
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
            <div class="wc-metric"><div class="wc-metric-label">Trajets</div><div class="wc-metric-value">${ds.total_trips}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">Anomalies</div><div class="wc-metric-value">${ds.total_anomalies}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">Taux d'anomalie</div><div class="wc-metric-value">${ds.anomaly_rate.toFixed(1)}%</div></div>
        </div>
        ${dom ? `<p class="wc-muted">Cause la plus fréquente : <strong>${esc(TOP_FEATURE_LABELS[dom.top_feature] || dom.top_feature)}</strong> (${dom.pct.toFixed(1)}% de ses anomalies)</p>` : ""}
        ${chartsOk ? `<div class="wc-charts-grid">
            <div class="wc-chart-wrap"><canvas id="wc-dr-chart-cause"></canvas></div>
            <div class="wc-chart-wrap"><canvas id="wc-dr-chart-line"></canvas></div>
        </div>` : `<div class="wc-banner warn">Graphiques indisponibles (bibliothèque de graphiques injoignable) — les chiffres ci-dessus et la liste ci-dessous restent à jour.</div>`}
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
    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            tabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            root.innerHTML = "";
            VIEWS[tab.dataset.view](root);
        });
    });
    renderTripsView(root); // default view on load
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();

})();
