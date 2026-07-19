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
            <div class="wc-loader-ring r1"></div>
            <div class="wc-loader-ring r2"></div>
            <div class="wc-loader-ring r3"></div>
            <div class="wc-loader-core"></div>
            <div class="wc-loader-dot d1"></div>
            <div class="wc-loader-dot d2"></div>
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
// Petite note discrète quand la réponse vient du cache PÉRIMÉ (une actualisation tourne
// déjà en arrière-plan, voir proxy.php) -- pas affichée pour HIT (cache frais, rien à
// signaler) ni MISS (déjà la donnée la plus fraîche possible).
function cacheNote(data) {
    if (!data || data.__cache !== "STALE") return "";
    return `<p class="wc-muted wc-cache-note">${icon("clock")}Résultat en cache -- actualisation en cours en arrière-plan, réessayez dans une minute pour les toutes dernières données.</p>`;
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
            <div class="wc-metric"><div class="wc-metric-label">Durée du trajet</div><div class="wc-metric-value">${fmtDuration(dur)}</div></div>
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

    for (const r of reasons) html += `<div class="wc-reason">${esc(r)}</div>`;

    // Stationnement terminus DÉTAILLÉ (quel terminus, de quand à quand) -- parité avec le
    // dashboard Streamlit (chip_origin_idle/chip_end_idle) : le chiffre "~N min avant
    // départ/après arrivée" de la raison modèle est mesuré sur les pings GPS (bus immobile
    // au terminus, temps DÉJÀ RETIRÉ de la durée du trajet affichée), et méritait d'être
    // nommé + horodaté au lieu d'un chiffre nu (retour utilisateur 2026-07-18).
    if ((a.origin_idle_min || 0) >= 30 && a.origin_idle_stop) {
        html += `<div class="wc-chip">${icon("parking")}Stationné au terminus <strong>${esc(a.origin_idle_stop)}</strong> avant le départ : <strong>${a.origin_idle_min.toFixed(0)} min</strong> — le traceur pingait sur place de ${fmtTime(a.origin_idle_from)} à ${dep} (départ réel). Temps non compté dans la durée du trajet ci-dessus.</div>`;
    }
    if ((a.end_idle_min || 0) >= 30 && a.end_idle_stop) {
        html += `<div class="wc-chip">${icon("parking")}Stationné au terminus <strong>${esc(a.end_idle_stop)}</strong> après l'arrivée : <strong>${a.end_idle_min.toFixed(0)} min</strong> — immobile de ${arr} (arrivée réelle) à ${fmtTime(a.end_idle_to)}. Temps non compté dans la durée du trajet ci-dessus.</div>`;
    }
    if (ps.longest_stop && ps.longest_stop.dwell_min >= 5) {
        html += `<div class="wc-chip">${icon("parking")}Immobilisation la plus longue : <strong>${esc(ps.longest_stop.stop)}</strong> (${ps.longest_stop.dwell_min.toFixed(0)} min)</div>`;
    }
    if (ps.signal_loss_stop) {
        html += `<div class="wc-chip">${icon("signal")}Perte de signal à <strong>${esc(ps.signal_loss_stop.stop)}</strong> (~${ps.signal_loss_stop.dark_min.toFixed(0)} min)</div>`;
    }
    if (ps.farthest_stop) {
        html += `<div class="wc-chip">${icon("pin")}Écart de position à <strong>${esc(ps.farthest_stop.stop)}</strong> (~${ps.farthest_stop.dist_m.toFixed(0)} m)</div>`;
    }
    if (ps.off_route_stops && ps.off_route_stops.length) {
        html += `<div class="wc-chip">${icon("ban")}Arrêts non desservis : ${esc(ps.off_route_stops.join(", "))}</div>`;
    }
    if (a.has_detour && a.detour) {
        html += `<div class="wc-chip detour">${icon("detour")}Détour non-officiel confirmé — ~${a.detour.distance_km} km pendant ~${a.detour.duration_min.toFixed(0)} min avant de revenir.</div>`;
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

function categoryFilterPills(container, anomalies, onChange) {
    const present = [...new Set(anomalies.flatMap(rowCategories))]
        .sort((a, b) => (TOP_FEATURE_LABELS[a] || "").localeCompare(TOP_FEATURE_LABELS[b] || ""));
    if (present.length < 2) { container.innerHTML = ""; return () => anomalies; }
    const selected = new Set(present);
    container.innerHTML = "";
    for (const cat of present) {
        const pill = el(`<span class="wc-pill selected" data-cat="${esc(cat)}">${esc(TOP_FEATURE_LABELS[cat] || cat)}</span>`);
        pill.addEventListener("click", () => {
            if (selected.has(cat)) { selected.delete(cat); pill.classList.remove("selected"); }
            else { selected.add(cat); pill.classList.add("selected"); }
            onChange(anomalies.filter((a) => rowCategories(a).some((c) => selected.has(c))));
        });
        container.appendChild(pill);
    }
    return () => anomalies.filter((a) => rowCategories(a).some((c) => selected.has(c)));
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

// ── View: Trajets signalés (aujourd'hui en direct + historique complet) ─────────────────
// Miroir de l'onglet tab_live du dashboard Streamlit -- séparé de "Expliquer un bus"
// (décision utilisateur 2026-07-17 : revenir aux 2 onglets comme dans Streamlit).
async function renderTripsView(root) {
    // Fusion "Trajets signalés" + "Expliquer un bus" (décision utilisateur 2026-07-19) --
    // un seul onglet : la vue rapide du jour + l'historique s'affiche tout de suite comme
    // avant, et un bouton révèle/masque le panneau de filtres+analyse (ex-onglet séparé,
    // voir renderExplainPanel) sans jamais le charger tant qu'on n'en a pas besoin.
    root.innerHTML = `
    <button id="wc-t-explain-toggle" class="wc-btn-secondary wc-explain-toggle">
        ${icon("search")}<span>Expliquer un bus</span>
    </button>
    <div id="wc-t-explain-panel" class="wc-explain-panel" hidden></div>
    <div id="wc-t-results"></div>
    `;
    const toggleBtn = root.querySelector("#wc-t-explain-toggle");
    const explainPanel = root.querySelector("#wc-t-explain-panel");
    let explainLoaded = false;
    toggleBtn.addEventListener("click", () => {
        const opening = explainPanel.hidden;
        explainPanel.hidden = !opening;
        toggleBtn.classList.toggle("active", opening);
        toggleBtn.querySelector("span").textContent = opening ? "Masquer l'analyse par bus" : "Expliquer un bus";
        if (opening && !explainLoaded) {
            explainLoaded = true;
            renderExplainPanel(explainPanel);
        }
        if (opening) explainPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    const resBox = root.querySelector("#wc-t-results");
    resBox.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Chargement des trajets signalés aujourd'hui…</p>`;
    try {
        const today = await api("/api/current-anomalies", {});
        const freshness = (today.live
            ? `<div class="wc-banner success">${LIVE_DOT}Données en direct — ${fmtDay(today.date)}</div>`
            : `<div class="wc-banner info">${icon("chart")}Dernier jour historique disponible — ${fmtDay(today.date)}</div>`)
            + cacheNote(today);
        if (!today.anomalies || !today.anomalies.length) {
            resBox.innerHTML = freshness + `<div class="wc-banner success">${icon("check")}Aucune anomalie aujourd'hui pour cet opérateur.</div>`
                + `<div id="wc-t-history"></div>`;
        } else {
            resBox.innerHTML = `<div class="wc-card"><h4>Trajets ce jour</h4>${freshness}<div id="wc-t-pills"></div><div id="wc-t-cards"></div></div>
                <div id="wc-t-history"></div>`;
            const cardsBox = resBox.querySelector("#wc-t-cards");
            const pillsBox = resBox.querySelector("#wc-t-pills");
            function draw(list) {
                cardsBox.innerHTML = "";
                for (const a of list) cardsBox.appendChild(renderAlertCard(a, { withMap: true }));
            }
            const getFiltered = categoryFilterPills(pillsBox, today.anomalies, draw);
            draw(getFiltered());
        }
        loadHistory(resBox.querySelector("#wc-t-history"));
    } catch (e) {
        resBox.innerHTML = `<div class="wc-banner error">Erreur : ${esc(e.message)}</div>`;
    }

    // Historique complet des anomalies sous la vue du jour -- même structure que l'onglet
    // "Trajets signalés" du dashboard Streamlit. Chargé séparément APRÈS la vue du jour
    // pour que le direct s'affiche vite ; paginé côté client (bouton "Afficher plus") pour
    // ne pas insérer des centaines de cartes DOM d'un coup.
    async function loadHistory(box) {
        box.innerHTML = `<div class="wc-card"><h4>Historique des anomalies</h4>
            <p class="wc-muted"><span class="wc-spin"></span> Chargement de l'historique…</p></div>`;
        let hist;
        try {
            hist = await api("/api/anomaly-history", { limit: 300 });
        } catch (e) {
            box.innerHTML = `<div class="wc-card"><h4>Historique des anomalies</h4>
                <div class="wc-banner error">${icon("alert")}Erreur : ${esc(e.message)}</div></div>`;
            return;
        }
        const all = (hist || {}).anomalies || [];
        if (!all.length) {
            box.innerHTML = `<div class="wc-card"><h4>Historique des anomalies</h4>
                <div class="wc-banner info">${icon("info")}Aucun historique d'anomalies pour cet opérateur.</div></div>`;
            return;
        }
        box.innerHTML = `<div class="wc-card"><h4>Historique des anomalies</h4>${cacheNote(hist)}
            <div class="wc-sort-row"><label>Trier par</label><select id="wc-th-sort">${
                Object.entries(SORT_OPTIONS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")
            }</select></div>
            <div id="wc-th-pills"></div><div id="wc-th-cards"></div>
            <button id="wc-th-more" class="wc-btn-secondary" style="margin-top:10px">Afficher plus</button></div>`;
        const cardsBox = box.querySelector("#wc-th-cards");
        const pillsBox = box.querySelector("#wc-th-pills");
        const moreBtn = box.querySelector("#wc-th-more");
        const sortSel = box.querySelector("#wc-th-sort");
        const PAGE = 15;
        let shown = PAGE;
        let current = all;
        function draw() {
            cardsBox.innerHTML = "";
            const sorted = sortAnomalies(current, sortSel.value);
            for (const a of sorted.slice(0, shown)) cardsBox.appendChild(renderAlertCard(a, { withMap: true }));
            moreBtn.style.display = shown < sorted.length ? "" : "none";
            moreBtn.textContent = `Afficher plus (${Math.min(shown, sorted.length)}/${sorted.length})`;
        }
        moreBtn.addEventListener("click", () => { shown += PAGE; draw(); });
        sortSel.addEventListener("change", () => { shown = PAGE; draw(); });
        const getFiltered = categoryFilterPills(pillsBox, all, (list) => { current = list; shown = PAGE; draw(); });
        current = getFiltered();
        draw();
    }
}

// ── Panneau "Expliquer un bus" (filtres + verdict de ligne + référence + analyse) ───────
// Miroir de l'ex-onglet tab_explain du dashboard Streamlit, maintenant repliable DANS
// l'onglet "Trajets signalés" (fusion décidée 2026-07-19) plutôt qu'un onglet séparé --
// `root` ici est le panneau repliable (#wc-t-explain-panel), pas la racine de l'onglet ;
// self-contained comme avant, seule sa position dans le DOM change. Verdict de ligne,
// trajet de référence avec carte, avertissements modèle, métriques avec explications,
// tri, catégories, cartes détaillées avec carte par trajet.
async function renderExplainPanel(root) {
    root.innerHTML = `
    <div class="wc-card">
        <div class="wc-filters">
            <div class="wc-field">
                <label>Ligne</label>
                <select id="wc-e-line"><option value="">Toutes les lignes</option></select>
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
            <div class="wc-field">
                <label>&nbsp;</label>
                <button id="wc-e-analyze">Analyser</button>
            </div>
        </div>
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
        hint.textContent = "Choisissez une ligne précise ou laissez « Toutes les lignes » et cliquez Analyser.";
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
            <div class="wc-ref-dir">
                <h4>${esc(d)}</h4>
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

        // Cartes chargées à l'OUVERTURE du bloc (pas d'office) -- Leaflet dans un <details>
        // fermé mesure une taille nulle et rend une carte grise ; on attend le premier
        // toggle pour instancier, puis c'est fait une fois pour toutes.
        const details = refBox.querySelector("details");
        let mapsDone = false;
        details.addEventListener("toggle", async () => {
            if (!details.open || mapsDone) return;
            mapsDone = true;
            try { await ensureLeaflet(); } catch { return; }
            for (const d of dirNames) {
                const holder = refBox.querySelector(`.wc-ref-map[data-dir="${CSS.escape(d)}"]`);
                if (holder && dirs[d].sequence) renderTripMap(holder, dirs[d].sequence, d, null);
            }
        });
    }

    async function runAnalysis() {
        const line = lineSel.value || null;
        const bus = busSel.value || null;
        const dir = dirSel.value || null;
        let day = daySel.value || null;
        if (manualDate.value) day = manualDate.value.replace(/-/g, "");

        showLoadingIn(resBox);
        try {
            // check_detours toujours actif sur un clic "Analyser" délibéré -- mesuré
            // ~30-40s sur une ligne très signalée (et plus sur "toutes les lignes"),
            // assumé derrière l'animation de chargement.
            const res = await api("/api/anomaly-explain", { line, bus, day, dir, check_detours: true });
            renderExplainResults(res, { line, bus });
        } catch (e) {
            resBox.innerHTML = `<div class="wc-banner error">${icon("alert")}Erreur d'analyse : ${esc(e.message)}</div>`;
        } finally {
            stopLoading();
        }
    }
    root.querySelector("#wc-e-analyze").addEventListener("click", runAnalysis);

    function renderExplainResults(res, scope) {
        if (!res || res.anomaly_count === 0) {
            const label = scope.bus ? `Bus ${scope.bus} · Ligne ${scope.line}` : (scope.line ? `Ligne ${scope.line}` : "Toutes les lignes");
            resBox.innerHTML = `<div class="wc-banner success">${icon("check")}${label} : aucun trajet anormal détecté — tout est dans la normale.</div>`;
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
            <h4 style="margin-top:14px">Trajets signalés</h4>
            <div class="wc-sort-row"><label>Trier par</label><select id="wc-e-sort">${
                Object.entries(SORT_OPTIONS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")
            }</select></div>
            <div class="wc-pills" id="wc-e-pills"></div>
            <div id="wc-e-cards"></div>
        </div>`;

        const cardsBox = resBox.querySelector("#wc-e-cards");
        const pillsBox = resBox.querySelector("#wc-e-pills");
        const sortSel = resBox.querySelector("#wc-e-sort");
        let current = res.anomalies;
        function draw() {
            cardsBox.innerHTML = "";
            for (const a of sortAnomalies(current, sortSel.value)) {
                cardsBox.appendChild(renderAlertCard(a, { withMap: true }));
            }
        }
        sortSel.addEventListener("change", draw);
        const getFiltered = categoryFilterPills(pillsBox, res.anomalies, (list) => { current = list; draw(); });
        current = getFiltered();
        draw();
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
            for (const a of anomalies) cardsBox.appendChild(renderTicketCard(a));
        } catch (e) {
            body.innerHTML = `<div class="wc-banner error">Erreur : ${esc(e.message)}</div>`;
        }
    }
    viewSel.addEventListener("change", load);
    lineSel.addEventListener("change", load);
    load();
}
function renderTicketCard(a) {
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
    if (a.line_median_avg_fare !== undefined && a.line_median_avg_fare !== null) {
        html += `<p class="wc-muted">Médiane ligne : ${a.line_median_nbr_ticket ?? "—"} tickets · ${(a.line_median_recette ?? 0).toFixed(0)} DT · ${(a.line_median_avg_fare ?? 0).toFixed(2)} DT/ticket</p>`;
    }
    html += `</div>`;
    return el(html);
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
