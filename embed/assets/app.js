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
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
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

// ── Shared: alert card rendering (used by Trips view, Chauffeurs view) ─────────────────
function rowCategories(a) {
    const cats = [a.top_feature ?? null];
    if (a.has_detour) cats.push("unofficial_detour");
    return cats;
}
function renderAlertCard(a, { showDriverStatsHint = true } = {}) {
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
    return el(html);
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

// ── View: Trips & Analyse (merged Trajets signalés + Expliquer un bus) ─────────────────
async function renderTripsView(root) {
    root.innerHTML = `
    <div class="wc-card">
        <div class="wc-filters">
            <div class="wc-field">
                <label>Ligne</label>
                <select id="wc-t-line"><option value="">Toutes les lignes</option></select>
            </div>
            <div class="wc-field">
                <label>Bus</label>
                <select id="wc-t-bus"><option value="">Tous les bus</option></select>
            </div>
            <div class="wc-field">
                <label>Jour</label>
                <select id="wc-t-day"><option value="">Tous les jours (historique + aujourd'hui)</option></select>
            </div>
            <div class="wc-field">
                <label>Ou date précise</label>
                <input type="date" id="wc-t-manual-date">
            </div>
            <div class="wc-field">
                <label>Direction</label>
                <select id="wc-t-dir"><option value="">Les deux</option><option value="ALLER">ALLER</option><option value="RETOUR">RETOUR</option></select>
            </div>
            <div class="wc-field">
                <label>&nbsp;</label>
                <button id="wc-t-analyze">Analyser</button>
            </div>
        </div>
        <p class="wc-muted" id="wc-t-hint">Chargement des lignes…</p>
    </div>
    <div id="wc-t-badge"></div>
    <div id="wc-t-results"></div>
    `;

    const lineSel = root.querySelector("#wc-t-line");
    const busSel = root.querySelector("#wc-t-bus");
    const daySel = root.querySelector("#wc-t-day");
    const manualDate = root.querySelector("#wc-t-manual-date");
    const dirSel = root.querySelector("#wc-t-dir");
    const hint = root.querySelector("#wc-t-hint");

    api("/api/lines-ranked").then((d) => {
        for (const line of (d.lines || [])) {
            lineSel.appendChild(el(`<option value="${esc(line)}">${esc(line)}</option>`));
        }
        hint.textContent = "Choisissez une ligne précise ou laissez « Toutes les lignes » et cliquez Analyser.";
    }).catch(() => { hint.textContent = "Impossible de charger la liste des lignes."; });

    lineSel.addEventListener("change", async () => {
        busSel.innerHTML = `<option value="">Tous les bus</option>`;
        daySel.innerHTML = `<option value="">Tous les jours (historique + aujourd'hui)</option>`;
        if (!lineSel.value) return;
        const [busesD, daysD] = await Promise.all([
            api("/api/buses-for-line", { line: lineSel.value }),
            api("/api/days-for-line", { line: lineSel.value }),
        ]);
        for (const b of (busesD.buses || [])) busSel.appendChild(el(`<option value="${esc(b)}">${esc(b)}</option>`));
        for (const d of (daysD.days || [])) daySel.appendChild(el(`<option value="${esc(d)}">${esc(fmtDay(d))}</option>`));
    });

    async function runAnalysis() {
        const line = lineSel.value || null;
        const bus = busSel.value || null;
        const dir = dirSel.value || null;
        let day = daySel.value || null;
        if (manualDate.value) day = manualDate.value.replace(/-/g, "");

        showLoadingIn(root.querySelector("#wc-t-results"));
        try {
            // check_detours always on for an explicit "Analyser" click -- measured ~30-40s
            // on a heavily-flagged single line, and much more across "all lines" (confirmed
            // 2026-07-15: minutes-long on a large operator) -- acceptable for a deliberate
            // deep-dive click backed by the loading animation, NOT for the page's default
            // landing view (see the fast path below).
            const res = await api("/api/anomaly-explain", { line, bus, day, dir, check_detours: true });
            renderTripsResults(root, res);
        } catch (e) {
            root.querySelector("#wc-t-results").innerHTML = `<div class="wc-banner error">${icon("alert")}Erreur d'analyse : ${esc(e.message)}</div>`;
        } finally {
            stopLoading();
        }
    }
    root.querySelector("#wc-t-analyze").addEventListener("click", runAnalysis);

    // Fast default landing view: today's live-scored flagged trips only (no detour check,
    // no "all lines + all history" scan) -- mirrors the old lightweight "Trajets signalés"
    // list. The filters + Analyser button above are the deliberate, slower deep-dive (the
    // old "Expliquer un bus"), which is the actual merge the user asked for: one view, but
    // the expensive path only runs when asked for.
    (async () => {
        const resBox = root.querySelector("#wc-t-results");
        resBox.innerHTML = `<p class="wc-muted"><span class="wc-spin"></span> Chargement des trajets signalés aujourd'hui…</p>`;
        try {
            const today = await api("/api/current-anomalies", {});
            const freshness = today.live
                ? `<div class="wc-banner success">${LIVE_DOT}Données en direct — ${fmtDay(today.date)}</div>`
                : `<div class="wc-banner info">${icon("chart")}Dernier jour historique disponible — ${fmtDay(today.date)}</div>`;
            if (!today.anomalies || !today.anomalies.length) {
                resBox.innerHTML = freshness + `<div class="wc-banner success">${icon("check")}Aucune anomalie aujourd'hui pour cet opérateur.</div>`;
                return;
            }
            resBox.innerHTML = `<div class="wc-card">${freshness}<div id="wc-t-pills"></div><div id="wc-t-cards"></div></div>`;
            const cardsBox = resBox.querySelector("#wc-t-cards");
            const pillsBox = resBox.querySelector("#wc-t-pills");
            function draw(list) {
                cardsBox.innerHTML = "";
                for (const a of list) cardsBox.appendChild(renderAlertCard(a));
            }
            const getFiltered = categoryFilterPills(pillsBox, today.anomalies, draw);
            draw(getFiltered());
        } catch (e) {
            resBox.innerHTML = `<div class="wc-banner error">Erreur : ${esc(e.message)}</div>`;
        }
    })();
}

function renderTripsResults(root, res) {
    const badgeBox = root.querySelector("#wc-t-badge");
    const resBox = root.querySelector("#wc-t-results");
    badgeBox.innerHTML = "";
    resBox.innerHTML = "";

    if (!res || res.anomaly_count === 0) {
        resBox.innerHTML = `<div class="wc-banner success">${icon("check")}Aucune anomalie trouvée pour ce périmètre.</div>`;
        return;
    }

    const pct = res.total_trips ? (100 * res.anomaly_count / res.total_trips) : 0;
    resBox.innerHTML = `
    <div class="wc-card">
        <div class="wc-metrics">
            <div class="wc-metric"><div class="wc-metric-label">Trajets analysés</div><div class="wc-metric-value">${res.total_trips}</div></div>
            <div class="wc-metric"><div class="wc-metric-label">Trajets anormaux</div><div class="wc-metric-value">${res.anomaly_count} (${pct.toFixed(1)}%)</div></div>
            ${res.avg_duration_min ? `<div class="wc-metric"><div class="wc-metric-label">Durée normale (médiane)</div><div class="wc-metric-value">${fmtDuration(res.avg_duration_min)}</div></div>` : ""}
        </div>
        <div class="wc-pills" id="wc-t-pills"></div>
        <div id="wc-t-cards"></div>
    </div>`;

    const cardsBox = resBox.querySelector("#wc-t-cards");
    const pillsBox = resBox.querySelector("#wc-t-pills");

    function draw(list) {
        cardsBox.innerHTML = "";
        const sorted = [...list].sort((a, b) => (b.day || "").localeCompare(a.day || ""));
        for (const a of sorted) cardsBox.appendChild(renderAlertCard(a));
    }
    const getFiltered = categoryFilterPills(pillsBox, res.anomalies, draw);
    draw(getFiltered());
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
