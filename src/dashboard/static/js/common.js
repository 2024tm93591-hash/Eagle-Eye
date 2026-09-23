/* ThreatVision dashboard - shared helpers: live event stream, alerts UI, tiny SVG charts. */
const TV = (() => {
  const handlers = { alert: [], stats: [], ack: [] };
  let es = null, audioCtx = null, soundOn = true;
  const LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
  const PALETTE = ["#3b82f6", "#22d3ee", "#f5b400", "#d946ef", "#22c55e", "#ff7a1a", "#a78bfa", "#ef4444"];

  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const on = (kind, fn) => handlers[kind].push(fn);
  const fmtType = t => String(t || "").replace(/_/g, " ");
  const ago = ts => { const s = Math.max(0, Date.now() / 1000 - ts); return s < 60 ? `${s | 0}s ago` : s < 3600 ? `${(s / 60) | 0}m ago` : `${(s / 3600) | 0}h ago`; };

  function connect() {
    es = new EventSource("/api/stream");
    es.onopen = () => setConn(true);
    es.onerror = () => setConn(false);
    es.onmessage = ev => {
      const m = JSON.parse(ev.data);
      if (m.kind === "alert") { onAlert(m.alert); handlers.alert.forEach(f => f(m.alert)); }
      else if (m.kind === "stats") handlers.stats.forEach(f => f(m.stats));
      else if (m.kind === "ack") handlers.ack.forEach(f => f(m.id));
    };
  }
  function setConn(ok) {
    document.getElementById("connDot").classList.toggle("off", !ok);
    document.getElementById("connText").textContent = ok ? "Live - connected" : "Reconnecting…";
  }

  function beep(level) {
    if (!soundOn) return;
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.frequency.value = level === "CRITICAL" ? 880 : 620; g.gain.value = 0.05;
      o.connect(g); g.connect(audioCtx.destination); o.start(); o.stop(audioCtx.currentTime + (level === "CRITICAL" ? 0.45 : 0.18));
    } catch (e) { /* audio blocked until the user interacts with the page */ }
  }

  function onAlert(a) {
    refreshBadge();
    if (a.level === "HIGH" || a.level === "CRITICAL") {
      toast(a); beep(a.level);
    }
    if (a.level === "CRITICAL") {
      document.getElementById("critText").textContent = `CRITICAL: ${a.description} (${a.camera}, ${a.time.slice(11)})`;
      document.getElementById("critBanner").classList.add("show");
    }
  }
  function dismissBanner() { document.getElementById("critBanner").classList.remove("show"); }

  function toast(a) {
    const box = document.getElementById("toasts");
    const el = document.createElement("div");
    el.innerHTML = alertCard(a, false);
    box.prepend(el);
    while (box.children.length > 3) box.lastChild.remove();
    setTimeout(() => el.remove(), 7000);
  }

  function alertCard(a, withAck = true) {
    const snap = a.snapshot ? ` · <a href="#" onclick="TV.showSnapshot('${esc(a.snapshot)}','${esc(a.description)}');return false">snapshot</a>` : "";
    const ack = withAck && !a.acknowledged ? `<button class="btn small" onclick="TV.ack('${a.id}', this)">Acknowledge</button>` : (a.acknowledged ? `<span class="pill">acknowledged</span>` : "");
    return `<div class="alert-card ${a.level} ${a.acknowledged ? "acked" : ""}" data-id="${a.id}">
      <div class="d">${esc(a.description)}</div><div><span class="lvl ${a.level}">${a.level} ${Math.round(a.score)}</span></div>
      <div class="m">${esc(a.time)} · ${esc(fmtType(a.type))}${a.zone ? " · " + esc(a.zone) : ""}${snap}</div>
      <div class="m" title="How the score was built">${esc((a.reasons || []).join(" · "))}</div>
      <div class="m">${ack}</div></div>`;
  }

  async function ack(id, btn) {
    await fetch(`/api/alerts/${id}/ack`, { method: "POST" });
    const card = btn && btn.closest(".alert-card");
    if (card) { card.classList.add("acked"); btn.replaceWith(Object.assign(document.createElement("span"), { className: "pill", textContent: "acknowledged" })); }
    refreshBadge();
  }

  function showSnapshot(name, title) {
    document.getElementById("modalBox").innerHTML = `<h2 style="margin:0 0 10px;font-size:15px">${esc(title)}</h2><img src="/snapshots/${encodeURIComponent(name)}">`;
    document.getElementById("modal").classList.add("show");
  }

  async function refreshBadge() {
    try {
      const s = await (await fetch("/api/summary?hours=24")).json();
      const b = document.getElementById("unackBadge");
      b.textContent = s.unacknowledged; b.style.display = s.unacknowledged ? "" : "none";
    } catch (e) { }
  }

  /* ------------------------------------------------------------------ SVG charts */
  function svg(w, h, body) { return `<svg class="chart" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">${body}</svg>`; }

  // grouped vertical bar chart: labels (x groups), series [{name,color,values}]
  function barChart(el, { labels, series, max = null, fmt = v => v, height = 240 }) {
    const W = Math.max(320, el.clientWidth || 640), H = height, L = 40, B = 46, T = 14, R = 8;
    const all = series.flatMap(s => s.values).filter(v => v != null);
    const mx = max ?? Math.max(1e-9, ...all) * 1.12;
    const gw = (W - L - R) / Math.max(1, labels.length), bw = Math.max(2, Math.min(34, (gw - Math.min(12, gw * 0.3)) / series.length));
    let g = "";
    for (let i = 0; i <= 4; i++) {
      const y = T + (H - T - B) * (1 - i / 4), v = mx * i / 4;
      g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#2a3340"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${fmt(v)}</text>`;
    }
    labels.forEach((lab, i) => {
      const x0 = L + i * gw + (gw - bw * series.length) / 2;
      series.forEach((s, j) => {
        const v = s.values[i]; if (v == null) return;
        const h = (H - T - B) * v / mx, x = x0 + j * bw, y = H - B - h;
        g += `<rect x="${x + 1}" y="${y}" width="${bw - 2}" height="${h}" rx="3" fill="${s.color}"><title>${esc(s.name)} · ${esc(lab)}: ${fmt(v)}</title></rect>`;
        if (series.length <= 3 && bw > 20) g += `<text class="val" x="${x + bw / 2}" y="${y - 4}" text-anchor="middle">${fmt(v)}</text>`;
      });
      g += `<text x="${L + i * gw + gw / 2}" y="${H - B + 16}" text-anchor="middle">${esc(String(lab).slice(0, Math.max(6, gw / 6.5 | 0)))}</text>`;
    });
    el.innerHTML = svg(W, H, g) + legend(series);
  }

  // horizontal bars: items [{label, value, color}]
  function hbarChart(el, items, fmt = v => v) {
    if (!items.length) { el.innerHTML = `<div class="empty">No data yet</div>`; return; }
    const mx = Math.max(...items.map(i => i.value), 1);
    el.innerHTML = `<div class="bars">${items.map(i => `<div class="row"><span class="lab" title="${esc(i.label)}">${esc(i.label)}</span>
      <div class="track"><div class="fill" style="width:${100 * i.value / mx}%;background:${i.color || "var(--accent)"}"></div></div>
      <span class="n">${fmt(i.value)}</span></div>`).join("")}</div>`;
  }

  // multi-series line chart: x (epoch seconds), series [{name,color,values}]
  function lineChart(el, { x, series, height = 220 }) {
    if (!x.length) { el.innerHTML = `<div class="empty">No data yet - statistics are sampled every 5 seconds.</div>`; return; }
    const W = Math.max(320, el.clientWidth || 640), H = height, L = 34, B = 26, T = 10, R = 10;
    const mx = Math.max(4, Math.ceil(Math.max(...series.flatMap(s => s.values)) * 1.15 / 4) * 4), x0 = x[0], x1 = Math.max(x[x.length - 1], x0 + 1);
    const px = t => L + (W - L - R) * (t - x0) / (x1 - x0), py = v => T + (H - T - B) * (1 - v / mx);
    let g = "";
    for (let i = 0; i <= 4; i++) { const v = mx * i / 4; g += `<line x1="${L}" x2="${W - R}" y1="${py(v)}" y2="${py(v)}" stroke="#2a3340"/><text x="${L - 5}" y="${py(v) + 4}" text-anchor="end">${v.toFixed(0)}</text>`; }
    for (let i = 0; i <= 4; i++) { const t = x0 + (x1 - x0) * i / 4; g += `<text x="${px(t)}" y="${H - 8}" text-anchor="middle">${new Date(t * 1000).toTimeString().slice(0, 5)}</text>`; }
    series.forEach(s => { g += `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${x.map((t, i) => `${px(t).toFixed(1)},${py(s.values[i]).toFixed(1)}`).join(" ")}"/>`; });
    el.innerHTML = svg(W, H, g) + legend(series);
  }

  function legend(series) { return series.length > 1 ? `<div class="legend">${series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>` : ""; }

  // confusion matrix heat-map (rows = true label, cols = predicted), row-normalised colouring
  function confusion(el, labels, m, title = "") {
    const rows = m.map(r => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map(v => v / s); });
    let h = `<table class="cm"><tr><th class="rl muted">true ↓ / pred →</th>${labels.map(l => `<th class="muted">${esc(l)}</th>`).join("")}</tr>`;
    m.forEach((r, i) => {
      h += `<tr><th class="rl">${esc(labels[i])}</th>${r.map((v, j) => {
        const a = rows[i][j], bg = i === j ? `rgba(34,197,94,${0.12 + 0.75 * a})` : `rgba(239,68,68,${a > 0 ? 0.1 + 0.8 * a : 0.04})`;
        return `<td style="background:${bg}" title="${(100 * a).toFixed(1)}% of true ${esc(labels[i])}">${v}</td>`;
      }).join("")}</tr>`;
    });
    el.innerHTML = (title ? `<div class="muted" style="margin-bottom:6px">${esc(title)}</div>` : "") + h + "</table>";
  }

  document.addEventListener("DOMContentLoaded", () => { connect(); refreshBadge(); document.body.addEventListener("click", () => { soundOn = true; }, { once: true }); });
  return { on, esc, fmtType, ago, alertCard, ack, showSnapshot, dismissBanner, barChart, hbarChart, lineChart, confusion, LEVELS, PALETTE };
})();
