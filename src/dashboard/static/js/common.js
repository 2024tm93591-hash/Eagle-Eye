// Shared by every dashboard page: the live event stream, alert cards/toasts and the activity bars.
const TV = (() => {
  const handlers = { alert: [], stats: [], ack: [] };
  const LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
  let audioCtx = null;

  function esc(value) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return String(value ?? "").replace(/[&<>"']/g, c => map[c]);
  }

  const fmtType = type => String(type || "").replace(/_/g, " ");
  const on = (kind, fn) => handlers[kind].push(fn);

  function connect() {
    const source = new EventSource("/api/stream");
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = event => {
      const msg = JSON.parse(event.data);
      if (msg.kind === "alert") {
        onAlert(msg.alert);
        handlers.alert.forEach(fn => fn(msg.alert));
      } else if (msg.kind === "stats") {
        handlers.stats.forEach(fn => fn(msg.stats));
      } else if (msg.kind === "ack") {
        handlers.ack.forEach(fn => fn(msg.id));
      }
    };
  }

  function setConnected(ok) {
    document.getElementById("connDot").classList.toggle("off", !ok);
    document.getElementById("connText").textContent = ok ? "Live - connected" : "Reconnecting…";
  }

  function beep(level) {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.frequency.value = level === "CRITICAL" ? 880 : 620;
      gain.gain.value = 0.05;
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + (level === "CRITICAL" ? 0.45 : 0.18));
    } catch (e) {
      // browsers block audio until the user has clicked somewhere on the page
    }
  }

  function onAlert(alert) {
    refreshBadge();
    if (alert.level === "HIGH" || alert.level === "CRITICAL") {
      toast(alert);
      beep(alert.level);
    }
    if (alert.level === "CRITICAL") {
      document.getElementById("critText").textContent =
        `CRITICAL: ${alert.description} (${alert.camera}, ${alert.time.slice(11)})`;
      document.getElementById("critBanner").classList.add("show");
    }
  }

  function dismissBanner() {
    document.getElementById("critBanner").classList.remove("show");
  }

  function toast(alert) {
    const box = document.getElementById("toasts");
    const el = document.createElement("div");
    el.innerHTML = alertCard(alert, false);
    box.prepend(el);
    while (box.children.length > 3) box.lastChild.remove();
    setTimeout(() => el.remove(), 7000);
  }

  function alertCard(a, withAck = true) {
    const snapshot = a.snapshot
      ? ` · <a href="#" onclick="TV.showSnapshot('${esc(a.snapshot)}', '${esc(a.description)}'); return false">snapshot</a>`
      : "";
    let ack = "";
    if (a.acknowledged) ack = `<span class="pill">acknowledged</span>`;
    else if (withAck) ack = `<button class="btn small" onclick="TV.ack('${a.id}', this)">Acknowledge</button>`;

    return `<div class="alert-card ${a.level} ${a.acknowledged ? "acked" : ""}" data-id="${a.id}">
      <div class="d">${esc(a.description)}</div>
      <div><span class="lvl ${a.level}">${a.level} ${Math.round(a.score)}</span></div>
      <div class="m">${esc(a.time)} · ${esc(fmtType(a.type))}${snapshot}</div>
      <div class="m" title="How the score was built">${esc((a.reasons || []).join(" · "))}</div>
      <div class="m">${ack}</div>
    </div>`;
  }

  async function ack(id, button) {
    await fetch(`/api/alerts/${id}/ack`, { method: "POST" });
    const card = button && button.closest(".alert-card");
    if (card) card.classList.add("acked");
    if (button) {
      const label = document.createElement("span");
      label.className = "pill";
      label.textContent = "acknowledged";
      button.replaceWith(label);
    }
    refreshBadge();
  }

  function showSnapshot(name, title) {
    document.getElementById("modalBox").innerHTML =
      `<h2 style="margin:0 0 10px;font-size:15px">${esc(title)}</h2><img src="/snapshots/${encodeURIComponent(name)}">`;
    document.getElementById("modal").classList.add("show");
  }

  async function refreshBadge() {
    try {
      const { count } = await (await fetch("/api/alerts/unacknowledged")).json();
      const badge = document.getElementById("unackBadge");
      badge.textContent = count;
      badge.style.display = count ? "" : "none";
    } catch (e) {
      // the server is probably restarting; the next alert will try again
    }
  }

  // simple horizontal bars: items is [{label, value, color}]
  function hbarChart(el, items) {
    if (!items.length) {
      el.innerHTML = `<div class="empty">No data yet</div>`;
      return;
    }
    const max = Math.max(...items.map(i => i.value), 1);
    el.innerHTML = `<div class="bars">${items.map(i => `
      <div class="row">
        <span class="lab" title="${esc(i.label)}">${esc(i.label)}</span>
        <div class="track"><div class="fill" style="width:${100 * i.value / max}%;background:${i.color || "var(--accent)"}"></div></div>
        <span class="n">${i.value}</span>
      </div>`).join("")}</div>`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    connect();
    refreshBadge();
  });

  return { on, esc, fmtType, alertCard, ack, showSnapshot, dismissBanner, hbarChart, LEVELS };
})();
