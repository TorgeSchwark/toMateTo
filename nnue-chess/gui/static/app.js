const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const expandedLogs = new Set();

// ---- tab switching --------------------------------------------------------
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const tool = tab.dataset.tool;
    $("#fc-form").classList.toggle("hidden", tool !== "full_cycle");
    $("#sp-form").classList.toggle("hidden", tool !== "self_play");
    $("#st-form").classList.toggle("hidden", tool !== "strength_search");
  });
});

$("#fc-mode").addEventListener("change", (e) => {
  const mode = e.target.value;
  $(".fc-gen-fields").classList.toggle("hidden", mode === "only_train");
  $(".fc-train-fields").classList.toggle("hidden", mode !== "only_train");
});

$("#sp-td-mode").addEventListener("change", (e) => {
  const lambda = e.target.value === "lambda";
  $(".sp-leaf-fields").classList.toggle("hidden", lambda);
  $(".sp-lambda-fields").classList.toggle("hidden", !lambda);
});

$("#build-toggle").addEventListener("click", () => {
  $("#build-form").classList.toggle("hidden");
});

$("#st-gen-dataset").addEventListener("change", (e) => {
  $(".st-dataset-fields").classList.toggle("hidden", !e.target.checked);
});

// ---- bucket-NNUE rows (strength form) ---------------------------------------
let latestNetworks = [];

function bucketRowHtml(min, max) {
  return `
    <div class="bucket-row">
      <input type="number" class="bucket-min" min="2" max="32" value="${min}">
      <span class="hint">–</span>
      <input type="number" class="bucket-max" min="2" max="32" value="${max}">
      <select class="bucket-net"></select>
      <button type="button" class="bucket-remove">✕</button>
    </div>`;
}

function fillBucketNetSelect(sel) {
  const current = sel.value;
  sel.innerHTML = latestNetworks.length
    ? latestNetworks.map((n) => `<option value="${n}">${n}</option>`).join("")
    : `<option value="">(keine .nnue Dateien gefunden)</option>`;
  if (latestNetworks.includes(current)) sel.value = current;
}

function addBucketRow(min, max) {
  const wrap = document.getElementById("st-bucket-rows");
  const div = document.createElement("div");
  div.innerHTML = bucketRowHtml(min, max);
  const row = div.firstElementChild;
  wrap.appendChild(row);
  fillBucketNetSelect(row.querySelector(".bucket-net"));
  row.querySelector(".bucket-remove").addEventListener("click", () => {
    row.remove();
    updateBucketCoverage();
  });
  row.querySelectorAll(".bucket-min, .bucket-max").forEach((inp) =>
    inp.addEventListener("input", updateBucketCoverage)
  );
  updateBucketCoverage();
}

function readBucketRows() {
  return $$("#st-bucket-rows .bucket-row").map((row) => ({
    min: parseInt(row.querySelector(".bucket-min").value, 10),
    max: parseInt(row.querySelector(".bucket-max").value, 10),
    net: row.querySelector(".bucket-net").value,
  }));
}

// Mirrors validate_bucket_coverage() in toMateTo_nnue.cpp (client-side
// preview only - the server/C++ side is still the real check before any
// game is played) so a gap or overlap is visible while still editing
// instead of only after submitting.
function updateBucketCoverage() {
  const rows = readBucketRows().slice().sort((a, b) => a.min - b.min);
  const el = document.getElementById("st-bucket-coverage");
  if (rows.length === 0) {
    el.textContent = "";
    return;
  }
  if (rows[0].min !== 2) {
    el.textContent = `Abdeckung beginnt bei ${rows[0].min}, muss bei 2 beginnen`;
    el.style.color = "var(--bad)";
    return;
  }
  for (let i = 0; i + 1 < rows.length; i++) {
    if (rows[i].max + 1 < rows[i + 1].min) {
      el.textContent = `Lücke zwischen ${rows[i].max} und ${rows[i + 1].min}`;
      el.style.color = "var(--bad)";
      return;
    }
    if (rows[i].max >= rows[i + 1].min) {
      el.textContent = `Überlappung: ${rows[i].min}-${rows[i].max} und ${rows[i + 1].min}-${rows[i + 1].max}`;
      el.style.color = "var(--bad)";
      return;
    }
  }
  if (rows[rows.length - 1].max !== 32) {
    el.textContent = `Abdeckung endet bei ${rows[rows.length - 1].max}, muss bei 32 enden`;
    el.style.color = "var(--bad)";
    return;
  }
  el.textContent = `deckt 2-32 vollständig ab (${rows.length} Buckets)`;
  el.style.color = "var(--good)";
}

document.getElementById("st-add-bucket").addEventListener("click", () => addBucketRow(2, 32));

document.getElementById("st-use-buckets").addEventListener("change", (e) => {
  const on = e.target.checked;
  document.getElementById("st-buckets-wrap").classList.toggle("hidden", !on);
  document.getElementById("st-net-single-wrap").classList.toggle("hidden", on);
  if (on && $$("#st-bucket-rows .bucket-row").length === 0) {
    addBucketRow(2, 15);
    addBucketRow(16, 32);
  }
});

// ---- starting jobs ---------------------------------------------------------
function formToBody(form, tool, extra) {
  const data = new FormData(form);
  const body = { tool };
  for (const [k, v] of data.entries()) body[k] = v;
  if (extra) Object.assign(body, extra);
  return body;
}

async function startJob(tool, form, extra) {
  const errBox = $("#start-error");
  errBox.classList.add("hidden");
  try {
    const res = await fetch("/api/jobs/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formToBody(form, tool, extra)),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || "Start fehlgeschlagen");
    await refresh();
  } catch (e) {
    errBox.textContent = e.message;
    errBox.classList.remove("hidden");
  }
}

$("#fc-form").addEventListener("submit", (e) => {
  e.preventDefault();
  startJob("full_cycle", e.target);
});
$("#sp-form").addEventListener("submit", (e) => {
  e.preventDefault();
  startJob("self_play", e.target);
});
$("#st-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const useBuckets = document.getElementById("st-use-buckets").checked;
  if (useBuckets) {
    const rows = readBucketRows();
    if (rows.some((r) => !r.net)) {
      const errBox = $("#start-error");
      errBox.textContent = "jeder Bucket braucht ein ausgewähltes Netz";
      errBox.classList.remove("hidden");
      return;
    }
    startJob("strength_search", e.target, { net_buckets: rows });
  } else {
    startJob("strength_search", e.target);
  }
});

$("#build-start").addEventListener("click", async () => {
  const errBox = $("#build-status");
  errBox.textContent = "baue...";
  try {
    const res = await fetch("/api/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        acc: $("#b-acc").value,
        h1: $("#b-h1").value,
        h2: $("#b-h2").value,
        h3: $("#b-h3").value,
      }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || "Build fehlgeschlagen");
    errBox.textContent = "Build gestartet, läuft im Hintergrund...";
  } catch (e) {
    errBox.textContent = "Fehler: " + e.message;
  }
});

async function stopJob(id) {
  await fetch(`/api/jobs/${id}/stop`, { method: "POST" });
  await refresh();
}

// ---- rendering --------------------------------------------------------------
function fmtElapsed(start, end) {
  if (!start) return "-";
  const secs = Math.round((end || Date.now() / 1000) - start);
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function statusLabel(status) {
  return { running: "läuft", done: "fertig", failed: "fehlgeschlagen", stopped: "gestoppt",
           stopping: "stoppe..." }[status] || status;
}

function renderJob(job) {
  const pct = job.progress ? job.progress.pct : job.status === "done" ? 100 : 0;
  const showBar = job.progress || job.status === "running";
  const isOpen = expandedLogs.has(job.id);
  const extLabel = job.external ? ' <span class="job-sub">(extern, außerhalb der GUI gestartet)</span>' : "";
  return `
  <div class="job-card" data-id="${job.id}">
    <div class="job-head">
      <div>
        <span class="job-title">${job.tool}${job.name ? " · " + job.name : ""}</span>
        ${extLabel}
      </div>
      <span class="badge ${job.status}">${statusLabel(job.status)}</span>
    </div>
    <div class="job-sub">${job.cmd_str}</div>
    ${showBar ? `
    <div class="bar-outer"><div class="bar-inner" style="width:${pct}%"></div></div>
    ` : ""}
    <div class="job-meta">
      ${job.progress ? `<span>${job.progress.cur}/${job.progress.tot} (${job.progress.pct}%)</span>
      <span>${job.progress.rate_ms.toFixed(1)}ms/item</span>
      <span>ETA ${job.progress.eta_s}s</span>
      ${job.progress.suffix ? `<span>${job.progress.suffix}</span>` : ""}` : ""}
      <span>Threads: ${job.threads || "-"}</span>
      <span>Laufzeit: ${fmtElapsed(job.start_time, job.end_time)}</span>
      ${job.returncode !== null && job.returncode !== undefined ? `<span>Exit: ${job.returncode}</span>` : ""}
    </div>
    <div class="job-actions">
      ${job.status === "running" || job.status === "stopping" ?
        `<button onclick="stopJob('${job.id}')">Stoppen</button>` : ""}
      ${job.log_tail !== undefined ?
        `<button onclick="toggleLog('${job.id}')">${isOpen ? "Log verbergen" : "Log anzeigen"}</button>` : ""}
    </div>
    ${isOpen ? `<div class="log-tail">${escapeHtml(job.log_tail || "(kein Log)")}</div>` : ""}
  </div>`;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function toggleLog(id) {
  if (expandedLogs.has(id)) expandedLogs.delete(id);
  else expandedLogs.add(id);
  refresh();
}
window.stopJob = stopJob;
window.toggleLog = toggleLog;

function renderLeaderboard(rows) {
  const tbody = $("#leaderboard-table tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="hint">Noch keine Ergebnisse.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r, i) => `<tr>
      <td>${i + 1}</td>
      <td>${r.name}</td>
      <td>${r.method}</td>
      <td>${r.architecture}</td>
      <td>${r.best_loss.toFixed(5)}</td>
      <td>${r.validation_loss >= 0 ? r.validation_loss.toFixed(5) : "-"}</td>
      <td>${r.validation_correlation !== -1 ? r.validation_correlation.toFixed(3) : "-"}</td>
      <td>${(r.total_time_ms / 1000).toFixed(1)}s</td>
    </tr>`
    )
    .join("");
}

function renderStrengthLeaderboard(rows) {
  const tbody = $("#strength-leaderboard-table tbody");
  if (!rows || !rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="hint">Noch keine Spielstärke-Ergebnisse.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r, i) => `<tr>
      <td>${i + 1}</td>
      <td>${r.name}</td>
      <td>${r.net}</td>
      <td>${r.estimated_elo}</td>
      <td>[${r.bracket_lo}, ${r.bracket_hi}]</td>
      <td>${r.rounds}</td>
      <td>${r.total_games}</td>
    </tr>`
    )
    .join("");
}

async function populateSelect(sel, options, placeholder) {
  const current = sel.value;
  sel.innerHTML = placeholder ? `<option value="">${placeholder}</option>` : "";
  for (const opt of options) {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    sel.appendChild(o);
  }
  if (options.includes(current)) sel.value = current;
}

// ---- loss plot --------------------------------------------------------------
let currentLossRun = null;

function drawLossChart(steps, losses) {
  const canvas = document.getElementById("loss-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  document.getElementById("loss-empty").classList.toggle("hidden", steps.length > 1);
  if (steps.length < 2) return;

  const pad = { l: 55, r: 15, t: 10, b: 30 };
  const minLoss = Math.min(...losses), maxLoss = Math.max(...losses);
  const lossRange = maxLoss - minLoss || 1;
  const maxStep = steps[steps.length - 1] || 1;

  const x = (s) => pad.l + (s / maxStep) * (w - pad.l - pad.r);
  const y = (l) => h - pad.b - ((l - minLoss) / lossRange) * (h - pad.t - pad.b);

  ctx.strokeStyle = "#2a3341";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();

  ctx.fillStyle = "#8a97a8";
  ctx.font = "11px sans-serif";
  ctx.fillText(minLoss.toFixed(5), 4, y(minLoss));
  ctx.fillText(maxLoss.toFixed(5), 4, y(maxLoss) + 4);
  ctx.fillText("0", pad.l - 4, h - pad.b + 14);
  ctx.fillText(String(maxStep), w - pad.r - 20, h - pad.b + 14);

  ctx.strokeStyle = "#4f8cff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < steps.length; i++) {
    const px = x(steps[i]), py = y(losses[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

async function loadLossCurve(name) {
  if (!name) {
    drawLossChart([], []);
    return;
  }
  const res = await fetch(`/api/loss?name=${encodeURIComponent(name)}`);
  const data = await res.json();
  drawLossChart(data.steps || [], data.losses || []);
}

document.getElementById("loss-select").addEventListener("change", (e) => {
  currentLossRun = e.target.value;
  loadLossCurve(currentLossRun);
});

// ---- dataset visualization ---------------------------------------------------
function drawBarChart(canvasId, labels, values, opts = {}) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!values.length) return;

  const pad = { l: 45, r: 15, t: 10, b: 36 };
  const maxV = Math.max(...values, 1);
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const barW = plotW / values.length;

  ctx.strokeStyle = "#2a3341";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();

  ctx.fillStyle = opts.color || "#4f8cff";
  for (let i = 0; i < values.length; i++) {
    const barH = (values[i] / maxV) * plotH;
    ctx.fillRect(pad.l + i * barW + 1, h - pad.b - barH, Math.max(1, barW - 2), barH);
  }

  ctx.fillStyle = "#8a97a8";
  ctx.font = "10px sans-serif";
  ctx.fillText(String(maxV), 4, pad.t + 8);
  ctx.fillText("0", 4, h - pad.b);
  const labelEvery = Math.ceil(labels.length / 16);
  for (let i = 0; i < labels.length; i += labelEvery) {
    ctx.fillText(String(labels[i]), pad.l + i * barW, h - pad.b + 12);
  }
}

async function analyzeDataset() {
  const name = document.getElementById("dataset-select").value;
  const statusEl = document.getElementById("dataset-status");
  const emptyEl = document.getElementById("dataset-empty");
  const summaryEl = document.getElementById("dataset-summary");
  if (!name) return;
  statusEl.textContent = "analysiere...";
  summaryEl.innerHTML = "";
  try {
    const res = await fetch(`/api/dataset_stats?name=${encodeURIComponent(name)}`);
    const data = await res.json();
    if (data.error || !data.n_sampled) {
      statusEl.textContent = data.error || "keine Daten";
      drawBarChart("dataset-canvas", [], []);
      emptyEl.classList.remove("hidden");
      return;
    }
    emptyEl.classList.add("hidden");
    const labels = data.piece_histogram.map((_, i) => i);
    drawBarChart("dataset-canvas", labels, data.piece_histogram);
    statusEl.textContent = `${data.n_sampled} Stellungen analysiert (${data.file_size_mb} MB Datei)`;
    const p = data.phase_pct, cp = data.cp;
    summaryEl.innerHTML = `
      <span>Eröffnung: ${p.opening}%</span>
      <span>Mittelspiel: ${p.midgame}%</span>
      <span>Endspiel: ${p.endgame}%</span>
      <span>Tiefes Endspiel: ${p.deep_endgame}%</span>
      <span>Eval (cp): ${cp.min} / ${cp.p10} / ${cp.p50} / ${cp.p90} / ${cp.max} (min/p10/p50/p90/max)</span>
      <span>Weiß am Zug: ${data.white_pct}%</span>
    `;
  } catch (e) {
    statusEl.textContent = "Fehler: " + e.message;
  }
}

document.getElementById("dataset-analyze").addEventListener("click", analyzeDataset);

async function refresh() {
  const res = await fetch("/api/status");
  const state = await res.json();

  const cores = state.cores;
  const used = state.active_threads;
  const gauge = $("#cores-gauge");
  gauge.textContent = `Kerne: ${used}/${cores} belegt`;
  gauge.classList.toggle("warn", used > cores);

  const arch = state.build_arch;
  $("#arch-pill").textContent = arch
    ? `Architektur: ${arch.acc},${arch.h1},${arch.h2},${arch.h3}${state.building ? " (baut...)" : ""}`
    : "Architektur: nicht gebaut";

  const jobsList = $("#jobs-list");
  if (!state.jobs.length) {
    jobsList.innerHTML = `<p class="hint">Noch keine Trainings gestartet.</p>`;
  } else {
    const sorted = [...state.jobs].sort((a, b) => {
      const rank = (s) => (s === "running" || s === "stopping" ? 0 : 1);
      return rank(a.status) - rank(b.status) || (b.start_time || 0) - (a.start_time || 0);
    });
    jobsList.innerHTML = sorted.map(renderJob).join("");
  }

  renderLeaderboard(state.leaderboard);
  renderStrengthLeaderboard(state.strength_leaderboard);
  await populateSelect($("#fc-dataset"), state.datasets, state.datasets.length ? null : "(keine .dataset Dateien gefunden)");
  await populateSelect($("#sp-init-from"), state.networks, "(keins)");
  await populateSelect($("#st-net"), state.networks, state.networks.length ? null : "(keine .nnue Dateien gefunden)");
  latestNetworks = state.networks || [];
  $$("#st-bucket-rows .bucket-net").forEach(fillBucketNetSelect);
  await populateSelect($("#dataset-select"), state.datasets, state.datasets.length ? null : "(keine .dataset Dateien gefunden)");

  const lossSel = document.getElementById("loss-select");
  const curves = state.loss_curves || [];
  await populateSelect(lossSel, curves, curves.length ? null : "(keine Läufe mit Loss-Daten gefunden)");
  if (!currentLossRun && curves.length) {
    currentLossRun = curves[0];
    lossSel.value = currentLossRun;
  }
  if (currentLossRun) await loadLossCurve(currentLossRun);
}

refresh();
setInterval(refresh, 1500);
