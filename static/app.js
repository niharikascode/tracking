"use strict";

const $ = (id) => document.getElementById(id);
const LS = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* storage unavailable */ } },
};

const state = {
  status: null,
  video: null,        // {video_id, name, size, meta}
  jobId: null,        // job shown in the output panel
  pollTimer: null,
  previewVersion: -1,
  key: LS.get("sam3.key") || "",
};

// ------------------------------------------------------------------ prompt building
// Mirrors tracker.build_prompt so people see exactly what SAM 3 receives.
const TRAILING = new Set(("with without wearing in on holding carrying that which who near at under behind next " +
  "beside inside riding pushing pulling sitting standing walking running having from of by over using").split(" "));

function buildPrompt(label, description) {
  label = label.trim();
  const desc = description.split(/\s+/).filter(Boolean).join(" ");
  if (!desc) return label;
  const esc = label.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if (label && new RegExp(`\\b${esc}s?\\b`).test(desc.toLowerCase())) return desc;
  if (TRAILING.has(desc.split(" ")[0].toLowerCase())) return `${label} ${desc}`;
  return `${desc} ${label}`;
}

// ------------------------------------------------------------------ api
function withKey(url) {
  if (!state.key) return url;
  return url + (url.includes("?") ? "&" : "?") + "key=" + encodeURIComponent(state.key);
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (state.key) headers["X-Access-Key"] = state.key;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) {
    askKey(true);
    throw new Error("Access key required");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(errorText(data) || `Request failed (${res.status})`);
  return data;
}

function errorText(data) {
  if (!data || !data.detail) return "";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((d) => {
      const where = (d.loc || []).filter((x) => x !== "body").join(" → ");
      return where ? `${where}: ${d.msg}` : d.msg;
    }).join("; ");
  }
  return JSON.stringify(data.detail);
}

// ------------------------------------------------------------------ access key
function askKey(failed) {
  const dlg = $("key-dialog");
  $("key-error").hidden = !failed || !state.key;
  if (!dlg.open) dlg.showModal();
  $("key-input").focus();
}

$("key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  state.key = $("key-input").value.trim();
  try {
    const res = await fetch("/api/auth-check", { headers: { "X-Access-Key": state.key } });
    if (!res.ok) throw new Error();
    LS.set("sam3.key", state.key);
    $("key-dialog").close();
    restore();
  } catch {
    $("key-error").hidden = false;
  }
});
$("key-dialog").addEventListener("cancel", (e) => e.preventDefault());

// ------------------------------------------------------------------ server status
async function refreshStatus() {
  try {
    const s = await fetch("/api/status").then((r) => r.json());
    state.status = s;
    const pill = $("server-pill");
    if (s.model === "loading") {
      setPill(pill, "pill-warn", "Model loading…");
    } else if (s.model === "error") {
      setPill(pill, "pill-bad", "Model failed to load");
      pill.title = s.model_error;
    } else if (s.busy || s.queue_length) {
      setPill(pill, "pill-info", s.queue_length ? `Busy · ${s.queue_length} waiting` : "Busy · GPU in use");
    } else {
      setPill(pill, "pill-ok", "Ready");
    }
    $("upload-limit").textContent = `MP4, MOV, MKV, WEBM, AVI · up to ${s.limits.max_upload_mb} MB`;
    $("opt-frames").max = s.limits.max_frames;
    updateEstimate();
  } catch {
    setPill($("server-pill"), "pill-bad", "Server offline");
  }
}

function setPill(el, cls, text) {
  el.className = "pill " + cls;
  el.textContent = text;
}

// ------------------------------------------------------------------ upload
const drop = $("dropzone");
const fileInput = $("file-input");

drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) upload(fileInput.files[0]); fileInput.value = ""; });
$("change-video").addEventListener("click", () => {
  setVideo(null);
  fileInput.click();
});

function upload(file) {
  hideFormError();
  const max = state.status?.limits?.max_upload_mb;
  if (max && file.size > max * 1024 * 1024) {
    showFormError(`That file is ${(file.size / 1048576).toFixed(0)} MB; the limit is ${max} MB.`);
    return;
  }
  drop.hidden = true;
  $("video-card").hidden = true;
  $("upload-progress").hidden = false;
  $("upload-name").textContent = `Uploading ${file.name}`;
  $("upload-bar").style.width = "0%";
  $("upload-pct").textContent = "0%";

  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/videos");
  if (state.key) xhr.setRequestHeader("X-Access-Key", state.key);
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    $("upload-bar").style.width = pct + "%";
    $("upload-pct").textContent = pct < 100 ? pct + "%" : "Checking video…";
  };
  xhr.onload = () => {
    $("upload-progress").hidden = true;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch { /* not json */ }
    if (xhr.status === 401) { setVideo(null); askKey(true); return; }
    if (xhr.status !== 200) {
      setVideo(null);
      showFormError(errorText(data) || `Upload failed (${xhr.status})`);
      return;
    }
    setVideo(data);
    showJob(null);
  };
  xhr.onerror = () => {
    $("upload-progress").hidden = true;
    setVideo(null);
    showFormError("Upload failed. Check your connection and try again.");
  };
  xhr.send(form);
}

function setVideo(v) {
  state.video = v;
  if (v) LS.set("sam3.video", JSON.stringify(v)); else LS.del("sam3.video");
  drop.hidden = !!v;
  $("video-card").hidden = !v;
  if (v) {
    $("video-thumb").src = withKey(`/api/videos/${v.video_id}/thumb.jpg`);
    $("video-name").textContent = v.name;
    const m = v.meta;
    $("video-meta").textContent =
      `${fmtDuration(m.duration)} · ${m.width}×${m.height} · ${m.fps.toFixed(1)} fps · ${(v.size / 1048576).toFixed(1)} MB`;
    $("opt-end").placeholder = m.duration.toFixed(1) + " (end)";
    $("opt-end").max = m.duration.toFixed(1);
    $("opt-start").max = m.duration.toFixed(1);
    loadHistory();
  } else {
    $("history").hidden = true;
  }
  updateEstimate();
}

// ------------------------------------------------------------------ targets
const targetsEl = $("targets");

function addTarget(label = "", description = "", focus = true) {
  const row = document.createElement("div");
  row.className = "target";
  row.innerHTML = `
    <div class="target-fields">
      <label class="field"><span>Label</span>
        <input type="text" class="t-label" maxlength="40" placeholder="e.g. person" autocomplete="off"></label>
      <label class="field desc-field"><span>Description <span class="muted">(optional)</span></span>
        <input type="text" class="t-desc" maxlength="160" placeholder="e.g. wearing a red jacket" autocomplete="off"></label>
      <button type="button" class="icon-btn t-remove" aria-label="Remove this label" title="Remove">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6 6 18"/></svg>
      </button>
    </div>
    <div class="target-prompt"><span>SAM 3 will look for:</span><code class="t-prompt">—</code></div>`;
  const [lab, desc] = [row.querySelector(".t-label"), row.querySelector(".t-desc")];
  lab.value = label;
  desc.value = description;
  const sync = () => {
    lab.classList.remove("invalid");
    row.querySelector(".t-prompt").textContent = lab.value.trim() ? buildPrompt(lab.value, desc.value) : "—";
    saveDraft();
    updateRunButton();
  };
  lab.addEventListener("input", sync);
  desc.addEventListener("input", sync);
  [lab, desc].forEach((inp) => inp.addEventListener("keydown", (e) => { if (e.key === "Enter") runJob(); }));
  row.querySelector(".t-remove").addEventListener("click", () => {
    row.remove();
    if (!targetsEl.children.length) addTarget("", "", false);
    refreshRemoveButtons();
    saveDraft();
    updateRunButton();
  });
  targetsEl.appendChild(row);
  sync();
  refreshRemoveButtons();
  if (focus) lab.focus();
}

function refreshRemoveButtons() {
  const rows = targetsEl.querySelectorAll(".target");
  rows.forEach((r) => { r.querySelector(".t-remove").disabled = rows.length === 1; });
  $("add-target").disabled = rows.length >= 10;
}

function readTargets() {
  return [...targetsEl.querySelectorAll(".target")]
    .map((r) => ({ label: r.querySelector(".t-label").value.trim(), description: r.querySelector(".t-desc").value.trim() }))
    .filter((t) => t.label);
}

function setTargets(list) {
  targetsEl.innerHTML = "";
  (list.length ? list : [{ label: "", description: "" }]).forEach((t) => addTarget(t.label, t.description || "", false));
}

$("add-target").addEventListener("click", () => addTarget());
document.querySelectorAll(".suggest .chip").forEach((chip) => chip.addEventListener("click", () => {
  const label = chip.dataset.label;
  const rows = [...targetsEl.querySelectorAll(".target")];
  if (rows.some((r) => r.querySelector(".t-label").value.trim().toLowerCase() === label)) return;
  const empty = rows.find((r) => !r.querySelector(".t-label").value.trim());
  if (empty) {
    const inp = empty.querySelector(".t-label");
    inp.value = label;
    inp.dispatchEvent(new Event("input"));
  } else if (rows.length < 10) {
    addTarget(label, "", false);
  }
}));

// ------------------------------------------------------------------ options + estimate
function readOptions() {
  const end = $("opt-end").value;
  return {
    start: parseFloat($("opt-start").value) || 0,
    end: end === "" ? null : parseFloat(end),
    max_frames: parseInt($("opt-frames").value, 10),
    score_thresh: parseFloat($("opt-score").value),
    boxes: $("opt-boxes").checked,
  };
}

function setOptions(o) {
  $("opt-start").value = o.start ?? 0;
  $("opt-end").value = o.end ?? "";
  $("opt-frames").value = o.max_frames ?? 150;
  $("opt-score").value = o.score_thresh ?? 0;
  $("opt-boxes").checked = o.boxes ?? true;
  syncOptionLabels();
}

function planFrames(meta, o) {
  const first = Math.max(0, Math.round(o.start * meta.fps));
  const last = o.end == null ? meta.frames : Math.min(meta.frames, Math.round(o.end * meta.fps));
  const span = Math.max(0, last - first);
  const stride = Math.max(1, Math.ceil(span / o.max_frames));
  return { n: Math.min(Math.ceil(span / stride), o.max_frames), stride, span };
}

function syncOptionLabels() {
  $("opt-frames-val").textContent = $("opt-frames").value;
  $("opt-score-val").textContent = parseFloat($("opt-score").value).toFixed(2);
}

function updateEstimate() {
  syncOptionLabels();
  const est = $("run-estimate");
  if (!state.video) { est.textContent = "Upload a video to begin."; updateRunButton(); return; }
  const o = readOptions();
  const { n, stride } = planFrames(state.video.meta, o);
  const spf = state.status?.sec_per_frame || 7.5;
  if (n <= 0) {
    est.textContent = "The selected time range has no frames.";
  } else {
    est.innerHTML = `<b>${n}</b> frames${stride > 1 ? ` (every ${ordinal(stride)} frame)` : ""} · about <b>${fmtDuration(n * spf + 30)}</b>`;
  }
  $("opt-frames-hint").textContent = stride > 1
    ? `Every ${ordinal(stride)} frame is used, so the output video plays at ${(state.video.meta.fps / stride).toFixed(1)} fps. Raise this for smoother output (slower).`
    : "Every frame in the range is used.";
  updateRunButton();
}

["opt-start", "opt-end", "opt-frames", "opt-score", "opt-boxes"].forEach((id) =>
  $(id).addEventListener("input", () => { updateEstimate(); saveDraft(); }));

function updateRunButton() {
  const ok = state.video && readTargets().length && state.status?.model !== "error";
  const running = state.currentJobState === "queued" || state.currentJobState === "running";
  $("run-btn").disabled = !ok || running;
  $("run-btn").textContent = running ? "Tracking in progress…" : "Start tracking";
}

// ------------------------------------------------------------------ drafts (survive a page refresh)
function saveDraft() {
  LS.set("sam3.draft", JSON.stringify({ targets: readTargets(), options: readOptions() }));
}

// ------------------------------------------------------------------ jobs
$("run-btn").addEventListener("click", runJob);

async function runJob() {
  hideFormError();
  if ($("run-btn").disabled) return;
  const targets = readTargets();
  if (!targets.length) {
    const first = targetsEl.querySelector(".t-label");
    first.classList.add("invalid");
    first.focus();
    showFormError("Add at least one label to track.");
    return;
  }
  $("run-btn").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", json: { video_id: state.video.video_id, targets, options: readOptions() } });
    showJob(job.id, job);
    loadHistory();
    if (window.matchMedia("(max-width: 960px)").matches) $("output-title").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    showFormError(e.message);
    updateRunButton();
  }
}

function showJob(jobId, job) {
  clearTimeout(state.pollTimer);
  state.jobId = jobId;
  state.previewVersion = -1;
  state.currentJobState = null;
  if (jobId) LS.set("sam3.job", jobId); else LS.del("sam3.job");
  $("live-preview").hidden = true;
  $("live-preview").removeAttribute("src");
  $("preview-placeholder").hidden = false;
  if (!jobId) { renderPanel("empty"); $("job-badge").hidden = true; updateRunButton(); return; }
  if (job) renderJob(job);
  poll();
}

async function poll() {
  const id = state.jobId;
  if (!id) return;
  try {
    const job = await api(`/api/jobs/${id}`);
    if (id !== state.jobId) return;
    renderJob(job);
    if (job.state === "queued" || job.state === "running") state.pollTimer = setTimeout(poll, 1200);
    else loadHistory();
  } catch (e) {
    if (id !== state.jobId) return;
    if (/not found/i.test(e.message)) { showJob(null); return; }
    state.pollTimer = setTimeout(poll, 3000);
  }
}

const STAGES = {
  queued: "Waiting in line",
  reading: "Reading video frames",
  preparing: "Finding objects in the first frames",
  tracking: "Tracking",
  encoding: "Saving video",
};

function renderJob(job) {
  state.currentJobState = job.state;
  updateRunButton();
  const badge = $("job-badge");
  badge.hidden = false;
  const badgeMap = {
    queued: ["pill-warn", "Queued"], running: ["pill-info", "Running"], done: ["pill-ok", "Done"],
    failed: ["pill-bad", "Failed"], cancelled: ["pill-muted", "Cancelled"],
  };
  setPill(badge, ...badgeMap[job.state]);

  if (job.state === "queued" || job.state === "running") {
    renderPanel("progress");
    const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
    const tracking = job.stage === "tracking" || job.stage === "encoding";
    $("stage-text").textContent = job.state === "queued"
      ? (job.position > 1 ? `Waiting in line · ${job.position - 1} run${job.position > 2 ? "s" : ""} ahead` : "Starting next…")
      : STAGES[job.stage] || "Working";
    $("eta-text").textContent = job.eta_seconds != null && job.state === "running" && tracking
      ? `about ${fmtDuration(job.eta_seconds)} left` : "";
    $("job-bar").classList.toggle("indeterminate", !tracking);
    $("job-bar").style.width = (job.stage === "reading" ? Math.round((job.done / Math.max(job.total, 1)) * 100) : pct) + "%";
    $("frames-text").textContent = tracking ? `Frame ${job.done} of ${job.total}` : `${job.total} frames to process`;
    $("pct-text").textContent = tracking ? pct + "%" : "";
    $("placeholder-text").textContent = job.state === "queued"
      ? "Another run is using the GPU. Yours starts automatically."
      : job.stage === "preparing" ? "SAM 3 is scanning the video. The first preview appears in a moment." : "Preparing…";
    renderCounts($("live-counts"), job.counts, job.targets);
    if (job.preview_version > 0 && job.preview_version !== state.previewVersion) {
      state.previewVersion = job.preview_version;
      const img = new Image();
      img.onload = () => {
        if (job.id !== state.jobId) return;
        $("live-preview").src = img.src;
        $("live-preview").hidden = false;
        $("preview-placeholder").hidden = true;
      };
      img.src = withKey(`/api/jobs/${job.id}/preview.jpg?v=${job.preview_version}`);
    }
    $("cancel-btn").disabled = false;
  } else if (job.state === "done") {
    renderPanel("done");
    const vurl = withKey(`/api/jobs/${job.id}/video`);
    const video = $("result-video");
    if (video.dataset.job !== job.id) {
      video.src = vurl;
      video.dataset.job = job.id;
    }
    $("dl-video").href = withKey(`/api/jobs/${job.id}/video?download=1`);
    $("dl-json").href = withKey(`/api/jobs/${job.id}/tracks.json`);
    const r = job.result || {};
    const stats = Object.entries(r.unique_tracks_per_class || {}).map(([k, v]) =>
      `<div class="stat"><div class="v">${v}</div><div class="k">${esc(k)} tracked</div></div>`);
    stats.push(`<div class="stat"><div class="v">${r.frames_processed ?? "–"}</div><div class="k">frames</div></div>`);
    stats.push(`<div class="stat"><div class="v">${r.seconds != null ? fmtDuration(r.seconds) : "–"}</div><div class="k">processing time</div></div>`);
    $("result-stats").innerHTML = stats.join("");
  } else {
    renderPanel("failed");
    $("fail-text").textContent = job.state === "cancelled" ? "This run was cancelled." : (job.error || "Something went wrong.");
  }
}

function renderPanel(which) {
  for (const p of ["empty", "progress", "done", "failed"]) $("out-" + p).hidden = p !== which;
  if (which !== "done") {
    const v = $("result-video");
    if (v.dataset.job) { v.pause(); v.removeAttribute("src"); v.load(); delete v.dataset.job; }
  }
}

function renderCounts(el, counts, targets) {
  const labels = [...new Set((targets || []).map((t) => t.label))];
  el.innerHTML = labels.map((l) => `<span class="count">${esc(l)} <b>${counts?.[l] ?? 0}</b> <span class="muted small">in frame</span></span>`).join("");
}

$("cancel-btn").addEventListener("click", async () => {
  if (!state.jobId) return;
  $("cancel-btn").disabled = true;
  try { renderJob(await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" })); } catch (e) { showFormError(e.message); }
});

function adjustAndRerun() {
  const id = state.jobId;
  const job = state.history?.find((j) => j.id === id);
  if (job) {
    setTargets(job.targets.map((t) => ({ label: t.label, description: t.description })));
    setOptions(job.options);
    saveDraft();
    updateEstimate();
  }
  $("step-targets").scrollIntoView({ behavior: "smooth", block: "start" });
  const first = targetsEl.querySelector(".t-desc") || targetsEl.querySelector(".t-label");
  setTimeout(() => first?.focus({ preventScroll: true }), 350);
}
$("retry-btn").addEventListener("click", adjustAndRerun);
$("fail-retry").addEventListener("click", adjustAndRerun);

// ------------------------------------------------------------------ history
async function loadHistory() {
  if (!state.video) return;
  try {
    const runs = await api(`/api/videos/${state.video.video_id}/jobs`);
    state.history = runs;
    $("history").hidden = runs.length < 1 || (runs.length === 1 && runs[0].id === state.jobId);
    const icon = { done: "pill-ok", failed: "pill-bad", cancelled: "pill-muted", queued: "pill-warn", running: "pill-info" };
    $("history-list").innerHTML = runs.map((j, i) => `
      <li><button type="button" data-id="${j.id}" class="${j.id === state.jobId ? "active" : ""}">
        <span class="h-prompts truncate"><b>Run ${runs.length - i}</b> · ${j.targets.map((t) => esc(t.prompt)).join(", ")}</span>
        <span class="pill ${icon[j.state]}">${j.state}</span>
      </button></li>`).join("");
    $("history-list").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      const job = runs.find((j) => j.id === b.dataset.id);
      showJob(job.id, job);
      loadHistory();
    }));
  } catch { /* history is optional */ }
}

// ------------------------------------------------------------------ misc
function showFormError(msg) { const el = $("form-error"); el.textContent = msg; el.hidden = false; }
function hideFormError() { $("form-error").hidden = true; }
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function ordinal(n) { const s = ["th", "st", "nd", "rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); }
function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return `${h}h ${m}m`;
  return s && m < 10 ? `${m}m ${s}s` : `${m} min`;
}

// ------------------------------------------------------------------ boot
async function restore() {
  let draft = null;
  try { draft = JSON.parse(LS.get("sam3.draft") || "null"); } catch { /* ignore */ }
  setTargets(draft?.targets || []);
  if (draft?.options) setOptions(draft.options);

  let video = null;
  try { video = JSON.parse(LS.get("sam3.video") || "null"); } catch { /* ignore */ }
  if (video) {
    // make sure the server still has it (uploads expire)
    try {
      setVideo(await api(`/api/videos/${video.video_id}`));
    } catch { setVideo(null); }
  } else {
    setVideo(null);
  }

  const jobId = LS.get("sam3.job");
  if (jobId && state.video) showJob(jobId); else showJob(null);
}

(async function init() {
  await refreshStatus();
  setInterval(refreshStatus, 5000);
  if (state.status?.auth_required && !state.key) {
    setTargets([]);
    askKey(false);
    return;
  }
  restore();
})();
