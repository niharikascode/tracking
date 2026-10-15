"use strict";

/* ===================================================================== helpers */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const LS = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};
// default label colours (same list as the server)
const PALETTE = ["#2f6fed", "#e5484d", "#30a46c", "#f5a524", "#8e4ec6", "#12a594", "#e93d82", "#0090ff", "#ff8b3e", "#6e56cf"];
function nextColor(targets) {
  const used = new Set(targets.map((t) => t.color));
  return PALETTE.find((c) => !used.has(c)) || PALETTE[targets.length % PALETTE.length];
}
const ICON = {
  check: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3.2"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  box: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="5" width="11" height="9" rx="1"/><rect x="10" y="10" width="10" height="9" rx="1" stroke-dasharray="3 2"/></svg>',
  mask: '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3c4.5 0 8 3 8 7.5 0 3-1.7 4.5-3.6 4.5-1.6 0-2 1.2-1.3 2.4.9 1.6-.3 3.6-3.1 3.6C7 21 4 17 4 12s3.5-9 8-9z" opacity=".85"/></svg>',
  upload: '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 16V4m0 0-4 4m4-4 4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>',
  dl: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 4v12m0 0 4-4m-4 4-4-4"/><path d="M4 18v1a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1"/></svg>',
};

function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return `${h}h ${m}m`;
  return s && m < 10 ? `${m}m ${s}s` : `${m} min`;
}
function fmtMoney(v) { return v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`; }
function fmtAgo(t) {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
}
function plural(n, word) { return `${n} ${word}${n === 1 ? "" : (word.endsWith("s") ? "es" : "s")}`; }

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

async function api(path, opts = {}) {
  const init = { ...opts, headers: { ...(opts.headers || {}) } };
  if (opts.json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, init);
  if (res.status === 401 && path !== "/api/login") {
    askKey();
    throw new Error("Access key required");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(errText(data) || `Request failed (${res.status})`);
  return data;
}
function errText(data) {
  if (!data?.detail) return "";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join("; ");
  return JSON.stringify(data.detail);
}

/* ===================================================================== dialogs */
function askKey(message) {
  const dlg = $("key-dialog");
  $("key-error").hidden = !message;
  $("key-error").textContent = message || "";
  if (!dlg.open) dlg.showModal();
  $("key-input").focus();
}
$("key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/login", { method: "POST", json: { key: $("key-input").value.trim() } });
    $("key-dialog").close();
    $("key-input").value = "";
    await refreshStatus();
    route();
  } catch (err) {
    askKey(err.message);
  }
});
$("key-dialog").addEventListener("cancel", (e) => e.preventDefault());

function promptText(title, value, okLabel = "Save") {
  return new Promise((resolve) => {
    const dlg = $("prompt-dialog");
    $("prompt-title").textContent = title;
    $("prompt-input").value = value || "";
    $("prompt-ok").textContent = okLabel;
    const done = (v) => { dlg.close(); cleanup(); resolve(v); };
    const onSubmit = (e) => { e.preventDefault(); const v = $("prompt-input").value.trim(); if (v) done(v); };
    const onCancel = () => done(null);
    const cleanup = () => {
      $("prompt-form").removeEventListener("submit", onSubmit);
      $("prompt-cancel").removeEventListener("click", onCancel);
      dlg.removeEventListener("cancel", onCancel);
    };
    $("prompt-form").addEventListener("submit", onSubmit);
    $("prompt-cancel").addEventListener("click", onCancel);
    dlg.addEventListener("cancel", onCancel);
    dlg.showModal();
    $("prompt-input").select();
  });
}

function confirmBox(title, text, okLabel = "Delete") {
  return new Promise((resolve) => {
    const dlg = $("confirm-dialog");
    $("confirm-title").textContent = title;
    $("confirm-text").textContent = text;
    $("confirm-ok").textContent = okLabel;
    const done = (v) => { dlg.close(); cleanup(); resolve(v); };
    const onSubmit = (e) => { e.preventDefault(); done(true); };
    const onCancel = () => done(false);
    const cleanup = () => {
      $("confirm-form").removeEventListener("submit", onSubmit);
      $("confirm-cancel").removeEventListener("click", onCancel);
      dlg.removeEventListener("cancel", onCancel);
    };
    $("confirm-form").addEventListener("submit", onSubmit);
    $("confirm-cancel").addEventListener("click", onCancel);
    dlg.addEventListener("cancel", onCancel);
    dlg.showModal();
  });
}

/* ===================================================================== status */
const state = {
  status: null,
  project: null,          // project detail with items
  selected: new Set(),
  filter: "all",
  pollTimer: null,
  viewer: null,           // {itemId, runId, mode}
  form: loadForm(),
};

function loadForm() {
  const defaults = { start: 0, end: null, max_frames: 150, detect_fps: 1, score_thresh: 0, boxes: true };
  try {
    const f = JSON.parse(LS.get("studio.form") || "null");
    if (f && Array.isArray(f.targets)) {
      f.options = { ...defaults, ...f.options };
      f.targets.forEach((t, i) => { if (!t.color) t.color = PALETTE[i % PALETTE.length]; });
      return f;
    }
  } catch { /* ignore */ }
  return { task: "segment", model: "qwen", targets: [{ label: "", description: "", color: PALETTE[0] }], options: defaults };
}
function saveForm() { LS.set("studio.form", JSON.stringify(state.form)); }

async function refreshStatus() {
  try {
    const s = await fetch("/api/status").then((r) => r.json());
    state.status = s;
    const pill = $("server-pill");
    pill.classList.add("dot");
    if (s.model === "loading") setPill(pill, "pill-warn", "GPU model loading…");
    else if (s.model === "error") { setPill(pill, "pill-bad", "GPU model failed"); pill.title = s.model_error; }
    else if (s.gpu_busy || s.gpu_queue) setPill(pill, "pill-info", s.gpu_queue ? `GPU busy · ${s.gpu_queue} waiting` : "GPU working");
    else setPill(pill, "pill-ok", "GPU ready");
    const b = s.budget;
    const bp = $("budget-pill");
    bp.hidden = !s.detection_enabled || !s.authorized;
    setPill(bp, b.spent_today >= b.daily_limit ? "pill-bad" : "pill-muted", `Detection credit today $${b.spent_today.toFixed(3)} / $${b.daily_limit.toFixed(2)}`);
    renderRunBox();
  } catch {
    setPill($("server-pill"), "pill-bad", "Server offline");
  }
}
function setPill(el, cls, text) { el.className = "pill dot " + cls; el.textContent = text; }

/* ===================================================================== routing */
window.addEventListener("hashchange", route);
function route() {
  clearTimeout(state.pollTimer);
  const m = location.hash.match(/^#\/p\/([0-9a-f]{32})(?:\/i\/([0-9a-f]{32}))?/);
  if (state.status?.auth_required && !state.status.authorized) { askKey(); return; }
  if (!m) { closeViewer(false); state.project = null; renderHome(); return; }
  const [, pid, iid] = m;
  if (!state.project || state.project.id !== pid) {
    state.selected.clear();
    loadProject(pid).then(() => { if (iid) openViewer(iid, false); });
  } else if (iid) {
    openViewer(iid, false);
  } else {
    closeViewer(false);
    schedulePoll();
  }
}

/* ===================================================================== home */
async function renderHome() {
  const app = $("app");
  app.innerHTML = `<div class="page-head"><h1>Projects</h1><span class="spacer"></span>
    <button class="btn btn-primary" id="new-project">+ New project</button></div>
    <div id="projects" class="projects"><div class="muted">Loading…</div></div>`;
  $("new-project").addEventListener("click", newProject);
  let projects = [];
  try { projects = await api("/api/projects"); } catch (e) { if (!/Access key/.test(e.message)) toast(e.message, "error"); return; }
  const box = $("projects");
  if (!projects.length) {
    app.querySelector(".page-head").insertAdjacentHTML("afterend", `
      <section class="card hero">
        <div><h2>Welcome! Start with a project.</h2>
          <p class="muted" style="margin-top:6px">A project is a folder of photos and videos you annotate together.</p></div>
        <div class="hero-steps">
          <div class="hero-step"><span class="step-num">1</span><span><b>Upload</b><span class="muted small">Add photos and videos — as many as you like.</span></span></div>
          <div class="hero-step"><span class="step-num">2</span><span><b>Choose a task</b><span class="muted small">Object detection (boxes) or segmentation (masks).</span></span></div>
          <div class="hero-step"><span class="step-num">3</span><span><b>Download</b><span class="muted small">Annotated media, JSON and a COCO export.</span></span></div>
        </div>
      </section>`);
  }
  box.innerHTML = `<button class="new-card" id="new-card"><span class="plus">+</span><strong>New project</strong><span class="small">Group photos and videos</span></button>` +
    projects.map((p) => {
      const n = Math.min(p.thumbs.length, 4);
      const mosaic = n ? `<div class="mosaic n${n}">${p.thumbs.map((iid) => `<img loading="lazy" alt="" src="/api/projects/${p.id}/items/${iid}/thumb.jpg">`).join("")}</div>`
        : `<div class="mosaic empty">No media yet</div>`;
      const parts = [p.images && plural(p.images, "photo"), p.videos && plural(p.videos, "video")].filter(Boolean).join(" · ") || "Empty";
      return `<a class="card project-card" href="#/p/${p.id}">${mosaic}
        <div class="project-meta"><strong class="truncate">${esc(p.name)}</strong>
        <span class="muted small">${parts} · updated ${fmtAgo(p.updated_at)}</span></div></a>`;
    }).join("");
  $("new-card").addEventListener("click", newProject);
}

async function newProject() {
  const name = await promptText("New project", "", "Create");
  if (!name) return;
  try {
    const p = await api("/api/projects", { method: "POST", json: { name } });
    location.hash = `#/p/${p.id}`;
  } catch (e) { toast(e.message, "error"); }
}

/* ===================================================================== project */
async function loadProject(pid) {
  try {
    state.project = await api(`/api/projects/${pid}`);
  } catch (e) {
    if (!/Access key/.test(e.message)) { toast(e.message, "error"); location.hash = "#/"; }
    return;
  }
  for (const id of [...state.selected]) if (!state.project.items.some((i) => i.id === id)) state.selected.delete(id);
  renderProject();
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  if (!state.project) return;
  const active = state.project.items.some((i) => i.latest && ["queued", "running"].includes(i.latest.state));
  state.pollTimer = setTimeout(async () => {
    if (!state.project) return;
    const before = new Map(state.project.items.map((i) => [i.id, i.latest?.state]));
    try { state.project = await api(`/api/projects/${state.project.id}`); } catch { schedulePoll(); return; }
    for (const it of state.project.items) {
      const prev = before.get(it.id), now = it.latest?.state;
      if (prev && prev !== now && ["queued", "running"].includes(prev)) {
        if (now === "done") toast(`${it.name}: ${it.latest.task === "detect" ? "detection" : "segmentation"} finished`, "ok");
        if (now === "failed") toast(`${it.name}: ${it.latest.error}`, "error");
      }
    }
    renderGrid();
    if (state.viewer) refreshViewer();
    schedulePoll();
  }, active ? 1500 : 8000);
}

function renderProject() {
  const p = state.project;
  $("app").innerHTML = `
    <div class="page-head">
      <a class="crumb" href="#/">← Projects</a>
      <button class="title-btn" id="rename" title="Rename project"><h1 class="truncate">${esc(p.name)}</h1></button>
      <span class="spacer"></span>
      <a class="btn btn-ghost btn-sm" id="export" href="/api/projects/${p.id}/export.zip" title="Annotated media, JSON per run and COCO annotations">${ICON.dl} Export all (ZIP)</a>
      <button class="btn btn-ghost btn-sm btn-danger" id="del-project">Delete project</button>
    </div>
    <div class="workspace">
      <section class="card media-panel" aria-label="Media">
        <label class="dropzone" id="drop" tabindex="0">
          <input type="file" id="files" accept="image/*,video/*" multiple hidden>
          ${ICON.upload}
          <span><strong>Add photos or videos</strong><span class="small">Drag &amp; drop or click · several at once · up to ${state.status?.limits.max_upload_mb ?? 1024} MB each</span></span>
        </label>
        <div id="uploads" class="uploads"></div>
        <div class="toolbar">
          <label class="check"><input type="checkbox" id="select-all"> <span id="sel-text">Select all</span></label>
          <span class="spacer"></span>
          <div class="seg" id="filter">
            <button data-f="all">All</button><button data-f="image">Photos</button><button data-f="video">Videos</button>
          </div>
        </div>
        <div id="grid" class="grid"></div>
      </section>
      <aside class="card annotate" id="annotate" aria-label="Annotate"></aside>
    </div>`;
  $("rename").addEventListener("click", async () => {
    const name = await promptText("Rename project", p.name);
    if (!name) return;
    try { await api(`/api/projects/${p.id}`, { method: "PATCH", json: { name } }); state.project.name = name; renderProject(); }
    catch (e) { toast(e.message, "error"); }
  });
  $("del-project").addEventListener("click", async () => {
    if (!await confirmBox("Delete this project?", `"${p.name}" and all its photos, videos and results will be deleted permanently.`)) return;
    try { await api(`/api/projects/${p.id}`, { method: "DELETE" }); state.project = null; location.hash = "#/"; toast("Project deleted"); }
    catch (e) { toast(e.message, "error"); }
  });
  wireUpload();
  $("select-all").addEventListener("change", (e) => {
    const vis = visibleItems();
    if (e.target.checked) vis.forEach((i) => state.selected.add(i.id)); else vis.forEach((i) => state.selected.delete(i.id));
    renderGrid();
  });
  $("filter").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.filter = b.dataset.f; renderGrid(); }));
  renderGrid();
  renderAnnotate();
}

function visibleItems() {
  const items = state.project?.items || [];
  return state.filter === "all" ? items : items.filter((i) => i.kind === state.filter);
}

function statusChip(it) {
  const r = it.latest;
  if (!r) return `<span class="pill pill-muted">Not annotated</span>`;
  const what = r.task === "detect" ? "Detected" : (r.kind === "video" ? "Tracked" : "Segmented");
  if (r.state === "done") {
    const n = Object.values(r.counts || {}).reduce((a, b) => a + b, 0);
    return `<span class="pill ${r.task === "detect" ? "pill-info" : "pill-violet"}">${what} · ${n}</span>`;
  }
  if (r.state === "queued") return `<span class="pill pill-warn">${r.position > 1 ? `Queued · #${r.position}` : "Queued"}</span>`;
  if (r.state === "running") return `<span class="pill pill-info dot">${r.total > 1 && r.done > 0 ? `${Math.round((r.done / r.total) * 100)}%` : r.total > 1 ? "Starting…" : "Working…"}</span>`;
  if (r.state === "failed") return `<span class="pill pill-bad" title="${esc(r.error)}">Failed</span>`;
  return `<span class="pill pill-muted">Cancelled</span>`;
}

function renderGrid() {
  const grid = $("grid");
  if (!grid) return;
  $("filter").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.f === state.filter));
  const items = visibleItems();
  const all = state.project.items.length;
  if (!all) {
    grid.innerHTML = `<div class="empty-grid" style="grid-column:1/-1"><strong>No media yet.</strong><br>Add photos or videos above to get started.</div>`;
  } else if (!items.length) {
    grid.innerHTML = `<div class="empty-grid" style="grid-column:1/-1">Nothing here with this filter.</div>`;
  } else {
    // update tiles in place (keyed by item id) so polling never swaps out an element mid-click
    grid.querySelector(".empty-grid")?.remove();
    const keep = new Set(items.map((i) => i.id));
    grid.querySelectorAll(".tile").forEach((t) => { if (!keep.has(t.dataset.id)) t.remove(); });
    items.forEach((it, idx) => {
      let t = grid.querySelector(`.tile[data-id="${it.id}"]`);
      if (!t) {
        t = document.createElement("div");
        t.className = "tile";
        t.dataset.id = it.id;
        t.innerHTML = `<button class="sel" type="button">${ICON.check}</button><span class="kind"></span>
          <img class="thumb" loading="lazy" alt=""><div class="mini-bar" hidden><span></span></div>
          <div class="info"><span class="name truncate"></span><div class="status"></div></div>`;
        t.addEventListener("click", (e) => {
          const id = t.dataset.id;
          if (e.target.closest(".sel") || e.shiftKey || e.metaKey || e.ctrlKey) {
            state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
            renderGrid();
          } else {
            location.hash = `#/p/${state.project.id}/i/${id}`;
          }
        });
      }
      if (grid.children[idx] !== t) grid.insertBefore(t, grid.children[idx] || null);
      const sel = state.selected.has(it.id);
      t.classList.toggle("selected", sel);
      const sb = t.querySelector(".sel");
      sb.setAttribute("aria-label", `${sel ? "Unselect" : "Select"} ${it.name}`);
      sb.setAttribute("aria-pressed", String(sel));
      t.querySelector(".kind").textContent = it.kind === "video" ? `▶ ${fmtDur(it.meta.duration)}` : "Photo";
      const thumb = it.latest_done_id ? `/api/projects/${state.project.id}/runs/${it.latest_done_id}/files/thumb.jpg`
        : `/api/projects/${state.project.id}/items/${it.id}/thumb.jpg`;
      const img = t.querySelector(".thumb");
      if (img.getAttribute("src") !== thumb) img.setAttribute("src", thumb);
      const name = t.querySelector(".name");
      name.textContent = it.name;
      name.title = it.name;
      const chip = statusChip(it);
      const st = t.querySelector(".status");
      if (st.innerHTML !== chip) st.innerHTML = chip;
      const r = it.latest;
      const running = r && r.state === "running" && r.total > 1;
      t.querySelector(".mini-bar").hidden = !running;
      if (running) t.querySelector(".mini-bar span").style.width = `${Math.round((r.done / r.total) * 100)}%`;
    });
  }
  const nSel = selectedItems().length;
  const sa = $("select-all");
  sa.checked = items.length > 0 && items.every((i) => state.selected.has(i.id));
  sa.indeterminate = !sa.checked && items.some((i) => state.selected.has(i.id));
  $("sel-text").textContent = nSel ? `${nSel} selected` : "Select all";
  renderRunBox();
}

/* ------------------------------------------------------------------ upload */
function wireUpload() {
  const drop = $("drop"), input = $("files");
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
  input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
}

async function uploadFiles(files) {
  if (!files.length) return;
  const max = (state.status?.limits.max_upload_mb || 1024) * 1048576;
  const ok = files.filter((f) => {
    if (f.size > max) { toast(`${f.name} is too large`, "error"); return false; }
    if (!/^(image|video)\//.test(f.type) && !/\.(mp4|mov|avi|mkv|webm|m4v|mpg|mpeg|jpe?g|png|webp|bmp|tiff?)$/i.test(f.name)) {
      toast(`${f.name} is not a photo or video`, "error"); return false;
    }
    return true;
  });
  const pid = state.project.id;
  // upload one file per request so each gets its own progress bar and a failure doesn't sink the rest
  for (const f of ok) {
    const row = document.createElement("div");
    row.className = "upload-row";
    row.innerHTML = `<div><div class="truncate">${esc(f.name)}</div><div class="bar"><div class="bar-fill"></div></div></div><span class="small muted">0%</span>`;
    $("uploads")?.appendChild(row);
    try {
      const res = await new Promise((resolve, reject) => {
        const form = new FormData();
        form.append("files", f);
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `/api/projects/${pid}/items`);
        xhr.upload.onprogress = (e) => {
          if (!e.lengthComputable) return;
          const pct = Math.round((e.loaded / e.total) * 100);
          row.querySelector(".bar-fill").style.width = pct + "%";
          row.querySelector("span").textContent = pct < 100 ? pct + "%" : "Processing…";
        };
        xhr.onload = () => {
          let data = {};
          try { data = JSON.parse(xhr.responseText); } catch { /* not json */ }
          if (xhr.status === 401) { askKey(); reject(new Error("Access key required")); }
          else if (xhr.status !== 200) reject(new Error(errText(data) || `Upload failed (${xhr.status})`));
          else resolve(data);
        };
        xhr.onerror = () => reject(new Error("Upload failed — check your connection"));
        xhr.send(form);
      });
      res.errors.forEach((m) => toast(m, "error"));
      if (state.project?.id === pid) {
        res.added.forEach((it) => { state.project.items.push(it); state.selected.add(it.id); });
        renderGrid();
      }
    } catch (e) {
      toast(`${f.name}: ${e.message}`, "error");
    } finally {
      row.remove();
    }
  }
}

/* ------------------------------------------------------------------ annotate panel */
const TRAILING = new Set(("with without wearing in on holding carrying that which who near at under behind next " +
  "beside inside riding pushing pulling sitting standing walking running having from of by over using").split(" "));
function buildPrompt(label, description) {
  label = label.trim();
  const desc = description.split(/\s+/).filter(Boolean).join(" ");
  if (!desc) return label;
  const e = label.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if (label && new RegExp(`\\b${e}s?\\b`).test(desc.toLowerCase())) return desc;
  if (TRAILING.has(desc.split(" ")[0].toLowerCase())) return `${label} ${desc}`;
  return `${desc} ${label}`;
}

function renderAnnotate() {
  const box = $("annotate");
  if (!box) return;
  const f = state.form;
  const s = state.status;
  const detOK = s?.detection_enabled;
  box.innerHTML = `
    <h2 style="margin-bottom:14px">Annotate</h2>
    <div class="a-step">
      <div class="a-head"><span class="step-num">1</span><h3>Choose a task</h3></div>
      <div class="tasks" role="radiogroup" aria-label="Task">
        <label class="task detect${f.task === "detect" ? " on" : ""}">
          <input type="radio" name="task" value="detect" ${f.task === "detect" ? "checked" : ""} ${detOK ? "" : "disabled"}>
          <span class="t-icon">${ICON.box}</span><strong>Object detection</strong>
          <span class="d">Boxes + labels from an AI model you pick. Uses OpenRouter credit.</span>
        </label>
        <label class="task segment${f.task === "segment" ? " on" : ""}">
          <input type="radio" name="task" value="segment" ${f.task === "segment" ? "checked" : ""}>
          <span class="t-icon">${ICON.mask}</span><strong>Segmentation</strong>
          <span class="d">Exact pixel masks with SAM 3 on the GPU. Videos are tracked. Free.</span>
        </label>
      </div>
      ${detOK ? "" : `<p class="muted small" style="margin-top:8px">Object detection isn't configured on this server.</p>`}
    </div>

    <div class="a-step">
      <div class="a-head"><span class="step-num">2</span><h3>What to find</h3></div>
      <p class="muted small" style="margin:-4px 0 10px">A short <b>label</b> per thing. Add a <b>description</b> to narrow it down (e.g. “wearing a red jacket”).</p>
      <div class="targets" id="targets"></div>
      <button type="button" class="btn btn-ghost btn-sm" id="add-target">+ Add label</button>
      <div class="chips"><span class="muted small">Quick add:</span>
        ${["person", "car", "bag", "dog", "bicycle", "bottle"].map((l) => `<button type="button" class="chip" data-l="${l}">${l}</button>`).join("")}
      </div>
    </div>

    <div class="a-step" id="model-step" ${f.task === "detect" ? "" : "hidden"}>
      <div class="a-head"><span class="step-num">3</span><h3>Detection model</h3></div>
      <div class="models" role="radiogroup" aria-label="Model">
        ${(s?.models || []).map((m) => `
          <label class="model${f.model === m.key ? " on" : ""}">
            <input type="radio" name="model" value="${m.key}" ${f.model === m.key ? "checked" : ""}>
            <span><span class="m-name">${esc(m.name)}</span><br><span class="m-note">${esc(m.note)}</span></span>
            <span class="m-price">≈ ${fmtMoney(m.est_per_image)}<br><span class="muted">per image</span></span>
          </label>`).join("")}
      </div>
    </div>

    <div class="a-step">
      <details class="settings" id="settings">
        <summary><span class="step-num">${f.task === "detect" ? 4 : 3}</span><h3>Settings <span class="muted small">(optional)</span></h3></summary>
        <div class="settings-grid">
          <label class="field"><span>Video start (s)</span><input type="number" id="o-start" min="0" step="0.1" value="${f.options.start ?? 0}"></label>
          <label class="field"><span>Video end (s)</span><input type="number" id="o-end" min="0" step="0.1" placeholder="end" value="${f.options.end ?? ""}"></label>
          <label class="field span-2" ${f.task === "segment" ? "" : "hidden"}>
            <span class="row-between"><span>Max video frames to track</span><b id="o-mf-v">${f.options.max_frames}</b></span>
            <input type="range" id="o-mf" min="20" max="${s?.limits.max_frames ?? 600}" step="10" value="${f.options.max_frames}">
            <span class="muted small">Longer videos use every Nth frame. More frames = smoother video, slower.</span>
          </label>
          <label class="field span-2" ${f.task === "detect" ? "" : "hidden"}>
            <span class="row-between"><span>Video: model checks per second</span><b id="o-fr-v">${fpsLabel(f.options.detect_fps)}</b></span>
            <input type="range" id="o-fr" min="0" max="${FPS_STEPS.length - 1}" step="1" value="${fpsIndex(f.options.detect_fps)}">
            <span class="muted small">The output is the full video; boxes update at each check. Each check is one paid model call
              (max ${s?.limits.detect_max_frames ?? 120} per video).</span>
          </label>
          <label class="field span-2" ${f.task === "segment" ? "" : "hidden"}>
            <span class="row-between"><span>Hide masks below confidence</span><b id="o-sc-v">${(+f.options.score_thresh).toFixed(2)}</b></span>
            <input type="range" id="o-sc" min="0" max="0.95" step="0.05" value="${f.options.score_thresh}">
          </label>
          <label class="check span-2" ${f.task === "segment" ? "" : "hidden"}><input type="checkbox" id="o-bx" ${f.options.boxes ? "checked" : ""}> Draw boxes around masks</label>
        </div>
      </details>
    </div>

    <div class="run-box" id="run-box"></div>`;

  box.querySelectorAll('input[name="task"]').forEach((r) => r.addEventListener("change", () => { f.task = r.value; saveForm(); renderAnnotate(); }));
  box.querySelectorAll('input[name="model"]').forEach((r) => r.addEventListener("change", () => {
    f.model = r.value; saveForm();
    box.querySelectorAll(".model").forEach((m) => m.classList.toggle("on", m.querySelector("input").checked));
    renderRunBox();
  }));
  const bindNum = (id, key, parse, label) => $(id)?.addEventListener("input", (e) => {
    f.options[key] = parse(e.target.value);
    if (label) $(label).textContent = key === "score_thresh" ? (+e.target.value).toFixed(2) : e.target.value;
    saveForm(); renderRunBox();
  });
  bindNum("o-start", "start", (v) => Math.max(0, parseFloat(v) || 0));
  bindNum("o-end", "end", (v) => (v === "" ? null : parseFloat(v)));
  bindNum("o-mf", "max_frames", (v) => parseInt(v, 10), "o-mf-v");
  $("o-fr")?.addEventListener("input", (e) => {
    f.options.detect_fps = FPS_STEPS[+e.target.value];
    $("o-fr-v").textContent = fpsLabel(f.options.detect_fps);
    saveForm(); renderRunBox();
  });
  bindNum("o-sc", "score_thresh", (v) => parseFloat(v), "o-sc-v");
  $("o-bx")?.addEventListener("change", (e) => { f.options.boxes = e.target.checked; saveForm(); });
  $("add-target").addEventListener("click", () => { f.targets.push({ label: "", description: "", color: nextColor(f.targets) }); saveForm(); renderTargets(true); });
  box.querySelectorAll(".chips .chip").forEach((c) => c.addEventListener("click", () => {
    const l = c.dataset.l;
    if (f.targets.some((t) => t.label.trim().toLowerCase() === l)) return;
    const empty = f.targets.find((t) => !t.label.trim());
    if (empty) empty.label = l; else if (f.targets.length < 10) f.targets.push({ label: l, description: "", color: nextColor(f.targets) });
    saveForm(); renderTargets();
  }));
  renderTargets();
  renderRunBox();
}

function renderTargets(focusLast) {
  const f = state.form, el = $("targets");
  if (!el) return;
  if (!f.targets.length) f.targets.push({ label: "", description: "", color: PALETTE[0] });
  el.innerHTML = f.targets.map((t, i) => `
    <div class="target" data-i="${i}">
      <div class="target-fields">
        <span class="label-row">
          <label class="swatch" title="Colour for this label" style="--c:${esc(t.color || PALETTE[i % PALETTE.length])}">
            <input type="color" class="t-color" value="${esc(t.color || PALETTE[i % PALETTE.length])}" aria-label="Colour for label ${i + 1}">
          </label>
          <input type="text" class="t-label" maxlength="40" placeholder="Label, e.g. person" value="${esc(t.label)}" aria-label="Label ${i + 1}">
        </span>
        <button type="button" class="icon-btn danger t-rm" aria-label="Remove label" ${f.targets.length === 1 ? "disabled" : ""}>✕</button>
        <input type="text" class="t-desc desc" maxlength="160" placeholder="Description (optional), e.g. wearing a red jacket" value="${esc(t.description)}" aria-label="Description ${i + 1}">
      </div>
      <div class="target-prompt"><span>Looks for:</span><code>${t.label.trim() ? esc(buildPrompt(t.label, t.description)) : "—"}</code></div>
    </div>`).join("");
  el.querySelectorAll(".target").forEach((row) => {
    const i = +row.dataset.i;
    const sync = () => {
      const color = row.querySelector(".t-color").value;
      f.targets[i] = { label: row.querySelector(".t-label").value, description: row.querySelector(".t-desc").value, color };
      row.querySelector(".swatch").style.setProperty("--c", color);
      row.querySelector(".t-label").classList.remove("invalid");
      row.querySelector("code").textContent = f.targets[i].label.trim() ? buildPrompt(f.targets[i].label, f.targets[i].description) : "—";
      saveForm(); renderRunBox();
    };
    row.querySelectorAll("input").forEach((inp) => {
      inp.addEventListener("input", sync);
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") startRuns(); });
    });
    row.querySelector(".t-rm").addEventListener("click", () => { f.targets.splice(i, 1); saveForm(); renderTargets(); });
  });
  $("add-target").disabled = f.targets.length >= 10;
  if (focusLast) el.querySelector(".target:last-child .t-label")?.focus();
  renderRunBox();
}

function selectedItems() { return visibleItems().filter((i) => state.selected.has(i.id)); }

function planFrames(meta, o) {
  const first = Math.max(0, Math.round((o.start || 0) * meta.fps));
  const last = o.end == null ? meta.frames : Math.min(meta.frames, Math.round(o.end * meta.fps));
  const span = Math.max(0, last - first);
  const stride = Math.max(1, Math.ceil(span / o.max_frames));
  return Math.min(Math.ceil(span / stride), o.max_frames);
}

const FPS_STEPS = [0.25, 0.5, 1, 2, 3, 5];
function fpsIndex(v) { const i = FPS_STEPS.indexOf(+v); return i < 0 ? 2 : i; }
function fpsLabel(v) { return +v >= 1 ? `${+v} per second` : `1 every ${Math.round(1 / v)} s`; }
function detectCalls(meta, o, maxCalls) {
  const first = Math.max(0, Math.round((o.start || 0) * meta.fps));
  const last = o.end == null ? meta.frames : Math.min(meta.frames, Math.round(o.end * meta.fps));
  const span = Math.max(0, last - first);
  return span ? Math.max(1, Math.min(maxCalls, Math.ceil((span / meta.fps) * (o.detect_fps || 1)))) : 0;
}

function renderRunBox() {
  const box = $("run-box");
  if (!box || !state.project) return;
  const f = state.form, s = state.status;
  const items = selectedItems();
  const targets = f.targets.filter((t) => t.label.trim());
  const imgs = items.filter((i) => i.kind === "image").length, vids = items.length - imgs;
  let estimate = "", disabled = !items.length || !targets.length;
  let budgetHtml = "";
  if (f.task === "detect") {
    const m = s?.models.find((x) => x.key === f.model);
    const maxCalls = s?.limits.detect_max_frames ?? 120;
    const calls = items.reduce((a, i) => a + (i.kind === "image" ? 1 : detectCalls(i.meta, f.options, maxCalls)), 0);
    const cost = (m?.est_per_image || 0) * calls;
    const left = s?.budget_remaining ?? 0;
    estimate = items.length ? `${plural(calls, "model call")} · about <b>${fmtMoney(cost)}</b> with ${esc(m?.name || "")}` : "";
    if (cost > left) disabled = true;
    const b = s?.budget;
    if (b) {
      const pct = Math.min(100, (b.spent_today / b.daily_limit) * 100);
      budgetHtml = `<div class="budget"><div class="row-between small"><span class="muted">Detection credit today</span><b>$${b.spent_today.toFixed(3)} of $${b.daily_limit.toFixed(2)}</b></div>
        <div class="meter ${pct > 90 ? "bad" : pct > 70 ? "warn" : ""}"><span style="width:${pct}%"></span></div>
        ${cost > left && items.length ? `<span class="small" style="color:var(--danger)">Not enough credit left today for this — pick a cheaper model or fewer items.</span>` : ""}</div>`;
    }
  } else if (items.length) {
    const spf = s?.sec_per_frame || 7.5;
    const frames = items.reduce((a, i) => a + (i.kind === "image" ? 1 : planFrames(i.meta, { ...f.options, max_frames: f.options.max_frames })), 0);
    estimate = `${plural(frames, "frame")} on the GPU · about <b>${fmtDur(frames * spf + 20 * vids + 3 * imgs)}</b>${s?.gpu_queue ? ` · ${s.gpu_queue} queued ahead` : ""}`;
  }
  const what = f.task === "detect" ? "Detect objects" : "Segment";
  const label = items.length ? `${what} in ${plural(items.length, "item")}` : `${what}`;
  const hint = !items.length ? "Select photos or videos first (click the ✓ on a tile)." : !targets.length ? "Add at least one label above." : "";
  // keep the button element stable so a click is never lost to a re-render
  if (!$("run-btn")) {
    box.innerHTML = `<div id="rb-info"></div><button class="btn btn-primary btn-lg" id="run-btn"></button>`;
    $("run-btn").addEventListener("click", startRuns);
  }
  $("rb-info").innerHTML = `${budgetHtml}${estimate ? `<div class="small">${estimate}</div>` : ""}${hint ? `<div class="small muted">${hint}</div>` : ""}`;
  $("rb-info").style.display = budgetHtml || estimate || hint ? "flex" : "none";
  $("rb-info").style.flexDirection = "column";
  $("rb-info").style.gap = "8px";
  const btn = $("run-btn");
  if (!btn.dataset.busy) btn.disabled = disabled;
  btn.textContent = label;
}

async function startRuns() {
  const f = state.form;
  const items = selectedItems();
  const targets = f.targets.filter((t) => t.label.trim()).map((t) => ({ label: t.label.trim(), description: t.description.trim(), color: t.color }));
  if (!targets.length) {
    document.querySelector(".t-label")?.classList.add("invalid");
    document.querySelector(".t-label")?.focus();
    toast("Add at least one label", "error");
    return;
  }
  if (!items.length) { toast("Select at least one photo or video", "error"); return; }
  const btn = $("run-btn");
  if (btn) { btn.disabled = true; btn.dataset.busy = "1"; }
  try {
    const runs = await api(`/api/projects/${state.project.id}/runs`, { method: "POST", json: {
      item_ids: items.map((i) => i.id), task: f.task, model: f.task === "detect" ? f.model : null, targets, options: f.options,
    } });
    toast(`Started ${plural(runs.length, "run")}`);
    for (const r of runs) {
      const it = state.project.items.find((i) => i.id === r.item_id);
      if (it) { it.latest = r; it.runs += 1; }
    }
    renderGrid();
    schedulePoll();
    refreshStatus();
  } catch (e) {
    toast(e.message, "error");
  } finally {
    if (btn) delete btn.dataset.busy;
    renderRunBox();
  }
}

/* ===================================================================== viewer */
function openViewer(itemId, push = true) {
  if (!state.project) return;
  const it = state.project.items.find((i) => i.id === itemId);
  if (!it) { location.hash = `#/p/${state.project.id}`; return; }
  if (push) { location.hash = `#/p/${state.project.id}/i/${itemId}`; return; }
  state.viewer = { itemId, runId: it.latest?.id || null, mode: "result", runs: [], shownKey: "" };
  $("viewer").hidden = false;
  document.body.style.overflow = "hidden";
  refreshViewer(true);
  schedulePoll();
}

function closeViewer(push = true) {
  if (!state.viewer) return;
  const v = $("viewer-stage").querySelector("video");
  if (v) v.pause();
  state.viewer = null;
  $("viewer").hidden = true;
  document.body.style.overflow = "";
  if (push && state.project) location.hash = `#/p/${state.project.id}`;
}

$("viewer-close").addEventListener("click", () => closeViewer());
$("viewer").addEventListener("click", (e) => { if (e.target.id === "viewer") closeViewer(); });
$("viewer-prev").addEventListener("click", () => stepViewer(-1));
$("viewer-next").addEventListener("click", () => stepViewer(1));
$("viewer-mode").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
  if (!state.viewer) return;
  state.viewer.mode = b.dataset.mode;
  state.viewer.shownKey = "";
  refreshViewer();
}));
document.addEventListener("keydown", (e) => {
  if (!state.viewer || document.querySelector("dialog[open]")) return;
  if (e.target.matches("input, textarea")) return;
  if (e.key === "Escape") closeViewer();
  if (e.key === "ArrowLeft") stepViewer(-1);
  if (e.key === "ArrowRight") stepViewer(1);
});
function stepViewer(d) {
  const items = visibleItems();
  const i = items.findIndex((x) => x.id === state.viewer?.itemId);
  const next = items[i + d];
  if (next) location.hash = `#/p/${state.project.id}/i/${next.id}`;
}

async function refreshViewer(first) {
  const v = state.viewer;
  if (!v || !state.project) return;
  const pid = state.project.id;
  const it = state.project.items.find((i) => i.id === v.itemId);
  if (!it) { closeViewer(); return; }
  try { v.runs = await api(`/api/projects/${pid}/runs?item=${it.id}`); } catch { return; }
  if (state.viewer !== v) return;
  if (first || !v.runs.some((r) => r.id === v.runId)) v.runId = (v.runs.find((r) => r.state !== "cancelled") || v.runs[0])?.id || null;
  const run = v.runs.find((r) => r.id === v.runId);
  const items = visibleItems();
  const idx = items.findIndex((x) => x.id === it.id);
  $("viewer-title").textContent = it.name;
  $("viewer-prev").disabled = idx <= 0;
  $("viewer-next").disabled = idx < 0 || idx >= items.length - 1;
  const showResult = v.mode === "result" && run;
  $("viewer-mode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.mode === (showResult ? "result" : "original")));
  renderStage(it, run, showResult);
  renderViewerSide(it, run);
}

function renderStage(it, run, showResult) {
  const pid = state.project.id, stage = $("viewer-stage"), v = state.viewer;
  let key, html;
  if (!showResult) {
    key = `orig:${it.id}`;
    html = it.kind === "video"
      ? `<video controls playsinline preload="metadata" src="/api/projects/${pid}/items/${it.id}/source"></video>`
      : `<img alt="${esc(it.name)}" src="/api/projects/${pid}/items/${it.id}/source">`;
  } else if (run.state === "done") {
    key = `done:${run.id}`;
    html = it.kind === "video"
      ? `<video controls playsinline preload="metadata" src="/api/projects/${pid}/runs/${run.id}/files/result.mp4"></video>`
      : `<img alt="Annotated ${esc(it.name)}" src="/api/projects/${pid}/runs/${run.id}/files/result.jpg">`;
  } else if (run.state === "queued" || run.state === "running") {
    key = `live:${run.id}:${run.preview_version}:${run.done}:${run.stage}`;
    const img = run.preview_version > 0
      ? `<img alt="Live preview" src="/api/projects/${pid}/runs/${run.id}/preview.jpg?v=${run.preview_version}">`
      : `<div class="stage-msg"><div class="spinner"></div><div>${run.state === "queued" ? (run.position > 1 ? `Waiting for the GPU · ${run.position - 1} ahead` : "Starting…") : stageText(run)}</div></div>`;
    const pct = run.total ? Math.round((run.done / run.total) * 100) : 0;
    const indet = run.state === "queued" || run.done === 0;
    html = `${img}<div class="overlay-progress"><div class="row-between"><span>${run.state === "queued" ? "Queued" : stageText(run)}</span>
      <span>${run.total > 1 ? `${run.done} / ${run.total}` : ""}${run.eta_seconds != null ? ` · ~${fmtDur(run.eta_seconds)} left` : ""}</span></div>
      <div class="bar"><div class="bar-fill${indet ? " indeterminate" : ""}" style="width:${pct}%"></div></div></div>`;
  } else {
    key = `fail:${run.id}`;
    html = `<div class="stage-msg"><div style="font-size:30px">⚠</div><div>${run.state === "cancelled" ? "This run was cancelled." : esc(run.error || "This run failed.")}</div></div>`;
  }
  if (v.shownKey === key) return;
  // keep the playing video if only the key's live part changed
  v.shownKey = key;
  stage.innerHTML = html;
}

function stageText(run) {
  return { starting: "Starting", reading: "Reading frames", preparing: "Finding objects in the first frames",
           tracking: "Tracking", encoding: "Saving video", detecting: run.total > 1 ? "Checking frames with the model" : "Asking the model",
           rendering: "Drawing boxes on the full video",
           segmenting: "Segmenting" }[run.stage] || "Working";
}

function taskName(r) { return r.task === "detect" ? "Detection" : r.kind === "video" ? "Segmentation + tracking" : "Segmentation"; }
function modelName(r) {
  if (r.task !== "detect") return "SAM 3";
  return state.status?.models.find((m) => m.key === r.model)?.name || r.model;
}

function renderViewerSide(it, run) {
  const pid = state.project.id;
  const side = $("viewer-run");
  // only rebuild when something shown here changed, so buttons never vanish under the cursor
  const sig = JSON.stringify([it.id, run && [run.id, run.state, run.counts, run.cost, run.error],
    state.viewer.runId, state.viewer.runs.map((r) => [r.id, r.state, r.cost])]);
  if (state.viewer.sideKey === sig) return;
  state.viewer.sideKey = sig;
  if (!run) {
    side.innerHTML = `<div class="run-card"><strong>Not annotated yet</strong>
      <p class="muted small">Close this view, keep this item selected, choose a task on the right and press the button.</p>
      <button class="btn btn-secondary btn-sm" id="v-select">Select this item</button></div>`;
    $("v-select").addEventListener("click", () => { state.selected.add(it.id); closeViewer(); renderGrid(); });
  } else {
    const counts = Object.entries(run.counts || {});
    const files = run.state === "done" ? [
      it.kind === "video" ? ["result.mp4", "Video"] : ["result.jpg", "Image"],
      ["result.json", "JSON"],
      ...(run.task === "segment" && it.kind === "image" ? [["mask.png", "Mask PNG"]] : []),
    ] : [];
    const r = run.result || {};
    side.innerHTML = `<div class="run-card">
      <div class="row-between"><strong>${taskName(run)}</strong><span class="pill ${run.task === "detect" ? "pill-info" : "pill-violet"}">${esc(modelName(run))}</span></div>
      <div class="labels">${run.targets.map((t) => `<code title="label: ${esc(t.label)}"><span class="dot-c" style="background:${esc(t.color || "#999")}"></span>${esc(t.prompt)}</code>`).join("")}</div>
      ${counts.length ? `<div class="stats">${counts.map(([k, n]) => `<div class="stat"><div class="v">${n}</div><div class="k">${esc(k)}</div></div>`).join("")}</div>
        <div class="small muted">${it.kind === "image" ? "Objects found in the photo." : run.task === "segment" ? "Different objects tracked over the video." : "Most found in a single check."}</div>` : ""}
      <div class="small muted">${[
        run.state === "done" && run.task === "detect" ? `cost ${fmtMoney(run.cost || 0)}` : "",
        r.frames ? `${r.frames} frames` : "",
        run.finished_at && run.started_at ? `took ${fmtDur(run.finished_at - run.started_at)}` : "",
        fmtAgo(run.created_at),
      ].filter(Boolean).join(" · ")}</div>
      ${run.task === "detect" && it.kind === "video" && run.state === "done" ? `<div class="small muted">Full ${fmtDur(r.duration || it.meta.duration)} video. The model checked ${r.frames} frames (${fpsLabel(run.options.detect_fps)}); in between, boxes move smoothly from one check to the next${r.failed_frames ? ` (${r.failed_frames} checks failed or looked wrong and were skipped)` : ""}. For fast motion use more checks per second.</div>` : ""}
      ${files.length ? `<div class="dl">${files.map(([n, l], i) => `<a class="btn ${i === 0 ? "btn-primary" : "btn-ghost"} btn-sm" href="/api/projects/${pid}/runs/${run.id}/files/${n}?download=1">${ICON.dl} ${l}</a>`).join("")}</div>` : ""}
      ${run.state === "failed" ? `<div class="alert alert-error">${esc(run.error)}</div>` : ""}
      <div class="dl">
        ${["queued", "running"].includes(run.state) ? `<button class="btn btn-ghost btn-sm btn-danger" id="v-cancel">Cancel run</button>` : `
          <button class="btn btn-secondary btn-sm" id="v-reuse" title="Load these labels and settings into the Annotate panel">Adjust &amp; re-run</button>
          <button class="btn btn-ghost btn-sm btn-danger" id="v-del">Delete run</button>`}
      </div></div>`;
    $("v-cancel")?.addEventListener("click", async (e) => {
      e.target.disabled = true;
      e.target.textContent = "Cancelling…";
      try {
        await api(`/api/projects/${pid}/runs/${run.id}/cancel`, { method: "POST" });
        toast("Cancelling — the run stops at its next step (can take a few seconds)");
      } catch (err) { toast(err.message, "error"); e.target.disabled = false; e.target.textContent = "Cancel run"; }
    });
    $("v-del")?.addEventListener("click", async () => {
      if (!await confirmBox("Delete this run?", "Its annotated media and JSON will be removed. The original stays.")) return;
      try {
        await api(`/api/projects/${pid}/runs/${run.id}`, { method: "DELETE" });
        state.project = await api(`/api/projects/${pid}`);
        renderGrid();
        state.viewer.runId = null;
        refreshViewer(true);
      } catch (e) { toast(e.message, "error"); }
    });
    $("v-reuse")?.addEventListener("click", () => {
      const f = state.form;
      f.task = run.task;
      if (run.task === "detect") f.model = run.model;
      f.targets = run.targets.map((t, i) => ({ label: t.label, description: t.description, color: t.color || PALETTE[i % PALETTE.length] }));
      f.options = { ...f.options, ...run.options };
      saveForm();
      state.selected = new Set([it.id]);
      closeViewer();
      renderGrid();
      renderAnnotate();
      $("annotate").scrollIntoView({ behavior: "smooth", block: "start" });
      setTimeout(() => document.querySelector(".t-desc")?.focus({ preventScroll: true }), 300);
      toast("Settings loaded — change labels or descriptions, then run again");
    });
  }
  const runs = state.viewer.runs;
  $("viewer-runs").innerHTML = runs.length ? runs.map((r) => `<li><button type="button" data-id="${r.id}" class="${r.id === state.viewer.runId ? "active" : ""}">
      <span class="r-top"><b class="small">${taskName(r)}</b><span class="spacer"></span><span class="pill ${{ done: "pill-ok", failed: "pill-bad", cancelled: "pill-muted", queued: "pill-warn", running: "pill-info" }[r.state]}">${r.state}</span></span>
      <span class="r-sub truncate">${esc(modelName(r))} · ${r.targets.map((t) => esc(t.prompt)).join(", ")}</span>
      <span class="r-sub">${fmtAgo(r.created_at)}${r.cost ? ` · ${fmtMoney(r.cost)}` : ""}</span>
    </button></li>`).join("") : `<li class="muted small">No runs yet.</li>`;
  $("viewer-runs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    state.viewer.runId = b.dataset.id;
    state.viewer.mode = "result";
    state.viewer.shownKey = "";
    refreshViewer();
  }));
}

/* ===================================================================== boot */
(async function init() {
  await refreshStatus();
  setInterval(refreshStatus, 6000);
  if (state.status?.auth_required && !state.status.authorized) { askKey(); return; }
  route();
})();
