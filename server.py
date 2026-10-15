"""SAM 3 Studio: project-based annotation web app.

Projects hold photos and videos. Each item can be annotated with
  - object detection: boxes from a vision-language model via OpenRouter (Qwen, DeepSeek,
    Gemini, GPT-6 Sol, GPT-6 Astra), with daily/total spending caps
  - segmentation: SAM 3 masks on the local GPU (videos are tracked frame by frame)

Segmentation runs one at a time on the GPU worker; detection runs on a small thread
pool. Access is protected by one shared access key (APP_PASSWORD).

Run:  .venv/bin/uvicorn server:app --host 127.0.0.1 --port 7860
"""

import hashlib
import hmac
import io
import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

import detector
import pipelines
from pipelines import Budget, BudgetExceeded
from tracker import Cancelled, Options, Sam3Tracker, build_prompt, plan_frames, video_meta

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))
MAX_FRAMES_LIMIT = int(os.getenv("MAX_FRAMES_LIMIT", "600"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "100"))
DTYPE = os.getenv("DTYPE", "fp16")
DETECT_DAILY_USD = env_float("DETECT_DAILY_USD", "2.00")
DETECT_TOTAL_USD = env_float("DETECT_TOTAL_USD", "20.00")
DETECT_MAX_FRAMES = int(os.getenv("DETECT_MAX_FRAMES", "120"))   # max model calls per video
DETECT_WORKERS = int(os.getenv("DETECT_WORKERS", "3"))
MAX_IMAGE_SIDE = 4096

PROJECTS = DATA_DIR / "projects"
PROJECTS.mkdir(parents=True, exist_ok=True)

ID_RE = re.compile(r"^[0-9a-f]{32}$")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
COOKIE = "sam3_key"

app = FastAPI(title="SAM 3 Studio")
budget = Budget(DATA_DIR / "usage.json", DETECT_DAILY_USD, DETECT_TOTAL_USD)


# ============================================================================ state

class State:
    def __init__(self):
        self.model_status = "loading"
        self.model_error = ""
        self.projects: dict[str, dict] = {}
        self.items: dict[str, dict] = {}       # item id -> item (has project_id)
        self.runs: dict[str, dict] = {}        # run id -> run (internals prefixed "_")
        self.gpu_queue: "queue.Queue[str]" = queue.Queue()
        self.gpu_order: list[str] = []
        self.gpu_running: str | None = None
        self.sec_per_frame = 3.0
        self.lock = threading.RLock()


S = State()
detect_pool = ThreadPoolExecutor(max_workers=DETECT_WORKERS, thread_name_prefix="detect")


def now():
    return time.time()


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def pdir(pid):
    return PROJECTS / pid


def idir(item):
    return PROJECTS / item["project_id"] / "items" / item["id"]


def rdir(run):
    return PROJECTS / run["project_id"] / "runs" / run["id"]


def save_project(p):
    write_json(pdir(p["id"]) / "project.json", p)


def save_item(it):
    write_json(idir(it) / "item.json", it)


def save_run(run):
    write_json(rdir(run) / "run.json", {k: v for k, v in run.items() if not k.startswith("_")})


def load_all():
    for pj in PROJECTS.glob("*/project.json"):
        try:
            p = json.loads(pj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        S.projects[p["id"]] = p
        for ij in (pj.parent / "items").glob("*/item.json"):
            try:
                it = json.loads(ij.read_text())
                S.items[it["id"]] = it
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        for rj in (pj.parent / "runs").glob("*/run.json"):
            try:
                run = json.loads(rj.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if run.get("state") in ("queued", "running"):
                run.update(state="failed", error="The server restarted before this run finished. Run it again.")
                write_json(rj, run)
            S.runs[run["id"]] = run


def touch_project(pid):
    p = S.projects.get(pid)
    if p:
        p["updated_at"] = now()
        save_project(p)


# ============================================================================ run execution

def public_run(run: dict) -> dict:
    out = {k: v for k, v in run.items() if not k.startswith("_")}
    if run["state"] == "queued" and run["task"] == "segment":
        with S.lock:
            out["position"] = S.gpu_order.index(run["id"]) + 1 if run["id"] in S.gpu_order else 0
    if run["state"] == "running" and run["task"] == "segment" and run["kind"] == "video" and run["stage"] == "tracking":
        out["eta_seconds"] = round(max(0, run["total"] - run["done"]) * S.sec_per_frame)
    return out


def make_progress(run):
    t0 = [None]

    def on_progress(p):
        run["stage"], run["done"], run["total"] = p.stage, p.done, p.total
        if p.stage == "tracking":
            if t0[0] is None:
                t0[0] = time.time()
            elif p.done > 1:
                spf = (time.time() - t0[0]) / (p.done - 1)
                S.sec_per_frame = 0.7 * S.sec_per_frame + 0.3 * spf
        if p.counts:
            run["counts"] = p.counts
        if p.preview is not None:
            img = p.preview
            if img.shape[1] > 960:
                img = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                run["_preview"] = buf.tobytes()
                run["preview_version"] = run.get("preview_version", 0) + 1
    return on_progress


def execute(run: dict, fn):
    """Run one pipeline function with uniform state handling."""
    run.update(state="running", stage="starting", started_at=now())
    out = rdir(run)
    out.mkdir(parents=True, exist_ok=True)
    try:
        summary = fn(out, make_progress(run), lambda: run.get("_cancel", False))
        run.update(state="done", stage="done", result=summary)
        if "counts" in summary:
            run["counts"] = summary["counts"]
    except Cancelled:
        run.update(state="cancelled", stage="cancelled")
    except (BudgetExceeded, detector.DetectionError) as e:
        run.update(state="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - surface every failure to the UI
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}"
        if "out of memory" in msg.lower():
            msg = ("The GPU ran out of memory (it is shared with other jobs on this server). "
                   "Try again, or use fewer frames / a shorter range.")
        run.update(state="failed", error=msg)
    finally:
        run["finished_at"] = now()
        if run["state"] == "done":
            run.pop("_preview", None)
        save_run(run)
        touch_project(run["project_id"])


def gpu_worker():
    try:
        tracker = Sam3Tracker(token=HF_TOKEN, dtype=DTYPE)
        S.model_status = "ready"
    except Exception as e:  # noqa: BLE001
        S.model_status, S.model_error = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return
    while True:
        rid = S.gpu_queue.get()
        with S.lock:
            if rid in S.gpu_order:
                S.gpu_order.remove(rid)
        run = S.runs.get(rid)
        if run is None or run["state"] != "queued":
            continue
        item = S.items.get(run["item_id"])
        if item is None:
            run.update(state="failed", error="The item was deleted.")
            continue
        S.gpu_running = rid
        src = idir(item) / item["file"]
        fn = pipelines.segment_video if item["kind"] == "video" else pipelines.segment_image
        execute(run, lambda out, prog, cancel: fn(tracker, src, out, run, prog, cancel))
        S.gpu_running = None


def detect_job(rid):
    run = S.runs.get(rid)
    if run is None or run["state"] != "queued":
        return
    item = S.items.get(run["item_id"])
    if item is None:
        run.update(state="failed", error="The item was deleted.")
        return
    src = idir(item) / item["file"]
    fn = pipelines.detect_video if item["kind"] == "video" else pipelines.detect_image
    if item["kind"] == "video":
        execute(run, lambda out, prog, cancel: fn(src, out, run, budget, OPENROUTER_KEY, prog, cancel,
                                                  max_calls=DETECT_MAX_FRAMES))
    else:
        execute(run, lambda out, prog, cancel: fn(src, out, run, budget, OPENROUTER_KEY, prog, cancel))


@app.on_event("startup")
def startup():
    load_all()
    threading.Thread(target=gpu_worker, daemon=True, name="gpu-worker").start()


# ============================================================================ auth

def key_ok(key: str) -> bool:
    return bool(key) and hmac.compare_digest(key.encode(), APP_PASSWORD.encode())


def cookie_value() -> str:
    # store a hash, not the key itself, in the cookie
    return hashlib.sha256(("sam3:" + APP_PASSWORD).encode()).hexdigest()


def authorized(request: Request) -> bool:
    if not APP_PASSWORD:
        return True
    if hmac.compare_digest(request.cookies.get(COOKIE, ""), cookie_value()):
        return True
    return key_ok(request.headers.get("x-access-key", ""))


@app.middleware("http")
async def require_key(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in ("/api/status", "/api/login") and not authorized(request):
        return JSONResponse({"detail": "Access key required"}, status_code=401)
    return await call_next(request)


class LoginIn(BaseModel):
    key: str


@app.post("/api/login")
def login(body: LoginIn, request: Request, response: Response):
    if APP_PASSWORD and not key_ok(body.key.strip()):
        raise HTTPException(401, "That access key is not right")
    if APP_PASSWORD:
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(COOKIE, cookie_value(), max_age=60 * 60 * 24 * 365, httponly=True,
                            samesite="lax", secure=secure)
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


# ============================================================================ helpers

def check_id(v: str) -> str:
    if not ID_RE.match(v or ""):
        raise HTTPException(404, "not found")
    return v


def get_project(pid) -> dict:
    p = S.projects.get(check_id(pid))
    if not p:
        raise HTTPException(404, "Project not found")
    return p


def get_item(pid, iid) -> dict:
    it = S.items.get(check_id(iid))
    if not it or it["project_id"] != pid:
        raise HTTPException(404, "Item not found")
    return it


def get_run(pid, rid) -> dict:
    run = S.runs.get(check_id(rid))
    if not run or run["project_id"] != pid:
        raise HTTPException(404, "Run not found")
    return run


def item_public(it: dict) -> dict:
    runs = sorted((r for r in S.runs.values() if r["item_id"] == it["id"]), key=lambda r: r["created_at"])
    # a cancelled run shouldn't hide the item's last real result
    shown = [r for r in runs if r["state"] != "cancelled"] or runs
    latest = shown[-1] if shown else None
    latest_done = next((r for r in reversed(runs) if r["state"] == "done"), None)
    return {**it, "runs": len(runs),
            "latest": public_run(latest) if latest else None,
            "latest_done_id": latest_done["id"] if latest_done else None}


def project_public(p: dict) -> dict:
    items = sorted((i for i in S.items.values() if i["project_id"] == p["id"]), key=lambda i: i["created_at"])
    return {**p, "item_count": len(items),
            "images": sum(i["kind"] == "image" for i in items),
            "videos": sum(i["kind"] == "video" for i in items),
            "thumbs": [i["id"] for i in items[:4]]}


# ============================================================================ status

@app.get("/api/status")
def status(request: Request):
    return {
        "model": S.model_status, "model_error": S.model_error,
        "gpu_busy": S.gpu_running is not None, "gpu_queue": len(S.gpu_order),
        "sec_per_frame": round(S.sec_per_frame, 2),
        "auth_required": bool(APP_PASSWORD), "authorized": authorized(request),
        "detection_enabled": bool(OPENROUTER_KEY),
        "budget": budget.summary(), "budget_remaining": round(budget.remaining(), 4),
        "models": [{"key": k, "name": m["name"], "id": m["id"], "note": m["note"],
                    "est_per_image": round(detector.estimate_cost(k, 1), 5)} for k, m in detector.MODELS.items()],
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "max_frames": MAX_FRAMES_LIMIT,
                   "detect_max_frames": DETECT_MAX_FRAMES},
    }


# ============================================================================ projects

class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@app.get("/api/projects")
def list_projects():
    return sorted((project_public(p) for p in S.projects.values()), key=lambda p: p["updated_at"], reverse=True)


@app.post("/api/projects")
def create_project(body: ProjectIn):
    pid = uuid.uuid4().hex
    p = {"id": pid, "name": " ".join(body.name.split()), "created_at": now(), "updated_at": now()}
    S.projects[pid] = p
    save_project(p)
    return project_public(p)


@app.get("/api/projects/{pid}")
def project_detail(pid: str):
    p = get_project(pid)
    items = sorted((i for i in S.items.values() if i["project_id"] == pid), key=lambda i: i["created_at"])
    return {**project_public(p), "items": [item_public(i) for i in items]}


@app.patch("/api/projects/{pid}")
def rename_project(pid: str, body: ProjectIn):
    p = get_project(pid)
    p["name"] = " ".join(body.name.split())
    p["updated_at"] = now()
    save_project(p)
    return project_public(p)


def _cancel_runs(pred):
    for run in S.runs.values():
        if pred(run) and run["state"] in ("queued", "running"):
            run["_cancel"] = True
            if run["state"] == "queued":
                run.update(state="cancelled", stage="cancelled")
                with S.lock:
                    if run["id"] in S.gpu_order:
                        S.gpu_order.remove(run["id"])


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    get_project(pid)
    if S.gpu_running and S.runs.get(S.gpu_running, {}).get("project_id") == pid:
        _cancel_runs(lambda r: r["project_id"] == pid)
        raise HTTPException(409, "A run in this project is still stopping on the GPU. Try again in a few seconds.")
    _cancel_runs(lambda r: r["project_id"] == pid)
    for d in (S.items, S.runs):
        for k in [k for k, v in d.items() if v["project_id"] == pid]:
            d.pop(k)
    S.projects.pop(pid)
    shutil.rmtree(pdir(pid), ignore_errors=True)
    return {"ok": True}


# ============================================================================ items

async def _save_upload(file: UploadFile, dest: Path):
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with dest.open("wb") as f:
        while chunk := await file.read(4 * 1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"'{file.filename}' is larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    return size


def _video_thumb(src: Path, dest: Path, frames: int):
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frames // 10, 30))
    ok, frame = cap.read()
    cap.release()
    if ok:
        h, w = frame.shape[:2]
        if w > 480:
            frame = cv2.resize(frame, (480, round(h * 480 / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])


@app.post("/api/projects/{pid}/items")
async def upload_items(pid: str, files: list[UploadFile]):
    get_project(pid)
    added, errors = [], []
    for file in files:
        name = Path(file.filename or "file").name
        ext = Path(name).suffix.lower()
        kind = "video" if ext in VIDEO_EXTS else "image" if ext in IMAGE_EXTS else None
        if kind is None:
            errors.append(f"{name}: unsupported file type")
            continue
        iid = uuid.uuid4().hex
        item = {"id": iid, "project_id": pid, "name": name, "kind": kind, "created_at": now()}
        d = idir(item)
        d.mkdir(parents=True)
        try:
            raw = d / ("upload" + ext)
            item["size"] = await _save_upload(file, raw)
            if kind == "image":
                img = ImageOps.exif_transpose(Image.open(raw)).convert("RGB")
                if max(img.size) > MAX_IMAGE_SIDE:
                    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
                img.save(d / "source.jpg", quality=95)
                raw.unlink()
                item.update(file="source.jpg", meta={"width": img.width, "height": img.height})
                img.thumbnail((480, 480))
                img.save(d / "thumb.jpg", quality=82)
            else:
                src = d / ("source" + ext)
                raw.rename(src)
                item.update(file=src.name, meta=video_meta(src))
                _video_thumb(src, d / "thumb.jpg", item["meta"]["frames"])
        except HTTPException as e:
            shutil.rmtree(d, ignore_errors=True)
            errors.append(e.detail)
            continue
        except Exception:  # noqa: BLE001
            shutil.rmtree(d, ignore_errors=True)
            errors.append(f"{name}: could not read this file")
            continue
        S.items[iid] = item
        save_item(item)
        added.append(item_public(item))
    touch_project(pid)
    return {"added": added, "errors": errors}


@app.delete("/api/projects/{pid}/items/{iid}")
def delete_item(pid: str, iid: str):
    item = get_item(pid, iid)
    if S.gpu_running and S.runs.get(S.gpu_running, {}).get("item_id") == iid:
        _cancel_runs(lambda r: r["item_id"] == iid)
        raise HTTPException(409, "This item is still stopping on the GPU. Try again in a few seconds.")
    _cancel_runs(lambda r: r["item_id"] == iid)
    for rid in [r["id"] for r in S.runs.values() if r["item_id"] == iid]:
        shutil.rmtree(rdir(S.runs[rid]), ignore_errors=True)
        S.runs.pop(rid)
    shutil.rmtree(idir(item), ignore_errors=True)
    S.items.pop(iid)
    touch_project(pid)
    return {"ok": True}


@app.get("/api/projects/{pid}/items/{iid}/thumb.jpg")
def item_thumb(pid: str, iid: str):
    path = idir(get_item(pid, iid)) / "thumb.jpg"
    if not path.exists():
        raise HTTPException(404, "no thumbnail")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/projects/{pid}/items/{iid}/source")
def item_source(pid: str, iid: str):
    item = get_item(pid, iid)
    return FileResponse(idir(item) / item["file"], media_type="image/jpeg" if item["kind"] == "image" else None)


# ============================================================================ runs

class TargetIn(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=160)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class RunOptions(BaseModel):
    start: float = Field(default=0.0, ge=0)
    end: float | None = Field(default=None, ge=0)
    max_frames: int = Field(default=150, ge=1)          # video segmentation
    detect_fps: float = Field(default=1.0, ge=0.1, le=5)  # video detection: model checks per second
    score_thresh: float = Field(default=0.0, ge=0, le=1)
    boxes: bool = True


class RunIn(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=200)
    task: str = Field(pattern="^(detect|segment)$")
    model: str | None = None
    targets: list[TargetIn] = Field(min_length=1, max_length=10)
    options: RunOptions = RunOptions()


@app.post("/api/projects/{pid}/runs")
def create_runs(pid: str, body: RunIn):
    get_project(pid)
    items = [get_item(pid, iid) for iid in dict.fromkeys(body.item_ids)]
    targets = [{"label": " ".join(t.label.split()), "description": " ".join(t.description.split()),
                "color": (t.color or pipelines.PALETTE[i % len(pipelines.PALETTE)]).lower()}
               for i, t in enumerate(body.targets) if t.label.strip()]
    if not targets:
        raise HTTPException(400, "Add at least one label")
    for t in targets:
        t["prompt"] = build_prompt(t["label"], t["description"])
    o = body.options
    if o.end is not None and o.end <= o.start:
        raise HTTPException(400, "End time must be after start time")

    def detect_calls(it):
        if it["kind"] == "image":
            return 1
        opts = {"start": o.start, "end": o.end, "detect_fps": o.detect_fps}
        return len(pipelines.detect_video_plan(it["meta"], opts, DETECT_MAX_FRAMES)["keyframes"])

    if body.task == "detect":
        if not OPENROUTER_KEY:
            raise HTTPException(503, "Object detection is not configured on this server (no OpenRouter key).")
        if body.model not in detector.MODELS:
            raise HTTPException(400, "Choose a detection model")
        n_calls = sum(detect_calls(it) for it in items)
        try:
            budget.check(detector.estimate_cost(body.model, n_calls))
        except BudgetExceeded as e:
            raise HTTPException(402, str(e))
    else:
        if S.model_status == "error":
            raise HTTPException(503, "SAM 3 failed to load on the server: " + S.model_error)
        if len(S.gpu_order) + len(items) > MAX_QUEUE:
            raise HTTPException(429, "The GPU queue is full right now. Please try again in a few minutes.")

    planned = []
    for it in items:
        opts = {"start": o.start, "end": o.end, "score_thresh": o.score_thresh, "boxes": o.boxes}
        if it["kind"] == "video":
            if o.start >= it["meta"]["duration"]:
                raise HTTPException(400, f"'{it['name']}' is only {it['meta']['duration']:.1f}s long; "
                                         "the start time is past its end")
            if body.task == "segment":
                opts["max_frames"] = min(o.max_frames, MAX_FRAMES_LIMIT)
                _, _, _, total = plan_frames(it["meta"], Options(start=o.start, end=o.end,
                                                                 max_frames=opts["max_frames"]))
            else:
                opts["detect_fps"] = o.detect_fps
                total = detect_calls(it)
        else:
            total = 1
        planned.append((it, opts, total))

    created = []
    for it, opts, total in planned:
        rid = uuid.uuid4().hex
        run = {
            "id": rid, "project_id": pid, "item_id": it["id"], "item_name": it["name"], "kind": it["kind"],
            "task": body.task, "model": body.model if body.task == "detect" else "sam3",
            "targets": targets, "options": opts, "created_at": now(),
            "state": "queued", "stage": "queued", "done": 0, "total": total,
            "counts": {}, "preview_version": 0, "error": "", "result": None, "cost": 0.0,
        }
        S.runs[rid] = run
        save_run(run)
        if body.task == "segment":
            with S.lock:
                S.gpu_order.append(rid)
            S.gpu_queue.put(rid)
        else:
            detect_pool.submit(detect_job, rid)
        created.append(public_run(run))
    touch_project(pid)
    return created


@app.get("/api/projects/{pid}/runs")
def list_runs(pid: str, item: str | None = None, active: int = 0):
    get_project(pid)
    runs = [r for r in S.runs.values() if r["project_id"] == pid and (item is None or r["item_id"] == item)
            and (not active or r["state"] in ("queued", "running"))]
    return [public_run(r) for r in sorted(runs, key=lambda r: r["created_at"], reverse=True)]


@app.get("/api/projects/{pid}/runs/{rid}")
def run_status(pid: str, rid: str):
    return public_run(get_run(pid, rid))


@app.post("/api/projects/{pid}/runs/{rid}/cancel")
def cancel_run(pid: str, rid: str):
    run = get_run(pid, rid)
    if run["state"] == "queued":
        run.update(state="cancelled", stage="cancelled", finished_at=now())
        with S.lock:
            if rid in S.gpu_order:
                S.gpu_order.remove(rid)
        save_run(run)
    elif run["state"] == "running":
        run["_cancel"] = True
    return public_run(run)


@app.delete("/api/projects/{pid}/runs/{rid}")
def delete_run(pid: str, rid: str):
    run = get_run(pid, rid)
    if run["state"] in ("queued", "running"):
        raise HTTPException(409, "Cancel the run before deleting it")
    shutil.rmtree(rdir(run), ignore_errors=True)
    S.runs.pop(rid)
    return {"ok": True}


@app.get("/api/projects/{pid}/runs/{rid}/preview.jpg")
def run_preview(pid: str, rid: str):
    run = get_run(pid, rid)
    data = run.get("_preview")
    if data is not None:
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    path = rdir(run) / "thumb.jpg"
    if not path.exists():
        raise HTTPException(404, "no preview yet")
    return FileResponse(path, media_type="image/jpeg")


def _dl_name(run, suffix):
    stem = Path(run["item_name"]).stem
    labels = "-".join(t["label"] for t in run["targets"])
    task = "detected" if run["task"] == "detect" else ("tracked" if run["kind"] == "video" else "segmented")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{stem}_{labels}")[:80] + f"_{task}{suffix}"


RUN_FILES = {"result.jpg": "image/jpeg", "result.mp4": "video/mp4", "result.json": "application/json",
             "mask.png": "image/png", "thumb.jpg": "image/jpeg"}


@app.get("/api/projects/{pid}/runs/{rid}/files/{name}")
def run_file(pid: str, rid: str, name: str, download: int = 0):
    run = get_run(pid, rid)
    if name not in RUN_FILES:
        raise HTTPException(404, "not found")
    path = rdir(run) / name
    if run["state"] != "done" or not path.exists():
        raise HTTPException(404, "not ready")
    suffix = {"result.json": ".json", "mask.png": "_mask.png"}.get(name, Path(name).suffix)
    return FileResponse(path, media_type=RUN_FILES[name],
                        filename=_dl_name(run, suffix) if download else None,
                        content_disposition_type="attachment" if download else "inline")


# ============================================================================ export

def coco_for(pid: str, task: str) -> dict:
    """COCO-style annotations from the latest finished `task` run of every image item."""
    images, anns, cats = [], [], {}
    for it in sorted((i for i in S.items.values() if i["project_id"] == pid and i["kind"] == "image"),
                     key=lambda i: i["created_at"]):
        done = sorted((r for r in S.runs.values()
                       if r["item_id"] == it["id"] and r["state"] == "done" and r["task"] == task),
                      key=lambda r: r["created_at"])
        if not done:
            continue
        run = done[-1]
        try:
            res = json.loads((rdir(run) / "result.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": f"images/{it['id'][:6]}_{it['name']}", "width": res["width"],
                       "height": res["height"], "task": run["task"], "model": run["model"]})
        for o in res["objects"]:
            cid = cats.setdefault(o["class"], len(cats) + 1)
            x1, y1, x2, y2 = o["box"]
            ann = {"id": len(anns) + 1, "image_id": img_id, "category_id": cid, "iscrowd": 0,
                   "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                   "area": o.get("area", round((x2 - x1) * (y2 - y1), 1))}
            if o.get("polygons"):
                ann["segmentation"] = o["polygons"]
            if "score" in o:
                ann["score"] = o["score"]
            anns.append(ann)
    return {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for n, i in cats.items()]}


@app.get("/api/projects/{pid}/export.zip")
def export_project(pid: str):
    p = get_project(pid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for task in ("detect", "segment"):
            coco = coco_for(pid, task)
            if coco["images"]:
                name = "coco_detection.json" if task == "detect" else "coco_segmentation.json"
                z.writestr(name, json.dumps(coco, indent=1))
        for it in (i for i in S.items.values() if i["project_id"] == pid):
            prefix = f"{it['id'][:6]}_{Path(it['name']).stem}"
            if it["kind"] == "image":
                z.write(idir(it) / it["file"], f"images/{it['id'][:6]}_{it['name']}")
            for run in (r for r in S.runs.values() if r["item_id"] == it["id"] and r["state"] == "done"):
                base = f"results/{prefix}/{run['task']}_{run['model']}_{run['id'][:6]}"
                for name in ("result.jpg", "result.mp4", "result.json", "mask.png"):
                    f = rdir(run) / name
                    if f.exists():
                        z.write(f, f"{base}/{name}",
                                compress_type=zipfile.ZIP_STORED if name.endswith((".mp4", ".jpg", ".png")) else None)
    buf.seek(0)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", p["name"])[:60] or "project"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}_annotations.zip"'})


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
