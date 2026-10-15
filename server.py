"""Web app for SAM 3 video tracking.

One GPU worker thread holds the model and processes jobs from a queue, one at a
time. Browsers upload a video, submit labels + descriptions, poll progress (with a
live preview frame) and download the result.

Run:  .venv/bin/uvicorn server:app --host 0.0.0.0 --port 7860
"""

import hmac
import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tracker import Cancelled, Options, Sam3Tracker, Target, build_prompt, plan_frames, video_meta

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))
MAX_FRAMES_LIMIT = int(os.getenv("MAX_FRAMES_LIMIT", "600"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "20"))
RETENTION_HOURS = float(os.getenv("RETENTION_HOURS", "48"))
DTYPE = os.getenv("DTYPE", "fp16")

VIDEOS = DATA_DIR / "videos"
JOBS = DATA_DIR / "jobs"
VIDEOS.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)

ID_RE = re.compile(r"^[0-9a-f]{32}$")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

app = FastAPI(title="SAM 3 Tracker")


# ----------------------------------------------------------------------------- state

class State:
    def __init__(self):
        self.model_status = "loading"     # loading | ready | error
        self.model_error = ""
        self.jobs: dict[str, dict] = {}   # job id -> job record (public fields + internals prefixed "_")
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.order: list[str] = []         # queued job ids, for position display
        self.running: str | None = None
        self.sec_per_frame = 7.5           # running estimate, updated after each job
        self.lock = threading.Lock()


S = State()


def public(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    if job["state"] == "queued":
        with S.lock:
            out["position"] = S.order.index(job["id"]) + 1 if job["id"] in S.order else 0
    remaining = max(0, job["total"] - job["done"])
    if job["state"] == "running":
        out["eta_seconds"] = round(remaining * S.sec_per_frame)
    elif job["state"] == "queued":
        out["eta_seconds"] = round(job["total"] * S.sec_per_frame)
    return out


def save_job(job: dict):
    path = JOBS / job["id"] / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in job.items() if not k.startswith("_")}, indent=1))


def load_saved_jobs():
    for p in JOBS.glob("*/job.json"):
        try:
            job = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("state") in ("queued", "running"):
            job.update(state="failed", error="The server restarted before this run finished. Please run it again.")
        S.jobs[job["id"]] = job


# ----------------------------------------------------------------------------- worker

def worker():
    try:
        tracker = Sam3Tracker(token=HF_TOKEN, dtype=DTYPE)
        S.model_status = "ready"
    except Exception as e:  # noqa: BLE001 - surface any load failure to the UI
        S.model_status, S.model_error = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return

    while True:
        jid = S.queue.get()
        with S.lock:
            if jid in S.order:
                S.order.remove(jid)
        job = S.jobs.get(jid)
        if job is None or job["state"] != "queued":
            continue
        S.running = jid
        job.update(state="running", stage="reading", started_at=time.time())
        tracking_t0 = [None]

        def on_progress(p):
            job["stage"], job["done"], job["total"] = p.stage, p.done, p.total
            if p.stage == "tracking":
                if tracking_t0[0] is None:
                    tracking_t0[0] = time.time()
                elif p.done > 1:
                    # blend this job's speed into the global estimate
                    spf = (time.time() - tracking_t0[0]) / (p.done - 1)
                    S.sec_per_frame = 0.7 * S.sec_per_frame + 0.3 * spf
            if p.counts:
                job["counts"] = p.counts
            if p.preview is not None:
                img = p.preview
                if img.shape[1] > 960:
                    img = cv2.resize(img, (960, round(img.shape[0] * 960 / img.shape[1])), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    job["_preview"] = buf.tobytes()
                    job["preview_version"] = p.done

        try:
            summary = tracker.run(
                VIDEOS / job["video_id"] / "source", [Target(t["label"], t["description"]) for t in job["targets"]],
                JOBS / jid, Options(**job["options"]),
                on_progress=on_progress, should_cancel=lambda: job.get("_cancel", False))
            job.update(state="done", stage="done", result={
                k: summary[k] for k in ("unique_tracks_per_class", "frames_processed", "seconds",
                                        "stride", "output_fps", "range_seconds")})
            if job.get("_preview"):
                (JOBS / jid / "preview.jpg").write_bytes(job["_preview"])
        except Cancelled:
            job.update(state="cancelled", stage="cancelled")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            if "out of memory" in msg.lower():
                msg = "The GPU ran out of memory. Try fewer frames (lower 'Max frames'), a shorter time range, or fewer labels."
            job.update(state="failed", error=msg)
        finally:
            job["finished_at"] = time.time()
            S.running = None
            save_job(job)


def janitor():
    """Delete uploads and results older than RETENTION_HOURS."""
    while True:
        cutoff = time.time() - RETENTION_HOURS * 3600
        active = {j["video_id"] for j in S.jobs.values() if j["state"] in ("queued", "running")}
        for d in list(VIDEOS.iterdir()) + list(JOBS.iterdir()):
            try:
                if d.stat().st_mtime < cutoff and d.name not in active and d.name != S.running:
                    shutil.rmtree(d, ignore_errors=True)
                    S.jobs.pop(d.name, None)
            except FileNotFoundError:
                pass
        time.sleep(3600)


@app.on_event("startup")
def startup():
    load_saved_jobs()
    threading.Thread(target=worker, daemon=True, name="gpu-worker").start()
    threading.Thread(target=janitor, daemon=True, name="janitor").start()


# ----------------------------------------------------------------------------- auth

@app.middleware("http")
async def require_password(request: Request, call_next):
    path = request.url.path
    if APP_PASSWORD and path.startswith("/api/") and path != "/api/status":
        key = request.headers.get("x-access-key") or request.query_params.get("key") or ""
        if not hmac.compare_digest(key.encode(), APP_PASSWORD.encode()):
            return JSONResponse({"detail": "Access key required"}, status_code=401)
    return await call_next(request)


# ----------------------------------------------------------------------------- helpers

def check_id(value: str) -> str:
    if not ID_RE.match(value):
        raise HTTPException(404, "not found")
    return value


def get_job(jid: str) -> dict:
    job = S.jobs.get(check_id(jid))
    if job is None:
        raise HTTPException(404, "run not found")
    return job


# ----------------------------------------------------------------------------- API

@app.get("/api/status")
def status():
    return {
        "model": S.model_status, "model_error": S.model_error,
        "queue_length": len(S.order), "busy": S.running is not None,
        "sec_per_frame": round(S.sec_per_frame, 2),
        "auth_required": bool(APP_PASSWORD),
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "max_frames": MAX_FRAMES_LIMIT},
    }


@app.get("/api/auth-check")
def auth_check():
    return {"ok": True}


@app.post("/api/videos")
async def upload_video(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext or '?'}'. Use one of: {', '.join(sorted(VIDEO_EXTS))}")
    vid = uuid.uuid4().hex
    vdir = VIDEOS / vid
    vdir.mkdir(parents=True)
    dest = vdir / "source"
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
                f.write(chunk)
        meta = video_meta(dest)
    except HTTPException:
        shutil.rmtree(vdir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(400, "Could not read this video. Try an MP4 (H.264) file.")

    cap = cv2.VideoCapture(str(dest))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(vdir / "thumb.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    info = {"video_id": vid, "name": Path(file.filename).name, "size": size, "meta": meta}
    (vdir / "info.json").write_text(json.dumps(info))
    return info


@app.get("/api/videos/{vid}")
def video_info(vid: str):
    path = VIDEOS / check_id(vid) / "info.json"
    if not path.exists():
        raise HTTPException(404, "Video not found (it may have expired)")
    return json.loads(path.read_text())


@app.get("/api/videos/{vid}/thumb.jpg")
def video_thumb(vid: str):
    path = VIDEOS / check_id(vid) / "thumb.jpg"
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/jpeg")


class TargetIn(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=160)


class OptionsIn(BaseModel):
    start: float = Field(default=0.0, ge=0)
    end: float | None = Field(default=None, ge=0)
    max_frames: int = Field(default=150, ge=1)
    score_thresh: float = Field(default=0.0, ge=0, le=1)
    boxes: bool = True


class JobIn(BaseModel):
    video_id: str
    targets: list[TargetIn] = Field(min_length=1, max_length=10)
    options: OptionsIn = OptionsIn()


@app.post("/api/prompt-preview")
def prompt_preview(targets: list[TargetIn]):
    return [build_prompt(t.label, t.description) for t in targets]


@app.post("/api/jobs")
def create_job(body: JobIn):
    vid = check_id(body.video_id)
    info_path = VIDEOS / vid / "info.json"
    if not info_path.exists():
        raise HTTPException(404, "Video not found (it may have expired). Please upload it again.")
    if S.model_status == "error":
        raise HTTPException(503, "The model failed to load on the server: " + S.model_error)
    if len(S.order) >= MAX_QUEUE:
        raise HTTPException(429, "The queue is full right now. Please try again in a few minutes.")
    info = json.loads(info_path.read_text())

    targets = [{"label": " ".join(t.label.split()), "description": " ".join(t.description.split())}
               for t in body.targets if t.label.strip()]
    if not targets:
        raise HTTPException(400, "Add at least one label")
    o = body.options
    if o.end is not None and o.end <= o.start:
        raise HTTPException(400, "End time must be after start time")
    options = {"start": o.start, "end": o.end, "max_frames": min(o.max_frames, MAX_FRAMES_LIMIT),
               "score_thresh": o.score_thresh, "boxes": o.boxes}
    _, _, stride, n = plan_frames(info["meta"], Options(**options))
    if n == 0:
        raise HTTPException(400, "The selected time range has no frames")

    jid = uuid.uuid4().hex
    job = {
        "id": jid, "video_id": vid, "video_name": info["name"], "created_at": time.time(),
        "targets": [dict(t, prompt=build_prompt(t["label"], t["description"])) for t in targets],
        "options": options, "planned": {"frames": n, "stride": stride},
        "state": "queued", "stage": "queued", "done": 0, "total": n,
        "counts": {}, "preview_version": 0, "error": "", "result": None,
    }
    S.jobs[jid] = job
    with S.lock:
        S.order.append(jid)
    S.queue.put(jid)
    return public(job)


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    return public(get_job(jid))


@app.get("/api/videos/{vid}/jobs")
def video_jobs(vid: str):
    check_id(vid)
    runs = [public(j) for j in S.jobs.values() if j["video_id"] == vid]
    return sorted(runs, key=lambda j: j["created_at"], reverse=True)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    job = get_job(jid)
    if job["state"] == "queued":
        job.update(state="cancelled", stage="cancelled", finished_at=time.time())
        with S.lock:
            if jid in S.order:
                S.order.remove(jid)
        save_job(job)
    elif job["state"] == "running":
        job["_cancel"] = True
    return public(job)


@app.get("/api/jobs/{jid}/preview.jpg")
def job_preview(jid: str):
    job = get_job(jid)
    data = job.get("_preview")
    if data is None:
        path = JOBS / jid / "preview.jpg"
        if not path.exists():
            raise HTTPException(404, "no preview yet")
        data = path.read_bytes()
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _download_name(job, ext):
    stem = Path(job.get("video_name") or "video").stem
    labels = "-".join(t["label"] for t in job["targets"])
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{stem}_{labels}")[:80]
    return f"{safe}_tracked.{ext}"


@app.get("/api/jobs/{jid}/video")
def job_video(jid: str, download: int = 0):
    job = get_job(jid)
    path = JOBS / jid / "tracked.mp4"
    if job["state"] != "done" or not path.exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(path, media_type="video/mp4",
                        filename=_download_name(job, "mp4") if download else None,
                        content_disposition_type="attachment" if download else "inline")


@app.get("/api/jobs/{jid}/tracks.json")
def job_tracks(jid: str):
    job = get_job(jid)
    path = JOBS / jid / "tracks.json"
    if job["state"] != "done" or not path.exists():
        raise HTTPException(404, "tracks not ready")
    return FileResponse(path, media_type="application/json", filename=_download_name(job, "json"))


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
