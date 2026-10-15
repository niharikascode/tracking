"""The four annotation pipelines (detect/segment x image/video) and the OpenRouter budget.

Every run writes into its own directory:
  result.jpg | result.mp4   annotated media
  result.json               machine-readable annotations
  mask.png                  instance-id mask (image segmentation only; 0 = background)
  thumb.jpg                 small preview for the media grid
"""

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import detector
from tracker import (Cancelled, FfmpegWriter, Options, Progress, Target, counts_banner, hex_to_rgb,
                     text_color, video_meta)

# default label colours, in the order labels are added (the page uses the same list)
PALETTE = ["#2f6fed", "#e5484d", "#30a46c", "#f5a524", "#8e4ec6",
           "#12a594", "#e93d82", "#0090ff", "#ff8b3e", "#6e56cf"]
DETECT_PARALLEL = 4      # concurrent model calls within one video run
MAX_OUT_FPS = 30         # detection videos are re-encoded at most at this frame rate


# ----------------------------------------------------------------------------- budget

class BudgetExceeded(Exception):
    pass


class Budget:
    """Tracks real OpenRouter spend (USD) per UTC day and in total, persisted to disk."""

    def __init__(self, path: Path, daily_limit: float, total_limit: float):
        self.path, self.daily_limit, self.total_limit = path, daily_limit, total_limit
        self.lock = threading.Lock()
        try:
            self.data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            self.data = {"day": "", "spent_today": 0.0, "spent_total": 0.0, "calls_today": 0}

    def _roll(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.data["day"] != today:
            self.data.update(day=today, spent_today=0.0, calls_today=0)

    def remaining(self) -> float:
        with self.lock:
            self._roll()
            day_left = self.daily_limit - self.data["spent_today"]
            total_left = self.total_limit - self.data["spent_total"] if self.total_limit > 0 else math.inf
            return max(0.0, min(day_left, total_left))

    def check(self, estimate: float):
        left = self.remaining()
        if estimate > left + 1e-9:
            raise BudgetExceeded(
                f"This needs about ${estimate:.3f} of detection credit but only ${left:.3f} is left today "
                f"(daily limit ${self.daily_limit:.2f}). Choose a cheaper model, fewer items or fewer "
                f"checks per second, or try tomorrow.")

    def add(self, cost: float):
        with self.lock:
            self._roll()
            self.data["spent_today"] += cost
            self.data["spent_total"] += cost
            self.data["calls_today"] += 1
            self.path.write_text(json.dumps(self.data))

    def summary(self) -> dict:
        with self.lock:
            self._roll()
            return {"spent_today": round(self.data["spent_today"], 4), "daily_limit": self.daily_limit,
                    "spent_total": round(self.data["spent_total"], 4), "total_limit": self.total_limit,
                    "calls_today": self.data["calls_today"]}


# ----------------------------------------------------------------------------- colours & drawing

def label_colors(run) -> dict:
    """label -> RGB, from each target's chosen colour (or the default palette)."""
    out = {}
    for i, t in enumerate(run["targets"]):
        out.setdefault(t["label"], hex_to_rgb(t.get("color") or PALETTE[i % len(PALETTE)]))
    return out


def draw_boxes(rgb: np.ndarray, objects: list[dict], colors: dict, labels: list[str]) -> np.ndarray:
    out = rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(out.shape[:2]) / 1100)
    thick = max(1, int(round(scale * 2)))
    for o in objects:
        x1, y1, x2, y2 = map(int, o["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colors.get(o["class"], (255, 255, 255)), thick + 1)
    for o in objects:  # labels after boxes so no box covers a label
        x1, y1 = int(o["box"][0]), int(o["box"][1])
        c = colors.get(o["class"], (255, 255, 255))
        text = f'{o["class"]} #{o["id"]}'
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        ty = y1 - 4 if y1 - th - base - 4 > 0 else y1 + th + base + 4
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 4, ty + base // 2), c, -1)
        cv2.putText(out, text, (x1 + 2, ty - base // 2), font, scale, text_color(c), thick, cv2.LINE_AA)
    counts_banner(out, {lab: sum(o["class"] == lab for o in objects) for lab in labels})
    return out


def save_jpg(path: Path, rgb: np.ndarray, quality=92):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_thumb(path: Path, rgb: np.ndarray, width=480):
    h, w = rgb.shape[:2]
    if w > width:
        rgb = cv2.resize(rgb, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    save_jpg(path, rgb, 80)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not read image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------- planning

def detect_video_plan(meta: dict, options: dict, max_calls: int) -> dict:
    """Which frames the detector looks at: `detect_fps` checks per second over [start, end]."""
    fps = meta["fps"]
    first = max(0, int(round(options.get("start", 0) * fps)))
    last = meta["frames"] if options.get("end") is None else min(meta["frames"], int(round(options["end"] * fps)))
    span = max(0, last - first)
    rate = float(options.get("detect_fps", 1.0))
    n = max(1, min(max_calls, math.ceil(span / fps * rate) + 1)) if span else 0
    if n == 1:
        keys = [first + span // 2] if span else []
    else:  # evenly from the first to the last frame, so boxes exist for the whole range
        keys = sorted({first + round(i * (span - 1) / (n - 1)) for i in range(n)})
    return {"first": first, "last": last, "keyframes": keys}


# ----------------------------------------------------------------------------- box interpolation

def _box_dist(a, b) -> float:
    """Centre distance relative to the boxes' mean diagonal (0 = same place)."""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    diag = (math.hypot(a[2] - a[0], a[3] - a[1]) + math.hypot(b[2] - b[0], b[3] - b[1])) / 2 or 1
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / diag


def match_objects(prev: list[dict], nxt: list[dict], max_dist=0.75) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of same-label boxes between two keyframes."""
    pairs = sorted((_box_dist(p["box"], q["box"]), i, j)
                   for i, p in enumerate(prev) for j, q in enumerate(nxt) if p["class"] == q["class"])
    used_i, used_j, out = set(), set(), []
    for d, i, j in pairs:
        if d > max_dist:
            break
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def interpolate(prev: list[dict], nxt: list[dict], t: float, matches) -> list[dict]:
    """Boxes at fraction t (0..1) between two keyframes: matched boxes glide, unmatched
    ones are only shown near their own keyframe."""
    out, mi, mj = [], {i for i, _ in matches}, {j for _, j in matches}
    for i, j in matches:
        a, b = prev[i]["box"], nxt[j]["box"]
        out.append({"class": prev[i]["class"], "box": [a[k] + (b[k] - a[k]) * t for k in range(4)]})
    out += [{"class": o["class"], "box": o["box"]} for i, o in enumerate(prev) if i not in mi and t < 0.35]
    out += [{"class": o["class"], "box": o["box"]} for j, o in enumerate(nxt) if j not in mj and t > 0.65]
    for n, o in enumerate(sorted(out, key=lambda o: (o["box"][1], o["box"][0]))):
        o["id"] = n + 1
    return out


def reject_outlier_keyframes(results: dict, labels: list[str]) -> list[int]:
    """Keyframes whose count is far above the others (a model glitch), to be ignored."""
    if len(results) < 4:
        return []
    bad = []
    for f, objs in results.items():
        for lab in labels:
            n = sum(o["class"] == lab for o in objs)
            others = sorted(sum(o["class"] == lab for o in results[g]) for g in results if g != f)
            med = others[len(others) // 2]
            # far above the others (a looping answer) or a sudden empty answer among busy frames
            if n > 2 * med + 5 or (med >= 6 and n < med / 4):
                bad.append(f)
                break
    return bad


# ----------------------------------------------------------------------------- pipelines

def _targets(run) -> list[dict]:
    return [{"label": t["label"], "prompt": t["prompt"]} for t in run["targets"]]


def _labels(run) -> list[str]:
    return list(dict.fromkeys(t["label"] for t in run["targets"]))


def _counts(objects, labels):
    return {lab: sum(o["class"] == lab for o in objects) for lab in labels}


def detect_image(src: Path, out: Path, run: dict, budget: Budget, api_key: str,
                 on_progress, should_cancel) -> dict:
    rgb = load_rgb(src)
    labels, colors = _labels(run), label_colors(run)
    on_progress(Progress("detecting", 0, 1))
    budget.check(detector.estimate_cost(run["model"], 1))
    r = detector.detect(Image.fromarray(rgb), _targets(run), run["model"], api_key)
    budget.add(r["cost"])
    run["cost"] = round(run.get("cost", 0.0) + r["cost"], 5)
    objects = [{"id": i + 1, "class": o["label"], "box": o["box"]} for i, o in enumerate(r["objects"])]
    vis = draw_boxes(rgb, objects, colors, labels)
    save_jpg(out / "result.jpg", vis)
    save_thumb(out / "thumb.jpg", vis)
    on_progress(Progress("detecting", 1, 1, vis, _counts(objects, labels)))
    result = {"task": "detect", "model": detector.MODELS[run["model"]]["id"], "width": rgb.shape[1],
              "height": rgb.shape[0], "targets": run["targets"], "objects": objects, "cost_usd": round(r["cost"], 5),
              "tokens": {"in": r["tokens_in"], "out": r["tokens_out"]}}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return {"counts": _counts(objects, labels), "cost": r["cost"]}


def detect_video(src: Path, out: Path, run: dict, budget: Budget, api_key: str,
                 on_progress, should_cancel, max_calls: int = 120) -> dict:
    """Detect on keyframes (`detect_fps` per second), then render the WHOLE range at normal
    speed, each frame showing the boxes of the most recent keyframe."""
    meta = video_meta(src)
    plan = detect_video_plan(meta, run["options"], max_calls)
    keys = plan["keyframes"]
    if not keys:
        raise ValueError("the selected time range has no frames")
    labels, colors, targets = _labels(run), label_colors(run), _targets(run)
    budget.check(detector.estimate_cost(run["model"], len(keys)))

    # 1) read the keyframes
    cap = cv2.VideoCapture(str(src))
    key_rgb = {}
    for f in keys:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, bgr = cap.read()
        if ok:
            key_rgb[f] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    if not key_rgb:
        raise ValueError("could not read frames from this video")

    # 2) ask the model about each keyframe, several at a time
    results, errors = {}, []
    lock = threading.Lock()
    total = len(key_rgb)
    on_progress(Progress("detecting", 0, total))

    def one(f):
        if should_cancel():
            return f, None, "cancelled"
        try:
            budget.check(detector.estimate_cost(run["model"], 1))
            r = detector.detect(Image.fromarray(key_rgb[f]), targets, run["model"], api_key)
        except (detector.DetectionError, BudgetExceeded) as e:
            return f, None, str(e)
        budget.add(r["cost"])
        with lock:
            run["cost"] = round(run.get("cost", 0.0) + r["cost"], 5)
        return f, r, None

    with ThreadPoolExecutor(max_workers=DETECT_PARALLEL) as pool:
        futures = [pool.submit(one, f) for f in key_rgb]
        for fut in as_completed(futures):
            f, r, err = fut.result()
            if err == "cancelled":
                continue
            if err:
                errors.append(err)
            else:
                objs = [{"id": i + 1, "class": d["label"], "box": d["box"]} for i, d in enumerate(r["objects"])]
                results[f] = objs
                vis = draw_boxes(key_rgb[f], objs, colors, labels)
                on_progress(Progress("detecting", len(results) + len(errors), total, vis, _counts(objs, labels)))
            if should_cancel():
                for fu in futures:
                    fu.cancel()
    if should_cancel():
        raise Cancelled()
    if not results or len(errors) > total / 2:
        raise detector.DetectionError(
            f"{len(errors)} of {total} frames failed. First error: {errors[0] if errors else 'unknown'}")

    outliers = reject_outlier_keyframes(results, labels)
    for f in outliers:
        if len(results) > 1:
            del results[f]

    # 3) render the full range; boxes glide between keyframes (matched by position)
    fps = meta["fps"]
    stride = max(1, math.ceil(fps / MAX_OUT_FPS))
    out_fps = fps / stride
    done_keys = sorted(results)
    seg_matches = {k: match_objects(results[done_keys[k]], results[done_keys[k + 1]])
                   for k in range(len(done_keys) - 1)}
    cap = cv2.VideoCapture(str(src))
    if plan["first"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, plan["first"])
    writer, idx, k = None, plan["first"], 0
    n_out = math.ceil((plan["last"] - plan["first"]) / stride)
    written = 0
    try:
        while idx < plan["last"]:
            ok, bgr = cap.read()
            if not ok:
                break
            if (idx - plan["first"]) % stride == 0:
                while k + 1 < len(done_keys) and done_keys[k + 1] <= idx:
                    k += 1
                if idx <= done_keys[0] or k == len(done_keys) - 1:
                    objs = results[done_keys[0] if idx <= done_keys[0] else done_keys[-1]]
                else:
                    a_f, b_f = done_keys[k], done_keys[k + 1]
                    objs = interpolate(results[a_f], results[b_f], (idx - a_f) / (b_f - a_f), seg_matches[k])
                vis = draw_boxes(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), objs, colors, labels)
                if writer is None:
                    writer = FfmpegWriter(out / "result.mp4", vis.shape[1], vis.shape[0], out_fps)
                writer.write(vis)
                written += 1
                if written % 30 == 0:
                    if should_cancel():
                        raise Cancelled()
                    on_progress(Progress("rendering", written, n_out))
            idx += 1
        if writer is None:
            raise ValueError("no frames could be read from the video")
        writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        cap.release()
    first_key = done_keys[len(done_keys) // 2]
    save_thumb(out / "thumb.jpg", draw_boxes(key_rgb[first_key], results[first_key], colors, labels))

    peak = {lab: max(sum(o["class"] == lab for o in results[f]) for f in done_keys) for lab in labels}
    result = {"task": "detect", "model": detector.MODELS[run["model"]]["id"], "source": meta,
              "detect_fps": run["options"].get("detect_fps", 1.0), "output_fps": round(out_fps, 3),
              "keyframes": [{"frame": f, "time": round(f / fps, 3), "objects": results[f]} for f in done_keys],
              "failed_keyframes": len(errors), "rejected_keyframes": [round(f / fps, 3) for f in outliers],
              "cost_usd": round(run.get("cost", 0.0), 5), "targets": run["targets"],
              "note": "Keyframes were checked by the model; frames in between show boxes interpolated "
                      "between matching objects of neighbouring keyframes."}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return {"counts": peak, "cost": run.get("cost", 0.0), "frames": len(done_keys),
            "failed_frames": len(errors) + len(outliers), "duration": round((plan["last"] - plan["first"]) / fps, 2)}


def segment_image(tracker, src: Path, out: Path, run: dict, on_progress, should_cancel) -> dict:
    rgb = load_rgb(src)
    labels = _labels(run)
    on_progress(Progress("segmenting", 0, 1))
    o = run["options"]
    r = tracker.segment_image(rgb, [Target(t["label"], t["description"]) for t in run["targets"]],
                              score_thresh=o.get("score_thresh", 0.0), boxes=o.get("boxes", True),
                              label_colors=label_colors(run))
    save_jpg(out / "result.jpg", r["image"])
    save_thumb(out / "thumb.jpg", r["image"])
    cv2.imwrite(str(out / "mask.png"), r["instance_mask"])
    counts = {lab: sum(ob["class"] == lab for ob in r["objects"]) for lab in labels}
    on_progress(Progress("segmenting", 1, 1, r["image"], counts))
    result = {"task": "segment", "model": "facebook/sam3", "width": rgb.shape[1], "height": rgb.shape[0],
              "mask_png": "mask.png (pixel value = object id, 0 = background)", "targets": run["targets"],
              "objects": r["objects"]}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return {"counts": counts}


def segment_video(tracker, src: Path, out: Path, run: dict, on_progress, should_cancel) -> dict:
    o = run["options"]
    opts = Options(start=o.get("start", 0.0), end=o.get("end"), max_frames=o.get("max_frames", 150),
                   score_thresh=o.get("score_thresh", 0.0), boxes=o.get("boxes", True))
    first_preview = [True]

    def progress(p):
        if p.preview is not None and first_preview[0]:
            save_thumb(out / "thumb.jpg", p.preview)
            first_preview[0] = False
        on_progress(p)

    s = tracker.run(src, [Target(t["label"], t["description"]) for t in run["targets"]], out, opts,
                    on_progress=progress, should_cancel=should_cancel, label_colors=label_colors(run))
    (out / "tracked.mp4").rename(out / "result.mp4")
    (out / "tracks.json").rename(out / "result.json")
    return {"counts": s["unique_tracks_per_class"], "frames": s["frames_processed"],
            "seconds": s["seconds"], "stride": s["stride"], "output_fps": s["output_fps"]}
