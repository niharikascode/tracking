"""SAM 3 text-prompted video tracking engine.

Loads facebook/sam3 once and runs tracking jobs, reporting progress per frame so
callers (the CLI and the web server) can show live previews.
"""

import colorsys
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

MODEL_ID = "facebook/sam3"

# Words that start a trailing qualifier ("person wearing a hat"), as opposed to
# adjectives that go in front of the noun ("red car").
_TRAILING_STARTERS = {
    "with", "without", "wearing", "in", "on", "holding", "carrying", "that", "which", "who",
    "near", "at", "under", "behind", "next", "beside", "inside", "riding", "pushing", "pulling",
    "sitting", "standing", "walking", "running", "having", "from", "of", "by", "over", "using",
}


class Cancelled(Exception):
    pass


@dataclass
class Target:
    """One thing to track: the display label plus an optional free-text description."""
    label: str
    description: str = ""

    @property
    def prompt(self) -> str:
        return build_prompt(self.label, self.description)


@dataclass
class Options:
    start: float = 0.0
    end: float | None = None
    stride: int = 0              # 0 = auto, chosen from max_frames
    max_frames: int = 150
    score_thresh: float = 0.0
    alpha: float = 0.45
    boxes: bool = True
    stream: bool = False


@dataclass
class Progress:
    stage: str                   # "reading" | "preparing" | "tracking" | "encoding"
    done: int = 0
    total: int = 0
    preview: np.ndarray | None = None   # RGB annotated frame
    counts: dict = field(default_factory=dict)


def build_prompt(label: str, description: str) -> str:
    """Merge a label and its description into one SAM3 noun phrase.

    "person" + "wearing a red jacket" -> "person wearing a red jacket"
    "car"    + "red"                  -> "red car"
    "car"    + "a red sports car"     -> "a red sports car"  (already names the label)
    """
    label, desc = label.strip(), " ".join(description.split())
    if not desc:
        return label
    if re.search(rf"\b{re.escape(label.lower())}s?\b", desc.lower()):
        return desc
    if desc.split()[0].lower() in _TRAILING_STARTERS:
        return f"{label} {desc}"
    return f"{desc} {label}"


def video_meta(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta = dict(fps=fps,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if meta["frames"] <= 0 or meta["width"] <= 0:
        raise ValueError("video has no readable frames")
    meta["duration"] = meta["frames"] / fps
    return meta


def plan_frames(meta: dict, opts: Options) -> tuple[int, int, int, int]:
    """Return (first, last_exclusive, stride, n_processed) for the requested range."""
    fps = meta["fps"]
    first = max(0, int(round(opts.start * fps)))
    last = meta["frames"] if opts.end is None else min(meta["frames"], int(round(opts.end * fps)))
    span = max(0, last - first)
    if opts.stride:
        stride = opts.stride
    elif opts.max_frames:
        stride = max(1, math.ceil(span / opts.max_frames))
    else:
        stride = 1
    n = math.ceil(span / stride) if span else 0
    if opts.max_frames:
        n = min(n, opts.max_frames)
    return first, last, stride, n


def read_frames(path, first, last, stride, limit):
    cap = cv2.VideoCapture(str(path))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    idx, kept = first, 0
    while idx < last and (not limit or kept < limit):
        ok, bgr = cap.read()
        if not ok:
            break
        if (idx - first) % stride == 0:
            yield idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            kept += 1
        idx += 1
    cap.release()


def color_for(obj_id):
    # golden-ratio hue spacing gives distinct, stable colors per track id
    h = (obj_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw(frame_rgb, dets, alpha, draw_boxes):
    overlay = frame_rgb.copy()
    for d in dets:
        overlay[d["mask"]] = d["color"]
    out = cv2.addWeighted(overlay, alpha, frame_rgb, 1 - alpha, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(out.shape[:2]) / 1100)
    thick = max(1, int(round(scale * 2)))
    for d in dets:
        contours, _ = cv2.findContours(d["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, d["color"], thick)
        x1, y1, x2, y2 = map(int, d["box"])
        if draw_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), d["color"], thick)
        label = f'{d["class"]} #{d["id"]} {d["score"]:.2f}'
        (tw, th), base = cv2.getTextSize(label, font, scale, thick)
        ty = y1 - 4 if y1 - th - base - 4 > 0 else y1 + th + base + 4
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 4, ty + base // 2), d["color"], -1)
        cv2.putText(out, label, (x1 + 2, ty - base // 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def counts_banner(img, counts):
    text = "  ".join(f"{c}: {n}" for c, n in counts.items())
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.5, img.shape[0] / 1300)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 16, th + base + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (8, th + 6), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


class FfmpegWriter:
    """Pipe RGB frames to ffmpeg -> H.264/yuv420p mp4 (plays in browsers and players)."""

    def __init__(self, path, width, height, fps):
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, rgb):
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed: " + self.proc.stderr.read().decode(errors="replace")[-500:])

    def abort(self):
        self.proc.kill()
        self.proc.wait()
        Path(self.path).unlink(missing_ok=True)


class Sam3Tracker:
    def __init__(self, token=None, device=None, dtype="fp16"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.model = Sam3VideoModel.from_pretrained(MODEL_ID, token=token).to(self.device, dtype=self.dtype).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(MODEL_ID, token=token)

    def run(self, video_path, targets: list[Target], out_dir, opts: Options = Options(),
            on_progress=lambda p: None, should_cancel=lambda: False) -> dict:
        """Track `targets` in the video and write tracked.mp4 + tracks.json into out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = video_meta(video_path)
        first, last, stride, n_total = plan_frames(meta, opts)
        if n_total == 0:
            raise ValueError("the selected time range contains no frames")

        # SAM3 dedupes identical prompt text, so several labels may share one prompt
        prompt_to_label = {}
        for t in targets:
            prompt_to_label.setdefault(t.prompt, t.label)
        prompts = list(prompt_to_label)
        labels = list(dict.fromkeys(t.label for t in targets))
        out_fps = meta["fps"] / stride
        frame_iter = read_frames(video_path, first, last, stride, n_total)

        if opts.stream:
            session = self.processor.init_video_session(
                inference_device=self.device, video_storage_device="cpu", dtype=self.dtype)
            session = self.processor.add_text_prompt(session, prompts)

            def results():
                for src_idx, rgb in frame_iter:
                    inputs = self.processor(images=rgb, device=self.device, return_tensors="pt")
                    with torch.inference_mode():
                        out = self.model(inference_session=session, frame=inputs.pixel_values[0])
                    yield src_idx, rgb, self.processor.postprocess_outputs(
                        session, out, original_sizes=inputs.original_sizes)
        else:
            on_progress(Progress("reading", 0, n_total))
            frames = []
            for f in frame_iter:
                frames.append(f)
                if len(frames) % 20 == 0:
                    if should_cancel():
                        raise Cancelled()
                    on_progress(Progress("reading", len(frames), n_total))
            n_total = len(frames)
            on_progress(Progress("preparing", 0, n_total))
            src_indices = [i for i, _ in frames]
            session = self.processor.init_video_session(
                video=[f for _, f in frames], inference_device=self.device,
                video_storage_device="cpu", processing_device="cpu", dtype=self.dtype)
            session = self.processor.add_text_prompt(session, prompts)

            def results():
                with torch.inference_mode():
                    for out in self.model.propagate_in_video_iterator(session):
                        i = out.frame_idx
                        yield src_indices[i], frames[i][1], self.processor.postprocess_outputs(session, out)
                        frames[i] = (src_indices[i], None)  # free the frame once rendered

        video_out = out_dir / "tracked.mp4"
        writer = None
        frames_json, seen = [], {lab: set() for lab in labels}
        t0, n = time.time(), 0
        try:
            for src_idx, rgb, res in results():
                if should_cancel():
                    raise Cancelled()
                obj_to_label = {oid: prompt_to_label.get(p, p)
                                for p, ids in res["prompt_to_obj_ids"].items() for oid in ids}
                masks = res["masks"].cpu().numpy()
                dets = []
                for k, oid in enumerate(res["object_ids"].tolist()):
                    score = float(res["scores"][k])
                    if score < opts.score_thresh:
                        continue
                    lab = obj_to_label.get(oid, "?")
                    seen.setdefault(lab, set()).add(oid)
                    dets.append({"id": oid, "class": lab, "score": score, "box": res["boxes"][k].tolist(),
                                 "mask": masks[k], "color": color_for(oid)})

                vis = draw(rgb, dets, opts.alpha, opts.boxes)
                counts = {lab: sum(d["class"] == lab for d in dets) for lab in labels}
                counts_banner(vis, counts)
                if writer is None:
                    writer = FfmpegWriter(video_out, vis.shape[1], vis.shape[0], out_fps)
                writer.write(vis)
                frames_json.append({
                    "frame": src_idx, "time": round(src_idx / meta["fps"], 3),
                    "objects": [{"id": d["id"], "class": d["class"], "score": round(d["score"], 4),
                                 "box": [round(v, 1) for v in d["box"]]} for d in dets]})
                n += 1
                on_progress(Progress("tracking", n, n_total, vis, counts))
            if writer is None:
                raise ValueError("no frames were processed")
            on_progress(Progress("encoding", n, n_total))
            writer.close()
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        finally:
            del session
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = {
            "model": MODEL_ID,
            "targets": [{"label": t.label, "description": t.description, "prompt": t.prompt} for t in targets],
            "mode": "stream" if opts.stream else "offline",
            "range_seconds": [round(first / meta["fps"], 2), round(last / meta["fps"], 2)],
            "stride": stride, "output_fps": round(out_fps, 3),
            "frames_processed": n, "seconds": round(time.time() - t0, 1),
            "unique_tracks_per_class": {lab: len(ids) for lab, ids in seen.items()},
            "source": meta,
            "frames": frames_json,
        }
        (out_dir / "tracks.json").write_text(json.dumps(summary, indent=1))
        return summary
