"""Object detection with vision-language models through OpenRouter.

Each model gets the image and the label list and must answer with JSON boxes in
0-1000 normalised coordinates. OpenRouter reports the real cost of every call,
which the server uses to enforce a daily spending cap.
"""

import base64
import io
import json
import re
import time

import httpx
from PIL import Image

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The only models offered. Prices are USD per million tokens (input, output),
# from OpenRouter's model list; real per-call cost comes back in the response.
MODELS = {
    # qwen3-vl-235b put whole crowds in one box in testing; 3.8 Flash boxes each person
    "qwen": {"id": "qwen/qwen3.8-flash", "name": "Qwen 3.8 Flash",
             "price": (0.15, 0.47), "note": "Cheap, good with crowds"},
    # OpenRouter has no Qwen3-VL 4B; 8B and 30B-A3B are its smallest Qwen3-VL models
    "qwen-vl-8b": {"id": "qwen/qwen3-vl-8b-instruct", "name": "Qwen3-VL 8B",
                   "price": (0.25, 0.75), "note": "Small and cheap; unreliable on crowded scenes"},
    "qwen-vl-30b": {"id": "qwen/qwen3-vl-30b-a3b-instruct", "name": "Qwen3-VL 30B-A3B",
                    "price": (0.15, 0.60), "note": "Cheap; can miss or loop on crowded scenes"},
    "deepseek": {"id": "deepseek/deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash",
                 "price": (0.02, 0.72), "note": "Cheapest",
                 "extra": {"reasoning": {"enabled": False}}},
    # Gemini is trained on [ymin, xmin, ymax, xmax]; asking for x-first gives swapped boxes
    "gemini": {"id": "google/gemini-3.8-flash", "name": "Gemini 3.8 Flash",
               "price": (0.75, 3.75), "note": "Fast, good all-rounder", "order": "yxyx"},
    "sol": {"id": "openai/gpt-6-sol", "name": "GPT-6 Sol",
            "price": (2.00, 10.00), "note": "OpenAI, mid price", "est_out": 3000},
    "astra": {"id": "openai/gpt-6-astra", "name": "GPT-6 Astra",
              "price": (10.00, 50.00), "note": "OpenAI, most expensive", "est_out": 3000},
}

# Rough per-image token use, for up-front cost estimates (real cost is billed after).
EST_TOKENS_IN, EST_TOKENS_OUT = 2500, 1500
MAX_SIDE = 1280          # images are downscaled to this before sending
MAX_OUTPUT_TOKENS = 4096
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

PROMPT = """You are an object detector. Find every visible instance of these object types in the image:
{targets}

Answer with JSON only, in exactly this form:
{{"objects": [{{"label": "<one of the object types above, copied exactly>", "box": {box_form}}}]}}

Rules:
- {box_rule}
- Coordinates are integers from 0 to 1000, relative to the image: x from left (0) to right (1000), y from top (0) to bottom (1000).
- One entry per separate object instance; do not merge a group into one box.
- List each object once. Stop when every visible object is listed.
- Only use the listed object types. If none are visible, return {{"objects": []}}."""


BOX_FORMS = {
    "xyxy": ("[x1, y1, x2, y2]", "box = [x1, y1, x2, y2]: left, top, right, bottom edges of a tight box around the object."),
    "yxyx": ("[ymin, xmin, ymax, xmax]", "box = [ymin, xmin, ymax, xmax]: top, left, bottom, right edges of a tight box around the object."),
}


class DetectionError(Exception):
    pass


def estimate_cost(model_key: str, n_images: int) -> float:
    cfg = MODELS[model_key]
    pin, pout = cfg["price"]
    return n_images * (EST_TOKENS_IN * pin + cfg.get("est_out", EST_TOKENS_OUT) * pout) / 1e6


def _encode(image: Image.Image) -> str:
    img = image.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- parsing

def _pairs_hook(pairs):
    """Keep objects whose keys repeat ({"label":..,"box":..,"label":..,"box":..}, seen from
    some Qwen providers) as several objects instead of letting later keys overwrite earlier ones."""
    keys = [k for k, _ in pairs]
    if len(keys) == len(set(keys)):
        return dict(pairs)
    groups, cur = [], {}
    for k, v in pairs:
        if k in cur:
            groups.append(cur)
            cur = {}
        cur[k] = v
    groups.append(cur)
    return {"__multi__": groups}


def _loads(text):
    return json.loads(text, object_pairs_hook=_pairs_hook)


_ENTRY_RE = re.compile(
    r'"label"\s*:\s*"([^"]*)"\s*,\s*"(?:box|box_2d|bbox_2d|bbox)"\s*:\s*\[([^\[\]]*)\]'
    r'|"(?:box|box_2d|bbox_2d|bbox)"\s*:\s*\[([^\[\]]*)\]\s*,\s*"label"\s*:\s*"([^"]*)"')


def _salvage(text: str):
    """Pull complete label/box pairs out of an answer that was cut off or is not valid JSON."""
    objs = []
    for m in _ENTRY_RE.finditer(text):
        label, nums = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(4), m.group(3))
        try:
            box = [float(v) for v in nums.split(",")]
        except ValueError:
            continue
        if len(box) == 4:
            objs.append({"label": label, "box": box})
    return {"objects": objs} if objs else None


def _extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return _loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return _loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    salvaged = _salvage(text)
    if salvaged is not None:
        return salvaged
    raise DetectionError("The model's answer was not valid JSON")


def _match_label(raw: str, targets: list[dict]) -> str | None:
    """Map the model's label back to one of the user's labels."""
    raw_l = raw.strip().lower()
    for t in targets:
        if raw_l in (t["label"].lower(), t["prompt"].lower()):
            return t["label"]
    for t in targets:  # loose match, e.g. "red car" -> "car"
        if t["label"].lower() in raw_l or raw_l in t["prompt"].lower():
            return t["label"]
    if len(targets) == 1:
        return targets[0]["label"]
    return None


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _staircase_runs(boxes, min_run=5):
    """Indices inside degenerate runs: consecutive boxes with the same width (or height) that
    march steadily in one direction -- what a looping model emits until its token limit."""
    drop = set()
    for dim, pos in ((2, 0), (3, 1)):          # (width, x) and (height, y)
        i, n = 0, len(boxes)
        while i < n:
            j = i
            while j + 1 < n:
                a, b = boxes[j], boxes[j + 1]
                size = a[dim] - a[pos]
                same = abs(size - (b[dim] - b[pos])) <= 1.5
                step = b[pos] - a[pos]
                # loop boxes overlap their neighbour (step < 60% of the size); real objects in a row don't
                same = same and abs(step) < 0.6 * size
                first_step = boxes[i + 1][pos] - boxes[i][pos] if i + 1 < n else 0
                if not same or step == 0 or (first_step and step * first_step < 0):
                    break
                j += 1
            if j - i + 1 >= min_run:
                drop.update(range(i + 1, j + 1))
            i = j + 1 if j > i else i + 1
    return drop


def _drop_loops(boxes: list[list[float]]) -> list[int]:
    """Indices to keep after removing looping 'staircase' runs and near-duplicate boxes."""
    drop = _staircase_runs(boxes)
    keep = []
    for idx, b in enumerate(boxes):
        if idx in drop:
            continue
        if any(_iou(b, boxes[k]) > 0.8 for k in keep):   # duplicates / near-duplicates
            continue
        keep.append(idx)
    return keep


def _parse_objects(data, targets, width, height, order="xyxy"):
    items = data.get("objects", data.get("detections", [])) if isinstance(data, dict) else data
    flat = []
    for it in items if isinstance(items, list) else []:
        flat.extend(it["__multi__"] if isinstance(it, dict) and "__multi__" in it else [it])
    parsed = []
    for it in flat:
        if not isinstance(it, dict):
            continue
        box = it.get("box") or it.get("box_2d") or it.get("bbox_2d") or it.get("bbox")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        try:
            a, b, c, d = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = (b, a, d, c) if order == "yxyx" else (a, b, c, d)
        # accept 0-1 fractions too
        scale = 1.0 if max(x1, y1, x2, y2) <= 1.0 else 1000.0
        x1, x2 = sorted((min(max(x1 / scale, 0), 1), min(max(x2 / scale, 0), 1)))
        y1, y2 = sorted((min(max(y1 / scale, 0), 1), min(max(y2 / scale, 0), 1)))
        if x2 - x1 < 0.002 or y2 - y1 < 0.002:
            continue
        label = _match_label(str(it.get("label", "")), targets)
        if label is None:
            continue
        parsed.append((label, [x1, y1, x2, y2]))
    keep = _drop_loops([[v * 1000 for v in b] for _, b in parsed])
    return [{"label": parsed[k][0], "box": [round(parsed[k][1][0] * width, 1), round(parsed[k][1][1] * height, 1),
                                            round(parsed[k][1][2] * width, 1), round(parsed[k][1][3] * height, 1)]}
            for k in keep]


# ----------------------------------------------------------------------------- calls

def detect(image: Image.Image, targets: list[dict], model_key: str, api_key: str, timeout=180) -> dict:
    """Detect `targets` ([{label, prompt}]) in a PIL image. Returns objects, usage, cost, seconds."""
    if model_key not in MODELS:
        raise DetectionError(f"Unknown model '{model_key}'")
    cfg = MODELS[model_key]
    order = cfg.get("order", "xyxy")
    box_form, box_rule = BOX_FORMS[order]
    listing = "\n".join(f"- {t['prompt']}" for t in targets)
    body = {
        "model": cfg["id"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(targets=listing, box_form=box_form, box_rule=box_rule)},
            {"type": "image_url", "image_url": {"url": _encode(image)}},
        ]}],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "usage": {"include": True},
        **cfg.get("extra", {}),
    }
    headers = {"Authorization": f"Bearer {api_key}", "X-Title": "SAM 3 Studio"}
    t0 = time.time()
    total = {"cost": 0.0, "in": 0, "out": 0}
    for attempt in range(2):  # one retry when the answer can't be parsed
        content = _call(body, headers, cfg["name"], timeout, total)
        try:
            objects = _parse_objects(_extract_json(content), targets, image.width, image.height, order)
            break
        except DetectionError:
            if attempt == 1:
                raise
    return {"objects": objects, "cost": total["cost"], "seconds": round(time.time() - t0, 1),
            "tokens_in": total["in"], "tokens_out": total["out"]}


def _call(body, headers, name, timeout, total, attempts=4) -> str:
    """POST to OpenRouter, retrying rate limits / transient errors with backoff."""
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            r = httpx.post(OPENROUTER_URL, json=body, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            if last:
                raise DetectionError(f"Could not reach OpenRouter: {e}") from e
            time.sleep(2 * 2 ** attempt)
            continue
        if r.status_code in RETRY_STATUS and not last:
            wait = r.headers.get("retry-after")
            time.sleep(min(float(wait), 30) if wait and wait.replace(".", "", 1).isdigit() else 2 * 2 ** attempt)
            continue
        if r.status_code != 200:
            try:
                msg = r.json().get("error", {}).get("message", r.text[:300])
            except ValueError:
                msg = r.text[:300]
            hint = " (the provider is rate-limiting; try again in a minute)" if r.status_code == 429 else ""
            raise DetectionError(f"{name} request failed ({r.status_code}): {msg}{hint}")
        data = r.json()
        if data.get("error"):
            if not last and data["error"].get("code") in RETRY_STATUS:
                time.sleep(2 * 2 ** attempt)
                continue
            raise DetectionError(f"{name}: {data['error'].get('message', data['error'])}")
        usage = data.get("usage") or {}
        total["cost"] += float(usage.get("cost") or 0.0)
        total["in"] += usage.get("prompt_tokens") or 0
        total["out"] += usage.get("completion_tokens") or 0
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise DetectionError(f"Empty answer from {name}")
    raise DetectionError(f"{name}: no answer after {attempts} attempts")
