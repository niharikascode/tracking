# SAM 3 Studio

A web app for annotating photos and videos on your own GPU, organised into **projects**:

- **Object detection.** You get boxes and labels from a vision-language model you pick: Qwen, DeepSeek, Gemini, GPT-6 Sol or GPT-6 Astra, all through OpenRouter. Spending is capped.
- **Segmentation.** SAM 3 runs on the local GPU and draws pixel-exact masks. Videos are tracked frame by frame. It's free.

## Using it

1. Create a project.
2. Drag in as many photos and videos as you like.
3. Select the items you want with the ✓ on each tile, or use *Select all*.
4. In **Annotate**:
   - pick **Object detection** or **Segmentation**;
   - add labels, each with an optional description such as `person` + `wearing a red jacket`;
   - for detection, pick a model;
   - press the button.
5. Tiles update live. Click a tile to open the viewer, which has:
   - the result or the original, with live progress while it runs;
   - every run on that item;
   - downloads: the annotated image or video, JSON, and a mask PNG for segmented photos.
6. If a result isn't right, use **Adjust & re-run**. It loads that run's labels and settings so you can change them and run again.
7. **Export all (ZIP)** gives you, for every finished run, the annotated media and JSON, plus `coco_detection.json` and `coco_segmentation.json` (with polygons) for the photos.

### How descriptions are used

Each label and its description are merged into one phrase, which is shown under the row as "Looks for:". The phrase goes to SAM 3, or to the detection model as one of the object types to find.

| label | description | phrase |
|---|---|---|
| person | wearing a red jacket | `person wearing a red jacket` |
| car | red | `red car` |

Results are still labelled with the plain label.

## Detection models

| Option | OpenRouter model | Price (in / out per 1M tokens) | Notes |
|---|---|---|---|
| Qwen | `qwen/qwen3.8-flash` | $0.15 / $0.47 | Cheap, and it handles crowds well (41 people in the market test photo) |
| Qwen3-VL 8B | `qwen/qwen3-vl-8b-instruct` | $0.25 / $0.75 | Small and cheap; unreliable on crowded scenes |
| Qwen3-VL 30B-A3B | `qwen/qwen3-vl-30b-a3b-instruct` | $0.15 / $0.60 | Cheap; can miss objects or loop on crowded scenes. OpenRouter has no Qwen3-VL 4B |
| DeepSeek | `deepseek/deepseek-v4.1-flash` | $0.02 / $0.72 | The cheapest, at about $0.0005 per photo. Its "thinking" is turned off |
| Gemini | `google/gemini-3.8-flash` | $0.75 / $3.75 | Gemini is asked for its native `[ymin, xmin, ymax, xmax]` box order |
| Sol | `openai/gpt-6-sol` | $2 / $10 | |
| Astra | `openai/gpt-6-astra` | $10 / $50 | The most expensive, at about $0.02–0.05 per photo |

Every model answers with JSON boxes in 0–1000 coordinates. The parser also accepts code fences, 0–1 fractions, `bbox_2d` keys, and repeated keys inside one object; some Qwen providers produce the last of these. A reply that can't be parsed is retried once.

For a video, the model checks frames at a steady rate (the "model checks per second" setting, default 1 per second), from the first frame to the last. The result is the **full video at its normal speed**:
- **Between checks:** each box moves smoothly from one check to the next, matched to the same object by position.
- **Objects seen in only one check:** shown just around that check.
- **Glitched checks are skipped:** answers that loop or come back suddenly empty are thrown out, and so are checks with a count far above or below the others.
- **Fast motion:** use more checks per second; each check is one paid call.

**Label colours:** every label has a colour picker. Boxes and tags use that colour; segmentation masks use shades of it, so neighbouring objects stay distinguishable.

### Spending limits

- **Daily cap:** `DETECT_DAILY_USD`, default **$2.00/day**, resetting at UTC midnight.
- **Overall cap:** `DETECT_TOTAL_USD`, default **$20.00**.
- **Real costs:** spend is counted from the actual cost OpenRouter reports for each call. It's kept in `data/usage.json` and shown in the page header.
- **Checked before starting:** before a batch starts, its estimated cost is compared with what's left. A batch that doesn't fit is refused with a clear message, and each call is checked again as it runs.
- **Calls per video:** capped by `DETECT_MAX_FRAMES` (default 120). Output is limited to 4096 tokens per call. Rate-limited calls (429) and server errors are retried with backoff.
- **The key's own limit:** the OpenRouter key also has its own credit limit, which OpenRouter enforces.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env      # set HF_TOKEN, OPENROUTER_API_KEY, APP_PASSWORD
```

`kernels` must be installed. Without it, SAM 3 skips mask cleanup and stray pixels produce huge bogus boxes.

## Configuration (`.env`)

| var | default | |
|---|---|---|
| `HF_TOKEN` | – | required (gated `facebook/sam3`) |
| `OPENROUTER_API_KEY` | – | enables object detection |
| `APP_PASSWORD` | empty | the access key; empty means open to anyone |
| `DETECT_DAILY_USD` / `DETECT_TOTAL_USD` | 2.00 / 20.00 | detection spending caps |
| `DETECT_MAX_FRAMES` | 120 | maximum model calls (checks) per video in detection |
| `DETECT_WORKERS` | 3 | detection calls running in parallel |
| `MAX_UPLOAD_MB` | 1024 | per-file upload limit |
| `MAX_FRAMES_LIMIT` | 600 | maximum frames tracked per video in segmentation |
| `MAX_QUEUE` | 100 | maximum queued GPU runs |
| `DTYPE` | fp16 | `fp16`, `fp32` or `bf16` |

**Access:** the access key is checked once at `/api/login`. After that, the browser holds an HttpOnly cookie (a hash of the key), so the key never appears in links.

## Run locally

```bash
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 7860
```

## Permanent deployment (systemd + Tailscale Funnel)

The live site runs this way at a fixed, free URL, `https://sam3-tracker.<tailnet>.ts.net`. There's no domain to buy and no upload size cap. Everything runs on this machine's GPU; Tailscale only forwards the web traffic.

1. **App as a user service.** It restarts on crash and starts at boot if lingering is enabled (`loginctl enable-linger $USER`):
   ```bash
   cp deploy/sam3-tracker.service ~/.config/systemd/user/   # edit the WorkingDirectory/ExecStart paths
   systemctl --user daemon-reload && systemctl --user enable --now sam3-tracker
   ```
2. **Tailscale in userspace mode.** This needs no root. Put the `tailscale` and `tailscaled` binaries from https://pkgs.tailscale.com/ in `~/bin`, then:
   ```bash
   cp deploy/tailscaled.service ~/.config/systemd/user/
   systemctl --user daemon-reload && systemctl --user enable --now tailscaled
   alias ts="$HOME/bin/tailscale --socket=$HOME/.local/share/tailscale/tailscaled.sock"
   ts up --hostname=sam3-tracker           # open the printed login link and sign in
   ```
   In the Tailscale admin console, turn on **MagicDNS** and **HTTPS certificates** (DNS page).
3. **Expose it.** The first run prints a link to enable Funnel for the tailnet:
   ```bash
   ts funnel --bg 7860
   ```

**Use Tailscale 1.105.59 (unstable) or newer.** Stable 1.104.1 has an open bug ([tailscale#21796](https://github.com/tailscale/tailscale/issues/21796)): Funnel connections that arrive over IPv6 are dropped with `Drop: TCP{...} no rules matched`, so most outside visitors can't connect.

Useful commands: `systemctl --user status sam3-tracker`, `journalctl --user -u sam3-tracker -f`, `ts funnel --https=443 off` (take the site offline).

## How it works

- `server.py`: FastAPI. It handles projects, items and runs, uploads, the budget, the access key and the ZIP/COCO export. One GPU worker thread runs segmentation one job at a time; a small thread pool runs detection calls. State is stored as JSON under `data/projects/` and survives restarts.
- `pipelines.py`: the four pipelines (detect or segment × photo or video) and the spending budget.
- `detector.py`: OpenRouter calls, prompts, the per-model box format, and tolerant JSON parsing.
- `tracker.py`: the SAM 3 engine (video tracking, single-image segmentation, mask polygons, H.264 output).
- `static/`: the page (plain HTML, CSS and JS, no build step).
- `track_video.py`: the SAM 3 tracking engine from the command line.

## Speed (Tesla T4)

- **Photo segmentation:** about 2–5 s.
- **Video tracking:** about 1–7.5 s per frame, depending on how many objects there are.
- **Detection:** about 2–30 s per photo, depending on the model.
