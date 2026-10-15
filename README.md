# SAM 3 Tracker

A small web app for text-prompted video tracking with [SAM 3](https://huggingface.co/facebook/sam3), running on your own GPU.

1. Upload a video.
2. Add **labels** (`person`, `bag`, …). For each label you can add an optional **description** (`wearing a red jacket`, `with a shopping bag`).
3. Watch the progress live: a progress bar, time remaining, the frame being tracked and per-label counts.
4. Play the result and download the MP4 or the per-frame tracks as JSON.
5. If the result isn't right, click **Adjust & re-run**, change the labels or descriptions and run again on the same video. Earlier runs stay listed so you can compare them.

## How descriptions are used

SAM 3 takes short noun phrases. Each label and its description are merged into one phrase, and the page shows that phrase under each row before you run:

| label | description | SAM 3 prompt |
|---|---|---|
| person | wearing a red jacket | `person wearing a red jacket` |
| car | red | `red car` |
| car | a red sports car | `a red sports car` |

A description that starts with *with, wearing, holding, carrying, in, on, near, …* goes after the label. Anything else goes before it. If the description already contains the label, it's used as is. Output videos and JSON still use the plain label (`person #3 0.91`).

Keep descriptions to short visual attributes, such as colour, clothing, what the object is holding or where it is. SAM 3 is not a language model, so long sentences or scene context won't help.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env      # then set HF_TOKEN (needs access to the gated facebook/sam3 model)
```

`kernels` must be installed. Without it, SAM 3 skips mask cleanup, and stray pixels produce huge bogus boxes.

## Run locally

```bash
./run.sh            # http://localhost:7860, this machine only
./run.sh --public   # also prints a temporary https://<random>.trycloudflare.com URL (changes on restart)
```

**Before going public, set `APP_PASSWORD` in `.env`.** Anyone with the URL can otherwise use your GPU. When it's set, the page asks for the access key once and remembers it.

## Permanent deployment (systemd + Tailscale Funnel)

The live site runs this way at a fixed, free URL, `https://sam3-tracker.<tailnet>.ts.net`. There's no domain to buy and no upload size cap. The tracking always runs on this machine's GPU; Tailscale only forwards the web traffic.

1. **App as a user service.** It restarts on crash and starts at boot if lingering is enabled (`loginctl enable-linger $USER`):
   ```bash
   cp deploy/sam3-tracker.service ~/.config/systemd/user/   # edit the WorkingDirectory/ExecStart paths
   systemctl --user daemon-reload && systemctl --user enable --now sam3-tracker
   ```
2. **Tailscale in userspace mode.** This needs no root and makes no system-wide network changes. Put the `tailscale` and `tailscaled` binaries from https://pkgs.tailscale.com/stable/ in `~/bin`, then:
   ```bash
   cp deploy/tailscaled.service ~/.config/systemd/user/
   systemctl --user daemon-reload && systemctl --user enable --now tailscaled
   alias ts="$HOME/bin/tailscale --socket=$HOME/.local/share/tailscale/tailscaled.sock"
   ts up --hostname=sam3-tracker           # open the printed login link and sign in
   ```
   In the Tailscale admin console, turn on **MagicDNS** and **HTTPS certificates** (DNS page).
3. **Expose it.** The first run prints a link to enable Funnel for the tailnet. After that, the setting persists across restarts:
   ```bash
   ts funnel --bg 7860
   ts funnel status
   ```

**Use Tailscale 1.105.59 (unstable) or newer.** Stable 1.104.1 has an open bug ([tailscale#21796](https://github.com/tailscale/tailscale/issues/21796)): Funnel connections that arrive over IPv6 are dropped with `Drop: TCP{...} no rules matched` in the tailscaled log, so most outside visitors get a TLS/"broken pipe" error. Get the build from https://pkgs.tailscale.com/unstable/.

Useful commands: `systemctl --user status sam3-tracker`, `journalctl --user -u sam3-tracker -f`, `ts funnel --https=443 off` (take the site offline).

## Configuration (`.env`)

| var | default | |
|---|---|---|
| `HF_TOKEN` | – | required |
| `APP_PASSWORD` | empty | access key required in the browser; empty = open |
| `MAX_UPLOAD_MB` | 1024 | upload size limit |
| `MAX_FRAMES_LIMIT` | 600 | upper bound on the "Max frames" slider |
| `MAX_QUEUE` | 20 | queued runs before new ones are refused |
| `RETENTION_HOURS` | 48 | uploads and results older than this are deleted |
| `DTYPE` | fp16 | `fp16`, `fp32` or `bf16` |

## How it works

- `tracker.py`: the SAM 3 engine. It reads frames, runs `Sam3VideoModel` with all prompts at once, draws masks, contours, boxes and labels, and pipes frames to ffmpeg (H.264).
- `server.py`: FastAPI. One background thread loads the model once and runs jobs **one at a time** from a queue, since the GPU is shared. Other users see their place in the line. Files are stored under `data/`.
- `static/`: the page (plain HTML, CSS and JS, no build step).
- `track_video.py`: the same engine from the command line.

The long-video default processes at most 150 frames. If the range is longer, it uses every Nth frame and the output plays at the reduced fps. Raise "Max frames" for smoother output.

## Speed (Tesla T4)

The cost grows with the number of tracked objects. On a 1270×720 store clip in fp16:

- 1–3 matching objects: about 1 s per frame.
- About 13 objects: about 7.5 s per frame.

The page's time estimates learn from recent runs.

## CLI

```bash
.venv/bin/python track_video.py --video in.mp4 --target person "wearing a red jacket" --target bag -o outputs/run1
```
