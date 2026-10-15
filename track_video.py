"""Command-line SAM 3 tracking (same engine as the web app).

  .venv/bin/python track_video.py --video in.mp4 --target person "wearing a red jacket" --target bag -o outputs/run1

Writes outputs/run1/tracked.mp4 and outputs/run1/tracks.json.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tracker import Options, Sam3Tracker, Target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--target", nargs="+", action="append", required=True, metavar=("LABEL", "DESCRIPTION"),
                   help='a label, optionally followed by a description; repeat for more labels')
    p.add_argument("-o", "--output", default=None, help="output directory (default: outputs/<video stem>)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=150, help="0 = every frame (watch memory)")
    p.add_argument("--stride", type=int, default=0, help="process every Nth frame (0 = auto from --max-frames)")
    p.add_argument("--score-thresh", type=float, default=0.0)
    p.add_argument("--no-boxes", action="store_true")
    p.add_argument("--stream", action="store_true", help="constant-memory streaming mode for long videos")
    p.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16")
    args = p.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    targets = [Target(t[0], " ".join(t[1:])) for t in args.target]
    out = Path(args.output or f"outputs/{Path(args.video).stem}")
    opts = Options(start=args.start, end=args.end, stride=args.stride, max_frames=args.max_frames,
                   score_thresh=args.score_thresh, boxes=not args.no_boxes, stream=args.stream)

    for t in targets:
        print(f"[sam3] {t.label!r:>16} -> prompt {t.prompt!r}")
    tracker = Sam3Tracker(token=os.getenv("HF_TOKEN"), dtype=args.dtype)

    def on_progress(pr):
        if pr.stage == "tracking":
            print(f"\r[sam3] tracking {pr.done}/{pr.total}  {pr.counts}", end="", file=sys.stderr, flush=True)

    summary = tracker.run(args.video, targets, out, opts, on_progress=on_progress)
    print(f"\n[sam3] done: {summary['frames_processed']} frames in {summary['seconds']}s, "
          f"unique tracks {summary['unique_tracks_per_class']}")
    print(f"[sam3] {out / 'tracked.mp4'}\n[sam3] {out / 'tracks.json'}")


if __name__ == "__main__":
    main()
