"""
detect_track_ttc.py — Day 1 integration demo.

- Reads frames from a camera (USB)
- Detects with YOLOv8n, tracks with ByteTrack (ultralytics built-in)
- Computes TTC and overlays results
- Saves the demo video as mp4
- Prints PyTorch baseline latency/FPS statistics on exit

Usage:
  python -m app.detect_track_ttc --source 0
  python -m app.detect_track_ttc --source 0 --record poster/demo_video/day1.mp4
  python -m app.detect_track_ttc --source data/sample.mp4 --no-show
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ensure the capstone1 root is on sys.path when running as a script
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.overlay import (  # noqa: E402
    TrackTrails, draw_box_with_ttc, draw_critical_banner, draw_hud, draw_trail,
)
from app.ttc import TTCEstimator, classify_ttc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8n + ByteTrack + TTC demo")
    p.add_argument("--model", default="yolov8n.pt",
                   help="ultralytics weights path or model name")
    p.add_argument("--source", default="0",
                   help="camera index (e.g. '0') or video file path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0", help="0,1,... or cpu")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="ultralytics built-in tracker config (bytetrack.yaml / botsort.yaml)")
    p.add_argument("--record", default=None,
                   help="output mp4 path. if set, video is recorded.")
    p.add_argument("--record-fps", type=float, default=0.0,
                   help="force output video fps. 0 uses source file fps "
                        "(file input) or measured fps (camera).")
    p.add_argument("--no-show", action="store_true",
                   help="disable GUI (for headless servers)")
    p.add_argument("--ttc-debug", default=None,
                   help="if set, saves per-track TTC state to CSV every frame")
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 for unlimited (run until manually stopped)")
    p.add_argument("--warmup", type=int, default=10,
                   help="number of initial frames to exclude from baseline statistics")
    return p.parse_args()


def _open_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, max(fps, 1.0), (w, h))


def _resolve_source(s: str):
    if s.isdigit():
        return int(s)
    return s


def main() -> int:
    args = parse_args()

    # ultralytics import triggers GPU warm-up, so defer until after arg parsing
    from ultralytics import YOLO  # noqa: WPS433

    print(f"[init] loading model: {args.model}")
    model = YOLO(args.model)

    # model.track uses generator interface (stream=True)
    # ultralytics manages cv2.VideoCapture / VideoStream internally
    source = _resolve_source(args.source)
    print(f"[init] source = {source!r}, tracker = {args.tracker}")

    # probe source fps before ultralytics opens it, so the writer's
    # fps metadata matches the original.
    # for camera (int) sources, leave as None and use measured fps.
    src_fps: Optional[float] = None
    if isinstance(source, str) and Path(source).is_file():
        _probe = cv2.VideoCapture(source)
        if _probe.isOpened():
            v = _probe.get(cv2.CAP_PROP_FPS)
            if v and v > 1:
                src_fps = float(v)
        _probe.release()
        if src_fps:
            print(f"[init] source fps from file = {src_fps:.2f}")

    ttc_est = TTCEstimator(alpha=0.3, stale_after_s=1.0)
    trails = TrackTrails(max_len=30)

    # per-frame inference time log (PyTorch baseline measurement)
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    fps_window: collections.deque[float] = collections.deque(maxlen=30)

    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    last_t = time.perf_counter()

    # TTC debug CSV
    debug_csv = None
    debug_writer = None
    if args.ttc_debug:
        Path(args.ttc_debug).parent.mkdir(parents=True, exist_ok=True)
        debug_csv = open(args.ttc_debug, "w", newline="")
        debug_writer = csv.writer(debug_csv)
        debug_writer.writerow([
            "frame", "t_sec", "track_id", "class",
            "x1", "y1", "x2", "y2",
            "scale", "growth_ema", "ttc", "level", "n_updates",
        ])
        print(f"[ttc-debug] logging to {args.ttc_debug}")

    # ultralytics track stream
    stream = model.track(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
    )

    try:
        for result in stream:
            # Results object for one frame
            frame = result.orig_img.copy()  # H, W, 3 BGR
            h, w = frame.shape[:2]

            # inference timing filled by ultralytics (ms)
            speed = getattr(result, "speed", None)  # {'preprocess', 'inference', 'postprocess'}
            if speed and "inference" in speed:
                inference_ms.append(float(speed["inference"]))

            # tracking results
            active_ids: set[int] = set()
            frame_levels: dict[int, str] = {}  # per-track classification for this frame
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                xyxys = result.boxes.xyxy.cpu().numpy()
                clss = result.boxes.cls.int().cpu().tolist()
                names = result.names

                for tid, xyxy, cls_idx in zip(ids, xyxys, clss):
                    active_ids.add(tid)
                    x1, y1, x2, y2 = xyxy.tolist()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    trails.add(tid, (cx, cy))

                    ttc = ttc_est.update(tid, (x1, y1, x2, y2))
                    # stable classification: hysteresis + persistence (v2)
                    level = ttc_est.classify(tid)
                    frame_levels[tid] = level
                    label = names.get(cls_idx, str(cls_idx))

                    draw_trail(frame, trails.get(tid))
                    draw_box_with_ttc(frame, (x1, y1, x2, y2), label, tid, ttc, level)

                    # debug CSV
                    if debug_writer is not None:
                        st = ttc_est.debug_state(tid)
                        if st is not None:
                            debug_writer.writerow([
                                frame_idx,
                                f"{frame_idx / max(src_fps or 30.0, 1.0):.4f}",
                                tid, label,
                                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                                f"{st['scale']:.3f}",
                                f"{st['growth_ema']:.4f}",
                                f"{st['ttc']:.4f}",
                                level,
                                st["n_updates"],
                            ])

            ttc_est.prune(active_ids=active_ids)
            trails.prune(active_ids=active_ids)

            # danger banner if any track is critical this frame
            if any(L == "critical" for L in frame_levels.values()):
                draw_critical_banner(frame)

            # end-to-end frame time → FPS
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            end_to_end_ms.append(dt * 1000.0)
            fps_window.append(1.0 / dt if dt > 0 else 0.0)
            fps = sum(fps_window) / len(fps_window)

            draw_hud(frame, fps=fps, backend="PyTorch FP32",
                     extra=f"frame {frame_idx} | tracks: {len(active_ids)}")

            # create writer on first frame once resolution is known
            if args.record and writer is None:
                # priority: --record-fps explicit value > source file fps > measured fps
                if args.record_fps > 0:
                    target_fps = args.record_fps
                    fps_src = "manual"
                elif src_fps is not None:
                    target_fps = src_fps
                    fps_src = "source"
                else:
                    target_fps = max(fps, 15.0)
                    fps_src = "measured"
                writer = _open_writer(args.record, w, h, fps=target_fps)
                print(f"[rec] writing to {args.record} @ {target_fps:.2f} fps ({fps_src})")
            if writer is not None:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("detect+track+TTC", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[exit] q/ESC pressed")
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[exit] reached --max-frames {args.max_frames}")
                break

    except KeyboardInterrupt:
        print("[exit] KeyboardInterrupt")
    finally:
        if writer is not None:
            writer.release()
        if debug_csv is not None:
            debug_csv.close()
        if not args.no_show:
            cv2.destroyAllWindows()

    # ── PyTorch baseline measurement summary ─────────────────────────
    print("\n" + "=" * 60)
    print("PyTorch baseline (Day 1 first measurement)")
    print("=" * 60)
    print(f"  total frames        : {frame_idx}")
    if len(inference_ms) > args.warmup + 10:
        warm = inference_ms[args.warmup:]
        warm_sorted = sorted(warm)
        n = len(warm_sorted)
        p50 = warm_sorted[n // 2]
        p95 = warm_sorted[int(n * 0.95)]
        p99 = warm_sorted[min(n - 1, int(n * 0.99))]
        print(f"  inference (ms)      : mean={statistics.mean(warm):.2f}  "
              f"p50={p50:.2f}  p95={p95:.2f}  p99={p99:.2f}  (n={n})")
    else:
        print("  inference samples too few for stats")
    if end_to_end_ms[args.warmup:]:
        e2e = end_to_end_ms[args.warmup:]
        mean_e2e = statistics.mean(e2e)
        print(f"  end-to-end (ms)     : mean={mean_e2e:.2f}  "
              f"FPS={1000.0 / mean_e2e:.2f}  (n={len(e2e)})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
