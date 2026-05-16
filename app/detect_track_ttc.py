"""
detect_track_ttc.py — Day 1 통합 데모.

- 카메라(USB)에서 프레임을 받아
- YOLOv8n 으로 검출, ByteTrack 으로 추적 (ultralytics 내장)
- TTC 를 계산해 오버레이
- 데모 영상을 mp4 로 저장
- 종료 시 PyTorch baseline latency/FPS 통계 출력

사용 예:
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

# 같은 capstone1 폴더에서 실행되도록 sys.path 보정
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
                   help="ultralytics 가중치 경로 또는 이름")
    p.add_argument("--source", default="0",
                   help="카메라 index (예: '0') 또는 비디오 파일 경로")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0", help="0,1,... 또는 cpu")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="ultralytics 내장 트래커 설정 (bytetrack.yaml / botsort.yaml)")
    p.add_argument("--record", default=None,
                   help="저장할 mp4 경로. 지정하면 영상으로 기록.")
    p.add_argument("--record-fps", type=float, default=0.0,
                   help="저장 영상 fps 강제값. 0이면 입력 영상의 fps 사용 "
                        "(파일 입력) 또는 측정 fps 사용(카메라).")
    p.add_argument("--no-show", action="store_true",
                   help="GUI 미사용 (헤드리스 서버용)")
    p.add_argument("--ttc-debug", default=None,
                   help="지정하면 매 프레임 트랙별 TTC 상태를 CSV로 저장")
    p.add_argument("--max-frames", type=int, default=0,
                   help="0이면 무제한 (수동 종료까지)")
    p.add_argument("--warmup", type=int, default=10,
                   help="베이스라인 통계에서 제외할 초기 프레임 수")
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

    # ultralytics import 는 GPU 워밍업을 유발하므로 인자 파싱 후에.
    from ultralytics import YOLO  # noqa: WPS433

    print(f"[init] loading model: {args.model}")
    model = YOLO(args.model)

    # model.track 은 generator 인터페이스 (stream=True)
    # ultralytics 가 내부적으로 cv2.VideoCapture / VideoStream 을 관리.
    source = _resolve_source(args.source)
    print(f"[init] source = {source!r}, tracker = {args.tracker}")

    # ultralytics 가 영상 파일을 열기 전에, 원본 fps 를 우리가 직접 한 번 읽는다.
    # writer 의 fps 메타데이터를 원본과 맞추기 위함이다.
    # 카메라(int)일 때는 None 으로 두고, 실측 fps 를 쓴다.
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

    # 프레임별 추론 시간 기록 (PyTorch baseline 1차 측정)
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    fps_window: collections.deque[float] = collections.deque(maxlen=30)

    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    last_t = time.perf_counter()

    # TTC 디버그 CSV
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
            # 한 프레임에 해당하는 Results 객체
            frame = result.orig_img.copy()  # H, W, 3 BGR
            h, w = frame.shape[:2]

            # ultralytics 가 채워주는 추론 시간 (ms)
            speed = getattr(result, "speed", None)  # {'preprocess', 'inference', 'postprocess'}
            if speed and "inference" in speed:
                inference_ms.append(float(speed["inference"]))

            # 트래킹 결과
            active_ids: set[int] = set()
            frame_levels: dict[int, str] = {}  # 이번 프레임 트랙별 분류
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
                    # 안정 분류: 히스테리시스 + 지속성 적용 (v2)
                    level = ttc_est.classify(tid)
                    frame_levels[tid] = level
                    label = names.get(cls_idx, str(cls_idx))

                    draw_trail(frame, trails.get(tid))
                    draw_box_with_ttc(frame, (x1, y1, x2, y2), label, tid, ttc, level)

                    # 디버그 CSV
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

            # 위험 배너 (이번 프레임에 critical 인 트랙이 있으면)
            if any(L == "critical" for L in frame_levels.values()):
                draw_critical_banner(frame)

            # end-to-end 프레임 시간 → FPS
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            end_to_end_ms.append(dt * 1000.0)
            fps_window.append(1.0 / dt if dt > 0 else 0.0)
            fps = sum(fps_window) / len(fps_window)

            draw_hud(frame, fps=fps, backend="PyTorch FP32",
                     extra=f"frame {frame_idx} | tracks: {len(active_ids)}")

            # writer 첫 프레임에서 생성 (해상도 확정)
            if args.record and writer is None:
                # 우선순위: --record-fps 명시값 > 입력 파일 fps > 측정 fps
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

    # ── PyTorch baseline 1차 측정 요약 ─────────────────────────
    print("\n" + "=" * 60)
    print("PyTorch baseline (Day 1 1차 측정)")
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
