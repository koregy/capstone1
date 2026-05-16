# Capstone 1 — On-Device Inference Optimization Infrastructure

> System- and hardware-level post-training optimization of a YOLOv8n object
> detection pipeline on Jetson Orin Nano.

## Overview

Using an ADAS scenario (YOLOv8n + ByteTrack + TTC) as the application,
this project builds a **5-way quantitative comparison infrastructure**
across PyTorch FP32, ONNX Runtime, and TensorRT (FP32 / FP16 / INT8).

- **Application (`app/`)** — Detection, tracking, and TTC demo (for showcase).
- **Infrastructure (`infra/`)** — **The core deliverable.** Backend
  abstraction, automated measurement, and result analysis. Designed to be
  reused as-is in Capstone 2.

## Environment

| Component | Version |
|---|---|
| Hardware | Jetson Orin Nano 8 GB |
| JetPack | 6.1 (L4T R36.4, CUDA 12.6) |
| PyTorch | 2.5.0a0 (NVIDIA Jetson wheel) |
| torchvision | 0.20.0 (built from source) |
| TensorRT | 10.3 |
| ONNX Runtime | 1.19 (with CUDA + TensorRT EP) |
| Camera | USB webcam |

## Repository Layout

```
capstone1/
├── app/                  Application scenario (detect + track + TTC)
├── infra/                ★ Measurement / conversion / analysis (reused in Capstone 2)
├── experiments/          One-off experiment & diagnostic scripts
├── configs/              YAML configs
├── data/                 Input videos (gitignored)
├── models/               Weights (gitignored)
├── results/              Measurement outputs (gitignored)
├── poster/               Demo videos, figures (gitignored)
└── docs/                 Work log
```

## Quick Start

```bash
# 0) Activate venv
source .venv/bin/activate

# 1) Environment check
python experiments/env_check.py

# 2) Detect + track + TTC demo (camera)
python -m app.detect_track_ttc --source 0

# 3) Demo on a video file + record output
python -m app.detect_track_ttc \
    --source data/clips/dashcam.mp4 \
    --record poster/demo_video/day1.mp4 \
    --no-show
```

## Day 1 Results

First-pass PyTorch FP32 baseline measured on a 30 FPS dashcam clip,
about 30 seconds long, with `--no-show`:

| Metric | Value |
|---|---|
| Inference (model only) | mean **25.10 ms** (≈ 40 FPS) |
| End-to-end (decode + infer + visualize) | mean 66.69 ms (≈ 15 FPS) |

> The official baseline will be re-measured in Day 2 using the automated
> benchmark script with dummy tensors, eliminating I/O and visualization
> overhead.

## Data & Model Preparation

Large binaries are excluded from git. Pull them locally:

```bash
# YOLOv8n weights (auto-downloaded by ultralytics on first use)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Demo dashcam clip (example)
pip install yt-dlp
mkdir -p data/clips
yt-dlp -f "best[height<=720][fps<=30][ext=mp4]" \
    -o "data/clips/sample.mp4" "<YouTube URL>"
```

## TTC Algorithm

`app/ttc.py` estimates Time-To-Collision from the rate of change of
bbox scale, with the following stabilization layers (final version):

1. **Scale outlier rejection** — drop frames where bbox scale jumps by
   more than 30 % or 15 px between frames.
2. **Scale EMA** (α = 0.4) and **growth EMA** (α = 0.15) — two-stage
   smoothing for the raw bbox signal and its derivative.
3. **`inst_growth` clamp** at ±200 — prevents small jitter from blowing
   up `ds/dt` when `dt` is small (high frame-rate inputs).
4. **`min_updates = 5`** — suppress noise in the first few frames of a track.
5. **Hysteresis (`critical_hold = 5`) + persistence (3-of-5)** — prevents
   color flicker by requiring multiple critical observations before
   raising the alarm and holding the level briefly afterward.

Diagnostic tool: `experiments/plot_ttc_debug.py`.

## License

MIT — see `LICENSE`.

## Author

(Your name / student ID / contact)
