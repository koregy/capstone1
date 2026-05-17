"""Build the percentile_500 INT8 engine via cache injection.

Pipeline:
  1. (Read) entropy_500 cache for template (tensor names + header).
  2. Run onnxruntime forward over 500 calibration images, gather
     per-tensor 99.9 percentile of |activations|.
  3. Write percentile_500.cache with scales = percentile / 127.
  4. Build INT8 engine using EntropyCalibrator -- TRT will cache-hit
     and use our percentile-derived scales directly.

Run from repo root:
    python3 scripts/build_percentile_500.py
"""

import sys
import logging
from pathlib import Path

# Make repo modules importable when script is run from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from infra.convert.calibration_dataset import list_coco_val_images, CalibrationDataset
from infra.convert.calibrators import EntropyCalibrator
from infra.convert.onnx_to_trt import build_trt_engine_int8
from infra.convert.percentile_cache import build_percentile_cache


def main():
    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

    # Stage 1: setup paths
    onnx_path = REPO_ROOT / "models/onnx/yolov8n.onnx"
    template_cache = REPO_ROOT / "cache/entropy_500.cache"
    percentile_cache_file = REPO_ROOT / "cache/percentile_500.cache"
    engine_path = REPO_ROOT / "models/trt/yolov8n_int8_percentile_500.engine"
    log_path = REPO_ROOT / "logs/trt_build_int8_percentile_500.log"

    # Sanity: template must exist
    if not template_cache.is_file():
        raise FileNotFoundError(
            f"Template cache not found: {template_cache}\n"
            f"Build entropy_500.engine first (this provides the template)."
        )

    # Stage 2: collect percentile + write cache (skip if already done)
    if percentile_cache_file.is_file():
        print(f"[skip] percentile cache already exists: {percentile_cache_file}")
        print(f"       delete it to recompute.")
    else:
        paths = list_coco_val_images(REPO_ROOT / "data/coco_val/images")
        dataset = CalibrationDataset(paths[:500], batch_size=1, img_size=640)

        print(f"[1/2] collecting percentile activations over 500 images...")
        thresholds = build_percentile_cache(
            onnx_path=onnx_path,
            template_cache_path=template_cache,
            dataset=dataset,
            output_cache_path=percentile_cache_file,
            percentile=99.9,
            histogram_bins=8192,
        )
        # Show a few for sanity
        sample = list(thresholds.items())[:5]
        print(f"      collected {len(thresholds)} percentile thresholds, e.g.:")
        for n, t in sample:
            print(f"         {n:60s} p99.9(|x|) = {t:.4f}  -> scale = {t/127:.6f}")

    # Stage 3: build the engine with EntropyCalibrator -- TRT will read our
    # injected cache and skip its own calibration.
    print(f"[2/2] building INT8 engine via cache injection...")
    # Calibrator's dataset is technically unused (cache hit), but we still
    # need the object for the API. Use a small dataset to keep memory low.
    paths = list_coco_val_images(REPO_ROOT / "data/coco_val/images")
    tiny_ds = CalibrationDataset(paths[:10], batch_size=1, img_size=640)
    cal = EntropyCalibrator(
        tiny_ds,
        cache_file=percentile_cache_file,
        input_name="images",
    )

    result = build_trt_engine_int8(
        onnx_path=str(onnx_path),
        engine_path=str(engine_path),
        calibrator=cal,
        workspace_mb=4096,
        log_path=str(log_path),
        verbose=True,
    )

    import json
    print()
    print("=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
