"""Build an INT8 TensorRT engine using EntropyCalibrator or MinMaxCalibrator.

Generalized entry point that subsumes the ad-hoc yolov8n entropy_500 /
entropy_100 / minmax_500 builds from Day 3. Path convention enforced so
new models (yolov8s, yolov8m, ...) work by changing only --model.

Path convention (model={model}, calibrator={cal}, num={N}):
    onnx   : models/onnx/{model}.onnx
    engine : models/trt/{model}_int8_{cal}_{N}.engine
    cache  : cache/{model}_{cal}_{N}.cache
    log    : (auto by build_trt_engine_int8, includes calibrator class name)

For percentile calibration (cache injection path), use scripts/build_percentile_500.py
instead -- the percentile pipeline has a Stage A activation-collection step
that does not fit this template.

Run from repo root:
    python3 scripts/build_int8_engine.py --model yolov8s --calibrator entropy --num-calib 500
    python3 scripts/build_int8_engine.py --model yolov8s --calibrator minmax  --num-calib 500
    python3 scripts/build_int8_engine.py --model yolov8n --calibrator entropy --num-calib 100
"""
import sys
import argparse
import logging
from pathlib import Path

# Make repo modules importable when script is run from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from infra.convert.calibration_dataset import list_coco_val_images, CalibrationDataset
from infra.convert.calibrators import EntropyCalibrator, MinMaxCalibrator
from infra.convert.onnx_to_trt import build_trt_engine_int8


CALIBRATOR_CLASSES = {
    "entropy": EntropyCalibrator,
    "minmax":  MinMaxCalibrator,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Model name stem (e.g. yolov8n, yolov8s). "
                             "Path convention: models/onnx/{model}.onnx etc.")
    parser.add_argument("--calibrator", required=True, choices=list(CALIBRATOR_CLASSES),
                        help="Calibration algorithm. percentile uses scripts/build_percentile_500.py.")
    parser.add_argument("--num-calib", type=int, default=500,
                        help="Number of COCO calibration images (default: 500).")
    parser.add_argument("--workspace-mb", type=int, default=4096,
                        help="TRT builder workspace in MB (default: 4096).")
    parser.add_argument("--log", type=str, default=None,
                        help="TRT verbose build log path. Default: "
                             "logs/trt_build_int8_<calibrator>.log (may overwrite!).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

    # Path assembly via convention.
    model      = args.model
    cal        = args.calibrator
    N          = args.num_calib
    onnx_path   = REPO_ROOT / f"models/onnx/{model}.onnx"
    engine_path = REPO_ROOT / f"models/trt/{model}_int8_{cal}_{N}.engine"
    cache_file  = REPO_ROOT / f"cache/{model}_{cal}_{N}.cache"

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX not found: {onnx_path}\n"
            f"Run: python3 infra/convert/pt_to_onnx.py "
            f"--weights models/pt/{model}.pt --output models/onnx/{model}.onnx"
        )

    print(f"[build_int8] model      = {model}")
    print(f"[build_int8] calibrator = {cal}")
    print(f"[build_int8] num_calib  = {N}")
    print(f"[build_int8] onnx       = {onnx_path}")
    print(f"[build_int8] engine     = {engine_path}")
    print(f"[build_int8] cache      = {cache_file}")

    # Stage 1: calibration dataset (first N images of COCO val).
    # Sliced [:N] matches accuracy eval's [500:1500] -- no overlap with eval set.
    paths = list_coco_val_images(REPO_ROOT / "data/coco_val/images")
    if len(paths) < N:
        raise RuntimeError(
            f"Need {N} calibration images but only {len(paths)} available in "
            f"data/coco_val/images"
        )
    dataset = CalibrationDataset(paths[:N], batch_size=1, img_size=640)

    # Stage 2: instantiate calibrator. Both Entropy and MinMax share the same
    # __init__(dataset, cache_file, input_name) signature via _CalibratorMixin.
    CalClass = CALIBRATOR_CLASSES[cal]
    calibrator = CalClass(
        dataset,
        cache_file=cache_file,
        input_name="images",
    )

    # Stage 3: build engine. If cache_file already exists, TRT cache-hits and
    # skips the calibration forward pass (~50% faster build).
    print(f"[build_int8] building engine via {CalClass.__name__}...")
    result = build_trt_engine_int8(
        onnx_path=str(onnx_path),
        engine_path=str(engine_path),
        calibrator=calibrator,
        workspace_mb=args.workspace_mb,
        log_path=args.log,
        verbose=True,
    )

    import json
    print()
    print("=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
