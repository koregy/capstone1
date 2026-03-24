"""Accuracy (mAP) measurement orchestrator.

Parallel to experiments/baseline_demo.py but runs COCO mAP evaluation
instead of latency. For each backend in configs/accuracy.yaml, this
script:

    1) builds the backend via the same factory baseline_demo uses
    2) loads weights/engine
    3) runs evaluate_backend() over the 1000-image COCO subset
    4) saves a record to results/accuracy/{name}.json

The factory and YAML schema are shared with baseline_demo.py so adding
new backends only requires touching one place.

Run:
    python3 experiments/accuracy_demo.py --config configs/accuracy.yaml
    python3 experiments/accuracy_demo.py --config configs/accuracy.yaml --only trt_int8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra.backends.base import BaseBackend  # noqa: E402
from infra.backends.pytorch_backend import PyTorchBackend  # noqa: E402
from infra.benchmark.accuracy import (  # noqa: E402
    evaluate_backend,
    load_coco_eval_subset,
)
from infra.benchmark.reporter import build_accuracy_record, save_record  # noqa: E402

# Reuse the same factory as baseline_demo so the YAML schema stays unified.
from experiments.baseline_demo import build_backend, load_weights_path  # noqa: E402


def measure_accuracy_one(
    backend_cfg: dict,
    eval_cfg: dict,
    image_records: list[dict],
    coco_gt,
    results_dir: str,
) -> dict:
    """Full measurement cycle for one backend. Returns the accuracy dict."""
    name = backend_cfg["name"]
    print(f"\n{'=' * 60}")
    print(f"  Evaluating backend: {name}")
    print(f"{'=' * 60}")

    backend: BaseBackend = build_backend(backend_cfg)
    weights = load_weights_path(backend_cfg)
    print(f"[1/3] load({weights})")
    backend.load(weights, device=eval_cfg["device"])

    print(f"[2/3] evaluate over {len(image_records)} images")
    accuracy = evaluate_backend(
        backend=backend,
        image_records=image_records,
        coco_gt=coco_gt,
        imgsz=eval_cfg["imgsz"],
        conf_thres=eval_cfg["conf_thres"],
        iou_thres=eval_cfg["iou_thres"],
        max_det=eval_cfg["max_det"],
        nc=eval_cfg["nc"],
        device=eval_cfg["device"],
        progress_interval=eval_cfg.get("progress_interval", 100),
    )

    print(f"[3/3] save record")
    record = build_accuracy_record(
        backend_name=name,
        precision=backend_cfg.get("precision", backend.precision),
        accuracy=accuracy,
        run_config={
            "imgsz": eval_cfg["imgsz"],
            "n_images": len(image_records),
            "image_subset": f"paths[{eval_cfg['start']}:{eval_cfg['start'] + eval_cfg['count']}]",
            "conf_thres": eval_cfg["conf_thres"],
            "iou_thres": eval_cfg["iou_thres"],
            "max_det": eval_cfg["max_det"],
            "nc": eval_cfg["nc"],
            "ann_file": eval_cfg["ann_file"],
            "device": eval_cfg["device"],
        },
        extra=backend.device_info,
    )

    # write_csv=False: history.csv has latency columns hardcoded.
    json_path, _ = save_record(record, results_dir=results_dir, write_csv=False)
    print(f"      -> {json_path}")

    backend.teardown()
    return accuracy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Accuracy (mAP) measurement orchestrator (COCO).",
    )
    parser.add_argument(
        "--config",
        default="configs/accuracy.yaml",
        help="Path to accuracy YAML.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="If set, evaluate only this backend (by name).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override eval.count from YAML (useful for sanity tests).",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = dict(cfg["eval"])
    eval_cfg["imgsz"] = cfg["input"]["imgsz"]
    if args.count is not None:
        eval_cfg["count"] = args.count

    results_dir = cfg["output"]["results_dir"]

    # Filter backends.
    backends_to_run = [b for b in cfg["backends"] if b.get("enabled", True)]
    if args.only:
        backends_to_run = [b for b in backends_to_run if b["name"] == args.only]
        if not backends_to_run:
            print(f"No enabled backend named {args.only!r}", file=sys.stderr)
            return 1

    print(f"Config:        {args.config}")
    print(f"Image subset:  paths[{eval_cfg['start']}:{eval_cfg['start'] + eval_cfg['count']}] "
          f"({eval_cfg['count']} images)")
    print(f"Conf/IoU:      {eval_cfg['conf_thres']} / {eval_cfg['iou_thres']}")
    print(f"Results dir:   {results_dir}")
    print(f"Backends:      {[b['name'] for b in backends_to_run]}")

    # Load COCO subset once and reuse across all backends.
    print("\nLoading COCO annotations + image subset...")
    image_records, coco_gt = load_coco_eval_subset(
        ann_file=eval_cfg["ann_file"],
        images_dir=eval_cfg["images_dir"],
        start=eval_cfg["start"],
        count=eval_cfg["count"],
    )
    print(f"  Loaded {len(image_records)} image records")

    all_accuracy: dict[str, dict] = {}
    for b in backends_to_run:
        all_accuracy[b["name"]] = measure_accuracy_one(
            b, eval_cfg, image_records, coco_gt, results_dir,
        )

    # Final cross-backend summary.
    if len(all_accuracy) > 1:
        print("\n" + "=" * 60)
        print("  ACCURACY SUMMARY")
        print("=" * 60)
        print(f"  {'backend':28s} {'mAP@.5:.95':>10s} {'mAP@.50':>10s} {'mAP@.75':>10s} {'n_det':>8s}")
        for name, a in all_accuracy.items():
            print(f"  {name:28s} {a['mAP_50_95']:>10.4f} "
                  f"{a['mAP_50']:>10.4f} {a['mAP_75']:>10.4f} "
                  f"{a['n_detections']:>8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
