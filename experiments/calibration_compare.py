"""Measure latency across the 4 INT8 calibration variants.

This is a thin twin of `experiments/baseline_demo.py`, restricted to the
INT8 calibration variants for §4.2 / §12.7.5 of the project doc:

    trt_int8_entropy_100   (small calib pool, KL-div algorithm)
    trt_int8_entropy_500   (large calib pool, KL-div algorithm)
    trt_int8_minmax_500    (large calib pool, min/max algorithm)
    trt_int8_percentile_500 (large calib pool, 99.9 percentile via cache injection)

We reuse `measure_one()` from baseline_demo. The control-flow code (argparse,
YAML loading, filter, summary) is short enough that duplicating it here is
cheaper than abstracting it into a library function in baseline_demo.

Usage:
    python3 experiments/calibration_compare.py
    python3 experiments/calibration_compare.py --config configs/calibration_compare.yaml
    python3 experiments/calibration_compare.py --only trt_int8_minmax_500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make repo modules importable when run from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.baseline_demo import measure_one  # reuse the per-backend cycle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/calibration_compare.yaml",
        help="Path to calibration comparison YAML config.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="If set, run only the backend with this name.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Override run.n_iter from the YAML.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=None,
        help="Override run.n_warmup from the YAML.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    input_shape = (inp["batch_size"], inp["channels"], inp["imgsz"], inp["imgsz"])
    run_cfg = dict(cfg["run"])
    if args.n_iter is not None:
        run_cfg["n_iter"] = args.n_iter
    if args.n_warmup is not None:
        run_cfg["n_warmup"] = args.n_warmup
    results_dir = cfg["output"]["results_dir"]

    # Same backend filtering logic as baseline_demo
    backends_to_run = []
    for b in cfg["backends"]:
        if args.only is not None:
            if b["name"] == args.only:
                backends_to_run.append(b)
        else:
            if b.get("enabled", False):
                backends_to_run.append(b)

    if not backends_to_run:
        print("No backends to run. Check --only or 'enabled' flags in YAML.")
        return 1

    print(f"Config:        {args.config}")
    print(f"Input shape:   {input_shape}")
    print(f"Warmup / iter: {run_cfg['n_warmup']} / {run_cfg['n_iter']}")
    print(f"Results dir:   {results_dir}")
    print(f"Variants:      {[b['name'] for b in backends_to_run]}")

    all_stats: dict[str, dict] = {}
    for b in backends_to_run:
        all_stats[b["name"]] = measure_one(b, input_shape, run_cfg, results_dir)

    # Cross-variant summary
    if len(all_stats) > 1:
        print("\n" + "=" * 70)
        print("  CALIBRATION VARIANT COMPARISON")
        print("=" * 70)
        # Use a slightly wider name column since variant names are longer
        for name, s in all_stats.items():
            print(f"  {name:30s}  mean={s['mean_ms']:7.3f} ms  "
                  f"p95={s['p95_ms']:7.3f}  fps={s['fps']:6.2f}")

        # Relative comparison vs entropy_500 (our "canonical" INT8 baseline)
        baseline_name = "trt_int8_entropy_500"
        if baseline_name in all_stats:
            print()
            print(f"  vs {baseline_name} (= 5-way table's trt_int8):")
            base = all_stats[baseline_name]["mean_ms"]
            for name, s in all_stats.items():
                if name == baseline_name:
                    continue
                delta = s["mean_ms"] - base
                pct = 100.0 * delta / base
                sign = "+" if delta >= 0 else ""
                print(f"    {name:30s}  {sign}{delta:6.3f} ms ({sign}{pct:5.2f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
