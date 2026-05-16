"""Day 2 baseline measurement orchestrator.

Reads configs/benchmark.yaml, runs each enabled backend through
warmup -> timed loop -> stats -> save. Backends are measured sequentially
with explicit teardown between them to keep Orin Nano 8GB happy.

Usage:
    python experiments/baseline_demo.py
    python experiments/baseline_demo.py --only pytorch_fp32
    python experiments/baseline_demo.py --n-iter 50   # quick check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Make the project root importable when the script is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.backends.base import BaseBackend  # noqa: E402
from infra.backends.pytorch_backend import PyTorchBackend  # noqa: E402
from infra.benchmark.metrics import (  # noqa: E402
    compute_latency_stats,
    format_stats_table,
)
from infra.benchmark.reporter import build_record, save_record  # noqa: E402
from infra.benchmark.runner import run_benchmark  # noqa: E402


def build_backend(backend_cfg: dict) -> BaseBackend:
    """Factory: turn a YAML backend dict into a concrete BaseBackend.

    Day 2 morning: pytorch_fp32 only.
    Day 2 afternoon: onnxrt_fp32, trt_fp32, trt_fp16 added.
    Day 3: trt_int8.
    """
    name = backend_cfg["name"]
    if name == "pytorch_fp32":
        return PyTorchBackend()
    if name == "onnxrt_fp32":
        from infra.backends.onnxrt_backend import ONNXRuntimeBackend
        return ONNXRuntimeBackend()
    if name in ("trt_fp32", "trt_fp16"):
        from infra.backends.tensorrt_backend import TensorRTBackend
        precision = backend_cfg.get("precision", name.split("_")[1])
        return TensorRTBackend(name=name, precision=precision)
    raise ValueError(f"Unknown or not-yet-implemented backend: {name}")
def load_weights_path(backend_cfg: dict) -> str:
    """Each backend stores its model under a different key (weights/engine).

    Returns the path the backend's load() should receive.
    """
    if "weights" in backend_cfg:
        return backend_cfg["weights"]
    if "engine" in backend_cfg:
        return backend_cfg["engine"]
    raise KeyError(
        f"Backend {backend_cfg.get('name')} has neither 'weights' nor 'engine'."
    )


def measure_one(
    backend_cfg: dict,
    input_shape: tuple[int, ...],
    run_cfg: dict,
    results_dir: str,
) -> dict:
    """Full measurement cycle for one backend. Returns the stats dict."""
    name = backend_cfg["name"]
    print(f"\n{'=' * 60}")
    print(f"  Measuring backend: {name}")
    print(f"{'=' * 60}")

    backend = build_backend(backend_cfg)
    weights = load_weights_path(backend_cfg)

    print(f"[1/4] load({weights})")
    backend.load(weights, device=run_cfg["device"])

    print(f"[2/4] warmup x {run_cfg['n_warmup']}, infer x {run_cfg['n_iter']}")
    latencies = run_benchmark(
        backend=backend,
        input_shape=input_shape,
        n_warmup=run_cfg["n_warmup"],
        n_iter=run_cfg["n_iter"],
        device=run_cfg["device"],
        seed=run_cfg["seed"],
    )

    print(f"[3/4] compute stats")
    stats = compute_latency_stats(
        latencies_ms=latencies,
        n_warmup_excluded=run_cfg["n_warmup"],
    )

    print(f"[4/4] save record")
    record = build_record(
        backend_name=name,
        precision=backend_cfg.get("precision", backend.precision),
        stats=stats,
        run_config={
            "input_shape": list(input_shape),
            "n_warmup": run_cfg["n_warmup"],
            "n_iter": run_cfg["n_iter"],
            "seed": run_cfg["seed"],
            "device": run_cfg["device"],
        },
        extra=backend.device_info,
    )
    json_path, csv_path = save_record(record, results_dir=results_dir)
    print(f"      -> {json_path}")
    print(f"      -> {csv_path}")

    print("\n" + format_stats_table(stats, title=name))

    # Release GPU memory before moving to the next backend.
    backend.teardown()
    del backend
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="If set, run only the backend with this name (overrides 'enabled').",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Override run.n_iter from the YAML (useful for quick checks).",
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

    # Resolve input shape into a tuple.
    inp = cfg["input"]
    input_shape = (inp["batch_size"], inp["channels"], inp["imgsz"], inp["imgsz"])

    run_cfg = dict(cfg["run"])
    if args.n_iter is not None:
        run_cfg["n_iter"] = args.n_iter
    if args.n_warmup is not None:
        run_cfg["n_warmup"] = args.n_warmup

    results_dir = cfg["output"]["results_dir"]

    # Filter the backend list.
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
    print(f"Backends:      {[b['name'] for b in backends_to_run]}")

    all_stats: dict[str, dict] = {}
    for b in backends_to_run:
        all_stats[b["name"]] = measure_one(b, input_shape, run_cfg, results_dir)

    # Final cross-backend summary.
    if len(all_stats) > 1:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        for name, s in all_stats.items():
            print(f"  {name:20s}  mean={s['mean_ms']:7.3f} ms  p95={s['p95_ms']:7.3f}  fps={s['fps']:6.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
