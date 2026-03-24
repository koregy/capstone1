"""Persist benchmark results to disk.

Writes two artifacts per measurement:

1) results/{subdir}/{backend_name}.json  — overwritten each run, full snapshot.
2) results/{subdir}/history.csv          — appended each run, flat summary.

The JSON is the source of truth for Day 7 plots (one file per backend).
The CSV is for tracking variance across re-runs of the same backend.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


# CSV column order. Kept stable across the project; do not reorder, only append.
CSV_COLUMNS = [
    "timestamp",
    "backend",
    "precision",
    "n",
    "mean_ms",
    "std_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "fps",
    "n_warmup",
    "n_iter",
    "input_shape",
]


def _collect_env_info() -> dict[str, Any]:
    """Collect environment metadata. Each field is best-effort."""
    info: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    try:
        info["torch"] = torch.__version__
    except Exception as e:
        info["torch"] = f"<error: {e}>"
    try:
        info["cuda"] = torch.version.cuda
    except Exception as e:
        info["cuda"] = f"<error: {e}>"
    try:
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        info["gpu"] = f"<error: {e}>"
    return info


def build_record(
    backend_name: str,
    precision: str,
    stats: dict,
    run_config: dict,
    extra: dict | None = None,
) -> dict:
    """Assemble the full record dict that will be written to JSON.

    Args:
        backend_name: e.g. "pytorch_fp32", "onnxrt_fp32", "trt_fp16".
        precision: "fp32" | "fp16" | "int8".
        stats: Output of metrics.compute_latency_stats().
        run_config: dict with keys input_shape, n_warmup, n_iter, seed, device.
        extra: Backend-specific extras (engine path, ORT providers, etc.).
    """
    return {
        "backend": {
            "name": backend_name,
            "precision": precision,
        },
        "run": run_config,
        "stats": stats,
        "env": _collect_env_info(),
        "extra": extra or {},
    }


def build_accuracy_record(
    backend_name: str,
    precision: str,
    accuracy: dict,
    run_config: dict,
    extra: dict | None = None,
) -> dict:
    """Build a record for accuracy (mAP) measurement, parallel to build_record().

    Args:
        backend_name: e.g. "pytorch_fp32", "trt_int8_entropy_500".
        precision: "fp32" | "fp16" | "int8".
        accuracy: Output of evaluate_backend(). Expected keys:
            mAP_50_95, mAP_50, mAP_75, mAP_small, mAP_medium, mAP_large,
            n_images, n_detections, eval_time_sec, inference_time_sec.
        run_config: dict with keys:
            imgsz, n_images, image_subset, conf_thres, iou_thres,
            max_det, nc, ann_file, device.
        extra: Backend-specific extras (engine path, ORT providers, etc.).
    """
    return {
        "backend": {
            "name": backend_name,
            "precision": precision,
        },
        "run": run_config,
        "accuracy": accuracy,
        "env": _collect_env_info(),
        "extra": extra or {},
    }


def save_record(
    record: dict,
    results_dir: str | Path = "results/baseline",
    write_csv: bool = True,
) -> tuple[Path, Path | None]:
    """Write the record as JSON, and append a summary row to history.csv.

    Args:
        record: Output of build_record().
        results_dir: Directory under which to write. Created if missing.
        write_csv: If False, only JSON is written.

    Returns:
        (json_path, csv_path_or_None)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    backend_name = record["backend"]["name"]
    json_path = results_dir / f"{backend_name}.json"
    with json_path.open("w") as f:
        json.dump(record, f, indent=2)

    csv_path: Path | None = None
    if write_csv:
        csv_path = results_dir / "history.csv"
        is_new = not csv_path.exists()
        row = {
            "timestamp": record["env"]["timestamp"],
            "backend": backend_name,
            "precision": record["backend"]["precision"],
            "n": record["stats"]["n"],
            "mean_ms": f"{record['stats']['mean_ms']:.4f}",
            "std_ms": f"{record['stats']['std_ms']:.4f}",
            "p50_ms": f"{record['stats']['p50_ms']:.4f}",
            "p95_ms": f"{record['stats']['p95_ms']:.4f}",
            "p99_ms": f"{record['stats']['p99_ms']:.4f}",
            "fps": f"{record['stats']['fps']:.4f}",
            "n_warmup": record["run"].get("n_warmup", ""),
            "n_iter": record["run"].get("n_iter", ""),
            "input_shape": str(record["run"].get("input_shape", "")),
        }
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    return json_path, csv_path


if __name__ == "__main__":
    # Self-check: build a fake record, save it, read it back.
    import tempfile

    fake_stats = {
        "n": 200,
        "n_warmup_excluded": 20,
        "mean_ms": 24.96,
        "std_ms": 0.43,
        "min_ms": 23.8,
        "max_ms": 26.1,
        "p50_ms": 24.95,
        "p90_ms": 25.50,
        "p95_ms": 25.78,
        "p99_ms": 26.05,
        "fps": 40.06,
    }
    fake_run = {
        "input_shape": [1, 3, 640, 640],
        "n_warmup": 20,
        "n_iter": 200,
        "seed": 42,
        "device": "cuda",
    }
    rec = build_record(
        backend_name="pytorch_fp32",
        precision="fp32",
        stats=fake_stats,
        run_config=fake_run,
        extra={"weights": "models/pt/yolov8n.pt"},
    )

    with tempfile.TemporaryDirectory() as td:
        jp, cp = save_record(rec, results_dir=td)
        print(f"JSON: {jp}")
        print(f"CSV:  {cp}")
        # Verify roundtrip
        with open(jp) as f:
            loaded = json.load(f)
        assert loaded["backend"]["name"] == "pytorch_fp32"
        assert loaded["stats"]["mean_ms"] == 24.96
        print("Roundtrip OK.")
        # Show CSV content
        print("--- CSV content ---")
        print(cp.read_text())
