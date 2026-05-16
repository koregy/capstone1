"""ONNX -> TensorRT engine via `trtexec`.

Thin wrapper around the NVIDIA `trtexec` CLI. Choosing the CLI path over the
Python tensorrt.Builder API because:
  1. Day 4 fusion analysis parses `--verbose` build logs; using trtexec
     keeps build and analysis on the same tool.
  2. FP32 / FP16 builds are dramatically simpler this way.
  3. Build progress is visible in real time.

INT8 with real calibration data is NOT handled here. `trtexec --int8` falls
back to random calibration which destroys accuracy; proper calibration uses
the Python IInt8EntropyCalibrator2 path and lives in Day 3 work.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

# Default trtexec location on JetPack 6.1.
DEFAULT_TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def find_trtexec() -> str:
    """Locate the trtexec binary. Returns absolute path or raises."""
    # 1) Default JetPack location.
    if Path(DEFAULT_TRTEXEC).exists():
        return DEFAULT_TRTEXEC
    # 2) PATH lookup as a fallback.
    found = shutil.which("trtexec")
    if found:
        return found
    raise FileNotFoundError(
        f"trtexec not found. Looked at {DEFAULT_TRTEXEC} and on PATH."
    )


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str = "fp32",
    workspace_mb: int = 2048,
    log_path: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Build a TensorRT engine from an ONNX file.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Where to write the .engine. Parent dir is created.
        precision: 'fp32' or 'fp16'. INT8 is handled separately (Day 3).
        workspace_mb: Build-time scratch memory in MB. 2 GB is plenty for
            YOLOv8n on Orin Nano 8 GB.
        log_path: If given, full verbose stdout is teed to this file
            (used by Day 4 fusion analysis). Default: logs/trt_build_<precision>.log.
        extra_args: Additional trtexec flags appended to the command.

    Returns:
        dict with keys: engine_path, log_path, build_time_s, precision,
        engine_size_mb, return_code.

    Raises:
        ValueError: invalid precision.
        FileNotFoundError: onnx_path missing or trtexec missing.
        RuntimeError: build failed (non-zero return code).
    """
    if precision not in ("fp32", "fp16"):
        raise ValueError(
            f"precision must be 'fp32' or 'fp16', got {precision!r}. "
            f"INT8 builds use a separate Python calibrator path (Day 3)."
        )

    onnx_path = Path(onnx_path).resolve()
    engine_path = Path(engine_path).resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    if log_path is None:
        log_path = Path(f"logs/trt_build_{precision}.log").resolve()
    else:
        log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    trtexec = find_trtexec()

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--verbose",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    if extra_args:
        cmd.extend(extra_args)

    print(f"[trt] precision = {precision}")
    print(f"[trt] onnx      = {onnx_path}")
    print(f"[trt] engine    = {engine_path}")
    print(f"[trt] log       = {log_path}")
    print(f"[trt] cmd       = {' '.join(cmd)}")
    print(f"[trt] starting build...")

    t0 = time.perf_counter()

    # Tee stdout to file AND show selected lines on console.
    # The full verbose log lives in log_path; the console only echoes
    # progress markers to keep the screen readable.
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            env={**os.environ},
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            # Show only key progress milestones on the console.
            stripped = line.rstrip()
            if any(
                marker in line
                for marker in (
                    "[I] Engine built in",
                    "[I] === ",
                    "[E]",
                    "[W]",
                    "FAILED",
                    "Error",
                )
            ) and "[V]" not in line:
                print(f"  {stripped}")

        proc.wait()
        ret = proc.returncode

    elapsed = time.perf_counter() - t0
    print(f"[trt] build finished in {elapsed:.1f}s (return code {ret})")

    if ret != 0:
        raise RuntimeError(
            f"trtexec failed with return code {ret}. "
            f"See full log at {log_path}"
        )

    if not engine_path.exists():
        raise RuntimeError(
            f"trtexec reported success but engine file is missing: {engine_path}"
        )

    engine_size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"[trt] engine size: {engine_size_mb:.2f} MB")

    return {
        "engine_path": str(engine_path),
        "log_path": str(log_path),
        "build_time_s": elapsed,
        "precision": precision,
        "engine_size_mb": engine_size_mb,
        "return_code": ret,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX.")
    parser.add_argument("--onnx", default="models/onnx/yolov8n.onnx")
    parser.add_argument("--engine", default=None,
                        help="Output engine path. Default: models/trt/yolov8n_<precision>.engine")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    if args.engine is None:
        args.engine = f"models/trt/yolov8n_{args.precision}.engine"

    result = build_trt_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace_mb=args.workspace_mb,
        log_path=args.log,
    )
    print("\n[result]")
    print(json.dumps(result, indent=2))
