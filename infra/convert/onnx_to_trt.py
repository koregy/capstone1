"""Thin wrapper around the NVIDIA `trtexec` CLI. Choosing the CLI path over the
Python tensorrt.Builder API because:
  1. Day 4 fusion analysis parses `--verbose` build logs; using trtexec
     keeps build and analysis on the same tool.
  2. FP32 / FP16 builds are dramatically simpler this way.
  3. Build progress is visible in real time.

INT8 with real calibration data CANNOT use trtexec: `trtexec --int8` falls back
to random calibration which destroys accuracy. INT8 builds therefore use the
Python tensorrt.Builder API in `build_trt_engine_int8()` below, which accepts
a user-provided `IInt8Calibrator` (Entropy / MinMax / Percentile, see
`infra/convert/calibrators/`).
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
        precision: 'fp32' or 'fp16'. For INT8 use `build_trt_engine_int8()`.
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
            f"INT8 builds use a separate Python calibrator path; "
            f"call build_trt_engine_int8() instead."
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


# ---------------------------------------------------------------------------
# INT8 path (Python tensorrt.Builder API)
# ---------------------------------------------------------------------------
#
# Why a separate function instead of an `int8` branch in build_trt_engine:
#   * Different tool: trtexec vs. tensorrt.Builder.
#   * Different inputs: calibrator object cannot be serialized to a CLI flag.
#   * Keeping the FP32/FP16 path byte-identical to Day 2 protects the existing
#     baseline measurements (7.36 ms / 3.98 ms) from accidental regression
#     when trtexec build options shift between TRT versions.
#
# The dict shape returned is the same as build_trt_engine() so callers
# (experiments/baseline_demo.py, accuracy.py) can treat the two uniformly.


def build_trt_engine_int8(
    onnx_path: str | Path,
    engine_path: str | Path,
    calibrator,  # trt.IInt8Calibrator -- not typed to avoid import-time TRT dep
    workspace_mb: int = 4096,
    log_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Build an INT8 TensorRT engine from ONNX using a user-provided calibrator.

    Unlike `build_trt_engine` (trtexec) this uses the Python tensorrt.Builder
    API because INT8 requires feeding real data through the network during
    the build, which trtexec cannot do without a precomputed calibration cache.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Where to write the .engine. Parent dir is created.
        calibrator: An instance of `trt.IInt8Calibrator` (subclass).
            See infra/convert/calibrators/ for our three implementations.
        workspace_mb: Build-time scratch memory in MB. INT8 calibration is
            heavier than FP32/FP16; default raised to 4 GB.
        log_path: TRT build log destination. Default:
            logs/trt_build_int8_<calibrator_class>.log.
        verbose: If True, TRT logger uses VERBOSE severity so Day 4 fusion
            analysis can parse the INT8 build log too.

    Returns:
        dict with the same keys as build_trt_engine().

    Raises:
        FileNotFoundError: onnx_path missing.
        RuntimeError: ONNX parse failure, shape/batch mismatch, or build failure.
    """
    # Late import: keeps `import infra.convert.onnx_to_trt` cheap for callers
    # that only use trtexec (no TRT Python module needed for that).
    import tensorrt as trt

    onnx_path = Path(onnx_path).resolve()
    engine_path = Path(engine_path).resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    if log_path is None:
        cal_name = type(calibrator).__name__.lower().replace("calibrator", "")
        log_path = Path(f"logs/trt_build_int8_{cal_name}.log").resolve()
    else:
        log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom logger: write everything to file, echo only milestones to console.
    # Mirrors what the trtexec path does via subprocess line-tee.
    class _FileTeeLogger(trt.ILogger):
        def __init__(self, file_handle, verbose_to_file: bool):
            trt.ILogger.__init__(self)
            self._fh = file_handle
            self._verbose = verbose_to_file
            self._severity_name = {
                trt.ILogger.INTERNAL_ERROR: "INTERNAL_ERROR",
                trt.ILogger.ERROR: "E",
                trt.ILogger.WARNING: "W",
                trt.ILogger.INFO: "I",
                trt.ILogger.VERBOSE: "V",
            }

        def log(self, severity, msg):
            name = self._severity_name.get(severity, "?")
            # File: keep everything if verbose, else INFO+.
            if self._verbose or severity <= trt.ILogger.INFO:
                self._fh.write(f"[{name}] {msg}\n")
            # Console: errors/warnings always, plus "Engine built in" milestone.
            if severity <= trt.ILogger.WARNING or "Engine built in" in msg:
                print(f"  [{name}] {msg}")

    print(f"[trt-int8] onnx      = {onnx_path}")
    print(f"[trt-int8] engine    = {engine_path}")
    print(f"[trt-int8] log       = {log_path}")
    print(f"[trt-int8] workspace = {workspace_mb} MB")
    print(f"[trt-int8] calibrator= {type(calibrator).__name__} "
          f"(batch_size={calibrator.get_batch_size()})")
    print(f"[trt-int8] TRT       = {trt.__version__}")
    print(f"[trt-int8] starting build...")

    t0 = time.perf_counter()

    with open(log_path, "w") as log_f:
        trt_logger = _FileTeeLogger(log_f, verbose_to_file=verbose)

        builder = trt.Builder(trt_logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)

        parser = trt.OnnxParser(network, trt_logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = "\n".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise RuntimeError(f"ONNX parse failed:\n{errors}")

        # Sanity-check network input matches calibrator expectations.
        if network.num_inputs != 1:
            raise RuntimeError(
                f"This builder assumes single-input networks, but ONNX has "
                f"{network.num_inputs} inputs."
            )
        inp = network.get_input(0)
        inp_shape = tuple(inp.shape)
        print(f"[trt-int8] network input: name={inp.name}, shape={inp_shape}")

        if any(d <= 0 for d in inp_shape):
            raise RuntimeError(
                f"Dynamic shape detected in input: {inp_shape}. "
                f"This builder only supports static shapes (Day 2 ONNX is "
                f"static B=1). Add an optimization profile for dynamic batching."
            )
        if inp_shape[0] != calibrator.get_batch_size():
            raise RuntimeError(
                f"Calibrator batch_size={calibrator.get_batch_size()} but "
                f"network input batch_dim={inp_shape[0]}. These must match "
                f"or TRT will silently miscalibrate."
            )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
        )
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                f"build_serialized_network returned None. "
                f"See log at {log_path} for the actual error."
            )

        engine_bytes = bytes(serialized)
        engine_path.write_bytes(engine_bytes)

    elapsed = time.perf_counter() - t0
    engine_size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"[trt-int8] build finished in {elapsed:.1f}s, "
          f"engine size {engine_size_mb:.2f} MB")

    return {
        "engine_path": str(engine_path),
        "log_path": str(log_path),
        "build_time_s": elapsed,
        "precision": "int8",
        "engine_size_mb": engine_size_mb,
        "return_code": 0,  # success: Python exceptions otherwise
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX.")
    parser.add_argument("--onnx", default="models/onnx/yolov8n.onnx")
    parser.add_argument("--engine", default=None,
                        help="Output engine path. Default: models/trt/yolov8n_<precision>.engine")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                        help="INT8 builds are not exposed in this CLI -- they need a "
                             "calibrator object. Use a separate entrypoint script.")
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
