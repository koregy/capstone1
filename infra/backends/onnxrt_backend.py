"""ONNX Runtime FP32 backend (CUDA Execution Provider).

Loads the ONNX model produced by infra/convert/pt_to_onnx.py and runs it
through ORT's CUDA EP. TensorrtExecutionProvider is intentionally excluded
from the provider list: TRT acceleration is measured separately via the
trtexec path, and mixing the two would invalidate the 5-way comparison.

Input handling: the benchmark runner passes the same torch.cuda.Tensor on
every iteration. We convert it to numpy once and cache the result, keyed on
data_ptr() + shape. This isolates the measurement to ORT's session.run()
itself; the cost of host-to-device transfers in real deployment is a
separate concern (touched on in Day 6 analysis).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from infra.backends.base import BaseBackend


class ONNXRuntimeBackend(BaseBackend):
    """FP32 ONNX Runtime backend with CUDA Execution Provider."""

    name = "onnxrt_fp32"
    precision = "fp32"

    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_names: list[str] | None = None
        self._model_path: str | None = None
        self._providers_used: list[str] = []

        # Cache the numpy view of the most recent input tensor.
        # Keyed by (data_ptr, shape) so re-used tensors hit the cache.
        self._np_cache_key: tuple | None = None
        self._np_cache_val: np.ndarray | None = None

    def load(self, model_path: str | Path, device: str = "cuda") -> None:
        """Build the ORT session with CUDA EP first, CPU as fallback."""
        self._model_path = str(model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.log_severity_level = 3  # ERROR only

        # Provider config. CUDA settings are tuned for Orin Nano 8GB.
        # cudnn_conv_algo_search=EXHAUSTIVE makes the first run slow but
        # finds the fastest conv algorithm; that's why we warm up 20 iters.
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024 ** 3,  # 4 GB cap
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]

        self.session = ort.InferenceSession(
            self._model_path,
            sess_options=sess_options,
            providers=providers,
        )
        self._providers_used = self.session.get_providers()

        # Cache I/O names (avoids repeated str lookups in the hot loop).
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Hard-fail if CUDA EP did not actually attach. A silent CPU fallback
        # would make the latency numbers meaningless.
        if "CUDAExecutionProvider" not in self._providers_used:
            raise RuntimeError(
                f"CUDAExecutionProvider not active. "
                f"Got providers: {self._providers_used}"
            )

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert input to numpy float32, caching the result for re-used tensors.

        runner.run_benchmark reuses the same dummy tensor every iter, so the
        cache hits every time after the first.
        """
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)

        if isinstance(x, torch.Tensor):
            key = (x.data_ptr(), tuple(x.shape))
            if key == self._np_cache_key and self._np_cache_val is not None:
                return self._np_cache_val
            arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
            self._np_cache_key = key
            self._np_cache_val = arr
            return arr

        raise TypeError(f"Unsupported input type: {type(x)}")

    def warmup(self, input_shape: tuple[int, ...], n_iter: int = 20) -> None:
        """Stabilize cuDNN auto-tune and ORT internal buffers."""
        assert self.session is not None, "Call load() before warmup()."
        dummy = np.random.randn(*input_shape).astype(np.float32)
        for _ in range(n_iter):
            self.session.run(self.output_names, {self.input_name: dummy})
        torch.cuda.synchronize()

    def infer(self, x: Any) -> Any:
        """Run one inference. Returns list[np.ndarray] (one per output)."""
        assert self.session is not None, "Call load() before infer()."
        arr = self._to_numpy(x)
        return self.session.run(self.output_names, {self.input_name: arr})

    def teardown(self) -> None:
        """Drop the session and the numpy cache. Frees GPU memory held by ORT."""
        self.session = None
        self._np_cache_key = None
        self._np_cache_val = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @property
    def device_info(self) -> dict:
        return {
            "name": self.name,
            "precision": self.precision,
            "model": self._model_path,
            "providers": self._providers_used,
            "ort_version": ort.__version__,
        }


if __name__ == "__main__":
    # Self-check: load ONNX model, run 5 inferences, print output shape and mem.
    import time

    backend = ONNXRuntimeBackend()
    print(f"Backend created: {backend!r}")

    onnx_path = "models/onnx/yolov8n.onnx"
    print(f"Loading model from {onnx_path}...")
    t0 = time.perf_counter()
    backend.load(onnx_path, device="cuda")
    t1 = time.perf_counter()
    print(f"  Load time: {t1 - t0:.2f} s")
    print(f"  Device info: {backend.device_info}")
    print(f"  Input name : {backend.input_name}")
    print(f"  Output names: {backend.output_names}")

    input_shape = (1, 3, 640, 640)
    print(f"\nWarming up (5 iters, shape={input_shape})...")
    backend.warmup(input_shape, n_iter=5)
    print("  Warmup done.")

    print("\nRunning 3 timed inferences (torch.Tensor input)...")
    dummy = torch.randn(*input_shape, device="cuda")
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = backend.infer(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        shapes = [tuple(o.shape) for o in out]
        print(f"  iter {i}: {(t1 - t0) * 1000:.3f} ms, output shapes: {shapes}")

    print(f"\nGPU memory before teardown: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    backend.teardown()
    print(f"GPU memory after teardown:  {torch.cuda.memory_allocated() / 1e6:.1f} MB")
