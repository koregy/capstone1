"""TensorRT backend using the Python API (TRT 10.x).

Loads an .engine file built by infra/convert/onnx_to_trt.py and runs
inference via execute_async_v3 on the current CUDA stream.

Design notes:
- CUDA memory is accessed through PyTorch tensors. torch.empty(..., device='cuda')
  gives us a managed allocation, and tensor.data_ptr() gives a raw CUDA pointer
  that TensorRT accepts via set_tensor_address. This avoids pulling in pycuda
  as a separate dependency.
- Input and output buffers are pre-allocated in load() and reused. Allocating
  inside infer() would contaminate the latency measurement.
- We do not call torch.cuda.synchronize() inside infer(); the benchmark runner
  is responsible for that. Calling sync twice per iter would skew comparisons
  against PyTorch / ORT backends.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from infra.backends.base import BaseBackend


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    """Map TensorRT DataType -> torch.dtype.

    Imported lazily because tensorrt is heavy and not needed for type hints.
    """
    import tensorrt as trt
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    if trt_dtype not in mapping:
        raise ValueError(f"Unsupported TRT dtype: {trt_dtype}")
    return mapping[trt_dtype]


class TensorRTBackend(BaseBackend):
    """TensorRT inference backend. Subclass per precision to set name/precision.

    Usage:
        backend = TensorRTBackend(name='trt_fp32', precision='fp32')
        backend.load('models/trt/yolov8n_fp32.engine')
    """

    def __init__(self, name: str = "trt_fp32", precision: str = "fp32") -> None:
        self.name = name
        self.precision = precision

        self._engine = None
        self._context = None
        self._runtime = None
        self._logger = None
        self._engine_path: str | None = None

        # Pre-allocated I/O buffers, keyed by tensor name.
        self._buffers: dict[str, torch.Tensor] = {}
        self._input_names: list[str] = []
        self._output_names: list[str] = []

        # CUDA stream handle (an int pointer to a cudaStream_t).
        self._stream: int | None = None

    def load(self, model_path: str | Path, device: str = "cuda") -> None:
        """Deserialize the .engine file and allocate I/O buffers."""
        import tensorrt as trt

        self._engine_path = str(model_path)
        engine_path = Path(model_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        # Build logger / runtime / engine / context.
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")
        self._context = self._engine.create_execution_context()

        # Discover I/O tensors and allocate buffers.
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            trt_dtype = self._engine.get_tensor_dtype(name)
            t_dtype = _trt_dtype_to_torch(trt_dtype)
            mode = self._engine.get_tensor_mode(name)

            buf = torch.empty(shape, dtype=t_dtype, device=device)
            self._buffers[name] = buf
            self._context.set_tensor_address(name, buf.data_ptr())

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

        # Use the current default CUDA stream so the runner's cuda.synchronize()
        # picks up our work.
        self._stream = torch.cuda.current_stream().cuda_stream

    def warmup(self, input_shape: tuple[int, ...], n_iter: int = 20) -> None:
        """Run dummy inferences to stabilize TRT internals."""
        assert self._context is not None, "Call load() before warmup()."
        if not self._input_names:
            raise RuntimeError("Engine has no input tensors.")

        # Fill input buffer with random data once; warmup just keeps calling.
        in_name = self._input_names[0]
        in_buf = self._buffers[in_name]
        in_buf.copy_(torch.randn(*in_buf.shape, dtype=in_buf.dtype, device=in_buf.device))

        for _ in range(n_iter):
            self._context.execute_async_v3(stream_handle=self._stream)
        torch.cuda.synchronize()

    def infer(self, x: Any) -> Any:
        """Run one inference. Input x is copied into the pre-allocated input buffer.

        Returns:
            The output buffer tensor(s). Single output: returns the tensor.
            Multiple outputs: returns dict[name] -> tensor.
            The runner discards the return value; do not synchronize here.
        """
        assert self._context is not None, "Call load() before infer()."
        in_name = self._input_names[0]
        in_buf = self._buffers[in_name]

        if isinstance(x, torch.Tensor):
            in_buf.copy_(x, non_blocking=True)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

        self._context.execute_async_v3(stream_handle=self._stream)

        if len(self._output_names) == 1:
            return self._buffers[self._output_names[0]]
        return {n: self._buffers[n] for n in self._output_names}

    def teardown(self) -> None:
        """Drop engine, context, and buffers. Frees both TRT and torch memory."""
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None
        self._buffers.clear()
        self._input_names.clear()
        self._output_names.clear()
        self._stream = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @property
    def device_info(self) -> dict:
        info = {
            "name": self.name,
            "precision": self.precision,
            "engine": self._engine_path,
            "input_names": list(self._input_names),
            "output_names": list(self._output_names),
        }
        try:
            import tensorrt as trt
            info["tensorrt_version"] = trt.__version__
        except Exception:
            pass
        return info


if __name__ == "__main__":
    # Self-check: load FP32 engine, run 5 inferences, print shape and memory.
    import time

    backend = TensorRTBackend(name="trt_fp32", precision="fp32")
    print(f"Backend created: {backend!r}")

    engine_path = "models/trt/yolov8n_fp32.engine"
    print(f"Loading engine from {engine_path}...")
    t0 = time.perf_counter()
    backend.load(engine_path, device="cuda")
    t1 = time.perf_counter()
    print(f"  Load time: {t1 - t0:.2f} s")
    print(f"  Device info: {backend.device_info}")

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
        if isinstance(out, dict):
            shapes = {n: tuple(t.shape) for n, t in out.items()}
        else:
            shapes = tuple(out.shape)
        print(f"  iter {i}: {(t1 - t0) * 1000:.3f} ms, output shape: {shapes}")

    print(f"\nGPU memory before teardown: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    backend.teardown()
    print(f"GPU memory after teardown:  {torch.cuda.memory_allocated() / 1e6:.1f} MB")
