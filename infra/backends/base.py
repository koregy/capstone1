"""Abstract base class for all inference backends.

Every backend (PyTorch, ONNX Runtime, TensorRT) must implement this interface.
The benchmark runner relies on .warmup() and .infer() only; .load() and
.teardown() are lifecycle hooks invoked by the caller (e.g. baseline_demo).

Design notes:
- infer() returns Any. The benchmark loop discards the return value; output
  postprocessing happens in a separate stage (Day 5 accuracy measurement).
  Forcing numpy conversion here would add a per-backend copy cost that
  contaminates latency comparison.
- precision is a string field, not a property. It is set at construction time
  and never changes.
- device_info is backend-specific metadata (engine path, providers, etc.)
  forwarded to the reporter's `extra` field.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseBackend(ABC):
    """Common interface for inference backends.

    Subclasses set `name` and `precision` as class attributes or in __init__.
    Lifecycle: __init__ -> load() -> warmup() -> infer() x N -> teardown().
    """

    name: str = "base"           # e.g. "pytorch_fp32", "onnxrt_fp32", "trt_fp16"
    precision: str = "fp32"      # "fp32" | "fp16" | "int8"

    @abstractmethod
    def load(self, model_path: str | Path, device: str = "cuda") -> None:
        """Load the model into memory and prepare for inference.

        Backend-specific meaning of model_path:
            PyTorch: .pt weights file
            ONNX Runtime: .onnx model file
            TensorRT: .engine serialized engine file
        """

    @abstractmethod
    def warmup(self, input_shape: tuple[int, ...], n_iter: int = 20) -> None:
        """Run n_iter dummy inferences to stabilize the runtime.

        Must run with the same input shape that infer() will receive.
        Should leave CUDA in a synchronized state on return.
        """

    @abstractmethod
    def infer(self, x: Any) -> Any:
        """Run a single inference.

        Args:
            x: Input tensor. The runner passes a torch.Tensor on CUDA;
                each backend is responsible for converting to its native
                format (numpy for ORT, GPU pointer for TRT, etc.) WITHOUT
                that conversion being part of the timed region in real
                deployment. For benchmarking, conversion cost is included
                inside infer() and that is intentional: it reflects the
                end-to-end cost of using that backend.

        Returns:
            Backend-native output. Discarded by the runner. Used downstream
            only by accuracy measurement (Day 5).
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release GPU memory and any backend-specific resources.

        Important on Orin Nano (8 GB): the caller measures backends
        sequentially and calls teardown() between them to avoid OOM.
        """

    @property
    def device_info(self) -> dict:
        """Backend-specific metadata for reporting.

        Default returns just the name and precision. Subclasses override
        to add engine_path, providers, opset, etc.
        """
        return {
            "name": self.name,
            "precision": self.precision,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} precision={self.precision}>"


if __name__ == "__main__":
    # Self-check: verify ABC enforcement.
    try:
        b = BaseBackend()  # type: ignore[abstract]
    except TypeError as e:
        print(f"OK: BaseBackend cannot be instantiated directly.")
        print(f"    Error: {e}")
    else:
        print("FAIL: BaseBackend should not be directly instantiable.")

    # Minimal concrete subclass to verify the interface compiles.
    class _DummyBackend(BaseBackend):
        name = "dummy"
        precision = "fp32"

        def load(self, model_path, device="cuda"):
            self._loaded = True

        def warmup(self, input_shape, n_iter=20):
            pass

        def infer(self, x):
            return None

        def teardown(self):
            self._loaded = False

    d = _DummyBackend()
    d.load("fake/path")
    d.warmup((1, 3, 640, 640))
    d.infer(None)
    d.teardown()
    print(f"OK: Concrete subclass works. repr = {d!r}")
    print(f"OK: device_info = {d.device_info}")
