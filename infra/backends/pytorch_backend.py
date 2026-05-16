"""PyTorch FP32 backend wrapping ultralytics YOLOv8n.

We access the underlying torch.nn.Module directly (yolo.model) and bypass
ultralytics' predict() / preprocessing / NMS. This is deliberate: the
benchmark must measure raw forward pass only. Postprocessing (NMS, class
filter) is backend-agnostic and would contaminate cross-backend comparison.

Day 5 accuracy measurement (mAP) uses a separate path that goes through
the full predict() pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from infra.backends.base import BaseBackend


class PyTorchBackend(BaseBackend):
    """FP32 PyTorch backend.

    Constructor does not load the model; call .load(weights_path) explicitly.
    """

    name = "pytorch_fp32"
    precision = "fp32"

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.device: str = "cuda"
        self._weights_path: str | None = None

        # Optimize for fixed input shape (batch=1, 640x640).
        # cuDNN benchmark mode: search best conv algo on first call, then cache.
        # Worth it because every iteration uses the same shape.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def load(self, model_path: str | Path, device: str = "cuda") -> None:
        """Load YOLOv8n weights and extract the underlying torch.nn.Module."""
        from ultralytics import YOLO  # heavy import; defer until needed

        self.device = device
        self._weights_path = str(model_path)

        yolo = YOLO(str(model_path))
        # yolo.model is the raw torch.nn.Module (DetectionModel).
        self.model = yolo.model
        self.model.eval()
        self.model.to(self.device)

        # Ensure all params are FP32 (ultralytics may load as FP16 on some configs).
        self.model.float()

    def warmup(self, input_shape: tuple[int, ...], n_iter: int = 20) -> None:
        """Run dummy forward passes to stabilize cuDNN and caches."""
        assert self.model is not None, "Call load() before warmup()."
        dummy = torch.randn(*input_shape, device=self.device)
        with torch.inference_mode():
            for _ in range(n_iter):
                _ = self.model(dummy)
        torch.cuda.synchronize()

    def infer(self, x: Any) -> Any:
        """Single forward pass. Input is a CUDA tensor (B, C, H, W)."""
        assert self.model is not None, "Call load() before infer()."
        with torch.inference_mode():
            out = self.model(x)
        return out

    def teardown(self) -> None:
        """Release GPU memory. Critical between backends on Orin Nano 8GB."""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @property
    def device_info(self) -> dict:
        return {
            "name": self.name,
            "precision": self.precision,
            "weights": self._weights_path,
            "device": self.device,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }


if __name__ == "__main__":
    # Self-check: load YOLOv8n, run 5 inferences, print output shape and mem.
    import time

    backend = PyTorchBackend()
    print(f"Backend created: {backend!r}")

    weights = "models/pt/yolov8n.pt"
    print(f"Loading weights from {weights}...")
    t0 = time.perf_counter()
    backend.load(weights, device="cuda")
    t1 = time.perf_counter()
    print(f"  Load time: {t1 - t0:.2f} s")
    print(f"  Device info: {backend.device_info}")

    input_shape = (1, 3, 640, 640)
    print(f"\nWarming up (5 iters, shape={input_shape})...")
    backend.warmup(input_shape, n_iter=5)
    print("  Warmup done.")

    print("\nRunning 3 timed inferences...")
    dummy = torch.randn(*input_shape, device="cuda")
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = backend.infer(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # YOLOv8 typically returns a tuple/list; first element is the pred tensor.
        if isinstance(out, (tuple, list)):
            shapes = [tuple(o.shape) if hasattr(o, "shape") else type(o).__name__ for o in out]
            print(f"  iter {i}: {(t1 - t0) * 1000:.3f} ms, output (tuple/list): {shapes}")
        else:
            print(f"  iter {i}: {(t1 - t0) * 1000:.3f} ms, output shape: {tuple(out.shape)}")

    print(f"\nGPU memory before teardown: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    backend.teardown()
    print(f"GPU memory after teardown:  {torch.cuda.memory_allocated() / 1e6:.1f} MB")
