"""Measurement loop. Runs n_iter timed inferences against any backend.

Two measurement boundaries are supported via `include_io`:

  * include_io=False (default, "Boundary A" = GPU compute time):
      - Input: CUDA tensor (already on GPU)
      - Output: CUDA tensor (stays on GPU)
      - H2D/D2H copy is NOT included in latency
      - This is the standard "inference engine benchmark" -- TensorRT, MLPerf,
        and NVIDIA's published Jetson numbers all use this boundary.

  * include_io=True ("Boundary B" = host latency):
      - Input: numpy array (in CPU memory)
      - Output: numpy array (in CPU memory)
      - H2D + D2H copy IS included in latency
      - Equivalent to trtexec's "Host Latency" metric. Reflects the user
        scenario where data starts and ends in CPU memory.

Backends must accept either input form in their `infer()`. The runner
itself never converts during the timed loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from infra.backends.base import BaseBackend


def run_benchmark(
    backend: "BaseBackend",
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    n_warmup: int = 20,
    n_iter: int = 200,
    device: str = "cuda",
    seed: int = 42,
    include_io: bool = False,
) -> list[float]:
    """Run a measurement loop on a single backend.

    The same input is reused across all iterations to avoid memory allocation
    noise. CUDA synchronization is called before and after every timed
    iteration; without it, latency reads as artificially low because of
    async kernel queuing.

    Args:
        backend: Anything with .warmup(input_shape, n_iter) and .infer(x).
        input_shape: Input tensor shape. Default = (1, 3, 640, 640) for YOLOv8n.
        n_warmup: Warmup iterations before measurement starts. Min 20 recommended
            (first few iters initialize CUDA context and cuDNN auto-tuner).
        n_iter: Measured iterations.
        device: 'cuda' or 'cpu'. CPU mode skips cuda sync (for debugging only).
        seed: torch seed for deterministic dummy tensor.
        include_io: If True, pass a numpy array as input (Boundary B = host
            latency includes H2D/D2H). If False (default), pass a CUDA tensor
            (Boundary A = GPU compute only). The backend's `infer()` must
            accept whichever form is requested.

    Returns:
        List of n_iter latencies in ms. Warmup is NOT included.
    """
    torch.manual_seed(seed)

    # Build dummy CUDA tensor first (deterministic via seed)
    dummy_cuda = torch.randn(*input_shape, device=device)

    if include_io:
        # Boundary B: convert to numpy in CPU memory ONCE, free the GPU copy.
        # The backend's infer() is responsible for H2D/D2H during the timed
        # iterations.
        dummy = dummy_cuda.cpu().numpy()
        del dummy_cuda
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        # Boundary A: keep the CUDA tensor; backend's infer() runs without I/O.
        dummy = dummy_cuda

    is_cuda = device.startswith("cuda")

    # The backend's own warmup uses its native input form (typically CUDA);
    # this stabilizes cuDNN auto-tune, JIT, allocator, etc.
    backend.warmup(input_shape=input_shape, n_iter=n_warmup)

    # In include_io mode, run a short additional warmup with the actual numpy
    # input the timed loop will use, so any H2D-specific kernels also stabilize.
    if include_io and n_warmup > 0:
        extra = min(n_warmup, 10)
        for _ in range(extra):
            if is_cuda:
                torch.cuda.synchronize()
            backend.infer(dummy)
            if is_cuda:
                torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(n_iter):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        backend.infer(dummy)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    return latencies_ms


if __name__ == "__main__":
    # Self-check with a fake backend (no real model needed).
    class FakeBackend:
        name = "fake"
        def warmup(self, input_shape, n_iter):
            for _ in range(n_iter):
                _ = torch.randn(input_shape, device="cuda").sum()
                torch.cuda.synchronize()
        def infer(self, x):
            # Accept both numpy and torch CUDA tensor
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).cuda()
            return (x.flatten() @ x.flatten()).item()

    if not torch.cuda.is_available():
        print("CUDA not available, skipping self-check")
    else:
        fake = FakeBackend()
        lats_a = run_benchmark(fake, n_warmup=5, n_iter=50, include_io=False)
        lats_b = run_benchmark(fake, n_warmup=5, n_iter=50, include_io=True)
        print(f"Boundary A (GPU only):    mean = {sum(lats_a)/len(lats_a):.3f} ms")
        print(f"Boundary B (host inc IO): mean = {sum(lats_b)/len(lats_b):.3f} ms")
