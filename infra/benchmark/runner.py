"""Backend-agnostic measurement loop.

Calls backend.warmup() and backend.infer() with CUDA synchronization on both
sides of every timed iteration. Returns the raw latency list; statistics are
computed separately by metrics.compute_latency_stats().
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
) -> list[float]:
    """Run a measurement loop on a single backend.

    The same input tensor is reused across all iterations to avoid memory
    allocation noise. CUDA synchronization is called before and after every
    timed iteration; without it, latency reads as artificially low because of
    async kernel queuing.

    Args:
        backend: Anything with .warmup(input_shape, n_iter) and .infer(x).
        input_shape: Input tensor shape. Default = (1, 3, 640, 640) for YOLOv8n.
        n_warmup: Warmup iterations before measurement starts. Min 20 recommended
            (first few iters initialize CUDA context and cuDNN auto-tuner).
        n_iter: Measured iterations.
        device: 'cuda' or 'cpu'. CPU mode skips cuda sync (for debugging only).
        seed: torch seed for deterministic dummy tensor.

    Returns:
        List of n_iter latencies in ms. Warmup is NOT included.
    """
    torch.manual_seed(seed)

    # Build the dummy input ONCE, then reuse. torch.randn allocates new memory
    # every call, which adds noise; building once keeps the timed loop clean.
    dummy = torch.randn(*input_shape, device=device)

    is_cuda = device.startswith("cuda")

    # Warmup: let the backend stabilize (cuDNN auto-tune, JIT, memory allocator).
    # The backend may internally call its own infer() loop here.
    backend.warmup(input_shape=input_shape, n_iter=n_warmup)

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
            # Simulate work: small matmul on GPU.
            return (x.flatten() @ x.flatten()).item()

    if not torch.cuda.is_available():
        print("CUDA not available, skipping self-check")
    else:
        fake = FakeBackend()
        lats = run_benchmark(fake, n_warmup=5, n_iter=50)
        print(f"Self-check: {len(lats)} iters, mean = {sum(lats)/len(lats):.3f} ms")
        print(f"  first 3:   {lats[:3]}")
        print(f"  last 3:    {lats[-3:]}")
