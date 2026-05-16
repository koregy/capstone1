"""Latency statistics computation.

Pure functions called by all measurements.
Backend/model agnostic. Only depends on numpy.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_latency_stats(
    latencies_ms: Iterable[float],
    n_warmup_excluded: int = 0,
) -> dict:
    """Convert a list of latency measurements into a statistics dict.

    Args:
        latencies_ms: Per-iter latency in ms. Warmup must already be excluded.
        n_warmup_excluded: Number of warmup iters the caller dropped
            (recorded as metadata only; not used in computation).

    Returns:
        JSON-serializable dict with keys:
        n, n_warmup_excluded, mean_ms, std_ms, min_ms, max_ms,
        p50_ms, p90_ms, p95_ms, p99_ms, fps.

    Raises:
        ValueError: If latencies_ms is empty or contains negative values.
    """
    arr = np.asarray(list(latencies_ms), dtype=np.float64)

    if arr.size == 0:
        raise ValueError("latencies_ms is empty")
    if np.any(arr < 0):
        raise ValueError(f"latencies_ms contains negative values: min={arr.min()}")

    mean = float(arr.mean())
    stats = {
        "n": int(arr.size),
        "n_warmup_excluded": int(n_warmup_excluded),
        "mean_ms": mean,
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "fps": 1000.0 / mean if mean > 0 else float("inf"),
    }
    return stats


def format_stats_table(stats: dict, title: str = "") -> str:
    """Render stats dict as a human-readable text block.

    Used by baseline_demo to print results to the console right after measurement.
    """
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(f"  n         = {stats['n']} (warmup {stats['n_warmup_excluded']} excluded)")
    lines.append(f"  mean      = {stats['mean_ms']:.3f} ms  ({stats['fps']:.2f} FPS)")
    lines.append(f"  std       = {stats['std_ms']:.3f} ms")
    lines.append(f"  min / max = {stats['min_ms']:.3f} / {stats['max_ms']:.3f} ms")
    lines.append(f"  p50 / p90 = {stats['p50_ms']:.3f} / {stats['p90_ms']:.3f} ms")
    lines.append(f"  p95 / p99 = {stats['p95_ms']:.3f} / {stats['p99_ms']:.3f} ms")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-check: works without any backend.
    import random
    random.seed(42)
    fake = [25.0 + random.gauss(0, 0.5) for _ in range(200)]
    s = compute_latency_stats(fake, n_warmup_excluded=20)
    print(format_stats_table(s, title="self-check (synthetic ~N(25, 0.5))"))
