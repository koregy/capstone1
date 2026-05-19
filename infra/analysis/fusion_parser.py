"""TensorRT verbose log parser for layer fusion analysis.

Parses `trtexec --verbose` build logs and extracts:
  * Layer-count pipeline per build phase (e.g. "After vertical fusions: 129 layers")
  * Individual fusion decisions (e.g. "PointWiseFusion: Fusing X with Y")
  * Fusion type counts (PointWiseFusion: 65, GenericConvActFusion: 44, ...)
  * Build metadata (precision, phase count, final layer count)

INT8 builds have TWO phases:
  Phase 1 = calibration run (FP32-style network, no vertical fusions applied,
            quant/dequant layers inserted -> layer count GROWS)
  Phase 2 = real engine build using collected calibration scales
            (full fusion pipeline, layer count SHRINKS)

FP32/FP16 builds have only one phase. The parser detects this from the
number of "Original: N layers" markers (one per phase).

The parser is intentionally line-oriented and split into small functions
(one per concern) so each piece can be unit-tested in isolation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Regex patterns — kept as module-level constants for grep-ability.
# ---------------------------------------------------------------------------

# Pipeline lines look like:
#   [05/17/2026-05:06:16] [V] [TRT] After vertical fusions: 129 layers
#   [V] After vertical fusions: 124 layers
# We match the optional timestamp/level prefix then "Original: N layers" or
# "After <something>: N layers".
_RE_PIPELINE = re.compile(
    r"^(?:\[[^\]]*\]\s*)*\[V\](?:\s*\[TRT\])?\s+"
    r"(Original|After [^:]+):\s+(\d+)\s+layers\s*$"
)

# Fusion decision lines look like:
#   [V] [TRT] PointWiseFusion: Fusing PWN(/model.0/act/Sigmoid) with PWN(/model.0/act/Mul)
#   [V] GenericConvActFusion: Fusing /model.0/conv/Conv with PWN(...)
_RE_FUSION = re.compile(
    r"^(?:\[[^\]]*\]\s*)*\[V\](?:\s*\[TRT\])?\s+"
    r"([A-Z][a-zA-Z]+Fusion):\s+Fusing\s+(.+?)\s+with\s+(.+?)\s*$"
)

# Precision detection from the trtexec command line that's echoed at the top
# of every verbose log:
#   &&&& RUNNING TensorRT.trtexec [TensorRT v100300] # /usr/src/tensorrt/bin/trtexec ... --fp16
_RE_CMDLINE = re.compile(r"&&&& RUNNING.*trtexec\s+(.+)$")

# Engine-build elapsed time:
#   [I] Engine built in 480.55 sec.
# or per the python-stdout wrapper:
#   [trt-int8] build finished in 649.7s
_RE_BUILD_TIME = re.compile(
    r"(?:Engine built in|build finished in)\s+([\d.]+)\s*(?:sec|s)\b"
)


# ---------------------------------------------------------------------------
# Per-concern parsers
# ---------------------------------------------------------------------------

def _parse_layer_pipeline(lines: list[str]) -> list[list[tuple[str, int]]]:
    """Extract layer-count pipeline, split into phases.

    A new phase starts at each "Original: N layers" line. Within a phase,
    every subsequent "After <stage>: N layers" line is collected in order.
    Stages that appear more than once within the SAME phase are disambiguated
    with an occurrence index suffix (e.g. "After dupe layer removal #1",
    "After dupe layer removal #2") so the full pipeline order is preserved.
    "Original" never repeats inside a single phase, so it's never suffixed.

    Returns:
        List of phases. Each phase is a list of (stage_name, layer_count) tuples.
    """
    phases: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    seen_in_phase: dict[str, int] = {}

    def _push(stage: str, count: int) -> None:
        seen_in_phase[stage] = seen_in_phase.get(stage, 0) + 1
        idx = seen_in_phase[stage]
        # Only suffix on the SECOND and later occurrences so common case
        # (1 occurrence per phase) stays unchanged.
        if idx > 1:
            current.append((f"{stage} #{idx}", count))
            # Also retroactively rename the first occurrence to "#1" if we
            # just hit "#2", so the table doesn't mix unsuffixed and "#2".
            if idx == 2:
                for i, (s, c) in enumerate(current[:-1]):
                    if s == stage:
                        current[i] = (f"{stage} #1", c)
                        break
        else:
            current.append((stage, count))

    for line in lines:
        m = _RE_PIPELINE.match(line.rstrip("\n"))
        if not m:
            continue
        stage, count = m.group(1), int(m.group(2))
        if stage == "Original":
            if current:
                phases.append(current)
            current = []
            seen_in_phase = {}
            _push(stage, count)
        else:
            _push(stage, count)
    if current:
        phases.append(current)
    return phases


def _parse_fusions(lines: list[str]) -> list[dict[str, str]]:
    """Extract individual fusion decisions.

    Each fusion line names a fusion type and two operands. We return
    them as ordered dicts so JSON output is human-readable.
    """
    out: list[dict[str, str]] = []
    for line in lines:
        m = _RE_FUSION.match(line.rstrip("\n"))
        if not m:
            continue
        out.append({
            "type": m.group(1),
            "a": m.group(2),
            "b": m.group(3),
        })
    return out


def _count_fusion_types(fusions: list[dict[str, str]]) -> dict[str, int]:
    """Tally fusion decisions by type. Sorted descending for readability."""
    c = Counter(f["type"] for f in fusions)
    return dict(c.most_common())


def _detect_precision(lines: list[str]) -> str:
    """Detect engine precision.

    Strategy (in order):
      1. trtexec command line flags (--int8 / --fp16)  — for trtexec builds
      2. Presence of "INT8 Inference Tensor scales"     — for Python TRT API builds
         (INT8 calibration uses Python API which doesn't echo a trtexec cmdline)
      3. Defaults to 'fp32' if a trtexec cmdline was seen but no flag
      4. 'unknown' if neither marker is present
    """
    cmdline_seen = False
    for line in lines:
        m = _RE_CMDLINE.search(line)
        if m:
            cmdline_seen = True
            cmd = m.group(1).lower()
            if "--int8" in cmd:
                return "int8"
            if "--fp16" in cmd:
                return "fp16"
            return "fp32"
        # Strategy 2: INT8 indicator line (Python API builds).
        if "INT8 Inference Tensor scales" in line:
            return "int8"
    return "fp32" if cmdline_seen else "unknown"


def _detect_build_time(lines: list[str]) -> float | None:
    """Return build elapsed time in seconds, if available."""
    for line in lines:
        m = _RE_BUILD_TIME.search(line)
        if m:
            return float(m.group(1))
    return None


def _final_layer_count(phases: list[list[tuple[str, int]]]) -> int | None:
    """Last layer count in the LAST phase = the engine's actual layer count."""
    if not phases:
        return None
    last_phase = phases[-1]
    if not last_phase:
        return None
    return last_phase[-1][1]


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def parse_build_log(log_path: str | Path) -> dict[str, Any]:
    """Parse one trtexec verbose log into a structured dict.

    Returns:
        {
            'log_path': '...',
            'meta': {
                'precision': 'int8',
                'phase_count': 2,
                'has_calibration_phase': True,
                'final_layer_count': 135,
                'build_time_sec': 649.7,
            },
            'build_phases': [
                {
                    'pipeline': [(stage_name, layer_count), ...],
                    'fusions': [{'type', 'a', 'b'}, ...],
                    'fusion_type_counts': {'PointWiseFusion': 65, ...},
                },
                ...
            ],
            'totals': {
                'fusion_type_counts': {...},   # summed across all phases
                'total_fusions': 117,
            },
        }
    """
    log_path = Path(log_path)
    with log_path.open(encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    phases_pipeline = _parse_layer_pipeline(lines)
    all_fusions = _parse_fusions(lines)
    precision = _detect_precision(lines)
    build_time = _detect_build_time(lines)

    # Fusion decisions are not tagged with their phase in the log, so we don't
    # try to split them per phase. We attach the full list to the LAST phase
    # (the real engine build) since that's where the fusions actually apply.
    build_phases: list[dict[str, Any]] = []
    for i, pipeline in enumerate(phases_pipeline):
        is_last = (i == len(phases_pipeline) - 1)
        fusions_here = all_fusions if is_last else []
        build_phases.append({
            "pipeline": pipeline,
            "fusions": fusions_here,
            "fusion_type_counts": _count_fusion_types(fusions_here),
        })

    return {
        "log_path": str(log_path),
        "meta": {
            "precision": precision,
            "phase_count": len(phases_pipeline),
            "has_calibration_phase": len(phases_pipeline) > 1,
            "final_layer_count": _final_layer_count(phases_pipeline),
            "build_time_sec": build_time,
        },
        "build_phases": build_phases,
        "totals": {
            "fusion_type_counts": _count_fusion_types(all_fusions),
            "total_fusions": len(all_fusions),
        },
    }


def save_parsed(parsed: dict[str, Any], out_path: str | Path) -> Path:
    """Persist parsed result as JSON. Returns the written Path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(parsed, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m infra.analysis.fusion_parser <log_path>")
        sys.exit(1)
    p = parse_build_log(sys.argv[1])
    print(json.dumps(p["meta"], indent=2))
    print(f"\nphases: {p['meta']['phase_count']}")
    for i, ph in enumerate(p["build_phases"]):
        print(f"\nphase {i}:")
        for stage, n in ph["pipeline"]:
            print(f"  {stage:40s} {n:6d} layers")
    print(f"\nfusion type totals: {p['totals']['fusion_type_counts']}")
