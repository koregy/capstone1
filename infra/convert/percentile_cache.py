"""Percentile INT8 calibration via cache injection.

Why this module exists
----------------------
TRT 10's `IInt8LegacyCalibrator` (the official percentile calibrator) does not
produce valid INT8 implementations for some layers in small models -- on
YOLOv8n it fails at the first convolution with
    [E] Could not find any implementation for node /model.0/conv/Conv ...
We confirmed this is a known TRT 10 limitation (LegacyCalibrator is in
soft-deprecation), not a bug in our calibrator code.

To get TRUE percentile-based INT8 calibration anyway, we exploit a property
of TRT's calibration cache: when a cache file is present, TRT skips the
calibration forward passes entirely and uses the scales in the cache as-is.
The cache file is plain text with format:

    TRT-100300-EntropyCalibration2
    tensor_name_1: <hex_BE_float32>     # scale = abs_max / 127  (NOT abs_max!)
    tensor_name_2: <hex_BE_float32>
    ...

(Confirmed empirically: sigmoid outputs in the cache are 1/127 = 0.00787,
 which matches the canonical scale = abs_max(=1.0) / 127 definition.)

So we:
  1. Run the ONNX model on calibration data ourselves (with onnxruntime).
  2. Collect histogram of |activations| per intermediate tensor.
  3. Compute the 99.9 percentile of |activations| for each tensor.
  4. Take an existing entropy_500.cache as a TEMPLATE (for tensor names
     and header) and overwrite each scale with our percentile/127.
  5. Hand that cache to `EntropyCalibrator2` -- TRT cache-hits and applies
     our percentile-derived scales without ever running its own algorithm.

This is "percentile calibration" in every meaningful sense: the INT8 scale
of each tensor is set to its 99.9-percentile absolute value divided by 127.
The only thing that's different from a hypothetical functional
`IInt8PercentileCalibrator` is the mechanism by which TRT receives the
scales -- through cache injection instead of through Python callbacks.

Reported finding
----------------
Day 3 §4.2: "Percentile calibration was implemented via cache injection
because TRT 10 IInt8LegacyCalibrator cannot produce valid INT8 tactics for
the first conv layer of YOLOv8n. The injected scales come from a direct
99.9 percentile measurement of |activations| over the same 500 calibration
images, providing an apples-to-apples comparison with entropy and minmax."
"""

from __future__ import annotations

import logging
import re
import struct
from pathlib import Path
from typing import Dict, Union

import numpy as np

# onnxruntime / onnx are imported lazily inside functions so the module can
# be loaded for cache-only operations (e.g. write_cache) without ORT installed.

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


# --------------------------------------------------------------------------
# Stage A: collect 99.9-percentile |activation| per intermediate tensor
# --------------------------------------------------------------------------

def expose_all_intermediates_as_outputs(onnx_path: PathLike) -> "onnx.ModelProto":
    """Load an ONNX model and add every intermediate tensor to graph.output.

    Returns a modified ModelProto. The original file on disk is unchanged.

    The TRT calibration cache contains scales for *every* tensor that
    becomes an INT8 quantization target, including activations between
    fused conv/silu nodes. Onnxruntime will expose all of those if we
    register them as graph outputs.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph

    # Existing outputs we keep as-is
    existing_output_names = {o.name for o in graph.output}

    # All intermediate tensors = (every node output) - (existing outputs)
    new_outputs = []
    seen = set(existing_output_names)
    for node in graph.node:
        for out_name in node.output:
            if out_name and out_name not in seen:
                # Make a ValueInfoProto for this tensor.
                # Shape inference would give us better metadata, but the
                # tensor name is the only thing onnxruntime strictly needs.
                vi = onnx.helper.make_tensor_value_info(
                    out_name, onnx.TensorProto.FLOAT, None,
                )
                new_outputs.append(vi)
                seen.add(out_name)

    graph.output.extend(new_outputs)
    log.info("expose_all_intermediates: added %d outputs (was %d, now %d)",
             len(new_outputs), len(existing_output_names), len(graph.output))
    return model


class _StreamingHistogram:
    """Per-tensor running absolute-value histogram with growing range.

    Each tensor gets one of these. We don't know the tensor's true max
    until we've seen all batches, so the histogram range expands on demand:
      * Initially: range = [0, observed_max * 1.2]
      * On batch i: if max(|new_batch|) > current upper bound,
                    expand the histogram (re-bin existing counts).
    """

    def __init__(self, n_bins: int = 8192):
        self.n_bins = n_bins
        self.counts = np.zeros(n_bins, dtype=np.int64)
        self.upper = 0.0   # current histogram upper bound (exclusive)
        self.total = 0

    def update(self, abs_values: np.ndarray) -> None:
        """abs_values: flattened |activations| of one batch for one tensor."""
        if abs_values.size == 0:
            return

        batch_max = float(abs_values.max())
        if batch_max == 0.0:
            # All zeros -- count them all in bin 0
            self.counts[0] += abs_values.size
            self.total += abs_values.size
            if self.upper == 0.0:
                self.upper = 1e-6  # tiny sentinel so subsequent calls don't divide by 0
            return

        # Expand range if needed
        target_upper = batch_max * 1.2
        if target_upper > self.upper:
            self._rebin(new_upper=target_upper)

        # Bin the new values
        bin_idx = np.clip(
            (abs_values / self.upper * self.n_bins).astype(np.int64),
            0, self.n_bins - 1,
        )
        np.add.at(self.counts, bin_idx, 1)
        self.total += abs_values.size

    def _rebin(self, new_upper: float) -> None:
        """Re-bin existing counts into a wider range."""
        if self.total == 0:
            self.upper = new_upper
            return
        # Existing bin i corresponds to range [i, i+1) * (self.upper / n_bins).
        # In the new histogram, bin centers move closer together.
        old_centers = (np.arange(self.n_bins) + 0.5) * self.upper / self.n_bins
        new_bin_idx = np.clip(
            (old_centers / new_upper * self.n_bins).astype(np.int64),
            0, self.n_bins - 1,
        )
        new_counts = np.zeros(self.n_bins, dtype=np.int64)
        np.add.at(new_counts, new_bin_idx, self.counts)
        self.counts = new_counts
        self.upper = new_upper

    def percentile(self, q: float) -> float:
        """q = 99.9 means 99.9 percentile (0..100 scale)."""
        if self.total == 0:
            return 0.0
        target = self.total * q / 100.0
        cumsum = np.cumsum(self.counts)
        bin_idx = int(np.searchsorted(cumsum, target))
        bin_idx = min(bin_idx, self.n_bins - 1)
        # Return upper edge of the bin (conservative; slightly overestimates
        # the percentile, which is safer for clipping)
        return (bin_idx + 1) * self.upper / self.n_bins


def collect_activation_percentiles(
    onnx_path: PathLike,
    dataset,   # CalibrationDataset
    percentile: float = 99.9,
    histogram_bins: int = 8192,
    providers: list = None,
) -> Dict[str, float]:
    """Forward-run the ONNX model on calibration data, return per-tensor percentile.

    Parameters
    ----------
    onnx_path : path to the ONNX model
    dataset : CalibrationDataset (yields (B,3,H,W) float32 numpy batches)
    percentile : default 99.9
    histogram_bins : bins per tensor (8192 = ~0.012% resolution)
    providers : onnxruntime providers list, default CUDA->CPU

    Returns
    -------
    Dict[tensor_name -> percentile of |activation|]
    """
    import onnxruntime as ort

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Stage A1: patch the model to expose all intermediates
    model = expose_all_intermediates_as_outputs(onnx_path)

    # Serialize to bytes (avoid touching disk) and load into ORT
    model_bytes = model.SerializeToString()

    # Some ORT versions warn about unused initializers when intermediates
    # are exposed; turn down log level to keep output clean.
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # warnings only

    session = ort.InferenceSession(
        model_bytes, sess_options=sess_opts, providers=providers,
    )
    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    log.info("collect_activation_percentiles: %d output tensors to track, "
             "providers=%s", len(out_names), session.get_providers())

    # Stage A2: forward all batches, accumulate histograms
    histograms: Dict[str, _StreamingHistogram] = {
        name: _StreamingHistogram(n_bins=histogram_bins) for name in out_names
    }

    n_total = len(dataset)
    for i, batch in enumerate(dataset):
        outputs = session.run(out_names, {in_name: batch})
        for name, arr in zip(out_names, outputs):
            histograms[name].update(np.abs(arr).ravel())
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            log.info("  forward: %d / %d batches", i + 1, n_total)

    # Stage A3: extract percentile per tensor
    result = {name: h.percentile(percentile) for name, h in histograms.items()}
    log.info("collected percentiles for %d tensors at q=%.1f", len(result), percentile)
    return result


# --------------------------------------------------------------------------
# Stage B: write a new calibration cache file from percentile values
# --------------------------------------------------------------------------

# Cache line format: "<tensor_name>: <hex_BE_float32>\n"
# Empirically verified against entropy_500.cache:
#   sigmoid outputs have hex 3c010a14 (BE) = 0.007876 ~= 1.0/127
#   => scale stored in cache = abs_max / 127

_CACHE_LINE_RE = re.compile(r"^([^:]+):\s+([0-9a-fA-F]+)\s*$")
INT8_RANGE = 127.0


def float_to_cache_hex(value: float) -> str:
    """float32 -> 8-char big-endian hex string (TRT cache format)."""
    return struct.pack(">f", float(value)).hex()


def cache_hex_to_float(hexstr: str) -> float:
    """8-char big-endian hex -> float32. For testing / inspection."""
    return struct.unpack(">f", bytes.fromhex(hexstr))[0]


def write_percentile_cache(
    template_cache_path: PathLike,
    percentile_thresholds: Dict[str, float],
    output_cache_path: PathLike,
) -> None:
    """Take an existing TRT calibration cache, replace each tensor's scale
    with `percentile_thresholds[tensor] / 127`, and write the result.

    The template cache provides:
      * the file header (e.g. "TRT-100300-EntropyCalibration2"),
      * the exact tensor names and their order, and
      * scale values for any tensor NOT in `percentile_thresholds` (we
        keep the original entropy scale as a fallback to avoid breaking
        the cache when our forward pass missed a tensor).

    Mismatched tensor names are logged but not fatal.
    """
    template_cache_path = Path(template_cache_path)
    output_cache_path = Path(output_cache_path)
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(template_cache_path) as f:
        lines = f.readlines()

    out_lines = []
    n_replaced = 0
    n_kept = 0
    n_unknown_in_template = 0
    available = set(percentile_thresholds.keys())

    for i, line in enumerate(lines):
        raw = line.rstrip("\n")
        if i == 0 and raw.startswith("TRT-"):
            # Header line -- keep as-is
            out_lines.append(line)
            continue

        m = _CACHE_LINE_RE.match(raw)
        if not m:
            # Blank or unrecognized -- preserve verbatim
            out_lines.append(line)
            continue

        name = m.group(1).strip()
        if name in percentile_thresholds:
            new_scale = percentile_thresholds[name] / INT8_RANGE
            new_hex = float_to_cache_hex(new_scale)
            out_lines.append(f"{name}: {new_hex}\n")
            n_replaced += 1
        else:
            # Tensor in template but not in our measurements -- keep original
            out_lines.append(line)
            n_kept += 1
            n_unknown_in_template += 1

    # Tensors we measured but template doesn't have
    measured_names = set(percentile_thresholds.keys())
    template_names = set()
    for line in lines:
        m = _CACHE_LINE_RE.match(line.rstrip("\n"))
        if m:
            template_names.add(m.group(1).strip())
    extra = measured_names - template_names
    if extra:
        log.warning("write_percentile_cache: %d measured tensors are NOT in "
                    "the template cache (they will be ignored): e.g. %s",
                    len(extra), list(extra)[:3])

    log.info("write_percentile_cache: replaced %d scales, kept %d original "
             "(%d template tensors had no percentile measurement), wrote %s",
             n_replaced, n_kept, n_unknown_in_template, output_cache_path)
    output_cache_path.write_text("".join(out_lines))


# --------------------------------------------------------------------------
# Convenience: end-to-end pipeline
# --------------------------------------------------------------------------

def build_percentile_cache(
    onnx_path: PathLike,
    template_cache_path: PathLike,
    dataset,   # CalibrationDataset
    output_cache_path: PathLike,
    percentile: float = 99.9,
    histogram_bins: int = 8192,
) -> Dict[str, float]:
    """Run Stage A + Stage B end-to-end."""
    thresholds = collect_activation_percentiles(
        onnx_path=onnx_path,
        dataset=dataset,
        percentile=percentile,
        histogram_bins=histogram_bins,
    )
    write_percentile_cache(
        template_cache_path=template_cache_path,
        percentile_thresholds=thresholds,
        output_cache_path=output_cache_path,
    )
    return thresholds
