"""TensorRT INT8 Percentile calibrator.

Implements `IInt8LegacyCalibrator` with a user-specified quantile cutoff.
The "legacy" name is unfortunate -- it's still fully supported in TRT 10
and is the canonical way to get percentile-based calibration.

How it works
------------
1. Histogram each tensor's activations across all calibration batches.
2. Choose the threshold T such that `quantile` (e.g. 0.999 = 99.9%) of
   activation values fall below T.
3. Map [-T, T] to INT8 range. Activations above T are clipped.

This is a sweet spot between MinMax (keep all outliers, lose resolution)
and Entropy (KL-optimal clipping). It's often the best choice when
activations have moderate tails -- which is YOLOv8's typical regime.

Hyperparameters
---------------
- quantile (default 0.999): fraction of values to keep within the range.
- regression_cutoff (default 1.0): how aggressively the algorithm extrapolates
  on rare values. The TRT docs are sparse here; we keep the default.

Day 3 §4.2 compares this against Entropy and MinMax at the same 500-image
calibration set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

try:
    import tensorrt as trt
except ImportError as e:
    raise ImportError(
        "TensorRT is required for INT8 calibration. "
        "On Jetson (JetPack 6.1), TRT 10.3 is preinstalled at /usr/lib/python3*/dist-packages."
    ) from e

from infra.convert.calibration_dataset import CalibrationDataset
from infra.convert.calibrators._base import _CalibratorMixin


PathLike = Union[str, Path]


class PercentileCalibrator(_CalibratorMixin, trt.IInt8LegacyCalibrator):
    """Percentile-based INT8 calibrator.

    Parameters
    ----------
    dataset : CalibrationDataset
    cache_file : Path or str
    input_name : str
        Network input tensor name (default "images" for our YOLOv8n ONNX).
    quantile : float
        Fraction of activation values to keep within the calibrated range.
        Default 0.999 (99.9 percentile -- standard).
    regression_cutoff : float
        TRT-internal extrapolation aggressiveness. Default 1.0 (the TRT default).
    """

    def __init__(
        self,
        dataset: CalibrationDataset,
        cache_file: PathLike,
        input_name: str = "images",
        quantile: float = 0.999,
        regression_cutoff: float = 1.0,
    ):
        trt.IInt8LegacyCalibrator.__init__(self)
        _CalibratorMixin.__init__(self, dataset, cache_file, input_name)
        # These are required for IInt8LegacyCalibrator -- TRT will call them.
        self._quantile = quantile
        self._regression_cutoff = regression_cutoff

    # -- IInt8LegacyCalibrator additional interface --

    def get_quantile(self) -> float:
        return self._quantile

    def get_regression_cutoff(self) -> float:
        return self._regression_cutoff

    # Legacy calibrator also has these two cache hooks (separate from the
    # standard calibration cache -- TRT uses them for histogram persistence).
    def read_histogram_cache(self, length: int):
        # Not implemented: no separate histogram cache file. TRT will
        # rebuild histograms each run (still fast because it's only used
        # during build, not inference).
        return None

    def write_histogram_cache(self, ptr, length: int) -> None:
        # Same: skip.
        pass
