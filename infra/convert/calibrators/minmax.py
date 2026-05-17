"""TensorRT INT8 MinMax calibrator.

Implements `IInt8MinMaxCalibrator` -- per-tensor min/max calibration.
For each tensor, the scale factor is chosen so that the FP32 dynamic range
[min, max] exactly maps to INT8's [-128, 127] (or [0, 255] for unsigned).

When this differs from Entropy
------------------------------
- MinMax preserves outliers exactly: even a single extreme activation
  dictates the full range.
- Entropy (KL-div) clips outliers in favor of resolution for the bulk of
  the distribution. Usually better mAP, sometimes worse on activations
  with heavy tails.

Day 3 §4.2 compares the two strategies head-to-head.
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


class MinMaxCalibrator(_CalibratorMixin, trt.IInt8MinMaxCalibrator):
    """Per-tensor min/max INT8 calibrator.

    Parameters identical to EntropyCalibrator -- the only difference is the
    TRT base class.
    """

    def __init__(
        self,
        dataset: CalibrationDataset,
        cache_file: PathLike,
        input_name: str = "images",
    ):
        trt.IInt8MinMaxCalibrator.__init__(self)
        _CalibratorMixin.__init__(self, dataset, cache_file, input_name)
