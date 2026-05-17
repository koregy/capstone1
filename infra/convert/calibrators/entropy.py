"""TensorRT INT8 entropy calibrator.

Implements `IInt8EntropyCalibrator2` (KL-divergence based), which is TRT 10's
recommended default. The non-strategy-specific logic (batch supply, GPU
buffers, cache IO) lives in `_base._CalibratorMixin`.
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


class EntropyCalibrator(_CalibratorMixin, trt.IInt8EntropyCalibrator2):
    """KL-divergence-based INT8 calibrator (TRT default, recommended).

    Parameters
    ----------
    dataset : CalibrationDataset
    cache_file : Path or str
    input_name : str
        Network input tensor name (default "images" for our YOLOv8n ONNX).
    """

    def __init__(
        self,
        dataset: CalibrationDataset,
        cache_file: PathLike,
        input_name: str = "images",
    ):
        # Call TRT C++ base first, then the mixin. We do NOT use super()
        # because the TRT binding doesn't cooperate with MRO chains; explicit
        # is safer (see _base.py docstring).
        trt.IInt8EntropyCalibrator2.__init__(self)
        _CalibratorMixin.__init__(self, dataset, cache_file, input_name)
