"""TensorRT INT8 calibrators.

Each class implements one calibration strategy from §4.2 of the project doc:
    EntropyCalibrator    -- KL-divergence (TRT default, IInt8EntropyCalibrator2)
    MinMaxCalibrator     -- per-tensor min/max (IInt8MinMaxCalibrator)
    PercentileCalibrator -- 99.9th percentile (IInt8LegacyCalibrator)

All share the same `CalibrationDataset` source and the same GPU buffer /
cache logic via `_base._CalibratorMixin`.
"""

from infra.convert.calibrators.entropy import EntropyCalibrator
from infra.convert.calibrators.minmax import MinMaxCalibrator
from infra.convert.calibrators.percentile import PercentileCalibrator

__all__ = [
    "EntropyCalibrator",
    "MinMaxCalibrator",
    "PercentileCalibrator",
]
