"""Shared logic for all TRT INT8 calibrators.

The three calibrators we implement (Entropy / MinMax / Percentile) differ
only in which `trt.IInt8XxxCalibrator` base they inherit from. The rest --
batch supply, GPU buffer management, cache IO, progress logging -- is
identical. This module factors that identical part out.

Design note: the mixin does NOT inherit from any TRT type. The concrete
calibrators do:

    class EntropyCalibrator(_CalibratorMixin, trt.IInt8EntropyCalibrator2):
        def __init__(self, ...):
            trt.IInt8EntropyCalibrator2.__init__(self)   # call TRT base FIRST
            _CalibratorMixin.__init__(self, ...)         # then mixin state

We avoid `super().__init__()` chains because TRT's C++ binding classes
don't cooperate well with Python's MRO; calling each base's __init__
explicitly is the documented safe pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  -- side effect: initializes CUDA context
except ImportError as e:
    raise ImportError(
        "pycuda is required for INT8 calibration. "
        "On Jetson: sudo apt install python3-pycuda  (or pip install pycuda)."
    ) from e

from infra.convert.calibration_dataset import CalibrationDataset


log = logging.getLogger(__name__)
PathLike = Union[str, Path]


class _CalibratorMixin:
    """Common state and methods for all our INT8 calibrators.

    Holds: dataset iterator, pre-allocated device buffer, cache file path,
    and progress counter. Provides the four TRT-required methods.

    Concrete subclasses must:
      1. Inherit from this mixin AND a `trt.IInt8XxxCalibrator` base.
      2. Call the TRT base's `__init__()` first, then this mixin's.
      3. Optionally accept extra hyperparameters (e.g. percentile's quantile).
    """

    def __init__(
        self,
        dataset: CalibrationDataset,
        cache_file: PathLike,
        input_name: str = "images",
    ):
        self.dataset = dataset
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.input_name = input_name

        self._iter = iter(self.dataset)
        self._batch_index = 0
        self._total_batches = len(self.dataset)

        # One reusable device buffer. float32 = 4 bytes.
        b, c, h, w = self.dataset.shape
        self._nbytes = b * c * h * w * 4
        self._device_input = cuda.mem_alloc(self._nbytes)

        log.info(
            "%s: %d batches of shape %s, cache=%s",
            type(self).__name__, self._total_batches,
            self.dataset.shape, self.cache_file,
        )

    # -- TRT-required interface --

    def get_batch_size(self) -> int:
        return self.dataset.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self.input_name not in names:
            raise RuntimeError(
                f"Calibrator expected input name '{self.input_name}' but TRT "
                f"asked for {names}. Check the ONNX graph input name."
            )

        try:
            host_batch = next(self._iter)
        except StopIteration:
            log.info("%s: exhausted after %d batches",
                     type(self).__name__, self._batch_index)
            return None

        if host_batch.nbytes != self._nbytes:
            raise RuntimeError(
                f"Batch {self._batch_index} has nbytes={host_batch.nbytes}, "
                f"expected {self._nbytes}. Likely batch_size or img_size mismatch."
            )

        cuda.memcpy_htod(self._device_input, host_batch)

        self._batch_index += 1
        if self._batch_index % 50 == 0 or self._batch_index == self._total_batches:
            log.info("%s: %d / %d batches",
                     type(self).__name__,
                     self._batch_index, self._total_batches)

        return [int(self._device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file.is_file():
            data = self.cache_file.read_bytes()
            log.info(
                "%s: reusing cache %s (%d bytes) -- calibration forward passes will be SKIPPED",
                type(self).__name__, self.cache_file, len(data),
            )
            return data
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_file.write_bytes(cache)
        log.info("%s: wrote cache %s (%d bytes)",
                 type(self).__name__, self.cache_file, len(cache))

    def __del__(self):
        try:
            if hasattr(self, "_device_input") and self._device_input is not None:
                self._device_input.free()
                self._device_input = None
        except Exception:
            pass
