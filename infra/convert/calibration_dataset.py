"""Calibration dataset for TRT INT8 quantization.

Loads COCO val2017 images, applies YOLOv8-compatible letterbox preprocessing,
and yields batches in the ONNX input contract:

    np.ndarray, dtype=float32, shape=(B, 3, H, W), range=[0.0, 1.0], RGB, C-contiguous.

Verified against `models/onnx/yolov8n.onnx` (Day 2 export, opset 17):
    input "images", shape [1, 3, 640, 640], dtype float32.

Notes
-----
The `letterbox` function is exported separately because mAP evaluation
(`infra/benchmark/accuracy.py`, Day 3 afternoon) must use the *exact same*
preprocessing as calibration. Any drift between the two would conflate
"INT8 quantization error" with "preprocessing mismatch error".

§12.8-4 (capstone1 doc): calibration set and mAP set must NOT overlap.
This module does not enforce that on its own -- callers must pass
non-overlapping slices of the sorted file list. The recommended split is:
    paths = sorted(Path("data/coco_val/images").glob("*.jpg"))
    calib_paths = paths[0:500]    # entropy_500, minmax_500, percentile_500
    eval_paths  = paths[500:1500] # mAP measurement (1000 images)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence, Tuple, Union

import cv2
import numpy as np


PathLike = Union[str, Path]


def letterbox(
    img: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize+pad an image to `new_shape` while preserving aspect ratio.

    Mirrors ultralytics 8.4's `LetterBox(auto=False, scaleup=True, center=True)`
    behavior, which is what `yolo.predict()` uses by default and therefore
    what the exported ONNX expects.

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (H, W, 3), dtype uint8, BGR (cv2 native).
    new_shape : int or (H, W)
        Target shape. Int means square.
    color : (B, G, R)
        Padding fill value. ultralytics default = 114 (gray).

    Returns
    -------
    np.ndarray
        Letterboxed image, shape (new_H, new_W, 3), dtype uint8, BGR.
        Color space conversion to RGB is done later in `preprocess()`.
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h0, w0 = img.shape[:2]
    # Scale ratio (new / old); choose smaller dim so the image fits inside new_shape
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (W, H) for cv2.resize

    # Padding to reach new_shape, centered
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color,
    )
    return img


def preprocess(img_path: PathLike, img_size: int = 640) -> np.ndarray:
    """Load one image from disk and produce the model's input tensor (single sample).

    Returns
    -------
    np.ndarray
        Shape (3, H, W), dtype float32, range [0.0, 1.0], RGB, C-contiguous.
        Note: NOT batched yet -- caller (CalibrationDataset) stacks.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed: {img_path}")
    img = letterbox(img, new_shape=img_size)
    # BGR -> RGB, HWC -> CHW, uint8 -> float32 / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)                     # HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img


class CalibrationDataset:
    """Iterable batch source for TRT INT8 calibrators.

    Parameters
    ----------
    image_paths : sequence of Path or str
        Sorted, deterministic list of image paths. Caller is responsible for
        slicing to avoid overlap with the mAP evaluation set (§12.8-4).
    batch_size : int
        Number of images per batch. Must match the calibrator's batch_size
        (which must match the engine's optimization profile / static batch dim,
        i.e. 1 for our Day 2 ONNX export).
    img_size : int
        Letterbox target. Must match the model's input H/W (640 for yolov8n).

    Lifecycle
    ---------
    - Iterate with `for batch in dataset: ...` -- yields np.ndarray (B, 3, H, W).
    - Last batch may be smaller than batch_size if `len(image_paths) % batch_size != 0`.
      TRT calibrators typically expect a fixed batch size, so the dataset *drops*
      the remainder by default to avoid silent shape mismatches at calibration time.
      Set `drop_last=False` only if you know the calibrator handles ragged batches.
    - `reset()` rewinds the iterator (some calibrators may re-iterate).
    """

    def __init__(
        self,
        image_paths: Sequence[PathLike],
        batch_size: int = 1,
        img_size: int = 640,
        drop_last: bool = True,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not image_paths:
            raise ValueError("image_paths is empty")
        # Materialize once so len() works and the order is fixed.
        self.paths: List[Path] = [Path(p) for p in image_paths]
        self.batch_size = batch_size
        self.img_size = img_size
        self.drop_last = drop_last
        self._cursor = 0

        n_full = len(self.paths) // batch_size
        self.n_batches = n_full if drop_last else (
            n_full + (1 if len(self.paths) % batch_size else 0)
        )
        self._dropped = len(self.paths) - n_full * batch_size if drop_last else 0

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """The shape of each batch this dataset yields."""
        return (self.batch_size, 3, self.img_size, self.img_size)

    @property
    def dropped(self) -> int:
        """How many trailing images are skipped due to drop_last=True."""
        return self._dropped

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[np.ndarray]:
        self._cursor = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._cursor >= self.n_batches:
            raise StopIteration
        start = self._cursor * self.batch_size
        end = start + self.batch_size
        # drop_last=False can produce a smaller last batch
        batch_paths = self.paths[start:end]
        samples = [preprocess(p, self.img_size) for p in batch_paths]
        batch = np.stack(samples, axis=0)
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        self._cursor += 1
        return batch

    def reset(self) -> None:
        """Rewind to the start of the dataset."""
        self._cursor = 0


def list_coco_val_images(images_dir: PathLike = "data/coco_val/images") -> List[Path]:
    """Return COCO val2017 image paths in deterministic (sorted) order.

    The COCO val2017 filenames are zero-padded 12-digit IDs
    (e.g. `000000000139.jpg`), so lexical sort == numerical sort == COCO's
    canonical order. Callers can then take `[0:500]` for calibration and
    `[500:1500]` for mAP without further sort logic.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    paths = sorted(images_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no .jpg files in {images_dir}")
    return paths
