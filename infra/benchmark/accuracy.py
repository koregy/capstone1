"""COCO mAP measurement for object detection backends.

Parallel to infra/benchmark/runner.py + metrics.py but for accuracy instead of
latency. The pipeline is:

    image_path
        -> preprocess_image (LetterBox + RGB + CHW + float/255)
        -> backend.infer(x)               # Boundary A: torch.Tensor CUDA in
        -> postprocess_predictions
              -> normalize output dtype to torch.Tensor (B, 84, 8400)
              -> non_max_suppression(conf=0.001, iou=0.7, max_det=300, nc=80)
              -> scale_boxes back to original image coords
              -> xyxy -> xywh
              -> COCO category_id remap (0..79 -> 1..90 with gaps)
        -> list of COCO-format detection dicts
        -> aggregate over all images
        -> COCO.loadRes() + COCOeval(imgIds=our_subset)
        -> mAP_50_95, mAP_50, mAP_75, mAP_small/medium/large

All backends share this pipeline so accuracy differences are attributable
solely to the forward output's numerical differences (i.e. quantization),
not to NMS or preprocessing implementation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


# YOLO class index (0..79) -> COCO category_id (1..90 with gaps).
# Standard mapping; same across all YOLOv8 COCO checkpoints.
COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
assert len(COCO80_TO_COCO91) == 80


# ---------------------------------------------------------------------------
# 1) Load COCO eval subset
# ---------------------------------------------------------------------------

def load_coco_eval_subset(
    ann_file: str | Path,
    images_dir: str | Path,
    start: int = 500,
    count: int = 1000,
) -> tuple[list[dict], COCO]:
    """Return (image_records, coco_gt) for [start:start+count] of sorted img IDs.

    image_records: list of dicts with keys
        id, file_name, path, height, width.

    Sorted by id so the subset is reproducible (same on every run).
    Skips records whose image file is missing on disk (with a warning).
    """
    ann_file = str(ann_file)
    images_dir = Path(images_dir)

    coco_gt = COCO(ann_file)
    all_ids = sorted(coco_gt.getImgIds())
    subset_ids = all_ids[start : start + count]

    records: list[dict] = []
    missing = 0
    for iid in subset_ids:
        info = coco_gt.loadImgs(iid)[0]
        path = images_dir / info["file_name"]
        if not path.exists():
            missing += 1
            continue
        records.append({
            "id": iid,
            "file_name": info["file_name"],
            "path": str(path),
            "height": info["height"],
            "width": info["width"],
        })

    if missing:
        print(f"  [warn] {missing}/{len(subset_ids)} image files missing on disk")
    return records, coco_gt


# ---------------------------------------------------------------------------
# 2) Preprocess
# ---------------------------------------------------------------------------

def preprocess_image(
    img_path: str | Path,
    imgsz: int = 640,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """Read image, LetterBox to (imgsz, imgsz), convert to backend input format.

    Returns:
        x: torch.Tensor of shape (1, 3, imgsz, imgsz), CUDA, float32, [0, 1].
        meta: {
            'orig_shape': (h, w) of original image,
            'letterbox_shape': (imgsz, imgsz),
            'image_id': not set here, caller fills.
        }

    The LetterBox is configured with auto=False (no auto-stride trim),
    scaleup=True (small images get enlarged), padding_value=114 (YOLO default).
    BGR -> RGB conversion happens after LetterBox, then HWC -> CHW.
    """
    img_bgr = cv2.imread(str(img_path))  # HWC, BGR, uint8
    if img_bgr is None:
        raise FileNotFoundError(f"cv2.imread failed: {img_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=True, padding_value=114)
    img_lb = lb(image=img_bgr)  # HWC, BGR, uint8, padded to (imgsz, imgsz)

    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))  # 3, imgsz, imgsz

    x = torch.from_numpy(img_chw).to(device=device, dtype=torch.float32)
    x /= 255.0
    x = x.unsqueeze(0)  # (1, 3, imgsz, imgsz)

    meta = {
        "orig_shape": (orig_h, orig_w),
        "letterbox_shape": (imgsz, imgsz),
    }
    return x, meta


# ---------------------------------------------------------------------------
# 3) Postprocess: raw output -> COCO-format detections
# ---------------------------------------------------------------------------

def _normalize_output(raw: Any, device: str = "cuda") -> torch.Tensor:
    """Coerce backend output to torch.Tensor (B, 84, 8400) on `device`.

    Backends return different shapes:
      * PyTorch (raw forward, no NMS): torch.Tensor (1, 84, 8400) or tuple/list
      * ONNXRuntime: np.ndarray or list of np.ndarray
      * TensorRT: torch.Tensor (single output) or dict[name -> Tensor]
    """
    # Unwrap containers
    if isinstance(raw, (tuple, list)):
        # Common pattern: (predictions, ...) where the rest is feature maps.
        raw = raw[0]
    if isinstance(raw, dict):
        # Single-output engine -> pick the only value.
        if len(raw) == 1:
            raw = next(iter(raw.values()))
        else:
            # Multi-output: pick the (B, 84, 8400)-shaped one.
            cand = [v for v in raw.values()
                    if hasattr(v, "shape") and len(v.shape) == 3 and v.shape[1] == 84]
            if len(cand) != 1:
                raise ValueError(f"Cannot disambiguate multi-output dict: {[v.shape for v in raw.values()]}")
            raw = cand[0]

    # To torch
    if isinstance(raw, np.ndarray):
        raw = torch.from_numpy(raw)
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Unsupported raw output type: {type(raw)}")

    if raw.device.type != device.split(":")[0]:
        raw = raw.to(device)

    # Sanity: expect (B, 84, 8400)
    if raw.dim() != 3 or raw.shape[1] != 84:
        raise ValueError(f"Unexpected output shape {tuple(raw.shape)}; expected (B, 84, 8400)")

    # Clone to escape "inference tensor" state (PyTorch backend uses
    # torch.inference_mode internally; NMS does in-place ops on prediction[..., :4]
    # and rejects inference tensors). Clone is cheap for (1, 84, 8400) float.
    return raw.clone()


def postprocess_predictions(
    raw_output: Any,
    meta: dict,
    image_id: int,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    nc: int = 80,
    device: str = "cuda",
) -> list[dict]:
    """raw_output -> NMS -> scale to original -> xywh -> COCO format dicts.

    Returns list of dicts: {image_id, category_id, bbox: [x,y,w,h], score}.
    Length is 0..max_det.
    """
    pred = _normalize_output(raw_output, device=device)

    # NMS expects (B, 84, num_boxes); returns list[Tensor(N, 6)] = (x1,y1,x2,y2,conf,cls).
    nms_out = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        nc=nc,
        agnostic=False,
        multi_label=False,
    )
    det = nms_out[0]  # batch size is 1
    if det is None or det.shape[0] == 0:
        return []

    # Scale boxes from letterbox space back to original image space.
    # scale_boxes(img1_shape, boxes, img0_shape) -- img1 = letterbox, img0 = original.
    boxes_xyxy = det[:, :4].clone()
    boxes_xyxy = scale_boxes(
        meta["letterbox_shape"],
        boxes_xyxy,
        meta["orig_shape"],
    )

    # xyxy -> xywh (COCO format: top-left x, y, width, height).
    # Compute directly to avoid relying on ultralytics.xyxy2xywh, which returns
    # CENTER xywh despite its docstring claiming top-left.
    boxes_xywh = boxes_xyxy.clone()
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]   # w = x2 - x1
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]   # h = y2 - y1
    # boxes_xywh[:, 0] and [:, 1] are already x1, y1 (top-left), no change.

    scores = det[:, 4]
    classes = det[:, 5].long()

    # Move to CPU once.
    boxes_xywh = boxes_xywh.cpu().tolist()
    scores = scores.cpu().tolist()
    classes = classes.cpu().tolist()

    detections: list[dict] = []
    for box, score, cls in zip(boxes_xywh, scores, classes):
        x, y, w, h = box
        detections.append({
            "image_id": int(image_id),
            "category_id": COCO80_TO_COCO91[int(cls)],
            "bbox": [round(float(x), 3), round(float(y), 3),
                     round(float(w), 3), round(float(h), 3)],
            "score": round(float(score), 5),
        })
    return detections


# ---------------------------------------------------------------------------
# 4) Top-level: evaluate one backend
# ---------------------------------------------------------------------------

def evaluate_backend(
    backend,
    image_records: list[dict],
    coco_gt: COCO,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    nc: int = 80,
    device: str = "cuda",
    progress_interval: int = 100,
) -> dict:
    """Run backend over all images, accumulate detections, run COCOeval.

    The backend MUST already be loaded. We do not call load() or teardown()
    here; the caller (accuracy_demo.py) owns the backend lifecycle so it
    matches baseline_demo's pattern.

    Returns dict with mAP metrics plus diagnostics.
    """
    # Brief inference warmup with random tensor (avoid first-image bias).
    warmup_x = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)
    for _ in range(5):
        _ = backend.infer(warmup_x)
    torch.cuda.synchronize()

    all_detections: list[dict] = []
    inference_time_total = 0.0
    n = len(image_records)

    t_loop_start = time.perf_counter()
    for idx, rec in enumerate(image_records):
        x, meta = preprocess_image(rec["path"], imgsz=imgsz, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw = backend.infer(x)
        torch.cuda.synchronize()
        inference_time_total += time.perf_counter() - t0

        dets = postprocess_predictions(
            raw_output=raw,
            meta=meta,
            image_id=rec["id"],
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            nc=nc,
            device=device,
        )
        all_detections.extend(dets)

        if progress_interval and (idx + 1) % progress_interval == 0:
            elapsed = time.perf_counter() - t_loop_start
            mean_infer_ms = inference_time_total / (idx + 1) * 1000
            print(f"    [{idx + 1}/{n}] mean infer={mean_infer_ms:.2f} ms, "
                  f"detections so far={len(all_detections)}, "
                  f"elapsed={elapsed:.1f}s")

    loop_time = time.perf_counter() - t_loop_start
    print(f"  Inference loop done: {loop_time:.1f}s, {len(all_detections)} detections")

    # ---- COCOeval ----
    if not all_detections:
        print("  [warn] No detections produced; skipping COCOeval.")
        return {
            "mAP_50_95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
            "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
            "n_images": n,
            "n_detections": 0,
            "eval_time_sec": 0.0,
            "inference_time_sec": inference_time_total,
        }

    subset_ids = [r["id"] for r in image_records]
    t_eval_start = time.perf_counter()

    coco_dt = coco_gt.loadRes(all_detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = subset_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    eval_time = time.perf_counter() - t_eval_start
    s = coco_eval.stats  # length 12: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl

    return {
        "mAP_50_95":  float(s[0]),
        "mAP_50":     float(s[1]),
        "mAP_75":     float(s[2]),
        "mAP_small":  float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large":  float(s[5]),
        "n_images": n,
        "n_detections": len(all_detections),
        "eval_time_sec": round(eval_time, 3),
        "inference_time_sec": round(inference_time_total, 3),
    }
