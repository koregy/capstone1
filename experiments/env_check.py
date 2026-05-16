"""
환경 점검 스크립트 (Day 1 오전)

Jetson Orin Nano 환경에서 다음을 점검한다:
- Python / OS / JetPack 정보
- PyTorch + CUDA 사용 가능 여부
- ultralytics 동작 여부
- onnxruntime CUDA EP 동작 여부
- TensorRT 동작 여부 (trtexec, python binding)
- OpenCV + USB 카메라 인식 여부

실패해도 멈추지 않고 모든 항목을 시도한 뒤 요약을 출력한다.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Tuple


def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_system() -> None:
    section("1. System")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine: {platform.machine()}")
    # JetPack version (L4T)
    if os.path.exists("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release") as f:
            print(f"  L4T    : {f.read().strip().splitlines()[0]}")
    else:
        warn("/etc/nv_tegra_release not found (non-Jetson?)")


def check_torch() -> Tuple[bool, bool]:
    section("2. PyTorch / CUDA")
    try:
        import torch  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        fail(f"torch import failed: {e}")
        return False, False
    ok(f"torch {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        ok(f"CUDA available: device={torch.cuda.get_device_name(0)}")
        ok(f"CUDA version (torch built): {torch.version.cuda}")
        # quick op
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        ok(f"matmul on CUDA sanity: scalar={y:.2f}")
    else:
        fail("torch.cuda.is_available() == False")
    return True, cuda_ok


def check_ultralytics() -> bool:
    section("3. ultralytics (YOLOv8)")
    try:
        import ultralytics  # noqa: WPS433
        from ultralytics import YOLO  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        fail(f"ultralytics import failed: {e}")
        return False
    ok(f"ultralytics {ultralytics.__version__}")
    # try loading yolov8n (will download if missing — small file)
    try:
        model = YOLO("yolov8n.pt")
        ok(f"YOLO('yolov8n.pt') loaded: task={model.task}")
    except Exception as e:  # noqa: BLE001
        fail(f"YOLO load failed: {e}")
        return False
    return True


def check_onnxruntime() -> bool:
    section("4. ONNX Runtime")
    try:
        import onnxruntime as ort  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        fail(f"onnxruntime import failed: {e}")
        return False
    ok(f"onnxruntime {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"  Providers: {providers}")
    if "CUDAExecutionProvider" in providers:
        ok("CUDAExecutionProvider available")
    else:
        warn("CUDAExecutionProvider NOT available — falling back to CPU")
    if "TensorrtExecutionProvider" in providers:
        ok("TensorrtExecutionProvider available (bonus)")
    return True


def check_tensorrt() -> bool:
    section("5. TensorRT")
    # python binding
    try:
        import tensorrt as trt  # noqa: WPS433
        ok(f"tensorrt python binding {trt.__version__}")
    except Exception as e:  # noqa: BLE001
        warn(f"tensorrt python import failed: {e}")
    # trtexec binary
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        # common Jetson location
        candidate = "/usr/src/tensorrt/bin/trtexec"
        if os.path.exists(candidate):
            trtexec = candidate
    if trtexec:
        ok(f"trtexec found: {trtexec}")
        try:
            out = subprocess.check_output([trtexec, "--version"], stderr=subprocess.STDOUT, timeout=10)
            print(f"  trtexec --version: {out.decode().strip().splitlines()[0]}")
        except Exception as e:  # noqa: BLE001
            warn(f"trtexec --version failed: {e}")
    else:
        fail("trtexec not found in PATH or /usr/src/tensorrt/bin/")
    return True


def check_opencv_and_camera() -> bool:
    section("6. OpenCV + USB camera")
    try:
        import cv2  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        fail(f"cv2 import failed: {e}")
        return False
    ok(f"opencv {cv2.__version__}")
    # list /dev/video*
    devs = sorted([d for d in os.listdir("/dev") if d.startswith("video")])
    if devs:
        ok(f"/dev/video*: {devs}")
    else:
        warn("no /dev/video* devices found")
        return False
    # try opening the first one
    for d in devs:
        idx = int(d.replace("video", ""))
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                ok(f"video{idx} opened, frame {w}x{h}")
                cap.release()
                return True
            cap.release()
            warn(f"video{idx} opened but read() failed")
        else:
            warn(f"video{idx} could not be opened")
    return False


def main() -> int:
    print("Jetson Orin Nano — env_check.py")
    check_system()
    check_torch()
    check_ultralytics()
    check_onnxruntime()
    check_tensorrt()
    check_opencv_and_camera()
    print("\n[done] Review [FAIL]/[WARN] above before moving on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
