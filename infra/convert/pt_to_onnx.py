"""PyTorch YOLOv8 -> ONNX export.

A thin wrapper over ultralytics' YOLO.export(format='onnx', ...) that pins
the options needed for fair 5-way backend comparison:

  - static shape (dynamic=False)
  - batch=1, imgsz=640
  - opset 17 (well-supported by TensorRT 10.3)
  - simplify=True (onnxsim) to drop redundant ops
  - nms=False (NMS stays out of the graph; we benchmark raw forward only)

The resulting .onnx file is moved into models/onnx/ regardless of where
ultralytics drops it.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def export_yolo_to_onnx(
    weights_path: str | Path = "models/pt/yolov8n.pt",
    output_path: str | Path = "models/onnx/yolov8n.onnx",
    imgsz: int = 640,
    batch: int = 1,
    opset: int = 17,
    simplify: bool = True,
) -> Path:
    """Export a YOLOv8 .pt to ONNX with options fixed for benchmarking.

    Args:
        weights_path: Path to the .pt weights file.
        output_path: Final location for the .onnx file. Parent dir is created.
        imgsz: Square input size (e.g. 640).
        batch: Static batch size. Must be 1 for the camera scenario.
        opset: ONNX opset version. 17 works with TensorRT 10.3.
        simplify: Whether to apply onnxsim.

    Returns:
        Absolute path to the exported .onnx file.

    Raises:
        FileNotFoundError: weights_path missing.
        RuntimeError: export produced no .onnx file.
    """
    from ultralytics import YOLO  # heavy import, defer

    weights_path = Path(weights_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"[export] loading {weights_path.name}")
    yolo = YOLO(str(weights_path))

    print(f"[export] running ultralytics export")
    print(f"         imgsz={imgsz}, batch={batch}, opset={opset}, "
          f"simplify={simplify}, dynamic=False, nms=False, half=False")
    exported = yolo.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        opset=opset,
        simplify=simplify,
        dynamic=False,
        nms=False,
        half=False,
    )
    # `exported` is the path that ultralytics returns.
    exported = Path(exported).resolve()
    print(f"[export] ultralytics wrote: {exported}")

    if not exported.exists():
        # Fallback: ultralytics sometimes returns a stale path; search next to weights.
        guess = weights_path.with_suffix(".onnx")
        if guess.exists():
            exported = guess
        else:
            raise RuntimeError(
                f"Export reported success but no .onnx file found "
                f"(checked {exported} and {guess})."
            )

    # Move into models/onnx/.
    if exported.resolve() != output_path.resolve():
        print(f"[export] moving to {output_path}")
        shutil.move(str(exported), str(output_path))
    else:
        print(f"[export] already at target {output_path}")

    return output_path


def verify_onnx(onnx_path: str | Path) -> dict:
    """Run onnx.checker and extract basic graph metadata.

    Returns:
        dict with keys: ir_version, opset, input_shape, output_shapes,
        n_nodes, n_initializers, file_size_mb.
    """
    import onnx

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    def _shape_of(value_info) -> list:
        shape = []
        for d in value_info.type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)
            else:
                shape.append(d.dim_value)
        return shape

    info = {
        "ir_version": model.ir_version,
        "opset": [{"domain": op.domain or "ai.onnx", "version": op.version}
                  for op in model.opset_import],
        "input_name": model.graph.input[0].name,
        "input_shape": _shape_of(model.graph.input[0]),
        "output_shapes": {o.name: _shape_of(o) for o in model.graph.output},
        "n_nodes": len(model.graph.node),
        "n_initializers": len(model.graph.initializer),
        "file_size_mb": onnx_path.stat().st_size / (1024 * 1024),
    }
    return info


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Export YOLOv8 .pt to .onnx.")
    parser.add_argument("--weights", default="models/pt/yolov8n.pt")
    parser.add_argument("--output", default="models/onnx/yolov8n.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    out = export_yolo_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        imgsz=args.imgsz,
        batch=1,
        opset=args.opset,
        simplify=not args.no_simplify,
    )
    print(f"\n[export] done -> {out}")

    print("\n[verify] onnx.checker + graph metadata")
    info = verify_onnx(out)
    print(json.dumps(info, indent=2, default=str))
