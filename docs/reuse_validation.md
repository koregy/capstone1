# §4.5 Infrastructure Reusability Validation (YOLOv8s)

## Hypothesis

The measurement infrastructure (§4.1, §4.2) was developed on YOLOv8n. The
infrastructure's claim is **domain/model independence**: changing the model
should require zero code changes, only configuration. This section validates
that claim by applying the infrastructure to YOLOv8s — a 3.5× larger model
of the same family — with identical protocol.

## Procedure

Day 7 work, total wall-clock ~2.5 hours including ORT package recovery and
ORT-inversion diagnostics.

**Code changes**: 0 (zero) modifications to `infra/`, `experiments/baseline_demo.py`,
or `experiments/accuracy_demo.py`. One new utility script
`scripts/build_int8_engine.py` was authored as a generalization of the
ad-hoc YOLOv8n INT8 build path — this codifies the existing convention
rather than adds new logic.

**Configuration changes**: Two new YAML files, cloned from YOLOv8n
counterparts with `yolov8n` → `yolov8s` path substitution and separate
`results_dir`:
- `configs/benchmark_v8s.yaml` (5-way × 2-boundary latency)
- `configs/accuracy_v8s.yaml` (5-way mAP)

**Build pipeline** (sequential):
python3 infra/convert/pt_to_onnx.py --weights models/pt/yolov8s.pt --output models/onnx/yolov8s.onnx
python3 infra/convert/onnx_to_trt.py --onnx models/onnx/yolov8s.onnx --engine models/trt/yolov8s_fp32.engine --precision fp32 --log logs/trt_build_yolov8s_fp32.log
python3 infra/convert/onnx_to_trt.py --onnx models/onnx/yolov8s.onnx --engine models/trt/yolov8s_fp16.engine --precision fp16 --log logs/trt_build_yolov8s_fp16.log
python3 scripts/build_int8_engine.py --model yolov8s --calibrator entropy --num-calib 500 --log logs/trt_build_yolov8s_int8_entropy.log

**Measurement** (sequential):
python3 experiments/baseline_demo.py --config configs/benchmark_v8s.yaml --boundaries both
python3 experiments/accuracy_demo.py --config configs/accuracy_v8s.yaml

## Build Metadata

| stage              | yolov8n      | yolov8s      | ratio (s/n) |
|--------------------|--------------|--------------|-------------|
| pt file size       | 6.2 MB       | 22.6 MB      | 3.6×        |
| onnx file size     | ~12 MB       | 42.8 MB      | ~3.6×       |
| onnx n_nodes       | 248          | 233          | 0.94×       |
| trt FP16 build     | 492 s        | 629.8 s      | 1.28×       |
| trt INT8 build     | 648 s        | 739.8 s      | 1.14×       |
| trt FP16 engine    | ~7 MB        | 24.5 MB      | ~3.5×       |
| trt INT8 engine    | 5.0 MB       | 12.7 MB      | 2.5×        |

Note: ONNX node count is *lower* for yolov8s than yolov8n after simplification.
This is a graph-topology coincidence (yolov8s consolidates into fewer but
larger operations after `onnx-simplifier`); runtime cost depends on FLOPs per
node, not node count.

## Latency Results (5-way × 2-boundary, n=200, warmup=20)

### Boundary A (GPU compute only)

| backend       | yolov8n (ms) | yolov8s (ms) | s/n ratio | yolov8s speedup vs PT |
|---------------|--------------|--------------|-----------|------------------------|
| pytorch_fp32  | 19.75        | 19.77        | 1.00×     | 1.00× (baseline)       |
| onnxrt_fp32   | 13.53        | 22.66        | 1.67×     | 0.87× ⚠️ (slower than PT) |
| trt_fp32      |  7.32        | 13.18        | 1.80×     | 1.50×                  |
| trt_fp16      |  3.99        |  6.47        | 1.62×     | 3.06×                  |
| trt_int8      |  3.29        |  4.36        | 1.32×     | 4.53×                  |

(yolov8s values are 2nd measurement; both runs agreed within ±0.01 ms on
TRT/ORT backends. PT noise is ±0.27 ms — within its typical pattern.)

### Boundary B (host latency, includes H2D + D2H)

| backend       | yolov8n B-A (ms) | yolov8s B-A (ms) |
|---------------|-------------------|-------------------|
| pytorch_fp32  | +2.14             | +2.20             |
| onnxrt_fp32   | +1.26             | +1.19             |
| trt_fp32      | +1.31             | +1.36             |
| trt_fp16      | +1.32             | +1.29             |
| trt_int8      | +1.31             | +1.29             |

**Observation:** B-A delta is identical across models (±0.07 ms). H2D/D2H
transfer cost depends only on tensor size (640×640×3 float32 in, 84×8400
float32 out), which is identical across yolov8n and yolov8s. This
independently confirms the infrastructure's I/O accounting (§13.6.A2 of
Day 3) generalizes without modification.

## Accuracy Results (5-way mAP, COCO val [500:1500] = 1000 images)

| backend       | yolov8n mAP@.5:.95 | yolov8s mAP@.5:.95 | yolov8s Δ vs PT | yolov8s rel. loss |
|---------------|---------------------|---------------------|-------------------|---------------------|
| pytorch_fp32  | 0.3942              | 0.4716              | baseline          | —                   |
| onnxrt_fp32   | 0.3949              | 0.4717              | +0.0001           | +0.02%              |
| trt_fp32      | 0.3948              | 0.4717              | +0.0001           | +0.02%              |
| trt_fp16      | 0.3942              | 0.4716              |  0.0000           |  0.00% ← zero loss  |
| trt_int8      | 0.2658              | 0.3297              | -0.1419           | -30.1%              |

YOLOv8s baseline mAP@.5:.95 = 0.4716 (1000-image subset; matches published
Ultralytics checkpoint quality on full val).

**Key cross-model observations** (§4.2 findings generalize):
- **FP16 zero-loss reproduces exactly**: 0.4716 = 0.4716 (yolov8s), 0.3942 = 0.3942 (yolov8n).
- **TRT FP32 / ORT / PT 일치 reproduces**: all three within 0.0001.
- **INT8 entropy collapse reproduces**: -30% on yolov8s, -33% on yolov8n.
  The KL-divergence calibration's saturation bias on classification Sigmoid
  output (§4.2 "smoking gun") is a model-independent failure of the algorithm.

## Findings

### F1. Infrastructure reusability claim — supported

The infrastructure required **zero modifications to `infra/`** to measure
YOLOv8s. All five backends loaded, the `CalibrationDataset` accepted yolov8s
without changes, and the COCO mAP pipeline handled yolov8s output shape
`(1, 84, 8400)` — identical to yolov8n — directly. Only two YAML files
(path-substituted) and one generalization script (`build_int8_engine.py`)
were added.

### F2. Measurement discipline preserved across models

B-A delta identical across models (±0.07 ms) shows the infrastructure's
accounting of GPU vs host cost generalizes. The PT B-A (+2.20 ms) is larger
than TRT/ORT (+1.30 ms) for the same reason as yolov8n: PT's `ultralytics`
forward includes numpy↔tensor conversion overhead. The pattern reproduces.

### F3. Infrastructure's hard-fail guard caught a regression

During yolov8s ONNX export, the `ultralytics` package auto-installed
`onnxruntime==1.23.2` (CPU build, PyPI x86_64 wheel), shadowing the working
`onnxruntime-gpu==1.23.0`. The benchmark's hard-fail guard
(`OnnxRTBackend.load()` line 87-89) detected `CUDAExecutionProvider not
active` and aborted explicitly — no silent CPU fallback that would have
produced ~10× inflated ORT latency. After restoring the GPU wheel,
measurement resumed with no spurious data in `history.csv`.

This is the *direct working example* of Day 2 §12.4 ("ORT CPU build shadowing
trap"). The trap reappeared in Day 7; the guard handled it.

### F4. Speedup pattern preserved across models

Both models show the same backend ordering: TRT INT8 > TRT FP16 > TRT FP32 > PT
(with ORT slotting between TRT FP32 and PT on yolov8n, but *below* PT on
yolov8s — see F5).

The PT→TRT INT8 speedup is 6.0× (yolov8n) vs 4.5× (yolov8s). The smaller
yolov8s speedup is *not* an infrastructure artifact: PT's baseline barely
changes across models (19.77 vs 19.75 ms), compressing the available
speedup headroom. This is a measurement *finding* about PyTorch's batch-1
inefficiency at small models, surfaced cleanly by the infrastructure.

### F5. ORT inversion at yolov8s — mechanism identified

In yolov8n, ORT (13.53 ms) was 0.69× of PT (19.75 ms). In yolov8s, ORT
(22.66 ms) is 1.15× of PT (19.77 ms) — slower than the PyTorch baseline.

**Confirmed mechanism**: ORT's `ConvActivationFusion` graph optimizer does
not match yolov8's SiLU pattern, leaving Conv and (Sigmoid+Mul) as separate
kernels.

Evidence (Day 7 diagnostics):

1. Saved `optimized_model_filepath` of yolov8s through ORT's full
   `ORT_ENABLE_ALL` optimization pipeline. The saved graph contains:
   - 64 separate `Conv` nodes (no `FusedConv`)
   - 64 separate `QuickGelu` nodes (ORT's name for the fused Sigmoid×Mul =
     SiLU pattern, produced by `QuickGeluFusion modified: 1` in the VERBOSE log)
   - Op types: `{'Conv', 'QuickGelu', 'MaxPool', 'Concat', 'Split', 'Add',
     'Sub', 'Mul', 'Div', 'Sigmoid', 'Softmax', 'Resize', 'Transpose', 'Reshape'}`
2. The VERBOSE optimization log shows `ConvActivationFusion modified: 0` and
   `ConvAddActivationFusion modified: 0` — neither attempted to fuse the
   Conv→QuickGelu pattern, because `ConvActivationFusion` only fuses
   classic ReLU/Sigmoid/Tanh, not QuickGelu.
3. TRT's graph compiler, by contrast, recognizes the Conv+SiLU pattern.
   §4.1 reports ~44 `GenericConvActFusion` per yolov8n FP16 engine; yolov8s
   shows the same pattern type (verified by re-running fusion_parser on
   yolov8s FP16 log: same fusion count distribution scaled to model depth).
4. Each unfused Conv→QuickGelu pair adds ~70 µs per pair to per-inference
   cost (64 pairs × ~70 µs ≈ 4.5 ms of the 9 ms ORT-vs-TRT gap; remainder
   is non-Conv kernel launches and memory layout differences).

**Eliminated alternative hypotheses** (each tested empirically):

| hypothesis                | option tested                    | result vs baseline |
|---------------------------|----------------------------------|---------------------|
| workspace too small       | `gpu_mem_limit: 4GB → 6GB`       | 23.74 → 23.76 ms (no change) |
| max workspace not used    | `cudnn_conv_use_max_workspace=1` | 23.74 → 23.77 ms (no change) |
| EXHAUSTIVE picks bad algo | `cudnn_conv_algo_search=HEURISTIC` | 23.74 → 23.76 ms (no change) |
| NCHW suboptimal           | `prefer_nhwc=1`                  | 23.74 → 27.22 ms (3.5 ms slower from transposes) |
| Conv+Bias separate        | `fuse_conv_bias=1`               | 23.74 → 32.44 ms (8.7 ms slower from de-fusion) |
| graph opt not applied     | use pre-optimized model          | 23.74 → 23.73 ms (no change) |

ORT's fallback Conv path warning observed in earlier multi-session
diagnostics (`OP Conv(/model.0/conv/Conv) running in Fallback mode`) did
NOT reproduce in a fresh single-session process and is therefore not the
primary cause; it appears to be an artifact of cuDNN context state across
sessions.

**Implication**: ORT's slowdown at yolov8s relative to PyTorch is not a
configuration bug to fix — it is a structural limitation of ORT's CUDA EP
graph-optimization coverage. The infrastructure correctly measured this:
ORT's reported 22.66 ms is the real cost paid by an ORT user on this model.
For deployment, this argues for TRT over ORT when the model contains SiLU
activations.

### F6. INT8 model-size scaling is the most favorable

INT8's model-size ratio (1.32×) is the smallest among all backends; FP16
(1.62×) and FP32 (1.80×) scale more steeply. As models grow, INT8's
relative advantage widens. This is direct evidence for INT8's appeal at
the next model tier — directly relevant to capstone2's OHT model, which is
expected to be larger than yolov8s.

## Cross-checks with §4.1 and §4.2

- §4.1 (fusion analysis) methodology transfers to yolov8s if needed
  (re-run `experiments/fusion_analysis.py` with
  `--engine-glob "models/trt/yolov8s_*.engine"`).
- §4.2 (calibration scale analysis) is **strongly reinforced** by yolov8s
  mAP results: the entropy collapse pattern (-30% on yolov8s, -33% on
  yolov8n) matches within 3 percentage points despite the 3.5× model size
  difference. The KL-divergence's saturation bias on classification Sigmoid
  output is a property of the algorithm interacting with detection P-R
  curves, not a yolov8n-specific artifact.

## Wall-clock breakdown (Day 7 first segment, ~2.5 hours)

| phase | time |
|---|---|
| setup (yolov8s.pt download, ONNX export, FP32/FP16/INT8 builds) | ~32 min |
| 5-way × 2-boundary latency measurement (×2 runs) | ~2 min |
| 5-way mAP measurement | ~3 min |
| ORT package recovery (CPU wheel shadowing trap) | ~10 min |
| ORT-inversion diagnostics (5+ option variants tested) | ~25 min |
| document writing | ~25 min |

The setup phase dominates. For a 3rd model (e.g. yolov8m), the repeated
cost would be ~50 min build + ~5 min measurement, with no code or
document-template work — exactly the reusability claim.
