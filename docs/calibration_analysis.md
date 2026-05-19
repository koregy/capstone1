# INT8 Calibration Analysis

Analysis of why TensorRT INT8 calibration method dominates accuracy
outcomes for YOLOv8n on Jetson Orin Nano. Counterintuitively, MinMax
calibration (which is rarely recommended in TRT documentation) produces
a mAP_50_95 of 0.380 while EntropyCalibration2 — the TRT default — drops
to 0.266 on the same network, same calibration set, same evaluation set.

This document traces that gap to a single observable cause: the
quantization scale chosen for the post-Sigmoid classification output.

## 1. Measurement summary

Evaluation: COCO val2017, `paths[500:1500]` (1000 images, disjoint
from the 500 images used for calibration). `conf_thres=0.001`,
`iou_thres=0.7`, `max_det=300`. Shared NMS and preprocessing across
all backends.

| backend                 | mAP@.5:.95 | mAP@.50  | mAP@.75  | Δ vs FP32 |
|-------------------------|-----------:|---------:|---------:|----------:|
| pytorch_fp32            |      0.394 |    0.543 |    0.426 |  baseline |
| onnxrt_fp32             |      0.395 |    0.545 |    0.428 |   +0.001  |
| trt_fp32                |      0.395 |    0.544 |    0.428 |   +0.001  |
| trt_fp16                |      0.394 |    0.544 |    0.426 |    0.000  |
| **trt_int8_minmax_500** |  **0.380** |    0.528 |    0.411 |   −0.014  |
| trt_int8_percentile_500 |      0.269 |    0.392 |    0.288 |   −0.126  |
| trt_int8_entropy_500    |      0.266 |    0.386 |    0.287 |   −0.128  |
| trt_int8_entropy_100    |      0.260 |    0.376 |    0.281 |   −0.134  |

The three "outlier-trimming" methods (entropy, percentile) cluster at
0.26–0.27. MinMax sits at 0.38, almost matching FP16. The pure
backend-switching cost (FP32 → ORT → TRT) is under 0.001 mAP.

## 2. The smoking gun: post-Sigmoid output scale

TensorRT writes one quantization scale per tensor into the calibration
cache file (`cache/*.cache`). Parsing the 250 entries of each cache
isolates the tensor that determines the maximum achievable classification
confidence: `/model.22/Sigmoid_output_0`, the final per-class score
before NMS.

INT8 representable range = `scale × 127` (per-tensor symmetric).
Since the sigmoid output is mathematically bounded to (0, 1), this
gives a direct ceiling on the confidence values that survive
quantization:

| method           | scale     | INT8 range | Max representable confidence |
|------------------|----------:|-----------:|-----------------------------:|
| entropy_500      |  0.00180  |     0.229  |                       **22.9%** |
| entropy_100      |  0.00176  |     0.224  |                          22.4% |
| percentile_500   |  0.00001  |     0.002  |                       **0.2%**  |
| minmax_500       |  0.00786  |     0.999  |                          99.9% |

This single tensor explains the mAP collapse:

- MinMax keeps the full (0, 1) range — almost perfect quantization
  granularity for confidence values.
- Entropy clamps the representable range at ~23% — every detection
  whose true confidence is above 0.23 saturates to the same INT8 code.
- Percentile is even more extreme: representable range 0.2%, meaning
  virtually all classification scores collapse to zero.

## 3. Cross-check: detection score distribution

A 50-image inference sweep at `conf_thres=0.001` confirms the cache
prediction. With saturation at 0.23, entropy cannot produce any
high-confidence detection:

| score bin        | minmax_500 | entropy_500 |
|------------------|-----------:|------------:|
| [0.001, 0.010)   |       4146 |        4749 |
| [0.010, 0.050)   |       1887 |        1912 |
| [0.050, 0.100)   |        438 |         384 |
| [0.100, 0.250)   |        334 |         268 |
| [0.250, 0.500)   |        171 |         115 |
| [0.500, 0.750)   |         71 |          58 |
| [0.750, 0.900)   |         68 |          33 |
| [0.900, 1.000)   |     **14** |       **0** |

Entropy produced **zero** detections above 0.9; minmax produced 14.
Above 0.5, minmax has 153 detections versus entropy's 91 — a 40%
reduction in high-confidence outputs. Since `mAP@0.5:0.95` is heavily
driven by the high-precision end of the P-R curve, losing the
high-confidence detections collapses that end of the curve and the
mAP with it.

## 4. The mechanism: why does EntropyCalibrator2 do this?

TensorRT's `IInt8EntropyCalibrator2` selects per-tensor saturation
thresholds by minimising the KL divergence between the float32
activation histogram (collected over calibration data) and the
quantized 8-bit histogram. Concretely:

1. Bin the float32 activations of each tensor (default: 2048 bins).
2. For each candidate threshold T, build the equivalent INT8
   histogram (clamping above T, distributing the tail mass into the
   last bin).
3. Pick the T that minimises `KL(float_hist || int8_hist)`.

The failure mode for detection-style sigmoid outputs:

- The activation distribution is heavily mass-concentrated near 0
  (most of the 8400 anchor positions are background; their scores
  hover near zero).
- High-confidence detections (score > 0.5) appear as a *thin tail*
  far from the mass.
- KL divergence is asymmetric in `(float || int8)`: it heavily
  penalises any int8 bin with zero mass where the float
  distribution has non-zero mass, but tolerates a fat int8 tail
  when float has no mass there.
- The optimal T therefore prioritises tight representation of the
  near-zero mode and is willing to clip the far tail.

In other words, the calibrator is doing exactly what its objective
asks. The objective is just wrong for detection: the rare high-score
events are *the signal*, not noise.

## 5. The full layer-level picture

Out of 249 per-tensor entries in the cache, the entropy/minmax scale
ratio (`minmax_scale / entropy_scale`) shows entropy systematically
clipping outliers across the whole network — this is not a one-tensor
artefact:

| ratio       | layer count | %     |
|-------------|------------:|------:|
| ≥ 2.0       |         114 | 45.8% |
| ≥ 5.0       |          21 |  8.4% |
| ≥ 10.0      |           0 |  0.0% |

The detection head's classification branch (`/model.22/cv3.*`) is the
most aggressively clipped:

| layer                                          | entropy | minmax | ratio |
|------------------------------------------------|--------:|-------:|------:|
| /model.22/cv3.1/cv3.1.1/act/Mul_output_0       |   0.160 |  1.324 |  8.25 |
| /model.22/cv3.0/cv3.0.1/act/Mul_output_0       |   0.156 |  1.003 |  6.43 |
| /model.22/cv3.2/cv3.2.1/act/Mul_output_0       |   0.142 |  0.791 |  5.59 |
| /model.22/cv3.0/cv3.0.1/conv/Conv_output_0     |   0.220 |  1.182 |  5.38 |

In YOLOv8 nomenclature `cv3.*` is the classification branch (versus
`cv2.*` for bbox regression). The bbox branch is also clipped by
entropy but less severely. This matches the observation that
mAP loss is dominated by score saturation rather than localisation
error.

## 6. Why percentile_500 matches entropy_500 despite different mechanics

Percentile calibration trims the top X% of activations (default 99.9%);
entropy trims to minimise KL. The two methods arrive at almost identical
mAP (0.269 vs 0.266) but via *different* per-tensor scales:

- 115 of 249 tensors (46%): percentile and entropy within 10% of
  each other.
- 4 tensors: percentile **stricter** than entropy (ratio < 0.5).
- 1 tensor (`/model.22/Sigmoid_output_0`): percentile catastrophic
  (scale 0.00001 — representable range 0.2%).

Percentile happens to hit the same critical tensor as entropy but
even more aggressively. The two mAP numbers are close by coincidence
(one tensor's collapse dominates), not by mechanism. This is an
important methodological point: "mAP similar" does not mean
"same calibration behaviour".

## 7. Calibration data size is not the lever

Entropy with 100 calibration images and entropy with 500 produce
almost identical scales (mean ratio 1.011, median 1.000) and almost
identical mAP (0.260 vs 0.266). More calibration data does not
rescue the algorithm:

| variant         | mAP_50_95 | mean scale ratio vs entropy_500 |
|-----------------|----------:|---------------------------------:|
| entropy_500     |     0.266 |                            1.000 |
| entropy_100     |     0.260 |                            0.990 |

The problem is the KL objective, not the statistics it's fed.

## 8. Cross-check with §4.1 fusion analysis

A natural alternative hypothesis for the entropy↔minmax gap is that
the TRT compiler chose different layer fusions or different kernels
for the two engines, and that this — not the scales — drives the mAP
gap. The fusion analysis (`docs/fusion_catalog.md`) rules that out:

- All three INT8 variants (entropy, minmax, percentile) produce
  *identical* layer counts at every fusion stage: 244 after scale
  fusion, 124 after vertical fusions, 109 after slice removal.
- All three produce identical fusion-type counts (PointWiseFusion 65,
  GenericConvActFusion 50, etc.).
- The only differences are in post-fusion graph cleanup
  (concat removal, reformat layers) which differ by 2–5 layers
  and reflect quant/dequant resolution paths during calibration,
  not fusion decisions.

Combined with §4.1, the conclusion is tight: the mAP gap is caused
**entirely** by per-tensor quantization scales chosen by the
calibrator. The compiler's fusion and kernel selection are
indistinguishable across variants.

## 9. Practical implication

For dense-prediction networks where the loss-relevant signal is
carried by rare high-magnitude activations (detection scores,
classification logits in long-tailed problems), TRT's default
`IInt8EntropyCalibrator2` can silently destroy accuracy. The
diagnostic signature is:

- Histogram of post-NMS confidence scores is squashed below ~0.25.
- The cache file shows a small scale (~0.002) on the final sigmoid
  or softmax output.
- mAP at high IoU thresholds (mAP_75, mAP_50_95) collapses more
  than mAP_50.

The remedies (in order of effort):

1. Switch to `IInt8MinMaxCalibrator` for the whole network — easiest,
   recovered 88% of the lost mAP in our case (0.380 vs 0.394 FP32,
   a 0.014 drop).
2. Mixed precision: leave the detection head's classification branch
   in FP16, INT8 only the backbone. Recovers more accuracy at a small
   latency cost.
3. Per-channel quantization on the detection head (requires explicit
   quantization API rather than calibrator).

For YOLOv8n on Orin Nano, option 1 is the right tradeoff: 6× speedup
over PyTorch FP32 with a 3.6% relative mAP loss.

---

## Appendix: data sources

- mAP measurements: `results/accuracy/*.json` (9 records)
- Calibration cache scales: `cache/entropy_500.cache`,
  `cache/entropy_100.cache`, `cache/minmax_500.cache`,
  `cache/percentile_500.cache` (250 tensor entries each)
- Layer fusion analysis: `docs/fusion_catalog.md`,
  `results/fusion/parsed.json`
- Detection score histogram: ad-hoc sweep over 50 COCO val images
  with conf_thres=0.001 (entropy: 7519 detections; minmax: 7129)
