# WS7 int8 quantisation study

Date: 16 October 2026. Branch: `ws7/quant`, from main `e6e12be`.
Stage 1 complete. Stage 2 is a proposal only. No GPU jobs were submitted.

## Decision

Keep the deployed fp16 model. PTQ optimisation recovers much of the stock KL
loss, but none of the 31 int8 or mixed-precision configurations passes the
existing six-metric accuracy policy. The best faster configuration chosen on
val also fails the test confuser FAR gate. The coordinator's int8 exclusion,
`deploy/model_manifest.json`, `app/config.yaml` and `ws7/export` are unchanged.

The best faster configuration by val R@P90 is
`kl-200-first-body15-body39`. It keeps `conv_3`, `conv_15` through `conv_26`
and `conv_39` through `conv_50` in float. Thus 25 convolution layers use the
native fp16 arithmetic path, and 77 remain int8. At its fresh val threshold
**0.5546875**, measured xcng0 val/test R@P90 is **0.777778 / 0.648725**.
Estimated Pi p95 is **33.81 ms**, compared with the supplied fp16 reference
36.54 ms and a fresh fp16 measurement scaled to **36.49 ms**.

Against the fresh fp16 reference, the best configuration loses **0.018373**
val R@P90 and **0.042493** test R@P90. Its test confuser FAR is **4/73 =
0.054795**, above 0.05; fp16 has 3/73. No alternative was chosen because of
test results. The final configurations and ranking use val only.

The fastest tested mixed configuration, `kl-200-first-winograd`, is estimated
at **31.47 ms**, with val/test R@P90 **0.769029 / 0.640227**. It keeps the
first convolution and all 32 ordinary 3x3 stride-1 convolutions in float.
Its test confuser FAR is also 4/73. More float layers do not necessarily mean
higher latency on this NCNN build; that relationship was measured rather than
assumed. No per-layer kernel profiling was performed.

## Model and measurement protocol

Only `yolo26n-v2s-800-round2-plain` seed1 was studied. Source checkpoint
SHA-256 is `4b753fef63b230c0808f7d5c831f6a0951e059db095746f9ccc9188cee9aa53d`.
Every export uses standard `decode: yolo11` at **448x800**, calibrated from the
existing fp32 export. This study does not reconsider one-to-one decode,
deployment selection, or the requirement for all three seeds before promotion.

Accuracy and FAR are **measured** through `columba_native` on xcng0/xcng1,
using the same A76-compatible NCNN build as WS7. Int8 and mixed exports retain
the runtime's default fp16 flags for floating operations. The PyTorch reference
is WS7's measured Mac CPU/batch-1 rectangular, same-head reference, not its
square training-size scores. Reference val/test R@P90 is 0.789151 / 0.691218.

The data is the original `columba-train-v2` val split, matching the original
KL experiment: 1,089 images, 1,143 pigeon boxes, 381 empty images and 218
confuser images. The v2s-trained checkpoint is unchanged. Deterministic
seed-42 hash ordering gives nested 200- and 1,000-image calibration sets.
No training or test images enter calibration. Every selection is checked
against frozen-test image SHA-256s, and every job verifies the frozen test
archive and manifest before scoring the 319 test images.

Each export independently selects its operating threshold on the entire val
split with WS1's P90 rule. Test uses that threshold unchanged. AP50 is WS1's
COCO metric, and operating recall is measured at each model's own val
threshold. Test R@P90 is the evaluator's descriptive PR metric; it is not a
threshold-selection step.

Latency is **measured** with the existing WS4-backed native C7 service:
ACTIVE waiting, four pinned cores, 20 warm-ups, 200 timed frames, 20 val
photographs cycled at the service's registered 640x480 camera source shape.
All inference uses 448x800. Pi figures are **estimated**, scaled by
1.527110713, not Pi measurements. WS10 must verify the real camera pipeline
at its 800x450 capture geometry and confirm precision-specific latency.

Before inspecting results, the runner defined clearly faster as at least a
5% reduction from the supplied 36.543667 ms fp16 reference, hence p95 below
34.716484 ms. The body-4-14 float block measured 35.39 ms and received no
further tuning as that configuration. Optimised fp32 is a pipeline check,
not a speed candidate. No deployment speed gate was changed.

## Pipeline audit and established cause

**The large KL failure is mainly input clipping caused by calibration.**
The reproduced 200-image int8 binary is byte-identical to the earlier export,
SHA-256 `5af414b6245ee002ebd3a41c37739fd35fb98722eaf6a3a22d32b2976bbaadbb`.
It exactly reproduces selected-seed val/test R@P90 0.529309 / 0.322946.

The native preprocessing audit covers 1,000 val images. Calibration PNGs
match native centred OpenCV bilinear letterboxing pixel for pixel: BGR file
storage, RGB model input, pad 114 and /255 normalisation. No frozen-test SHA
overlap was found. NCNN's PNG loader converts file RGB to BGR, and the
requested pixel conversion returns RGB. Its resize routine bypasses
interpolation when dimensions already equal 800x448. An explicit
`shape=[0,0,3]` calibration produces an identical table, identical weights,
and identical accuracy to `shape=[800,448,3]`. There is no second resize.

The optimised fp32 companion preserves all six PyTorch accuracy metrics
exactly and passes strict `compare_exports` parity. Thus the loss is not
introduced by `ncnnoptimize` before quantisation.

The first convolution's KL activation scale is **283.483368**, giving
`127 / scale = 0.447998064`. This is almost exactly pad value **114/255**.
In the audited 1,000-image population, padding occupies **30.10%** of RGB
channel values; **37.31%** of all values and **53.38%** of non-padding values
exceed that clipping bound. Those brighter values lose their differences at
the first int8 conversion. The input clipping is observed directly; the
histogram's padding peak as the reason KL chooses it is an inference from
the histogram and threshold, not an instrumented account of the optimiser.

Two interventions establish its effect:

- Keep only `conv_3` in float: val/test R@P90 becomes
  **0.743657 / 0.637394**, estimated Pi p95 **32.41 ms**.
- Keep all 102 convolutions int8, changing only the physically known input
  range to [0,1], table scale 127: **0.746282 / 0.637394**, **33.73 ms**.

By contrast, keeping the entire Detect head in float alone gives
0.531059 / 0.322946. It does not repair the clipping upstream. The archived
640 exports have first-input bounds **0.997803** for all three seeds;
the archived 800 exports have **0.447998** for all three. This explains why
stock KL was much worse at 448x800 than at 384x640. Increasing calibration
from 200 to 1,000 images leaves the 800 input bound unchanged and does not
solve the collapse.

A residual gap remains after correcting that input range. Two backbone blocks
in float improve it further. Stock NCNN also reserves a signed **6-bit**
weight range, 31/absmax per output channel, for ordinary 3x3 stride-1,
dilation-1 convolutions eligible for int8 Winograd F43. Other ordinary
convolutions use 127/absmax. This is a constraint a matching QAT scheme must
respect; this study does not establish it as the sole residual-error cause.

Historical-number clarification: the brief's approximately 0.786 to 0.541
are fp16-to-int8 **three-seed val means**, a loss of 0.245. The previous
approximately -0.363 figure was the **three-seed test loss against PyTorch**,
0.688385 to 0.325779. This study's tables below are selected-seed measurements.

## Calibration methods

Stock KL and ACIQ completed at both sizes. ACIQ has a first-input bound of
0.804432 at both counts, so it avoids the extreme KL input clipping but has
lower val recall than the targeted range correction. Its better test R@P90
was not used to select it over the higher-val configurations.

Stock EQ crashed with SIGSEGV at both 200 and 1,000 images, after its initial
KL stage and on the first scale-search message. No EQ package exists, so
accuracy, FAR and latency are unavailable, not zero. The pinned source has
incorrect channel/layer indices inside the input/image loops of
`quantize_EQ`; the precise crash instruction was not established with a
debugger. Both complete failure logs are retained. No patched EQ algorithm,
NCNN dependency update or custom kernel was introduced. EQ's 50-image local
search cap would also mean that 1,000 supplied images affect its initial KL
stage, not the size of its later local search.

## Results for every completed configuration

Accuracy and FAR below are **measured on xcng**. Pi p95 is **estimated**.
FAR columns list **empty / confuser**, separately for val and test.
The accuracy column checks all six losses against same-head rectangular
PyTorch, each at most 0.01. Strict int8 per-image parity is retained as an
additional diagnostic in the evidence files. The fp16 companion passes its
registered policy through strict fp32 parity plus its accuracy result.

| Variant | Val R@P90 | Test R@P90 | Val FAR E/C | Test FAR E/C | Est. Pi p95 ms | Int8 convs | Accuracy policy |
|---|---:|---:|---:|---:|---:|---:|---|
| `aciq-1000` | 0.5932 | 0.6572 | 0.0000 / 0.0183 | 0.0000 / 0.0411 | 33.81 | 102/102 | fail |
| `aciq-200` | 0.6072 | 0.6601 | 0.0000 / 0.0138 | 0.0000 / 0.0411 | 33.72 | 102/102 | fail |
| `fp16-reference` | 0.7962 | 0.6912 | 0.0000 / 0.0321 | 0.0000 / 0.0411 | 36.49 | 0/102 | pass |
| `kl-1000` | 0.5328 | 0.3314 | 0.0026 / 0.0092 | 0.0000 / 0.0411 | 33.70 | 102/102 | fail |
| `kl-200` | 0.5293 | 0.3229 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.61 | 102/102 | fail |
| `kl-200-first-body15` | 0.7594 | 0.6459 | 0.0000 / 0.0138 | 0.0000 / 0.0548 | 33.41 | 89/102 | fail |
| `kl-200-first-body15-body39` | 0.7778 | 0.6487 | 0.0000 / 0.0183 | 0.0000 / 0.0548 | 33.81 | 77/102 | fail |
| `kl-200-first-body15-head` | 0.7673 | 0.6516 | 0.0000 / 0.0138 | 0.0000 / 0.0548 | 32.66 | 65/102 | fail |
| `kl-200-first-box` | 0.7437 | 0.6374 | 0.0000 / 0.0183 | 0.0000 / 0.0548 | 32.26 | 92/102 | fail |
| `kl-200-first-class` | 0.7437 | 0.6374 | 0.0000 / 0.0183 | 0.0000 / 0.0548 | 32.52 | 86/102 | fail |
| `kl-200-first-head` | 0.7480 | 0.6374 | 0.0000 / 0.0183 | 0.0000 / 0.0548 | 32.13 | 77/102 | fail |
| `kl-200-first-winograd` | 0.7690 | 0.6402 | 0.0000 / 0.0183 | 0.0000 / 0.0548 | 31.47 | 69/102 | fail |
| `kl-200-float-body-15-26` | 0.5556 | 0.3343 | 0.0026 / 0.0092 | 0.0000 / 0.0411 | 34.17 | 90/102 | fail |
| `kl-200-float-body-27-38` | 0.5468 | 0.3286 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.81 | 90/102 | fail |
| `kl-200-float-body-39-50` | 0.5494 | 0.3286 | 0.0026 / 0.0092 | 0.0000 / 0.0411 | 33.90 | 90/102 | fail |
| `kl-200-float-body-4-14` | 0.5433 | 0.3399 | 0.0026 / 0.0092 | 0.0000 / 0.0274 | 35.39 | 91/102 | fail |
| `kl-200-float-body-51-62` | 0.5241 | 0.3258 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 34.23 | 90/102 | fail |
| `kl-200-float-body-63-78` | 0.5293 | 0.3229 | 0.0026 / 0.0092 | 0.0000 / 0.0411 | 34.11 | 86/102 | fail |
| `kl-200-float-box` | 0.5293 | 0.3201 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.25 | 93/102 | fail |
| `kl-200-float-class` | 0.5311 | 0.3286 | 0.0026 / 0.0092 | 0.0000 / 0.0274 | 33.40 | 87/102 | fail |
| `kl-200-float-class-p3` | 0.5354 | 0.3229 | 0.0052 / 0.0092 | 0.0000 / 0.0411 | 33.77 | 101/102 | fail |
| `kl-200-float-class-p4` | 0.5223 | 0.3229 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.84 | 101/102 | fail |
| `kl-200-float-class-p5` | 0.5284 | 0.3229 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.65 | 101/102 | fail |
| `kl-200-float-first` | 0.7437 | 0.6374 | 0.0000 / 0.0183 | 0.0000 / 0.0548 | 32.41 | 101/102 | fail |
| `kl-200-float-head` | 0.5311 | 0.3229 | 0.0026 / 0.0092 | 0.0000 / 0.0274 | 32.85 | 78/102 | fail |
| `kl-200-float-other-depthwise` | 0.5319 | 0.3201 | 0.0026 / 0.0092 | 0.0000 / 0.0274 | 33.63 | 100/102 | fail |
| `kl-200-float-terminal-box` | 0.5284 | 0.3201 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.57 | 99/102 | fail |
| `kl-200-float-terminal-class` | 0.5249 | 0.3229 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.76 | 99/102 | fail |
| `kl-200-float-terminal-head` | 0.5232 | 0.3201 | 0.0026 / 0.0092 | 0.0000 / 0.0274 | 33.75 | 96/102 | fail |
| `kl-200-input-range` | 0.7463 | 0.6374 | 0.0000 / 0.0183 | 0.0000 / 0.0548 | 33.73 | 102/102 | fail |
| `kl-200-input-range-body15` | 0.7524 | 0.6402 | 0.0000 / 0.0138 | 0.0000 / 0.0548 | 34.24 | 90/102 | fail |
| `kl-200-no-resize` | 0.5293 | 0.3229 | 0.0052 / 0.0092 | 0.0000 / 0.0274 | 33.61 | 102/102 | fail |
| `optimised-fp32` | 0.7892 | 0.6912 | 0.0000 / 0.0321 | 0.0000 / 0.0411 | 70.90 | 0/102 | pass |
| `eq-200` | unavailable | unavailable | unavailable | unavailable | unavailable | no package | SIGSEGV |
| `eq-1000` | unavailable | unavailable | unavailable | unavailable | unavailable | no package | SIGSEGV |

The float sensitivity scan partitions all 102 convolution layers into ten
blocks: first convolution; ordinary convs 4-14, 15-26, 27-38, 39-50, 51-62,
63-78; Detect box branch; Detect class branch; remaining depthwise layers.
It leaves one block out at a time, retaining its weights and activations in
float by removing both activation and weight table entries. Inventory checks
verify the actual converted graph, including every requested exclusion.
These are groups of layers, not per-block quantisation granularity.

The class branch includes `conv_88`-`conv_96` and `convdw_212`-`convdw_217`;
the box branch is `conv_79`-`conv_87`. All terminal class/box sets and each
terminal class layer were also tested. The final backbone combinations were
chosen from the initial **val** sensitivity ranking, with the first-layer
correction retained. The known-range intervention comes from input semantics,
not a fitted test-set clipping bound. No further variants are planned.

### AP50, operating recall and fresh thresholds

All values are **measured on xcng**. Operating recall uses each row's own
val-selected threshold; test never selects one.

| Variant | Val AP50 | Test AP50 | Val operating recall | Test operating recall | Val threshold |
|---|---:|---:|---:|---:|---:|
| `aciq-1000` | 0.8292 | 0.7036 | 0.5932 | 0.3654 | 0.740234375 |
| `aciq-200` | 0.8355 | 0.7008 | 0.6072 | 0.3909 | 0.708007812 |
| `fp16-reference` | 0.8740 | 0.7479 | 0.7962 | 0.6176 | 0.626464844 |
| `kl-1000` | 0.6985 | 0.4181 | 0.5328 | 0.3258 | 0.474121094 |
| `kl-200` | 0.6957 | 0.4109 | 0.5293 | 0.3229 | 0.452392578 |
| `kl-200-first-body15` | 0.8595 | 0.7198 | 0.7594 | 0.5836 | 0.484375000 |
| `kl-200-first-body15-body39` | 0.8656 | 0.7223 | 0.7778 | 0.5864 | 0.554687500 |
| `kl-200-first-body15-head` | 0.8616 | 0.7151 | 0.7673 | 0.6006 | 0.472656250 |
| `kl-200-first-box` | 0.8539 | 0.7077 | 0.7437 | 0.5637 | 0.503906250 |
| `kl-200-first-class` | 0.8529 | 0.7054 | 0.7437 | 0.5666 | 0.482910156 |
| `kl-200-first-head` | 0.8543 | 0.7076 | 0.7480 | 0.5694 | 0.474121094 |
| `kl-200-first-winograd` | 0.8627 | 0.7137 | 0.7690 | 0.5751 | 0.532226562 |
| `kl-200-float-body-15-26` | 0.7074 | 0.4232 | 0.5556 | 0.3286 | 0.474853516 |
| `kl-200-float-body-27-38` | 0.6972 | 0.4155 | 0.5468 | 0.3258 | 0.430908203 |
| `kl-200-float-body-39-50` | 0.7046 | 0.4181 | 0.5494 | 0.3286 | 0.548828125 |
| `kl-200-float-body-4-14` | 0.6963 | 0.4194 | 0.5433 | 0.3314 | 0.451171875 |
| `kl-200-float-body-51-62` | 0.6985 | 0.4177 | 0.5241 | 0.3258 | 0.549316406 |
| `kl-200-float-body-63-78` | 0.6995 | 0.4116 | 0.5293 | 0.3229 | 0.489501953 |
| `kl-200-float-box` | 0.6976 | 0.4152 | 0.5293 | 0.3201 | 0.452392578 |
| `kl-200-float-class` | 0.6950 | 0.4111 | 0.5311 | 0.3229 | 0.444091797 |
| `kl-200-float-class-p3` | 0.6959 | 0.4110 | 0.5354 | 0.3229 | 0.436767578 |
| `kl-200-float-class-p4` | 0.6960 | 0.4107 | 0.5223 | 0.3229 | 0.453857422 |
| `kl-200-float-class-p5` | 0.6951 | 0.4110 | 0.5284 | 0.3229 | 0.453857422 |
| `kl-200-float-first` | 0.8530 | 0.7057 | 0.7437 | 0.5637 | 0.503906250 |
| `kl-200-float-head` | 0.6962 | 0.4117 | 0.5311 | 0.3201 | 0.444091797 |
| `kl-200-float-other-depthwise` | 0.6957 | 0.4071 | 0.5319 | 0.3201 | 0.447265625 |
| `kl-200-float-terminal-box` | 0.6954 | 0.4092 | 0.5284 | 0.3201 | 0.453857422 |
| `kl-200-float-terminal-class` | 0.6959 | 0.4109 | 0.5249 | 0.3229 | 0.442138672 |
| `kl-200-float-terminal-head` | 0.6954 | 0.4089 | 0.5232 | 0.3201 | 0.443603516 |
| `kl-200-input-range` | 0.8531 | 0.7064 | 0.7463 | 0.5637 | 0.495117188 |
| `kl-200-input-range-body15` | 0.8587 | 0.7159 | 0.7524 | 0.5779 | 0.499511719 |
| `kl-200-no-resize` | 0.6957 | 0.4109 | 0.5293 | 0.3229 | 0.452392578 |
| `optimised-fp32` | 0.8738 | 0.7479 | 0.7892 | 0.6147 | 0.639187396 |

### Accuracy deltas against rectangular PyTorch

These are differences derived from the measured native and published CPU
PyTorch metrics. Positive means the export improved that metric. Every one
must be at least -0.01 to pass the current accuracy policy.

| Variant | Val ΔR@P90 | Test ΔR@P90 | Val ΔAP50 | Test ΔAP50 | Val Δoperating recall | Test Δoperating recall |
|---|---:|---:|---:|---:|---:|---:|
| `aciq-1000` | -0.195976 | -0.033994 | -0.044620 | -0.044378 | -0.195976 | -0.249292 |
| `aciq-200` | -0.181977 | -0.031161 | -0.038386 | -0.047163 | -0.181977 | -0.223796 |
| `fp16-reference` | +0.006999 | +0.000000 | +0.000161 | -0.000005 | +0.006999 | +0.002833 |
| `kl-1000` | -0.256343 | -0.359773 | -0.175354 | -0.329852 | -0.256343 | -0.288952 |
| `kl-200` | -0.259843 | -0.368272 | -0.178190 | -0.337077 | -0.259843 | -0.291785 |
| `kl-200-first-body15` | -0.029746 | -0.045326 | -0.014374 | -0.028161 | -0.029746 | -0.031161 |
| `kl-200-first-body15-body39` | -0.011374 | -0.042493 | -0.008204 | -0.025633 | -0.011374 | -0.028329 |
| `kl-200-first-body15-head` | -0.021872 | -0.039660 | -0.012241 | -0.032804 | -0.021872 | -0.014164 |
| `kl-200-first-box` | -0.045494 | -0.053824 | -0.019909 | -0.040219 | -0.045494 | -0.050992 |
| `kl-200-first-class` | -0.045494 | -0.053824 | -0.020923 | -0.042588 | -0.045494 | -0.048159 |
| `kl-200-first-head` | -0.041120 | -0.053824 | -0.019577 | -0.040392 | -0.041120 | -0.045326 |
| `kl-200-first-winograd` | -0.020122 | -0.050992 | -0.011134 | -0.034276 | -0.020122 | -0.039660 |
| `kl-200-float-body-15-26` | -0.233596 | -0.356941 | -0.166475 | -0.324716 | -0.233596 | -0.286119 |
| `kl-200-float-body-27-38` | -0.242345 | -0.362606 | -0.176658 | -0.332472 | -0.242345 | -0.288952 |
| `kl-200-float-body-39-50` | -0.239720 | -0.362606 | -0.169221 | -0.329805 | -0.239720 | -0.286119 |
| `kl-200-float-body-4-14` | -0.245844 | -0.351275 | -0.177501 | -0.328527 | -0.245844 | -0.283286 |
| `kl-200-float-body-51-62` | -0.265092 | -0.365439 | -0.175318 | -0.330219 | -0.265092 | -0.288952 |
| `kl-200-float-body-63-78` | -0.259843 | -0.368272 | -0.174352 | -0.336296 | -0.259843 | -0.291785 |
| `kl-200-float-box` | -0.259843 | -0.371105 | -0.176204 | -0.332757 | -0.259843 | -0.294618 |
| `kl-200-float-class` | -0.258093 | -0.362606 | -0.178869 | -0.336808 | -0.258093 | -0.291785 |
| `kl-200-float-class-p3` | -0.253718 | -0.368272 | -0.177917 | -0.336992 | -0.253718 | -0.291785 |
| `kl-200-float-class-p4` | -0.266842 | -0.368272 | -0.177806 | -0.337230 | -0.266842 | -0.291785 |
| `kl-200-float-class-p5` | -0.260717 | -0.368272 | -0.178724 | -0.336940 | -0.260717 | -0.291785 |
| `kl-200-float-first` | -0.045494 | -0.053824 | -0.020882 | -0.042282 | -0.045494 | -0.050992 |
| `kl-200-float-head` | -0.258093 | -0.368272 | -0.177630 | -0.336201 | -0.258093 | -0.294618 |
| `kl-200-float-other-depthwise` | -0.257218 | -0.371105 | -0.178162 | -0.340806 | -0.257218 | -0.294618 |
| `kl-200-float-terminal-box` | -0.260717 | -0.371105 | -0.178465 | -0.338699 | -0.260717 | -0.294618 |
| `kl-200-float-terminal-class` | -0.264217 | -0.368272 | -0.177969 | -0.336995 | -0.264217 | -0.291785 |
| `kl-200-float-terminal-head` | -0.265967 | -0.371105 | -0.178440 | -0.339026 | -0.265967 | -0.294618 |
| `kl-200-input-range` | -0.042870 | -0.053824 | -0.020753 | -0.041496 | -0.042870 | -0.050992 |
| `kl-200-input-range-body15` | -0.036745 | -0.050992 | -0.015104 | -0.032037 | -0.036745 | -0.036827 |
| `kl-200-no-resize` | -0.259843 | -0.368272 | -0.178190 | -0.337077 | -0.259843 | -0.291785 |
| `optimised-fp32` | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## NCNN granularity

The ordinary convolution weights already have one symmetric scale per output
channel. In this model every depthwise group is one channel, so its group
scales also give one per output channel. Activations use one scale per input
tensor. NCNN's deployed ARM int8 kernels provide no per-block activation or
weight mode. Splitting these tensors into independently scaled blocks would
need a different graph/kernel path, outside this study. No such path was
built. Selective float exclusions and valid calibration-table scale changes
are supported and were the only granularity changes tested.

## Stage 2 proposal, awaiting user approval

Stage 1 meets the condition for a QAT proposal: the val-chosen faster PTQ
configuration still loses more than 0.01 R@P90 against fp16 on both val and
test. **Do not start it without the coordinator's go-ahead.** Given fp16
already meets the latency target, keep fp16 unless the user has a reason to
spend GPU time on this remaining 3-5 ms opportunity.

1. Start from the fixed plain-800 seed1 `best.pt`. Fine-tune on the v2s
   **training split only** for 5-10 epochs at a small pre-registered learning
   rate, with BN statistics frozen and export folding matched. Keep the
   deployed one-to-many head mapping. Use val for stopping/ranking and fresh
   thresholds; keep test frozen until the export/configuration is fixed.
2. Use symmetric signed quantisation, zero point zero, per-output-channel
   weights and per-tensor activations, with straight-through fake quantisation.
   Fix the model input range to [0,1]. Match NCNN's 6-bit weight range for its
   eligible Winograd convolutions and full signed int8 for the others; merely
   using PyTorch's stock full-int8 observer everywhere would not reproduce
   this converter. Use the mixed-precision exclusions as an optional registered
   comparison, not a new kernel mode.
3. Prefer exporting learned clip ranges into the NCNN table: a scalar
   activation multiplier `127 / clip`, and one weight multiplier per output
   channel, `127 / clip` or `31 / clip` for the Winograd subset. Map each folded
   module to its exported NCNN layer name before applying the table. Confirm
   identical folding, tensor ordering and ranges; NCNN has no learned zero
   points to import. Validate the table against the fake-quant graph on val.
4. If reliable range/name transfer is unavailable, strip fake quantisation,
   export fp32 QAT weights and recalibrate those weights on val with the known
   input range fixed. This is a separate measured result, not an assumption
   that the QAT ranges survived export. Re-run strict fp32 parity, the int8
   accuracy policy, fresh thresholds and C7 timings. Check all three seeds
   before any promotion, with an explicit coordinator decision.
5. **Estimated pilot GPU budget:** one GPU, at most **one H100-equivalent GPU
   hour** for one seed. The original H100 NVL run measured approximately
   35 seconds per epoch; assuming 2-4x QAT overhead gives approximately 12-24
   minutes for ten epochs, with the remainder for setup, validation and one
   restart. This multiplier is an estimate, not a QAT benchmark. Re-estimate
   before choosing a slower GPU. A later three-seed study would need a separate
   budget and approval. Export and NCNN evaluation remain ARM CPU work.

No GPU allocation, training, package update or deployment change has occurred.

## Reproduction and evidence

`tools/quantize_ncnn.py` preserves its default KL/200 behaviour and adds
`--method kl|aciq|eq`, `--count`, provenance-checked `--table-source`, exact-name
`--float-layers`, `--activation-range LAYER ABSMAX`, `--no-resize` and a bounded
calibration timeout. Removing a float layer removes its weight and activation
entries; prefix collisions such as conv_3 versus conv_30 are tested.

`tools/quantization_study.py` runs the native evaluation and C7 service. The
commands below run **inside a guarded four-core xcng Slurm allocation**, from
this checkout, after staging and verifying the two datasets. Exact allocation,
staging, snapshot and budget-guard submission scripts are in
`WS7-quant-evidence/jobs.json`. Never run these on a login node.

```bash
export PATH="$HOME/columba/scratch/ws8/venv/bin:$PATH"
export PYTHONPATH="$HOME/columba/scratch/ws8/round3/repo/native/build/linux-aarch64:$PWD"
export OMP_WAIT_POLICY=ACTIVE OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
source_model="$HOME/columba/scratch/ws7/exports/yolo26n-v2s-800-round2-plain/seed1/yolo11/fp32_ncnn_model"
study_output="$HOME/columba/scratch/ws7-quant"
validation_data="/tmp/cz-$SLURM_JOB_ID/columba-train-v2"
frozen_test_data="/tmp/cz-$SLURM_JOB_ID/columba-test-v2"
reference_predictions="$HOME/columba/scratch/ws7/published-references/yolo26n-v2s-800-round2-plain/seed1/yolo11/pt"
study_arguments=(--model "$source_model" --output "$study_output"
  --dataset "$validation_data" --test-dataset "$frozen_test_data"
  --reference "$reference_predictions" --prefix "$HOME/columba/scratch/ws4/ncnn-xcng")
srun --cpu-bind=cores python -m tools.quantization_study calibration \
  "${study_arguments[@]}" --methods kl aciq --counts 200 1000
srun --cpu-bind=cores python -m tools.quantization_study convert \
  "${study_arguments[@]}" --variant kl-200-input-range --method kl --count 200 \
  --table-source "$study_output/packages/kl-200_ncnn_model/calibration.table" \
  --activation-range conv_3 1
srun --cpu-bind=cores python -m tools.quantization_study convert \
  "${study_arguments[@]}" --variant kl-200-first-body15-body39 --method kl --count 200 \
  --table-source "$study_output/packages/kl-200_ncnn_model/calibration.table" \
  --float-layers conv_3 conv_{15..26} conv_{39..50}
srun --cpu-bind=cores python -m tools.quantization_study recompare \
  "${study_arguments[@]}"
```

Use a distinct variant name for a changed configuration; the runner rejects
mismatching cached packages and model/result hashes. `scan`, `heads`, `audit`
and `optimise-reference` reproduce the other registered comparisons. EQ
failures and calibration timeouts are recorded without manufacturing a model.

The initial study used an older GPU/batch-16 PyTorch reference cache for
comparison deltas. Native C3/C4 accuracy, thresholds and C7 timing were
unaffected. The final job **recomputed every parity and accuracy comparison**
against the published CPU/batch-1 reference. A new guard rejects the wrong
cache before future runs. The optimised fp32 companion now has zero strict
parity failures and zero metric deltas. No initial comparison result is used
in this report.

Committed evidence:

- `WS7-quant-results.json`: all 33 completed configurations and exact metrics.
- `WS7-quant-evidence/`: C4s, C3 sidecars, corrected parity and accuracy deltas,
  C7s with all 200 samples, calibration tables/provenance, layer inventory,
  calibration image hashes, preprocessing histogram, source hashes and failures.
- `WS7-quant-evidence/audit.json`: all 33 results checked against reference
  metrics, fresh thresholds, calibrated samples, actual layer coverage and hashes.
- `WS7-quant-evidence/jobs.json` and `job-accounting.txt`: every submission,
  snapshot, CPU allocation, completed job and initial staging failure.

The leaderboard contains 37 `ws7-quant/` diagnostic/failure rows, including
the two staging failures and two EQ failures. Each evidence directory records
its source snapshot and job in `run.json`. These are export studies, with zero
training epochs, not new training arms.

Raw C3 arrays and experimental C8 packages remain at
`~/columba/scratch/ws7-quant/`, with local copies under `outputs/ws7-quant/`.
No new model binaries enter Git. C7 copies omit duplicated embedded calibration
images and link their separate provenance instead; original report hashes are
retained. Calibration image lists are shared by count rather than duplicated
for every float set. Source hashes and raw prediction hashes make those
compacted evidence files auditable. Copied calibration tables and EQ logs
normalise trailing whitespace; their original and normalised hashes are
recorded, with numerical table values unchanged.

Completed jobs: **959718** KL/ACIQ/block scan, **959719** recorded EQ failures,
**959731** head scan, **959798** input/graph controls, **959814** backbone
controls and final reference comparisons. Jobs **959716/959717** stopped at
staging before calibration and were resubmitted. All had `cz-` names,
`nice=1000`, four CPUs and 12 GiB requested memory, at most two CPU jobs
project-wide. About **2.68 CPU core-hours** were allocated, **zero GPU-hours**.
All node-local staging was removed. No study jobs remain in flight. No SSH
retry was needed after the tunnel was restored; no password was requested,
used or stored.

## Validation and remaining decisions

After the last tooling change: full pytest **332 passed, 233 subtests passed,
zero skipped**; full unittest **332 passed**; Ruff check and format check on
`app tests tools cluster training native` pass, **96 files formatted**; full
mypy passes, **93 source files**. `git diff --check` passes. The added tests
cover table filtering, exact layer names, physical range validation, quantised
layer inventory, scan coverage and the parity-reference cache guard.

No contract change is requested. The unresolved study limitation is stock
EQ's crash; its repaired algorithm was not evaluated. The coordinator must
decide whether the optional QAT pilot merits user approval. Existing deployment
acceptance remains with WS10; this study does not change its decisions.

## Primary implementation sources

- [Pinned NCNN calibration source](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/tools/quantize/ncnn2table.cpp).
- [NCNN int8 conversion source](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/tools/quantize/ncnn2int8.cpp).
- [NCNN pixel resize bypass](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/src/mat_pixel.cpp).
- [ARM convolution activation scaling](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/src/layer/arm/convolution_arm.cpp).
