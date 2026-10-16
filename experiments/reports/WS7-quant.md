# WS7 int8 quantisation study

Date: 16 October 2026. Branch: `ws7/quant`, based on local main `e6e12be`.
Status: experiments in progress, ARM CPUs only. No QAT or GPU work.

## Established findings so far

The initial connection failure was resolved by the coordinator. Jobs 959716
and 959717 stopped before calibration because the freeze verifier also needs
the test archive beside the extracted dataset. Staging was corrected, and
jobs 959718 and 959719 ran on xcng0 and xcng1. Job 959731 tests smaller float
head sets. All submissions use the project-wide budget guard, four CPU cores,
`nice=1000`, node-local datasets and `OMP_WAIT_POLICY=ACTIVE`.

The 200-image KL reproduction has exactly the same int8 weight SHA-256 as the
previous export, `5af414b6245ee002ebd3a41c37739fd35fb98722eaf6a3a22d32b2976bbaadbb`.
It reproduces selected-seed val R@P90 0.529309 and test 0.322946. These are
measured on xcng0. The brief's 0.786 and 0.541 are arm-level figures rather
than these selected-seed numbers; their arithmetic difference is -0.245,
not -0.363. The previous test arm-mean collapse is a separate comparison.

A native pixel audit passed on 1,000 val images, with no pixel differences
against the calibration letterbox and no frozen-test SHA overlap. The PNGs
are centred 448x800 letterboxes, OpenCV bilinear, pad 114, with RGB /255 at
NCNN input. NCNN's source confirms that `from_pixels_resize` bypasses
interpolation when the source already has the requested dimensions.

The stock export quantises 94 Convolution and eight ConvolutionDepthWise
layers, all 102 supported convolution layers. Swish, concatenation, attention
and other graph operations retain their supported floating implementations.
The quantiser now writes a per-layer inventory rather than inferring coverage
from file size.

Keeping only the first convolution in float improves val R@P90 to 0.743657
and test to 0.637394, with estimated Pi p95 32.41 ms. The first-layer KL
activation scale is 283.483368, giving a symmetric bound of 0.447998, almost
114/255. A known-input-range correction is being prepared to test whether
this clipping explains the loss.

Both stock EQ attempts, 200 and 1,000 val images, crashed with SIGSEGV.
The pinned tool's `quantize_EQ` uses channel/layer indices where its nested
input/image loops require different indices. No EQ accuracy number exists.
No NCNN runtime kernel or pinned dependency was changed.

## Protocol and remaining measurements

All variants use plain-800 seed1, standard yolo11 decode at 448x800, calibrated
from the same fp32 graph. Calibration images are deterministic nested subsets
of the original `columba-train-v2` validation split, matching the previous
study. The 200-image set is contained in the 1,000-image set. Neither uses
training or test images. The frozen test archive is verified before scoring.

Every variant gets its own val-selected P90 threshold. Test uses that
threshold without tuning. Accuracy is measured through `columba_native` with
the A76-compatible NCNN build. C7 latency uses the WS4 service, four pinned
cores, 20 warm-ups and 200 measurements with ACTIVE waiting. Pi estimates use
WS4's factor 1.527110713 and require hardware confirmation. The existing
service cycles 20 val photographs at its registered 640x480 camera shape.

A 5% reduction from the historical fp16 Pi p95, 36.543667 ms, is the study's
explicit interpretation of clearly faster. Variants at or above 34.716484 ms
will not receive further tuning. Configuration choices use val only; test
scores remain diagnostics. Deployment eligibility and the int8 exclusion
are unchanged.

Results, all six accuracy deltas against same-head rectangular PyTorch,
FARs, calibration-method comparison, sensitivity scan and the final
recommendation will be added after the CPU jobs finish.

## Sources

- [Pinned NCNN calibration source](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/tools/quantize/ncnn2table.cpp).
- [NCNN int8 conversion source](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/tools/quantize/ncnn2int8.cpp).
- [NCNN pixel resize bypass](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/src/mat_pixel.cpp).

## Verification so far

Full pytest passed, 329 tests and 225 subtests. Full unittest discovery passed,
329 tests. Ruff check, format check and full mypy passed. Further range-control
changes will receive the same checks before finalisation.

The deploy manifest and application config are unchanged. No password was
requested, used or stored. `ws7/export` is unchanged. The cluster workspace is
`~/columba/scratch/ws7-quant/`; raw local copies are in `outputs/ws7-quant/`.
