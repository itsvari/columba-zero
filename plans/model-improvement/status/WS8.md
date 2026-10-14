# WS8 — Native C++ detector

- State: done, including frozen FP32 parity on both platforms
- Branch: `ws8/native`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14
- Rebased onto main: `3af2d02`, including native mypy and Ruff coverage from `2cb3308`.

## Done

1. `columba_native.Detector(model_dir, threads=4, compute=None)`: C8 and legacy metadata,
   OpenCV `INTER_LINEAR` letterboxing, NCNN `in0`/`out0`, both decoders,
   zero-copy uint8 BGR input, released GIL and owned float32 N×6 output.
   Properties: `input_hw`, `names`, `decode`, effective `compute` and actual
   loaded-network `compute_flags`. `detect_with_timings(frame, conf)`
   adds C++ `preprocess`, `extract`, `postprocess`, `total` measurements in ms.
2. Builds and C++ tests pass on Mac ARM64 and Linux aarch64 `xcng1`, using
   NCNN tag `20260526`, commit `e54f7b1f88434e1d844ea0551b880a1cfb079ce1`,
   Vulkan off, and OpenCV `5.0.0`. Plain CMake supports WS4's standard
   `-Dncnn_DIR=<prefix>/lib/cmake/ncnn` target. Both WS4 prefixes also build
   and pass C++ tests; its exact Mac/Linux commands are in `native/README.md`.
3. `NativeDetector(path, image_size, confidence, threads=4, compute=None)` matches WS9's
   factory. It reuses shared package validation, permits only pigeon triggers,
   returns secondary species too, uses lower-case metadata names and passes
   `class_id=`. `last_stage_ms` supplies `preprocess`, `inference`, `postprocess`.
   No edit to WS9-owned application files was needed.
4. **Measured smoke parity, Mac `vexcalibur` and Linux `xcng1`:** seven packages
   each pass all 13 local images: YOLO11n, standard YOLO26n and head-swapped
   one-to-one YOLO26n at 640×640 and 384×640, plus legacy v1-2 at 640×640.
   The set contains eight COCO8 photos, two Ultralytics assets, two historical
   images and one blank control. These are parity checks, not accuracy results.
   Additional Mac checks pass against Ultralytics NCNN, including a package
   without the `_ncnn_model` suffix.
5. **Measured overhead** for the baseline NCNN model on both hosts is recorded
   below and in `native/results/ws8-fp32-smoke.json`.
6. Round 2: all **208 Mac tests pass** with the extension, including
   59 runtime tests and 18 native tests. Ruff lint/format and strict mypy pass
   on `app tests tools cluster training native`, with `native/build` excluded
   from mypy. Without the extension the seven extension-only tests skip cleanly.
   The 12 package/override combinations inspect the actual loaded NCNN options;
   wrapper forwarding and invalid options are covered too. CTest passes.
   In round 1, all **201 Linux tests passed** against WS4's prefix, along with
   Ruff and strict mypy. Job `945441` completed with exit 0. No WS8 background
   process or running Slurm job remains.

## In progress

Nothing. Frozen/proxy FP32 parity passes on both platforms with the documented
zero-area comparison rule. FP16 package-default diagnostics are recorded for WS7.

## Blockers and requests

- The coordinator completed native mypy and Ruff coverage at `2cb3308`.
  The plain CMake build needs no build-system change. WS8 did not edit pyproject.
- Required frozen FP32 parity passes on both hosts after rescoring unchanged
  outputs with the corrected degenerate-box comparator. The 63-image source
  test directory is a proxy, not claimed to be the proven original split.
- FP16 smoke results below are informational. WS7 still owns FP16/int8 accuracy
  gates on frozen test-v2. Package weight quantisation and runtime compute
  precision are distinct.
- WS9/coordinator, optional future config: the wrapper now accepts `compute=`;
  the current factory uses the package default. Exposing a config override
  would require a WS9-owned config/factory change. No such file was edited;
  no outstanding class-ID or timing switch remains.
- The documented Mac build now uses merged `tools/build_ncnn_a76.sh` and the
  stable main-checkout prefix, not a disposable WS4 worktree. The OpenCV helper
  supports `--opencv-only`; Pi system `libopencv-dev` is documented too.

## Round-3 frozen parity results

**Measured**, Mac ARM64 `vexcalibur` and Linux aarch64 `xcng1`, four threads,
confidence 0.25. Both verify frozen **columba-test-v2, 319 images**, against
`data_sources/columba-test-v2.sha256`. The **63-image Roboflow YOLO1199 source
test proxy** has matching image-set SHA-256 on both hosts:
`67e84e6413cb0e0627cc5e09d26eb84184be22e55378cf1dccba2dfc4c0eefc7`.

YOLO11n and both YOLO26n heads use trained single-class pigeon checkpoints
from `~/columba/runs/{yolo11n,yolo26n}-v1clean-640-r1/seed0/weights/best.pt`,
exported at 384×640. No COCO fallback was used. FP16 exports use `quantize=16`,
C8 `quantization: fp16`, and `compute=None` via `--compute package`.
Legacy FP32 uses the original v1-2 package at 640×640; its FP16 default check
uses a C8 metadata copy with byte-identical legacy weights. Checkpoint and
package hashes are in the committed summary.

All **3,056 required FP32 image/package comparisons pass**, including
**2,552 frozen-set** and **504 proxy** comparisons. Maximum associated FP32
coordinate delta is **0.002319 px** and score delta **0.000102580**.
The table shows passing images / total; FP16 uses looser informational
IoU ≥ 0.7 and score delta ≤ 0.10, with equal classes and counts.

| Host / model | Frozen FP32 | Frozen FP16 info | Proxy FP32 | Proxy FP16 info |
|---|---|---|---|---|
| vexcalibur / v1-2 · 640×640 | PASS 319/319 | FAIL 306/319 | PASS 63/63 | FAIL 61/63 |
| vexcalibur / YOLO11n · 384×640 | PASS 319/319 | FAIL 138/319 | PASS 63/63 | FAIL 5/63 |
| vexcalibur / YOLO26n standard · 384×640 | PASS 319/319 | FAIL 311/319 | PASS 63/63 | FAIL 60/63 |
| vexcalibur / YOLO26n one-to-one · 384×640 | PASS 319/319 | FAIL 309/319 | PASS 63/63 | FAIL 59/63 |
| xcng1 / v1-2 · 640×640 | PASS 319/319 | FAIL 303/319 | PASS 63/63 | FAIL 59/63 |
| xcng1 / YOLO11n · 384×640 | PASS 319/319 | FAIL 137/319 | PASS 63/63 | FAIL 5/63 |
| xcng1 / YOLO26n standard · 384×640 | PASS 319/319 | FAIL 313/319 | PASS 63/63 | FAIL 60/63 |
| xcng1 / YOLO26n one-to-one · 384×640 | PASS 319/319 | FAIL 309/319 | PASS 63/63 | FAIL 61/63 |

FP16 has **560 failed image/package comparisons**, including **94 count
mismatches**. Maximum associated FP16 box/score deltas are **68.162842 px /
0.271183**. Many YOLO11n FP16 failures involve only clipped zero-area box
endpoints; the summary records those separately. These are parity diagnostics,
not accuracy results. WS7 owns the final FP16/int8 accuracy gate.

Zero-area boxes have undefined IoU. Trained exports exposed cancellation in
small endpoints that the original per-coordinate ULP check could not handle.
The final numeric rule is
`abs(delta_px) <= 8 * float32 epsilon * max(abs(coordinates))` across either
box, about one part per million of the box's coordinate scale. Normal boxes
retain `compare_exports.py` IoU ≥ 0.8 and score delta ≤ 0.05, with equal
classes and counts. Tests reproduce both Mac and Linux endpoint cases and
reject larger shifts and excessive score differences. No C++ inference code
changed during round 3.

One Slurm job, **945637**, `cz-ws8-parity`, normal/xcng1, nice 1000, four CPUs,
eight GB, two-hour limit; all model cases pinned to **cores 0–3**, verified
in report metadata. It was requeued once after archive-sibling staging failed
in preflight. The runner now stages the archive beside the extracted dataset.
The final allocation ran **18 min 7 s** and produced all 16 reports, passed
freeze verification, CTest and **78 native/runtime tests**. Its raw batch exit
was **1**, because the superseded fixed-eight-ULP comparator rejected one
zero-width endpoint: **7.390136 vs 7.391154 px**, at a 1276.99 px box scale.
All required outputs pass the corrected relative comparator.

The original raw reports and predictions remain unchanged. Both hosts' saved
predictions were rescored on Mac using `native.rescore_parity`; the reports
record source-report, raw-prediction and comparison-code hashes. This is
explicitly post-inference comparison, not a second arm inference run. No
second Slurm job was submitted. The allocation is closed; no WS8 CPU job remains.

Mac checks pass: **212 tests**, including **20 native tests**, plus Ruff,
strict mypy including native, and CTest. Without the extension, all tests pass
with seven clean skips. Linux's 78-test result precedes the final comparison-only
change; the new Linux-prediction rescore regression passes on Mac.

Committed summary: `native/results/ws8-frozen-parity.json` and
`native/results/frozen-parity.html`. Raw and rescored reports, freeze logs,
Slurm accounting and check logs: `outputs/ws8/round3/`. Cluster inputs/results
remain under `~/columba/scratch/ws8/round3/`; node-local data is cleaned by the
job's EXIT trap. Repeatable runners are `native/run_frozen_parity.sh` and
`native/run_frozen_arm_job.sh`; `python -m native.rescore_parity --source RAW
--output RESCORED` repeats comparison without modifying inference outputs.

## Round-2 compute modes and measured results

`compute=None` selects FP16 for C8 `quantization: fp16` and `int8`, FP32 for
`fp32` and legacy packages. `compute="fp16"` explicitly sets
`use_fp16_packed`, `use_fp16_storage`, `use_fp16_arithmetic` all true;
`compute="fp32"` sets all three false. Int8 kernels remain enabled. Options
are applied before loading the model, matching WS4's benchmark flags.

**Measured**, Mac `vexcalibur`, M2 Pro, shared-machine noise, four threads,
same 13-image set and seven packages. FP32 is forced explicitly for strict
PyTorch parity: **91/91 image/package comparisons pass**, with no count
mismatches, maximum associated box delta **0.004044 px** and score delta
**0.000006050**. FP16 uses documented looser IoU ≥ 0.7 and score delta ≤ 0.10,
equal classes/counts; strict FP32 results are retained alongside it.

| Package / input H×W | FP16 max box delta px | FP16 max score delta | Count-mismatch images | Loose failures / 13 |
|---|---:|---:|---:|---:|
| v1-2 / 640×640 | 1.072 | 0.009265 | 0 | 1 |
| YOLO11n / 640×640 | 1.680 | 0.032253 | 0 | 0 |
| YOLO11n / 384×640 | 36.477 | 0.019075 | 0 | 0 |
| YOLO26n standard / 640×640 | 4.745 | 0.013528 | 0 | 0 |
| YOLO26n standard / 384×640 | 11.656 | 0.052865 | 0 | 0 |
| YOLO26n one-to-one / 640×640 | 4.011 | 0.027678 | 1 | 1 |
| YOLO26n one-to-one / 384×640 | 6.631 | 0.021514 | 1 | 1 |

Maximum deltas apply to above-threshold, same-class diagnostic associations
with IoU ≥ 0.5, independently of score drift; unmatched boxes remain reported.
Across all packages there are **two count-mismatch image/package pairs**:
one-to-one 640 has PT/native **5/4** on `000000000009.jpg`; one-to-one 384 has
**5/6** on `bus.jpg`. The baseline's extra failure has equal counts and a
zero-height clipped box: right edge **636.948303 vs 637.0 px** fails the
unchanged one-float32-ULP degenerate rule. There are three loose failures
and four strict failures in 91 comparisons; the fourth strict failure is
standard YOLO26n 384's score delta above 0.05. No FP16 accuracy pass is claimed.

**Measured** baseline v1-2, 640×640 input, 480×640 frames, four threads,
20 warm-ups, 100 alternating samples per backend, WS4 `mac-a76compat` prefix:

| Compute / backend | Total p50/p95 ms | Extract p50/p95 ms | Overhead p50/p95 ms |
|---|---|---|---|
| FP16 / native | 14.729 / 17.677 | 14.441 / 17.335 | 0.310 / 0.360 |
| FP16 / Ultralytics NCNN | 15.742 / 18.255 | 14.363 / 17.018 | 1.307 / 1.608 |
| FP32 / native | 23.977 / 27.246 | 23.672 / 26.918 | 0.318 / 0.345 |
| FP32 / Ultralytics NCNN | 24.766 / 26.706 | 23.592 / 25.438 | 1.194 / 1.356 |

WS4's matching v1-2 640×640 Mac FP16 record has inference p50/p95
**14.515 / 17.512 ms**, versus native **14.441 / 17.335 ms**: differences
**−0.51% / −1.01%**, consistent within shared-machine noise. WS4 used 200 timed
iterations and includes a Mat copy plus input submission in its inference
boundary; native times extract only. The model weights have the same SHA-256
`50463d299ed5d373aa6a32f23e58f603a67f3c214718242678911e4f677776e6`;
parameter files differ. Native FP16 extract is **1.64× faster** at p50 than
the new FP32 measurement. These are Mac host measurements, not Pi estimates.

Committed summary: `native/results/ws8-compute-modes.json`. Raw reports and
samples: `outputs/ws8/results-round2-{fp32,fp16}/`. Comparison source:
`experiments/latency/vexcalibur-pigeon_baseline_v1-2-ncnn-fp16-640x640.json`
and `experiments/latency/summary.md`. The stable NCNN prefix was built using
the merged script at
`/Users/vari/Developer/columba-zero/native/build/ncnn-mac-a76compat`, and the
new `--opencv-only` build was executed successfully. Build commands are in
`native/README.md`. Round-2 check logs are `outputs/ws8/*round2.log`.

## Round-1 results

**Measured**, baseline NCNN, 640×640 input, 480×640 BGR frames, four threads,
20 warm-ups and 100 alternating samples per backend. Both networks receive
matching FP32 compute and thread options before model loading. Overhead is
full Python-call wall time minus that call's NCNN extract time; IPC is excluded.
Mac is an M2 Pro shared with other workers, so its timings include machine noise.
Linux is Neoverse-N1 `xcng1`, with `taskset -c 0-3`, verified in the JSON.
These are host measurements, not Pi measurements.

| Host / backend | Total p50/p95 ms | Extract p50/p95 ms | Overhead p50/p95 ms |
|---|---|---|---|
| Mac `vexcalibur` / native | 28.630 / 42.796 | 28.032 / 42.028 | 0.333 / 0.844 |
| Mac `vexcalibur` / Ultralytics NCNN | 29.649 / 45.915 | 27.546 / 43.991 | 1.553 / 2.996 |
| Linux `xcng1` / native | 82.822 / 124.774 | 81.872 / 123.644 | 0.987 / 1.377 |
| Linux `xcng1` / Ultralytics NCNN | 91.169 / 133.872 | 86.648 / 128.634 | 4.770 / 5.855 |

Committed summary: `native/results/ws8-fp32-smoke.json`. Raw samples and box
comparisons: `outputs/ws8/results-{mac,linux}/`. Ignored build logs and test
logs are under `outputs/ws8/`. Cluster artifacts remain under
`~/columba/scratch/ws8/`, never WS3's environments; staged datasets under
`/tmp/cz-$SLURM_JOB_ID/` are cleaned at job exit. Source weights and baseline
packages in the main checkout were read and copied, never modified.

Verified details and limits:

- Installed Ultralytics 8.4.175 scales x/y with separate gains from rounded
  resized dimensions. Letterboxing matches it byte for byte on ten shape cases.
- FP16 arithmetic previously changed a one-to-one COCO8 score from 0.286
  to 0.226. Round 2 restores FP16 arithmetic as an explicit mode and the default
  for FP16/int8 packages, while exact parity explicitly forces FP32.
- One legacy box clips to zero height. Its IoU is undefined. The runner records
  a same-class match only within one float32 ULP per coordinate, while retaining
  the 0.05 confidence tolerance. Normal boxes use `compare_exports.py` IoU ≥ 0.8,
  confidence delta ≤ 0.05, equal classes and equal counts. Tests reject larger
  degenerate-coordinate differences.
- Mac parity/benchmark tools use separate native/PyTorch processes. Loading
  Homebrew and PyTorch OpenMP together was reproduced as a deadlock; the native
  application path has no PyTorch dependency.
- Both model decoders run all classes. Class-aware one-to-one top-k can keep
  multiple classes at one anchor; agnostic top-k takes the best class only.

Cluster execution trail, all sequential `cz-ws8-build` jobs with nice 1000:
`945168` installed packages but hit unsupported `/tmp` permission preservation;
`945271` built and passed six export cases before the degenerate-IoU comparison;
`945326` passed all parity and measured overhead, then needed Git deployment
fixtures for two repository tests; `945343` passed all 57 tests and exposed a
missing staged `.gitignore` that made Ruff scan dependencies; `945360` passed
all static checks after staging it. `945441` checks the new WS9 integration and
WS4 prefix and completed with exit 0. Full Git staging is now documented,
and no vendor source was edited.
