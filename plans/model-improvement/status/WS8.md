# WS8 — Native C++ detector

- State: done
- Branch: `ws8/native`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14
- Rebased onto main: `d57919a`, including WS9 `b902e04` and CONTRACTS v1.5.

## Done

1. `columba_native.Detector(model_dir, threads=4)`: C8 and legacy metadata,
   OpenCV `INTER_LINEAR` letterboxing, NCNN `in0`/`out0`, both decoders,
   zero-copy uint8 BGR input, released GIL and owned float32 N×6 output.
   Properties: `input_hw`, `names`, `decode`. `detect_with_timings(frame, conf)`
   adds C++ `preprocess`, `extract`, `postprocess`, `total` measurements in ms.
2. Builds and C++ tests pass on Mac ARM64 and Linux aarch64 `xcng1`, using
   NCNN tag `20260526`, commit `e54f7b1f88434e1d844ea0551b880a1cfb079ce1`,
   Vulkan off, and OpenCV `5.0.0`. Plain CMake supports WS4's standard
   `-Dncnn_DIR=<prefix>/lib/cmake/ncnn` target. Both WS4 prefixes also build
   and pass C++ tests; its exact Mac/Linux commands are in `native/README.md`.
3. `NativeDetector(path, image_size, confidence, threads=4)` matches WS9's
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
6. After WS9 rebase: all **201 Mac tests pass** with the extension, including
   59 runtime tests and 15 native tests. Ruff lint/format and strict mypy pass
   on `app tests tools cluster training native`, with `native/build` excluded
   from mypy. Without the extension the five extension-only tests skip cleanly.
   After rebase, all **201 Linux tests pass** against WS4's prefix, along with
   Ruff and strict mypy. Job `945441` completed with exit 0. No WS8 background
   process or running Slurm job remains.

## In progress

Nothing. WS8 tasks 1–5 are complete for the coordinator-authorised smoke dataset.
The unavailable frozen datasets remain an explicit follow-up, not a claimed pass.

## Blockers and requests

- WS0/coordinator: add `native` to mypy's `files` and exclude `^native/build/`;
  include `native` in repository ruff commands. The plain CMake build requires
  no top-level build-system change. `pyproject.toml` was not edited by WS8.
- Coordinator: rerun parity on frozen `columba-test-v2` when ready. The old
  63-image split cannot be reconstructed. Use
  `python -m native.verify_native --model DIR --reference ORIGINAL.pt
  --dataset DATASET --split test --output REPORT.json`.
- Real-model checks cover FP32. FP16/int8 need their export-specific parity
  gates. Precision is explicit: FP32/legacy use FP32 activations; FP16 allows
  half storage with FP32 arithmetic. Package weight quantisation and runtime
  compute precision are distinct, as CONTRACTS v1.5 C3 records.
- The temporary WS8 dependency helper can be reconciled with WS4's script after
  merge. CMake already accepts both WS4 prefixes. WS9 wiring is implemented;
  no outstanding class-ID or timing switch remains.

## Results

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
- NCNN's default FP16 arithmetic changed one one-to-one COCO8 score from 0.286
  to 0.226. FP32 compute resolved that discrepancy. Arithmetic remains FP32
  when half storage is enabled; no int8 accuracy claim is made.
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
