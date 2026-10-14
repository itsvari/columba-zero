# WS8 — Native C++ detector

- State: in progress
- Branch: `ws8/native`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14

## Done

- `columba_native.Detector(model_dir, threads=4)`: C8 and legacy metadata,
  OpenCV-exact letterboxing, NCNN `in0`/`out0`, both decoders, zero-copy input,
  released GIL and owned float32 N×6 output. Added `detect_with_timings`.
- Mac ARM64 build against NCNN tag `20260526` and OpenCV `5.0.0`; C++ tests pass.
- Wrapper matches WS9's factory signature and exposes `last_stage_ms` with
  `preprocess`, `inference`, `postprocess`. Names become lower-case labels.
- Reusable `native.verify_native --dataset DIR --split test` and local-image
  input. Ultralytics NCNN directories without a suffix use a temporary symlink.
- Mac parity passes for all seven packages on 13 real/local images: YOLO11n,
  standard YOLO26n, and one-to-one YOLO26n at 640×640 and 384×640, plus legacy
  v1-2 at 640×640. These are smoke parity checks, not accuracy measurements.
- Mac full suite: 57 tests pass with the extension. Ruff and strict mypy pass,
  including native Python sources with the build directory excluded.

## In progress

Linux aarch64 build, parity and overhead are running in Slurm job `945271`,
`cz-ws8-build`, `normal`, `xcng1`, eight allocated CPUs, 16 GB, two-hour limit,
`--nice=1000`. The first job, `945168`, installed dependencies but stopped when
`cp -a` tried to preserve unsupported permission metadata on `/tmp`. The retry
uses `cp -R --no-preserve=all` and reuses the environment. Only one WS8 CPU job
runs at a time. Cluster scratch is `~/columba/scratch/ws8/`; datasets are staged
under `/tmp/cz-$SLURM_JOB_ID/` and cleaned at job exit.

## Blockers and requests

- WS0: include `native` in mypy's `files`, exclude `^native/build/` from mypy,
  and include `native` in repository ruff commands.
  The plain CMake build needs no top-level build-system change.
- WS9: add `class_id: int = 0` to `Detection`. WS8 will preserve native class IDs
  internally. After that merge, add `class_id=class_id` to the wrapper's
  `Detection(...)` call. `_last_class_ids` already retains the IDs. Current
  wrapper validation/filtering deliberately follows current main; if WS9
  changes that contract to return secondary species too, reconcile that filter.
- The frozen test-v2 and old 63-image split are unavailable. This round uses
  local real images and a reusable C1-compatible parity runner.

## Results

**Measured, Mac M2 Pro, host `vexcalibur`, ARM64, shared-machine noise:**
baseline NCNN, 640×640 input, 480×640 BGR frame, four threads, 20 warm-ups,
100 alternating samples per backend in isolated processes. FP32 activations
and thread options are set before model loading on both backends. Overhead is
each full Python call minus that call's NCNN extract time; IPC is excluded.

| Backend | Total p50/p95 ms | Extract p50/p95 ms | Overhead p50/p95 ms |
|---|---|---|---|
| Native | 23.733 / 28.721 | 23.416 / 28.132 | 0.312 / 0.357 |
| Ultralytics NCNN predict | 24.646 / 30.955 | 23.401 / 29.591 | 1.204 / 1.664 |

Raw Mac results: `outputs/ws8/results-mac/`; build logs:
`outputs/ws8/{dependencies,configure,build}-mac.log`; Python suite logs:
`outputs/ws8/unittest-{with,without}-extension.log`. Shared ignored models are
read from the main checkout and copied to `outputs/ws8/weights/`; source
weights and the baseline package are unchanged.

Two verified implementation details:

- Installed Ultralytics 8.4.175 uses gains from the rounded resized dimensions
  separately for x/y. Letterbox output matches it byte for byte on ten shapes.
- NCNN's default FP16 arithmetic changed one one-to-one COCO8 score from
  0.286 to 0.226. C8 FP32 packages now use FP32 activations; FP16 storage is
  opt-in via metadata and arithmetic remains FP32. One identical baseline
  box clips to zero height, so the parity runner records an exact-coordinate
  match for it, while retaining the existing 0.05 confidence tolerance. Normal
  boxes use `compare_exports.py` IoU ≥ 0.8 and confidence delta ≤ 0.05.

Mac parity/benchmark tools isolate native and PyTorch OpenMP runtimes in
separate processes. Loading Homebrew and PyTorch OpenMP together was reproduced
as a deadlock; the runtime native path has no PyTorch dependency.

<!-- Label numbers as measured / estimated / simulated, and name the host. Link artifacts by path. -->
