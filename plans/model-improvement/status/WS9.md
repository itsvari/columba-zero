# WS9 — Runtime pipeline features

- State: done  <!-- not started | in progress | blocked | done -->
- Branch: `ws9/runtime`
- Agent: Claude Opus 5.5 via T3 delegate
- Last updated: 2026-10-14

## Done

All eight PLAN.md WS9 tasks, each with mocked unit tests (85 tests in total, up from 44).
Defaults keep the previous behaviour. Every new feature is opt-in, apart from the
render/log thread, which keeps the previous exit and cleanup semantics.

1. **Rectangular input.** `model.image_size` takes an int (square) or `[h, w]`. Both
   sides must be multiples of 32. `check_model`/`load_model_info` compare it with the
   export's `imgsz` and with `columba.input_hw`.
2. **Multiple classes.** `Detection(label, confidence, l, t, r, b, class_id=0,
   track_id=None)`. Names and `trigger_classes` come from C8 metadata.
   `validate_names` requires `pigeon` at index 0 with contiguous, unique names.
   Packages without a `columba:` block (v1-2) are read as `decode: yolo11`,
   `trigger_classes: [pigeon]`. The Ultralytics backend accepts `decode` yolo11 and
   yolo26 and asks for `backend: native` otherwise. Backend switch:
   `model.backend: ultralytics | native`, `model.threads`; `app.main.create_detector`
   imports `app.native_detector.NativeDetector` lazily and raises `DetectorError` if
   it is missing.
3. **Tracker and confirmation** (`app/tracking.py`, `confirmation:` section;
   `tracking:` is still rejected). Greedy IoU association, class-agnostic. A frame
   is a hit only when the matched detection is a trigger class. k-of-n per track,
   one `trigger` per track, tracks end after `max_age_frames` misses and reset on a
   camera switch.
4. **Render/log thread** (`app/pipeline.py`). Capture, detection and confirmation
   run on a processing thread. The main thread keeps the OpenCV window (needed on
   macOS/Qt), the JSONL log and the recording. The live render queue holds 2 frames
   and drops the oldest. Replay waits instead, so recordings keep every frame. Log
   records are never dropped.
5. **Camera controls** (`camera:`): `stream`, `main_size`, `lens_position`,
   `max_exposure_us`, `tuning_file`, `camera_index`, `name`, each with a fallback
   and a warning when unsupported. Frames now come from `capture_request`, so
   packets carry lux and sensor-to-delivery time.
6. **Day/night switching** (`app/switching.py`, `cameras:` list and
   `camera_switch:`). One camera open at a time, lux hysteresis with a hold time,
   C9 `camera_switch` events. A failed switch reopens the working camera and
   retries after 60 s.
7. **Instrumentation.** Frame records gain C9 `stage_ms`, `camera`, plus `lux`,
   `delivery_ms`, `decided_at` and `decision_ms`. `tools/summarize_run.py` reports
   distributions (0.1 ms histograms, bounded memory).
8. **`--reaction-test IMAGE [--reaction-trials N]`** (`app/reaction.py`). No image
   is committed; a path is required.

README configuration docs and `app/config.yaml` are updated.

## In progress

Nothing.

## Blockers and requests

For the coordinator:

- **C8 vs. Ultralytics directory naming.** Ultralytics 8.4.175 recognises an NCNN
  export only when its directory name ends in `_ncnn_model`, so a C8 package at
  `models/<model_id>/` fails to load. The runtime now works around it: it links
  the package under a `_ncnn_model` name in a temporary directory. Other tools
  that call `YOLO(path)` on a C8 package (WS1, WS7, `compare_exports.py`) need the
  same handling, or C8 should note the suffix.
- **C9 additions beyond the contract** (please fold them into C9 and bump the
  version):
  - frame: `lux`, `delivery_ms`, `decided_at`, `decision_ms`
  - trigger: `first_seen_frame`, `frame_number`
  - camera_switch: `active`, `switch_ms`, `error` (the `reason` values are `dark`
    and `bright`)
  - health and run_end: `render_dropped_frames`
  - new events `stimulus` and `reaction_test`
- **WS8 interface notes.** `create_detector` calls `NativeDetector(model.path,
  model.image_size, model.confidence, threads=model.threads)`. `image_size` is an
  `int` or a `tuple[int, int]` (`(h, w)`), never a list. Requested:
  - Expose an optional `last_stage_ms: dict[str, float | None]` with keys
    `preprocess`, `inference` and `postprocess`, updated by each `detect`.
    Without it the whole detect call is logged as `inference`.
  - Use lower-case labels from `names`, and `class_id=`.
- **Behaviour change in `app/dataset.py`.** Through `validate_names`,
  `tools/train_model.py` and `tools/dataset_audit.py` now accept multi-class
  datasets with `pigeon` at index 0 (C2 species taxonomy) instead of exactly one
  class.
- **WS10 checks (no Pi here, all mocked):**
  - Is `SensorTimestamp` on CLOCK_BOOTTIME (`delivery_ms` assumes it; implausible
    values log as null)?
  - Does the lores stream give RGB888 on the Pi 5?
  - Do the installed tuning files use AGC `shutter` or `exposure` keys (both are
    handled)?
  - Lux thresholds for switching (the NoIR camera reads high under IR).
  - Display latency in `--reaction-test`.

## Results

Host: Mac M2 Pro, shared with other workers. Python 3.14, Ultralytics 8.4.175,
ncnn 1.0.20260526. Timings are noisy and say nothing about the Pi.

| Check | Result | Label |
|---|---|---|
| Tracker overhead, 20 detections × 1000 frames (unit benchmark, best of 3) | ≈ 0.07 ms/frame (budget 1 ms) | measured |
| Replay, `pigeon_baseline_v1-2/best_ncnn_model`, 5-frame clip (blank/positive tile alternating, from `outputs/validation/style-cleanup/images/`), default config | pigeon counts `0, 1, 0, 1, 0` (conf 0.866), 0 drops, `video_complete`, recording decodes to 5 frames; inference p50 ≈ 12 ms, decision max ≈ 21 ms | measured |
| Same, `confirmation.enabled: true` | identical counts; 1 `trigger` (track 1, first seen frame 1, confirmed at frame 3, k=2/n=3) | measured |
| Rectangular C8-style package: WS0's `yolo26n` NCNN export at `[384, 640]`, class 0 relabelled `pigeon` for the smoke test, `columba:` block, `image_size: [384, 640]`, confirmation on | runs via Ultralytics; tile detected as `bird` (class_id 14, tracked as track 1), no trigger (not a trigger class) | measured |

Artifacts (git-ignored, in this worktree): `outputs/ws9/replay/smoke.avi`,
`outputs/ws9/replay/{default,confirm}/*/detections.jsonl` and `summary.json`,
`outputs/ws9/rect/runs/*/detections.jsonl`.
