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

### Round 2 (review fixes, rebased onto main 222ba8f)

Each fix has a regression test that failed on the old code.

1. **Trigger classes:** `PERMITTED_TRIGGER_CLASSES = {"pigeon"}` (mirrors C2).
   Packages listing any other trigger class are rejected.
   Test: `DetectorTests.test_trigger_classes_limited_to_pigeon` (`spotted_dove`,
   both `load_model_info` and the native factory).
2. **Camera-switch cleanup:** a target camera that fails to start and then fails
   cleanup raises `CameraCleanupError`, which carries both errors. The previous
   camera is not reopened and the run ends with an error that is recorded in
   `run_end`. A capture cleanup error in `run` is now appended to an earlier
   failure instead of replacing it.
   Test: `DayNightTests.test_failed_switch_cleanup_failure_is_fatal`.
3. **Records after a render failure:** logging now runs on its own `LogWriter`
   thread. After the worker is joined, `finish()` writes every queued record before
   `run_end`, including after display or recording failures.
   Test: `RuntimeTests.test_render_failure_keeps_frame_and_trigger_records` (four
   processed frames, four logged, and the frame-3 trigger is present).
4. **`decode: yolo26` on Ultralytics:** the Ultralytics backend rejects it with a
   `DetectorError` naming `model.backend: native`.
   Test: `DetectorTests.test_ultralytics_backend_rejects_nms_free_decode`.
5. **Abnormal worker exit:** the worker catches `BaseException`, and
   `ProcessingResult.completed` marks normal completion; anything else is reported
   as an error.
   Test: `RuntimeTests.test_worker_base_exceptions_are_failures` (`SystemExit`,
   `KeyboardInterrupt`).
6. **Bounded log queue:** capacity 4096, separate from the render queue. On
   overflow only `frame` records are dropped and counted (`log_dropped_records`
   in `health` and `run_end`). The unused trigger history is removed.
   Tests: `OutputQueueTests.test_log_queue_is_bounded_but_keeps_events`,
   `RuntimeTests.test_display_stall_does_not_block_logging`,
   `RuntimeTests.test_log_writer_stall_drops_frames_but_keeps_triggers`.
7. **Summary overflow:** values of 10 s or more are no longer put in the last
   histogram bin. Up to 1000 of them are kept exactly; beyond that, affected
   percentiles are reported as 10000 ms and listed in `lower_bound`.
   Test: `InstrumentationTests.test_summary_reports_values_beyond_the_histogram`
   (60 s sample).

### Round 3 (review fixes, rebased onto main a15ed4c)

Each fix has a regression test that failed on the old code.

1. **Log queue bound:** frame records and other events wait in separate deques
   and are merged back by sequence number, so enqueueing is O(1). When the queue
   is full and no queued frame can be evicted, the incoming frame is dropped and
   counted. Protected events may exceed the capacity by
   `LOG_EVENT_ALLOWANCE = 1024`. Past that, `overflow_error` is set, the run stops,
   events already accepted are still written, and `run_end` reports
   `reason: error` with exit 1. Tests:
   - `OutputQueueTests.test_event_only_backlog_drops_frames_and_fails_past_allowance`
   - `OutputQueueTests.test_log_enqueue_cost_does_not_grow_with_backlog` (the old
     queue was ≈ 178× slower with a full backlog; the limit is 5×)
   - `RuntimeTests.test_trigger_backlog_beyond_allowance_stops_the_run`
2. **Reaction-test race:** `ReactionAnalysis` only collects stimulus, trigger and
   detection records; `report()` matches them by timestamp afterwards, so arrival
   order does not matter. Test:
   `ReactionTests.test_analysis_is_independent_of_arrival_order` (trigger
   delivered before the stimulus is registered).
3. **Abnormal camera start:** `SwitchingCameraStream._open` cleans up on any
   `BaseException` and re-raises it. A stream whose cleanup failed is kept, and
   `close()` retries it once, reporting failures. Tests:
   - `DayNightTests.test_abnormal_camera_start_is_cleaned_up` (`KeyboardInterrupt`)
   - `DayNightTests.test_failed_cleanup_is_retried_at_teardown`

### Round 4 (review fixes, rebased onto main 435e7b0)

1. **Permanently blocked log writer:** the writer thread is a daemon, and
   `LogWriter.finish(LOG_DRAIN_TIMEOUT_S = 2.0)` drains with a deadline. On
   timeout it returns the unwritten count, and the run reports "log writer
   stalled; N records unwritten" on stderr. `run_end` goes to stderr and is not
   written to the blocked sink, and the log is not closed under the writer. The
   camera and recording are still closed, and the run exits 1.
   Test: `RuntimeTests.test_permanently_blocked_log_writer_cannot_block_exit`
   (the old code hung until killed).
2. **Real camera release retry:** `Picamera2Capture.close()` keeps the camera
   until its release succeeds, remembers a successful `stop()` so it is not
   repeated, and lets a failed release propagate. `SwitchingCameraStream` also
   keeps a stream whose close fails during a switch.
   Test: `DayNightTests.test_teardown_retries_the_real_camera_release` (real
   `SwitchingCameraStream`, `LatestFrameStream` and `Picamera2Capture` over a
   mocked Picamera2; release fails once, and always).

Reviewer probes (Mac):
- `/tmp/columba_round3_extra.py`, unchanged: all pass.
- `/tmp/columba_round3_review.py`: queue, concurrency and reaction probes pass.
  - `camera_retry` asserted the old behaviour (1 close); it now shows 2 closes,
    1 stop and the camera released.
  - With that assertion and the stall probe's `finish` wrapper signature adapted
    (`/tmp/columba_round3_review_fixed.py`), the runtime probes pass. The stalled
    run returns status 1 within the 4 s limit (whole script 3.5 s), with a daemon
    writer and "log writer stalled; 9 records unwritten".

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
  - health and run_end: `render_dropped_frames`, `log_dropped_records`
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
| Round 4 rerun of both replays (after rebase onto 435e7b0 and fixes) | identical: counts `0, 1, 0, 1, 0`, trigger at frame 3 with confirmation, 5 processed, 0 drops, 5 recorded | measured |
| Round 3 rerun of both replays (after rebase onto a15ed4c and fixes) | identical: counts `0, 1, 0, 1, 0`, trigger at frame 3 with confirmation, 5 processed, 0 drops, 5 recorded | measured |
| Round 2 rerun of both replays above (after rebase and fixes) | identical: counts `0, 1, 0, 1, 0`, trigger at frame 3 with confirmation, 5 processed, 0 log and render drops, 5 recorded frames | measured |
| Rectangular C8-style package: WS0's `yolo26n` NCNN export at `[384, 640]`, class 0 relabelled `pigeon` for the smoke test, `columba:` block, `image_size: [384, 640]`, confirmation on | runs via Ultralytics; tile detected as `bird` (class_id 14, tracked as track 1), no trigger (not a trigger class) | measured |

Artifacts (git-ignored, in this worktree): `outputs/ws9/replay/smoke.avi`,
`outputs/ws9/replay/{default,confirm}/*/detections.jsonl` and `summary.json`,
`outputs/ws9/rect/runs/*/detections.jsonl`.
