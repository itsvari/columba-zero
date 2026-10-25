# Validation status

## Pan-tilt software validation — 25 October 2026

- Calibration follow-up: 60 total unittest cases passed after swapping default
  pan/tilt channels to 0/1 and adding immediate arrow-key movement. Coverage
  includes terminal key decoding, command editing, bounds, routing, CLI overrides,
  and restoration of terminal settings after interruption. Real SSH keyboard
  behavior and physical servo routing still require verification on the Pi.
- 51 automated unittest cases passed on the existing Windows desktop environment.
  New coverage includes target association, loss/reacquisition, return-home timing,
  stale observations, deadband, speed/travel limits, calibration/profile validation,
  PCA9685 register/pulse writes and channel-specific disable, startup/worker/cleanup
  failures, worker shutdown, runtime modes, overlays, and JSONL tracking fields.
- `ruff check app tests` passed using Ruff through `uv tool run`.
- A real NCNN replay of `outputs/validation/replay_fixture.avi` processed five frames
  in simulation mode, selected a target in every frame, logged five tracking records,
  and produced an annotated MJPEG AVI that decoded to five frames. It exited cleanly
  at the requested frame limit. Evidence is in ignored
  `outputs/validation/pan-tilt-replay/20261025T071858Z-f72ae117/`.
- No physical Pi, HAT, servo, power, or CSI motion acceptance is claimed. Wiring,
  axis direction, calibrated travel, stable following, target-loss return, real
  shutdown behavior, and one-hour operation remain pending. Follow the pan-tilt
  section of `../RASPBERRY_PI_PROTOTYPE_PLAN.md`.

## Detection baseline — 10 October 2026

## Completed on Windows desktop

- 25 automated unittest cases passed: configuration and old CLI rejection, model
  metadata/files/classes, inference filtering/warmup, sequential video completion,
  bounded live buffer and drops, capture errors/timeouts, GUI exit behavior,
  JSONL output, recording failure, startup/runtime cleanup, frame limit, and
  absence of GPIO imports, CLI overrides, SIGTERM handling/restoration, and
  cleanup failures.
- Runtime model copies match SHA-256 values in `model_manifest.json`.
- CPU PT/NCNN comparison: **63 test images, zero comparison failures** at confidence
  0.5, input 640, square preprocessing, NMS IoU 0.7. All counts matched and matched
  boxes met IoU >=0.8 and confidence difference <=0.05. See
  `export_comparison_summary.json`; raw predictions are in ignored
  `outputs/validation/export_comparison.json`.
- Authoritative `best.pt` evaluated on the existing **test** split: 63 images,
  183 labeled instances, zero background-only images. Precision **0.9049**,
  recall **0.8318**, mAP50 **0.9048**, mAP50–95 **0.5728**. These are test metrics,
  separate from the training log. Confidence floor 0.001 enables AP integration;
  precision/recall are curve-selected, not fixed-threshold 0.5 measurements.
  See `pt_test_metrics.json`. Split independence/session provenance was not audited.
- Real NCNN runtime processed a five-frame AVI made from held-out images in
  sequence, exited at EOF, logged all five frames with zero drops, and produced
  an annotated MJPEG AVI that decoded to five frames. GUI was disabled for this
  integration check; window behavior was tested with mocked OpenCV calls.

Environment: Windows 11, Intel Core i7-13700K, Python 3.12.10,
Ultralytics 8.4.175, NCNN 1.0.20260526, Torch 2.14.1+cu132 (CPU inference),
OpenCV 5.0.0.93, NumPy 2.5.3. Desktop timing was measured while two validation
processes ran concurrently; do not treat it as a performance benchmark. The
existing desktop venv contains both OpenCV pip variants; new installs must use
only one GUI-capable OpenCV provider.

The evaluator could not write its optional label cache outside the workspace;
validation completed successfully without it. Original labels were unchanged.

## Pending on physical Pi 5

Pi OS installation and ARM dependency compatibility, CSI colors and box alignment,
actual monitor interaction, camera restart/shutdown, recording overhead,
one-hour memory/thermal/throttling behavior, and representative positive/negative
footage review. No Raspberry Pi or camera is connected to this workspace, so no
Pi performance or hardware acceptance is claimed. Follow
`../RASPBERRY_PI_PROTOTYPE_PLAN.md` and append actual measurements here.
