# Columba Zero

Columba Zero is a safe-by-default Raspberry Pi 4 prototype for detecting pigeons and
driving a mechanically bounded, two-axis bench deterrent. It implements the runtime
described in `RASPBERRY_PI_PROTOTYPE_PLAN.md`; it does **not** make outdoor or live-animal
laser use safe or approved.

The default configuration uses a simulated turret, an in-memory indicator output, and an
uncalibrated identity transform. It cannot energise a laser. Real laser output additionally
requires an enclosed bench rig, explicit configuration, `--hardware`, a physical E-stop,
and a closed hardware interlock. A normally-closed physical shutter and hardwired power
cutoff remain mandatory; software is not the primary safety barrier.

## What is implemented

- bounded latest-frame Picamera2 acquisition and recorded-video replay;
- YOLO/NCNN inference plus deterministic frame-indexed replay detections;
- permitted-ROI filtering, 3-of-5 temporal confirmation, target clustering, and release
  hysteresis;
- person, vehicle, bicycle, and pet blocking, stale-frame checks, E-stop and interlock
  inputs, activation-rate limiting, cooldown, and fault-to-`SAFE` behavior;
- homography-based pixel-to-plane conversion and yaw/pitch geometry;
- speed-limited, clamped, stale-command-rejecting simulation and GPIO Zero turret drivers;
- bounded indicator/laser pulses with output and shutter forced off in every exit path;
- structured JSONL event and health logging;
- held-out evaluation, dataset auditing, NCNN export comparison, calibration, and Pi
  performance benchmark tools;
- a hardened systemd service template and automated safety-logic tests.

## Development setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,vision,metrics]'
pytest
```

On Raspberry Pi OS, install Picamera2 through the OS packages so it matches libcamera:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv
python -m pip install -e '.[pi,metrics]'
```

Do not copy the example config into real operation unchanged. In particular, survey the
ROI, install mechanical stops, calibrate servo pulse widths, and verify the full beam
envelope against a matte beam dump.

## Run modes

The live camera runtime loads the configured YOLO/NCNN model:

```bash
columba-zero --config app/config.yaml
```

The runtime warms up the detector before starting capture or opening GPIO outputs,
so lazy NCNN initialization does not make the first camera frame stale. Startup logs
report the warm-up duration. The configured frame freshness limit still applies to
every real frame; persistent `stale_frame` faults require measuring inference latency.

Replay a recording through the real detector:

```bash
columba-zero --config app/config.yaml --video test.mp4
```

Replay known detections without a model (useful for closed-loop fault testing):

```bash
columba-zero --config config.deployment.yaml \
  --video test.mp4 --detections examples/replay_detections.json
```

`geometry.calibrated` must be true before the controller will aim. Laser mode also forces
`geometry.aim_mode: fixed_safe_surface`, which points only at the surveyed
`approved_target_plane_m`; it never follows the animal coordinate. The calibration tool
intentionally emits `calibrated: false`; set it to true only after physically measuring
pointing error across the whole ROI with the laser disconnected or replaced by an LED.

GPIO is never touched unless `--hardware` is supplied. An enabled laser configuration is
rejected unless physical E-stop and interlock pins are configured. Start hardware tests
with `actuator.mode: indicator` and the laser driver physically disconnected.

## Dataset and model workflow

The saved notebook is retained as a baseline, but its general COCO bird boxes must not be
relabeled as pigeons for deployment training. Build a dataset with explicit `pigeon`,
`other_bird`, and `person` classes, background-only images, and splits grouped by recording
session/location. Then run:

```bash
python tools/dataset_audit.py /data/pigeons/data.yaml
python tools/train_model.py /data/pigeons/data.yaml
python tools/evaluate_model.py best.pt /data/pigeons/data.yaml --confidence 0.55
python tools/export_ncnn.py best.pt --imgsz 416
python tools/compare_exports.py best.pt best_ncnn_model /data/fixed-export-check
python tools/benchmark_detector.py best_ncnn_model sustained-test.mp4
python tools/summarize_events.py logs/events.jsonl
```

The structural audit cannot prove that adjacent frames were kept in the same split; keep a
session manifest and review it. For threshold selection, supplement detector metrics with
false positives per hour and false negatives per complete pigeon event on continuous held-
out recordings.

## Calibration

Measure at least four non-collinear image points and corresponding target-plane positions
in metres:

```json
{
  "image_points": [[100, 100], [500, 100], [500, 400], [100, 400]],
  "plane_points": [[1.0, -0.5], [1.0, 0.5], [2.0, 0.5], [2.0, -0.5]]
}
```

Fit the transform with:

```bash
python tools/calibrate_homography.py measured_points.json
```

Merge the emitted `geometry` values into a private deployment config, add the measured
turret origin and servo offsets, then verify actual pointing error with inert targets at
multiple interior and boundary points. Camera distortion should be corrected before the
homography is fitted.

## State machine and fault behavior

The runtime follows `SAFE → IDLE → VERIFYING → AIMING → ACTIVE → COOLDOWN → IDLE`.
Aiming consumes one confirmed event. Activation requires a later, fresh frame after the
turret settles; the target must still be present and near its original location. A person
or other blocked object rejects the event. E-stop, stale camera data, interlock loss,
calibration errors, detector exceptions, servo timeouts, and actuator exceptions force
`SAFE`, close outputs, and stop servo commands.

The JSONL log defaults to `logs/events.jsonl`. It includes state transitions, rejected
activations, faults, target/angle data, activation durations, inference latency, dropped
frames, memory use, CPU use, and temperature where available.

## Deployment

`deploy/columba-zero.service` is a template. Create the service account and deployment
directories, place the reviewed configuration under `/etc/columba-zero`, write logs to
`/var/log/columba-zero`, and install the unit only after indicator-only bench acceptance.
The service stops outputs on `SIGTERM` and restarts only after a process failure.

Before any field trial, complete every acceptance criterion in the prototype plan and the
qualified laser-safety/site/agency reviews. Never aim the beam at an animal, person,
vehicle, aircraft, reflective surface, or open sky.
