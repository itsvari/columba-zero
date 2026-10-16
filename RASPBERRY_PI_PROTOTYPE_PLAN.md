# Pi 5 detection prototype acceptance

This document replaces the historical actuation prototype plan. Scope is pigeon
detection only: no laser, servos, interlocks, or hardware outputs. It is the
working checklist for WS10 in the
[model-improvement plan](plans/model-improvement/PLAN.md#ws10--pi-hardware-acceptance-blocked-until-hardware-arrives),
the only open workstream. It is blocked until the Pi 5 and cameras arrive.

The model under test is `yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`: a
448×800 fp16 NCNN package with yolo11 decode and threshold 0.62646484375, run on
the native backend with k2/n3 confirmation and an 800×450 lores stream at 30 FPS
([app/config.yaml](app/config.yaml), [WS7 report](experiments/reports/WS7-final.md)).
`pigeon_baseline_v1-2` stays available as the fallback (see the README).

## Completed locally

See [validation results](deploy/VALIDATION.md) for tests, package verification
and replay checks. `python tools/prepare_models.py` verifies and copies both
packages. All pre-Pi latency numbers are estimates from xcng and a published
calibration factor; reaction times are simulated.

## 1. Platform and latency calibration

1. Install using the README instructions, including NCNN (`--profile pi5`) and
   the native extension (`native/README.md`). Confirm 64-bit OS and capture
   dependency versions in `outputs/setup/installed.txt`.
2. Record the `/proc/cpuinfo` flags and compare them with PLAN section 3.
3. Run the WS4 benchmark bundle ([experiments/latency/README.md](experiments/latency/README.md))
   with `--host-profile pi5` on the shipped package and the baseline. Replace the
   calibration factor with the measured one and save records as
   `experiments/latency/pi5-*.json`.

## 2. Camera and monitor acceptance

1. With recording off, run `columba-zero`. Check that frames reach the detector
   as 800×450 from the full field of view (`camera.stream: lores`), red/blue
   objects have correct colours, boxes align, and labels, count and separate
   capture/processing FPS are shown. Confirm a clear scene leaves no stale boxes.
2. Test q, window closure, Ctrl+C, SIGTERM, missing model, invalid configuration,
   unavailable camera, and failed video opening. Confirm clean exit, released
   camera, readable logs, and successful restart. Do not disconnect CSI cables
   while powered; test unavailability with the camera disabled or disconnected
   before boot.
3. Run with `--record`; confirm the annotated AVI is playable and matches
   processed frames. Compare performance with recording disabled.

## 3. One-hour endurance test

Run on the actual Pi with its active cooler and intended power supply:

```bash
timeout --signal=TERM --preserve-status 3600 columba-zero
python tools/summarize_run.py outputs/<run>/detections.jsonl > outputs/<run>/summary.json
vcgencmd get_throttled
```

Repeat with `--record` if recording will be used in practice. Review health
samples and the summary for sustained processing FPS, stage and decision-time
percentiles, buffer drops, RSS trend, maximum temperature, throttling, and
errors. Confirm the run lasted at least one hour, logs and recordings remain
readable, and RSS settles rather than growing. Record actual values, OS, camera,
cooling, and recording setting; no target FPS is claimed in advance.

## 4. Reaction time

Measure with `columba-zero --reaction-test /path/to/pigeon.jpg` for both
cameras, by day and at night with the IR illuminator. The target is p95 at or
below 500 ms (WS7 simulated 345 ms).

## 5. OpenMP wait policy

Time the full pipeline with `OMP_WAIT_POLICY=ACTIVE` and `PASSIVE` set in the
launch environment before NCNN loads. WS7 measured PASSIVE at more than twice
ACTIVE's NCNN time on xcng, but ACTIVE spins idle threads that may slow capture,
tracking and rendering on four cores. Set the winner in the launcher and record
it (C7 `omp_wait_policy`).

## 6. Site footage, fine-tune and thresholds

1. Collect site footage: dawn, day, dusk and night, both cameras, including long
   empty periods. Label about 300 frames as `columba-test-site-v1`, the first
   real check of 16:9 deploy-geometry accuracy.
2. Fine-tune on the cluster with a round-4 recipe based on the shipped
   `yolo26n-v2s-800-round2-plain` recipe, then re-select and package with
   `tools/select_model.py` ([cluster/README.md](cluster/README.md)).
3. Set the final threshold and k/n from site footage. Choose them on tuning
   footage and evaluate on untouched footage; never tune on a test split.

Record results in [deploy/VALIDATION.md](deploy/VALIDATION.md). Unattended
startup and actuation remain outside this version.
