# Pi 5 detection prototype acceptance

This document replaces the historical actuation prototype plan. Scope is pigeon
detection only: no laser, servos, interlocks, or hardware outputs.

## Completed locally

See [validation results](deploy/VALIDATION.md) for tests, checkpoint evaluation,
PT/NCNN comparison, and replay/recording checks. Preserve the original training
run and verify runtime model copies using `tools/prepare_models.py`.

## Camera and monitor acceptance on Pi 5

1. Install using README instructions. Confirm 64-bit OS and capture the dependency
   versions in `outputs/setup/installed.txt`.
2. With recording off, run `columba-zero`. Check red/blue objects for correct
   colors, full-frame box alignment, confidence labels, count, and separate
   capture/processing FPS. Confirm a clear scene does not produce stale boxes.
3. Test q, window closure, Ctrl+C, SIGTERM, missing model, invalid configuration,
   unavailable camera, and failed video opening. Confirm clean exit, released
   camera, readable logs, and successful restart. Do not disconnect CSI cables
   while powered; test unavailability with the camera disabled or disconnected
   before boot.
4. Run with `--record`; confirm annotated AVI is playable and matches processed
   frames. Compare performance with recording disabled.

## One-hour endurance test

Run on the actual Pi with its intended cooling and power supply:

```bash
timeout --signal=TERM --preserve-status 3600 columba-zero
python tools/summarize_run.py outputs/<run>/detections.jsonl > outputs/<run>/summary.json
vcgencmd get_throttled
```

Repeat with `--record` if recording will be used in practice. Review health
samples and summary for sustained processing FPS, inference mean/max latency and
last-1000-frame p95, buffer drops, RSS trend, maximum temperature, throttling, and
errors. Confirm duration is at least one hour, logs/recordings remain readable,
and RSS settles rather than growing continuously. Record actual values, OS,
camera, cooling, and recording setting; no target FPS is claimed in advance.

## Field footage review

Use representative positive and negative footage covering intended lighting,
distance, backgrounds, and other birds. Record missed pigeon events and false
positives per hour, including clips with no pigeons. The current held-out test
split has no background-only images, so it cannot establish negative-scene
performance. Choose confidence on separate tuning footage, then evaluate it on
untouched footage. If accuracy is insufficient, improve data/model separately
from the runtime rewrite; do not repeatedly tune on the held-out test split.

Record Pi acceptance results in `deploy/VALIDATION.md` when hardware testing is
available. Unattended startup, tracking, and actuation remain outside this version.
