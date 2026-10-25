# Pi 5 camera detection and tracking acceptance

This document covers pigeon detection and the Waveshare Pan-Tilt HAT camera mount.
Run the detection checks with `--tracking off`, then follow the calibration and
tracking checks below.

## Completed locally

See [validation results](deploy/VALIDATION.md) for tests, checkpoint evaluation,
PT/NCNN comparison, and replay/recording checks. Preserve the original training
run and verify runtime model copies using `tools/prepare_models.py`.

## Camera and monitor acceptance on Pi 5

1. Install using README instructions. Confirm 64-bit OS and capture the dependency
   versions in `outputs/setup/installed.txt`.
2. With recording off, run `columba-zero --tracking off`. Check red/blue objects for correct
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
timeout --signal=TERM --preserve-status 3600 columba-zero --tracking off
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
available. Unattended startup remains outside this version.

## Pan-tilt acceptance on Pi 5 (pending)

1. Follow README wiring, power, and calibration instructions. Verify pan S1,
   tilt S0, I2C address, each direction, measured travel limits, home position,
   and camera cable clearance. Preserve the local calibration profile separately.
2. Run live simulation and confirm selected-target overlays and JSONL commands.
   Run physical tracking and verify stable centering with a single pigeon,
   movement on both axes, deadband behavior, speed limits, and bounded travel.
3. Introduce multiple pigeons and crossing paths; document association switches.
   Cover the target briefly: movement should stop. After two seconds without a
   match, verify a smooth return home and successful reacquisition.
4. Measure inference latency and validate observation expiry. Check q, window
   closure, Ctrl+C, and SIGTERM release both channels, close resources, and permit
   restart. Test missing calibration and unavailable I2C at startup. Use injected
   errors for runtime I2C failure rather than disconnecting powered wiring.
5. Repeat the one-hour endurance test with `--tracking hardware` instead of
   `--tracking off` and the intended
   recording setting. Record centering stability, temperatures, power behavior,
   memory, processing FPS, errors, and any cable/mechanical issues.
