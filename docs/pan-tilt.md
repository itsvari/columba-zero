# Prepare the pan-tilt demonstration

Physical movement and Pi performance have not been verified. The software uses the
Waveshare PCA9685 mount described in the [assembly manual](https://files.waveshare.com/upload/b/b9/Pan-Tilt_HAT_user_manual_en.pdf).
Complete assembly and check cable clearance before calibration.

## Calibrate the mount

Use the repository's [Pi installation instructions](../README.md) first. Install the
optional hardware driver in that environment:

```sh
python -m pip install -e '.[hardware]'
```

The calibration command moves both axes to 1500 microseconds immediately. Position
the assembly so that this movement is within its mechanical and cable limits.
Run calibration as the same user that runs detection, locally or over an interactive
terminal:

```sh
python -m app.calibrate --output calibration.local.yaml
```

Use `--line-input` when immediate arrow keys are unavailable. The default channels
are pan 0 and tilt 1. Set `--pan-channel`, `--tilt-channel`, `--bus`, or `--address`
when the wiring differs. The command's `--help` lists pulse-envelope overrides.

1. Move one axis at a time with `pan +`, `pan -`, `tilt +`, or `tilt -`.
2. Mark the safe endpoints and home with `pan min`, `pan home`, and `pan max`.
   Repeat for tilt. Marks refer to increasing pulse width.
3. Set `pan direction 1` or `pan direction -1`, then set the tilt direction.
4. Use `status` to inspect the marks, then `save` to write the profile.

Direction refers to the raw, unflipped camera image. A positive direction means
increasing the pulse aims the camera towards a target to the right for pan, or
down for tilt. A stationary target then moves left or up in the raw image. Use this
convention even when a separate viewer flips or mirrors its preview.

The command refuses to save incomplete or unordered marks. It replaces the profile
atomically after validation. `quit`, Ctrl+C, SIGTERM, and SIGHUP close the driver;
the interactive terminal settings and previous signal handlers are restored.
The local profile is ignored by Git.

The angle mapping uses the nominal 500–2500 microsecond servo range as 0–180 degrees.
The pulse envelope and marked travel limits restrict movement within that range.
Actual angle, speed, and acceleration require measurement on the assembled mount.
Optional `--horizontal-fov-deg` and `--vertical-fov-deg` flags store measured camera
field of view. Leave these unset until measured.

## Follow a recorded video in simulation

Install the desktop following dependencies in the project environment:

```sh
python -m pip install -e '.[follow]'
```

On the Pi, retain OS OpenCV as described in the main installation guide and install
SciPy alongside it. The `follow` extra is for desktop environments using pip OpenCV.

Copy `app/config.yaml` to `outputs/follow.yaml`. Change these sections in the copy,
keeping the shipped model settings:

```yaml
follow:
  enabled: true
pan_tilt:
  mode: simulate
display:
  enabled: false
```

Run the same detector on a recorded video:

```sh
python -m app.main --config outputs/follow.yaml --video /path/to/clip.mp4
```

Replay always uses simulation, including when the copied configuration says
`mode: hardware`. It never opens a physical driver or requires a calibration file.
Motion, selection, and simulated control use frame number divided by source FPS.
When the source reports no valid FPS, replay uses `camera.fps` and logs that fallback.
Existing delivery timestamps in frame records retain their decoding-time meaning.

The tracker uses Kalman prediction and sparse background optical flow. Detections
above `model.confidence` can create tracks and count towards `confirmation.k` of
`confirmation.n` hits. Detections down to `follow.maintenance_confidence`, default
0.15, can maintain an existing identity. They cannot create or confirm a track.
Following requires `confirmation.enabled: true`. The shipped configuration already
enables it. A configuration that disables confirmation while enabling following is rejected.

The selector chooses the confirmed pigeon with the largest smoothed box area.
Another pigeon must remain at least 25% larger for 0.5 seconds to replace it.
An unmatched target keeps its identity for 0.75 seconds. Unmatched observations
stop corrections, and observations at least 250 milliseconds old also cause a hold.
When sensor timing is available, observation age includes sensor-to-delivery time.
The logs distinguish sensor, delivery, replay, and unavailable timing.

## Enable physical following after bring-up

First verify manual calibration on the assembled mount. Change `pan_tilt.mode` to
`hardware` and set `pan_tilt.calibration` to that mount's profile. Run the command
without `--video` to use the live camera.

The controller runs independently at 20 Hz. Defaults are a central deadband of
±5% per image dimension, maximum speed of 30 nominal degrees per second,
acceleration of 60 nominal degrees per second squared, and home speed of 10 nominal
degrees per second. Each command respects the marked travel limits. Tune the gain
and motion limits on hardware before field use.

`controller_start` records the resolved calibration and raw image orientation.
Frame records contain the selected target, controller state, limits, and observation
age. `target_acquired`, `target_switched`, and `target_lost` record identity changes.
The overlay labels the selected bird as "nearest-looking" and draws the deadband.
Box area provides a ranking, with no distance estimate.

In following mode, `pigeon_count` counts candidates at the maintenance floor.
Each detection has a `maintenance_only` flag, and the frame records both thresholds.
Low-confidence boxes appear grey with an association or maintenance label; the
selected target remains cyan. Detection-only counts and thresholds are unchanged.

Following with day/night camera switching is rejected before driver construction.
`--reaction-test` disables following. Hardware compatibility, centring, latency,
and a one-hour run remain unverified. The fixed-zone mode is deferred until the core
demonstration passes hardware acceptance; `zone.enabled: true` reports that explicitly.

## Watch the browser preview

In your copied configuration, enable the preview:

```yaml
preview:
  enabled: true
  host: 127.0.0.1
  port: 8080
  max_fps: 10
```

Start the application and open `http://127.0.0.1:8080` on the same computer. For
another LAN device, bind `host: 0.0.0.0` and open `http://<Pi address>:8080` there.
This exposes the stream and mount controls to clients on that network.

The page shows the processed image, nearest-looking target, controller state,
timing, limits, and recording status. Pause holds position. Home returns the mount
and keeps following paused until Resume. Controls are unavailable when following
is disabled or the connection is lost. The page labels simulation explicitly and
marks stale images. Preview also works with detection-only operation.

The service provides `/stream.mjpg`, `/snapshot.jpg`, and `/api/state`. The processed
image and its boxes are encoded together, once per selected frame, to a fixed
800×450 canvas by default. Slow clients skip to the newest frame. Short videos may
finish before you open the page because replay runs without real-time pacing.
Requests accept literal IPs, `localhost`, and the configured bind hostname. Use a
literal Pi address when binding all interfaces with `0.0.0.0`.

## Record a demonstration with elapsed timing

Install FFmpeg and ffprobe in the runtime environment. On the Pi:

```sh
sudo apt install ffmpeg python3-scipy
```

Enable `preview.record: true` in the copied configuration. It works with or without
the browser server. Set `preview.enabled: false` for recording-only operation.
After processing stops, the application exports `annotated.mp4` in the run directory.
The source JPEGs and `demo/index.jsonl` remain there, with matching detections,
target and controller snapshots, and presentation timestamps.

Live recordings use delivery-monotonic timestamps. Replay uses source-frame time.
Dropped samples keep their elapsed gaps. The final frame holds for one preview
period in live operation or one source-frame period in replay. FFmpeg appends a
duplicate final image to represent that hold. MP4 conversion happens after camera
control stops and does not count towards processing elapsed time or FPS.

Defaults stop spooling after 600 seconds or 512 MiB, and reserve 256 MiB of free
space. `preview.record_max_duration_s`, `record_max_bytes`, and `record_min_free_bytes`
set those limits. The writer has an eight-frame queue and counts samples dropped
when storage falls behind. `preview.export_timeout_s` defaults to 900 seconds.
Ctrl+C or SIGTERM during conversion cancels FFmpeg and retains the source frames.
The export uses x264's `veryfast` preset; actual Pi conversion time remains unmeasured.
Export has its own output byte bound, equal to `record_max_bytes`, in addition to
the retained spool. Both outputs therefore need disk space.

Storage faults are isolated from live detection and control. Export faults occur
after processing stops. Both retain the source frames and report the error.
To retry conversion afterward:

```sh
python tools/export_demo.py outputs/<run>/demo outputs/<run>/annotated.mp4 --timeout-s 900
```

For replay, also set `--frame-duration-s` to one divided by the source FPS, such as
`0.0666666667` for 15 FPS. `--record` and `output.record` continue to select the
separate fixed-FPS diagnostic AVI mode.

## Compare output overhead with replay

Use one representative clip for every mode:

```sh
python tools/benchmark_demo.py /path/to/clip.mp4 --frames 300 --output outputs/demo-benchmark
```

The command runs tracking alone, preview, recording, and both outputs. Each run
uses simulation and saves its configuration, logs, and summary. The comparison is
`outputs/demo-benchmark/benchmark.json`. `--backend ultralytics` selects the
Ultralytics adapter when the native extension is unavailable.

This comparison has no browser client and measures unpaced replay. Use the real
browser, live camera, and mount for LAN delivery, frame-to-command latency, centring,
and thermal acceptance. If output overhead affects control, reduce `preview.max_fps`
to 5, then 2, and repeat the measurements.

## Verify the software on a desktop

Activate the project environment, then run:

```sh
python tools/verify_demo.py
```

This runs the complete pytest, Ruff, formatting, and strict mypy checks. Driver
tests use a mock I²C bus. They do not establish physical compatibility or servo
behaviour. Software evidence and hardware measurements are recorded separately in
[the validation record](../deploy/VALIDATION.md).
