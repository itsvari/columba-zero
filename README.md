# Columba Zero

Pigeon detection and camera tracking prototype for **Raspberry Pi 5, 64-bit Raspberry Pi OS,
CSI camera, and a connected monitor**. It displays pigeon boxes, confidence,
current-frame count, capture FPS, and processing FPS. The Waveshare Pan-Tilt HAT
can move the camera to follow one pigeon. Live runs default to physical tracking
and require calibration; use `--tracking off` for detection alone. This version
does not count unique birds.

The runtime uses the existing `pigeon_baseline_v1-2` YOLO11n single-class model.
The original training notebook, results, and weights remain intact. The notebook
and `GEN2002Y Report.md` describe historical work; the current application is
specified here and in [the acceptance guide](RASPBERRY_PI_PROTOTYPE_PLAN.md).

## Model preparation

From the repository root:

```bash
python tools/prepare_models.py
```

This copies `best.pt` and the complete `best_ncnn_model` directory from
`runs/detect/pigeon_detector/pigeon_baseline_v1-2/weights/` into ignored
`models/pigeon_baseline_v1-2/`, checking SHA-256 against
[the tracked manifest](deploy/model_manifest.json). Original artifacts are never
modified. When transferring to the Pi, copy `models/pigeon_baseline_v1-2/` as
well as the application; ignored model binaries do not arrive via Git alone.
Keep original training artifacts backed up separately.

## Pi setup

Use Raspberry Pi OS Desktop **64-bit**, Python 3.11 or newer, with a local desktop
session and CSI camera enabled and connected. Run manually in a terminal on the
Pi monitor. The old hardware systemd unit has been removed; disable any previously
installed `columba-zero` service before running this application.

Install camera and GUI-capable OpenCV through the OS, then use the system Python:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 python3-opencv python3-numpy python3-yaml
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
mkdir -p outputs/setup
python -c "import numpy; print('numpy==' + numpy.__version__)" > outputs/setup/pi-constraints.txt
python -m pip install -c outputs/setup/pi-constraints.txt torch torchvision matplotlib requests psutil polars nvidia-ml-py 'ultralytics-thop>=2.2.2' 'ultralytics-platform>=0.1.45' 'ncnn==1.0.20260526'
python -m pip install --no-deps 'ultralytics==8.4.175'
python -m pip install --no-deps -e .
python -c "import cv2, picamera2, ncnn, ultralytics; print(cv2.__file__); print(cv2.getBuildInformation())"
python -m pip freeze > outputs/setup/installed.txt
```

This Pi-specific install intentionally retains OS NumPy, OpenCV, and libcamera
bindings. Ultralytics is installed without dependency resolution to avoid replacing
OS OpenCV. Its remaining runtime dependencies are explicitly installed above.
`pip check` may report the OS OpenCV distribution as missing or too old even when
`cv2` works: verify imports, GUI build information, and actual predictions.
Do not install `opencv-python-headless` or combine multiple pip OpenCV variants.
If OS NumPy is older than 1.24, use a newer supported Pi OS image. Resolve any
missing ARM wheels before proceeding; this setup has not yet been verified on a Pi.

The model was exported with Ultralytics 8.4.175; that version and the locally tested
NCNN version are pinned. Camera format `RGB888` provides BGR bytes for OpenCV
(despite its name). See the [official Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
for OS packages, virtual environments, and pixel formats, and the
[Ultralytics Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) for NCNN support.

## Run

Launch from the repository root; configuration paths are relative to the current
working directory. Default live settings: 640x480 capture, requested 15 FPS,
640x640 model input, confidence 0.5, recording off.

```bash
columba-zero --tracking off
columba-zero --tracking off --confidence 0.6 --record
columba-zero --video sample.mp4 --max-frames 300
columba-zero --model models/pigeon_baseline_v1-2/best.pt --video sample.mp4
```

Press **q**, close the window, or press **Ctrl+C** to stop. SIGTERM also requests
shutdown. Capture timeouts and inference failures exit nonzero after cleanup.
A termination request waits for the current capture/inference call to return;
there is no hard watchdog for a hung native backend.

`--config` selects a YAML file. [app/config.yaml](app/config.yaml) contains all
settings. CLI model/confidence values override YAML; `--record` enables recording.
Set `display.enabled: false` for headless replay or tests. The new `tracking`
section configures camera following; other old actuation sections and CLI
switches are rejected with migration guidance.

## Pan-tilt camera setup

The [Waveshare Pan-Tilt HAT](https://www.waveshare.com/wiki/Pan-Tilt_HAT) uses
a PCA9685 over I2C. Defaults for this mount are bus 1, address `0x40`, PWM 50 Hz,
pan on S0 (channel 0), and tilt on S1 (channel 1). Only these two channels are driven or
disabled; PWM frequency is shared by the chip, so reserve the HAT for this app.
The ambient light sensor is unused. The driver uses Linux I2C through `smbus2`,
without WiringPi or RPi.GPIO. Actual Pi 5 hardware acceptance remains pending.

Enable I2C using `sudo raspi-config` (Interface Options), then install into the
Pi virtual environment prepared above:

```bash
sudo apt install i2c-tools
python -m pip install 'smbus2>=0.4.3,<1'
# Alternatively: python -m pip install -e '.[hardware]'
i2cdetect -y 1
```

Confirm `40` appears and your user can access `/dev/i2c-1` (normally through the
`i2c` group). Stop other programs that control the mount. Follow
[Waveshare's assembly and power instructions](https://files.waveshare.com/upload/b/b9/Pan-Tilt_HAT_user_manual_en.pdf).
Center the servos before attaching their horns and mount the CSI camera with
enough cable slack throughout the intended travel. Calibration initially commands
both servos to 1500 microseconds; their physical orientation depends on assembly.
Power off before attaching horns or changing wiring.

### Calibrate the assembled mount

```bash
python -m app.calibrate
```

There is no automatic sweep. In an interactive terminal (including SSH), arrow
keys move immediately without Enter: left/right decrease/increase the pan pulse;
up/down decrease/increase the tilt pulse. Each keypress moves 10 microseconds;
holding a key repeats using your terminal's keyboard repeat. Physical direction
depends on how the servos are mounted. Typed `pan +`, `pan -`, `tilt +`, and
`tilt -` commands remain available. Type marking, direction, and save commands
and press Enter as before. Arrow keys are ignored while a typed command is unfinished.
Use `--line-input` for command-only input; redirected input also uses this mode.
Start with a narrow, clear travel range;
stop before binding, cable tension, or mechanical end stops. The initial pulse
envelope is 1000–2000 microseconds. Flags such as `--pan-pulse-min-us` and
`--pan-pulse-max-us` let you specify the installed servo's envelope within
500–2500 microseconds; this is a mapping range, not a measured mechanical limit.
`--bus`, `--address 0x40`, `--pan-channel`, and `--tilt-channel` override wiring.
The default pan/tilt channels were swapped to match this assembled mount.
Existing calibration files retain their saved channel assignments: recalibrate
and save a fresh profile after updating. For the original Waveshare channel
layout, use `--pan-channel 1 --tilt-channel 0` when calibrating.

For each axis:

1. Move to a lower **pulse** position within clear travel and enter `pan min`.
2. Move to the desired home position and enter `pan home`.
3. Move to an upper pulse position within clear travel and enter `pan max`.
4. With a fixed scene and detection disabled, check movement direction. Set
   `pan direction 1` if increasing the pulse turns the camera toward objects
   on the right of the image; otherwise use `pan direction -1`.
5. Repeat using `tilt`; direction 1 must move toward objects below image center.

`status` shows pulses and marks. `save` requires all four marks for both axes
and writes `calibration.local.yaml`; `quit`, Ctrl+C, or EOF leaves it unsaved.
Exiting disables the two PWM channels. The profile is ignored by Git; copy it
separately when deploying. Do not reuse another mount's travel limits. Angles
are logical 0–180-degree coordinates interpolated over each pulse envelope, not
measured camera angles. `tracking.calibration` selects the profile path.

### Run tracking

```bash
columba-zero                          # live CSI camera, physical tracking
columba-zero --tracking simulate      # live camera, virtual mount
columba-zero --tracking off           # detection only; no hardware dependency
columba-zero --video sample.mp4       # simulated tracking using video frame time
```

Physical tracking with `--video` is rejected. Simulation uses your local profile
if present, otherwise nominal home 90 and travel 60–120 on each axis. Physical
mode requires a complete profile and working I2C hardware. At startup it commands
the calibrated home position; the initial move cannot be speed limited because
these servos provide no position feedback. Later commands obey configured speed
and travel limits.

The largest pigeon is acquired first. Following frames match the closest center
within `tracking.match_distance` (default 0.25 in frame-width/height units), with
confidence breaking ties. This is approximate association; crossing birds or
camera movement can cause a switch. A missed match stops movement immediately.
After two seconds without a match the camera returns home; new acquisition
interrupts the return. Brief gaps retain the previous target for matching.

The controller updates at up to 20 Hz with a latest-observation slot, 5% frame
deadband on each side of center, gain 30 degrees/second at full normalized error,
and maximum speed 30 degrees/second. These values are configurable in YAML.
Observations older than one second cannot steer toward a target. The timeout
also applies to inference latency: increase it only after measuring Pi latency.
After two seconds without a fresh match the controller returns home even if
inference stalls. A prolonged worker scheduling delay does not produce a large
catch-up step. Simulation replay advances at virtual 20 Hz using frame sequence
and video FPS (falling back to configured camera FPS if unavailable), independent
of desktop inference speed. It does not model the visual effect of camera motion.

The preview and recorded frames show a center crosshair, the currently matched
target in magenta, state, and commanded pan/tilt angles. Frame JSONL records add
`tracking` with `mode`, `state`, `selected_target`, `commanded_pan_deg`, and
`commanded_tilt_deg`; it is `null` when disabled. Angles are commands without
position feedback. A worker I2C failure attempts to disable both channels and
is reported as a runtime error. Normal shutdown disables PWM and closes the bus;
the mount may relax when PWM is removed. Native camera/inference or I2C calls
cannot be forcibly interrupted, and a killed process cannot guarantee cleanup.

The supplied configuration sets `output.flip_vertical: true` for this camera mount.
It flips the preview and new annotated recordings top-to-bottom, with boxes aligned
to the flipped image and text drawn upright. Set it to `false` for the original
view. Detection, tracking, calibration directions, and JSONL box coordinates keep
using the original camera image. Previously recorded files are unchanged.

The NCNN export is fixed at 640x640. To benchmark a different inference size,
export a separate model with `tools/export_ncnn.py` and use a matching
`model.image_size`. Camera FPS is not inference FPS. Live capture discards old
buffered frames when inference falls behind; video replay processes every decoded
frame sequentially, without real-time pacing or frame dropping.

## Outputs

Each run creates `outputs/<UTC timestamp>-<unique suffix>/` with
`detections.jsonl` and optional `annotated.avi` (MJPEG). Frame records include
zero-based source frame number, Unix frame-delivery timestamp, inference seconds,
pigeon count, confidence and original-frame pixel boxes, cumulative live buffer
drops, and capture/processing FPS. Drops measure this application's buffer, not
camera driver losses. Replay timestamps reflect decoding time, not the recording's
original capture time. Health samples include RSS, CPU use, and Pi temperature
when available. `run_end` records completion or failure; a forcibly killed process
may have no final record.

Recording writes **only processed frames** at `output.recording_fps` (default 15).
It is diagnostic, not a timing-faithful camera recording. It may speed up or slow
down playback and adds processing/storage overhead. Files grow for the duration
of a run; monitor available disk space and stop recording when evidence is sufficient.

```bash
python tools/summarize_run.py outputs/<run>/detections.jsonl
```

## Desktop development and validation

```bash
python -m venv .venv
# Activate .venv using your shell's activation script.
python -m pip install -e '.[vision,metrics,dev]'
python -m unittest discover -s tests -v
python tools/prepare_models.py
python tools/compare_exports.py models/pigeon_baseline_v1-2/best.pt models/pigeon_baseline_v1-2/best_ncnn_model /path/to/test/images --output outputs/validation/export_comparison.json
python tools/evaluate_model.py models/pigeon_baseline_v1-2/best.pt /path/to/data.yaml --output outputs/validation/pt_test_metrics.json
```

For clean desktop environments use GUI-capable `opencv-python`. The Pi-specific
installation above deliberately uses OS OpenCV instead of the desktop extra.
The runtime CLI is also available as `python -m app.main`.

Export comparison uses matching CPU, 640-square preprocessing, confidence 0.5,
and NMS IoU 0.7. It records counts, boxes, matched IoUs, and confidence deltas;
acceptance defaults are IoU >=0.8, confidence delta <=0.05, and equal counts.
Test-set evaluation uses confidence floor 0.001 for AP curves; reported precision
and recall are Ultralytics' curve-selected metrics, not necessarily the deployment
threshold 0.5. Review continuous negative and positive footage before tuning that
threshold. Dataset metrics and desktop timing cannot establish Pi performance.

[Local validation results](deploy/VALIDATION.md) separate completed software/model
checks from pending Pi camera and one-hour endurance acceptance.
