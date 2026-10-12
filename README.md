# Columba Zero

Detection-only pigeon prototype for **Raspberry Pi 5, 64-bit Raspberry Pi OS,
CSI camera, and a connected monitor**. It displays pigeon boxes, confidence,
current-frame count, capture FPS, and processing FPS. There is no GPIO or
actuation code, calibration, tracking, or unique-bird counting.

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

Git checks out the NCNN `metadata.yaml` with LF line endings on every platform.
Its manifest entry hashes the tracked 427-byte file. The other four manifest
files retain their exact bytes during checkout. Preparation verifies both file
size and SHA-256 before copying and again after copying.

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
columba-zero
columba-zero --confidence 0.6 --record
columba-zero --video sample.mp4 --max-frames 300
columba-zero --model models/pigeon_baseline_v1-2/best.pt --video sample.mp4
```

Press **q**, close the window, or press **Ctrl+C** to stop. SIGTERM also requests
shutdown. Capture timeouts and inference failures exit nonzero after cleanup.
A termination request waits for the current capture/inference call to return;
there is no hard watchdog for a hung native backend.

`--config` selects a YAML file. [app/config.yaml](app/config.yaml) contains all
settings. CLI model/confidence values override YAML; `--record` enables recording.
Set `display.enabled: false` for headless replay or tests. Old actuation config
sections and CLI switches are rejected with migration guidance.

The NCNN export is fixed at 640x640. To benchmark a different inference size,
export a separate model with `tools/export_ncnn.py` and use a matching
`model.image_size`. Camera FPS is not inference FPS. Live capture discards old
buffered frames when inference falls behind; video replay processes every decoded
frame sequentially, without real-time pacing or frame dropping.

Continuous autofocus is enabled only when the camera advertises `AfMode`.
Fixed-focus cameras retain the frame-rate setting without an autofocus request.

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
python -m ruff check app tests tools
python -m ruff format --check app tests tools
python -m mypy
python tools/prepare_models.py
python tools/compare_exports.py models/pigeon_baseline_v1-2/best.pt models/pigeon_baseline_v1-2/best_ncnn_model /path/to/test/images --output outputs/validation/export_comparison.json
python tools/evaluate_model.py models/pigeon_baseline_v1-2/best.pt /path/to/data.yaml --output outputs/validation/pt_test_metrics.json
```

For clean desktop environments use GUI-capable `opencv-python`. The Pi-specific
installation above deliberately uses OS OpenCV instead of the desktop extra.
The runtime CLI is also available as `python -m app.main`.

Strict type checking covers application code, tests, and tools. Ultralytics and
the Pi camera bindings lack complete type information; the detector adapter uses
explicit prediction protocols, and camera imports remain deferred until use.

Export comparison uses matching CPU, 640-square preprocessing, confidence 0.5,
and NMS IoU 0.7. It records counts, boxes, matched IoUs, and confidence deltas;
acceptance defaults are IoU >=0.8, confidence delta <=0.05, and equal counts.
Test-set evaluation uses confidence floor 0.001 for AP curves; reported precision
and recall are Ultralytics' curve-selected metrics, not necessarily the deployment
threshold 0.5. Review continuous negative and positive footage before tuning that
threshold. Dataset metrics and desktop timing cannot establish Pi performance.

[Local validation results](deploy/VALIDATION.md) separate completed software/model
checks from pending Pi camera and one-hour endurance acceptance.
