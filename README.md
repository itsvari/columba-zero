# Columba Zero

Detection-only pigeon prototype for **Raspberry Pi 5, 64-bit Raspberry Pi OS,
CSI camera, and a connected monitor**. It displays pigeon boxes, confidence,
current-frame count, capture FPS, and processing FPS. There is no GPIO or
actuation code, calibration, or unique-bird counting. Per-track trigger
confirmation is available but off by default.

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
sections (including `tracking:`) and CLI switches are rejected with migration
guidance.

An NCNN export has a fixed input size. To benchmark a different inference size,
export a separate model with `tools/export_ncnn.py` and use a matching
`model.image_size`. Camera FPS is not inference FPS. Live capture discards old
buffered frames when inference falls behind; video replay processes every decoded
frame sequentially, without real-time pacing or frame dropping.

Capture, inference and confirmation run on a processing thread. The main thread
draws the window and records; a separate log-writer thread writes the JSONL log,
so a stalled display never delays logging and the processing loop never waits
for either. When drawing falls behind, the oldest waiting frame is dropped from
the display and recording (`render_dropped_frames` in `health` and `run_end`).
Replay waits for the display instead, so replay recordings keep every frame. If
the log writer falls more than 4096 records behind, the oldest queued `frame`
record is dropped (or the incoming one, when only other events are queued) and
counted as `log_dropped_records`. `trigger`, `camera_switch`, `health` and run
records are never dropped; they may exceed the limit by 1024. Beyond that the
run stops with an error, because triggers could no longer be recorded; records
already accepted are still written. Every queued record is written before
`run_end`, also after a display or recording failure.

### Configuration reference

Defaults (in brackets) keep the behaviour of earlier versions; every new feature
is off until configured.

`camera:`

| Key | Default | Meaning |
|---|---|---|
| `width`, `height` | 640, 480 | Frame size delivered to the model. 16:9 sizes such as 640x360 suit rectangular models. |
| `fps` | 15 | Requested camera frame rate. |
| `frame_timeout_s` | 2.0 | Capture timeout; exceeding it stops the run with an error. |
| `stream` | `main` | `lores` makes the camera hardware scale the full field of view down to `width` x `height`. Where the low-resolution stream cannot be RGB (Pi 4 and earlier), it is captured as YUV420 and converted. |
| `main_size` | null | With `lores`: size of the main stream that sets the field of view. Null picks half the sensor resolution at the frame's aspect ratio (a 2x2 binned mode). |
| `lens_position` | null | Manual focus in dioptres (0 = infinity), clamped to the camera's range. Without `LensPosition` support the camera keeps its default (continuous autofocus when `AfMode` exists, else fixed focus) and a warning is printed. |
| `max_exposure_us` | null | Caps automatic exposure (e.g. 16666 for 1/60 s) by clamping the AGC exposure modes of the sensor's tuning, as described in the Picamera2 manual. |
| `tuning_file` | null | Camera tuning by name (e.g. `imx708_noir.json` for the NoIR camera) or path. If it cannot be loaded, the default tuning is used with a warning. |
| `camera_index` | 0 | Which CSI camera to open. |
| `name` | null | Camera name in frame records; null means `camera<index>`. |

`model:`

| Key | Default | Meaning |
|---|---|---|
| `path` | `models/pigeon_baseline_v1-2/best_ncnn_model` | `.pt` checkpoint or NCNN package directory. |
| `image_size` | 640 | An integer for a square input, or `[height, width]` (e.g. `[384, 640]`), both multiples of 32. Must equal the export's `imgsz`. |
| `confidence` | 0.5 | Per-frame detection threshold. |
| `backend` | `ultralytics` | `native` uses the `columba_native` C++ detector (`app/native_detector.py`); it fails with an error if that module is not built. |
| `threads` | 4 | CPU threads for the native backend. |

Class names and trigger classes come from the package's `metadata.yaml`
(`names`, and `trigger_classes` in the `columba:` block, default `[pigeon]`).
`pigeon` must be class 0, and `pigeon` is the only class the runtime lets
trigger (CONTRACTS.md C2): a package listing any other trigger class is rejected.
Packages without a `columba:` block, such as `pigeon_baseline_v1-2`, are read as
a YOLO11 export with trigger class `pigeon`. Other classes are drawn in orange
and logged, but never trigger. The Ultralytics backend always applies NMS, so
packages with `columba.decode: yolo26` (NMS-free) need `model.backend: native`.

`confirmation:` (off by default)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | Track detections and log `trigger` events. |
| `k`, `n` | 2, 3 | A track confirms when k of its last n frames had a trigger-class detection. |
| `iou` | 0.3 | Minimum box overlap to continue a track (greedy, highest overlap first, class ignored). |
| `max_age_frames` | 2 | A track ends after this many consecutive frames without a match. |

Each track confirms at most once. Confirmed tracks are drawn in red.

`cameras:` and `camera_switch:` (off unless `cameras` is given)

List exactly one `role: day` and one `role: night` camera, each with a `name`
and optional `camera_index`, `tuning_file`, `lens_position` and
`max_exposure_us`; other settings come from `camera:`. Only one camera is open
at a time, starting with the day camera. The run switches to the night camera
when the active camera's lux estimate stays below `camera_switch.night_below_lux`
[10] for `hold_s` [5] seconds, and back when it stays above `day_above_lux` [40].
Each switch is logged as `camera_switch`. If the other camera fails to start,
the run reopens the working camera and retries after 60 s. The NoIR camera reads
a higher lux under an infrared illuminator than the scene appears to have, so
calibrate both thresholds on site.

### Reaction-time self-test

```bash
columba-zero --reaction-test /path/to/pigeon.jpg --reaction-trials 10
```

Point the camera at the monitor. The test shows a blank screen for 5 s, then
alternately shows the image fullscreen for 3 s and blanks it for 3–6 s (random),
logging each change as a `stimulus` event. Confirmation is enabled for the run
(with the configured k and n) and the annotated window stays closed. At the end
it writes `reaction_test.json` and a `reaction_test` log record: per-trial time
from display to confirmed trigger and to first pigeon detection, misses, and
triggers outside trials. Results are matched by timestamps after the run, so the
order in which records reach the log does not matter. Display times are taken
when OpenCV finishes drawing, so
monitor latency makes the results slight overestimates. No image ships with the
repository; use a pigeon photo whose licence you have checked.

## Outputs

Each run creates `outputs/<UTC timestamp>-<unique suffix>/` with
`detections.jsonl` and optional `annotated.avi` (MJPEG). Frame records include
zero-based source frame number, Unix frame-delivery timestamp, inference seconds,
pigeon count, detections (label, confidence, original-frame pixel box,
`class_id`, and `track_id` when confirmation is on), cumulative live buffer
drops, and capture/processing FPS. Timing fields: `stage_ms` (`wait` from frame
delivery to pickup, `preprocess`, `inference` and `postprocess` from
Ultralytics' per-result timings, `track`; a backend without stage timings reports
its whole detect call as `inference`), `delivery_ms` (start of sensor exposure
to delivery, live cameras only), `decided_at` and `decision_ms` (delivery to
decision). Frames also record `camera` and its `lux` estimate. `trigger`
records carry the track id, class, first-seen frame and time, confirmation time,
frames seen, and k/n. Drops measure this application's buffer, not
camera driver losses. Replay timestamps reflect decoding time, not the recording's
original capture time. Health samples include RSS, CPU use, and Pi temperature
when available. `run_end` records completion or failure; a forcibly killed process
may have no final record.

Recording and display cleanup finishes before `run_end` is written. Cleanup
failures appear in that record and cause a nonzero exit.

Recording writes **only processed frames** at `output.recording_fps` (default 15).
It is diagnostic, not a timing-faithful camera recording. It may speed up or slow
down playback and adds processing/storage overhead. Files grow for the duration
of a run; monitor available disk space and stop recording when evidence is sufficient.

```bash
python tools/summarize_run.py outputs/<run>/detections.jsonl
```

The summary adds count, mean, p50, p95, p99 and maximum for each stage, frame
delivery, decision time and trigger reaction time (first frame of a track to its
confirmation), in 0.1 ms histogram bins, plus trigger, camera-switch and
per-camera frame counts. Values of 10 s or more are kept exactly (up to 1000 of
them); past that, affected percentiles are reported as 10000 ms and listed under
`lower_bound`.

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
Matching finds the maximum number of one-to-one, same-class assignments that meet
both tolerances, so detection ordering cannot change acceptance.
Test-set evaluation uses confidence floor 0.001 for AP curves; reported precision
and recall are Ultralytics' curve-selected metrics, not necessarily the deployment
threshold 0.5. Review continuous negative and positive footage before tuning that
threshold. Dataset metrics and desktop timing cannot establish Pi performance.

`tools/benchmark_detector.py` warms up each input size on the first decoded frame,
then reopens the video to measure every frame from the beginning. The requested
frame limit must be positive. Empty videos and decoder failures before the
declared end cause a nonzero exit instead of a successful benchmark report.

[Local validation results](deploy/VALIDATION.md) separate completed software/model
checks from pending Pi camera and one-hour endurance acceptance.
