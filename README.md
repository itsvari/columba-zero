# Columba Zero

Pigeon detection and opt-in camera following for **Raspberry Pi 5, 64-bit Raspberry
Pi OS, and a CSI camera**. Detection-only operation remains the default. The local
view displays boxes, confidence, capture FPS, processing FPS, and confirmed tracks.
Following adds shared motion identities, target selection, and a calibrated I²C
pan-tilt controller. Physical operation and Pi performance await hardware acceptance.

The shipped model is `yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`, a
single-class YOLO26n NCNN package: 448×800, fp16, yolo11 decode, threshold
0.62646484375. It runs on the native C++ backend with k2/n3 confirmation and an
800×450 camera stream at 30 FPS ([WS7 report](experiments/reports/WS7-final.md)).
The earlier `pigeon_baseline_v1-2` YOLO11n 640 model stays as the comparison
baseline and fallback.

Further documents:

- [Pan-tilt calibration, following, and desktop verification](docs/pan-tilt.md).
- [Pi setup over SSH and pan-tilt hardware testing](docs/pi-hardware-testing.md).
- [Pi acceptance checklist](RASPBERRY_PI_PROTOTYPE_PLAN.md) and
  [validation results](deploy/VALIDATION.md).
- [Model-improvement plan](plans/model-improvement/PLAN.md) (closed except WS10,
  Pi hardware acceptance) and its [contracts](plans/model-improvement/CONTRACTS.md).
- [Experiment reports](experiments/README.md), the [native backend](native/README.md)
  and [cluster tooling](cluster/README.md).
- History: the original [training notebook](docs/history/pigeon_detection_training.ipynb)
  and [GEN2002Y report](docs/history/GEN2002Y%20Report.md).

## Model preparation

From the repository root:

```bash
python tools/prepare_models.py
```

This verifies and copies both packages listed in
[the tracked manifest](deploy/model_manifest.json) into ignored `models/`:

- the shipped package from
  `runs/detect/pigeon_detector/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16/weights/best_ncnn_model/`
  to `models/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16/`;
- the fallback `best.pt` and `best_ncnn_model/` from
  `runs/detect/pigeon_detector/pigeon_baseline_v1-2/weights/` to
  `models/pigeon_baseline_v1-2/`.

`--model-id <id>` prepares one entry only. Preparation checks file size and
SHA-256 before and after copying; tracked originals are never modified. When
transferring to the Pi, copy `models/` as well as the application, since
ignored model files do not arrive via Git alone. Git checks out each NCNN
`metadata.yaml` with LF line endings and the other package files with their
exact bytes (`.gitattributes`), so the manifest hashes hold on every platform.

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

PyTorch, Ultralytics and pip `ncnn` serve the Ultralytics backend, `.pt`
checkpoints and the tools; the native backend does not import them. This install
keeps OS NumPy, OpenCV and libcamera bindings; Ultralytics is installed without
dependency resolution so it cannot replace OS OpenCV. `pip check` may report OS
OpenCV as missing or too old even when `cv2` works: verify imports, GUI build
information and predictions. Do not install `opencv-python-headless` or combine
pip OpenCV variants. If OS NumPy is older than 1.24, use a newer Pi OS image.
This Ultralytics dependency set has not yet been verified on a Pi. The native-only
runtime was built and exercised without Torch or Ultralytics. See
[the Pi validation record](deploy/VALIDATION.md#pi-bring-up-27-october-2026-in-progress).

Ultralytics 8.4.175 and ncnn 1.0.20260526 are pinned; the models were exported
with them. Camera format `RGB888` provides BGR bytes for OpenCV (despite its
name). See the [official Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
for OS packages, virtual environments, and pixel formats.

## Native backend

The shipped configuration uses `model.backend: native`, the `columba_native`
C++ extension (NCNN tag `20260526`, OpenCV core/imgproc, nanobind). The app
exits with an error if the module cannot be imported. On the Pi, inside the
venv above:

```bash
sudo apt install libopencv-dev
python -m pip install nanobind 'cmake>=3.26'
bash tools/build_ncnn_a76.sh --profile pi5 --jobs 4
cmake -S native -B native/build/pi5 -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(command -v python)" \
    -Dncnn_DIR="$PWD/native/build/ncnn-pi5/lib/cmake/ncnn"
cmake --build native/build/pi5 --parallel 4
ctest --test-dir native/build/pi5 --output-on-failure
cmake --install native/build/pi5 --prefix "$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
```

The Pi 5 build, CTest, and native Python tests pass on the assembled device.
Live capture and inference run, with throughput acceptance still pending.
[native/README.md](native/README.md)
covers the macOS and cluster builds, the Python API, decoding and parity
tools. To use the Ultralytics backend instead, set `model.backend: ultralytics`;
it only supports yolo11-decode packages and applies its own NMS.

## Run

Launch from the repository root; configuration paths are relative to the current
working directory. [app/config.yaml](app/config.yaml) runs the shipped model:
800×450 lores capture at 30 FPS, 448×800 native inference, threshold
0.62646484375, confirmation k2/n3, recording off.

```bash
columba-zero
columba-zero --record
columba-zero --video sample.mp4 --max-frames 300
columba-zero --config outputs/baseline.yaml
```

Press **q**, close the window, or press **Ctrl+C** to stop. SIGTERM also requests
shutdown. Capture timeouts and inference failures exit nonzero after cleanup.
A termination request waits for the current capture/inference call to return;
there is no hard watchdog for a hung native backend.

`--config` selects a YAML file. `--model` and `--confidence` override the YAML
values and `--record` enables recording. Set `display.enabled: false` for
headless replay or tests. Old actuation config sections (including `tracking:`)
and CLI switches are rejected with migration guidance.

To run the fallback, copy `app/config.yaml` (for example to `outputs/baseline.yaml`)
and set `model.path: models/pigeon_baseline_v1-2/best_ncnn_model`,
`model.image_size: 640` and `model.confidence: 0.5`. An NCNN export has a fixed
input size, so `model.image_size` must equal the export's `imgsz`; export another
size with `tools/export_ncnn.py`. Camera FPS is not inference FPS. Live capture
discards old buffered frames when inference falls behind; video replay processes
every decoded frame sequentially, without real-time pacing or frame dropping.

Capture, inference and confirmation run on a processing thread. The main thread
draws the window and records; a separate log-writer thread writes the JSONL log,
so a stalled display never delays logging and the processing loop never waits
for either. When drawing falls behind, the oldest waiting frame is dropped from
the display and recording (`render_dropped_frames` in `health` and `run_end`).
Replay waits for the display instead, so replay recordings keep every frame. If
the log writer falls more than 4096 records behind, the oldest queued `frame` or
`controller_state` sample is dropped (or the incoming sample, when only lifecycle
events are queued) and
counted as `log_dropped_records`. `trigger`, `camera_switch`, `health` and run
records are never dropped; they may exceed the limit by 1024. Beyond that the
run stops with an error, because triggers could no longer be recorded; records
already accepted are still written. Every queued record is written before
`run_end`, also after a display or recording failure. Shutdown waits at most 2 s
for the log writer; if it is stuck inside a write, the run reports "log writer
stalled; N records unwritten" and the `run_end` summary on stderr instead of the
log, still closes the camera and recording, and exits nonzero.

### Configuration reference

The tables give the values in `app/config.yaml`. An omitted key takes the schema
default, which keeps the behaviour of earlier versions: 640×480 at 15 FPS on the
`main` stream, `models/pigeon_baseline_v1-2/best_ncnn_model` at 640 with
confidence 0.5 on the `ultralytics` backend, and confirmation off. Other
defaults equal the values shown.

`camera:`

| Key | config.yaml | Meaning |
|---|---|---|
| `width`, `height` | 800, 450 | Frame size delivered to the model. |
| `fps` | 30 | Requested camera frame rate. |
| `frame_timeout_s` | 2.0 | Capture timeout; exceeding it stops the run with an error. |
| `stream` | `lores` | `lores` makes the camera hardware scale the full field of view down to `width` x `height`. Where the low-resolution stream cannot be RGB (Pi 4 and earlier), it is captured as YUV420 and converted. |
| `main_size` | null | With `lores`: size of the main stream that sets the field of view. Null picks half the sensor resolution at the frame's aspect ratio (a 2x2 binned mode). |
| `lens_position` | null | Manual focus in dioptres (0 = infinity), clamped to the camera's range. Without `LensPosition` support the camera keeps its default (continuous autofocus when `AfMode` exists, else fixed focus) and a warning is printed. |
| `exposure_compensation_ev` | null | CSI auto-exposure bias in stops (EV), e.g. -1.5 darkens the image. null preserves the camera default; explicit 0 requests neutral compensation. Unsupported or out-of-range values fail when that camera starts; a failed day/night switch falls back to the previous camera and records an event. Values are never clamped. Replay uses the recorded exposure and ignores this setting. |
| `max_exposure_us` | null | Caps automatic exposure (e.g. 16666 for 1/60 s) by clamping the AGC exposure modes of the sensor's tuning, as described in the Picamera2 manual. |
| `tuning_file` | null | Camera tuning by name (e.g. `imx708_noir.json` for the NoIR camera) or path. If it cannot be loaded, the default tuning is used with a warning. |
| `camera_index` | 0 | Which CSI camera to open. |
| `name` | null | Camera name in frame records; null means `camera<index>`. |
| `rotation_deg` | 0 | Live capture rotation, either 0 or 180 degrees, applied before detection, tracking, and preview. Hardware calibration must record the same orientation. Replay uses the video frames as stored. |

`model:`

| Key | config.yaml | Meaning |
|---|---|---|
| `path` | `models/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16` | `.pt` checkpoint or NCNN package directory. |
| `image_size` | `[448, 800]` | An integer for a square input, or `[height, width]`, both multiples of 32. Must equal the export's `imgsz`. |
| `confidence` | 0.62646484375 | Per-frame detection threshold (the shipped model's val-selected operating threshold). |
| `backend` | `native` | `native` uses the `columba_native` C++ detector (`app/native_detector.py`); `ultralytics` uses Ultralytics `YOLO.predict`. |
| `threads` | 4 | CPU threads for the native backend. |

Class names and trigger classes come from the package's `metadata.yaml`
(`names`, and `trigger_classes` in the `columba:` block, default `[pigeon]`).
`pigeon` must be class 0, and `pigeon` is the only class the runtime lets
trigger (CONTRACTS.md C2): a package listing any other trigger class is rejected.
Packages without a `columba:` block, such as `pigeon_baseline_v1-2`, are read as
a YOLO11 export with trigger class `pigeon`. Other classes are drawn in orange
and logged, but never trigger. The Ultralytics backend always applies NMS, so
packages with `columba.decode: yolo26` (NMS-free) need `model.backend: native`.

`confirmation:`

| Key | config.yaml | Meaning |
|---|---|---|
| `enabled` | true | Track detections and log `trigger` events. |
| `k`, `n` | 2, 3 | A track confirms when k of its last n frames had a trigger-class detection. |
| `iou` | 0.3 | Minimum box overlap to continue a track (greedy, highest overlap first, class ignored). |
| `max_age_frames` | 2 | A track ends after this many consecutive frames without a match. |

Each track confirms at most once. Confirmed tracks are drawn in red. `output:`
sets `directory` [`outputs`], `record` [false], `recording_fps` [15] and
`health_interval_s` [30]; `display.enabled` defaults to true.

`follow.enabled: true` uses motion tracking and confirmed target selection, with
low-confidence detections retained only for association. `pan_tilt.mode` defaults
to `simulate`; physical following requires `hardware` and a local calibration
profile. Replay always simulates. See the [following guide](docs/pan-tilt.md).
Following with day/night camera switching is rejected until mounted-camera
calibration and orientation are defined. Reaction tests disable following.

`cameras:` and `camera_switch:` (off unless `cameras` is given)

List exactly one `role: day` and one `role: night` camera, each with a `name`
and optional `camera_index`, `tuning_file`, `lens_position`,
`max_exposure_us`, and `exposure_compensation_ev`; other settings come from `camera:`. Only one camera is open
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
when OpenCV finishes drawing, so monitor latency makes the results slight
overestimates. No image ships with the repository; use a pigeon photo whose
licence you have checked.

## Outputs

Each run creates `outputs/<UTC timestamp>-<unique suffix>/` with
`detections.jsonl` and optional `annotated.avi` (MJPEG). Frame records include
zero-based source frame number, Unix frame-delivery timestamp, inference seconds,
pigeon count, detections (label, confidence, original-frame pixel box,
`class_id`, and `track_id` when confirmation is on), cumulative live buffer
drops, and capture/processing FPS. Timing fields: `stage_ms` (`wait` from frame
delivery to pickup, `preprocess`, `inference` and `postprocess` from the backend,
`track`; a backend without stage timings reports its whole detect call as
`inference`), `delivery_ms` (start of sensor exposure to delivery, live cameras
only), `decided_at` and `decision_ms` (delivery to decision). Frames also record
`camera` and its `lux` estimate. `trigger` records carry the track id, class,
first-seen frame and time, confirmation time, frames seen, and k/n. Drops measure
this application's buffer, not camera driver losses. Replay timestamps reflect
decoding time, not the recording's original capture time. Health samples include
RSS, CPU use, and Pi temperature when available. `run_end` records completion or
failure; a forcibly killed process may have no final record.

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
python -m pip install -e '.[vision,follow,metrics,eval,data,native,dev]'
python -m pytest
python -m ruff check app tests tools cluster training native
python -m ruff format --check app tests tools cluster training native
python -m mypy
python tools/prepare_models.py
```

`python -m unittest discover -s tests -v` runs the same suite. Tests that need
`columba_native` skip unless it is built and on `PYTHONPATH`; C++ tests run with
`ctest`. Strict mypy covers all six source directories. Ultralytics and the Pi
camera bindings lack complete type information, so the detector adapter uses
explicit prediction protocols and camera imports are deferred until use. Desktop
environments use GUI-capable `opencv-python`; the Pi install deliberately uses
OS OpenCV. The runtime CLI is also available as `python -m app.main`.

Model tools for square Ultralytics exports, such as the baseline:

```bash
python tools/compare_exports.py models/pigeon_baseline_v1-2/best.pt models/pigeon_baseline_v1-2/best_ncnn_model /path/to/test/images --output outputs/validation/export_comparison.json
python tools/evaluate_model.py models/pigeon_baseline_v1-2/best.pt /path/to/data.yaml --output outputs/validation/pt_test_metrics.json
python tools/benchmark_detector.py models/pigeon_baseline_v1-2/best_ncnn_model sample.mp4 --sizes 640
```

Export comparison uses CPU, square preprocessing (`--imgsz`, default 640),
confidence 0.5 and NMS IoU 0.7. It accepts IoU >=0.8, confidence delta <=0.05
and equal counts, with one-to-one same-class matching so detection order cannot
change the result. Evaluation uses confidence floor 0.001 for AP curves; its
precision and recall are curve-selected, not at the deployment threshold. The
benchmark warms up, then reopens the video to time every frame; empty videos and
decoder failures exit nonzero. Rectangular fp16 packages such as the shipped
model are scored with `tools/select_model.py` and `native.verify_native`
([WS7 methods](experiments/reports/WS7-methods.md)). Dataset metrics and desktop
timing cannot establish Pi performance.
