# Native NCNN detector

`columba_native` runs detection without importing PyTorch or Ultralytics. The
extension uses NCNN tag `20260526`, nanobind and OpenCV core/imgproc. Python needs
NumPy and PyYAML. The application wrapper imports the extension only when a
`NativeDetector` is constructed.

## Build on macOS ARM64

Run from the repository root. The development helper clones NCNN and OpenCV
`5.0.0` into the ignored `native/build/sources/` directory and installs static
libraries into separate prefixes. Homebrew `libomp` supplies OpenMP on this Mac.
The architecture flags also handle an x86_64 CMake binary running under Rosetta.

```sh
native_python=/Users/vari/Developer/columba-zero/.venv/bin/python
BUILD_THREADS=4 bash native/build_dependencies.sh mac-arm64
cmake -S native -B native/build/mac-arm64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 \
    -DPython_EXECUTABLE="$native_python" \
    -Dncnn_DIR="$PWD/native/build/ncnn-mac-arm64/lib/cmake/ncnn" \
    -DOpenCV_DIR="$PWD/native/build/opencv-mac-arm64/lib/cmake/opencv5"
cmake --build native/build/mac-arm64 --parallel 4
ctest --test-dir native/build/mac-arm64 --output-on-failure
export PYTHONPATH="$PWD/native/build/mac-arm64:$PWD"
"$native_python" -m unittest discover -s tests -v
```

For an installed module, pass Python's site-packages directory as the install
prefix. Installation includes the extension and its type stub.

```sh
site_packages_directory="$($native_python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
cmake --install native/build/mac-arm64 --prefix "$site_packages_directory"
```

No top-level `pyproject.toml` build-system change is required for this plain
CMake build. The existing `native` extra supplies nanobind and build tools.

## Build on Linux aarch64

Use Python 3.12.3 on the arm lane. Create the verification environment under
`~/columba/scratch/ws8/venv`, outside WS3's environments. The Slurm runner
`native/run_arm_job.sh` bundles dependency installation, build, C++ and Python
tests, parity, and overhead measurement. It requests eight CPUs on `xcng1`,
`normal`, a two-hour limit and `--nice=1000`. Check `squeue -u $USER` immediately
before submission and respect the project caps and WS8's single CPU-job share.
Stage the repository at `~/columba/scratch/ws8/repo` and models/images at
`~/columba/scratch/ws8/artifacts`; logs and results stay under the same root.

The direct build commands, inside the allocated arm job, are:

```sh
native_python="$HOME/columba/scratch/ws8/venv/bin/python"
BUILD_THREADS=8 bash native/build_dependencies.sh linux-aarch64
cmake -S native -B native/build/linux-aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$native_python" \
    -Dncnn_DIR="$PWD/native/build/ncnn-linux-aarch64/lib/cmake/ncnn" \
    -DOpenCV_DIR="$PWD/native/build/opencv-linux-aarch64/lib/cmake/opencv5"
cmake --build native/build/linux-aarch64 --parallel 8
ctest --test-dir native/build/linux-aarch64 --output-on-failure
export PYTHONPATH="$PWD/native/build/linux-aarch64:$PWD"
"$native_python" -m unittest discover -s tests -v
```

A WS4 prefix can replace the development NCNN prefix using
`-Dncnn_DIR=<prefix>/lib/cmake/ncnn`; no NCNN source checkout is required by
`native/CMakeLists.txt`. Vulkan is disabled. OpenCV may likewise come from an
existing installation via `OpenCV_DIR`.

## Python interfaces

```python
from columba_native import Detector

detector = Detector(model_dir, threads=4)
rows = detector.detect(frame, conf=0.5)
```

`frame` must be a non-empty C-contiguous NumPy `uint8` H×W×3 BGR array. The
binding borrows the existing buffer and rejects dtype/layout conversions.
Read-only inputs work. Do not modify the buffer from another thread during a
call. The GIL is released during preprocessing, inference and decoding.

`rows` is an owned, C-contiguous `float32` N×6 array. Columns are
`x1, y1, x2, y2, score, class_id` in original-frame pixels; an empty result has
shape `(0, 6)`. Properties are `input_hw`, `names` and `decode`.

`detect_with_timings(frame, conf)` additionally returns a dictionary with
`preprocess`, `extract`, `postprocess`, `total`, all in milliseconds.
`extract` times only NCNN's `Extractor::extract("out0", ...)`; extractor creation
and input submission count towards preprocessing. Output-array construction
occurs after the C++ total timer, so the benchmark uses the full Python-call
wall time when calculating overhead.

The application API is:

```python
from app.native_detector import NativeDetector

detector = NativeDetector(model_path, image_size, confidence, threads=4)
detector.warmup(width, height)
detections = detector.detect(packet)
```

`image_size` is an integer or `(height, width)` tuple. `names` is a class-index
mapping. `last_stage_ms` is updated on every successful call with `preprocess`,
`inference`, `postprocess`; `inference` is the C++ extract time. Validation and
inference errors are `DetectorError`. The wrapper currently matches main's
single-pigeon validation and confidence filtering; labels use lower-case
metadata names. After WS9 merges, add `class_id=class_id` to the `Detection(...)`
call. `_last_class_ids` already retains the IDs alongside the returned list.
The factory call specified by WS9 requires no API change.

## Metadata and decoding

The constructor reads a C8 package directly, with no directory-name suffix
requirement. NCNN blobs are `in0` and `out0`; output is unpacked float32
`(4 + nc, N)`, with cx/cy/w/h in letterboxed pixels and sigmoid class scores.
Names must have contiguous indices starting at zero.

* `decode: yolo11` selects the best class per anchor, filters with `score > conf`,
  applies per-class NMS or agnostic NMS, then keeps at most `max_det` boxes.
  NMS defaults to IoU 0.7 and `max_det` defaults to 300. Standard YOLO26 exports
  use this decoder too.
* `decode: yolo26` reads the head-swapped one-to-one export and takes top-k
  scores without NMS. Class-aware top-k can select multiple classes for an
  anchor, matching Ultralytics 8.4.175. Agnostic top-k uses the best class only.
* Legacy packages without `columba` must have `imgsz: [640, 640]`. They use
  `yolo11` and metadata names.

Preprocessing uses centred OpenCV `INTER_LINEAR` letterboxing, Python-compatible
rounding and pad value 114, followed by BGR→RGB and `/255`. C8 `color_order: bgr`
is also supported. Normalisation and output-layout variants outside the C8
contract are rejected. Boxes are clipped after NMS. The installed 8.4.175
`scale_boxes` uses separate gains from rounded resized width and height; the
native decoder follows that source, including odd padding.

C8 `quantization: fp32` and legacy packages use FP32 activations. `fp16` permits
half storage/packing with FP32 arithmetic; `int8` leaves NCNN's int8 kernels
enabled. FP16 arithmetic is disabled because it changed a one-to-one score
from 0.286 to 0.226 on a real COCO8 image during verification. Real-model parity
in this workstream covers FP32 packages; int8 needs WS7's separate parity gate.

## Repeatable verification

Set `YOLO_CONFIG_DIR` to scratch and create its `Ultralytics/settings.json` with
`sync: false` and scratch datasets, weights and runs directories before using
these tools. Model preparation never changes the source checkpoints:

```sh
"$native_python" -m native.prepare_models \
    --yolo11 /absolute/path/to/yolo11n.pt \
    --yolo26 /absolute/path/to/yolo26n.pt \
    --output outputs/ws8/weights
```

This writes six C8 exports: YOLO11n, standard YOLO26n and head-swapped YOLO26n,
each at 640×640 and 384×640. For the one-to-one decoder, parity references the
original YOLO26 checkpoint with `nms=False`, not the swapped checkpoint.

```sh
"$native_python" -m native.verify_native \
    --model /absolute/path/to/package \
    --reference /absolute/path/to/original.pt \
    --dataset /absolute/path/to/columba-test-v2 --split test \
    --output outputs/ws8/parity.json
```

`--images DIR` can replace `--dataset DIR --split test` for local images.
`--reference` also accepts a standard NCNN package for `yolo11`; a temporary
symlink supplies Ultralytics' required `_ncnn_model` suffix. One-to-one parity
requires the original PyTorch checkpoint. A native-only child process avoids
loading conflicting OpenMP runtimes together on macOS.

The matching tolerances come from `tools/compare_exports.py`: IoU ≥ 0.8,
confidence difference ≤ 0.05, equal classes and equal counts. The old model
produces one box clipped to zero height on a COCO8 image. Since its IoU is
undefined, the runner accepts it only when both outputs have identical
coordinates and class, still applying the confidence tolerance; it records
this case explicitly. It never removes unmatched degenerate boxes.

```sh
taskset -c 0-3 "$native_python" -m native.benchmark_native \
    --model /absolute/path/to/package \
    --image /absolute/path/to/bus.jpg --threads 4 \
    --warmup 20 --iterations 100 --frame-size 480 640 \
    --output outputs/ws8/overhead.json
```

Omit `taskset` on macOS. Persistent spawned workers isolate OpenMP runtimes;
only one backend runs a measured call at a time, alternating order each frame.
Timers exclude process communication. Both NCNN networks receive matching
thread and precision options before model loading, because NCNN caches kernel
choices then. Overhead is each full call minus its own instrumented extract.
The JSON includes every sample and p50/p95/mean/std. Mac results are labelled
as shared-machine measurements, never Pi measurements.

Python verification commands include the new source directory, excluding its
ignored dependency checkouts from mypy:

```sh
"$native_python" -m ruff check app tests tools cluster training native
"$native_python" -m ruff format --check app tests tools cluster training native
"$native_python" -m mypy --exclude '^native/build/' app tests tools cluster training native
```

The Python tests mock `columba_native` for the wrapper. Extension-only tests
skip cleanly when it is absent and use a tiny synthetic NCNN graph when it is
present. C++ tests cover letterbox geometry, channel order, normalisation,
synthetic decode, clipping, per-class/agnostic NMS and one-to-one top-k.
