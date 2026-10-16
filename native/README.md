# Native NCNN detector

`columba_native` runs detection without importing PyTorch or Ultralytics. The
extension uses NCNN tag `20260526`, nanobind and OpenCV core/imgproc. Python needs
NumPy and PyYAML. The application wrapper imports the extension only when a
`NativeDetector` is constructed. The shipped configuration (`model.backend:
native` in `app/config.yaml`) needs it, so build it before running the app.

## Build on macOS ARM64

Run from the main checkout. The merged WS4 script builds NCNN tag `20260526`
with Vulkan and post-A76 CPU features disabled, installing its CMake package in
`native/build/ncnn-mac-a76compat/lib/cmake/ncnn`. The OpenCV helper clones
`5.0.0` into ignored `native/build/sources/opencv`, builds core/imgproc and
installs static libraries in `native/build/opencv-mac-arm64`. `--opencv-only`
avoids building a second NCNN prefix. Homebrew `libomp` supplies OpenMP.
The architecture flags also handle an x86_64 CMake binary running under Rosetta.

```sh
cd /Users/vari/Developer/columba-zero
native_python="$PWD/.venv/bin/python"
brew install libomp
bash tools/build_ncnn_a76.sh --profile mac-a76compat --jobs 4
BUILD_THREADS=4 bash native/build_dependencies.sh mac-arm64 --opencv-only
cmake -S native -B native/build/mac-arm64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 \
    -DPython_EXECUTABLE="$native_python" \
    -Dncnn_DIR="$PWD/native/build/ncnn-mac-a76compat/lib/cmake/ncnn" \
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

Use Python 3.12.3 on the arm lane. The cluster keeps one working build, which
`tools/select_model.py` hard-codes for its ARM evaluation bundles:

- environment: `~/columba/scratch/ws8/venv`, outside WS3's environments;
- extension: `~/columba/scratch/ws8/round3/repo/native/build/linux-aarch64`;
- NCNN: `~/columba/scratch/ws4/ncnn-xcng`.

The staged checkout `~/columba/scratch/ws8/repo`, the WS8 parity artefacts and
the Linux OpenCV prefix were deleted after WS7. A rebuild needs a fresh Git
checkout and `build_dependencies.sh --opencv-only` (below).

The Slurm runner `native/run_arm_job.sh` bundles dependency installation,
build, C++ and Python tests, parity and overhead measurement. It requests
eight CPUs on `xcng1`, `normal`, a two-hour limit and `--nice=1000`. Check
`squeue -u $USER` immediately before submission and respect the project caps.
It expects a Git checkout at `~/columba/scratch/ws8/repo` and parity models
and images at `~/columba/scratch/ws8/artifacts` (create them with
`native.prepare_models`, below); logs and results stay under the same root.
The full project tests need Git history and tracked deployment fixtures because
they create local clones. `--checks-only` reruns those Python checks against an
existing build without repeating model inference.

The direct build commands, inside the allocated arm job, are:

```sh
native_python="$HOME/columba/scratch/ws8/venv/bin/python"
BUILD_THREADS=8 bash native/build_dependencies.sh linux-aarch64 --opencv-only
cmake -S native -B native/build/linux-aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$native_python" \
    -Dncnn_DIR="$HOME/columba/scratch/ws4/ncnn-xcng/lib/cmake/ncnn" \
    -DOpenCV_DIR="$PWD/native/build/opencv-linux-aarch64/lib/cmake/opencv5"
cmake --build native/build/linux-aarch64 --parallel 8
ctest --test-dir native/build/linux-aarch64 --output-on-failure
export PYTHONPATH="$PWD/native/build/linux-aarch64:$PWD"
"$native_python" -m unittest discover -s tests -v
```

CMake accepts `-Dncnn_DIR=<prefix>/lib/cmake/ncnn`; it never fetches NCNN.
For a fresh Linux aarch64 prefix, run
`bash tools/build_ncnn_a76.sh --profile xcng --jobs 4` from the checkout and use
`$PWD/native/build/ncnn-xcng/lib/cmake/ncnn`. Work stays under `~/columba/` on
the cluster. Existing WS4 installations can be reused without modification.

On a Raspberry Pi, `sudo apt install libopencv-dev` supplies OpenCV core/imgproc
and its CMake config, usually under `/usr/lib/aarch64-linux-gnu/cmake/opencv4`.
Build NCNN with `bash tools/build_ncnn_a76.sh --profile pi5 --jobs 4`, then use
`-Dncnn_DIR="$PWD/native/build/ncnn-pi5/lib/cmake/ncnn"` and omit `OpenCV_DIR`
if CMake finds the system installation; otherwise set it to that OpenCV config
directory. The extension supports OpenCV 4 or 5. The recorded byte-exact
letterbox tests use OpenCV 5.0.0; a physical Pi build is a WS10 follow-up.

## Python interfaces

```python
from columba_native import Detector

detector = Detector(model_dir, threads=4, compute=None)
rows = detector.detect(frame, conf=0.5)
```

`frame` must be a non-empty C-contiguous NumPy `uint8` H×W×3 BGR array. The
binding borrows the existing buffer and rejects dtype/layout conversions.
Read-only inputs work. Do not modify the buffer from another thread during a
call. The GIL is released during preprocessing, inference and decoding.

`rows` is an owned, C-contiguous `float32` N×6 array. Columns are
`x1, y1, x2, y2, score, class_id` in original-frame pixels; an empty result has
shape `(0, 6)`. Properties are `input_hw`, `names`, `decode`, effective `compute`
and `compute_flags`, the three actual NCNN FP16 options on the loaded network.

`detect_with_timings(frame, conf)` additionally returns a dictionary with
`preprocess`, `extract`, `postprocess`, `total`, all in milliseconds.
`extract` times only NCNN's `Extractor::extract("out0", ...)`; extractor creation
and input submission count towards preprocessing. Output-array construction
occurs after the C++ total timer, so the benchmark uses the full Python-call
wall time when calculating overhead.

The application API is:

```python
from app.native_detector import NativeDetector

detector = NativeDetector(model_path, image_size, confidence, threads=4, compute=None)
detector.warmup(width, height)
detections = detector.detect(packet)
```

`image_size` is an integer or `(height, width)` tuple. `names` is a class-index
mapping. `last_stage_ms` is updated on every successful call with `preprocess`,
`inference`, `postprocess`; `inference` is the C++ extract time. Validation and
inference errors are `DetectorError`. The wrapper uses main's shared package validation, including the restriction
that only pigeon may trigger. All species detections above the confidence
threshold are returned with lower-case metadata labels and `class_id` keywords.
It matches WS9's factory and C9 timing contract. The optional compute argument
preserves that factory's existing positional calls; a config override is not
currently wired, so runtime construction uses the package default.

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

`compute` accepts `None`, `"fp16"` or `"fp32"`. `None` selects the package
default: C8 `quantization: fp16` and `int8` turn on FP16 packing, storage and
arithmetic; `fp32` and legacy packages turn all three off. An explicit override
sets all three together, independently of stored weight quantisation. Int8
kernels remain enabled. The FP16 options match WS4's latency bridge exactly.
FP16 arithmetic can shift scores and confidence-threshold counts; WS7 applied
the final FP16/int8 accuracy gate on frozen test-v2 (fp16 shipped, int8 rejected).

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
    --compute fp32 \
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
undefined, the runner accepts it only when coordinates differ by at most
`8 * float32 epsilon * max(abs(coordinates))` across either box, and the class
agrees, still applying the confidence tolerance. The whole-box scale accounts
for cancellation in small endpoints after subtracting padding and box widths.
The frozen trained exports exposed differences below 0.001 px that were many
ULPs at a small endpoint, despite agreement at the original coordinate scale.
The runner records these cases explicitly and retains unmatched degenerate boxes.

Parity always explicitly selects FP32 unless `--compute fp16` is supplied.
`--compute package` passes `None` to the extension and records the resolved
mode and package quantisation. A resolved FP16 mode is informational too.
FP16 arithmetic checks are informational, exit successfully despite reported
mismatches, and use looser IoU ≥ 0.7 and score difference ≤ 0.10, with equal
classes and counts still required. The report also records failures at the
strict FP32 tolerances, count-mismatch images and unmatched boxes. Maximum
box/score deltas use same-class diagnostic associations with IoU ≥ 0.5 and no
score restriction beyond the sigmoid range; unmatched detections remain
reported separately. The degenerate-box rule uses the same relative bound in both
modes, using the whole-box scale described above. These comparisons are not a
deployment accuracy gate.

Run all seven prepared packages and a baseline latency measurement with
`bash native/run_verification.sh outputs/ws8 outputs/ws8/results-fp16 fp16`;
the optional third argument defaults to `fp32`.

```sh
taskset -c 0-3 "$native_python" -m native.benchmark_native \
    --model /absolute/path/to/package \
    --image /absolute/path/to/bus.jpg --threads 4 \
    --warmup 20 --iterations 100 --frame-size 480 640 \
    --compute fp16 \
    --output outputs/ws8/overhead.json
```

Omit `taskset` on macOS. Persistent spawned workers isolate OpenMP runtimes;
only one backend runs a measured call at a time, alternating order each frame.
Timers exclude process communication. Both NCNN networks receive matching
thread and precision options before model loading, because NCNN caches kernel
choices then. Overhead is each full call minus its own instrumented extract.
The JSON includes every sample and p50/p95/mean/std. Mac results are labelled
as shared-machine measurements, never Pi measurements. `--compute` defaults
to `fp32`; use `fp16` to compare against WS4's FP16 arithmetic measurements.

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

Measured round-2 summaries are in [results/ws8-compute-modes.json](results/ws8-compute-modes.json)
and [WS8 status](../plans/model-improvement/status/WS8.md). FP32 passes 91/91
image/package comparisons. FP16 has two count-mismatch pairs and a baseline
degenerate-box mismatch; results remain informational. Native FP16 v1-2
640×640 extract p50/p95 is 14.441/17.335 ms on the shared M2 Pro, consistent
with WS4's 14.515/17.512 ms for the same model/size and NCNN profile.

For trained pigeon checkpoints, `native.prepare_models` accepts
`--input-size 384 640 --quantization fp32` or `fp16`. The latter exports FP16
weights with `quantize=16` and sets the corresponding C8 package default.
Single-class pigeon names produce `taxonomy: single` and pigeon triggers.

Frozen test-v2 and old-split proxy parity for both platforms is recorded in
[results/ws8-frozen-parity.json](results/ws8-frozen-parity.json). Required FP32
comparisons pass; FP16 package-default results are informational. The relative
degenerate-box bound avoids a fixed-ULP discontinuity observed on Linux at a
0.001018 px endpoint difference. The one-off frozen-parity runners
(`run_frozen_parity.sh`, `run_frozen_arm_job.sh`) and the offline
`rescore_parity.py` were removed in the cleanup; they are in git history at
`6de04fb`.
