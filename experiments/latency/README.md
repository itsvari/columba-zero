# WS4 latency measurements

All timings name their host. Mac results are measured on an M2 Pro with an
A76-compatible NCNN build on a shared machine. xcng results are measured on
Neoverse-N1, not on Raspberry Pi. Pi timings are estimated through
`cal-v0-published`; reaction timings are simulated.

## Build and link

```sh
bash tools/build_ncnn_a76.sh --profile mac-a76compat
bash tools/build_ncnn_a76.sh --profile xcng --prefix ~/columba/scratch/ws4/ncnn-xcng
bash tools/build_ncnn_a76.sh --profile pi5 --prefix native/build/ncnn-pi5
```

The pinned upstream tag is `20260526`, commit
`e54f7b1f88434e1d844ea0551b880a1cfb079ce1`, corresponding to pip
`ncnn==1.0.20260526`. The script requires Git, CMake and a C++17 compiler.
Mac builds also require Homebrew libomp; they explicitly request ARM64 because
the installed CMake executable may run under Rosetta. Vulkan is off. ARM82
fp16 and ARM82DOT are on; post-A76 FML, bf16, i8mm and all available SVE
options are off on all profiles. The Pi profile should be rechecked against
actual Pi CPU flags by WS10.

Standard install prefix for WS8:

```text
PREFIX/
  include/ncnn/{net.h,mat.h,platform.h,...}
  lib/libncnn.a
  lib/cmake/ncnn/{ncnnConfig.cmake,ncnn.cmake,...}
  lib/libcolumba_bench.dylib   # Mac; .so on Linux
  bin/{ncnnoptimize,ncnn2table,ncnn2int8,...}
  build-evidence.txt
```

Use `-DCMAKE_PREFIX_PATH=PREFIX` then `find_package(ncnn CONFIG REQUIRED)`
and `target_link_libraries(target PRIVATE ncnn)`. On Mac, the consuming CMake
configure also needs the libomp discovery flags used by the build script:
`-DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp"`,
`-DOpenMP_CXX_LIB_NAMES=omp`, and
`-DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"`.
`build-evidence.txt`
records generated feature macros, actual compiled kernel objects, CPU-query
symbols and the source pin. The script rejects any unexpected generated ARM
feature setting. Hardware feature reports alone cannot prove which
kernels are enabled, because the Mac reports features this build disables.

`tools/bench_ncnn/bridge.cpp` is a small C++17 RAII bridge. Python calls it
through ctypes. It links against this custom static library, never the pip
module. One process owns one model at a time. Python owns all borrowed buffers.

## Measure and gate

Run from the repository root with the project Python environment:

```sh
python tools/bench_ncnn.py run --model MODEL_DIR --model-id MODEL_ID \
  --input-hw 384 640 --host-profile mac-a76compat --format fp16 --frame FRAME.jpg
python tools/bench_ncnn.py gate --model MODEL_DIR --input-hw 384 640 \
  --host-profile xcng --prefix ~/columba/scratch/ws4/ncnn-xcng --format fp16
```

The gate runs 20 warm-up and 200 timed iterations by default and writes C7.
It prints estimated Pi end-to-end p95, PASS/FAIL for the 100 ms hard limit
and for the 70 ms target. Exit status is 0 for hard PASS, 1 for hard FAIL,
2 for invalid inputs or unavailable calibration. `--iterations` must be at
least 200. `--camera-hw 864 1536` measures the larger camera stream.
`--output experiments/latency/highres` keeps the C7 basename without overwriting
the 480×640 measurements. `--calibration` selects another calibration record.
A direct `--host-profile pi5` gate uses measured p95 and requires no factor.

Supported output is C8 `(4+nc,N)` with pixel cxcywh and scores already sigmoid.
Standard YOLO11 and YOLO26 exports use class-aware IoU 0.7 NMS. The optional
`--nms-free` switch is only for a separately verified YOLO26 one-to-one export;
it is not valid for a standard YOLO26 export. Non-C8 challenger outputs need a
model-specific decoder before this end-to-end gate can evaluate them.

Input dimensions must match `metadata.yaml` and be multiples of 32. Preprocessing
is OpenCV letterboxing from BGR uint8 to RGB CHW float32 divided by 255.
Postprocessing includes NumPy decode, NMS, original-frame box mapping/clipping
and a boolean decision. Confidence defaults to 0.5, max detections 300.
Camera reads, sensor delay, tracking, rendering and logging are outside these
per-frame timings. End-to-end times include bridge/FFI and output-copy overhead.
The C++ inference timer includes the input Mat copy and extractor input/extract.

The bundled measurements use one COCO128 photograph, resized to both camera
sizes. Thus postprocessing is input-dependent and is not a worst-case crowded
pigeon scene. COCO models have 80 classes; `pigeon_baseline_v1-2` has one.
fp32/fp16 use the same fp32 export weights and explicitly select NCNN storage,
packing and arithmetic options. int8 uses actual quantised convolution weights.
Its 32-image COCO128 KL table uses direct RGB resizing and 1/255 normalisation.
This is **latency-only calibration**, not WS7 deployment calibration or an
accuracy/parity claim. No checkpoint, image or model binary is committed.

C7 extensions requested from the coordinator: paired `samples_ms` end-to-end
measurements, camera dimensions, class count, confidence, hashes, load averages,
raw CPU feature report, timing boundaries and int8 calibration provenance. Percentiles
use NumPy linear interpolation; std is population standard deviation. Samples
are collected after warm-up and stage percentiles are not added together.

## Calibrate and simulate

```sh
python tools/bench_ncnn.py calibrate \
  --reference experiments/latency/xcng0-yolo11n-coco-ncnn-fp32-640x640.json \
  --mac-reference experiments/latency/vexcalibur-yolo11n-coco-ncnn-fp32-640x640.json
python tools/reaction_budget.py C7.json --calibration experiments/latency/calibration.json \
  --camera-fps 30 --recall 0.8 --k 2 --n 3 --correlation 0 --seed 42
```

Calibration divides the published **mean inference**, 94.03 ms for YOLO11n
COCO80 NCNN FP32 at 640×640, by the matching host's mean inference.
The [versioned official Raspberry Pi guide](https://github.com/ultralytics/ultralytics/blob/v8.3.200/docs/en/guides/raspberry-pi.md)
records Ultralytics 8.3.152 and explicitly excludes preprocessing and
postprocessing. Accessed 2026-10-14. The current guide has changed to YOLO26,
so a versioned official source is necessary. Published NCNN version, thread
count and thermal conditions are unknown. Scaling end-to-end distributions,
other precisions and other sizes with this one factor remains an estimate.
Mac uses its own host-matched factor and cannot use the xcng factor directly.
The measured xcng factor is 1.527110713, using 61.573793715 ms mean inference;
the Mac factor is 3.792166804, using 24.795850200 ms mean inference.
The compiled specialised ARM objects are asimdhp and asimddp variants, with no
FML, ARM84 bf16/i8mm or SVE variants. Generic bf16 storage routines may still
appear in symbol reports; `use_bf16_storage` is off. The raw feature report is
`fp16_query + 2 * dotprod_query`: Mac returns 3, Linux N1 returns 2098176 because
NCNN returns HWCAP bit values rather than booleans on Linux. Both hosts support
fp16 and dot-product; generated macros and compiled objects prove the build
restrictions.

The simulation starts at the first visible frame's capture timestamp, without
an extra pre-capture uniform wait. Sensor delay defaults to uniform 1–2 camera
periods. It pre-rolls the latest-frame processor and then bootstraps processing
times, consumes each camera frame at most once, and overwrites queued frames
when busy. k hits in the last n processed visible frames confirm a trigger.
There is no actuation time or tracking/occlusion model. The sensor delay is
fixed within each trial and varies across trials.

Independent hits are the default. `--correlation RHO` uses a stationary Markov
Bernoulli model whose marginal hit probability is recall. Persistence decays as
`RHO**camera_frame_gap`, including skipped frames. This is a sensitivity model,
not a measured temporal error model. `--frame-policy fifo` models a queue from
appearance with no pre-existing backlog; it does not describe `app/capture.py`.
C7 samples are preferred; summary-only input fits an estimated lognormal to
p50/p95. Reports record the seed, trials, policy and source distribution.
