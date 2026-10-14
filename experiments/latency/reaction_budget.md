# Recomputed PLAN section 4 reaction budget

All per-frame Pi values are **estimated** from measured xcng0 distributions. Reaction values are **simulated** from 20,000 trials, seed 42. Camera 30 FPS, sensor delay uniform 1–2 frame periods, latest-frame buffer, recall 0.8. Each trial pre-rolls 128 camera frames. Origin is the first visible frame capture timestamp, so there is no extra pre-capture wait.

The main table uses FP16 and the realistic one-class incumbent alongside COCO80 model probes. The independent k=2/n=3 scenario matches the starting PLAN budget. rho=0.6 is an assumed temporal-correlation sensitivity scenario, not a measured error correlation. k=3/n=5 is the fallback confirmation policy. Accuracy and recall are assumptions here; int8 models are latency-only calibrated. See reaction_results.json for all precisions and scenario metadata.

| Model | Input H×W | Estimated Pi frame p95 | Simulated k2/n3 p50 | Simulated k2/n3 p95 | Simulated k3/n5 p95 | Simulated k2/n3 rho=.6 p95 |
|---|---|---:|---:|---:|---:|---:|
| pigeon_baseline_v1-2 | 352×608 | 29.14 | 120.07 | 199.82 | 239.64 | 299.36 |
| pigeon_baseline_v1-2 | 384×640 | 32.83 | 123.46 | 202.98 | 243.38 | 302.37 |
| pigeon_baseline_v1-2 | 448×800 | 48.89 | 174.38 | 273.88 | 329.98 | 380.23 |
| pigeon_baseline_v1-2 | 640×640 | 54.42 | 188.50 | 305.43 | 368.73 | 398.18 |
| yolo11n-coco | 352×608 | 33.53 | 123.31 | 203.23 | 242.66 | 302.72 |
| yolo11n-coco | 384×640 | 38.65 | 145.05 | 226.22 | 269.21 | 329.92 |
| yolo11n-coco | 448×800 | 50.11 | 181.14 | 290.89 | 350.31 | 389.95 |
| yolo11n-coco | 640×640 | 55.99 | 195.13 | 317.44 | 382.37 | 408.49 |
| yolo26n-coco | 352×608 | 32.36 | 122.73 | 202.40 | 242.55 | 302.25 |
| yolo26n-coco | 384×640 | 36.89 | 140.99 | 222.53 | 261.78 | 323.98 |
| yolo26n-coco | 448×800 | 51.61 | 182.07 | 292.69 | 349.13 | 386.03 |
| yolo26n-coco | 640×640 | 55.56 | 194.80 | 317.46 | 382.50 | 414.67 |

Calibration factor 1.527111 = 94.03 ms published mean inference / 61.573794 ms xcng0 mean inference for COCO YOLO11n FP32 640×640. [Official guide](https://github.com/ultralytics/ultralytics/blob/v8.3.200/docs/en/guides/raspberry-pi.md), accessed 2026-10-14. Scaling the whole pipeline and other precisions is approximate; it does not establish a Pi confidence interval.

Coordinator request: replace the provisional typical/p95 result in PLAN section 4 with this table or the chosen model row. Keep the 70/100 ms per-frame and 500 ms reaction gates. The rho=.6 sensitivity column shows how sustained misses can consume the reaction budget; replace this assumption with real temporal recall measurements in WS10.

Reproduce a row with:

```sh
python tools/reaction_budget.py experiments/latency/xcng0-pigeon_baseline_v1-2-ncnn-fp16-384x640.json \
  --calibration experiments/latency/calibration.json --camera-fps 30 --recall 0.8 --k 2 --n 3 --seed 42
```

## Round 2: s-model reaction budget

Per-frame Pi times are **estimated** from measured xcng0 samples. Reaction values are **simulated**, 20,000 trials, seed 42, 30 FPS, sensor delay uniform 1–2 camera periods, latest-frame policy, recall 0.8, k=2/n=3. Correlation rho=.6 is a sensitivity assumption on captured-frame hits, not a measured temporal recall. Both FP16 and latency-only int8 are shown.

| Model | Input H×W | Format | Estimated Pi frame p95 | Simulated reaction p50 | Simulated reaction p95 | Simulated rho=.6 p95 | 500 ms reaction gate |
|---|---|---|---:|---:|---:|---:|---|
| yolo11s-coco | 352×608 | ncnn-fp16 | 72.15 | 239.00 | 396.62 | 494.21 | PASS |
| yolo11s-coco | 384×640 | ncnn-fp16 | 81.83 | 260.55 | 438.67 | 532.67 | PASS |
| yolo11s-coco | 448×800 | ncnn-fp16 | 116.39 | 340.33 | 612.45 | 670.56 | FAIL |
| yolo11s-coco | 640×640 | ncnn-fp16 | 131.42 | 407.00 | 685.26 | 776.56 | FAIL |
| yolo11s-coco | 352×608 | ncnn-int8 | 67.40 | 227.10 | 378.60 | 466.41 | PASS |
| yolo11s-coco | 384×640 | ncnn-int8 | 77.72 | 263.40 | 433.65 | 532.36 | PASS |
| yolo11s-coco | 448×800 | ncnn-int8 | 113.34 | 315.41 | 600.20 | 646.79 | FAIL |
| yolo11s-coco | 640×640 | ncnn-int8 | 129.28 | 407.82 | 715.57 | 777.55 | FAIL |
| yolo26s-coco | 352×608 | ncnn-fp16 | 72.46 | 235.20 | 389.15 | 486.48 | PASS |
| yolo26s-coco | 384×640 | ncnn-fp16 | 83.54 | 264.98 | 448.37 | 535.15 | PASS |
| yolo26s-coco | 448×800 | ncnn-fp16 | 121.74 | 379.21 | 630.96 | 723.41 | FAIL |
| yolo26s-coco | 640×640 | ncnn-fp16 | 139.15 | 427.71 | 730.93 | 813.93 | FAIL |
| yolo26s-coco | 352×608 | ncnn-int8 | 70.35 | 241.94 | 393.42 | 496.26 | PASS |
| yolo26s-coco | 384×640 | ncnn-int8 | 80.52 | 263.60 | 446.44 | 528.45 | PASS |
| yolo26s-coco | 448×800 | ncnn-int8 | 119.39 | 384.12 | 721.02 | 755.59 | FAIL |
| yolo26s-coco | 640×640 | ncnn-int8 | 136.55 | 414.65 | 758.66 | 799.40 | FAIL |
