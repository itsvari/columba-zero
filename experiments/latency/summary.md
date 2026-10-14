# C7 latency summary

Inference and end-to-end columns are **measured** on the named host. Pi p95 is **estimated**, never measured. All values are milliseconds; 4 threads, 20 warm-up, 200 timed iterations. Mac M2 Pro measurements used a shared machine. xcng Neoverse-N1 measurements used four allocated cores.

Calibration xcng factor 1.527111; Mac factor 3.792167. Published Pi5 YOLO11n FP32 640×640 mean inference 94.03 ms. [Official versioned guide](https://github.com/ultralytics/ultralytics/blob/v8.3.200/docs/en/guides/raspberry-pi.md), accessed 2026-10-14.

## Camera 640×480

| Host | Model | Input H×W | Format | Inference p50 | Inference p95 | End-to-end p95 | Estimated Pi p95 |
|---|---|---|---|---:|---:|---:|---:|
| vexcalibur | pigeon_baseline_v1-2 | 352×608 | ncnn-fp16 | 10.91 | 17.02 | 17.96 | 68.10 |
| vexcalibur | pigeon_baseline_v1-2 | 384×640 | ncnn-fp16 | 10.12 | 16.39 | 17.06 | 64.70 |
| vexcalibur | pigeon_baseline_v1-2 | 448×800 | ncnn-fp16 | 12.53 | 13.46 | 14.25 | 54.05 |
| vexcalibur | pigeon_baseline_v1-2 | 640×640 | ncnn-fp16 | 14.52 | 17.51 | 18.35 | 69.60 |
| vexcalibur | pigeon_baseline_v1-2 | 352×608 | ncnn-fp32 | 16.33 | 23.58 | 24.10 | 91.37 |
| vexcalibur | pigeon_baseline_v1-2 | 384×640 | ncnn-fp32 | 16.20 | 22.67 | 23.49 | 89.10 |
| vexcalibur | pigeon_baseline_v1-2 | 448×800 | ncnn-fp32 | 23.59 | 33.85 | 34.92 | 132.44 |
| vexcalibur | pigeon_baseline_v1-2 | 640×640 | ncnn-fp32 | 22.82 | 23.99 | 24.73 | 93.76 |
| vexcalibur | pigeon_baseline_v1-2 | 352×608 | ncnn-int8 | 9.44 | 13.96 | 14.51 | 55.03 |
| vexcalibur | pigeon_baseline_v1-2 | 384×640 | ncnn-int8 | 11.09 | 17.15 | 17.81 | 67.55 |
| vexcalibur | pigeon_baseline_v1-2 | 448×800 | ncnn-int8 | 13.78 | 14.48 | 15.19 | 57.62 |
| vexcalibur | pigeon_baseline_v1-2 | 640×640 | ncnn-int8 | 16.06 | 17.41 | 18.18 | 68.96 |
| vexcalibur | yolo11n-coco | 352×608 | ncnn-fp16 | 8.31 | 8.65 | 9.33 | 35.39 |
| vexcalibur | yolo11n-coco | 384×640 | ncnn-fp16 | 9.39 | 9.83 | 10.63 | 40.30 |
| vexcalibur | yolo11n-coco | 448×800 | ncnn-fp16 | 13.24 | 13.73 | 14.76 | 55.98 |
| vexcalibur | yolo11n-coco | 640×640 | ncnn-fp16 | 15.23 | 18.75 | 20.25 | 76.77 |
| vexcalibur | yolo11n-coco | 352×608 | ncnn-fp32 | 13.40 | 13.98 | 14.64 | 55.51 |
| vexcalibur | yolo11n-coco | 384×640 | ncnn-fp32 | 14.93 | 15.74 | 16.50 | 62.58 |
| vexcalibur | yolo11n-coco | 448×800 | ncnn-fp32 | 21.17 | 24.34 | 25.67 | 97.35 |
| vexcalibur | yolo11n-coco | 640×640 | ncnn-fp32 | 24.26 | 27.93 | 29.10 | 110.35 |
| vexcalibur | yolo11n-coco | 352×608 | ncnn-int8 | 9.08 | 9.57 | 10.21 | 38.72 |
| vexcalibur | yolo11n-coco | 384×640 | ncnn-int8 | 10.47 | 11.50 | 12.42 | 47.11 |
| vexcalibur | yolo11n-coco | 448×800 | ncnn-int8 | 14.94 | 15.58 | 16.59 | 62.90 |
| vexcalibur | yolo11n-coco | 640×640 | ncnn-int8 | 16.80 | 17.81 | 18.92 | 71.76 |
| vexcalibur | yolo26n-coco | 352×608 | ncnn-fp16 | 8.43 | 9.02 | 9.74 | 36.95 |
| vexcalibur | yolo26n-coco | 384×640 | ncnn-fp16 | 9.18 | 9.55 | 10.33 | 39.16 |
| vexcalibur | yolo26n-coco | 448×800 | ncnn-fp16 | 13.25 | 14.58 | 15.62 | 59.22 |
| vexcalibur | yolo26n-coco | 640×640 | ncnn-fp16 | 15.27 | 15.75 | 16.74 | 63.46 |
| vexcalibur | yolo26n-coco | 352×608 | ncnn-fp32 | 13.44 | 13.82 | 14.43 | 54.72 |
| vexcalibur | yolo26n-coco | 384×640 | ncnn-fp32 | 14.96 | 15.44 | 16.19 | 61.40 |
| vexcalibur | yolo26n-coco | 448×800 | ncnn-fp32 | 21.03 | 21.71 | 22.71 | 86.13 |
| vexcalibur | yolo26n-coco | 640×640 | ncnn-fp32 | 25.21 | 30.12 | 31.14 | 118.10 |
| vexcalibur | yolo26n-coco | 352×608 | ncnn-int8 | 8.92 | 9.95 | 10.68 | 40.49 |
| vexcalibur | yolo26n-coco | 384×640 | ncnn-int8 | 9.97 | 10.30 | 11.00 | 41.73 |
| vexcalibur | yolo26n-coco | 448×800 | ncnn-int8 | 15.98 | 22.78 | 23.85 | 90.43 |
| vexcalibur | yolo26n-coco | 640×640 | ncnn-int8 | 17.32 | 25.22 | 26.35 | 99.91 |
| xcng0 | pigeon_baseline_v1-2 | 352×608 | ncnn-fp16 | 16.16 | 17.64 | 19.08 | 29.14 |
| xcng0 | pigeon_baseline_v1-2 | 384×640 | ncnn-fp16 | 18.28 | 19.88 | 21.50 | 32.83 |
| xcng0 | pigeon_baseline_v1-2 | 448×800 | ncnn-fp16 | 26.50 | 29.89 | 32.02 | 48.89 |
| xcng0 | pigeon_baseline_v1-2 | 640×640 | ncnn-fp16 | 30.29 | 33.23 | 35.63 | 54.42 |
| xcng0 | pigeon_baseline_v1-2 | 352×608 | ncnn-fp32 | 29.00 | 31.57 | 32.96 | 50.34 |
| xcng0 | pigeon_baseline_v1-2 | 384×640 | ncnn-fp32 | 33.57 | 35.28 | 36.85 | 56.27 |
| xcng0 | pigeon_baseline_v1-2 | 448×800 | ncnn-fp32 | 50.58 | 53.14 | 55.31 | 84.47 |
| xcng0 | pigeon_baseline_v1-2 | 640×640 | ncnn-fp32 | 57.78 | 60.61 | 62.51 | 95.46 |
| xcng0 | pigeon_baseline_v1-2 | 352×608 | ncnn-int8 | 18.71 | 19.68 | 21.07 | 32.18 |
| xcng0 | pigeon_baseline_v1-2 | 384×640 | ncnn-int8 | 21.04 | 22.37 | 23.95 | 36.57 |
| xcng0 | pigeon_baseline_v1-2 | 448×800 | ncnn-int8 | 31.65 | 33.84 | 35.97 | 54.93 |
| xcng0 | pigeon_baseline_v1-2 | 640×640 | ncnn-int8 | 35.03 | 37.98 | 39.86 | 60.87 |
| xcng0 | yolo11n-coco | 352×608 | ncnn-fp16 | 17.98 | 20.16 | 21.96 | 33.53 |
| xcng0 | yolo11n-coco | 384×640 | ncnn-fp16 | 20.22 | 23.31 | 25.31 | 38.65 |
| xcng0 | yolo11n-coco | 448×800 | ncnn-fp16 | 27.80 | 30.17 | 32.81 | 50.11 |
| xcng0 | yolo11n-coco | 640×640 | ncnn-fp16 | 31.13 | 34.11 | 36.67 | 55.99 |
| xcng0 | yolo11n-coco | 352×608 | ncnn-fp32 | 31.46 | 33.20 | 35.01 | 53.47 |
| xcng0 | yolo11n-coco | 384×640 | ncnn-fp32 | 35.89 | 37.75 | 39.69 | 60.61 |
| xcng0 | yolo11n-coco | 448×800 | ncnn-fp32 | 53.52 | 55.58 | 58.22 | 88.91 |
| xcng0 | yolo11n-coco | 640×640 | ncnn-fp32 | 61.51 | 64.01 | 66.51 | 101.56 |
| xcng0 | yolo11n-coco | 352×608 | ncnn-int8 | 20.87 | 23.27 | 25.04 | 38.24 |
| xcng0 | yolo11n-coco | 384×640 | ncnn-int8 | 23.30 | 25.66 | 27.62 | 42.19 |
| xcng0 | yolo11n-coco | 448×800 | ncnn-int8 | 33.16 | 35.08 | 37.72 | 57.60 |
| xcng0 | yolo11n-coco | 640×640 | ncnn-int8 | 37.25 | 38.99 | 41.49 | 63.35 |
| xcng0 | yolo26n-coco | 352×608 | ncnn-fp16 | 17.49 | 19.40 | 21.19 | 32.36 |
| xcng0 | yolo26n-coco | 384×640 | ncnn-fp16 | 19.82 | 22.15 | 24.16 | 36.89 |
| xcng0 | yolo26n-coco | 448×800 | ncnn-fp16 | 28.04 | 30.68 | 33.80 | 51.61 |
| xcng0 | yolo26n-coco | 640×640 | ncnn-fp16 | 31.22 | 33.51 | 36.38 | 55.56 |
| xcng0 | yolo26n-coco | 352×608 | ncnn-fp32 | 31.33 | 32.93 | 34.69 | 52.98 |
| xcng0 | yolo26n-coco | 384×640 | ncnn-fp32 | 35.61 | 38.08 | 40.07 | 61.19 |
| xcng0 | yolo26n-coco | 448×800 | ncnn-fp32 | 54.17 | 56.37 | 59.02 | 90.14 |
| xcng0 | yolo26n-coco | 640×640 | ncnn-fp32 | 62.15 | 64.54 | 67.04 | 102.37 |
| xcng0 | yolo26n-coco | 352×608 | ncnn-int8 | 19.97 | 22.04 | 23.78 | 36.32 |
| xcng0 | yolo26n-coco | 384×640 | ncnn-int8 | 22.70 | 25.06 | 27.00 | 41.22 |
| xcng0 | yolo26n-coco | 448×800 | ncnn-int8 | 33.25 | 34.78 | 37.39 | 57.10 |
| xcng0 | yolo26n-coco | 640×640 | ncnn-int8 | 37.05 | 38.95 | 41.45 | 63.30 |

## Camera 1536×864

| Host | Model | Input H×W | Format | Inference p50 | Inference p95 | End-to-end p95 | Estimated Pi p95 |
|---|---|---|---|---:|---:|---:|---:|
| vexcalibur | pigeon_baseline_v1-2 | 352×608 | ncnn-fp16 | 9.84 | 18.75 | 19.66 | 74.55 |
| vexcalibur | pigeon_baseline_v1-2 | 384×640 | ncnn-fp16 | 10.88 | 32.21 | 33.05 | 125.35 |
| vexcalibur | pigeon_baseline_v1-2 | 448×800 | ncnn-fp16 | 12.81 | 16.54 | 17.42 | 66.06 |
| vexcalibur | pigeon_baseline_v1-2 | 640×640 | ncnn-fp16 | 15.46 | 28.92 | 32.20 | 122.10 |
| vexcalibur | pigeon_baseline_v1-2 | 352×608 | ncnn-fp32 | 17.77 | 22.65 | 23.37 | 88.64 |
| vexcalibur | pigeon_baseline_v1-2 | 384×640 | ncnn-fp32 | 16.89 | 25.02 | 25.72 | 97.54 |
| vexcalibur | pigeon_baseline_v1-2 | 448×800 | ncnn-fp32 | 20.63 | 21.34 | 22.19 | 84.15 |
| vexcalibur | pigeon_baseline_v1-2 | 640×640 | ncnn-fp32 | 23.25 | 24.31 | 25.19 | 95.54 |
| vexcalibur | pigeon_baseline_v1-2 | 352×608 | ncnn-int8 | 11.18 | 21.69 | 23.12 | 87.67 |
| vexcalibur | pigeon_baseline_v1-2 | 384×640 | ncnn-int8 | 14.61 | 25.02 | 25.66 | 97.31 |
| vexcalibur | pigeon_baseline_v1-2 | 448×800 | ncnn-int8 | 13.74 | 14.11 | 14.87 | 56.38 |
| vexcalibur | pigeon_baseline_v1-2 | 640×640 | ncnn-int8 | 16.00 | 17.21 | 18.16 | 68.86 |
| vexcalibur | yolo11n-coco | 352×608 | ncnn-fp16 | 8.45 | 9.11 | 9.95 | 37.73 |
| vexcalibur | yolo11n-coco | 384×640 | ncnn-fp16 | 9.30 | 9.58 | 10.37 | 39.31 |
| vexcalibur | yolo11n-coco | 448×800 | ncnn-fp16 | 13.27 | 13.90 | 14.93 | 56.62 |
| vexcalibur | yolo11n-coco | 640×640 | ncnn-fp16 | 15.21 | 16.08 | 17.22 | 65.30 |
| vexcalibur | yolo11n-coco | 352×608 | ncnn-fp32 | 13.42 | 13.88 | 14.60 | 55.37 |
| vexcalibur | yolo11n-coco | 384×640 | ncnn-fp32 | 14.97 | 17.18 | 18.00 | 68.25 |
| vexcalibur | yolo11n-coco | 448×800 | ncnn-fp32 | 21.14 | 23.10 | 24.39 | 92.47 |
| vexcalibur | yolo11n-coco | 640×640 | ncnn-fp32 | 24.59 | 27.22 | 28.30 | 107.33 |
| vexcalibur | yolo11n-coco | 352×608 | ncnn-int8 | 9.09 | 9.56 | 10.24 | 38.81 |
| vexcalibur | yolo11n-coco | 384×640 | ncnn-int8 | 10.50 | 11.17 | 11.96 | 45.34 |
| vexcalibur | yolo11n-coco | 448×800 | ncnn-int8 | 14.44 | 15.47 | 16.56 | 62.78 |
| vexcalibur | yolo11n-coco | 640×640 | ncnn-int8 | 16.41 | 17.12 | 18.19 | 68.98 |
| vexcalibur | yolo26n-coco | 352×608 | ncnn-fp16 | 8.40 | 9.14 | 9.85 | 37.36 |
| vexcalibur | yolo26n-coco | 384×640 | ncnn-fp16 | 9.28 | 9.71 | 10.56 | 40.03 |
| vexcalibur | yolo26n-coco | 448×800 | ncnn-fp16 | 14.06 | 18.79 | 20.19 | 76.56 |
| vexcalibur | yolo26n-coco | 640×640 | ncnn-fp16 | 14.94 | 15.29 | 16.41 | 62.22 |
| vexcalibur | yolo26n-coco | 352×608 | ncnn-fp32 | 13.31 | 13.92 | 14.67 | 55.64 |
| vexcalibur | yolo26n-coco | 384×640 | ncnn-fp32 | 14.88 | 15.36 | 16.12 | 61.15 |
| vexcalibur | yolo26n-coco | 448×800 | ncnn-fp32 | 21.68 | 23.12 | 24.28 | 92.09 |
| vexcalibur | yolo26n-coco | 640×640 | ncnn-fp32 | 24.75 | 27.11 | 28.33 | 107.43 |
| vexcalibur | yolo26n-coco | 352×608 | ncnn-int8 | 9.20 | 10.18 | 10.94 | 41.47 |
| vexcalibur | yolo26n-coco | 384×640 | ncnn-int8 | 9.96 | 10.22 | 10.93 | 41.43 |
| vexcalibur | yolo26n-coco | 448×800 | ncnn-int8 | 15.98 | 22.77 | 23.83 | 90.39 |
| vexcalibur | yolo26n-coco | 640×640 | ncnn-int8 | 18.16 | 30.53 | 31.74 | 120.36 |
| xcng0 | pigeon_baseline_v1-2 | 352×608 | ncnn-fp16 | 16.77 | 18.28 | 19.92 | 30.42 |
| xcng0 | pigeon_baseline_v1-2 | 384×640 | ncnn-fp16 | 18.47 | 20.47 | 22.22 | 33.93 |
| xcng0 | pigeon_baseline_v1-2 | 448×800 | ncnn-fp16 | 26.96 | 29.64 | 32.15 | 49.09 |
| xcng0 | pigeon_baseline_v1-2 | 640×640 | ncnn-fp16 | 30.68 | 33.36 | 35.81 | 54.68 |
| xcng0 | pigeon_baseline_v1-2 | 352×608 | ncnn-fp32 | 29.89 | 31.88 | 33.53 | 51.20 |
| xcng0 | pigeon_baseline_v1-2 | 384×640 | ncnn-fp32 | 33.57 | 35.47 | 37.29 | 56.95 |
| xcng0 | pigeon_baseline_v1-2 | 448×800 | ncnn-fp32 | 51.71 | 54.70 | 57.21 | 87.37 |
| xcng0 | pigeon_baseline_v1-2 | 640×640 | ncnn-fp32 | 58.92 | 62.02 | 64.45 | 98.42 |
| xcng0 | pigeon_baseline_v1-2 | 352×608 | ncnn-int8 | 18.98 | 20.61 | 22.18 | 33.88 |
| xcng0 | pigeon_baseline_v1-2 | 384×640 | ncnn-int8 | 21.17 | 22.74 | 24.51 | 37.43 |
| xcng0 | pigeon_baseline_v1-2 | 448×800 | ncnn-int8 | 31.84 | 33.77 | 36.28 | 55.40 |
| xcng0 | pigeon_baseline_v1-2 | 640×640 | ncnn-int8 | 35.23 | 37.51 | 39.88 | 60.91 |
| xcng0 | yolo11n-coco | 352×608 | ncnn-fp16 | 18.00 | 20.70 | 22.70 | 34.66 |
| xcng0 | yolo11n-coco | 384×640 | ncnn-fp16 | 19.95 | 23.57 | 25.73 | 39.30 |
| xcng0 | yolo11n-coco | 448×800 | ncnn-fp16 | 28.02 | 30.81 | 33.87 | 51.72 |
| xcng0 | yolo11n-coco | 640×640 | ncnn-fp16 | 31.64 | 34.45 | 37.43 | 57.16 |
| xcng0 | yolo11n-coco | 352×608 | ncnn-fp32 | 32.16 | 33.80 | 35.82 | 54.71 |
| xcng0 | yolo11n-coco | 384×640 | ncnn-fp32 | 36.05 | 38.40 | 40.57 | 61.95 |
| xcng0 | yolo11n-coco | 448×800 | ncnn-fp32 | 53.69 | 56.47 | 59.51 | 90.88 |
| xcng0 | yolo11n-coco | 640×640 | ncnn-fp32 | 61.30 | 63.59 | 66.59 | 101.69 |
| xcng0 | yolo11n-coco | 352×608 | ncnn-int8 | 21.17 | 23.50 | 25.44 | 38.85 |
| xcng0 | yolo11n-coco | 384×640 | ncnn-int8 | 23.47 | 26.28 | 28.46 | 43.46 |
| xcng0 | yolo11n-coco | 448×800 | ncnn-int8 | 33.06 | 34.76 | 37.84 | 57.78 |
| xcng0 | yolo11n-coco | 640×640 | ncnn-int8 | 36.98 | 38.84 | 41.84 | 63.89 |
| xcng0 | yolo26n-coco | 352×608 | ncnn-fp16 | 17.76 | 20.19 | 22.20 | 33.91 |
| xcng0 | yolo26n-coco | 384×640 | ncnn-fp16 | 20.13 | 22.92 | 25.12 | 38.37 |
| xcng0 | yolo26n-coco | 448×800 | ncnn-fp16 | 28.05 | 30.47 | 33.56 | 51.24 |
| xcng0 | yolo26n-coco | 640×640 | ncnn-fp16 | 31.69 | 34.23 | 37.23 | 56.85 |
| xcng0 | yolo26n-coco | 352×608 | ncnn-fp32 | 31.60 | 33.41 | 35.41 | 54.07 |
| xcng0 | yolo26n-coco | 384×640 | ncnn-fp32 | 35.78 | 38.29 | 40.45 | 61.77 |
| xcng0 | yolo26n-coco | 448×800 | ncnn-fp32 | 53.74 | 56.97 | 60.03 | 91.68 |
| xcng0 | yolo26n-coco | 640×640 | ncnn-fp32 | 61.56 | 64.79 | 67.79 | 103.53 |
| xcng0 | yolo26n-coco | 352×608 | ncnn-int8 | 20.28 | 22.90 | 24.89 | 38.01 |
| xcng0 | yolo26n-coco | 384×640 | ncnn-int8 | 22.49 | 24.88 | 27.01 | 41.24 |
| xcng0 | yolo26n-coco | 448×800 | ncnn-int8 | 33.09 | 34.87 | 37.93 | 57.92 |
| xcng0 | yolo26n-coco | 640×640 | ncnn-int8 | 36.81 | 39.08 | 42.03 | 64.19 |

