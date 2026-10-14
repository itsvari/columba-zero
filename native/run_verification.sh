#!/usr/bin/env bash
set -euo pipefail

verification_python="${VERIFICATION_PYTHON:-python}"
artifact_root="${1:?usage: run_verification.sh ARTIFACT_ROOT REPORT_DIRECTORY [fp32|fp16]}"
report_directory="${2:?report directory required}"
compute="${3:-fp32}"
mkdir -p "$report_directory"

for size in 640x640 384x640; do
    for family in yolo11n yolo26n yolo26n-one2one; do
        reference="$artifact_root/weights/reference-yolo26n.pt"
        if [[ "$family" == yolo11n ]]; then
            reference="$artifact_root/weights/reference-yolo11n.pt"
        fi
        "$verification_python" -m native.verify_native \
            --model "$artifact_root/weights/${family}-${size}_ncnn_model" \
            --reference "$reference" --dataset "$artifact_root/parity-images" \
            --compute "$compute" \
            --split test --output "$report_directory/parity-${family}-${size}.json"
    done
done
"$verification_python" -m native.verify_native \
    --model "$artifact_root/weights/baseline_ncnn_model" \
    --reference "$artifact_root/weights/reference-baseline.pt" \
    --dataset "$artifact_root/parity-images" --split test \
    --compute "$compute" \
    --output "$report_directory/parity-baseline-640x640.json"
"$verification_python" -m native.benchmark_native \
    --model "$artifact_root/weights/baseline_ncnn_model" \
    --image "$artifact_root/parity-images/images/test/bus.jpg" \
    --threads 4 --warmup 20 --iterations 100 \
    --compute "$compute" \
    --output "$report_directory/overhead-baseline-640x640.json"
