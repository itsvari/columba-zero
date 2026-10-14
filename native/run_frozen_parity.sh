#!/usr/bin/env bash
set -euo pipefail

artifact_root="${1:?usage: run_frozen_parity.sh ARTIFACT_ROOT FROZEN_DATASET PROXY_IMAGES REPORT_DIRECTORY}"
frozen_dataset="${2:?frozen dataset required}"
proxy_images="${3:?proxy images required}"
report_directory="${4:?report directory required}"
verification_python="${VERIFICATION_PYTHON:-python}"
mkdir -p "$report_directory"
failed_cases=0

for family in baseline yolo11n yolo26n yolo26n-one2one; do
    reference="$artifact_root/weights/fp32/reference-yolo26n.pt"
    if [[ "$family" == yolo11n ]]; then
        reference="$artifact_root/weights/fp32/reference-yolo11n.pt"
    elif [[ "$family" == baseline ]]; then
        reference="$artifact_root/weights/reference-baseline.pt"
    fi
    for precision in fp32 fp16; do
        compute=fp32
        model="$artifact_root/weights/$precision/${family}-384x640_ncnn_model"
        if [[ "$precision" == fp16 ]]; then
            compute=package
        fi
        if [[ "$family" == baseline && "$precision" == fp32 ]]; then
            model="$artifact_root/weights/baseline_ncnn_model"
        elif [[ "$family" == baseline ]]; then
            model="$artifact_root/weights/fp16/baseline_ncnn_model"
        fi
        if ! "$verification_python" -m native.verify_native \
            --model "$model" --reference "$reference" --threads 4 --confidence 0.25 \
            --compute "$compute" --dataset "$frozen_dataset" --split test \
            --output "$report_directory/frozen-$family-$precision.json"; then
            failed_cases=$((failed_cases + 1))
        fi
        if ! "$verification_python" -m native.verify_native \
            --model "$model" --reference "$reference" --threads 4 --confidence 0.25 \
            --compute "$compute" --images "$proxy_images" \
            --output "$report_directory/proxy-$family-$precision.json"; then
            failed_cases=$((failed_cases + 1))
        fi
    done
done
[[ "$failed_cases" == 0 ]]
