#!/bin/bash
#SBATCH --job-name=cz-ws6-gate
#SBATCH --partition=normal
#SBATCH --nodelist=xcng0
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --nice=1000
#SBATCH --output=slurm-%j.log
set -euo pipefail
workspace="$HOME/columba/scratch/ws6"
stage="/tmp/cz-$SLURM_JOB_ID"
prefix="$HOME/columba/scratch/ws4/ncnn-xcng"
mkdir -p "$stage"
trap 'rm -rf "$stage"' EXIT
tar -xzf "$workspace/bundle.tar.gz" -C "$stage"
if [[ ! -x "$workspace/venv/bin/python" ]]; then
    python -m venv "$workspace/venv"
    "$workspace/venv/bin/python" -m pip install numpy opencv-python-headless pyyaml pillow > "$workspace/install-arm.log" 2>&1
fi
export PYTHONPATH="$stage"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CZ_WS6_STAGE="$stage"
export CZ_WS6_PYTHON="$workspace/venv/bin/python"
export CZ_WS6_PREFIX="$prefix"
srun --cpu-bind=cores bash -c '
    set -euo pipefail
    cd "$CZ_WS6_STAGE"
    for dimensions in "352 608" "384 640" "448 800"; do
        read -r height width <<< "$dimensions"
        taskset -c 0-3 "$CZ_WS6_PYTHON" -m training.adapters.verify_challenger \
            --framework nanodet --model "models/nanodet-${height}x${width}" \
            --prefix "$CZ_WS6_PREFIX" --input-hw "$height" "$width" \
            --output "results/parity-${height}x${width}.json" \
            "references/nanodet-${height}x${width}-bus.npz" \
            "references/nanodet-${height}x${width}-zidane.npz"
        for precision in fp16 fp32; do
            status=0
            taskset -c 0-3 "$CZ_WS6_PYTHON" -m training.adapters.bench_challenger \
                --framework nanodet --model "models/nanodet-${height}x${width}" \
                --prefix "$CZ_WS6_PREFIX" --host-profile xcng --format "$precision" \
                --input-hw "$height" "$width" --frame bus.jpg --output results || status=$?
            [[ "$status" -le 1 ]]
        done
    done
' > "$workspace/gate.log" 2>&1
cp -a "$stage/results" "$workspace/"
touch "$workspace/DONE"
