#!/bin/bash
#SBATCH --job-name=cz-ws4-bench-s
#SBATCH --partition=normal
#SBATCH --nodelist=xcng0
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=02:45:00
#SBATCH --nice=1000
#SBATCH --output=slurm-%j.log
set -euo pipefail
workspace="$HOME/columba/scratch/ws4"
round_workspace="$workspace/round2"
stage="/tmp/cz-$SLURM_JOB_ID"
mkdir -p "$stage"
trap 'rm -rf "$stage"' EXIT
tar -xzf "$round_workspace/bundle.tar.gz" -C "$stage"
test -f "$workspace/ncnn-xcng/lib/libcolumba_bench.so"
export PYTHONPATH="$stage"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CZ_WS4_STAGE="$stage"
export CZ_WS4_WORKSPACE="$workspace"
srun --cpu-bind=cores bash -c '
    set -euo pipefail
    cd "$CZ_WS4_STAGE"
    "$CZ_WS4_WORKSPACE/venv/bin/python" -c "import os,numpy,cv2,yaml; print(os.sched_getaffinity(0)); assert set(range(4)) <= os.sched_getaffinity(0)"
    taskset -c 0-3 "$CZ_WS4_WORKSPACE/venv/bin/python" run.py "$CZ_WS4_STAGE" xcng "$CZ_WS4_WORKSPACE/ncnn-xcng"
' > "$round_workspace/bench-xcng.log" 2>&1
cp -a "$stage/results" "$round_workspace/"
printf '%s\n' "$SLURM_JOB_ID" > "$round_workspace/DONE"
