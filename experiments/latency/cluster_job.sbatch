#!/bin/bash
#SBATCH --job-name=cz-ws4-bench
#SBATCH --partition=normal
#SBATCH --nodelist=xcng0
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=02:45:00
#SBATCH --nice=1000
#SBATCH --output=slurm-%j.log
set -euo pipefail
workspace="$HOME/columba/scratch/ws4"
stage="/tmp/cz-$SLURM_JOB_ID"
mkdir -p "$stage"
trap 'rm -rf "$stage"' EXIT
tar -xzf "$workspace/bundle.tar.gz" -C "$stage"
tar -xzf "$workspace/ncnn-source.tar.gz" -C "$stage"
cp "$workspace/run_bundle.py" "$stage/run_bundle.py"
if [[ ! -f "$workspace/ncnn-xcng/lib/libcolumba_bench.so" ]]; then
bash "$stage/tools/build_ncnn_a76.sh" --profile xcng --prefix "$stage/ncnn-install" --source "$stage/ncnn" --jobs 16 > "$workspace/build-xcng.log" 2>&1
mkdir -p "$workspace/ncnn-xcng"
cp -a "$stage/ncnn-install/lib" "$stage/ncnn-install/include" "$stage/ncnn-install/bin" "$stage/ncnn-install/build-evidence.txt" "$workspace/ncnn-xcng/"
fi
if ! "$workspace/venv/bin/python" -c 'import numpy, cv2, yaml' > "$workspace/venv-smoke.log" 2>&1; then
    python -m venv --clear "$workspace/venv"
    "$workspace/venv/bin/python" -m pip install numpy opencv-python-headless pyyaml > "$workspace/venv-install-arm.log" 2>&1
fi
export PYTHONPATH="$stage"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CZ_WS4_STAGE="$stage"
export CZ_WS4_WORKSPACE="$workspace"
# srun binds to this allocation, then taskset selects four of the allocated cores.
srun --cpu-bind=cores bash -c '
    set -euo pipefail
    cd "$CZ_WS4_STAGE"
    cores=$("$CZ_WS4_WORKSPACE/venv/bin/python" -c "import os; print(\",\".join(map(str, sorted(os.sched_getaffinity(0))[:4])))")
    taskset -c "$cores" "$CZ_WS4_WORKSPACE/venv/bin/python" run_bundle.py "$CZ_WS4_STAGE" xcng "$CZ_WS4_WORKSPACE/ncnn-xcng" "$CZ_WS4_STAGE/frame.jpg"
' > "$workspace/bench-xcng.log" 2>&1
cp -a "$stage/results" "$workspace/"
cat /proc/cpuinfo > "$workspace/cpuinfo-xcng.txt"
touch "$workspace/DONE"
