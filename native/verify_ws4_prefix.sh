#!/usr/bin/env bash
#SBATCH --job-name=cz-ws8-build
#SBATCH --partition=normal
#SBATCH --nodelist=xcng1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

workspace="$HOME/columba/scratch/ws8"
cd "$workspace/repo"
verification_python="$workspace/venv/bin/python"
export PATH="$workspace/venv/bin:$PATH"
export YOLO_CONFIG_DIR="$workspace/settings"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export PIP_CACHE_DIR="$workspace/pip-cache"
"$verification_python" -m pip freeze > "$workspace/constraints.txt"
"$verification_python" -m pip install -c "$workspace/constraints.txt" pycocotools imagehash zstandard
cmake -S native -B native/build/linux-ws4 -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$verification_python" \
    -Dncnn_DIR="$HOME/columba/scratch/ws4/ncnn-xcng/lib/cmake/ncnn" \
    -DOpenCV_DIR="$PWD/native/build/opencv-linux-aarch64/lib/cmake/opencv5"
cmake --build native/build/linux-ws4 --parallel 4
ctest --test-dir native/build/linux-ws4 --output-on-failure
export PYTHONPATH="$PWD/native/build/linux-ws4:$PWD"
taskset -c 0-3 "$verification_python" -m unittest discover -s tests -v \
    > "$workspace/results/unittest-after-ws9.log" 2>&1
"$verification_python" -m ruff check app tests tools cluster training native
"$verification_python" -m ruff format --check app tests tools cluster training native
"$verification_python" -m mypy --exclude '^native/build/' app tests tools cluster training native
date -u > "$workspace/results/DONE-after-ws9"
