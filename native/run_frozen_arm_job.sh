#!/usr/bin/env bash
#SBATCH --job-name=cz-ws8-parity
#SBATCH --partition=normal
#SBATCH --nodelist=xcng1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail

workspace="$HOME/columba/scratch/ws8"
round_directory="$workspace/round3"
cd "$round_directory/repo"
verification_python="$workspace/venv/bin/python"
export PATH="$workspace/venv/bin:$PATH"
export YOLO_CONFIG_DIR="$workspace/settings"
export PYTHONPATH="$PWD/native/build/linux-aarch64:$PWD"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
report_directory="$round_directory/results"
mkdir -p "$report_directory"
node_data="/tmp/cz-$SLURM_JOB_ID"
mkdir -p "$node_data"

cleanup() {
    "$verification_python" - "$node_data" <<'PY'
import shutil
import sys
from pathlib import Path
path = Path(sys.argv[1])
if path.parent == Path('/tmp') and path.name.startswith('cz-'):
    shutil.rmtree(path)
PY
}
trap cleanup EXIT

(
    cd "$HOME/columba/data"
    sha256sum --check columba-test-v2.tar.zst.sha256
)
tar --zstd -xf "$HOME/columba/data/columba-test-v2.tar.zst" -C "$node_data"
tar -xf "$round_directory/old-split-proxy.tar" -C "$node_data"
"$verification_python" -m tools.build_dataset verify-freeze \
    "$node_data/columba-test-v2" data_sources/columba-test-v2.sha256 \
    > "$report_directory/verify-freeze.log"
cmake -S native -B native/build/linux-aarch64 -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$verification_python" \
    -Dncnn_DIR="$HOME/columba/scratch/ws4/ncnn-xcng/lib/cmake/ncnn" \
    -DOpenCV_DIR="$workspace/repo/native/build/opencv-linux-aarch64/lib/cmake/opencv5"
cmake --build native/build/linux-aarch64 --parallel 4
ctest --test-dir native/build/linux-aarch64 --output-on-failure
"$verification_python" -m unittest tests.test_native_detector tests.test_runtime -v \
    > "$report_directory/native-runtime.log" 2>&1
export VERIFICATION_PYTHON="$verification_python"
taskset -c 0-3 bash native/run_frozen_parity.sh "$round_directory/artifacts" \
    "$node_data/columba-test-v2" "$node_data/old-split-proxy/images" "$report_directory"
date -u > "$report_directory/DONE"
