#!/usr/bin/env bash
#SBATCH --job-name=cz-ws8-build
#SBATCH --partition=normal
#SBATCH --nodelist=xcng1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

workspace="$HOME/columba/scratch/ws8"
cd "$workspace/repo"
mkdir -p "$workspace/results"
verification_python="$workspace/venv/bin/python"
export PATH="$workspace/venv/bin:$PATH"
export YOLO_CONFIG_DIR="$workspace/settings"
export PYTHONPATH="$PWD/native/build/linux-aarch64:$PWD"

verify_python() {
    if [[ "${1:-}" != --static-only ]]; then
        "$verification_python" -m unittest discover -s tests -v > "$workspace/results/unittest.log" 2>&1
    fi
    "$verification_python" -m ruff check app tests tools cluster training native
    "$verification_python" -m ruff format --check app tests tools cluster training native
    "$verification_python" -m mypy --exclude '^native/build/' app tests tools cluster training native
    date -u > "$workspace/results/DONE"
}

if [[ "${1:-}" == --checks-only || "${1:-}" == --static-only ]]; then
    verify_python "${1:-}"
    exit 0
fi

node_data="/tmp/cz-$SLURM_JOB_ID"
mkdir -p "$node_data"
python -m venv "$workspace/venv"
export PIP_CACHE_DIR="$workspace/pip-cache"
"$verification_python" -m pip install \
    ultralytics==8.4.175 ncnn==1.0.20260526 nanobind==3.1.0 \
    opencv-python==5.0.0.93 torch==2.14.1 torchvision==0.29.1 \
    numpy==2.5.2 PyYAML==6.0.3 cmake ruff mypy types-PyYAML types-requests types-psutil
"$verification_python" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['YOLO_CONFIG_DIR'])
(root / 'Ultralytics').mkdir(parents=True, exist_ok=True)
(root / 'Ultralytics/settings.json').write_text(json.dumps({
    'settings_version': '0.0.6', 'sync': False,
    'datasets_dir': str(root / 'datasets'), 'weights_dir': str(root / 'weights'),
    'runs_dir': str(root / 'runs'),
}))
PY

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
cp -R --no-preserve=all "$workspace/artifacts/parity-images" "$node_data/parity-images"
ln -s "$workspace/artifacts/weights" "$node_data/weights"
BUILD_THREADS=8 bash native/build_dependencies.sh linux-aarch64
cmake -S native -B native/build/linux-aarch64 -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$verification_python" \
    -Dncnn_DIR="$PWD/native/build/ncnn-linux-aarch64/lib/cmake/ncnn" \
    -DOpenCV_DIR="$PWD/native/build/opencv-linux-aarch64/lib/cmake/opencv5"
cmake --build native/build/linux-aarch64 --parallel 8
ctest --test-dir native/build/linux-aarch64 --output-on-failure
export PYTHONPATH="$PWD/native/build/linux-aarch64:$PWD"
"$verification_python" -m unittest discover -s tests -p test_native_detector.py -v
export VERIFICATION_PYTHON="$verification_python"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
uname -a > "$workspace/results/host.txt"
taskset -c 0-3 bash native/run_verification.sh "$node_data" "$workspace/results"
verify_python
