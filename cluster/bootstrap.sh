#!/usr/bin/env bash
set -euo pipefail

project_root="${COLUMBA_ROOT:-$HOME/columba}"
case "$project_root" in "$HOME/columba"|"$HOME/columba/"*) ;; *) echo 'COLUMBA_ROOT must be under ~/columba' >&2; exit 2 ;; esac
mkdir -p "$project_root"/{venvs,weights,cache/tmp,logs,data,runs}
export TMPDIR="$project_root/cache/tmp"
export UV_CACHE_DIR="$project_root/cache/uv"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1
export PIP_CACHE_DIR="$project_root/cache/pip"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    settings_scratch="/tmp/cz-$SLURM_JOB_ID/bootstrap"
else
    settings_scratch="$project_root/cache/bootstrap"
fi
export YOLO_CONFIG_DIR="$settings_scratch/config"
export YOLO_AUTOINSTALL=false
python - "$settings_scratch" <<'PYTHON'
import json
import sys
from pathlib import Path
scratch = Path(sys.argv[1])
settings = scratch / "config/Ultralytics/settings.json"
settings.parent.mkdir(parents=True, exist_ok=True)
settings.write_text(json.dumps({"sync": False, "datasets_dir": str(scratch / "data"),
    "weights_dir": str(scratch / "weights"), "runs_dir": str(scratch / "runs")}))
PYTHON
architecture="$(uname -m)"
uv_command="$project_root/bin/$architecture/uv"
if [[ ! -x "$uv_command" ]]; then uv_command="$(command -v uv || true)"; fi
if [[ -z "$uv_command" && -x "$HOME/.local/bin/uv" ]]; then uv_command="$HOME/.local/bin/uv"; fi
if [[ -z "$uv_command" ]] || ! "$uv_command" --version >/dev/null 2>&1; then
    installer="$project_root/cache/install-uv-$architecture.sh"
    curl --fail --location https://astral.sh/uv/install.sh --output "$installer"
    install_directory="$project_root/bin/$architecture"
    if [[ ! -e "$HOME/.local/bin/uv" ]]; then install_directory="$HOME/.local/bin"; fi
    UV_INSTALL_DIR="$install_directory" UV_NO_MODIFY_PATH=1 sh "$installer"
    uv_command="$install_directory/uv"
fi
case "$architecture" in
    x86_64) acceleration=cu126; torch_version=2.8.0; vision_version=0.23.0 ;;
    aarch64) acceleration=cpu; torch_version=2.8.0; vision_version=0.23.0 ;;
    *) echo "unsupported architecture: $architecture" >&2; exit 2 ;;
esac
environment="$project_root/venvs/$architecture-$acceleration"
# a shared lock prevents parallel smoke jobs from changing the same environment.
exec 9>"$project_root/venvs/.$architecture-$acceleration.lock"
flock 9
if [[ ! -x "$environment/bin/python" ]]; then "$uv_command" venv --python python "$environment"; fi
if [[ ! -f "$environment/.columba-ready" ]] || ! "$environment/bin/python" -c 'import pnnx' >/dev/null 2>&1; then
    "$uv_command" pip install --python "$environment/bin/python" "torch==$torch_version" "torchvision==$vision_version" --index-url "https://download.pytorch.org/whl/$acceleration"
    printf 'torch==%s\ntorchvision==%s\nultralytics==8.4.175\nncnn==1.0.20260526\n' "$torch_version" "$vision_version" > "$environment/columba-constraints.txt"
    "$uv_command" pip install --python "$environment/bin/python" --constraint "$environment/columba-constraints.txt" 'ultralytics==8.4.175' 'ncnn==1.0.20260526' 'pnnx==20260526' PyYAML zstandard pycocotools
    "$environment/bin/python" -c 'import torch, ultralytics, ncnn; assert ultralytics.__version__ == "8.4.175"; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())'
    touch "$environment/.columba-ready"
fi
flock -u 9
if [[ "${1:-}" == --gpu-smoke ]]; then
    "$environment/bin/python" -m cluster.hardware --smoke
fi
printf '%s\n' "$environment"
