#!/usr/bin/env bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export UV_CACHE_DIR="$HOME/columba/cache/uv"
finish_job() {
    exit_status=$?
    trap - EXIT
    if [[ "$exit_status" != 0 ]]; then
        python -m cluster.job_failure --exit-status "$exit_status" || true
    fi
    rm -rf -- "/tmp/cz-${SLURM_JOB_ID:?}"
    exit "$exit_status"
}
trap finish_job EXIT
cd "$COLUMBA_CHECKOUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
bash cluster/bootstrap.sh --gpu-smoke
environment="$(bash cluster/bootstrap.sh --gpu --print-env)"
run_arguments=(--recipe "$COLUMBA_RECIPE" --seed "$COLUMBA_SEED")
if [[ -n "${COLUMBA_STOP_AFTER_EPOCH:-}" ]]; then
    run_arguments+=(--stop-after-epoch "$COLUMBA_STOP_AFTER_EPOCH")
fi
"$environment/bin/python" -m cluster.run_job "${run_arguments[@]}"
