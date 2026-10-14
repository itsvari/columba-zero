#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodelist=xcng0
#SBATCH --nice=1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
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
bash cluster/bootstrap.sh
"$HOME/columba/venvs/aarch64-cpu/bin/python" -m cluster.run_job --recipe "$COLUMBA_RECIPE" --seed "$COLUMBA_SEED"
