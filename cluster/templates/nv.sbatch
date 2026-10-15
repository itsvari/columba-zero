#!/usr/bin/env bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export UV_CACHE_DIR="$HOME/columba/cache/uv"
training_process_identifier=""
readonly signal_exit_status_base=128 requeue_exit_status=75
requeue_requested=false
request_checkpoint() {
    requeue_requested=true
    if [[ -n "$training_process_identifier" ]]; then kill -USR1 "$training_process_identifier" 2>/dev/null || true; fi
}
finish_job() {
    exit_status=$?
    trap - EXIT
    if [[ "$exit_status" != 0 && "$requeue_requested" != true ]]; then
        python -m cluster.job_failure --exit-status "$exit_status" || true
    fi
    rm -rf -- "/tmp/cz-${SLURM_JOB_ID:?}"
    exit "$exit_status"
}
trap request_checkpoint USR1
trap finish_job EXIT
cd "$COLUMBA_CHECKOUT"
python -m cluster.device_faults check --nvidia-only
bash cluster/bootstrap.sh --gpu
environment="$(bash cluster/bootstrap.sh --gpu --print-env)"
"$environment/bin/python" -m cluster.device_faults check
python -m cluster.requeue start --run-path "$COLUMBA_RUN_PATH"
bash cluster/bootstrap.sh --gpu-smoke
run_arguments=(--recipe "$COLUMBA_RECIPE" --seed "$COLUMBA_SEED")
if [[ -n "${COLUMBA_STOP_AFTER_EPOCH:-}" ]]; then
    run_arguments+=(--stop-after-epoch "$COLUMBA_STOP_AFTER_EPOCH")
fi
export COLUMBA_REQUEUE_REQUESTED="$requeue_requested"
set +e
"$environment/bin/python" -m cluster.run_job "${run_arguments[@]}" &
training_process_identifier=$!
wait "$training_process_identifier"
exit_status=$?
if [[ "$exit_status" -ge "$signal_exit_status_base" ]]; then
    wait "$training_process_identifier"
    exit_status=$?
fi
set -e
if [[ "$exit_status" == "$requeue_exit_status" ]]; then
    requeue_requested=true
    python -m cluster.requeue signal --run-path "$COLUMBA_RUN_PATH"
    exit 0
fi
exit "$exit_status"
