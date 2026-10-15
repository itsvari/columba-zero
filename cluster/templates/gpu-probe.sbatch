#!/usr/bin/env bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:01:00
#SBATCH --no-requeue
set -euo pipefail
trap 'rm -rf -- "/tmp/cz-${SLURM_JOB_ID:?}"' EXIT
cd "$COLUMBA_CHECKOUT"
probe_arguments=()
if [[ "${COLUMBA_PROBE_CLEAR_ON_SUCCESS:-}" == true ]]; then
    probe_arguments+=(--clear-on-success)
fi
python -m cluster.gpu_probe "${probe_arguments[@]}"
