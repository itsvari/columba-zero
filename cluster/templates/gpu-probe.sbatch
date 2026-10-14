#!/usr/bin/env bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
trap 'rm -rf -- "/tmp/cz-${SLURM_JOB_ID:?}"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$COLUMBA_CHECKOUT"
bash cluster/bootstrap.sh --gpu
environment="$(bash cluster/bootstrap.sh --gpu --print-env)"
"$environment/bin/python" -m cluster.gpu_probe
