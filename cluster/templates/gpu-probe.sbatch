#!/usr/bin/env bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:nv:1
#SBATCH --nice=1000
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$COLUMBA_CHECKOUT"
"$HOME/columba/venvs/x86_64-cu126/bin/python" -m cluster.gpu_probe
