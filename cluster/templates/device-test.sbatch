#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=t4
#SBATCH --nice=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:01:00
#SBATCH --requeue
set -euo pipefail
cd "$COLUMBA_CHECKOUT"
export COLUMBA_RUN_PATH="$HOME/columba/runs/smoke-device-r5-final/seed0"
export COLUMBA_RUN_METADATA='{"recipe_id":"smoke-device-r5-final","seed":0,"epochs_done":0}'
python -m cluster.device_faults check --simulate-once
python -m cluster.requeue start --run-path "$COLUMBA_RUN_PATH"
python - <<'PYTHON'
import json
import os
import platform
import subprocess
from pathlib import Path
root = Path.home() / "columba"
run = Path(os.environ["COLUMBA_RUN_PATH"])
ledger = json.loads((run / "device_faults.json").read_text())
target = os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
fields = subprocess.check_output(["scontrol", "show", "job", target, "-o"], text=True)
proof = {"job": target, "host": platform.node(), "restart_count": os.environ.get("SLURM_RESTART_COUNT"),
         "ledger": ledger, "scontrol": fields,
         "checkpoint_ledger": json.loads((run / "requeue.json").read_text())}
assert int(proof["restart_count"]) == 1
assert proof["checkpoint_ledger"].get("count", 0) == 0
assert ledger["events"][0]["node"] != platform.node().split(".")[0]
(run / "proof.json").write_text(json.dumps(proof, indent=2) + chr(10))
print(json.dumps(proof), flush=True)
PYTHON
