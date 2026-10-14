"""Read-only CUDA availability probe inside a permitted nv Slurm allocation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from app.json_data import JsonValue
from cluster.artifacts import atomic_json


def main() -> None:
    if not os.environ.get("SLURM_JOB_NAME", "").startswith("cz-"):
        raise ValueError("probe requires a cz- Slurm job")
    completed = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    print(completed.stdout, flush=True)
    print(completed.stderr, flush=True)
    import torch

    available = torch.cuda.is_available()
    report: dict[str, JsonValue] = {
        "host": platform.node(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "label": "measured",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_exit_status": completed.returncode,
        "nvidia_smi": completed.stdout,
        "cuda_available": available,
        "status": "probe_complete",
    }
    if available:
        report["gpu"] = torch.cuda.get_device_name(0)
        report["torch_arch_list"] = list(torch.cuda.get_arch_list())
    destination = Path(os.environ["COLUMBA_RUN_PATH"]).resolve()
    if not destination.is_relative_to((Path.home() / "columba").resolve()):
        raise ValueError("probe reports must stay inside ~/columba")
    atomic_json(destination / "run.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
