"""GPU guards, conservative memory-class batching and bootstrap smoke checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

from app.json_data import JsonValue

GIB_BYTES = 1024**3
REFERENCE_IMAGE_SIZE_PX = 640
MEMORY_CLASS_BATCHES = ((22, 64), (15, 48), (10, 32))
MODEL_MEMORY_FACTORS = {"n": 1, "s": 2, "m": 4, "l": 6, "x": 8}
ALLOWED_GPU_NAMES = ("TITAN V", "TITAN RTX", "T4")


def select_batch(
    memory_gb: float, base_model: str, image_size_px: int, teacher: bool = False
) -> int:
    import re

    model_size = re.search(r"([nsmlx])(?:[-.]|$)", base_model.replace(" ", ""))
    memory_factor = MODEL_MEMORY_FACTORS.get(model_size.group(1) if model_size else "x", 8)
    baseline = next((batch for minimum, batch in MEMORY_CLASS_BATCHES if memory_gb >= minimum), 8)
    area_factor = (image_size_px / REFERENCE_IMAGE_SIZE_PX) ** 2
    teacher_factor = 2 if teacher else 1
    return max(1, int(baseline / memory_factor / area_factor / teacher_factor))


def configure_ultralytics(scratch: Path) -> None:
    settings_root = scratch / "config"
    settings = settings_root / "Ultralytics/settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps(
            {
                "sync": False,
                "datasets_dir": str(scratch / "data"),
                "weights_dir": str(scratch / "weights"),
                "runs_dir": str(scratch / "runs"),
            }
        ),
        encoding="utf-8",
    )
    os.environ["YOLO_CONFIG_DIR"] = str(settings_root)
    os.environ["YOLO_AUTOINSTALL"] = "false"


def describe_hardware(require_gpu: bool) -> dict[str, JsonValue]:
    import torch
    import ultralytics

    report: dict[str, JsonValue] = {
        "host": platform.node(),
        "architecture": platform.machine(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "gpu": None,
        "compute_capability": None,
        "gpu_memory_gb": 0,
    }
    if not require_gpu:
        if platform.machine() != "aarch64" or platform.node().split(".")[0] not in {
            "xcng0",
            "xcng1",
        }:
            raise RuntimeError("arm lane must run on xcng0 or xcng1 (aarch64)")
        return report
    if not torch.cuda.is_available():
        raise RuntimeError("nv lane requires a visible CUDA GPU")
    device = torch.cuda.get_device_properties(0)
    if not any(name in device.name.upper() for name in ALLOWED_GPU_NAMES):
        raise RuntimeError(f"GPU is outside the permitted nv lane: {device.name}")
    capability = torch.cuda.get_device_capability(0)
    device_architecture = f"sm_{capability[0]}{capability[1]}"
    supported = torch.cuda.get_arch_list()
    if device_architecture not in supported:
        raise RuntimeError(f"{device_architecture} missing from wheel architectures: {supported}")
    report.update(
        {
            "gpu": device.name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "gpu_memory_gb": device.total_memory / GIB_BYTES,
            "torch_arch_list": cast(list[JsonValue], supported),
        }
    )
    return report


def smoke_gpu() -> dict[str, JsonValue]:
    import torch

    scratch = Path("/tmp") / f"cz-{os.environ['SLURM_JOB_ID']}" / "bootstrap"
    configure_ultralytics(scratch)
    report = describe_hardware(True)
    from ultralytics import YOLO

    class TrainingNetwork(Protocol):
        args: SimpleNamespace

        def train(self) -> TrainingNetwork: ...
        def to(self, device: str) -> TrainingNetwork: ...
        def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]: ...

    model = YOLO("yolo11n.yaml")
    network = cast(TrainingNetwork, model.model).to("cuda").train()
    from ultralytics.cfg import DEFAULT_CFG_DICT

    network.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    image_size_px = 64
    batch_size = 2
    batch = {
        "img": torch.rand(batch_size, 3, image_size_px, image_size_px, device="cuda"),
        "batch_idx": torch.tensor([0, 1], device="cuda"),
        "cls": torch.zeros(batch_size, 1, device="cuda"),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]] * batch_size, device="cuda"),
    }
    loss, _ = network.loss(batch)
    torch.autograd.backward(loss.sum())
    torch.cuda.synchronize()
    report["forward_backward"] = "passed"
    report["loss"] = float(loss.sum().detach().cpu())
    destination = Path.home() / "columba/logs" / f"bootstrap-{os.environ['SLURM_JOB_ID']}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(smoke_gpu(), indent=2))


if __name__ == "__main__":
    main()
