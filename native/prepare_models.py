"""Export square and rectangular C8 packages for native detector verification."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import torch
import yaml

from app.native_detector import MetadataValue

INPUT_SIZES = ((640, 640), (384, 640))


class DetectionHead(Protocol):
    cv2: torch.nn.ModuleList
    cv3: torch.nn.ModuleList
    one2one_cv2: torch.nn.ModuleList | None
    one2one_cv3: torch.nn.ModuleList | None


class DetectionNetwork(Protocol):
    model: Sequence[DetectionHead]


class ExportModel(Protocol):
    model: DetectionNetwork

    def save(self, filename: str) -> None: ...
    def export(
        self, *, format: str, imgsz: tuple[int, int], device: str, quantize: int | None
    ) -> str: ...


def export_package(
    checkpoint: Path,
    destination: Path,
    input_hw: tuple[int, int],
    one_to_one: bool,
    quantisation: str = "fp32",
) -> Path:
    from ultralytics import YOLO

    shutil.copy2(checkpoint, destination)
    model = cast(ExportModel, YOLO(str(destination), task="detect"))
    if one_to_one:
        head = model.model.model[-1]
        if head.one2one_cv2 is None or head.one2one_cv3 is None:
            raise ValueError("checkpoint has no one-to-one head")
        head.cv2, head.cv3 = head.one2one_cv2, head.one2one_cv3
        head.one2one_cv2, head.one2one_cv3 = None, None
        model.save(str(destination))
    exported = Path(
        model.export(
            format="ncnn",
            imgsz=input_hw,
            device="cpu",
            quantize=16 if quantisation == "fp16" else None,
        )
    )
    metadata_path = exported / "metadata.yaml"
    metadata = cast(
        dict[str | int, MetadataValue], yaml.safe_load(metadata_path.read_text("utf-8"))
    )
    single_class_pigeon = metadata.get("names") == {0: "pigeon"}
    metadata["columba"] = {
        "model_id": destination.stem,
        "decode": "yolo26" if one_to_one else "yolo11",
        "input_hw": list(input_hw),
        "letterbox": True,
        "pad_value": 114,
        "color_order": "rgb",
        "normalize": {"mean": [0, 0, 0], "std": [255, 255, 255]},
        "output_layout": "1x(4+nc)xN cxcywh-px sigmoid-scores",
        "nms": {"iou": 0.7, "agnostic": False},
        "max_det": 300,
        "quantization": quantisation,
        "taxonomy": "single" if single_class_pigeon else "coco-verification-only",
        "trigger_classes": ["pigeon"] if single_class_pigeon else [],
    }
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo11", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, nargs=2, action="append", metavar=("H", "W"))
    parser.add_argument("--quantization", choices=("fp32", "fp16"), default="fp32")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arguments.yolo11.resolve(), arguments.output / "reference-yolo11n.pt")
    shutil.copy2(arguments.yolo26.resolve(), arguments.output / "reference-yolo26n.pt")
    torch.set_num_threads(2)
    input_sizes = (
        [tuple(size) for size in arguments.input_size] if arguments.input_size else INPUT_SIZES
    )
    for input_hw in input_sizes:
        for name, checkpoint, one_to_one in (
            ("yolo11n", arguments.yolo11, False),
            ("yolo26n", arguments.yolo26, False),
            ("yolo26n-one2one", arguments.yolo26, True),
        ):
            destination = arguments.output / f"{name}-{input_hw[0]}x{input_hw[1]}.pt"
            exported = export_package(
                checkpoint.resolve(),
                destination.resolve(),
                input_hw,
                one_to_one,
                arguments.quantization,
            )
            print(exported, flush=True)


if __name__ == "__main__":
    main()
