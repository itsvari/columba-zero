#!/usr/bin/env python
"""Export a checkpoint to a rectangular C8 NCNN package without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import cast

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue
from tools.evaluate_predictions import class_names, input_shape, mapping
from tools.predict_to_coco import configure_ultralytics

MODEL_STRIDE_PX = 32
THREAD_COUNT = 4


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_package(
    checkpoint: Path,
    destination: Path,
    input_hw: tuple[int, int],
    *,
    model_id: str,
    decode: str = "yolo11",
    quantisation: str = "fp16",
    taxonomy: str = "single",
) -> Path:
    if decode not in ("yolo11", "yolo26") or quantisation not in ("fp16", "fp32"):
        raise ValueError("export supports yolo11/yolo26 and fp16/fp32; use quantize_ncnn for int8")
    if any(size <= 0 or size % MODEL_STRIDE_PX for size in input_hw):
        raise ValueError("input dimensions must be positive multiples of 32")
    checkpoint = checkpoint.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("outputs/ws7/ultralytics").resolve()))
    configure_ultralytics()
    import torch
    from ultralytics import YOLO

    from native.prepare_models import ExportModel

    torch.set_num_threads(THREAD_COUNT)
    source_sha256 = fingerprint(checkpoint)
    with tempfile.TemporaryDirectory(prefix="cz-export-", dir=destination.parent) as temporary:
        copied = Path(temporary) / "checkpoint.pt"
        shutil.copy2(checkpoint, copied)
        model = cast(ExportModel, YOLO(str(copied), task="detect"))
        if decode == "yolo26":
            head = model.model.model[-1]
            if head.one2one_cv2 is None or head.one2one_cv3 is None:
                raise ValueError("checkpoint has no YOLO26 one-to-one head")
            head.cv2, head.cv3 = head.one2one_cv2, head.one2one_cv3
            head.one2one_cv2, head.one2one_cv3 = None, None
            model.save(str(copied))
        exported = Path(
            model.export(
                format="ncnn",
                imgsz=input_hw,
                device="cpu",
                quantize=16 if quantisation == "fp16" else None,
            )
        )
        metadata_path = exported / "metadata.yaml"
        metadata = mapping(yaml.safe_load(metadata_path.read_text()), "metadata")
        names = class_names(metadata.get("names"))
        classes = mapping(yaml.safe_load(Path("training/classes.yaml").read_text()), "classes")
        taxonomies = mapping(classes["taxonomies"], "taxonomies")
        expected = class_names(mapping(taxonomies[taxonomy], "taxonomy")["names"])
        if names != expected:
            raise ValueError(f"checkpoint classes {names} differ from {taxonomy}: {expected}")
        columba: dict[str, JsonValue] = {
            "model_id": model_id,
            "decode": decode,
            "input_hw": list(input_hw),
            "letterbox": True,
            "pad_value": 114,
            "color_order": "rgb",
            "normalize": {"mean": [0, 0, 0], "std": [255, 255, 255]},
            "output_layout": "1x(4+nc)xN cxcywh-px sigmoid-scores",
            "max_det": 300,
            "quantization": quantisation,
            "taxonomy": taxonomy,
            "trigger_classes": ["pigeon"],
        }
        if decode == "yolo11":
            columba["nms"] = {"iou": 0.7, "agnostic": False}
        metadata["columba"] = columba
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
        (exported / "export.json").write_text(
            json.dumps(
                {
                    "source": str(checkpoint),
                    "source_sha256": source_sha256,
                    "decode": decode,
                    "quantization": quantisation,
                    "transformation": "one-to-one cv2/cv3 head swap"
                    if decode == "yolo26"
                    else "standard",
                },
                indent=2,
            )
            + "\n"
        )
        shutil.move(str(exported), destination)
    if fingerprint(checkpoint) != source_sha256:
        raise RuntimeError("source checkpoint changed during export")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640])
    parser.add_argument("--half", action="store_true", help="legacy alias for --quantization fp16")
    parser.add_argument("--quantization", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--decode", choices=("yolo11", "yolo26"), default="yolo11")
    parser.add_argument("--taxonomy", choices=("single", "species"), default="single")
    parser.add_argument("--model-id")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    dimensions = arguments.imgsz * 2 if len(arguments.imgsz) == 1 else arguments.imgsz
    try:
        print(
            export_package(
                arguments.model,
                arguments.output or arguments.model.with_name(arguments.model.stem + "_ncnn_model"),
                input_shape(dimensions),
                model_id=arguments.model_id or arguments.model.stem,
                decode=arguments.decode,
                taxonomy=arguments.taxonomy,
                quantisation="fp16" if arguments.half else arguments.quantization,
            )
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
