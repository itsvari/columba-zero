#!/usr/bin/env python
"""Evaluate C1 labels and framework-independent C3 detections with COCO AP."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue

Box: TypeAlias = tuple[float, float, float, float]
InputShape: TypeAlias = tuple[int, int]
PRECISION_TARGET = 0.90
MATCH_IOU = 0.50
SMALL_SHORT_SIDE_PX = 32
MAX_DETECTIONS = 300
DEFAULT_THRESHOLD = 0.5
NO_DETECTION_THRESHOLD = math.nextafter(1.0, math.inf)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
DEFAULT_CLASSES = Path(__file__).resolve().parents[1] / "training/classes.yaml"

METRIC_HELP = """
AP uses pycocotools: 101 recall points, IoU 0.50:0.05:0.95, all areas,
maxDets=300. AP50 is at IoU 0.50. Matching for operating metrics is greedy,
per image/class, descending score, one-to-one, IoU >= 0.50. Equal-score
predictions use file order within an image and enter a threshold together.
recall_at_p90 is the highest pigeon recall at a score threshold with overall
pigeon precision >= 0.90. small_recall_at_p90 uses the same eligible thresholds
and matching, counting only matched small pigeon ground truths; large matches
still contribute to overall precision. Small means GT short side < 32 pixels
multiplied by min(input_height/image_height, input_width/image_width).
Thresholds include predictions with score >= threshold. Selection chooses the
lowest observed score meeting precision >= 0.90, on val only. Test selection
is refused. If unattainable, selection writes a finite threshold just above 1,
attainable=false, and recall_at_p90=0. --threshold-from reuses a selected val
threshold, including this no-detection sentinel. Default threshold is 0.5,
explicitly recorded as default; supply a selected val file for test comparisons.
No ground truths: AP=null, recall=0. No predictions: precision=0, recall=0.
Empty/confuser FAR counts images with >=1 pigeon prediction at the operating
threshold, divided by images with no birds / only non-pigeon birds respectively,
as defined by manifest classes_present, even for unlabelled single-taxonomy
confusers. A missing denominator gives null. Missing predictions are fine;
unknown image/category IDs, malformed boxes and non-finite numbers are errors.
by_style/by_source recompute metrics within each group at the shared operating
threshold, with each group's own PR curve. Counts include manifest group IDs.
Without manifest.jsonl, IDs are unique image stems; style/source are unknown
and bird presence is inferred from labels, so unlabelled confusers are unknown.
"""


@dataclass(frozen=True)
class Annotation:
    category_id: int
    bbox: Box


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    width_px: int
    height_px: int
    style: str
    source: str
    group: str
    classes_present: tuple[str, ...]
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True)
class Dataset:
    name: str
    names: tuple[str, ...]
    images: tuple[ImageRecord, ...]
    id_source: str


@dataclass(frozen=True)
class Prediction:
    image_id: str
    category_id: int
    bbox: Box
    score: float


@dataclass(frozen=True)
class Match:
    score: float
    matched: bool
    small: bool


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    precision: float
    recall: float
    small_recall: float


def read_json(path: Path) -> JsonValue:
    return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))


def mapping(value: object, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must have string keys")
    return cast(dict[str, JsonValue], value)


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def string_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return tuple(string(item, name) for item in value)


def input_shape(value: object) -> InputShape:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("input_hw must contain height and width")
    return integer(value[0], "input height", 1), integer(value[1], "input width", 1)


def class_names(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        indexed = {int(key): string(name, "class name") for key, name in value.items()}
        if set(indexed) != set(range(len(indexed))):
            raise ValueError("class IDs must be contiguous from zero")
        names = tuple(indexed[index] for index in range(len(indexed)))
    else:
        names = string_sequence(value, "names")
    if not names or names[0] != "pigeon" or len(set(names)) != len(names):
        raise ValueError("names must be unique with pigeon at index zero")
    return names


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        position = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"image must be under an images directory: {image_path}") from error
    parts[position] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(
    path: Path, width_px: int, height_px: int, names: tuple[str, ...]
) -> tuple[Annotation, ...]:
    if not path.exists():
        return ()
    annotations: list[Annotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected class cx cy w h")
        category_value, centre_x, centre_y, width, height = (float(field) for field in fields)
        coordinates = (centre_x, centre_y, width, height)
        if not math.isfinite(category_value) or not category_value.is_integer():
            raise ValueError(f"{path}:{line_number}: invalid class ID")
        category_id = int(category_value)
        if category_id not in range(len(names)):
            raise ValueError(f"{path}:{line_number}: unknown class ID {category_id}")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: coordinates must be finite and normalised")
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: label box must have positive area")
        annotations.append(
            Annotation(
                category_id,
                (
                    (centre_x - width / 2) * width_px,
                    (centre_y - height / 2) * height_px,
                    width * width_px,
                    height * height_px,
                ),
            )
        )
    return tuple(annotations)


def manifest_images(root: Path, split: str, names: tuple[str, ...]) -> tuple[ImageRecord, ...]:
    images: list[ImageRecord] = []
    all_ids: set[str] = set()
    all_paths: set[Path] = set()
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = mapping(json.loads(line), "manifest entry")
        image_id = string(record.get("id"), "manifest id")
        image_path = (root / string(record.get("file"), "manifest file")).resolve()
        if not image_path.is_relative_to(root.resolve()):
            raise ValueError("manifest file must stay within the dataset")
        if image_id in all_ids or image_path in all_paths:
            raise ValueError(f"duplicate manifest id or file: {image_id}")
        all_ids.add(image_id)
        all_paths.add(image_path)
        if record.get("split") != split:
            continue
        if not image_path.is_file():
            raise ValueError(f"missing dataset image: {image_path}")
        width_px = integer(record.get("width"), "image width", 1)
        height_px = integer(record.get("height"), "image height", 1)
        present = string_sequence(record.get("classes_present"), "classes_present")
        labels = label_path(image_path)
        if not labels.is_file():
            raise ValueError(f"missing C1 label file: {labels}")
        annotations = read_labels(labels, width_px, height_px, names)
        if any(names[annotation.category_id] not in present for annotation in annotations):
            raise ValueError(f"label class absent from classes_present: {image_id}")
        images.append(
            ImageRecord(
                image_id,
                image_path,
                width_px,
                height_px,
                string(record.get("style"), "style"),
                string(record.get("source"), "source"),
                string(record.get("group"), "group"),
                present,
                annotations,
            )
        )
    return tuple(images)


def plain_images(
    root: Path, configuration: Mapping[str, JsonValue], split: str, names: tuple[str, ...]
) -> tuple[ImageRecord, ...]:
    import cv2

    base = (root / str(configuration.get("path", "."))).resolve()
    entry = configuration.get(split, f"images/{split}")
    entries = entry if isinstance(entry, list) else [entry]
    paths: list[Path] = []
    for item in entries:
        location = (base / string(item, f"{split} path")).resolve()
        if location.is_dir():
            paths.extend(
                sorted(
                    path for path in location.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif location.suffix == ".txt":
            paths.extend(
                (location.parent / line.strip()).resolve()
                for line in location.read_text().splitlines()
                if line.strip()
            )
        else:
            raise ValueError(f"split path must be an image directory or image list: {location}")
    images: list[ImageRecord] = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"cannot read image: {path}")
        height_px, width_px = frame.shape[:2]
        annotations = read_labels(label_path(path), width_px, height_px, names)
        present = tuple(sorted({names[annotation.category_id] for annotation in annotations}))
        images.append(
            ImageRecord(
                path.stem,
                path,
                width_px,
                height_px,
                "unknown",
                "unknown",
                path.stem,
                present,
                annotations,
            )
        )
    if len({image.image_id for image in images}) != len(images):
        raise ValueError("image stems must be unique without a manifest")
    return tuple(images)


def load_dataset(dataset: Path, split: str, classes: Path | None = DEFAULT_CLASSES) -> Dataset:
    configuration = mapping(yaml.safe_load((dataset / "data.yaml").read_text()), "data.yaml")
    names = class_names(configuration.get("names"))
    if classes is not None:
        taxonomy = mapping(yaml.safe_load(classes.read_text()), "classes.yaml")
        taxonomies = mapping(taxonomy.get("taxonomies"), "taxonomies")
        allowed = [
            class_names(mapping(value, "taxonomy").get("names")) for value in taxonomies.values()
        ]
        if names not in allowed:
            raise ValueError("data.yaml names do not match a classes.yaml taxonomy")
    has_manifest = (dataset / "manifest.jsonl").exists()
    images = (
        manifest_images(dataset, split, names)
        if has_manifest
        else plain_images(dataset, configuration, split, names)
    )
    if not images:
        raise ValueError(f"dataset split {split!r} has no images")
    summary = (
        mapping(read_json(dataset / "DATASET.json"), "DATASET.json")
        if (dataset / "DATASET.json").exists()
        else {}
    )
    return Dataset(
        str(summary.get("name", dataset.name)),
        names,
        images,
        "manifest" if has_manifest else "image_stem",
    )


def read_predictions(path: Path, dataset: Dataset) -> tuple[Prediction, ...]:
    values = read_json(path)
    if not isinstance(values, list):
        raise ValueError("predictions must be a COCO results list")
    image_ids = {image.image_id for image in dataset.images}
    predictions: list[Prediction] = []
    for value in values:
        record = mapping(value, "prediction")
        image_id = string(record.get("image_id"), "image_id")
        category_id = integer(record.get("category_id"), "category_id")
        if image_id not in image_ids:
            raise ValueError(f"prediction image not in dataset split: {image_id}")
        if category_id not in range(len(dataset.names)):
            raise ValueError(f"unknown prediction category_id: {category_id}")
        coordinates = record.get("bbox")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError("bbox must contain x y width height in original pixels")
        bbox = cast(Box, tuple(number(coordinate, "bbox coordinate") for coordinate in coordinates))
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("prediction bbox must have positive area")
        score = number(record.get("score"), "score")
        if not 0 <= score <= 1:
            raise ValueError("prediction score must be between zero and one")
        predictions.append(Prediction(image_id, category_id, bbox, score))
    counts: dict[str, int] = defaultdict(int)
    for prediction in predictions:
        counts[prediction.image_id] += 1
    if any(count > MAX_DETECTIONS for count in counts.values()):
        raise ValueError("C3 predictions exceed 300 boxes per image")
    return tuple(predictions)


def sidecar_path(predictions: Path) -> Path:
    return predictions.with_suffix(".meta.json")


def overlap(left: Box, right: Box) -> float:
    width_px = max(0.0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0]))
    height_px = max(0.0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1]))
    intersection_px = width_px * height_px
    return intersection_px / (left[2] * left[3] + right[2] * right[3] - intersection_px)


def is_small(annotation: Annotation, image: ImageRecord, input_hw: InputShape) -> bool:
    scale = min(input_hw[0] / image.height_px, input_hw[1] / image.width_px)
    return min(annotation.bbox[2:]) * scale < SMALL_SHORT_SIDE_PX


def match_image(
    image: ImageRecord, predictions: Sequence[Prediction], input_hw: InputShape
) -> tuple[Match, ...]:
    annotations = tuple(
        annotation for annotation in image.annotations if annotation.category_id == 0
    )
    unmatched = set(range(len(annotations)))
    matches: list[Match] = []
    for prediction in sorted(predictions, key=lambda prediction: -prediction.score):
        candidates = [
            (overlap(prediction.bbox, annotations[index].bbox), index) for index in unmatched
        ]
        best_overlap, index = max(candidates, default=(0.0, -1))
        matched = best_overlap >= MATCH_IOU
        if matched:
            unmatched.remove(index)
        matches.append(
            Match(
                prediction.score, matched, matched and is_small(annotations[index], image, input_hw)
            )
        )
    return tuple(matches)


def curve(matches: Sequence[Match], pigeon_boxes: int, small_boxes: int) -> tuple[CurvePoint, ...]:
    grouped: dict[float, list[Match]] = defaultdict(list)
    for match in matches:
        grouped[match.score].append(match)
    true_positives = small_positives = prediction_count = 0
    points: list[CurvePoint] = []
    for score, items in sorted(grouped.items(), reverse=True):
        prediction_count += len(items)
        true_positives += sum(item.matched for item in items)
        small_positives += sum(item.small for item in items)
        points.append(
            CurvePoint(
                score,
                true_positives / prediction_count,
                true_positives / pigeon_boxes if pigeon_boxes else 0.0,
                small_positives / small_boxes if small_boxes else 0.0,
            )
        )
    return tuple(points)


def coco_ap(
    dataset: Dataset, predictions: Sequence[Prediction]
) -> dict[str, tuple[float | None, float | None]]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = {image.image_id: index for index, image in enumerate(dataset.images, start=1)}
    annotations = [
        {
            "id": index,
            "image_id": image_ids[image.image_id],
            "category_id": annotation.category_id,
            "bbox": list(annotation.bbox),
            "area": annotation.bbox[2] * annotation.bbox[3],
            "iscrowd": 0,
        }
        for index, (image, annotation) in enumerate(
            ((image, annotation) for image in dataset.images for annotation in image.annotations),
            start=1,
        )
    ]
    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {},
        "images": [
            {"id": image_ids[image.image_id], "width": image.width_px, "height": image.height_px}
            for image in dataset.images
        ],
        "categories": [{"id": index, "name": name} for index, name in enumerate(dataset.names)],
        "annotations": annotations,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        if predictions:
            detections = ground_truth.loadRes(
                [
                    {
                        "image_id": image_ids[prediction.image_id],
                        "category_id": prediction.category_id,
                        "bbox": list(prediction.bbox),
                        "score": prediction.score,
                    }
                    for prediction in predictions
                ]
            )
        else:
            detections = COCO()
            detections.dataset = {**ground_truth.dataset, "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = np.asarray(evaluator.eval["precision"], dtype=np.float64)
    result: dict[str, tuple[float | None, float | None]] = {}
    for index, name in enumerate(dataset.names):
        ap50 = precision[0, :, index, 0, -1]
        all_iou = precision[:, :, index, 0, -1]
        result[name] = (
            float(ap50[ap50 >= 0].mean()) if (ap50 >= 0).any() else None,
            float(all_iou[all_iou >= 0].mean()) if (all_iou >= 0).any() else None,
        )
    return result


def metric_group(
    dataset: Dataset, predictions: Sequence[Prediction], input_hw: InputShape, threshold: float
) -> dict[str, JsonValue]:
    pigeons: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.category_id == 0:
            pigeons[prediction.image_id].append(prediction)
    annotations = [
        (image, annotation)
        for image in dataset.images
        for annotation in image.annotations
        if annotation.category_id == 0
    ]
    small_boxes = sum(is_small(annotation, image, input_hw) for image, annotation in annotations)
    matches = tuple(
        match
        for image in dataset.images
        for match in match_image(image, pigeons[image.image_id], input_hw)
    )
    points = curve(matches, len(annotations), small_boxes)
    eligible = [point for point in points if point.precision >= PRECISION_TARGET]
    at_threshold = [match for match in matches if match.score >= threshold]
    true_positives = sum(match.matched for match in at_threshold)
    empty = tuple(image for image in dataset.images if not image.classes_present)
    confusers = tuple(
        image
        for image in dataset.images
        if image.classes_present and "pigeon" not in image.classes_present
    )
    alarms = {
        image_id
        for image_id, detections in pigeons.items()
        if any(prediction.score >= threshold for prediction in detections)
    }
    ap = coco_ap(dataset, predictions)
    return {
        "pigeon": {
            "ap50": ap["pigeon"][0],
            "ap50_95": ap["pigeon"][1],
            "recall_at_p90": max((point.recall for point in eligible), default=0.0),
            "small_recall_at_p90": max((point.small_recall for point in eligible), default=0.0),
            "precision_at_threshold": true_positives / len(at_threshold) if at_threshold else 0.0,
            "recall_at_threshold": true_positives / len(annotations) if annotations else 0.0,
        },
        "pigeon_recall_at_p90": max((point.recall for point in eligible), default=0.0),
        "empty_false_alarm_rate": sum(image.image_id in alarms for image in empty) / len(empty)
        if empty
        else None,
        "confuser_false_alarm_rate": sum(image.image_id in alarms for image in confusers)
        / len(confusers)
        if confusers
        else None,
        "per_class_ap50": {name: values[0] for name, values in ap.items()},
        "counts": {
            "images": len(dataset.images),
            "empty_images": len(empty),
            "confuser_images": len(confusers),
            "pigeon_boxes": len(annotations),
            "small_pigeon_boxes": small_boxes,
            "groups": len({image.group for image in dataset.images}),
        },
    }


def select_operating_threshold(
    dataset: Dataset, predictions: Sequence[Prediction], input_hw: InputShape
) -> dict[str, JsonValue]:
    pigeons: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.category_id == 0:
            pigeons[prediction.image_id].append(prediction)
    matches = tuple(
        match
        for image in dataset.images
        for match in match_image(image, pigeons[image.image_id], input_hw)
    )
    pigeon_boxes = sum(
        annotation.category_id == 0 for image in dataset.images for annotation in image.annotations
    )
    eligible = [
        point.threshold
        for point in curve(matches, pigeon_boxes, 0)
        if point.precision >= PRECISION_TARGET
    ]
    return {
        "value": min(eligible, default=NO_DETECTION_THRESHOLD),
        "source": "selected",
        "attainable": bool(eligible),
        "precision_target": PRECISION_TARGET,
        "split": "val",
    }


def resolve_threshold(
    dataset: Dataset,
    predictions: Sequence[Prediction],
    input_hw: InputShape,
    split: str,
    select_threshold: bool,
    threshold_from: Path | None,
    threshold: float | None,
) -> dict[str, JsonValue]:
    if sum((select_threshold, threshold_from is not None, threshold is not None)) > 1:
        raise ValueError("threshold selection options are mutually exclusive")
    if select_threshold:
        if split != "val":
            raise ValueError("threshold selection is only permitted on val, never test or train")
        return select_operating_threshold(dataset, predictions, input_hw)
    if threshold_from is not None:
        report = mapping(read_json(threshold_from), "threshold metrics")
        selected = mapping(report.get("operating_threshold"), "operating_threshold")
        if (
            report.get("split", selected.get("split")) != "val"
            or selected.get("source") != "selected"
        ):
            raise ValueError("--threshold-from requires metrics selected on val")
        value = number(selected.get("value"), "operating threshold")
        source = str(threshold_from)
    else:
        value = DEFAULT_THRESHOLD if threshold is None else number(threshold, "threshold")
        source = "default" if threshold is None else "explicit"
    if not 0 <= value <= NO_DETECTION_THRESHOLD:
        raise ValueError("threshold must be in [0,1], or the selected no-detection sentinel")
    return {"value": value, "source": source}


def evaluate(
    dataset: str | Path,
    split: str,
    predictions: str | Path,
    *,
    input_hw: InputShape | None = None,
    classes: str | Path | None = DEFAULT_CLASSES,
    select_threshold: bool = False,
    threshold_from: str | Path | None = None,
    threshold: float | None = None,
) -> dict[str, JsonValue]:
    """Return C4 metrics; write the result as JSON for reuse by cluster jobs."""
    prediction_path = Path(predictions)
    loaded = load_dataset(Path(dataset), split, Path(classes) if classes is not None else None)
    detections = read_predictions(prediction_path, loaded)
    if input_hw is None:
        metadata = mapping(read_json(sidecar_path(prediction_path)), "prediction meta")
        input_hw = input_shape(metadata.get("input_hw"))
    else:
        input_hw = input_shape(input_hw)
    operating = resolve_threshold(
        loaded,
        detections,
        input_hw,
        split,
        select_threshold,
        Path(threshold_from) if threshold_from is not None else None,
        threshold,
    )
    operating_value = number(operating["value"], "operating threshold")
    report = metric_group(loaded, detections, input_hw, operating_value)
    report.update(
        {
            "dataset": loaded.name,
            "split": split,
            "predictions": str(prediction_path),
            "input_hw": list(input_hw),
            "operating_threshold": operating,
            "image_id_source": loaded.id_source,
            "label": "measured",
            "host": platform.node(),
            "definitions": METRIC_HELP.strip(),
        }
    )
    for field in ("style", "source"):
        breakdown: dict[str, JsonValue] = {}
        for value in sorted({getattr(image, field) for image in loaded.images}):
            images = tuple(image for image in loaded.images if getattr(image, field) == value)
            image_ids = {image.image_id for image in images}
            subset = Dataset(loaded.name, loaded.names, images, loaded.id_source)
            breakdown[value] = metric_group(
                subset,
                tuple(prediction for prediction in detections if prediction.image_id in image_ids),
                input_hw,
                operating_value,
            )
        report[f"by_{field}"] = breakdown
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=METRIC_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--input-hw", nargs=2, type=int, metavar=("H", "W"))
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--select-threshold", action="store_true")
    selection.add_argument("--threshold-from", type=Path)
    selection.add_argument("--threshold", type=float)
    arguments = parser.parse_args()
    try:
        report = evaluate(
            arguments.dataset,
            arguments.split,
            arguments.predictions,
            input_hw=input_shape(arguments.input_hw) if arguments.input_hw else None,
            classes=arguments.classes,
            select_threshold=arguments.select_threshold,
            threshold_from=arguments.threshold_from,
            threshold=arguments.threshold,
        )
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
