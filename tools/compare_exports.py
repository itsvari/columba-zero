"""Compare CPU PT and NCNN predictions on identical square-preprocessed images."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from app.inference import NMS_INTERSECTION_OVER_UNION, PredictionModel

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


class ExportPrediction(TypedDict):
    label: str
    confidence: float
    box: list[float]


class MatchedDetection(TypedDict):
    pt_index: int
    ncnn_index: int
    iou: float
    confidence_delta: float
    box_delta_px: list[float]


class ImageComparison(TypedDict):
    image: str
    pt_count: int
    ncnn_count: int
    pt: list[ExportPrediction]
    ncnn: list[ExportPrediction]
    matches: list[MatchedDetection]
    passed: bool


def intersection_over_union(left: Sequence[float], right: Sequence[float]) -> float:
    left_minimum_x_px, left_minimum_y_px, left_maximum_x_px, left_maximum_y_px = left
    right_minimum_x_px, right_minimum_y_px, right_maximum_x_px, right_maximum_y_px = right
    intersection_width_px = max(
        0,
        min(left_maximum_x_px, right_maximum_x_px) - max(left_minimum_x_px, right_minimum_x_px),
    )
    intersection_height_px = max(
        0,
        min(left_maximum_y_px, right_maximum_y_px) - max(left_minimum_y_px, right_minimum_y_px),
    )
    intersection_area_px = intersection_width_px * intersection_height_px
    left_area_px = (left_maximum_x_px - left_minimum_x_px) * (left_maximum_y_px - left_minimum_y_px)
    right_area_px = (right_maximum_x_px - right_minimum_x_px) * (
        right_maximum_y_px - right_minimum_y_px
    )
    union_area_px = left_area_px + right_area_px - intersection_area_px
    return intersection_area_px / union_area_px if union_area_px else 0.0


def predictions(
    model: PredictionModel,
    image: Path,
    image_size_px: int,
    confidence: float,
) -> list[ExportPrediction]:
    result = model.predict(
        str(image),
        imgsz=image_size_px,
        conf=confidence,
        device="cpu",
        rect=False,
        iou=NMS_INTERSECTION_OVER_UNION,
        verbose=False,
    )[0]
    return [
        {
            "label": result.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "box": box.xyxy[0].tolist(),
        }
        for box in result.boxes or ()
    ]


def match_predictions(
    expected: Sequence[ExportPrediction],
    actual: Sequence[ExportPrediction],
) -> list[MatchedDetection]:
    unmatched_indices = set(range(len(actual)))
    matches: list[MatchedDetection] = []
    for expected_index, expected_detection in enumerate(expected):
        candidates = [
            (
                intersection_over_union(expected_detection["box"], actual[actual_index]["box"]),
                actual_index,
            )
            for actual_index in unmatched_indices
            if expected_detection["label"] == actual[actual_index]["label"]
        ]
        overlap, matched_index = max(candidates, default=(0.0, -1))
        if matched_index < 0 or overlap == 0:
            continue
        unmatched_indices.remove(matched_index)
        matched_detection = actual[matched_index]
        matches.append(
            {
                "pt_index": expected_index,
                "ncnn_index": matched_index,
                "iou": overlap,
                "confidence_delta": abs(
                    expected_detection["confidence"] - matched_detection["confidence"]
                ),
                "box_delta_px": [
                    actual_coordinate_px - expected_coordinate_px
                    for expected_coordinate_px, actual_coordinate_px in zip(
                        expected_detection["box"],
                        matched_detection["box"],
                        strict=True,
                    )
                ],
            }
        )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt_model")
    parser.add_argument("ncnn_model")
    parser.add_argument("images")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--min-iou", type=float, default=0.8)
    parser.add_argument("--max-confidence-delta", type=float, default=0.05)
    parser.add_argument("--output", default="export_comparison.json")
    arguments = parser.parse_args()
    from ultralytics import YOLO

    checkpoint_model = cast(PredictionModel, YOLO(arguments.pt_model, task="detect"))
    exported_model = cast(PredictionModel, YOLO(arguments.ncnn_model, task="detect"))
    images = sorted(
        image
        for image in Path(arguments.images).iterdir()
        if image.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit("no images to compare")
    comparisons: list[ImageComparison] = []
    for image in images:
        expected = predictions(checkpoint_model, image, arguments.imgsz, arguments.confidence)
        actual = predictions(exported_model, image, arguments.imgsz, arguments.confidence)
        matches = match_predictions(expected, actual)
        passed = len(matches) == len(expected) == len(actual) and all(
            match["iou"] >= arguments.min_iou
            and match["confidence_delta"] <= arguments.max_confidence_delta
            for match in matches
        )
        comparisons.append(
            {
                "image": str(image),
                "pt_count": len(expected),
                "ncnn_count": len(actual),
                "pt": expected,
                "ncnn": actual,
                "matches": matches,
                "passed": passed,
            }
        )
    failures = sum(not comparison["passed"] for comparison in comparisons)
    report = {
        "images": len(images),
        "failures": failures,
        "image_size": arguments.imgsz,
        "confidence": arguments.confidence,
        "device": "cpu",
        "rect": False,
        "iou": NMS_INTERSECTION_OVER_UNION,
        "min_iou": arguments.min_iou,
        "max_confidence_delta": arguments.max_confidence_delta,
        "records": comparisons,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "failures": failures}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
