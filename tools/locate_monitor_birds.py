#!/usr/bin/env python
"""Locate reviewed visible-bird boxes using local photograph features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy
from numpy.typing import NDArray

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue
from tools.monitor_stimulus import PHOTOGRAPHS, Photograph

Image = NDArray[numpy.uint8]
Points = NDArray[numpy.float32]
Matrix = NDArray[numpy.float64]
MINIMUM_MATCHES = 16
MINIMUM_INLIERS = 8
MINIMUM_HELD_OUT = 8
MAXIMUM_ERROR_PX = 2.0
MINIMUM_HELD_OUT_FRACTION = 0.8
MAXIMUM_MEDIAN_ERROR_PX = 1.0
MINIMUM_HULL_FRACTION = 0.2
CAMERA_SCALE = 3
RATIO_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class Label:
    photograph: Photograph
    bounds: tuple[float, float, float, float]
    edge_allowance_px: float
    registration_allowance_px: float
    reviewer: str


@dataclass(frozen=True, slots=True)
class CameraCheck:
    image: str
    asset: str
    image_sha256: str
    corners: Points
    maximum_error_px: float
    corner_allowance_px: float
    reviewer: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapping(value: JsonValue, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def text_value(value: JsonValue, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def number(value: JsonValue, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def numbers(value: JsonValue, count: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} must have {count} numbers")
    return tuple(number(item, name) for item in value)


def load_labels(path: Path) -> list[Label]:
    data = mapping(cast(JsonValue, json.loads(path.read_text())), "labels")
    entries = data.get("labels")
    if not isinstance(entries, list) or not entries:
        raise ValueError("labels must be a nonempty list")
    result = []
    seen = set()
    for entry in entries:
        item = mapping(entry, "label")
        asset = text_value(item.get("asset_id"), "asset_id")
        photograph = next((photo for photo in PHOTOGRAPHS if photo.asset_id == asset), None)
        if photograph is None or asset in seen:
            raise ValueError("asset_id must be a unique catalogue asset")
        seen.add(asset)
        if item.get("confirmed") is not True:
            raise ValueError(f"{asset} source label requires explicit confirmation")
        reviewer = text_value(item.get("reviewer"), "reviewer")
        if item.get("source_sha256") != digest(Path(photograph.source)):
            raise ValueError(f"{asset} source hash mismatch")
        crop = photograph.crop
        if numbers(item.get("crop_xywh_px"), 4, "crop") != (
            crop.left_px,
            crop.top_px,
            crop.width_px,
            crop.height_px,
        ):
            raise ValueError(f"{asset} crop mismatch")
        left, top, right, bottom = numbers(item.get("visible_bbox_xyxy_px"), 4, "bbox")
        if not (0 <= left < right <= crop.width_px and 0 <= top < bottom <= crop.height_px):
            raise ValueError(f"{asset} bbox outside source crop")
        edge = number(item.get("edge_allowance_source_px"), "edge allowance")
        registration = number(
            item.get("registration_allowance_camera_px"), "registration allowance"
        )
        if edge < 0 or registration <= 0:
            raise ValueError(
                "edge allowance must be nonnegative and registration allowance positive"
            )
        result.append(Label(photograph, (left, top, right, bottom), edge, registration, reviewer))
    return result


def load_checks(path: Path | None) -> dict[tuple[str, str], CameraCheck]:
    if path is None:
        return {}
    data = mapping(cast(JsonValue, json.loads(path.read_text())), "camera checks")
    entries = data.get("checks")
    if not isinstance(entries, list):
        raise ValueError("camera checks must be a list")
    checks = {}
    for entry in entries:
        item = mapping(entry, "camera check")
        image = text_value(item.get("image"), "image")
        asset = text_value(item.get("asset_id"), "asset_id")
        reviewer = text_value(item.get("reviewer"), "reviewer")
        limit = number(item.get("maximum_error_px"), "maximum camera check error")
        image_sha256 = text_value(item.get("image_sha256"), "camera check image_sha256")
        if len(image_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in image_sha256
        ):
            raise ValueError("camera check image_sha256 must be a lowercase SHA-256 digest")
        allowance_value = item.get("corner_allowance_px", 0)
        allowance = number(allowance_value, "camera corner allowance")
        if allowance < 0:
            raise ValueError("camera corner allowance must be nonnegative")
        raw = item.get("corners_camera_px")
        if not isinstance(raw, list) or len(raw) != 4 or limit <= 0:
            raise ValueError("camera check requires four corners and a positive limit")
        corners = numpy.array(
            [numbers(point, 2, "camera corner") for point in raw], dtype=numpy.float32
        )
        key = (image, asset)
        if key in checks:
            raise ValueError("duplicate camera check")
        checks[key] = CameraCheck(image, asset, image_sha256, corners, limit, allowance, reviewer)
    return checks


def project(matrix: Matrix, points: Points) -> Points:
    return cast(Points, cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2))


def evaluate_geometry(
    source: Points, target: Points, label: Label, image_shape: tuple[int, ...]
) -> dict[str, JsonValue]:
    if len(source) < MINIMUM_MATCHES:
        return {
            "status": "rejected",
            "reason": "insufficient unique matches",
            "matches": len(source),
        }
    fit = numpy.arange(len(source)) % 2 == 0
    matrix, mask = cv2.findHomography(source[fit], target[fit], cv2.RANSAC, MAXIMUM_ERROR_PX)
    if matrix is None or mask is None:
        return {"status": "rejected", "reason": "homography fit failed", "matches": len(source)}
    matrix = cast(Matrix, matrix)
    fit_points = source[fit][mask.ravel().astype(bool)]
    inliers = len(fit_points)
    errors = numpy.linalg.norm(project(matrix, source[~fit]) - target[~fit], axis=1)
    held_count = len(errors)
    held_fraction = float(numpy.mean(errors <= MAXIMUM_ERROR_PX))
    median_error = float(numpy.median(errors))
    if inliers < MINIMUM_INLIERS or not numpy.isfinite(matrix).all():
        return {
            "status": "rejected",
            "reason": "insufficient fit inliers",
            "matches": len(source),
            "fit_inliers": inliers,
        }
    hull = cv2.convexHull(fit_points)
    crop = label.photograph.crop
    hull_fraction = float(cv2.contourArea(hull) / (crop.width_px * crop.height_px))
    left, top, right, bottom = label.bounds
    centre = numpy.array([[(left + right) / 2, (top + bottom) / 2]], dtype=numpy.float32)
    inside_hull = cv2.pointPolygonTest(hull, tuple(float(value) for value in centre[0]), False) >= 0
    corners = numpy.array(
        [[0, 0], [crop.width_px, 0], [crop.width_px, crop.height_px], [0, crop.height_px]],
        dtype=numpy.float32,
    )
    projected = project(matrix, corners)
    height, width = image_shape[:2]
    sides = numpy.linalg.norm(projected - numpy.roll(projected, -1, axis=0), axis=1)
    valid_rect = (
        numpy.isfinite(projected).all()
        and cv2.isContourConvex(projected.reshape(-1, 1, 2))
        and cv2.contourArea(projected, oriented=True) > 100
        and min(sides) >= 10
        and max(sides) <= 4 * min(sides)
        and numpy.all(projected[:, 0] >= 0)
        and numpy.all(projected[:, 0] < width)
        and numpy.all(projected[:, 1] >= 0)
        and numpy.all(projected[:, 1] < height)
    )
    metrics: dict[str, JsonValue] = {
        "matches": len(source),
        "fit_count": int(fit.sum()),
        "fit_inliers": inliers,
        "held_out_count": held_count,
        "held_out_within_2px_fraction": held_fraction,
        "held_out_median_error_px": median_error,
        "source_hull_fraction": hull_fraction,
        "centre_inside_source_hull": bool(inside_hull),
    }
    if (
        inliers < MINIMUM_INLIERS
        or held_count < MINIMUM_HELD_OUT
        or held_fraction < MINIMUM_HELD_OUT_FRACTION
        or median_error >= MAXIMUM_MEDIAN_ERROR_PX
        or hull_fraction < MINIMUM_HULL_FRACTION
        or not inside_hull
        or not valid_rect
    ):
        return {"status": "rejected", "reason": "geometry or quality check failed", **metrics}
    shifted = numpy.array(
        [
            [centre[0, 0] + x, centre[0, 1] + y]
            for x in (-label.edge_allowance_px, label.edge_allowance_px)
            for y in (-label.edge_allowance_px, label.edge_allowance_px)
        ],
        dtype=numpy.float32,
    )
    mapped_centre = project(matrix, centre)[0]
    mapped_shifted = project(matrix, shifted)
    if not numpy.isfinite(mapped_shifted).all() or not numpy.isfinite(mapped_centre).all():
        return {"status": "rejected", "reason": "non-finite uncertainty projection", **metrics}
    lower = mapped_shifted.min(axis=0) - label.registration_allowance_px
    upper = mapped_shifted.max(axis=0) + label.registration_allowance_px
    return {
        "status": "candidate",
        **metrics,
        "source_to_camera_homography": matrix.tolist(),
        "photograph_corners_camera_px": projected.tolist(),
        "visible_bbox_centre_camera_px": mapped_centre.tolist(),
        "centre_allowance_bounds_camera_xyxy_px": [
            float(lower[0]),
            float(lower[1]),
            float(upper[0]),
            float(upper[1]),
        ],
        "edge_allowance_source_px": label.edge_allowance_px,
        "registration_allowance_camera_px": label.registration_allowance_px,
    }


def feature_matches(label: Label, image: Image) -> tuple[Points, Points]:
    photograph = label.photograph
    source_image = cv2.imread(photograph.source, cv2.IMREAD_GRAYSCALE)
    if source_image is None:
        raise ValueError(f"cannot read source photograph {photograph.source}")
    crop = photograph.crop
    source_image = source_image[
        crop.top_px : crop.top_px + crop.height_px, crop.left_px : crop.left_px + crop.width_px
    ]
    detector = cv2.SIFT_create(  # type: ignore[attr-defined]
        nfeatures=4000, contrastThreshold=0.008, edgeThreshold=12
    )
    source_features, source_descriptors = detector.detectAndCompute(source_image, None)
    enlarged = cv2.resize(
        image, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_CUBIC
    )
    camera_features, camera_descriptors = detector.detectAndCompute(enlarged, None)
    if source_descriptors is None or camera_descriptors is None or len(camera_descriptors) < 2:
        return numpy.empty((0, 2), dtype=numpy.float32), numpy.empty((0, 2), dtype=numpy.float32)
    pairs = cv2.BFMatcher().knnMatch(source_descriptors, camera_descriptors, k=2)
    matches = sorted(
        (first for first, second in pairs if first.distance < RATIO_THRESHOLD * second.distance),
        key=lambda match: match.distance,
    )
    unique = []
    used = set()
    for match in matches:
        if match.trainIdx not in used:
            unique.append(match)
            used.add(match.trainIdx)
    source = numpy.array(
        [source_features[match.queryIdx].pt for match in unique], dtype=numpy.float32
    ).reshape(-1, 2)
    target = (
        numpy.array(
            [camera_features[match.trainIdx].pt for match in unique], dtype=numpy.float32
        ).reshape(-1, 2)
        / CAMERA_SCALE
    )
    return source, target.astype(numpy.float32)


def camera_check_precondition(
    check: CameraCheck, image_sha256: str, registration_allowance_px: float
) -> str | None:
    if check.image_sha256 != image_sha256:
        return "camera check image SHA-256 mismatch"
    if check.maximum_error_px > registration_allowance_px:
        return "camera check threshold exceeds registration allowance"
    return None


def apply_camera_check(
    result: dict[str, JsonValue],
    check: CameraCheck,
    image_sha256: str,
    registration_allowance_px: float,
) -> dict[str, JsonValue]:
    reason = camera_check_precondition(check, image_sha256, registration_allowance_px)
    if reason is not None:
        return {**result, "status": "rejected", "reason": reason}
    if result["status"] != "candidate":
        return result
    corners = numpy.asarray(result["photograph_corners_camera_px"], dtype=numpy.float32)
    difference = numpy.abs(corners - check.corners)
    error = float(numpy.max(numpy.linalg.norm(difference, axis=1)))
    worst_case = float(numpy.max(numpy.linalg.norm(difference + check.corner_allowance_px, axis=1)))
    return {
        **result,
        "status": "camera_checked" if worst_case <= check.maximum_error_px else "rejected",
        "camera_check": {
            "reviewer": check.reviewer,
            "image_sha256": check.image_sha256,
            "maximum_error_px": error,
            "raw_maximum_error_px": error,
            "worst_case_maximum_error_px": worst_case,
            "corner_allowance_px": check.corner_allowance_px,
            "allowed_error_px": check.maximum_error_px,
        },
        **(
            {"reason": "independent camera corner check failed"}
            if worst_case > check.maximum_error_px
            else {}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--source-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-checks", type=Path)
    parser.add_argument("--image", action="append", default=[])
    options = parser.parse_args()
    try:
        labels = load_labels(options.source_labels)
        checks = load_checks(options.camera_checks)
        entries = [
            mapping(cast(JsonValue, json.loads(line)), "index entry")
            for line in options.index.read_text().splitlines()
            if line.strip()
        ]
        names = [text_value(entry.get("filename"), "filename") for entry in entries]
        if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
            raise ValueError("index image filenames must be unique and local")
        if not names:
            raise ValueError("index has no images")
        requested = set(options.image)
        if requested - set(names):
            raise ValueError("requested image absent from index")
        selected = [name for name in names if not requested or name in requested]
        if any(
            image not in selected or asset not in {label.photograph.asset_id for label in labels}
            for image, asset in checks
        ):
            raise ValueError("camera check references unselected image or asset")
        results: list[JsonValue] = []
        for name in selected:
            path = options.index.parent / name
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"cannot read {name}")
            image_sha256 = digest(path)
            for label in labels:
                check = checks.get((name, label.photograph.asset_id))
                reason = (
                    camera_check_precondition(check, image_sha256, label.registration_allowance_px)
                    if check
                    else None
                )
                if reason is not None:
                    outcome: dict[str, JsonValue] = {"status": "rejected", "reason": reason}
                else:
                    source, target = feature_matches(label, cast(Image, image))
                    outcome = evaluate_geometry(source, target, label, image.shape)
                    if check is not None:
                        outcome = apply_camera_check(
                            outcome, check, image_sha256, label.registration_allowance_px
                        )
                results.append(
                    {
                        "image": name,
                        "image_sha256": image_sha256,
                        "asset_id": label.photograph.asset_id,
                        "source_sha256": digest(Path(label.photograph.source)),
                        "source_reviewer": label.reviewer,
                        **outcome,
                    }
                )
        report: dict[str, JsonValue] = {
            "index_sha256": digest(options.index),
            "source_labels_sha256": digest(options.source_labels),
            "camera_checks_sha256": digest(options.camera_checks)
            if options.camera_checks
            else None,
            "code_sha256": digest(Path(__file__)),
            "results": results,
        }
        options.output.mkdir(parents=True, exist_ok=False)
        (options.output / "locations.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    except (OSError, ValueError, cv2.error, json.JSONDecodeError) as error:
        parser.exit(2, f"{error}\n")
    print(options.output / "locations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
