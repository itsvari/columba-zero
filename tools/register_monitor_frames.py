#!/usr/bin/env python
"""Fit and independently check offline screen-to-camera registration candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

Json: TypeAlias = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
Point: TypeAlias = tuple[float, float]
Matrix: TypeAlias = NDArray[np.float64]
MARKER_IDS = ("top-left", "top-right", "bottom-right", "bottom-left")
MARKER_CENTRES_DESIGN_PX: tuple[Point, ...] = ((56, 56), (1544, 56), (1544, 844), (56, 844))
MARKER_SIZE_DESIGN_PX = 64.0
MARKER_MARGIN_DESIGN_PX = 24.0
TARGET_REGION_DESIGN_PX = (160.0, 180.0, 1280.0, 540.0)
MARKER_COLOUR_BGR = (255, 255, 0)
OBSERVED_COLOUR_BGR = (0, 255, 0)
PREDICTED_COLOUR_BGR = (0, 0, 255)
STATUS_COLOUR_BGR = (0, 255, 255)
MARKER_RADIUS_PX = 5
CHECK_RADIUS_PX = 4
CROSS_SIZE_PX = 8
ANNOTATION_LINE_WIDTH_PX = 1
ANNOTATION_FONT_SCALE = 0.35
STATUS_FONT_SCALE = 0.5
STATUS_LEFT_PX = 10
STATUS_BOTTOM_MARGIN_PX = 12
MINIMUM_QUADRILATERAL_AREA_PX2 = 16.0
MINIMUM_CHECK_COUNT = 5
CENTRE_TOLERANCE_FRACTION = 0.2
GEOMETRY_TOLERANCE_PX = 0.05
DENOMINATOR_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Check:
    name: str
    screen: Point
    camera: Point


def mapping(value: Json, name: str) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def number(value: Json, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def point(value: Json, name: str) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [x, y]")
    return number(value[0], name), number(value[1], name)


def rectangle(value: Json, name: str) -> tuple[float, float, float, float]:
    data = mapping(value, name)
    left = number(data.get("left_px"), name)
    top = number(data.get("top_px"), name)
    width = number(data.get("width_px"), name)
    height = number(data.get("height_px"), name)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must have positive dimensions")
    return left, top, width, height


def marker_screen_points(frame: dict[str, Json]) -> tuple[Point, ...]:
    markers = frame.get("registration_markers")
    if not isinstance(markers, list) or len(markers) != len(MARKER_IDS):
        raise ValueError("frame record must contain all four marker rectangles")
    rectangles: dict[str, tuple[float, float, float, float]] = {}
    for item in markers:
        marker = mapping(item, "marker")
        marker_id = marker.get("marker_id")
        if not isinstance(marker_id, str) or marker_id not in MARKER_IDS or marker_id in rectangles:
            raise ValueError("frame marker identities must be unique and complete")
        rectangles[marker_id] = rectangle(marker.get("box_screen_css_px"), marker_id)
    ordered = tuple(rectangles[marker_id] for marker_id in MARKER_IDS)
    scale = ordered[0][2] / MARKER_SIZE_DESIGN_PX
    origin_x = ordered[0][0] - MARKER_MARGIN_DESIGN_PX * scale
    origin_y = ordered[0][1] - MARKER_MARGIN_DESIGN_PX * scale
    centres = tuple((left + width / 2, top + height / 2) for left, top, width, height in ordered)
    for bounds, centre, design in zip(ordered, centres, MARKER_CENTRES_DESIGN_PX, strict=True):
        expected = (origin_x + design[0] * scale, origin_y + design[1] * scale)
        if (
            max(
                abs(centre[0] - expected[0]),
                abs(centre[1] - expected[1]),
                abs(bounds[2] - MARKER_SIZE_DESIGN_PX * scale),
                abs(bounds[3] - MARKER_SIZE_DESIGN_PX * scale),
            )
            > GEOMETRY_TOLERANCE_PX
        ):
            raise ValueError("marker rectangles do not match schema 2 geometry")
    region = rectangle(frame.get("target_region_screen_css_px"), "target region")
    expected_region = (origin_x + 160 * scale, origin_y + 180 * scale, 1280 * scale, 540 * scale)
    if (
        max(
            abs(actual - expected) for actual, expected in zip(region, expected_region, strict=True)
        )
        > GEOMETRY_TOLERANCE_PX
    ):
        raise ValueError("target region does not match schema 2 geometry")
    return centres


def polygon(points: tuple[Point, ...], name: str) -> NDArray[np.float32]:
    contour = np.asarray(points, dtype=np.float32)
    if (
        not cv2.isContourConvex(contour)
        or abs(cv2.contourArea(contour)) < MINIMUM_QUADRILATERAL_AREA_PX2
    ):
        raise ValueError(f"{name} is degenerate, crossed or non-convex")
    return contour


def project(matrix: Matrix, position: Point) -> Point:
    coordinates = matrix @ np.asarray((position[0], position[1], 1.0))
    denominator = float(coordinates[2])
    if abs(denominator) < DENOMINATOR_TOLERANCE:
        raise ValueError("mapping has a projective singularity")
    projected = (float(coordinates[0] / denominator), float(coordinates[1] / denominator))
    if not all(math.isfinite(value) for value in projected):
        raise ValueError("mapping produced non-finite coordinates")
    return projected


def checks_from(value: Json) -> tuple[Check, ...]:
    if not isinstance(value, list):
        raise ValueError("validation_points must be a list")
    checks: list[Check] = []
    for item in value:
        data = mapping(item, "validation point")
        name = data.get("name")
        if not isinstance(name, str) or not name or any(check.name == name for check in checks):
            raise ValueError("validation point names must be non-empty and unique")
        checks.append(
            Check(name, point(data.get("screen_css_px"), name), point(data.get("camera_px"), name))
        )
    return tuple(checks)


def check_coverage(checks: tuple[Check, ...], region: tuple[float, float, float, float]) -> None:
    if len(checks) < MINIMUM_CHECK_COUNT:
        raise ValueError("at least five independent validation points are required")
    left, top, width, height = region
    fractions = tuple(
        ((check.screen[0] - left) / width, (check.screen[1] - top) / height) for check in checks
    )
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in fractions):
        raise ValueError("validation points must lie in the target region")
    if len(set(check.screen for check in checks)) != len(checks):
        raise ValueError("validation screen points must be distinct")
    quadrants = {
        (int(x >= 0.5), int(y >= 0.5))
        for x, y in fractions
        if abs(x - 0.5) >= CENTRE_TOLERANCE_FRACTION and abs(y - 0.5) >= CENTRE_TOLERANCE_FRACTION
    }
    centre = any(
        abs(x - 0.5) <= CENTRE_TOLERANCE_FRACTION and abs(y - 0.5) <= CENTRE_TOLERANCE_FRACTION
        for x, y in fractions
    )
    if len(quadrants) != 4 or not centre:
        raise ValueError(
            "validation points must cover the centre and all four target-region quadrants"
        )


def camera_point_in_image(position: Point, width_px: int, height_px: int) -> bool:
    return 0 <= position[0] < width_px and 0 <= position[1] < height_px


def process_frame(
    data: dict[str, Json], annotations_directory: Path, output: Path, maximum_error_px: float
) -> dict[str, Json]:
    filename = data.get("image")
    sequence = data.get("image_sequence")
    result: dict[str, Json] = {
        "image": filename,
        "image_sequence": sequence,
        "model": "planar homography",
        "status": "failed",
        "failures": [],
        "transformed_points": [],
    }
    failures: list[Json] = []
    result["failures"] = failures
    image = None
    try:
        if not math.isfinite(maximum_error_px) or maximum_error_px <= 0:
            raise ValueError("maximum validation error must be finite and positive")
        if (
            not isinstance(filename, str)
            or not filename
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ValueError("image and non-negative image_sequence are required")
        image_path = annotations_directory / filename
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read image: {image_path}")
        height_px, width_px = image.shape[:2]
        frame = mapping(data.get("stimulus_frame"), "stimulus_frame")
        result["stimulus_reference"] = {
            key: frame.get(key)
            for key in (
                "presentation_id",
                "run_number",
                "scene_id",
                "scene_frame",
                "sequence_frame",
                "render_number",
            )
        }
        if frame.get("schema_version") != 2 or frame.get("kind") != "frame":
            raise ValueError("stimulus_frame must be a schema 2 frame record")
        screen = marker_screen_points(frame)
        camera_data = mapping(data.get("marker_centres_camera_px"), "marker_centres_camera_px")
        if (
            set(camera_data) != set(MARKER_IDS)
            or data.get("marker_identities_confirmed") is not True
            or data.get("all_markers_fully_visible") is not True
        ):
            raise ValueError(
                "all four marker centres, identities and full visibility must be confirmed"
            )
        camera = tuple(point(camera_data[marker_id], marker_id) for marker_id in MARKER_IDS)
        if not all(camera_point_in_image(position, width_px, height_px) for position in camera):
            raise ValueError("marker centres must lie inside the image")
        screen_polygon = polygon(screen, "screen marker quadrilateral")
        camera_polygon = polygon(camera, "camera marker quadrilateral")
        if np.sign(cv2.contourArea(screen_polygon, oriented=True)) != np.sign(
            cv2.contourArea(camera_polygon, oriented=True)
        ):
            raise ValueError("marker order is mirrored or inconsistent")
        matrix = cast(Matrix, cv2.getPerspectiveTransform(screen_polygon, camera_polygon))
        if (
            not np.isfinite(matrix).all()
            or abs(float(np.linalg.det(matrix))) < DENOMINATOR_TOLERANCE
        ):
            raise ValueError("homography is singular")
        denominators = tuple(float(matrix[2] @ np.asarray((x, y, 1.0))) for x, y in screen)
        if (
            min(abs(value) for value in denominators) < DENOMINATOR_TOLERANCE
            or min(denominators) * max(denominators) <= 0
        ):
            raise ValueError("homography has a singularity within the marker quadrilateral")
        result["matrix"] = cast(Json, matrix.tolist())
        result["fit_residuals_px"] = [
            math.dist(project(matrix, source), target)
            for source, target in zip(screen, camera, strict=True)
        ]
        result["status"] = "unvalidated"
        for marker_id, position in zip(MARKER_IDS, camera, strict=True):
            pixel = tuple(round(value) for value in position)
            cv2.circle(image, pixel, MARKER_RADIUS_PX, MARKER_COLOUR_BGR, ANNOTATION_LINE_WIDTH_PX)
            cv2.putText(
                image,
                marker_id,
                pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                ANNOTATION_FONT_SCALE,
                MARKER_COLOUR_BGR,
                ANNOTATION_LINE_WIDTH_PX,
            )
        try:
            checks = checks_from(data.get("validation_points", []))
            if not all(
                camera_point_in_image(check.camera, width_px, height_px) for check in checks
            ):
                raise ValueError("validation camera points must lie inside the image")
            if any(
                cv2.pointPolygonTest(screen_polygon, check.screen, False) <= 0 for check in checks
            ):
                raise ValueError(
                    "validation points must lie strictly inside the marker quadrilateral"
                )
            residuals: list[Json] = []
            errors_px: list[float] = []
            for check in checks:
                predicted = project(matrix, check.screen)
                error_px = math.dist(predicted, check.camera)
                errors_px.append(error_px)
                residuals.append(
                    {
                        "name": check.name,
                        "screen_css_px": list(check.screen),
                        "observed_camera_px": list(check.camera),
                        "predicted_camera_px": list(predicted),
                        "error_px": error_px,
                    }
                )
                observed_pixel = tuple(round(value) for value in check.camera)
                predicted_pixel = tuple(round(value) for value in predicted)
                cv2.circle(
                    image,
                    observed_pixel,
                    CHECK_RADIUS_PX,
                    OBSERVED_COLOUR_BGR,
                    ANNOTATION_LINE_WIDTH_PX,
                )
                cv2.drawMarker(
                    image,
                    predicted_pixel,
                    PREDICTED_COLOUR_BGR,
                    cv2.MARKER_CROSS,
                    CROSS_SIZE_PX,
                    ANNOTATION_LINE_WIDTH_PX,
                )
                cv2.line(
                    image,
                    observed_pixel,
                    predicted_pixel,
                    PREDICTED_COLOUR_BGR,
                    ANNOTATION_LINE_WIDTH_PX,
                )
            result["validation_residuals"] = residuals
            if errors_px:
                result["max_validation_error_px"] = max(errors_px)
            check_coverage(
                checks, rectangle(frame.get("target_region_screen_css_px"), "target region")
            )
            if data.get("validation_points_independently_observed") is not True:
                raise ValueError(
                    "independent camera observation of validation points must be confirmed"
                )
            if max(errors_px) > maximum_error_px:
                raise ValueError("independent validation error exceeds the supplied threshold")
            hull = cv2.convexHull(np.asarray([check.screen for check in checks], dtype=np.float32))
            result["validated_hull_screen_css_px"] = cast(Json, hull.reshape(-1, 2).tolist())
            requests = data.get("screen_points", [])
            if not isinstance(requests, list):
                raise ValueError("screen_points must be a list")
            transformed: list[Json] = []
            for index, request in enumerate(requests):
                requested = point(request, f"screen_points[{index}]")
                if cv2.pointPolygonTest(hull, requested, False) < 0:
                    raise ValueError("requested point lies outside the validation hull")
                projected = project(matrix, requested)
                if not camera_point_in_image(projected, width_px, height_px):
                    raise ValueError("requested point maps outside the image")
                transformed.append({"screen_css_px": list(requested), "camera_px": list(projected)})
            result["transformed_points"] = transformed
            result["status"] = "validated_within_sampled_hull"
        except ValueError as error:
            failures.append(str(error))
    except (ValueError, cv2.error) as error:
        failures.append(str(error))
    if image is not None and isinstance(sequence, int) and isinstance(filename, str):
        overlay = output / f"{sequence:09d}-{Path(filename).stem}-registration.png"
        cv2.putText(
            image,
            str(result["status"]),
            (STATUS_LEFT_PX, image.shape[0] - STATUS_BOTTOM_MARGIN_PX),
            cv2.FONT_HERSHEY_SIMPLEX,
            STATUS_FONT_SCALE,
            STATUS_COLOUR_BGR,
            ANNOTATION_LINE_WIDTH_PX,
        )
        if cv2.imwrite(str(overlay), image):
            result["overlay"] = str(overlay)
        else:
            failures.append(f"cannot write overlay: {overlay}")
            result["status"] = "failed"
            result["transformed_points"] = []
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--max-error-px",
        type=float,
        required=True,
        help="maximum independent camera-pixel residual",
    )
    options = parser.parse_args(arguments)
    if not math.isfinite(options.max_error_px) or options.max_error_px <= 0:
        parser.error("--max-error-px must be finite and positive")
    try:
        document = mapping(
            cast(Json, json.loads(options.annotations.read_text(encoding="utf-8"))), "annotations"
        )
        frames = document.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("frames must be a non-empty list")
        sequences = [mapping(frame, "frame").get("image_sequence") for frame in frames]
        if len({str(sequence) for sequence in sequences}) != len(sequences):
            raise ValueError("each image sequence must be unique")
        options.output.mkdir(parents=True, exist_ok=False)
        results = [
            process_frame(
                mapping(frame, "frame"),
                options.annotations.parent,
                options.output,
                options.max_error_px,
            )
            for frame in frames
        ]
        report: dict[str, Json] = {
            "model": "planar homography",
            "max_allowed_validation_error_px": options.max_error_px,
            "association": "operator supplied; not optically verified by this tool",
            "frames": cast(Json, results),
        }
        (options.output / "registration.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"{error}\n")
    print(options.output / "registration.json")
    return (
        0 if all(result["status"] == "validated_within_sampled_hull" for result in results) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
