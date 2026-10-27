#!/usr/bin/env python
"""Associate recorded JPEGs with optical frame codes and check registration offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy
from numpy.typing import NDArray

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue
from tools.register_monitor_frames import (
    MARKER_IDS,
    mapping,
    marker_screen_points,
    point,
    polygon,
    process_frame,
    project,
    rectangle,
)

CODE_COLUMNS = 32
CODE_ROWS = 6
CODE_BYTES = 24
CODE_LEFT_DESIGN_PX = 1056
CODE_TOP_DESIGN_PX = 740
CODE_CHECK_TOP_DESIGN_PX = 96
CODE_WIDTH_DESIGN_PX = 384
CODE_HEIGHT_DESIGN_PX = 72
MARKER_ORIGIN_DESIGN_PX = 56
MARKER_HORIZONTAL_DISTANCE_DESIGN_PX = 1488
MARKER_WIDTH_DESIGN_PX = 64
GEOMETRY_TOLERANCE_CSS_PX = 0.05
CRC64_POLYNOMIAL = 0x42F0E1EBA9EA3693
CRC64_MASK = 0xFFFFFFFFFFFFFFFF
MINIMUM_CELL_CAMERA_PX = 1.7
MINIMUM_CODE_CONTRAST = 45.0
MINIMUM_CLASS_MARGIN = 8.0
CROSS_RADIUS_DESIGN_PX = 14
CROSS_OUTLINE_DESIGN_PX = 7
CROSS_LINE_DESIGN_PX = 3
CROSS_PATCH_RADIUS_DESIGN_PX = 22
SEARCH_RADIUS_CAMERA_PX = 5
MINIMUM_CROSS_CORRELATION = 0.69
MINIMUM_CROSS_PEAK_GAP = 0.20
MINIMUM_CROSS_TEMPLATE_SIZE_PX = 5

Image = NDArray[numpy.uint8]
Matrix = NDArray[numpy.float64]
Point = tuple[float, float]


def read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    return [
        mapping(cast(JsonValue, json.loads(line)), str(path))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def identity_bytes(presentation: str, session: str, run: int, render: int) -> bytes:
    if (
        len(presentation) != 64
        or len(session) != 16
        or not (0 <= run <= 65535 and 0 <= render <= 4294967295)
    ):
        raise ValueError("invalid optical identity")
    try:
        bytes.fromhex(presentation)
        payload = (
            bytes.fromhex(presentation[:4] + session)
            + run.to_bytes(2, "big")
            + render.to_bytes(4, "big")
        )
    except ValueError as error:
        raise ValueError("invalid optical identity hexadecimal") from error
    return payload + crc64(payload).to_bytes(8, "big")


def crc64(payload: bytes) -> int:
    checksum = 0
    for byte in payload:
        checksum ^= byte << 56
        for _ in range(8):
            checksum = (
                (checksum << 1) ^ (CRC64_POLYNOMIAL if checksum & (1 << 63) else 0)
            ) & CRC64_MASK
    return checksum


def frame_identity(frame: dict[str, JsonValue]) -> bytes:
    presentation, session, run, render = (
        frame.get(key)
        for key in ("presentation_id", "optical_session", "run_number", "render_number")
    )
    if (
        not isinstance(presentation, str)
        or not isinstance(session, str)
        or type(run) is not int
        or type(render) is not int
    ):
        raise ValueError("frame lacks optical identity")
    return identity_bytes(presentation, session, run, render)


def camera_matrix(frame: dict[str, JsonValue], markers: dict[str, JsonValue]) -> Matrix:
    if set(markers) != set(MARKER_IDS):
        raise ValueError("all four manually checked marker centres are required")
    screen = polygon(marker_screen_points(frame), "screen marker quadrilateral")
    camera = polygon(
        tuple(point(markers[name], name) for name in MARKER_IDS), "camera marker quadrilateral"
    )
    if numpy.sign(cv2.contourArea(screen, oriented=True)) != numpy.sign(
        cv2.contourArea(camera, oriented=True)
    ):
        raise ValueError("marker order is mirrored or inconsistent")
    matrix = cast(Matrix, cv2.getPerspectiveTransform(screen, camera))
    if not numpy.isfinite(matrix).all() or abs(float(numpy.linalg.det(matrix))) < 1e-9:
        raise ValueError("singular marker geometry")
    return matrix


def decode_code(
    image: Image, frame: dict[str, JsonValue], matrix: Matrix, *, upper: bool = False
) -> bytes:
    key = "optical_check_screen_css_px" if upper else "optical_code_screen_css_px"
    left, top, width, height = rectangle(frame.get(key), "optical code")
    markers = marker_screen_points(frame)
    scale = (markers[1][0] - markers[0][0]) / MARKER_HORIZONTAL_DISTANCE_DESIGN_PX
    origin_x = markers[0][0] - MARKER_ORIGIN_DESIGN_PX * scale
    origin_y = markers[0][1] - MARKER_ORIGIN_DESIGN_PX * scale
    design_top = CODE_CHECK_TOP_DESIGN_PX if upper else CODE_TOP_DESIGN_PX
    expected = (
        origin_x + CODE_LEFT_DESIGN_PX * scale,
        origin_y + design_top * scale,
        CODE_WIDTH_DESIGN_PX * scale,
        CODE_HEIGHT_DESIGN_PX * scale,
    )
    if (
        max(
            abs(actual - reference)
            for actual, reference in zip((left, top, width, height), expected, strict=True)
        )
        > GEOMETRY_TOLERANCE_CSS_PX
    ):
        raise ValueError("optical code rectangle does not match stimulus geometry")
    intensities: list[float] = []
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cell_width, cell_height = width / CODE_COLUMNS, height / CODE_ROWS
    for row in range(CODE_ROWS):
        for column in range(CODE_COLUMNS):
            centre = (left + (column + 0.5) * cell_width, top + (row + 0.5) * cell_height)
            camera = project(matrix, centre)
            next_column = project(matrix, (centre[0] + cell_width, centre[1]))
            next_row = project(matrix, (centre[0], centre[1] + cell_height))
            if (
                min(math.dist(camera, next_column), math.dist(camera, next_row))
                < MINIMUM_CELL_CAMERA_PX
            ):
                raise ValueError("optical cells are too small for reliable sampling")
            samples: list[float] = []
            for offset_x, offset_y in (
                (0.0, 0.0),
                (-0.15, 0.0),
                (0.15, 0.0),
                (0.0, -0.15),
                (0.0, 0.15),
            ):
                x, y = project(
                    matrix, (centre[0] + offset_x * cell_width, centre[1] + offset_y * cell_height)
                )
                if x < 0 or y < 0 or x >= grey.shape[1] - 1 or y >= grey.shape[0] - 1:
                    raise ValueError("optical code is outside the image")
                samples.append(float(cv2.getRectSubPix(grey, (1, 1), (x, y))[0, 0]))
            intensities.append(float(numpy.median(samples)))
    low, high = numpy.percentile(intensities, (10, 90))
    if high - low < MINIMUM_CODE_CONTRAST:
        raise ValueError("insufficient optical code contrast")
    threshold = float((low + high) / 2)
    if any(abs(value - threshold) < MINIMUM_CLASS_MARGIN for value in intensities):
        raise ValueError("ambiguous optical cell")
    bits = [int(value < threshold) for value in intensities]
    decoded = bytes(
        sum(bits[index + bit] << (7 - bit) for bit in range(8)) for index in range(0, len(bits), 8)
    )
    if len(decoded) != CODE_BYTES or crc64(decoded[:-8]) != int.from_bytes(decoded[-8:], "big"):
        raise ValueError("optical code CRC-64 failed")
    return decoded


def cross_template(matrix: Matrix, position: Point, screen_scale: float) -> Image:
    radius = CROSS_PATCH_RADIUS_DESIGN_PX
    size = 2 * radius + 1
    template = numpy.full((size, size), 184, dtype=numpy.uint8)
    centre = (radius, radius)
    for colour, thickness in ((255, CROSS_OUTLINE_DESIGN_PX), (0, CROSS_LINE_DESIGN_PX)):
        cv2.line(
            template,
            (centre[0] - CROSS_RADIUS_DESIGN_PX, centre[1]),
            (centre[0] + CROSS_RADIUS_DESIGN_PX, centre[1]),
            colour,
            thickness,
        )
        cv2.line(
            template,
            (centre[0], centre[1] - CROSS_RADIUS_DESIGN_PX),
            (centre[0], centre[1] + CROSS_RADIUS_DESIGN_PX),
            colour,
            thickness,
        )
    camera = project(matrix, position)
    horizontal = project(matrix, (position[0] + screen_scale, position[1]))
    vertical = project(matrix, (position[0], position[1] + screen_scale))
    horizontal_step = (horizontal[0] - camera[0], horizontal[1] - camera[1])
    vertical_step = (vertical[0] - camera[0], vertical[1] - camera[1])
    half_width = math.ceil(radius * (abs(horizontal_step[0]) + abs(vertical_step[0])))
    half_height = math.ceil(radius * (abs(horizontal_step[1]) + abs(vertical_step[1])))
    width, height = half_width * 2 + 1, half_height * 2 + 1
    if min(width, height) < MINIMUM_CROSS_TEMPLATE_SIZE_PX:
        raise ValueError("registration cross is too small")
    affine = numpy.array(
        (
            (
                horizontal_step[0],
                vertical_step[0],
                half_width - radius * (horizontal_step[0] + vertical_step[0]),
            ),
            (
                horizontal_step[1],
                vertical_step[1],
                half_height - radius * (horizontal_step[1] + vertical_step[1]),
            ),
        ),
        dtype=numpy.float64,
    )
    return cast(
        Image,
        cv2.warpAffine(template, affine, (width, height), flags=cv2.INTER_AREA, borderValue=184),
    )


def observe_cross(
    image: Image, matrix: Matrix, position: Point, screen_scale: float
) -> tuple[Point, float]:
    predicted = project(matrix, position)
    template = cross_template(matrix, position, screen_scale)
    height, width = template.shape
    centre_x, centre_y = round(predicted[0]), round(predicted[1])
    left = centre_x - width // 2 - SEARCH_RADIUS_CAMERA_PX
    top = centre_y - height // 2 - SEARCH_RADIUS_CAMERA_PX
    right, bottom = (
        left + width + 2 * SEARCH_RADIUS_CAMERA_PX,
        top + height + 2 * SEARCH_RADIUS_CAMERA_PX,
    )
    if left < 0 or top < 0 or right > image.shape[1] or bottom > image.shape[0]:
        raise ValueError("cross search outside image")
    grey = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    scores = cv2.matchTemplate(grey, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, location = cv2.minMaxLoc(scores)
    competitors = scores.copy()
    x, y = location
    competitors[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = -1
    second = float(numpy.max(competitors))
    if best < MINIMUM_CROSS_CORRELATION or best - second < MINIMUM_CROSS_PEAK_GAP:
        raise ValueError("cross evidence is weak or ambiguous")
    return (float(left + x + width // 2), float(top + y + height // 2)), float(best)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("browser", type=Path)
    parser.add_argument("markers", type=Path, help="per-image manually confirmed marker JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-error-px", type=float, required=True)
    options = parser.parse_args(arguments)
    if not math.isfinite(options.max_error_px) or options.max_error_px <= 0:
        parser.error("--max-error-px must be finite and positive")
    try:
        index = read_jsonl(options.index)
        browser = read_jsonl(options.browser)
        annotations = mapping(cast(JsonValue, json.loads(options.markers.read_text())), "markers")
        entries = annotations.get("frames")
        if not isinstance(entries, list) or not entries:
            raise ValueError("markers.frames must be a non-empty list")
        index_by_name: dict[str, dict[str, JsonValue]] = {}
        sequences: set[int] = set()
        for record in index:
            name, sequence = record.get("filename"), record.get("sequence")
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or type(sequence) is not int
                or sequence < 0
                or name in index_by_name
                or sequence in sequences
            ):
                raise ValueError("index filenames and non-negative sequences must be unique")
            index_by_name[name] = record
            sequences.add(sequence)
        frames = [
            record
            for record in browser
            if record.get("kind") == "frame"
            and record.get("optical_code_screen_css_px") is not None
        ]
        by_code: dict[bytes, list[dict[str, JsonValue]]] = {}
        layouts: dict[str, dict[str, JsonValue]] = {}
        for frame in frames:
            code = frame_identity(frame)
            by_code.setdefault(code, []).append(frame)
            layout = json.dumps(
                [
                    frame.get("registration_markers"),
                    frame.get("target_region_screen_css_px"),
                    frame.get("optical_code_screen_css_px"),
                    frame.get("optical_check_screen_css_px"),
                ],
                sort_keys=True,
            )
            layouts[layout] = frame
        options.output.mkdir(parents=True, exist_ok=False)
        results: list[JsonValue] = []
        seen: set[str] = set()
        for entry in entries:
            annotation = mapping(entry, "marker frame")
            name = annotation.get("image")
            result: dict[str, JsonValue] = {"image": name, "status": "candidate", "failures": []}
            results.append(result)
            try:
                if not isinstance(name, str) or name in seen or name not in index_by_name:
                    raise ValueError("image must occur once in the index and marker file")
                seen.add(name)
                result["image_sequence"] = index_by_name[name]["sequence"]
                image_path = options.index.parent / name
                result["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                loaded_image = cv2.imread(str(image_path))
                if loaded_image is None:
                    raise ValueError("cannot decode JPEG")
                image = cast(Image, loaded_image)
                if not frames:
                    raise ValueError("browser export contains no optically coded frames")
                if (
                    annotation.get("marker_identities_confirmed") is not True
                    or annotation.get("all_markers_fully_visible") is not True
                ):
                    raise ValueError(
                        "marker identity and full visibility must be manually confirmed"
                    )
                markers = mapping(annotation.get("marker_centres_camera_px"), "marker centres")
                if any(
                    not (
                        0 <= point(value, name)[0] < image.shape[1]
                        and 0 <= point(value, name)[1] < image.shape[0]
                    )
                    for name, value in markers.items()
                ):
                    raise ValueError("marker centres must lie inside the image")
                matches: dict[bytes, dict[str, JsonValue]] = {}
                for layout_frame in layouts.values():
                    try:
                        matrix = camera_matrix(layout_frame, markers)
                        decoded = decode_code(image, layout_frame, matrix)
                        if decode_code(image, layout_frame, matrix, upper=True) != decoded:
                            continue
                        possible = by_code.get(decoded, [])
                        if len(possible) != 1:
                            continue
                        selected = possible[0]
                        selected_layout = [
                            selected.get("registration_markers"),
                            selected.get("target_region_screen_css_px"),
                            selected.get("optical_code_screen_css_px"),
                            selected.get("optical_check_screen_css_px"),
                        ]
                        if json.dumps(selected_layout, sort_keys=True) == json.dumps(
                            [
                                layout_frame.get("registration_markers"),
                                layout_frame.get("target_region_screen_css_px"),
                                layout_frame.get("optical_code_screen_css_px"),
                                layout_frame.get("optical_check_screen_css_px"),
                            ],
                            sort_keys=True,
                        ):
                            matches[decoded] = selected
                    except (ValueError, cv2.error):
                        continue
                if len(matches) != 1:
                    raise ValueError(
                        "no unique CRC-checked optical match for the supplied geometry"
                    )
                code, frame = next(iter(matches.items()))
                result["status"] = "unvalidated"
                result["optical_code_hex"] = code.hex()
                result["stimulus_frame_sha256"] = hashlib.sha256(
                    json.dumps(frame, sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                result["association"] = {
                    key: frame.get(key)
                    for key in (
                        "presentation_id",
                        "optical_session",
                        "run_number",
                        "render_number",
                        "scene_id",
                        "scene_frame",
                        "sequence_frame",
                    )
                }
                matrix = camera_matrix(frame, markers)
                scale = (
                    rectangle(
                        mapping(
                            cast(list[JsonValue], frame["registration_markers"])[0], "marker"
                        ).get("box_screen_css_px"),
                        "marker",
                    )[2]
                    / MARKER_WIDTH_DESIGN_PX
                )
                checks: list[JsonValue] = []
                rejected: list[JsonValue] = []
                points = frame.get("registration_points")
                if not isinstance(points, list):
                    raise ValueError("frame lacks registration points")
                fractions = (0.1, 0.34, 0.5, 0.66, 0.9)
                region_left, region_top, region_width, region_height = rectangle(
                    frame.get("target_region_screen_css_px"), "target region"
                )
                expected_points = {
                    f"row-{row + 1}-column-{column + 1}": (
                        region_left + x * region_width,
                        region_top + y * region_height,
                    )
                    for row, y in enumerate(fractions)
                    for column, x in enumerate(fractions)
                }
                if len(points) != len(expected_points):
                    raise ValueError("all 25 scripted registration points are required")
                seen_points: set[str] = set()
                for item in points:
                    reference = mapping(item, "registration point")
                    identifier = reference.get("point_id")
                    if (
                        not isinstance(identifier, str)
                        or identifier not in expected_points
                        or identifier in seen_points
                        or math.dist(
                            point(reference.get("screen_css_px"), "registration point"),
                            expected_points[identifier],
                        )
                        > GEOMETRY_TOLERANCE_CSS_PX
                    ):
                        raise ValueError("registration point identity or geometry is invalid")
                    seen_points.add(identifier)
                    if reference.get("visible") is not True:
                        rejected.append({"point_id": identifier, "reason": "scripted occlusion"})
                        continue
                    try:
                        screen = point(reference.get("screen_css_px"), "registration point")
                        observed, correlation = observe_cross(image, matrix, screen, scale)
                        checks.append(
                            {
                                "name": identifier,
                                "screen_css_px": list(screen),
                                "camera_px": list(observed),
                                "correlation": correlation,
                            }
                        )
                    except (ValueError, cv2.error) as error:
                        rejected.append({"point_id": identifier, "reason": str(error)})
                result["rejected_crosses"] = rejected
                result["observed_crosses"] = checks
                registration: dict[str, JsonValue] = {
                    "image": str(image_path),
                    "image_sequence": result["image_sequence"],
                    "stimulus_frame": frame,
                    "marker_identities_confirmed": True,
                    "all_markers_fully_visible": True,
                    "marker_centres_camera_px": markers,
                    "validation_points_independently_observed": True,
                    "validation_points": checks,
                    "screen_points": [],
                }
                checked = process_frame(
                    registration, options.index.parent, options.output, options.max_error_px
                )
                result["registration"] = checked
                result["status"] = (
                    "validated"
                    if checked["status"] == "validated_within_sampled_hull"
                    else "unvalidated"
                )
                if result["status"] == "validated":
                    hull = numpy.asarray(
                        checked["validated_hull_screen_css_px"], dtype=numpy.float32
                    )
                    projected: list[JsonValue] = []
                    targets = frame.get("targets")
                    if isinstance(targets, list):
                        for item in targets:
                            target = mapping(item, "target")
                            if target.get("present") is not True:
                                continue
                            left, top, width, height = rectangle(
                                target.get("box_screen_css_px"), "photograph"
                            )
                            centre = (left + width / 2, top + height / 2)
                            if cv2.pointPolygonTest(hull, centre, False) >= 0:
                                camera = project(matrix, centre)
                                if (
                                    0 <= camera[0] < image.shape[1]
                                    and 0 <= camera[1] < image.shape[0]
                                ):
                                    projected.append(
                                        {
                                            "intended_identity": target.get("intended_identity"),
                                            "photograph_centre_screen_css_px": list(centre),
                                            "projected_photograph_centre_camera_px": list(camera),
                                        }
                                    )
                    result["projected_photograph_centres"] = projected
            except (OSError, ValueError, cv2.error) as error:
                cast(list[JsonValue], result["failures"]).append(str(error))
        report: dict[str, JsonValue] = {
            "index_sha256": hashlib.sha256(options.index.read_bytes()).hexdigest(),
            "browser_jsonl_sha256": hashlib.sha256(options.browser.read_bytes()).hexdigest(),
            "marker_annotations_sha256": hashlib.sha256(options.markers.read_bytes()).hexdigest(),
            "maximum_error_camera_px": options.max_error_px,
            "frames": results,
        }
        (options.output / "association.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"{error}\n")
    print(options.output / "association.json")
    return 0 if all(mapping(result, "result")["status"] == "validated" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
