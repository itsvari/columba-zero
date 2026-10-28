#!/usr/bin/env python
"""Detect complete monitor corner markers as candidates for human confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


class MarkerCandidate(TypedDict):
    identity: str
    centre: list[float]
    corners: list[list[float]]
    contrast: float
    minimum_margin: float


PATTERNS = {
    "top-left": ("11111", "10001", "10101", "10001", "11111"),
    "top-right": ("11111", "11001", "10001", "10001", "11111"),
    "bottom-right": ("11111", "10001", "10001", "10101", "11111"),
    "bottom-left": ("11111", "10011", "10001", "10001", "11111"),
}
MARKER_SIZE_PX = 80
MINIMUM_SIDE_PX = 12
MINIMUM_CONTRAST = 60
MINIMUM_MARGIN = 10


def order_corners(points: NDArray[np.float32]) -> NDArray[np.float32]:
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    return ordered


def candidate(image: Image, corners: NDArray[np.float32]) -> MarkerCandidate | None:
    destination = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    patch = cv2.warpPerspective(image, transform, (MARKER_SIZE_PX, MARKER_SIZE_PX))
    values = []
    for row in range(5):
        for column in range(5):
            x = round(10 + (column + 0.5) * 12)
            y = round(10 + (row + 0.5) * 12)
            values.append(float(np.median(patch[y - 2 : y + 3, x - 2 : x + 3])))
    low, high = np.percentile(values, (20, 80))
    if high - low < MINIMUM_CONTRAST:
        return None
    threshold = (low + high) / 2
    if min(abs(value - threshold) for value in values) < MINIMUM_MARGIN:
        return None
    pattern = tuple(
        "".join("1" if values[row * 5 + column] < threshold else "0" for column in range(5))
        for row in range(5)
    )
    matches = [name for name, expected in PATTERNS.items() if expected == pattern]
    if not (
        pattern[0] == "11111"
        and pattern[4] == "11111"
        and all(row[0] == "1" and row[4] == "1" for row in pattern)
    ):
        return None
    quiet = np.concatenate(
        (
            patch[4:8, 12:68].ravel(),
            patch[72:76, 12:68].ravel(),
            patch[12:68, 4:8].ravel(),
            patch[12:68, 72:76].ravel(),
        )
    )
    if float(np.percentile(quiet, 15)) < threshold:
        return None
    centre = cv2.perspectiveTransform(
        np.array([[[39.5, 39.5]]], dtype=np.float32), np.linalg.inv(transform)
    )[0, 0]
    return {
        "identity": matches[0] if len(matches) == 1 else "unrecognised",
        "centre": [float(centre[0]), float(centre[1])],
        "corners": corners.tolist(),
        "contrast": float(high - low),
        "minimum_margin": float(min(abs(value - threshold) for value in values)),
    }


def detect(image: Image) -> tuple[list[MarkerCandidate], str]:
    grey = cast(Image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) if image.ndim == 3 else image
    _, binary = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    height, width = grey.shape
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        vertices = cv2.approxPolyDP(contour, 0.045 * perimeter, True)
        if len(vertices) != 4 or not cv2.isContourConvex(vertices):
            continue
        corners = order_corners(vertices.reshape(4, 2).astype(np.float32))
        lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        if min(lengths) < MINIMUM_SIDE_PX or max(lengths) > min(lengths) * 2:
            continue
        if (
            np.min(corners[:, 0]) < 1
            or np.min(corners[:, 1]) < 1
            or np.max(corners[:, 0]) >= width - 1
            or np.max(corners[:, 1]) >= height - 1
        ):
            continue
        found = candidate(grey, corners)
        if found is not None:
            candidates.append(found)
    if len(candidates) != 4 or {item["identity"] for item in candidates} != set(PATTERNS):
        return candidates, "expected exactly one complete candidate of each identity"
    points = np.array(
        [
            next(item["centre"] for item in candidates if item["identity"] == name)
            for name in PATTERNS
        ],
        dtype=np.float32,
    )
    if (
        not cv2.isContourConvex(points.reshape(-1, 1, 2))
        or points[0, 0] >= points[1, 0]
        or points[3, 0] >= points[2, 0]
        or points[0, 1] >= points[3, 1]
        or points[1, 1] >= points[2, 1]
    ):
        return candidates, "marker geometry is not globally upright"
    return candidates, "candidate"


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--image", action="append", default=[])
    options = parser.parse_args(arguments)
    raw_index = options.index.read_bytes()
    entries = [json.loads(line) for line in raw_index.splitlines() if line.strip()]
    requested = set(options.image)
    input_paths = {options.index.resolve()}
    input_paths.update((options.index.parent / entry["filename"]).resolve() for entry in entries)
    output_paths = {options.output.resolve(), options.diagnostics.resolve()}
    if len(output_paths) != 2 or output_paths & input_paths:
        raise ValueError("output paths must be distinct from each other and all inputs")
    frames, diagnostics = [], []
    for entry in entries:
        name = entry["filename"]
        if requested and name not in requested:
            continue
        if Path(name).name != name:
            raise ValueError("invalid image name")
        path = options.index.parent / name
        source = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {name}")
        candidates, status = detect(cast(Image, image))
        diagnostics.append(
            {
                "image": name,
                "image_sha256": hashlib.sha256(source).hexdigest(),
                "status": status,
                "candidates": candidates,
            }
        )
        if status == "candidate":
            frames.append(
                {
                    "image": name,
                    "marker_identities_confirmed": False,
                    "all_markers_fully_visible": False,
                    "marker_centres_camera_px": {
                        item["identity"]: item["centre"] for item in candidates
                    },
                }
            )
    if requested - {item["image"] for item in diagnostics}:
        raise ValueError("requested image absent from index")
    options.output.write_text(json.dumps({"frames": frames}, indent=2) + "\n")
    options.diagnostics.write_text(
        json.dumps(
            {"index_sha256": hashlib.sha256(raw_index).hexdigest(), "frames": diagnostics}, indent=2
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
