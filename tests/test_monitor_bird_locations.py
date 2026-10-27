from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import cv2
import numpy
import pytest
from numpy.typing import NDArray

from tools.locate_monitor_birds import (
    CameraCheck,
    Label,
    apply_camera_check,
    digest,
    evaluate_geometry,
    feature_matches,
    load_labels,
)
from tools.monitor_stimulus import PHOTOGRAPHS


def label(bounds: tuple[float, float, float, float] = (50, 50, 150, 150), edge: float = 5) -> Label:
    return Label(PHOTOGRAPHS[0], bounds, edge, 2, "synthetic test")


def points() -> tuple[NDArray[numpy.float32], NDArray[numpy.float32]]:
    coords = numpy.array(
        [(x, y) for y in (20, 60, 100, 140, 180) for x in (20, 60, 100, 140, 180)],
        dtype=numpy.float32,
    )
    return coords, (coords * 0.5 + numpy.array([100, 80], dtype=numpy.float32)).astype(
        numpy.float32
    )


def test_geometry_bbox_and_uncertainty_scale() -> None:
    source, target = points()
    result = evaluate_geometry(source, target, label(), (400, 500))
    assert result["status"] == "candidate"
    assert result["visible_bbox_centre_camera_px"] == pytest.approx([150, 130], abs=0.01)
    assert result["centre_allowance_bounds_camera_xyxy_px"] == pytest.approx(
        [145.5, 125.5, 154.5, 134.5], abs=0.01
    )
    shifted = evaluate_geometry(source, target, label((70, 50, 170, 150)), (400, 500))
    assert shifted["visible_bbox_centre_camera_px"] == pytest.approx([160, 130], abs=0.01)


def test_held_out_checks_are_not_used_for_fitting() -> None:
    source, target = points()
    original = evaluate_geometry(source, target, label(), (400, 500))
    corners = numpy.asarray(original["photograph_corners_camera_px"], dtype=numpy.float32)
    check = CameraCheck("frame.jpg", PHOTOGRAPHS[0].asset_id, corners + 10, 1, "test")
    checked = apply_camera_check(original, check)
    assert checked["status"] == "rejected"
    assert checked["source_to_camera_homography"] == original["source_to_camera_homography"]
    target[1::2] += 10
    assert evaluate_geometry(source, target, label(), (400, 500))["status"] == "rejected"


def test_outside_hull_and_frame_are_rejected() -> None:
    source, target = points()
    assert (
        evaluate_geometry(source, target, label((185, 185, 200, 204)), (400, 500))["status"]
        == "rejected"
    )
    assert evaluate_geometry(source, target - 150, label(), (400, 500))["status"] == "rejected"


def write_label(path: Path, **changes: object) -> None:
    photo = PHOTOGRAPHS[0]
    entry: dict[str, object] = {
        "asset_id": photo.asset_id,
        "source_sha256": digest(Path(photo.source)),
        "crop_xywh_px": [25, 777, 200, 204],
        "visible_bbox_xyxy_px": [50, 50, 150, 150],
        "edge_allowance_source_px": 5,
        "registration_allowance_camera_px": 2,
        "confirmed": True,
        "reviewer": "synthetic test",
    }
    entry.update(changes)
    path.write_text(json.dumps({"labels": [entry]}))


@pytest.mark.parametrize(
    "changes",
    [
        {"source_sha256": "bad"},
        {"crop_xywh_px": [26, 777, 200, 204]},
        {"confirmed": False},
        {"reviewer": ""},
        {"visible_bbox_xyxy_px": [-1, 0, 200, 204]},
    ],
)
def test_label_binding_and_review_are_required(tmp_path: Path, changes: dict[str, object]) -> None:
    path = tmp_path / "labels.json"
    write_label(path, **changes)
    with pytest.raises(ValueError):
        load_labels(path)


def test_low_texture_and_wrong_photo_are_rejected() -> None:
    empty = numpy.full((450, 800), 120, dtype=numpy.uint8)
    source, target = feature_matches(label(), empty)
    assert evaluate_geometry(source, target, label(), empty.shape)["status"] == "rejected"
    photo = PHOTOGRAPHS[-1]
    image = cv2.imread(photo.source, cv2.IMREAD_GRAYSCALE)
    assert image is not None
    crop = photo.crop
    canvas = numpy.full((450, 800), 220, dtype=numpy.uint8)
    canvas[100 : 100 + crop.height_px, 200 : 200 + crop.width_px] = image[
        crop.top_px : crop.top_px + crop.height_px, crop.left_px : crop.left_px + crop.width_px
    ]
    source, target = feature_matches(label(), canvas)
    assert evaluate_geometry(source, target, label(), canvas.shape)["status"] == "rejected"


@pytest.mark.parametrize("damage", ["repeated", "occluded", "partial"])
def test_repeated_occluded_or_partial_image_rejected(damage: str) -> None:
    photo = PHOTOGRAPHS[0]
    source_image = cv2.imread(photo.source, cv2.IMREAD_GRAYSCALE)
    assert source_image is not None
    crop = photo.crop
    original = source_image[
        crop.top_px : crop.top_px + crop.height_px, crop.left_px : crop.left_px + crop.width_px
    ]
    canvas = numpy.full((450, 800), 230, dtype=numpy.uint8)
    if damage == "partial":
        canvas[100:304, :100] = original[:, 100:200]
    else:
        canvas[100:304, 200:400] = original
        if damage == "repeated":
            canvas[100:304, 500:700] = original
        else:
            canvas[100:304, 290:400] = 230
    source, target = feature_matches(label(), cast(NDArray[numpy.uint8], canvas))
    assert evaluate_geometry(source, target, label(), canvas.shape)["status"] == "rejected"


@pytest.mark.parametrize(
    ("frame", "asset_index", "bounds"),
    [
        (720, 0, (0.0, 0.0, 200.0, 204.0)),
        (720, 1, (0.0, 0.0, 121.0, 125.0)),
        (1032, 4, (37.0, 0.0, 125.0, 75.0)),
    ],
)
def test_recorded_features_with_test_only_labels(
    frame: int, asset_index: int, bounds: tuple[float, float, float, float]
) -> None:
    directory = Path(
        "outputs/validation/monitor-20261027/outputs/monitor-hardware-clean-campaign/"
        "20261027T110751Z-63626eaf/demo"
    )
    path = directory / f"frame_{frame:09}.jpg"
    if not path.exists():
        pytest.skip("optional recorded evidence is not present")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    fixture = Label(PHOTOGRAPHS[asset_index], bounds, 10, 2, "test fixture; not reviewed")
    source, target = feature_matches(fixture, cast(NDArray[numpy.uint8], image))
    assert evaluate_geometry(source, target, fixture, image.shape)["status"] == "candidate"
