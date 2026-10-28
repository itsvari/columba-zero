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
    camera_features,
    digest,
    evaluate_geometry,
    load_labels,
    match_features,
    source_features,
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
    check = CameraCheck("frame.jpg", PHOTOGRAPHS[0].asset_id, "a" * 64, corners + 10, 1, 0, "test")
    checked = apply_camera_check(original, check, "a" * 64, 2)
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
    source, target = match_features(source_features(label()), camera_features(empty))
    assert evaluate_geometry(source, target, label(), empty.shape)["status"] == "rejected"
    photo = PHOTOGRAPHS[-1]
    image = cv2.imread(photo.source, cv2.IMREAD_GRAYSCALE)
    assert image is not None
    crop = photo.crop
    canvas = numpy.full((450, 800), 220, dtype=numpy.uint8)
    canvas[100 : 100 + crop.height_px, 200 : 200 + crop.width_px] = image[
        crop.top_px : crop.top_px + crop.height_px, crop.left_px : crop.left_px + crop.width_px
    ]
    source, target = match_features(source_features(label()), camera_features(canvas))
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
    source, target = match_features(
        source_features(label()), camera_features(cast(NDArray[numpy.uint8], canvas))
    )
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
    source, target = match_features(
        source_features(fixture), camera_features(cast(NDArray[numpy.uint8], image))
    )
    assert evaluate_geometry(source, target, fixture, image.shape)["status"] == "candidate"


def test_camera_hash_budget_and_corner_allowance() -> None:
    source, target = points()
    original = evaluate_geometry(source, target, label(), (400, 500))
    corners = numpy.asarray(original["photograph_corners_camera_px"], dtype=numpy.float32)
    image_sha256 = "a" * 64
    check = CameraCheck(
        "frame.jpg", PHOTOGRAPHS[0].asset_id, image_sha256, corners + 1, 2, 1, "independent review"
    )
    wrong_hash = apply_camera_check(original, check, "b" * 64, 2)
    assert wrong_hash["status"] == "rejected"
    assert wrong_hash["reason"] == "camera check image SHA-256 mismatch"
    over_budget = apply_camera_check(original, check, image_sha256, 1)
    assert over_budget["status"] == "rejected"
    assert over_budget["reason"] == "camera check threshold exceeds registration allowance"
    result = apply_camera_check(original, check, image_sha256, 2)
    assert result["status"] == "rejected"
    evidence = result["camera_check"]
    assert isinstance(evidence, dict)
    assert evidence["raw_maximum_error_px"] == pytest.approx(2**0.5)
    assert evidence["worst_case_maximum_error_px"] == pytest.approx(8**0.5)
    assert evidence["corner_allowance_px"] == 1
    passing = CameraCheck(
        "frame.jpg", PHOTOGRAPHS[0].asset_id, image_sha256, corners + 1, 3, 1, "independent review"
    )
    assert apply_camera_check(original, passing, image_sha256, 3)["status"] == "camera_checked"


def test_missing_hash_and_bad_annotation_allowance_are_rejected(tmp_path: Path) -> None:
    from tools.locate_monitor_birds import load_checks

    entry: dict[str, object] = {
        "image": "frame.jpg",
        "asset_id": PHOTOGRAPHS[0].asset_id,
        "corners_camera_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "maximum_error_px": 2,
        "reviewer": "test",
    }
    path = tmp_path / "checks.json"
    path.write_text(json.dumps({"checks": [entry]}))
    with pytest.raises(ValueError, match="image_sha256"):
        load_checks(path)
    entry.update({"image_sha256": "a" * 64, "corner_allowance_px": -1})
    path.write_text(json.dumps({"checks": [entry]}))
    with pytest.raises(ValueError, match="nonnegative"):
        load_checks(path)
    del entry["corner_allowance_px"]
    path.write_text(json.dumps({"checks": [entry]}))
    assert load_checks(path)[("frame.jpg", PHOTOGRAPHS[0].asset_id)].corner_allowance_px == 0


def test_cli_rejects_mismatched_image_before_feature_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from tools import locate_monitor_birds

    image = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image), numpy.full((30, 30), 128, dtype=numpy.uint8))
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"filename": image.name}) + "\n")
    labels = tmp_path / "labels.json"
    write_label(labels)
    checks = tmp_path / "checks.json"
    checks.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "image": image.name,
                        "image_sha256": "a" * 64,
                        "asset_id": PHOTOGRAPHS[0].asset_id,
                        "corners_camera_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "maximum_error_px": 2,
                        "reviewer": "test",
                    }
                ]
            }
        )
    )
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "locate_monitor_birds",
            str(index),
            "--source-labels",
            str(labels),
            "--camera-checks",
            str(checks),
            "--output",
            str(output),
        ],
    )

    def fail_if_called(image: NDArray[numpy.uint8]) -> None:
        raise AssertionError("feature matching should not run for mismatched camera check")

    monkeypatch.setattr(locate_monitor_birds, "camera_features", fail_if_called)
    assert locate_monitor_birds.main() == 0
    results = json.loads((output / "locations.json").read_text())["results"]
    assert results[0]["status"] == "rejected"
    assert results[0]["reason"] == "camera check image SHA-256 mismatch"


def test_cli_extracts_each_source_once_and_each_image_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from tools import locate_monitor_birds

    images = ["first.jpg", "second.jpg"]
    for name in images:
        assert cv2.imwrite(str(tmp_path / name), numpy.full((30, 30), 128, dtype=numpy.uint8))
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps({"filename": name}) + "\n" for name in images))
    labels = tmp_path / "labels.json"
    write_label(labels)
    entries = json.loads(labels.read_text())["labels"]
    photo = PHOTOGRAPHS[1]
    entries.append(
        {
            "asset_id": photo.asset_id,
            "source_sha256": digest(Path(photo.source)),
            "crop_xywh_px": [168, 620, 121, 125],
            "visible_bbox_xyxy_px": [0, 0, 121, 125],
            "edge_allowance_source_px": 5,
            "registration_allowance_camera_px": 2,
            "confirmed": True,
            "reviewer": "synthetic test",
        }
    )
    labels.write_text(json.dumps({"labels": entries}))
    source_calls: list[str] = []
    camera_calls: list[int] = []
    original_source = locate_monitor_birds.source_features
    original_camera = locate_monitor_birds.camera_features

    def count_source(label: Label) -> locate_monitor_birds.Features:
        source_calls.append(label.photograph.asset_id)
        return original_source(label)

    def count_camera(image: NDArray[numpy.uint8]) -> locate_monitor_birds.Features:
        camera_calls.append(1)
        return original_camera(image)

    monkeypatch.setattr(locate_monitor_birds, "source_features", count_source)
    monkeypatch.setattr(locate_monitor_birds, "camera_features", count_camera)
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "locate_monitor_birds",
            str(index),
            "--source-labels",
            str(labels),
            "--output",
            str(output),
        ],
    )
    assert locate_monitor_birds.main() == 0
    results = json.loads((output / "locations.json").read_text())["results"]
    assert len(results) == 4
    assert all(result["status"] == "rejected" for result in results)
    assert source_calls == [PHOTOGRAPHS[0].asset_id, PHOTOGRAPHS[1].asset_id]
    assert len(camera_calls) == len(images)
