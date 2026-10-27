from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from tools.register_monitor_frames import Json, main, process_frame, project


@pytest.fixture
def example(tmp_path: Path) -> tuple[dict[str, Json], Path]:
    image_path = tmp_path / "camera.jpg"
    assert cv2.imwrite(str(image_path), np.full((600, 800, 3), 100, dtype=np.uint8))
    rectangles: list[Json] = []
    for marker_id, left, top in (
        ("top-left", 24, 24),
        ("top-right", 1512, 24),
        ("bottom-right", 1512, 812),
        ("bottom-left", 24, 812),
    ):
        rectangles.append(
            {
                "marker_id": marker_id,
                "box_screen_css_px": {
                    "left_px": left,
                    "top_px": top,
                    "width_px": 64,
                    "height_px": 64,
                },
            }
        )
    frame: dict[str, Json] = {
        "schema_version": 2,
        "kind": "frame",
        "presentation_id": "test",
        "scene_id": "scene",
        "render_number": 123,
        "sequence_frame": 7,
        "registration_markers": rectangles,
        "target_region_screen_css_px": {
            "left_px": 160,
            "top_px": 180,
            "width_px": 1280,
            "height_px": 540,
        },
    }
    matrix = np.array(((0.36, 0.02, 40), (0.01, 0.4, 30), (0.00012, 0.00006, 1.0)))
    marker_centres = ((56, 56), (1544, 56), (1544, 844), (56, 844))
    marker_ids = ("top-left", "top-right", "bottom-right", "bottom-left")
    camera_markers: dict[str, Json] = {
        marker_id: list(project(matrix, position))
        for marker_id, position in zip(marker_ids, marker_centres, strict=True)
    }
    screen_checks = ((240, 220), (1360, 220), (1360, 680), (240, 680), (800, 450))
    checks: list[Json] = [
        {
            "name": str(index),
            "screen_css_px": list(position),
            "camera_px": list(project(matrix, position)),
        }
        for index, position in enumerate(screen_checks)
    ]
    data: dict[str, Json] = {
        "image": image_path.name,
        "image_sequence": 42,
        "stimulus_frame": frame,
        "marker_identities_confirmed": True,
        "all_markers_fully_visible": True,
        "validation_points_independently_observed": True,
        "marker_centres_camera_px": camera_markers,
        "validation_points": checks,
        "screen_points": [[800, 450]],
    }
    return data, tmp_path


def test_perspective_independent_validation_and_transform(
    example: tuple[dict[str, Json], Path],
) -> None:
    data, directory = example
    result = process_frame(data, directory, directory, 0.1)
    assert result["status"] == "validated_within_sampled_hull"
    assert result["image"] == "camera.jpg"
    assert result["image_sequence"] == 42
    assert cast(float, result["max_validation_error_px"]) < 0.1
    transformed = cast(list[dict[str, Json]], result["transformed_points"])
    check = cast(list[dict[str, Json]], data["validation_points"])[4]
    assert cast(list[float], transformed[0]["camera_px"]) == pytest.approx(
        check["camera_px"], abs=0.1
    )
    assert Path(cast(str, result["overlay"])).is_file()


def test_four_marker_fit_does_not_validate_curvature(example: tuple[dict[str, Json], Path]) -> None:
    data, directory = example
    checks = cast(list[dict[str, Json]], data["validation_points"])
    centre = cast(list[float], checks[4]["camera_px"])
    centre[1] += 20
    result = process_frame(data, directory, directory, 2)
    assert result["status"] == "unvalidated"
    assert max(cast(list[float], result["fit_residuals_px"])) < 0.1
    assert cast(float, result["max_validation_error_px"]) > 19
    assert result["transformed_points"] == []


def test_missing_checks_and_outside_hull_fail_closed(example: tuple[dict[str, Json], Path]) -> None:
    data, directory = example
    data["validation_points"] = []
    missing = process_frame(data, directory, directory, 2)
    assert missing["status"] == "unvalidated"
    assert missing["transformed_points"] == []
    assert "five independent" in str(missing["failures"])
    data, _ = example_with_checks(example)
    data["screen_points"] = [[170, 200]]
    outside = process_frame(data, directory, directory, 2)
    assert outside["status"] == "unvalidated"
    assert outside["transformed_points"] == []
    assert "outside the validation hull" in str(outside["failures"])


def example_with_checks(example: tuple[dict[str, Json], Path]) -> tuple[dict[str, Json], Path]:
    data, directory = example
    checks = ((240, 220), (1360, 220), (1360, 680), (240, 680), (800, 450))
    matrix = np.array(((0.36, 0.02, 40), (0.01, 0.4, 30), (0.00012, 0.00006, 1.0)))
    data["validation_points"] = [
        {
            "name": str(index),
            "screen_css_px": list(position),
            "camera_px": list(project(matrix, position)),
        }
        for index, position in enumerate(checks)
    ]
    return data, directory


@pytest.mark.parametrize("failure", ["partial", "crossed", "unconfirmed", "schema", "geometry"])
def test_invalid_marker_evidence_is_rejected(
    example: tuple[dict[str, Json], Path], failure: str
) -> None:
    data, directory = example
    markers = cast(dict[str, Json], data["marker_centres_camera_px"])
    frame = cast(dict[str, Json], data["stimulus_frame"])
    if failure == "partial":
        del markers["top-left"]
    elif failure == "crossed":
        markers["top-left"], markers["bottom-right"] = markers["bottom-right"], markers["top-left"]
    elif failure == "unconfirmed":
        data["marker_identities_confirmed"] = False
    elif failure == "schema":
        frame["schema_version"] = 1
    else:
        region = cast(dict[str, Json], frame["target_region_screen_css_px"])
        region["width_px"] = 200
    result = process_frame(data, directory, directory, 2)
    assert result["status"] == "failed"
    assert result["transformed_points"] == []
    assert result["failures"]
    assert Path(cast(str, result["overlay"])).is_file()


def test_insufficient_spatial_coverage_fails(example: tuple[dict[str, Json], Path]) -> None:
    data, directory = example
    checks = cast(list[dict[str, Json]], data["validation_points"])
    checks.pop(0)
    checks.append(copy.deepcopy(checks[0]))
    checks[-1]["name"] = "duplicate-position"
    result = process_frame(data, directory, directory, 1)
    assert result["status"] == "unvalidated"
    assert "distinct" in str(result["failures"])


def test_cli_keeps_each_frame_and_returns_failure_for_partial_evidence(
    example: tuple[dict[str, Json], Path],
) -> None:
    data, directory = example
    partial = copy.deepcopy(data)
    partial["image_sequence"] = 43
    partial["validation_points"] = []
    annotation_path = directory / "annotations.json"
    annotation_path.write_text(json.dumps({"frames": [data, partial]}), encoding="utf-8")
    output = directory / "report"
    assert main([str(annotation_path), "--output", str(output), "--max-error-px", "1"]) == 1
    report = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    assert [frame["image_sequence"] for frame in report["frames"]] == [42, 43]
    assert [frame["status"] for frame in report["frames"]] == [
        "validated_within_sampled_hull",
        "unvalidated",
    ]
    assert "not optically verified" in report["association"]


def test_independent_observation_requires_confirmation(
    example: tuple[dict[str, Json], Path],
) -> None:
    data, directory = example
    data["validation_points_independently_observed"] = False
    result = process_frame(data, directory, directory, 1)
    assert result["status"] == "unvalidated"
    assert result["transformed_points"] == []
    assert "independent camera observation" in str(result["failures"])


def test_degenerate_or_nonfinite_camera_points_fail(example: tuple[dict[str, Json], Path]) -> None:
    data, directory = example
    markers = cast(dict[str, Json], data["marker_centres_camera_px"])
    markers["top-left"] = markers["top-right"]
    assert process_frame(data, directory, directory, 1)["status"] == "failed"
    markers["top-left"] = [float("nan"), 1]
    assert process_frame(data, directory, directory, 1)["status"] == "failed"


def test_partial_validation_is_reported_without_transform(
    example: tuple[dict[str, Json], Path],
) -> None:
    data, directory = example
    checks = cast(list[dict[str, Json]], data["validation_points"])
    checks.pop()
    result = process_frame(data, directory, directory, 1)
    assert result["status"] == "unvalidated"
    assert len(cast(list[Json], result["validation_residuals"])) == 4
    assert result["transformed_points"] == []


def test_invalid_error_limit_fails_closed(example: tuple[dict[str, Json], Path]) -> None:
    data, directory = example
    for limit_px in (0.0, -1.0, float("nan"), float("inf")):
        result = process_frame(data, directory, directory, limit_px)
        assert result["status"] == "failed"
        assert result["transformed_points"] == []
