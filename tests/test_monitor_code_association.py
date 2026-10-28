from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.json_data import JsonValue
from tools import associate_monitor_codes as codes
from tools import detect_monitor_markers
from tools.associate_monitor_frames import Image, camera_matrix, identity_bytes
from tools.monitor_stimulus import build_manifest
from tools.register_monitor_frames import project

CELL_DESIGN_PX = 12
MARKER_QUIET_DESIGN_PX = 8
CAMERA_DIMENSIONS_PX = (800, 450)
CAMERA_MATRIX = np.array(((0.36, 0.004, 60), (0.002, 0.36, 30), (0.000001, 0.000004, 1.0)))


@dataclass(frozen=True)
class Capture:
    index: Path
    browser: Path
    markers: Path
    report: Path
    image: Path
    frame: dict[str, JsonValue]


def stimulus(
    render: int = 1, *, torn: bool = False, corrupt: bool = False
) -> tuple[Image, dict[str, JsonValue], dict[str, JsonValue]]:
    manifest = build_manifest("tuning", registration_grid=True, optical_code=True)
    session = "0123456789abcdef"
    payload = identity_bytes(manifest.presentation_id, session, 2, render)
    design = np.full((900, 1600, 3), 18, dtype=np.uint8)
    design[180:720, 160:1440] = 184
    cv2.putText(
        design,
        "STATIC MONITOR BOUNDARY",
        (210, 850),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (255, 255, 255),
        3,
    )
    markers: list[JsonValue] = []
    centres: dict[str, JsonValue] = {}
    for marker in manifest.registration_markers:
        left, top = round(marker.bounds.left_px), round(marker.bounds.top_px)
        design[top : top + 64, left : left + 64] = 255
        for row, pattern in enumerate(marker.pattern):
            for column, cell in enumerate(pattern):
                if cell == "1":
                    horizontal = left + MARKER_QUIET_DESIGN_PX + column * 9.6
                    vertical = top + MARKER_QUIET_DESIGN_PX + row * 9.6
                    design[
                        round(vertical) : round(vertical + 9.6),
                        round(horizontal) : round(horizontal + 9.6),
                    ] = 0
        markers.append(
            {
                "marker_id": marker.marker_id,
                "box_screen_css_px": {
                    "left_px": left,
                    "top_px": top,
                    "width_px": 64,
                    "height_px": 64,
                },
            }
        )
        centres[marker.marker_id] = list(project(CAMERA_MATRIX, (left + 32, top + 32)))
    for top in (96, 740):
        band = (
            identity_bytes(manifest.presentation_id, session, 2, render + 1)
            if torn and top == 96
            else payload
        )
        bits = np.unpackbits(np.frombuffer(band, dtype=np.uint8)).copy()
        if corrupt:
            bits[41] ^= 1
        patch_pixels = np.repeat(
            np.repeat((1 - bits.reshape(6, 32)) * 255, CELL_DESIGN_PX, axis=0),
            CELL_DESIGN_PX,
            axis=1,
        )
        design[top : top + 72, 1056:1440] = patch_pixels[:, :, None]
    frame: dict[str, JsonValue] = {
        "kind": "frame",
        "presentation_id": manifest.presentation_id,
        "optical_session": session,
        "run_number": 2,
        "render_number": render,
        "scene_id": "synthetic-code-only",
        "scene_frame": render - 1,
        "sequence_frame": render - 1,
        "registration_markers": markers,
        "target_region_screen_css_px": {
            "left_px": 160,
            "top_px": 180,
            "width_px": 1280,
            "height_px": 540,
        },
        "optical_code_screen_css_px": {
            "left_px": 1056,
            "top_px": 740,
            "width_px": 384,
            "height_px": 72,
        },
        "optical_check_screen_css_px": {
            "left_px": 1056,
            "top_px": 96,
            "width_px": 384,
            "height_px": 72,
        },
    }
    image = cast(
        Image,
        cv2.warpPerspective(design, CAMERA_MATRIX, CAMERA_DIMENSIONS_PX, flags=cv2.INTER_AREA),
    )
    return image, frame, centres


def capture(directory: Path, *, torn: bool = False, corrupt: bool = False) -> Capture:
    directory.mkdir()
    image, frame, centres = stimulus(torn=torn, corrupt=corrupt)
    image_path = directory / "frame_000000057.jpg"
    assert cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    index, browser, markers, report = (
        directory / name for name in ("index.jsonl", "browser.jsonl", "markers.json", "review.json")
    )
    index.write_text(json.dumps({"filename": image_path.name, "sequence": 340}) + "\n")
    browser.write_text(json.dumps(frame) + "\n")
    markers.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "image": image_path.name,
                        "marker_centres_camera_px": centres,
                        "marker_identities_confirmed": False,
                        "all_markers_fully_visible": False,
                        "review_evidence": "synthetic reference-centroid provenance",
                    }
                ]
            }
        )
    )
    report.write_text('{"purpose":"provenance only"}\n')
    return Capture(index, browser, markers, report, image_path, frame)


def arguments(target: Capture, reference: Capture, output: Path) -> list[str]:
    return [
        str(target.index),
        str(target.browser),
        "--reference-index",
        str(reference.index),
        "--reference-browser",
        str(reference.browser),
        "--reference-markers",
        str(reference.markers),
        "--reference-report",
        str(reference.report),
        "--output",
        str(output),
    ]


def prepared(reference: Capture) -> tuple[codes.Reference, ...]:
    references, results = codes.prepare_references(
        codes.read_index(reference.index), codes.read_browser(reference.browser), reference.markers
    )
    assert cast(dict[str, JsonValue], results[0])["status"] == "optically_associated", results
    return references


def test_cli_direct_identity_and_reference_provenance_without_manual_claims(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    target = capture(tmp_path / "target")
    output = tmp_path / "result"
    assert codes.main(arguments(target, reference, output)) == 0
    result = json.loads((output / "association.json").read_text())
    frame = result["frames"][0]
    assert frame["status"] == "optically_associated"
    assert frame["source_sequence"] == 340
    assert frame["image_path"] == str(target.image)
    assert frame["image_sha256"] == codes.digest(target.image)
    assert frame["direct_optical_association"] is True
    assert result["inputs"]["reference_index"]["sha256"] == codes.digest(reference.index)
    assert result["inputs"]["reference_browser"]["sha256"] == codes.digest(reference.browser)
    assert result["inputs"]["reference_markers"]["sha256"] == codes.digest(reference.markers)
    assert result["inputs"]["reference_report"]["sha256"] == codes.digest(reference.report)
    assert result["reference_results"][0]["image_sha256"] == codes.digest(reference.image)
    assert (output / "association.sha256").read_text().split()[0] == codes.digest(
        output / "association.json"
    )
    assert "marker_identities_confirmed" not in json.dumps(result)
    assert "projected_photograph_centres" not in json.dumps(result)
    assert "registration" not in frame


def test_fallback_tracks_transformed_camera_template_with_partial_marker(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    references = prepared(reference)
    original, _ = codes.read_image(reference.image)
    camera_transform = np.array(((1.0, 0.008, 13), (-0.003, 1.0, 5), (0.000005, -0.00001, 1)))
    transformed = cast(
        Image,
        cv2.warpPerspective(original, camera_transform, CAMERA_DIMENSIONS_PX),
    )
    top_left = project(camera_transform, project(CAMERA_MATRIX, (56, 56)))
    cv2.rectangle(
        transformed,
        (round(top_left[0]) - 18, round(top_left[1]) - 18),
        (round(top_left[0]) + 18, round(top_left[1]) + 18),
        (18, 18, 18),
        -1,
    )
    result = codes.associate_image(transformed, codes.read_browser(reference.browser), references)
    assert result["status"] == "optically_associated", result
    assert result["direct_optical_association"] is False
    proposals = cast(list[dict[str, JsonValue]], result["proposals"])
    assert any(
        cast(int, proposal.get("inliers", 0)) >= codes.MINIMUM_INLIERS for proposal in proposals
    )
    assert result["distinct_crc_valid_band_payloads"] == 1


def test_torn_crc_valid_bands_are_rejected(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    image, _, _ = stimulus(torn=True)
    result = codes.associate_image(
        image, codes.read_browser(reference.browser), prepared(reference)
    )
    assert result["status"] == "rejected"
    assert result["optical_ambiguity_veto"] is True
    assert result["distinct_crc_valid_band_payloads"] == 2


def test_competing_payloads_across_proposals_cannot_be_selected_by_browser(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    references = prepared(reference)
    image, _ = codes.read_image(reference.image)
    browser = codes.read_browser(reference.browser)
    first = next(iter(browser.frames_by_code))
    second = identity_bytes(cast(str, reference.frame["presentation_id"]), "0123456789abcdef", 2, 2)

    def optical(payload: bytes) -> tuple[dict[str, JsonValue], bytes, bool]:
        return (
            {
                "bands": {
                    band: {"crc_valid": True, "payload_hex": payload.hex()}
                    for band in ("lower", "upper")
                }
            },
            payload,
            False,
        )

    with (
        patch.object(
            detect_monitor_markers, "detect", return_value=([], "ambiguous direct markers")
        ),
        patch.object(codes, "sift_proposal", return_value=({"method": "proposal"}, CAMERA_MATRIX)),
        patch.object(codes, "decode_pair", side_effect=(optical(first), optical(second))),
    ):
        result = codes.associate_image(image, browser, references + references)
    assert result["status"] == "rejected"
    assert result["distinct_dual_band_payloads"] == 2
    assert result["distinct_crc_valid_band_payloads"] == 2


def test_layout_mismatch_rejected_before_output(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    target = capture(tmp_path / "target")
    changed = dict(target.frame)
    changed["optical_check_screen_css_px"] = {
        "left_px": 1056,
        "top_px": 97,
        "width_px": 384,
        "height_px": 72,
    }
    target.browser.write_text(json.dumps(changed) + "\n")
    output = tmp_path / "result"
    with pytest.raises(SystemExit) as error:
        codes.main(arguments(target, reference, output))
    assert error.value.code == 2
    assert not output.exists()


def test_failed_reference_is_retained_as_rejected_and_never_used(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference", corrupt=True)
    references, results = codes.prepare_references(
        codes.read_index(reference.index), codes.read_browser(reference.browser), reference.markers
    )
    assert not references
    rejected = cast(dict[str, JsonValue], results[0])
    assert rejected["status"] == "rejected"
    assert rejected["image_sha256"] == codes.digest(reference.image)
    assert "dual-CRC" in cast(str, rejected["reason"])


def test_ambiguous_direct_markers_supply_no_sampling_geometry(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    image, _ = codes.read_image(reference.image)
    with patch.object(
        detect_monitor_markers,
        "detect",
        return_value=([], "expected exactly one complete candidate of each identity"),
    ):
        result = codes.associate_image(image, codes.read_browser(reference.browser), ())
    assert result["status"] == "rejected"
    assert result["proposals"] == []


def test_reference_target_pixels_do_not_affect_sift_descriptors(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    image, _ = codes.read_image(reference.image)
    markers = json.loads(reference.markers.read_text())["frames"][0]["marker_centres_camera_px"]
    matrix = camera_matrix(reference.frame, markers)
    changed = image.copy()
    polygon = codes.rectangle_polygon(reference.frame, "target_region_screen_css_px", matrix)
    cv2.fillConvexPoly(changed, polygon, (0, 0, 255))
    points, descriptors, details = codes.reference_features(image, reference.frame, matrix, markers)
    other_points, other_descriptors, other_details = codes.reference_features(
        changed, reference.frame, matrix, markers
    )
    assert [point.pt for point in points] == [point.pt for point in other_points]
    np.testing.assert_array_equal(descriptors, other_descriptors)
    assert details == other_details


def test_duplicate_export_identity_fails_closed(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    image, _ = codes.read_image(reference.image)
    references = prepared(reference)
    reference.browser.write_text(reference.browser.read_text() * 2)
    result = codes.associate_image(image, codes.read_browser(reference.browser), references)
    assert result["status"] == "rejected"
    assert result["browser_match_count"] == 2


def test_index_names_and_sequences_cannot_alias_sources(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    reference.index.write_text(reference.index.read_text() * 2)
    with pytest.raises(ValueError, match="unique"):
        codes.read_index(reference.index)


def test_corrupted_target_bits_are_not_repaired_from_export(tmp_path: Path) -> None:
    reference = capture(tmp_path / "reference")
    image, _, _ = stimulus(corrupt=True)
    result = codes.associate_image(
        image, codes.read_browser(reference.browser), prepared(reference)
    )
    assert result["status"] == "rejected"
    assert result["distinct_dual_band_payloads"] == 0
