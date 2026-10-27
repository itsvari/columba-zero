from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy
import pytest

from app.json_data import JsonValue
from tools import locate_monitor_birds as locator
from tools import prepare_monitor_truth as preparation
from tools import score_monitor_run as scorer
from tools.associate_monitor_frames import frame_identity
from tools.monitor_stimulus import Photograph, Rectangle
from tools.register_monitor_frames import project

STANDING = "tuning-standing-pigeon"
PROFILE = "tuning-profile-pigeon"
CONFUSER = "shorebird-confuser"
PRESENTATION = "a" * 64
SESSION = "b" * 16
IMAGE_WIDTH_PX = 800
IMAGE_HEIGHT_PX = 450


def frame(
    number: int, *, left_px: float = 300, asset: str = STANDING, identity: str = "incumbent"
) -> dict[str, JsonValue]:
    centres = ((56, 56), (1544, 56), (1544, 844), (56, 844))
    names = ("top-left", "top-right", "bottom-right", "bottom-left")
    return {
        "kind": "frame",
        "presentation_id": PRESENTATION,
        "optical_session": SESSION,
        "run_number": 1,
        "repeat_index": 0,
        "scene_id": "static-scene",
        "scene_frame": number - 1,
        "sequence_frame": number - 1,
        "render_number": number,
        "registration_markers": [
            {
                "marker_id": name,
                "box_screen_css_px": {
                    "left_px": x - 32,
                    "top_px": y - 32,
                    "width_px": 64,
                    "height_px": 64,
                },
            }
            for name, (x, y) in zip(names, centres, strict=True)
        ],
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
        "targets": [
            {
                "intended_identity": identity,
                "asset_id": asset,
                "subject": "other_bird" if asset == CONFUSER else "pigeon",
                "box_screen_css_px": {
                    "left_px": left_px,
                    "top_px": 150,
                    "width_px": 200,
                    "height_px": 204,
                },
                "present": True,
                "visibility": "visible",
                "visible_fraction": 1,
                "covered_fraction": 0,
            }
        ],
    }


def manifest(
    labels: dict[str, locator.Label], *, rules: list[JsonValue] | None = None
) -> dict[str, JsonValue]:
    return {
        "assets": [
            {
                "photograph": {
                    "asset_id": asset,
                    "crop": {
                        "left_px": label.photograph.crop.left_px,
                        "top_px": label.photograph.crop.top_px,
                        "width_px": label.photograph.crop.width_px,
                        "height_px": label.photograph.crop.height_px,
                    },
                },
                "source_sha256": preparation.digest(Path(label.photograph.source)),
            }
            for asset, label in labels.items()
        ],
        "scenes": [
            {
                "scene_id": "static-scene",
                "targets": [{"identity": "incumbent", "asset_id": STANDING}],
                "annotations": rules
                or [
                    {
                        "start_frame": 0,
                        "end_frame": 100,
                        "selected_identity": "incumbent",
                        "behaviour": "synthetic static rule",
                    }
                ],
            }
        ],
    }


@dataclass(frozen=True)
class Fixture:
    recording: scorer.Recording
    browser_path: Path
    association_path: Path
    location_path: Path
    labels_path: Path
    qualification_path: Path
    labels: dict[str, locator.Label]
    browser: preparation.Browser
    policy: preparation.Qualification


def local_row(
    recorded: scorer.RecordedFrame,
    label: locator.Label,
    *,
    status: str = "candidate",
    camera_left_px: float = 300,
) -> dict[str, JsonValue]:
    mapping = numpy.array(((0.3, 0, camera_left_px), (0, 0.4, 150), (0, 0, 1)), dtype=numpy.float64)
    crop = label.photograph.crop
    centre = project(mapping, preparation.source_centre(label))
    bounds = preparation.projected_bounds(mapping, preparation.source_centre(label), (10, 10), 7)
    corners = [
        list(project(mapping, position))
        for position in (
            (0, 0),
            (crop.width_px, 0),
            (crop.width_px, crop.height_px),
            (0, crop.height_px),
        )
    ]
    return {
        "image": recorded.filename,
        "image_sha256": recorded.jpeg_sha256,
        "asset_id": label.photograph.asset_id,
        "source_sha256": preparation.digest(Path(label.photograph.source)),
        "status": status,
        "reason": "synthetic failed local fit" if status == "rejected" else None,
        "matches": 40,
        "fit_count": 20,
        "fit_inliers": 18,
        "held_out_count": 20,
        "held_out_within_2px_fraction": 1,
        "held_out_median_error_px": 0.2,
        "source_hull_fraction": 0.4,
        "centre_inside_source_hull": True,
        "source_to_camera_homography": mapping.tolist(),
        "photograph_corners_camera_px": corners,
        "visible_bbox_centre_camera_px": list(centre),
        "centre_allowance_bounds_camera_xyxy_px": list(bounds),
        "edge_allowance_source_px": 10,
        "registration_allowance_camera_px": 7,
    }


def write_json(path: Path, value: JsonValue) -> None:
    path.write_text(json.dumps(value) + "\n")


def make_fixture(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    offsets_s: tuple[float, ...] = (0, 0.5, 1),
    missing_photo: bool = False,
) -> Fixture:
    directory.mkdir()
    labels: dict[str, locator.Label] = {}
    dimensions = {STANDING: (200, 204), PROFILE: (121, 125), CONFUSER: (125, 86)}
    for asset, (width_px, height_px) in dimensions.items():
        source = directory / f"{asset}.source"
        source.write_bytes(asset.encode())
        photo = Photograph(
            asset,
            str(source),
            width_px,
            height_px,
            Rectangle(0, 0, width_px, height_px),
            "other_bird" if asset == CONFUSER else "pigeon",
            "synthetic source",
        )
        bounds = (
            (37.0, 0.0, 125.0, 75.0)
            if asset == CONFUSER
            else (0.0, 0.0, float(width_px), float(height_px))
        )
        labels[asset] = locator.Label(photo, bounds, 10, 7, "synthetic reviewer")
    monkeypatch.setattr(
        locator, "PHOTOGRAPHS", tuple(label.photograph for label in labels.values())
    )
    labels_path = directory / "source-labels.json"
    write_json(
        labels_path,
        {
            "labels": [
                {
                    "asset_id": asset,
                    "source_sha256": preparation.digest(Path(label.photograph.source)),
                    "crop_xywh_px": [
                        0,
                        0,
                        label.photograph.crop.width_px,
                        label.photograph.crop.height_px,
                    ],
                    "visible_bbox_xyxy_px": list(label.bounds),
                    "edge_allowance_source_px": 10,
                    "registration_allowance_camera_px": 7,
                    "confirmed": True,
                    "reviewer": "synthetic reviewer",
                }
                for asset, label in labels.items()
            ]
        },
    )
    run = directory / "run"
    demo = run / "demo"
    demo.mkdir(parents=True)
    index: list[JsonValue] = []
    logs: list[JsonValue] = [
        {"event": "run_start", "source": "csi_camera", "config": {"pan_tilt": {"mode": "hardware"}}}
    ]
    jpeg = (
        b"\xff\xd8\xff\xc0\x00\x0b\x08"
        + IMAGE_HEIGHT_PX.to_bytes(2, "big")
        + IMAGE_WIDTH_PX.to_bytes(2, "big")
        + b"\x01\x01\x11\x00\xff\xd9"
    )
    for position, offset_s in enumerate(offsets_s):
        name = f"frame_{position:09d}.jpg"
        (demo / name).write_bytes(jpeg + str(position).encode())
        target: dict[str, JsonValue] = {
            "track": {"track_id": 1, "detection": {"bbox": [300, 150, 360, 232]}},
            "state": "tracking",
        }
        metadata: dict[str, JsonValue] = {
            "annotations_included": False,
            "target": target,
            "controller": {"mode": "hardware"},
        }
        index.append(
            {
                "filename": name,
                "sequence": position * 4,
                "monotonic_s": 10 + offset_s,
                "captured_at": 1000 + offset_s,
                "timebase": "delivery_monotonic",
                "metadata": metadata,
            }
        )
        logs.append(
            {
                "event": "frame",
                "captured_at": 1000 + offset_s,
                "target": target,
                "controller": {"mode": "hardware"},
                "follow_timebase": "sensor",
            }
        )
    logs.append(
        {
            "event": "run_end",
            "error": None,
            "log_dropped_records": 0,
            "demo_recording": {
                "frames_written": len(index),
                "frames_dropped": 0,
                "first_monotonic_s": 10 + offsets_s[0],
                "last_monotonic_s": 10 + offsets_s[-1],
                "error": None,
            },
        }
    )
    for path, rows in ((demo / "index.jsonl", index), (run / "detections.jsonl", logs)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    recording = scorer.load_recording(run)
    manifest_value = manifest(labels)
    browser_path = directory / "browser.jsonl"
    browser_rows: list[JsonValue] = [
        {
            "kind": "session",
            "presentation_id": PRESENTATION,
            "optical_session": SESSION,
            "manifest": manifest_value,
        }
    ]
    browser_rows.extend(frame(number) for number in range(1, len(index) + 1))
    browser_path.write_text("".join(json.dumps(row) + "\n" for row in browser_rows))
    browser = preparation.read_browser(browser_path, labels)
    association_rows: list[JsonValue] = []
    for recorded, render in zip(recording.frames, browser.renders.values(), strict=True):
        association_rows.append(
            {
                "image": recorded.filename,
                "image_path": str(demo.resolve() / recorded.filename),
                "source_sequence": recorded.sequence,
                "image_sha256": recorded.jpeg_sha256,
                "status": "optically_associated",
                "optical_code_hex": frame_identity(render.raw).hex(),
                "browser_frame_sha256": render.sha256,
                "association": {
                    key: render.raw.get(key)
                    for key in (
                        "presentation_id",
                        "optical_session",
                        "run_number",
                        "render_number",
                        "scene_id",
                        "scene_frame",
                        "sequence_frame",
                    )
                },
            }
        )
    association_path = directory / "association.json"
    write_json(
        association_path,
        {
            "inputs": {
                "source_index": {
                    "path": str((demo / "index.jsonl").resolve()),
                    "sha256": recording.index_sha256,
                },
                "source_browser": {"sha256": browser.sha256},
            },
            "frames": association_rows,
        },
    )
    location_path = directory / "locations.json"
    location_rows = [
        local_row(
            recorded,
            label,
            status="candidate" if asset == STANDING and not missing_photo else "rejected",
        )
        for recorded in recording.frames
        for asset, label in labels.items()
    ]
    write_json(
        location_path,
        {
            "index_sha256": recording.index_sha256,
            "source_labels_sha256": preparation.digest(labels_path),
            "code_sha256": preparation.digest(Path(locator.__file__)),
            "results": cast(JsonValue, location_rows),
        },
    )
    heldout = directory / "heldout.json"
    equivalence = directory / "equivalence.json"
    write_json(
        heldout,
        {
            "source_labels_sha256": preparation.digest(labels_path),
            "code_sha256": "c" * 64,
            "results": [
                {
                    "asset_id": asset,
                    "status": "camera_checked",
                    "camera_check": {"allowed_error_px": 7, "worst_case_maximum_error_px": 6},
                }
                for asset in (STANDING, CONFUSER)
            ],
        },
    )
    write_json(
        equivalence,
        {
            "code_sha256_before": "c" * 64,
            "code_sha256_after": preparation.digest(Path(locator.__file__)),
            "asset_image_results": 30,
            "exact_result_matches": 30,
            "mismatches": [],
            "labels_sha256": preparation.digest(labels_path),
        },
    )
    home = directory / "home"
    home.mkdir()
    home_image = home / "frame_000000383.jpg"
    home_image.write_bytes(jpeg)
    (home / "index.jsonl").write_text(
        json.dumps({"filename": home_image.name, "sequence": 2294}) + "\n"
    )
    axis = directory / "axis.json"
    write_json(axis, {"scope": "synthetic empirical sweep"})
    qualification_path = directory / "qualification.json"
    reach: dict[str, JsonValue] = {
        "home_reference_image": str(home_image),
        "home_reference_image_sha256": preparation.digest(home_image),
        "home_reference_source_sequence": 2294,
        "axis_measurement_report": str(axis),
        "axis_measurement_sha256": preparation.digest(axis),
        "home_css_to_camera": cast(JsonValue, numpy.eye(3).tolist()),
        "home_mapping_allowance_camera_px": 7,
        "pan_usable_shift_px": [-110, 87],
        "tilt_usable_shift_px": [-74, 72],
        "pan_outer_shift_px": [-125, 103],
        "tilt_outer_shift_px": [-123, 121],
        "layout_reference_frame": frame(1),
        "image_width_px": 800,
        "image_height_px": 450,
    }
    write_json(
        qualification_path,
        {
            "source_labels_sha256": preparation.digest(labels_path),
            "qualified_assets": [STANDING, CONFUSER],
            "registration_allowance_camera_px": 7,
            "settling_s": 6,
            "maximum_sample_gap_s": 1,
            "camera_validation_report": str(heldout),
            "camera_validation_sha256": preparation.digest(heldout),
            "feature_reuse_equivalence_report": str(equivalence),
            "feature_reuse_equivalence_sha256": preparation.digest(equivalence),
            "qualified_locator_code_sha256": preparation.digest(Path(locator.__file__)),
            "reach": reach,
        },
    )
    policy = preparation.read_qualification(
        qualification_path, labels_path, preparation.document(location_path)
    )
    return Fixture(
        recording,
        browser_path,
        association_path,
        location_path,
        labels_path,
        qualification_path,
        labels,
        browser,
        policy,
    )


def evidence(render: preparation.Render) -> preparation.RenderEvidence:
    return preparation.RenderEvidence(
        render,
        render.number,
        render.number,
        True,
        {"kind": "synthetic optical anchor", "browser_frame_sha256": render.sha256},
    )


def test_exact_and_invariant_content_brackets_preserve_render_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    frames = fixture.recording.frames
    first, last = fixture.browser.renders[(SESSION, 1)], fixture.browser.renders[(SESSION, 3)]
    anchors = {frames[0].sequence: evidence(first), frames[2].sequence: evidence(last)}
    result, mode = preparation.bracket_render(1, frames, anchors, fixture.browser)
    assert mode == "invariant_interval" and result is not None and not result.exact
    assert (result.first_render, result.last_render) == (1, 3)
    anchors[frames[2].sequence] = evidence(first)
    result, mode = preparation.bracket_render(1, frames, anchors, fixture.browser)
    assert mode == "invariant_exact" and result is not None and result.exact
    assert result.first_render == 1
    assert preparation.bracket_render(0, frames, anchors, fixture.browser)[1] == "exact"


@pytest.mark.parametrize("damage", ["moving", "rule", "missing", "group", "span"])
def test_unsafe_brackets_remain_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    renders = dict(fixture.browser.renders)
    if damage == "missing":
        del renders[(SESSION, 2)]
    if damage == "moving":
        renders[(SESSION, 2)] = preparation.make_render(
            frame(2, left_px=301), manifest(fixture.labels)
        )
    if damage == "rule":
        renders[(SESSION, 2)] = replace(
            renders[(SESSION, 2)], content="different expected-selection rule"
        )
    if damage == "group":
        renders[(SESSION, 3)] = replace(
            renders[(SESSION, 3)], group=(PRESENTATION, SESSION, 2, 0, "static-scene")
        )
    frames = fixture.recording.frames
    if damage == "span":
        frames = (frames[0], frames[1], replace(frames[2], monotonic_s=frames[0].monotonic_s + 2.1))
    anchors = {
        frames[0].sequence: evidence(renders[(SESSION, 1)]),
        frames[2].sequence: evidence(renders[(SESSION, 3)]),
    }
    assert (
        preparation.bracket_render(1, frames, anchors, replace(fixture.browser, renders=renders))[0]
        is None
    )


@pytest.mark.parametrize("damage", ["index", "jpeg", "duplicate", "path", "identity"])
def test_cross_run_alias_and_changed_association_provenance_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    document = preparation.document(fixture.association_path)
    rows = cast(list[dict[str, JsonValue]], document["frames"])
    if damage == "index":
        cast(dict[str, dict[str, JsonValue]], document["inputs"])["source_index"]["sha256"] = (
            "f" * 64
        )
    if damage == "jpeg":
        rows[0]["image_sha256"] = "f" * 64
    if damage == "duplicate":
        rows.append(rows[0])
    if damage == "path":
        rows[0]["image_path"] = str(tmp_path / "another-run" / cast(str, rows[0]["image"]))
    if damage == "identity":
        cast(dict[str, JsonValue], rows[0]["association"])["render_number"] = 99
    write_json(fixture.association_path, document)
    with pytest.raises(ValueError):
        preparation.bind_associations(fixture.association_path, fixture.recording, fixture.browser)


def test_changed_location_hash_or_duplicate_asset_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    locations = preparation.document(fixture.location_path)
    rows = cast(list[dict[str, JsonValue]], locations["results"])
    rows[0]["image_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        preparation.bind_locations(locations, fixture.recording, fixture.labels)
    rows[0]["image_sha256"] = fixture.recording.frames[0].jpeg_sha256
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        preparation.bind_locations(locations, fixture.recording, fixture.labels)


def test_source_bird_box_and_mapped_edge_allowance_not_photo_centre(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    source = fixture.labels[CONFUSER]
    recorded = fixture.recording.frames[0]
    row = local_row(recorded, source)
    photo = preparation.local_photo(row, source, recorded, fixture.policy)
    assert photo is not None
    assert photo.centre_px == pytest.approx((324.3, 165.0))
    assert photo.centre_px != pytest.approx((318.75, 167.2))
    assert (
        photo.centre_px[0] - photo.bounds_px[0],
        photo.centre_px[1] - photo.bounds_px[1],
    ) == pytest.approx((10, 11))


@pytest.mark.parametrize(
    "centre_x_px,expected", [(400, True), (230, None), (100, False), (700, False)]
)
def test_reach_is_central_band_existence_with_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, centre_x_px: float, expected: bool | None
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    render = fixture.browser.renders[(SESSION, 1)]
    target = replace(render.targets[0], box_css_px=(centre_x_px - 100, 123, 200, 204))
    reachable, details = preparation.reach_bounds(
        target, render, fixture.labels[STANDING], fixture.policy
    )
    assert reachable is expected
    assert details["home_centre_bounds_camera_xyxy_px"] == pytest.approx(
        [centre_x_px - 17, 208, centre_x_px + 17, 242]
    )


def test_wrong_selected_photo_stays_wrong_and_overlap_ambiguity_stays_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    render = fixture.browser.renders[(SESSION, 1)]
    challenger = replace(render.targets[0], identity="challenger", asset=PROFILE)

    def photo(asset: str, left_px: float) -> preparation.Photo:
        return preparation.Photo(
            asset,
            (left_px + 20, 220),
            (left_px + 10, 210, left_px + 30, 230),
            ((left_px, 200), (left_px + 40, 200), (left_px + 40, 240), (left_px, 240)),
            "f" * 64,
        )

    photos = {STANDING: photo(STANDING, 300), PROFILE: photo(PROFILE, 500)}
    selected, _ = preparation.selection_mapping(
        (500, 200, 540, 240), (render.targets[0], challenger), photos
    )
    assert selected == "challenger" and selected != render.expected_identity
    photos[PROFILE] = photo(PROFILE, 300)
    assert (
        preparation.selection_mapping(
            (300, 200, 340, 240), (render.targets[0], challenger), photos
        )[0]
        is None
    )


def test_known_scene_missing_expected_photo_retains_partial_inventory_and_post_settle_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(
        tmp_path / "fixture",
        monkeypatch,
        offsets_s=tuple(float(index) for index in range(9)),
        missing_photo=True,
    )
    output = tmp_path / "prepared"
    assert (
        preparation.main(
            [
                str(fixture.recording.directory),
                str(fixture.browser_path),
                "--associations",
                str(fixture.association_path),
                "--locations",
                str(fixture.location_path),
                "--source-labels",
                str(fixture.labels_path),
                "--qualification",
                str(fixture.qualification_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    annotations = preparation.document(output / "annotations.json")
    rows = cast(list[dict[str, JsonValue]], annotations["frames"])
    label = cast(dict[str, JsonValue], rows[0]["label"])
    target = cast(list[dict[str, JsonValue]], label["targets"])[0]
    assert label["inventory_complete"] is True and label["expected_identity"] == "incumbent"
    assert target["identity"] == "incumbent" and target["visible"] is None
    assert target["registration_valid"] is False and target["centre_px"] is None
    score = preparation.document(output / "score.json")
    coverage = cast(dict[str, JsonValue], score["coverage"])
    exclusions = cast(dict[str, JsonValue], coverage["excluded_duration_s_by_reason"])
    assert exclusions["settling"] == pytest.approx(6)
    assert exclusions["visibility_unresolved"] == pytest.approx(2)
    assert coverage["unknown_duration_s"] == pytest.approx(2)
    assert score["global_verdict"] == "inconclusive"


def add_direct_qualification(fixture: Fixture, damage: str | None = None) -> Path:
    directory = fixture.qualification_path.parent
    checks_path, locations_path, comparison_path = (
        directory / name
        for name in ("direct-checks.json", "direct-locations.json", "direct-comparison.json")
    )
    first, second = fixture.recording.frames[:2]
    accepted = local_row(first, fixture.labels[PROFILE])
    rejected = local_row(second, fixture.labels[PROFILE], status="rejected")
    predicted = cast(list[float], accepted["visible_bbox_centre_camera_px"])
    bounds = [predicted[0] - 10, predicted[1] - 10, predicted[0] + 10, predicted[1] + 10]
    observed = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    checks: dict[str, JsonValue] = {
        "checks": [
            {
                "image": first.filename,
                "image_sha256": first.jpeg_sha256,
                "asset_id": PROFILE,
                "visible_bbox_camera_xyxy_px": cast(JsonValue, bounds),
                "edge_allowance_px": 2,
                "reviewer": "synthetic independent annotator",
            },
            {
                "image": second.filename,
                "image_sha256": second.jpeg_sha256,
                "asset_id": PROFILE,
                "visible_bbox_camera_xyxy_px": cast(JsonValue, bounds),
                "edge_allowance_px": 2,
                "reviewer": "synthetic independent annotator",
            },
        ]
    }
    locations: dict[str, JsonValue] = {
        "index_sha256": "d" * 64,
        "source_labels_sha256": preparation.digest(fixture.labels_path),
        "code_sha256": preparation.digest(Path(locator.__file__)),
        "results": [accepted, rejected],
    }
    error_px = float(numpy.linalg.norm(numpy.abs(numpy.asarray(predicted) - observed) + 2))
    results: list[JsonValue] = [
        {
            "image": first.filename,
            "asset_id": PROFILE,
            "locator_status": "candidate",
            "observed_centre_px": list(observed),
            "predicted_centre_px": cast(JsonValue, predicted),
            "annotation_edge_allowance_px": 2,
            "worst_case_error_px": error_px,
            "registration_budget_px": 7,
            "status": "within_budget",
        },
        {
            "image": second.filename,
            "asset_id": PROFILE,
            "locator_status": "rejected",
            "status": "rejected_by_locator",
        },
    ]
    if damage == "budget":
        cast(dict[str, JsonValue], results[0])["worst_case_error_px"] = 8
    if damage == "drop_rejection":
        results.pop()
    write_json(checks_path, checks)
    write_json(locations_path, locations)
    comparison: dict[str, JsonValue] = {
        "checks_sha256": preparation.digest(checks_path),
        "locations_sha256": preparation.digest(locations_path),
        "results": results,
    }
    write_json(comparison_path, comparison)
    if damage == "hash":
        checks_path.write_text(checks_path.read_text() + " ")
    policy = preparation.document(fixture.qualification_path)
    heldout_path = Path(cast(str, policy["camera_validation_report"]))
    heldout = preparation.document(heldout_path)
    heldout["index_sha256"] = "d" * 64
    write_json(heldout_path, heldout)
    policy["camera_validation_sha256"] = preparation.digest(heldout_path)
    policy["qualified_assets"] = [STANDING, CONFUSER, PROFILE]
    policy["qualified_direct_centre_assets"] = [PROFILE]
    policy["direct_centre_validation_report"] = str(comparison_path)
    policy["direct_centre_validation_sha256"] = preparation.digest(comparison_path)
    policy["direct_centre_checks"] = str(checks_path)
    policy["direct_centre_locations"] = str(locations_path)
    policy["direct_centre_scope"] = (
        "synthetic limited direct-centre observation, not photograph corners"
    )
    target = directory / "profile-qualification.json"
    write_json(target, policy)
    return target


def test_direct_profile_qualification_preserves_rejections_and_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    path = add_direct_qualification(fixture)
    policy = preparation.read_qualification(
        path, fixture.labels_path, preparation.document(fixture.location_path)
    )
    assert PROFILE in policy.qualified_assets
    assert policy.proof_hashes["direct_centre_accepted_by_asset"] == {PROFILE: 1}
    assert policy.proof_hashes["direct_centre_locator_rejections_by_asset"] == {PROFILE: 1}
    assert policy.proof_hashes["photo_corner_validation_claimed"] is False
    assert PROFILE not in fixture.policy.qualified_assets


@pytest.mark.parametrize("damage", ["budget", "drop_rejection", "hash"])
def test_direct_profile_evidence_cannot_lose_rejections_or_exceed_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    path = add_direct_qualification(fixture, damage)
    with pytest.raises(ValueError):
        preparation.read_qualification(
            path, fixture.labels_path, preparation.document(fixture.location_path)
        )


def test_selected_bbox_changes_only_identity_mapping_not_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    raw = frame(1)
    raw_targets = cast(list[dict[str, JsonValue]], raw["targets"])
    challenger = dict(raw_targets[0])
    challenger["intended_identity"] = "challenger"
    challenger["asset_id"] = PROFILE
    raw_targets.append(challenger)
    declared = manifest(fixture.labels)
    scene = cast(list[dict[str, JsonValue]], declared["scenes"])[0]
    scene["targets"] = [
        {"identity": "incumbent", "asset_id": STANDING},
        {"identity": "challenger", "asset_id": PROFILE},
    ]
    render = preparation.make_render(raw, declared)
    recorded = fixture.recording.frames[0]
    rows = {
        STANDING: local_row(recorded, fixture.labels[STANDING]),
        PROFILE: local_row(recorded, fixture.labels[PROFILE], camera_left_px=500),
    }
    first, _ = preparation.prepare_label(
        evidence(render), recorded, rows, fixture.labels, fixture.policy, (300, 150, 360, 232)
    )
    wrong, _ = preparation.prepare_label(
        evidence(render), recorded, rows, fixture.labels, fixture.policy, (500, 150, 537, 201)
    )
    assert first is not None and wrong is not None
    assert first["targets"] == wrong["targets"]
    assert wrong["expected_identity"] == "incumbent"
    assert cast(dict[str, JsonValue], wrong["selection"])["identity"] == "challenger"
    profile_target = cast(list[dict[str, JsonValue]], wrong["targets"])[1]
    assert profile_target["visible"] is True
    assert profile_target["registration_valid"] is False and profile_target["centre_px"] is None


def test_browser_counter_and_inventory_cannot_be_invented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path / "fixture", monkeypatch)
    rows = fixture.browser_path.read_text().splitlines()
    fixture.browser_path.write_text("\n".join((rows[0], rows[1], rows[3], rows[2])) + "\n")
    with pytest.raises(ValueError, match="increase strictly"):
        preparation.read_browser(fixture.browser_path, fixture.labels)
    raw = frame(1)
    raw["targets"] = []
    with pytest.raises(ValueError, match="intended target slot"):
        preparation.make_render(raw, manifest(fixture.labels))
