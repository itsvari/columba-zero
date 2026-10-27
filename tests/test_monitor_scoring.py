"""Synthetic acceptance fixtures; none of these labels describe hardware results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.json_data import JsonValue
from tools.score_monitor_run import (
    Recording,
    load_recording,
    main,
    make_review_template,
    read_json,
    score_recording,
)

IMAGE_WIDTH_PX = 800
IMAGE_HEIGHT_PX = 450
START_MONOTONIC_S = 10.0
CAPTURE_UNIX_OFFSET_S = 1000.0
SEQUENCE_STEP = 4
POLICY_EVIDENCE = "synthetic fixture: policy fixed before reviewing labels"


def mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def entries(document: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    values = document["frames"]
    assert isinstance(values, list)
    return [mapping(value) for value in values]


def write_records(path: Path, records: list[JsonValue]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def read_records(path: Path) -> list[JsonValue]:
    return [read_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_recording(
    directory: Path,
    offsets_s: tuple[float, ...] = (0.0, 1.0, 2.0),
    selected_tracks: tuple[int | None, ...] = (1, 1, 1),
    mode: str = "hardware",
) -> Recording:
    demo_directory = directory / "demo"
    demo_directory.mkdir()
    index: list[JsonValue] = []
    records: list[JsonValue] = [
        {
            "event": "run_start",
            "source": "csi_camera",
            "config": {"pan_tilt": {"mode": mode}},
        }
    ]
    jpeg_header = (
        b"\xff\xd8\xff\xc0\x00\x0b\x08"
        + IMAGE_HEIGHT_PX.to_bytes(2, "big")
        + IMAGE_WIDTH_PX.to_bytes(2, "big")
        + b"\x01\x01\x11\x00\xff\xd9"
    )
    for position, (offset_s, selected_track_id) in enumerate(
        zip(offsets_s, selected_tracks, strict=True)
    ):
        filename = f"frame_{position:09d}.jpg"
        (demo_directory / filename).write_bytes(jpeg_header)
        target: dict[str, JsonValue] = {
            "track": {"track_id": selected_track_id} if selected_track_id is not None else None,
            "state": "tracking" if selected_track_id is not None else "lost",
        }
        controller: dict[str, JsonValue] = {"mode": mode}
        captured_at = CAPTURE_UNIX_OFFSET_S + offset_s
        index.append(
            {
                "filename": filename,
                "sequence": position * SEQUENCE_STEP,
                "monotonic_s": START_MONOTONIC_S + offset_s,
                "captured_at": captured_at,
                "timebase": "delivery_monotonic",
                "metadata": {
                    "target": target,
                    "controller": controller,
                    "annotations_included": False,
                },
            }
        )
        records.append(
            {
                "event": "frame",
                "captured_at": captured_at,
                "target": target,
                "controller": controller,
                "follow_timebase": "sensor",
            }
        )
    records.append(
        {
            "event": "run_end",
            "error": None,
            "log_dropped_records": 0,
            "demo_recording": {
                "frames_written": len(index),
                "frames_dropped": 0,
                "first_monotonic_s": START_MONOTONIC_S + offsets_s[0],
                "last_monotonic_s": START_MONOTONIC_S + offsets_s[-1],
                "error": None,
            },
        }
    )
    write_records(directory / "detections.jsonl", records)
    write_records(demo_directory / "index.jsonl", index)
    return load_recording(directory)


def label(
    centre_x_px: float = IMAGE_WIDTH_PX / 2,
    expected_identity: str | None = "pigeon-a",
    selected_identity: str | None = "pigeon-a",
    scene_id: str = "synthetic-scene",
) -> dict[str, JsonValue]:
    return {
        "presentation_id": "synthetic-presentation",
        "run_number": 1,
        "repeat_index": 0,
        "scene_id": scene_id,
        "render_number": 1,
        "inventory_complete": True,
        "expected_identity": expected_identity,
        "targets": [
            {
                "identity": "pigeon-a",
                "visible": True,
                "reachable": True,
                "registration_valid": True,
                "centre_method": "manual_pixels",
                "centre_px": {"x_px": centre_x_px, "y_px": IMAGE_HEIGHT_PX / 2},
                "evidence": "synthetic camera truth and reach, independent of selected track",
            }
        ],
        "selection": (
            {"identity": selected_identity, "evidence": "synthetic independently matched identity"}
            if selected_identity is not None
            else None
        ),
        "evidence": "synthetic scene/render reference and complete target inventory",
    }


def labelled_document(
    recording: Recording,
    labels: tuple[JsonValue, ...],
    settling_s: float = 0.0,
    maximum_sample_gap_s: float = 2.0,
) -> dict[str, JsonValue]:
    document = make_review_template(recording, settling_s, maximum_sample_gap_s, POLICY_EVIDENCE)
    for entry, frame_label in zip(entries(document), labels, strict=True):
        entry["label"] = frame_label
    return document


def first_target(frame_label: dict[str, JsonValue]) -> dict[str, JsonValue]:
    targets = frame_label["targets"]
    assert isinstance(targets, list)
    return mapping(targets[0])


def equivalent_label(
    first_render_number: int = 1, last_render_number: int = 3
) -> dict[str, JsonValue]:
    return label() | {
        "render_number": None,
        "render_interval": {
            "first_render_number": first_render_number,
            "last_render_number": last_render_number,
            "equivalent_content_confirmed": True,
            "evidence": "synthetic independent review of every intervening render and selection",
        },
    }


def test_irregular_elapsed_time_and_settling_are_not_frame_averages(tmp_path: Path) -> None:
    recording = write_recording(tmp_path, (0.0, 0.25, 1.0, 2.0), (1, 1, 1, 1))
    document = labelled_document(
        recording, (label(100), label(400), label(100), label()), settling_s=0.1
    )
    report = score_recording(recording, document)
    centring = mapping(report["centring"])
    coverage = mapping(report["coverage"])
    assert centring["eligible_duration_s"] == pytest.approx(1.9)
    assert centring["centred_duration_s"] == pytest.approx(0.75)
    assert centring["occupancy_fraction_on_scored_time"] == pytest.approx(0.75 / 1.9)
    assert centring["verdict"] == "fail"
    assert mapping(coverage["excluded_duration_s_by_reason"])["settling"] == pytest.approx(0.1)
    assert coverage["recorded_duration_s"] == 2.0
    assert report["global_verdict"] == "inconclusive"


@pytest.mark.parametrize("centre_x_px, centre_y_px", [(320.0, 180.0), (480.0, 270.0)])
def test_band_includes_both_central_twenty_percent_boundaries(
    tmp_path: Path, centre_x_px: float, centre_y_px: float
) -> None:
    recording = write_recording(tmp_path)
    boundary = label(centre_x_px)
    first_target(boundary)["centre_px"] = {"x_px": centre_x_px, "y_px": centre_y_px}
    report = score_recording(recording, labelled_document(recording, (boundary,) * 3))
    assert mapping(report["centring"])["verdict"] == "pass"
    assert report["global_verdict"] == "inconclusive"


def test_missing_selection_does_not_remove_visible_truth_from_denominator(tmp_path: Path) -> None:
    recording = write_recording(tmp_path, selected_tracks=(None, None, None))
    document = labelled_document(
        recording,
        (
            label(100, selected_identity=None),
            label(selected_identity=None),
            label(selected_identity=None),
        ),
    )
    report = score_recording(recording, document)
    assert mapping(report["centring"])["eligible_duration_s"] == 2.0
    assert mapping(report["centring"])["occupancy_fraction_on_scored_time"] == 0.5
    durations = mapping(mapping(report["selection"])["post_settling_duration_s_by_result"])
    assert durations["incorrect"] == 2.0


@pytest.mark.parametrize("annotations_included", (True, None))
@pytest.mark.parametrize("centre_x_px", (100, 400))
def test_annotated_or_unknown_camera_evidence_cannot_pass_centring(
    tmp_path: Path, annotations_included: bool | None, centre_x_px: float
) -> None:
    write_recording(tmp_path)
    index_path = tmp_path / "demo" / "index.jsonl"
    records = read_records(index_path)
    metadata = mapping(mapping(records[0])["metadata"])
    metadata["annotations_included"] = annotations_included
    write_records(index_path, records)
    recording = load_recording(tmp_path)
    report = score_recording(recording, labelled_document(recording, (label(centre_x_px),) * 3))
    assert mapping(report["centring"])["verdict"] == "inconclusive"
    assert "unannotated image provenance" in str(recording.issues)


def test_selection_and_loss_observations_need_independent_identity_labels(tmp_path: Path) -> None:
    recording = write_recording(tmp_path, (0, 1, 2, 3, 4), (1, 2, 2, None, None))
    document = labelled_document(
        recording,
        (
            label(),
            label(selected_identity="pigeon-b"),
            label(expected_identity=None, selected_identity=None),
            label(expected_identity=None, selected_identity=None),
            label(expected_identity=None, selected_identity=None),
        ),
    )
    report = score_recording(recording, document)
    selection = mapping(report["selection"])
    assert selection["post_settling_duration_s_by_result"] == {
        "correct": 2.0,
        "incorrect": 1.0,
        "unresolved": 1.0,
    }
    observations = selection["observations"]
    assert isinstance(observations, list)
    assert mapping(observations[2])["selection_result"] == "unresolved"
    assert mapping(observations[3])["selection_result"] == "correct"
    assert mapping(observations[3])["selection_state"] == "lost"


def test_blank_partial_and_final_only_missing_labels_keep_explicit_coverage(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    blank = make_review_template(recording, 0, 2, POLICY_EVIDENCE)
    blank_report = score_recording(recording, blank)
    assert mapping(blank_report["centring"])["goal_met_on_scored_time"] is None
    blank_coverage = mapping(blank_report["coverage"])
    assert mapping(blank_coverage["excluded_duration_s_by_reason"])["missing_annotations"] == 2
    for labels, verdict in (
        ((label(), None, label()), "inconclusive"),
        ((label(), label(), None), "pass"),
    ):
        report = score_recording(recording, labelled_document(recording, labels))
        assert mapping(report["centring"])["goal_met_on_scored_time"] is True
        assert mapping(report["centring"])["verdict"] == verdict
        assert mapping(report["coverage"])["complete"] is False
    blank["frames"] = []
    assert mapping(score_recording(recording, blank)["centring"])["verdict"] == "inconclusive"


def test_long_gap_is_fully_excluded_and_unknown_time_has_a_conservative_bound(
    tmp_path: Path,
) -> None:
    recording = write_recording(tmp_path, (0.0, 0.25, 2.0), (1, 1, 1))
    report = score_recording(
        recording,
        labelled_document(recording, (label(),) * 3, maximum_sample_gap_s=0.5),
    )
    coverage = mapping(report["coverage"])
    assert mapping(coverage["excluded_duration_s_by_reason"])["sampling_gap"] == 1.75
    assert coverage["scored_fraction"] == 0.125
    centring = mapping(report["centring"])
    assert centring["occupancy_fraction_on_scored_time"] == 1.0
    assert centring["occupancy_lower_bound_including_unknown_time"] == 0.125
    assert centring["verdict"] == "inconclusive"


def test_every_excluded_duration_is_accounted_for_and_registration_failure_is_unknown(
    tmp_path: Path,
) -> None:
    recording = write_recording(tmp_path, (0, 1, 2, 3, 4, 5), (1,) * 6)
    invisible, unreachable, failed_registration = label(), label(), label()
    first_target(invisible)["visible"] = False
    first_target(invisible)["centre_px"] = None
    first_target(unreachable)["reachable"] = False
    first_target(failed_registration)["registration_valid"] = False
    first_target(failed_registration)["centre_method"] = "validated_registration"
    first_target(failed_registration)["centre_px"] = None
    report = score_recording(
        recording,
        labelled_document(
            recording,
            (
                invisible,
                unreachable,
                failed_registration,
                label(expected_identity=None),
                label(),
                label(),
            ),
        ),
    )
    coverage = mapping(report["coverage"])
    excluded = mapping(coverage["excluded_duration_s_by_reason"])
    for reason in ("invisible", "unreachable", "registration_failed", "no_expected_target"):
        assert excluded[reason] == 1.0
    assert mapping(report["centring"])["eligible_duration_s"] == 1.0
    assert coverage["unknown_duration_s"] == 1.0
    assert mapping(report["centring"])["verdict"] == "inconclusive"


def test_settling_restarts_on_scene_occurrence_not_tracker_changes(tmp_path: Path) -> None:
    recording = write_recording(tmp_path, (0, 1, 2, 3, 4), (1, 2, 2, 2, 2))
    labels = (
        label(),
        label(),
        label(scene_id="next"),
        label(scene_id="next"),
        label(scene_id="next"),
    )
    report = score_recording(recording, labelled_document(recording, labels, settling_s=0.5))
    coverage = mapping(report["coverage"])
    assert mapping(coverage["excluded_duration_s_by_reason"])["settling"] == 1.0
    assert mapping(report["centring"])["eligible_duration_s"] == 3.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("visible", "yes"),
        ("reachable", 1),
        ("reachable", 0),
        ("reachable", 1.0),
        ("reachable", "yes"),
        ("reachable", []),
        ("registration_valid", None),
        ("centre_method", "detector_box"),
        ("centre_px", None),
        ("centre_px", {"x_px": 800, "y_px": 225}),
        ("centre_px", {"x_px": -1, "y_px": 225}),
        ("evidence", " "),
    ],
)
def test_invalid_truth_is_rejected(tmp_path: Path, field: str, value: JsonValue) -> None:
    recording = write_recording(tmp_path)
    invalid_label = label()
    first_target(invalid_label)[field] = value
    with pytest.raises(ValueError):
        score_recording(recording, labelled_document(recording, (invalid_label, label(), label())))


@pytest.mark.parametrize(
    "contents", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}']
)
def test_duplicate_json_fields_and_nonfinite_numbers_are_rejected(contents: str) -> None:
    with pytest.raises(ValueError):
        read_json(contents)


def test_annotations_cannot_change_sources_dimensions_inventory_or_identity(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    for field, value in (("index_sha256", "wrong"), ("schema_version", True), ("extra", 1)):
        document = labelled_document(recording, (label(),) * 3)
        document[field] = value
        with pytest.raises(ValueError):
            score_recording(recording, document)
    for field, value in (
        ("filename", "different.jpg"),
        ("sequence", 1),
        ("width_px", 801),
        ("jpeg_sha256", "wrong"),
    ):
        document = labelled_document(recording, (label(),) * 3)
        entries(document)[0][field] = value
        with pytest.raises(ValueError):
            score_recording(recording, document)
    for field, value in (("inventory_complete", False), ("expected_identity", "not-in-inventory")):
        invalid_label = label()
        invalid_label[field] = value
        with pytest.raises(ValueError):
            score_recording(
                recording, labelled_document(recording, (invalid_label, label(), label()))
            )
    document = labelled_document(recording, (label(),) * 3)
    document["frames"] = [entries(document)[0], entries(document)[0]]
    with pytest.raises(ValueError, match="duplicate annotation"):
        score_recording(recording, document)
    invalid_label = label()
    target = first_target(invalid_label)
    invalid_label["targets"] = [target, target]
    with pytest.raises(ValueError, match="duplicate target"):
        score_recording(recording, labelled_document(recording, (invalid_label, label(), label())))


@pytest.mark.parametrize(
    "field, value",
    [
        ("timebase", "replay_source"),
        ("timebase", "processing_monotonic"),
        ("sequence", 0),
        ("monotonic_s", START_MONOTONIC_S),
        ("filename", "../outside.jpg"),
        ("captured_at", CAPTURE_UNIX_OFFSET_S),
    ],
)
def test_invalid_and_mixed_recordings_are_rejected(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    write_recording(tmp_path)
    path = tmp_path / "demo" / "index.jsonl"
    records = read_records(path)
    mapping(records[1])[field] = value
    write_records(path, records)
    with pytest.raises(ValueError):
        load_recording(tmp_path)


def test_log_provenance_and_mixed_modes_are_checked(tmp_path: Path) -> None:
    write_recording(tmp_path)
    path = tmp_path / "detections.jsonl"
    original = read_records(path)
    for field, value in (("source", "video"), ("source", "replay")):
        records = read_records(path)
        mapping(records[0])[field] = value
        write_records(path, records)
        with pytest.raises(ValueError, match="reject replay"):
            load_recording(tmp_path)
        write_records(path, original)
    records = read_records(path)
    mapping(records[1])["controller"] = {"mode": "simulate"}
    write_records(path, records)
    with pytest.raises(ValueError, match="mixed frame"):
        load_recording(tmp_path)
    write_records(path, original)
    index_path = tmp_path / "demo" / "index.jsonl"
    records = read_records(index_path)
    mapping(mapping(records[1])["metadata"])["target"] = {"state": "invented", "track": None}
    write_records(index_path, records)
    with pytest.raises(ValueError, match="does not match"):
        load_recording(tmp_path)


def test_scene_reuse_and_render_reversal_are_rejected(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    for labels in (
        (label(), label(scene_id="next"), label()),
        (label() | {"render_number": 2}, label(), label()),
    ):
        with pytest.raises(ValueError):
            score_recording(recording, labelled_document(recording, labels))


def test_equivalent_render_intervals_preserve_geometry_and_selection_durations(
    tmp_path: Path,
) -> None:
    recording = write_recording(tmp_path)
    exact = score_recording(recording, labelled_document(recording, (label(),) * 3))
    reviewed = (equivalent_label(1, 3), equivalent_label(2, 4), label() | {"render_number": 3})
    report = score_recording(recording, labelled_document(recording, reviewed))
    assert report["centring"] == exact["centring"]
    assert report["coverage"] == exact["coverage"]
    selection = mapping(report["selection"])
    assert selection["post_settling_duration_s_by_result"] == {
        "correct": 2.0,
        "incorrect": 0.0,
        "unresolved": 0.0,
    }
    observations = selection["observations"]
    assert isinstance(observations, list)
    for position in (0, 1):
        observation = mapping(observations[position])
        assert observation["render_number"] is None
        assert observation["render_interval"] == reviewed[position]["render_interval"]
    assert mapping(observations[2])["render_number"] == 3
    assert "render_interval" not in mapping(observations[2])
    assert report["global_verdict"] == "inconclusive"


def test_single_equivalent_render_zero_is_valid_without_claiming_exact_decode(
    tmp_path: Path,
) -> None:
    recording = write_recording(tmp_path)
    report = score_recording(recording, labelled_document(recording, (equivalent_label(0, 0),) * 3))
    observations = mapping(report["selection"])["observations"]
    assert isinstance(observations, list)
    assert mapping(observations[0])["render_number"] is None
    assert mapping(mapping(observations[0])["render_interval"])["first_render_number"] == 0
    assert mapping(report["centring"])["centred_duration_s"] == 2.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("first_render_number", -1),
        ("first_render_number", True),
        ("first_render_number", 1.5),
        ("last_render_number", -1),
        ("last_render_number", False),
        ("last_render_number", 3.0),
        ("last_render_number", 0),
        ("equivalent_content_confirmed", False),
        ("equivalent_content_confirmed", 1),
        ("equivalent_content_confirmed", None),
        ("evidence", " "),
        ("extra", "not permitted"),
    ],
)
def test_invalid_equivalent_render_interval_fields_are_rejected(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    recording = write_recording(tmp_path)
    reviewed = equivalent_label()
    mapping(reviewed["render_interval"])[field] = value
    with pytest.raises(ValueError):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))


@pytest.mark.parametrize("render_number", (1, True, -1, "1"))
def test_render_interval_cannot_accompany_an_exact_render_field(
    tmp_path: Path, render_number: JsonValue
) -> None:
    recording = write_recording(tmp_path)
    reviewed = equivalent_label() | {"render_number": render_number}
    with pytest.raises(ValueError, match="render_interval requires render_number: null"):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))


@pytest.mark.parametrize("interval", (None, {}, {"first_render_number": 1}))
def test_null_render_requires_a_complete_interval(tmp_path: Path, interval: JsonValue) -> None:
    recording = write_recording(tmp_path)
    reviewed = label() | {"render_number": None}
    with pytest.raises(ValueError, match="requires render_interval"):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))
    reviewed["render_interval"] = interval
    with pytest.raises(ValueError, match="render_interval"):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))


def test_render_chronology_rejects_reversal_even_after_overlapping_intervals(
    tmp_path: Path,
) -> None:
    recording = write_recording(tmp_path)
    for reviewed in (
        (label() | {"render_number": 5}, equivalent_label(1, 10), label() | {"render_number": 4}),
        (equivalent_label(5, 10), equivalent_label(1, 4), equivalent_label(5, 10)),
    ):
        with pytest.raises(ValueError, match="must not decrease"):
            score_recording(recording, labelled_document(recording, reviewed))


def test_render_interval_chronology_restarts_for_a_new_scene_occurrence(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    reviewed = (
        equivalent_label(5, 10),
        equivalent_label(0, 2) | {"scene_id": "next"},
        label(scene_id="next"),
    )
    report = score_recording(recording, labelled_document(recording, reviewed))
    assert mapping(report["centring"])["eligible_duration_s"] == 2.0


def test_unresolved_expected_reach_is_unknown_and_stays_in_worst_case_bounds(
    tmp_path: Path,
) -> None:
    recording = write_recording(tmp_path)
    unresolved = label()
    first_target(unresolved)["reachable"] = None
    first_target(unresolved)["centre_px"] = None
    report = score_recording(
        recording, labelled_document(recording, (unresolved, label(), label()))
    )
    coverage = mapping(report["coverage"])
    exclusions = mapping(coverage["excluded_duration_s_by_reason"])
    assert exclusions["reach_unresolved"] == 1.0
    assert exclusions["unreachable"] == 0.0
    assert exclusions["invisible"] == 0.0
    assert coverage["unknown_duration_s"] == 1.0
    assert coverage["complete"] is False
    centring = mapping(report["centring"])
    assert centring["eligible_duration_s"] == 1.0
    assert centring["occupancy_fraction_on_scored_time"] == 1.0
    assert centring["occupancy_lower_bound_including_unknown_time"] == 0.5
    assert centring["occupancy_upper_bound_including_unknown_time"] == 1.0
    assert centring["verdict"] == "inconclusive"


def test_nonexpected_null_reach_keeps_the_known_expected_target_eligible(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    reviewed = label()
    other_target = first_target(label()) | {
        "identity": "pigeon-b",
        "reachable": None,
        "centre_px": None,
    }
    reviewed["targets"] = [first_target(reviewed), other_target]
    report = score_recording(recording, labelled_document(recording, (reviewed,) * 3))
    assert mapping(report["coverage"])["unknown_duration_s"] == 0.0
    assert mapping(report["centring"])["eligible_duration_s"] == 2.0
    assert mapping(report["centring"])["centred_duration_s"] == 2.0


def test_invisible_null_reach_is_invisible_rather_than_unknown(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    invisible = label()
    first_target(invisible).update(visible=False, reachable=None, centre_px=None)
    report = score_recording(recording, labelled_document(recording, (invisible, label(), label())))
    coverage = mapping(report["coverage"])
    exclusions = mapping(coverage["excluded_duration_s_by_reason"])
    assert exclusions["invisible"] == 1.0
    assert exclusions["reach_unresolved"] == 0.0
    assert coverage["unknown_duration_s"] == 0.0
    assert coverage["complete"] is True
    assert mapping(report["centring"])["eligible_duration_s"] == 1.0


def test_incomplete_run_telemetry_cannot_pass_centring(tmp_path: Path) -> None:
    write_recording(tmp_path)
    path = tmp_path / "detections.jsonl"
    records = read_records(path)
    mapping(records[-1])["log_dropped_records"] = 1
    write_records(path, records)
    recording = load_recording(tmp_path)
    report = score_recording(recording, labelled_document(recording, (label(),) * 3))
    assert mapping(report["centring"])["goal_met_on_scored_time"] is True
    assert mapping(report["centring"])["verdict"] == "inconclusive"


def test_recording_cannot_be_cropped_to_cherry_pick_the_denominator(tmp_path: Path) -> None:
    write_recording(tmp_path)
    path = tmp_path / "demo" / "index.jsonl"
    records = read_records(path)
    write_records(path, records[1:])
    with pytest.raises(ValueError, match="reject cropped indexes"):
        load_recording(tmp_path)


def test_exact_eighty_percent_meets_numeric_goal(tmp_path: Path) -> None:
    recording = write_recording(tmp_path, (0.0, 1.0, 5.0), (1, 1, 1))
    report = score_recording(
        recording,
        labelled_document(recording, (label(100), label(), label()), maximum_sample_gap_s=4.0),
    )
    assert mapping(report["centring"])["occupancy_fraction_on_scored_time"] == 0.8
    assert mapping(report["centring"])["verdict"] == "pass"


@pytest.mark.parametrize("method", ("manual_pixels", "validated_registration"))
def test_legacy_labels_without_uncertainty_retain_zero_allowance(
    tmp_path: Path, method: str
) -> None:
    recording = write_recording(tmp_path)
    reviewed = label()
    first_target(reviewed)["centre_method"] = method
    report = score_recording(recording, labelled_document(recording, (reviewed,) * 3))
    centring = mapping(report["centring"])
    assert centring["centred_duration_s"] == 2.0
    assert centring["definitely_centred_duration_s"] == 2.0
    assert centring["possibly_centred_duration_s"] == 2.0
    assert centring["occupancy_lower_bound_including_unknown_time"] == 1.0
    assert centring["occupancy_upper_bound_including_unknown_time"] == 1.0
    assert centring["verdict"] == "pass"


@pytest.mark.parametrize("method", ("manual_pixels", "validated_registration"))
def test_legacy_methods_honour_explicit_uncertainty(tmp_path: Path, method: str) -> None:
    recording = write_recording(tmp_path)
    reviewed = label(320)
    target = first_target(reviewed)
    target["centre_method"] = method
    target["centre_uncertainty_px"] = {"x_px": 1, "y_px": 0}
    report = score_recording(recording, labelled_document(recording, (reviewed,) * 3))
    centring = mapping(report["centring"])
    assert centring["occupancy_fraction_on_scored_time"] == 1.0
    assert centring["occupancy_lower_bound_including_unknown_time"] == 0.0
    assert centring["occupancy_upper_bound_including_unknown_time"] == 1.0
    assert centring["verdict"] == "inconclusive"


@pytest.mark.parametrize(
    "centre_px, uncertainty_px, point, definite, possible, verdict",
    [
        ((400, 225), (80, 45), True, True, True, "pass"),
        ((400, 225), (81, 45), True, False, True, "inconclusive"),
        ((400, 225), (80, 46), True, False, True, "inconclusive"),
        ((320, 180), (0, 0), True, True, True, "pass"),
        ((480, 270), (0, 0), True, True, True, "pass"),
        ((320, 180), (1, 0), True, False, True, "inconclusive"),
        ((480, 270), (0, 1), True, False, True, "inconclusive"),
        ((310, 225), (10, 0), False, False, True, "inconclusive"),
        ((490, 225), (10, 0), False, False, True, "inconclusive"),
        ((400, 170), (0, 10), False, False, True, "inconclusive"),
        ((400, 280), (0, 10), False, False, True, "inconclusive"),
        ((310, 225), (9, 100), False, False, False, "fail"),
        ((400, 280), (100, 9), False, False, False, "fail"),
    ],
)
def test_uncertainty_rectangles_use_both_axes_and_include_band_edges(
    tmp_path: Path,
    centre_px: tuple[float, float],
    uncertainty_px: tuple[float, float],
    point: bool,
    definite: bool,
    possible: bool,
    verdict: str,
) -> None:
    recording = write_recording(tmp_path)
    reviewed = label()
    target = first_target(reviewed)
    target["centre_method"] = "validated_local_photograph"
    target["centre_px"] = {"x_px": centre_px[0], "y_px": centre_px[1]}
    target["centre_uncertainty_px"] = {"x_px": uncertainty_px[0], "y_px": uncertainty_px[1]}
    target["evidence"] = "synthetic independently checked local mapping and axis error allowances"
    report = score_recording(recording, labelled_document(recording, (reviewed,) * 3))
    centring = mapping(report["centring"])
    assert centring["centred_duration_s"] == 2.0 * point
    assert centring["definitely_centred_duration_s"] == 2.0 * definite
    assert centring["possibly_centred_duration_s"] == 2.0 * possible
    assert centring["occupancy_fraction_on_scored_time"] == float(point)
    assert centring["definite_occupancy_fraction_on_scored_time"] == float(definite)
    assert centring["possible_occupancy_fraction_on_scored_time"] == float(possible)
    assert centring["occupancy_lower_bound_including_unknown_time"] == float(definite)
    assert centring["occupancy_upper_bound_including_unknown_time"] == float(possible)
    assert centring["verdict"] == verdict
    assert report["global_verdict"] == "inconclusive"


@pytest.mark.parametrize("axis", ("x_px", "y_px"))
@pytest.mark.parametrize("invalid", (-1, float("nan"), float("inf"), True))
def test_uncertainty_rejects_negative_nonfinite_and_boolean_allowances(
    tmp_path: Path, axis: str, invalid: JsonValue
) -> None:
    recording = write_recording(tmp_path)
    reviewed = label()
    target = first_target(reviewed)
    target["centre_method"] = "validated_local_photograph"
    target["centre_uncertainty_px"] = {"x_px": 0, "y_px": 0, axis: invalid}
    with pytest.raises(ValueError, match="finite non-negative number"):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))


@pytest.mark.parametrize("uncertainty", (None, {"x_px": 1}, {"x_px": 1, "y_px": 1, "extra": 1}))
def test_uncertainty_has_strict_axis_fields(tmp_path: Path, uncertainty: JsonValue) -> None:
    recording = write_recording(tmp_path)
    reviewed = label()
    first_target(reviewed)["centre_uncertainty_px"] = uncertainty
    with pytest.raises(ValueError, match="centre_uncertainty_px"):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))


def test_local_photograph_method_requires_explicit_uncertainty(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    reviewed = label()
    first_target(reviewed)["centre_method"] = "validated_local_photograph"
    with pytest.raises(ValueError, match="requires centre_uncertainty_px"):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))


def test_local_photograph_method_requires_nonblank_review_evidence(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    reviewed = label()
    target = first_target(reviewed)
    target["centre_method"] = "validated_local_photograph"
    target["centre_uncertainty_px"] = {"x_px": 0, "y_px": 0}
    target["evidence"] = " "
    with pytest.raises(ValueError, match="target.evidence"):
        score_recording(recording, labelled_document(recording, (reviewed,) * 3))


@pytest.mark.parametrize("definite_duration_s, verdict", ((8.0, "pass"), (7.0, "inconclusive")))
def test_definite_time_controls_threshold_even_when_nominal_occupancy_is_perfect(
    tmp_path: Path, definite_duration_s: float, verdict: str
) -> None:
    recording = write_recording(tmp_path, (0.0, definite_duration_s, 10.0), (1, 1, 1))
    uncertain = label(320)
    first_target(uncertain)["centre_uncertainty_px"] = {"x_px": 1, "y_px": 0}
    report = score_recording(
        recording,
        labelled_document(recording, (label(), uncertain, label()), maximum_sample_gap_s=10.0),
    )
    centring = mapping(report["centring"])
    assert centring["occupancy_fraction_on_scored_time"] == 1.0
    assert centring["definitely_centred_duration_s"] == definite_duration_s
    assert centring["occupancy_lower_bound_including_unknown_time"] == definite_duration_s / 10.0
    assert centring["occupancy_upper_bound_including_unknown_time"] == 1.0
    assert centring["verdict"] == verdict


@pytest.mark.parametrize(
    "known_duration_s, verdict", ((9.0, "pass"), (8.0, "pass"), (7.0, "inconclusive"))
)
def test_unknown_time_stays_in_denominator_and_can_still_prove_scoped_goal(
    tmp_path: Path, known_duration_s: float, verdict: str
) -> None:
    recording = write_recording(tmp_path, (0.0, known_duration_s, 10.0), (None, 2, 1))
    report = score_recording(
        recording,
        labelled_document(
            recording,
            (label(selected_identity=None), None, label()),
            maximum_sample_gap_s=10.0,
        ),
    )
    centring = mapping(report["centring"])
    assert centring["eligible_duration_s"] == known_duration_s
    assert centring["definitely_centred_duration_s"] == known_duration_s
    assert centring["occupancy_fraction_on_scored_time"] == 1.0
    assert centring["occupancy_lower_bound_including_unknown_time"] == known_duration_s / 10.0
    assert centring["occupancy_upper_bound_including_unknown_time"] == 1.0
    assert centring["verdict"] == verdict
    assert mapping(report["coverage"])["unknown_duration_s"] == 10.0 - known_duration_s
    assert mapping(report["coverage"])["complete"] is False
    assert report["global_verdict"] == "inconclusive"


@pytest.mark.parametrize("unknown_duration_s, verdict", ((2.0, "fail"), (8.0, "inconclusive")))
def test_unknown_time_is_centred_in_upper_bound_for_failure_decision(
    tmp_path: Path, unknown_duration_s: float, verdict: str
) -> None:
    recording = write_recording(tmp_path, (0.0, unknown_duration_s, 10.0), (1, 1, 1))
    report = score_recording(
        recording,
        labelled_document(recording, (None, label(100), label()), maximum_sample_gap_s=10.0),
    )
    centring = mapping(report["centring"])
    assert centring["occupancy_lower_bound_including_unknown_time"] == 0.0
    assert centring["occupancy_upper_bound_including_unknown_time"] == unknown_duration_s / 10.0
    assert centring["verdict"] == verdict


def test_no_eligible_or_unknown_duration_is_inconclusive(tmp_path: Path) -> None:
    recording = write_recording(tmp_path)
    report = score_recording(
        recording, labelled_document(recording, (label(expected_identity=None),) * 3)
    )
    centring = mapping(report["centring"])
    assert centring["occupancy_lower_bound_including_unknown_time"] is None
    assert centring["occupancy_upper_bound_including_unknown_time"] is None
    assert centring["definite_occupancy_fraction_on_scored_time"] is None
    assert centring["possible_occupancy_fraction_on_scored_time"] is None
    assert centring["verdict"] == "inconclusive"


def test_real_cli_blank_template_and_report_never_overwrite_evidence(tmp_path: Path) -> None:
    write_recording(tmp_path, mode="simulate")
    template_path = tmp_path / "review.json"
    report_path = tmp_path / "report.json"
    arguments = [
        "template",
        str(tmp_path),
        "--settling-s",
        "0.5",
        "--maximum-sample-gap-s",
        "2",
        "--policy-evidence",
        POLICY_EVIDENCE,
        "--output",
        str(template_path),
    ]
    assert main(arguments) == 0
    document = mapping(read_json(template_path.read_text(encoding="utf-8")))
    assert all(entry["label"] is None for entry in entries(document))
    assert [entry["sequence"] for entry in entries(document)] == [0, 4, 8]
    assert (
        main(
            [
                "score",
                str(tmp_path),
                "--annotations",
                str(template_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = mapping(read_json(report_path.read_text(encoding="utf-8")))
    assert report["mode"] == "simulate"
    assert report["global_verdict"] == "inconclusive"
    assert mapping(report["centring"])["verdict"] == "inconclusive"
    assert "annotations_sha256" in report
    original = template_path.read_bytes()
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2
    assert template_path.read_bytes() == original
