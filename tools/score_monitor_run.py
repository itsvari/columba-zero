"""Score recorded monitor geometry against independently reviewed camera labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal, NoReturn, cast

from app.json_data import JsonValue

SCHEMA_VERSION = 1
CENTRAL_IMAGE_FRACTION = 0.20
OCCUPANCY_GOAL_FRACTION = 0.80
IMAGE_MIDPOINT_FRACTION = 0.50
JPEG_START = b"\xff\xd8"
JPEG_MARKER_PREFIX = b"\xff"
JPEG_LENGTH_BYTES = 2
JPEG_DIMENSION_BYTES = 2
JPEG_PRECISION_BYTES = 1
JPEG_START_OF_SCAN = 0xDA
JPEG_END = 0xD9
JPEG_START_OF_FRAME = frozenset(
    (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)
)
EXCLUSION_REASONS = (
    "missing_annotations",
    "settling",
    "sampling_gap",
    "no_expected_target",
    "invisible",
    "unreachable",
    "registration_failed",
)
UNKNOWN_REASONS = ("missing_annotations", "sampling_gap", "registration_failed")
Mode = Literal["hardware", "simulate"]
SelectionResult = Literal["correct", "incorrect", "unresolved"]


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    filename: str
    sequence: int
    monotonic_s: float
    width_px: int
    height_px: int
    jpeg_sha256: str
    selected_track_id: int | None
    selection_state: str


@dataclass(frozen=True, slots=True)
class Recording:
    directory: Path
    index_sha256: str
    detections_sha256: str
    mode: Mode
    frames: tuple[RecordedFrame, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Policy:
    settling_s: float
    maximum_sample_gap_s: float
    evidence: str


@dataclass(frozen=True, slots=True)
class Scene:
    presentation_id: str
    run_number: int
    repeat_index: int
    scene_id: str


@dataclass(frozen=True, slots=True)
class TargetTruth:
    identity: str
    visible: bool
    reachable: bool
    registration_valid: bool
    centre_px: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class Annotation:
    scene: Scene
    render_number: int
    expected_identity: str | None
    targets: tuple[TargetTruth, ...]
    selected_identity: str | None


def _mapping(value: JsonValue, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON mapping")
    return value


def _fields(value: JsonValue, context: str, names: str) -> dict[str, JsonValue]:
    result = _mapping(value, context)
    if set(result) != set(names.split()):
        raise ValueError(f"{context} must have exactly these fields: {names}")
    return result


def _text(value: JsonValue, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-empty text")
    return value


def _integer(value: JsonValue, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _number(value: JsonValue, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be a finite non-negative number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{context} must be finite") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context} must be a finite non-negative number")
    return number


def _boolean(value: JsonValue, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be boolean")
    return value


def _unique_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(contents: str) -> JsonValue:
    return cast(
        JsonValue,
        json.loads(
            contents,
            object_pairs_hook=_unique_fields,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        ),
    )


def _read_records(path: Path) -> tuple[dict[str, JsonValue], ...]:
    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        context = f"{path}:{line_number}"
        try:
            records.append(_mapping(read_json(line), context))
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error
    return tuple(records)


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as image:
        if image.read(len(JPEG_START)) != JPEG_START:
            raise ValueError(f"{path}: missing JPEG start")
        while image.read(1) == JPEG_MARKER_PREFIX:
            marker = image.read(1)
            while marker == JPEG_MARKER_PREFIX:
                marker = image.read(1)
            if not marker or marker[0] in (JPEG_START_OF_SCAN, JPEG_END):
                break
            length = int.from_bytes(image.read(JPEG_LENGTH_BYTES), "big")
            segment = image.read(max(0, length - JPEG_LENGTH_BYTES))
            if length < JPEG_LENGTH_BYTES or len(segment) != length - JPEG_LENGTH_BYTES:
                break
            if marker[0] not in JPEG_START_OF_FRAME:
                continue
            dimension_end = JPEG_PRECISION_BYTES + JPEG_DIMENSION_BYTES * 2
            if len(segment) < dimension_end:
                break
            width_start = dimension_end - JPEG_DIMENSION_BYTES
            height_px = int.from_bytes(segment[JPEG_PRECISION_BYTES:width_start], "big")
            width_px = int.from_bytes(segment[width_start:dimension_end], "big")
            if width_px > 0 and height_px > 0:
                return width_px, height_px
            break
    raise ValueError(f"{path}: missing or invalid JPEG dimensions")


def _recorded_frame(record: dict[str, JsonValue], directory: Path, mode: Mode) -> RecordedFrame:
    filename = _text(record.get("filename"), "filename")
    if Path(filename).name != filename or Path(filename).suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("filename must be a JPEG basename")
    if record.get("timebase") != "delivery_monotonic":
        raise ValueError("only live delivery_monotonic recordings are supported; reject replay")
    metadata = _mapping(record.get("metadata"), "metadata")
    controller = _mapping(metadata.get("controller"), "metadata.controller")
    if controller.get("mode") != mode:
        raise ValueError("recorded controller mode differs from run mode")
    target = _mapping(metadata.get("target"), "metadata.target")
    track = target.get("track")
    selected_track_id = (
        None if track is None else _integer(_mapping(track, "track").get("track_id"), "track_id")
    )
    width_px, height_px = _jpeg_dimensions(directory / "demo" / filename)
    return RecordedFrame(
        filename,
        _integer(record.get("sequence"), "sequence"),
        _number(record.get("monotonic_s"), "monotonic_s"),
        width_px,
        height_px,
        hashlib.sha256((directory / "demo" / filename).read_bytes()).hexdigest(),
        selected_track_id,
        _text(target.get("state"), "target.state"),
    )


def _recording_issues(
    ends: Sequence[dict[str, JsonValue]], frames: tuple[RecordedFrame, ...]
) -> tuple[str, ...]:
    if len(ends) > 1:
        raise ValueError("duplicate run_end")
    if not ends:
        return ("missing run_end",)
    end = ends[0]
    issues: list[str] = []
    if "error" not in end or end["error"] is not None:
        issues.append("missing or unsuccessful run_end status")
    if end.get("log_dropped_records") is None:
        issues.append("missing log-drop count")
    elif _integer(end["log_dropped_records"], "log_dropped_records") != 0:
        issues.append("dropped log records")
    if end.get("demo_recording") is None:
        return (*issues, "missing recording bounds/counts")
    status = _mapping(end["demo_recording"], "demo_recording")
    if _integer(status.get("frames_written"), "frames_written") != len(frames):
        raise ValueError("recording frame count differs from run_end; reject cropped indexes")
    if (
        _number(status.get("first_monotonic_s"), "first_monotonic_s") != frames[0].monotonic_s
        or _number(status.get("last_monotonic_s"), "last_monotonic_s") != frames[-1].monotonic_s
    ):
        raise ValueError("recording bounds differ from run_end; reject cropped indexes")
    if _integer(status.get("frames_dropped"), "frames_dropped") != 0:
        issues.append("dropped recording frames")
    if status.get("error") is not None:
        issues.append("recording error")
    return tuple(issues)


def load_recording(directory: Path) -> Recording:
    index_path = directory / "demo" / "index.jsonl"
    detections_path = directory / "detections.jsonl"
    records = _read_records(detections_path)
    starts = [record for record in records if record.get("event") == "run_start"]
    if len(starts) != 1 or starts[0].get("source") != "csi_camera":
        raise ValueError("require exactly one live csi_camera run_start; reject replay")
    configuration = _mapping(starts[0].get("config"), "run configuration")
    configured_mode = _mapping(configuration.get("pan_tilt"), "pan_tilt").get("mode")
    if configured_mode not in ("hardware", "simulate"):
        raise ValueError("run mode must be hardware or simulate")
    mode = cast(Mode, configured_mode)
    logged_frames: dict[float, dict[str, JsonValue]] = {}
    for record in records:
        if record.get("event") == "controller_state" and record.get("mode") != mode:
            raise ValueError("mixed controller modes")
        if record.get("event") != "frame":
            continue
        captured_at = _number(record.get("captured_at"), "frame.captured_at")
        if captured_at in logged_frames:
            raise ValueError("duplicate logged frame captured_at")
        logged_frames[captured_at] = record
        if _mapping(record.get("controller"), "frame.controller").get("mode") != mode:
            raise ValueError("mixed frame controller modes")
        if record.get("follow_timebase") not in ("sensor", "delivery"):
            raise ValueError("frame follow_timebase must be live sensor or delivery")
    index = _read_records(index_path)
    frames = tuple(_recorded_frame(record, directory, mode) for record in index)
    if len(frames) < 2:
        raise ValueError("require at least two recorded frames to measure elapsed time")
    if len({frame.filename for frame in frames}) != len(frames):
        raise ValueError("duplicate recorded filename")
    for previous, current in pairwise(frames):
        if current.sequence <= previous.sequence or current.monotonic_s <= previous.monotonic_s:
            raise ValueError("sequences and monotonic_s must be strictly increasing")
    captured_times_s: set[float] = set()
    for record in index:
        captured_at = _number(record.get("captured_at"), "index.captured_at")
        if captured_at in captured_times_s:
            raise ValueError("duplicate recorded captured_at")
        captured_times_s.add(captured_at)
        logged = logged_frames.get(captured_at)
        metadata = _mapping(record.get("metadata"), "metadata")
        if logged is None or metadata.get("target") != logged.get("target"):
            raise ValueError("recording target metadata does not match detections.jsonl")
    ends = [record for record in records if record.get("event") == "run_end"]
    return Recording(
        directory.resolve(),
        hashlib.sha256(index_path.read_bytes()).hexdigest(),
        hashlib.sha256(detections_path.read_bytes()).hexdigest(),
        mode,
        frames,
        _recording_issues(ends, frames),
    )


def _policy(value: JsonValue) -> Policy:
    policy = _fields(value, "policy", "settling_s maximum_sample_gap_s evidence target_inventory")
    if policy["target_inventory"] != "all_intended_targets_independent_of_selection":
        raise ValueError(
            "target_inventory must include all intended targets independently of selection"
        )
    maximum_sample_gap_s = _number(policy["maximum_sample_gap_s"], "maximum_sample_gap_s")
    if maximum_sample_gap_s <= 0:
        raise ValueError("maximum_sample_gap_s must be positive")
    return Policy(
        _number(policy["settling_s"], "settling_s"),
        maximum_sample_gap_s,
        _text(policy["evidence"], "policy.evidence"),
    )


def make_review_template(
    recording: Recording, settling_s: float, maximum_sample_gap_s: float, evidence: str
) -> dict[str, JsonValue]:
    policy: dict[str, JsonValue] = {
        "settling_s": settling_s,
        "maximum_sample_gap_s": maximum_sample_gap_s,
        "evidence": evidence,
        "target_inventory": "all_intended_targets_independent_of_selection",
    }
    _policy(policy)
    return {
        "schema_version": SCHEMA_VERSION,
        "index_sha256": recording.index_sha256,
        "detections_sha256": recording.detections_sha256,
        "policy": policy,
        "frames": [
            {
                "filename": frame.filename,
                "sequence": frame.sequence,
                "width_px": frame.width_px,
                "height_px": frame.height_px,
                "jpeg_sha256": frame.jpeg_sha256,
                "label": None,
            }
            for frame in recording.frames
        ],
    }


def _target_truth(value: JsonValue, frame: RecordedFrame) -> TargetTruth:
    target = _fields(
        value,
        "target truth",
        "identity visible reachable registration_valid centre_method centre_px evidence",
    )
    if target["centre_method"] not in ("manual_pixels", "validated_registration"):
        raise ValueError("centre_method must be manual_pixels or validated_registration")
    _text(target["evidence"], "target.evidence")
    centre_px: tuple[float, float] | None = None
    if target["centre_px"] is not None:
        centre = _fields(target["centre_px"], "centre_px", "x_px y_px")
        centre_px = (_number(centre["x_px"], "x_px"), _number(centre["y_px"], "y_px"))
        if centre_px[0] >= frame.width_px or centre_px[1] >= frame.height_px:
            raise ValueError("centre_px must lie within the recorded JPEG dimensions")
    truth = TargetTruth(
        _text(target["identity"], "identity"),
        _boolean(target["visible"], "visible"),
        _boolean(target["reachable"], "reachable"),
        _boolean(target["registration_valid"], "registration_valid"),
        centre_px,
    )
    if truth.visible and truth.reachable and truth.registration_valid and centre_px is None:
        raise ValueError("visible reachable targets with valid registration require centre_px")
    return truth


def _annotation(value: JsonValue, frame: RecordedFrame) -> Annotation:
    label = _fields(
        value,
        "label",
        "presentation_id run_number repeat_index scene_id render_number inventory_complete "
        "expected_identity targets selection evidence",
    )
    if label["inventory_complete"] is not True:
        raise ValueError("inventory_complete must affirm review of every intended target")
    _text(label["evidence"], "label.evidence")
    if not isinstance(label["targets"], list):
        raise ValueError("targets must be a list")
    targets = tuple(_target_truth(value, frame) for value in label["targets"])
    identities = {target.identity for target in targets}
    if len(identities) != len(targets):
        raise ValueError("duplicate target identity")
    expected_identity = label["expected_identity"]
    if expected_identity is not None:
        expected_identity = _text(expected_identity, "expected_identity")
        if expected_identity not in identities:
            raise ValueError("expected_identity must exist in the complete target inventory")
    selected_identity = None
    if label["selection"] is not None:
        selection = _fields(label["selection"], "selection", "identity evidence")
        if frame.selected_track_id is None:
            raise ValueError("cannot map selection when recording has no selected track")
        selected_identity = _text(selection["identity"], "selection.identity")
        _text(selection["evidence"], "selection.evidence")
    return Annotation(
        Scene(
            _text(label["presentation_id"], "presentation_id"),
            _integer(label["run_number"], "run_number"),
            _integer(label["repeat_index"], "repeat_index"),
            _text(label["scene_id"], "scene_id"),
        ),
        _integer(label["render_number"], "render_number"),
        expected_identity,
        targets,
        selected_identity,
    )


def _annotations(recording: Recording, value: JsonValue) -> tuple[Policy, dict[int, Annotation]]:
    document = _fields(
        value, "annotations", "schema_version index_sha256 detections_sha256 policy frames"
    )
    if _integer(document["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if (
        document["index_sha256"] != recording.index_sha256
        or document["detections_sha256"] != recording.detections_sha256
    ):
        raise ValueError("annotation provenance hashes do not match recording")
    if not isinstance(document["frames"], list):
        raise ValueError("frames must be a list")
    recorded_frames = {frame.sequence: frame for frame in recording.frames}
    annotations: dict[int, Annotation] = {}
    seen: set[int] = set()
    for value in document["frames"]:
        entry = _fields(
            value, "annotation frame", "filename sequence width_px height_px jpeg_sha256 label"
        )
        sequence = _integer(entry["sequence"], "sequence")
        frame = recorded_frames.get(sequence)
        if frame is None or frame.filename != entry["filename"]:
            raise ValueError("annotation filename and sequence must match the recording")
        if (
            _integer(entry["width_px"], "width_px") != frame.width_px
            or _integer(entry["height_px"], "height_px") != frame.height_px
            or entry["jpeg_sha256"] != frame.jpeg_sha256
        ):
            raise ValueError("annotation JPEG dimensions or hash do not match recording")
        if sequence in seen:
            raise ValueError("duplicate annotation sequence")
        seen.add(sequence)
        if entry["label"] is not None:
            annotations[sequence] = _annotation(entry["label"], frame)
    return _policy(document["policy"]), annotations


def _scene_starts(
    frames: tuple[RecordedFrame, ...], annotations: dict[int, Annotation]
) -> dict[Scene, float]:
    starts: dict[Scene, float] = {}
    previous: Annotation | None = None
    for frame in frames:
        annotation = annotations.get(frame.sequence)
        if annotation is None:
            continue
        if (
            previous is not None
            and previous.scene != annotation.scene
            and annotation.scene in starts
        ):
            raise ValueError("a scene occurrence cannot reappear; use the correct run/repeat IDs")
        if previous is not None and previous.scene == annotation.scene:
            if annotation.render_number < previous.render_number:
                raise ValueError("render_number must not decrease within a scene occurrence")
        starts.setdefault(annotation.scene, frame.monotonic_s)
        previous = annotation
    return starts


def _exclusion(annotation: Annotation | None) -> str | None:
    if annotation is None:
        return "missing_annotations"
    if annotation.expected_identity is None:
        return "no_expected_target"
    target = next(
        target for target in annotation.targets if target.identity == annotation.expected_identity
    )
    if not target.visible:
        return "invisible"
    if not target.reachable:
        return "unreachable"
    if not target.registration_valid:
        return "registration_failed"
    return None


def _centred(annotation: Annotation, frame: RecordedFrame) -> bool:
    target = next(
        target for target in annotation.targets if target.identity == annotation.expected_identity
    )
    assert target.centre_px is not None
    half_band_fraction = CENTRAL_IMAGE_FRACTION / 2
    return all(
        abs(coordinate_px / dimension_px - IMAGE_MIDPOINT_FRACTION) <= half_band_fraction
        for coordinate_px, dimension_px in zip(
            target.centre_px, (frame.width_px, frame.height_px), strict=True
        )
    )


def _selection_result(annotation: Annotation | None, frame: RecordedFrame) -> SelectionResult:
    if annotation is None:
        return "unresolved"
    if frame.selected_track_id is None:
        return "correct" if annotation.expected_identity is None else "incorrect"
    if annotation.selected_identity is None:
        return "unresolved"
    return (
        "correct" if annotation.selected_identity == annotation.expected_identity else "incorrect"
    )


def score_recording(recording: Recording, value: JsonValue) -> dict[str, JsonValue]:
    policy, annotations = _annotations(recording, value)
    scene_starts = _scene_starts(recording.frames, annotations)
    exclusions = dict.fromkeys(EXCLUSION_REASONS, 0.0)
    selection_durations = dict.fromkeys(("correct", "incorrect", "unresolved"), 0.0)
    eligible_duration_s = 0.0
    centred_duration_s = 0.0
    labelled_duration_s = 0.0
    sampling_gap_duration_s = 0.0
    observations: list[JsonValue] = []
    for position, frame in enumerate(recording.frames):
        annotation = annotations.get(frame.sequence)
        next_frame = recording.frames[min(position + 1, len(recording.frames) - 1)]
        duration_s = next_frame.monotonic_s - frame.monotonic_s
        if annotation is not None:
            labelled_duration_s += duration_s
        settling_duration_s = (
            min(
                duration_s,
                max(0.0, scene_starts[annotation.scene] + policy.settling_s - frame.monotonic_s),
            )
            if annotation is not None
            else 0.0
        )
        exclusions["settling"] += settling_duration_s
        remaining_duration_s = duration_s - settling_duration_s
        long_gap = duration_s > policy.maximum_sample_gap_s
        sampling_gap_duration_s += duration_s if long_gap else 0.0
        exclusion = "sampling_gap" if long_gap else _exclusion(annotation)
        if exclusion is not None:
            exclusions[exclusion] += remaining_duration_s
        else:
            assert annotation is not None
            eligible_duration_s += remaining_duration_s
            if _centred(annotation, frame):
                centred_duration_s += remaining_duration_s
        selection_result = _selection_result(annotation, frame)
        duration_result = "unresolved" if long_gap else selection_result
        selection_durations[duration_result] += remaining_duration_s
        observations.append(
            {
                "filename": frame.filename,
                "sequence": frame.sequence,
                "duration_s": duration_s,
                "settling_s": settling_duration_s,
                "exclusion": exclusion,
                "scene_id": annotation.scene.scene_id if annotation else None,
                "render_number": annotation.render_number if annotation else None,
                "expected_identity": annotation.expected_identity if annotation else None,
                "selected_track_id": frame.selected_track_id,
                "selection_state": frame.selection_state,
                "observed_selected_identity": annotation.selected_identity if annotation else None,
                "selection_result": selection_result,
            }
        )
    total_duration_s = recording.frames[-1].monotonic_s - recording.frames[0].monotonic_s
    unknown_duration_s = math.fsum(exclusions[reason] for reason in UNKNOWN_REASONS)
    incomplete_labels = len(annotations) != len(recording.frames)
    occupancy = centred_duration_s / eligible_duration_s if eligible_duration_s else None
    complete = not incomplete_labels and unknown_duration_s == 0 and not recording.issues
    goal_met = occupancy >= OCCUPANCY_GOAL_FRACTION if occupancy is not None else None
    centring_verdict = (
        "inconclusive" if not complete or goal_met is None else ("pass" if goal_met else "fail")
    )
    conservative_denominator_s = eligible_duration_s + unknown_duration_s
    return {
        "schema_version": SCHEMA_VERSION,
        "recording_directory": str(recording.directory),
        "index_sha256": recording.index_sha256,
        "detections_sha256": recording.detections_sha256,
        "mode": recording.mode,
        "timebase": "delivery_monotonic",
        "policy": _mapping(value, "annotations")["policy"],
        "window_start_monotonic_s": recording.frames[0].monotonic_s,
        "window_end_monotonic_s": recording.frames[-1].monotonic_s,
        "coverage": {
            "recorded_duration_s": total_duration_s,
            "recorded_frame_count": len(recording.frames),
            "annotated_frame_count": len(annotations),
            "labelled_duration_s": labelled_duration_s,
            "labelled_fraction": labelled_duration_s / total_duration_s,
            "scored_fraction": eligible_duration_s / total_duration_s,
            "unknown_duration_s": unknown_duration_s,
            "excluded_duration_s_by_reason": dict(exclusions),
            "long_sampling_gaps_total_s": sampling_gap_duration_s,
            "complete": complete,
            "recording_issues": list(recording.issues),
        },
        "centring": {
            "central_width_fraction": CENTRAL_IMAGE_FRACTION,
            "central_height_fraction": CENTRAL_IMAGE_FRACTION,
            "goal_fraction": OCCUPANCY_GOAL_FRACTION,
            "eligible_duration_s": eligible_duration_s,
            "centred_duration_s": centred_duration_s,
            "occupancy_fraction_on_scored_time": occupancy,
            "goal_met_on_scored_time": goal_met,
            "occupancy_lower_bound_including_unknown_time": (
                centred_duration_s / conservative_denominator_s
                if conservative_denominator_s
                else None
            ),
            "verdict": centring_verdict,
        },
        "selection": {
            "post_settling_duration_s_by_result": dict(selection_durations),
            "observations": observations,
        },
        "global_verdict": "inconclusive",
        "global_reason": (
            "This scorer assesses sampled recording geometry and labelled selection/loss only. "
            "Timing, recovery, movement envelopes, output modes and endurance "
            "require separate gates."
        ),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="write filename-linked labels with null truth")
    template.add_argument("run_directory", type=Path)
    template.add_argument("--settling-s", type=float, required=True)
    template.add_argument("--maximum-sample-gap-s", type=float, required=True)
    template.add_argument("--policy-evidence", required=True)
    template.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="score independently completed labels")
    score.add_argument("run_directory", type=Path)
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    try:
        recording = load_recording(options.run_directory)
        if options.command == "template":
            report = make_review_template(
                recording,
                options.settling_s,
                options.maximum_sample_gap_s,
                options.policy_evidence,
            )
        else:
            annotation_contents = options.annotations.read_bytes()
            report = score_recording(recording, read_json(annotation_contents.decode("utf-8")))
            report["annotations_sha256"] = hashlib.sha256(annotation_contents).hexdigest()
        with options.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
