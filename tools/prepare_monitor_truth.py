#!/usr/bin/env python
"""Prepare conservative, hash-bound truth from optical identity and independent local-photo fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy
from numpy.typing import NDArray

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.json_data import JsonValue
from tools import associate_monitor_codes as codes
from tools import associate_monitor_frames as optical
from tools import locate_monitor_birds as locator
from tools import score_monitor_run as scorer
from tools.register_monitor_frames import project, rectangle

MAXIMUM_ANCHOR_SPAN_S = 2.0
SELECTION_MINIMUM_IOU = 0.25
CENTRAL_BAND_FRACTION = 0.2
RECOMPUTATION_TOLERANCE_CAMERA_PX = 0.01
Point = tuple[float, float]
Bounds = tuple[float, float, float, float]
Matrix = NDArray[numpy.float64]
Group = tuple[str, str, int, int, str]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def value_hash(value: JsonValue) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def document(path: Path) -> dict[str, JsonValue]:
    return locator.mapping(scorer.read_json(path.read_text()), str(path))


def integer(value: JsonValue, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def boolean(value: JsonValue, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def matrix(value: JsonValue, name: str) -> Matrix:
    result = numpy.asarray(value, dtype=numpy.float64)
    if (
        result.shape != (3, 3)
        or not numpy.isfinite(result).all()
        or abs(float(numpy.linalg.det(result))) < 1e-9
    ):
        raise ValueError(f"{name} must be a finite nonsingular3x3 mapping")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Qualification:
    raw: dict[str, JsonValue]
    qualified_assets: frozenset[str]
    home_matrix: Matrix
    home_allowance_px: float
    usable_x_px: tuple[float, float]
    usable_y_px: tuple[float, float]
    outer_x_px: tuple[float, float]
    outer_y_px: tuple[float, float]
    home_layout: str
    width_px: int
    height_px: int
    proof_hashes: dict[str, JsonValue]


@dataclass(frozen=True)
class RenderTarget:
    identity: str
    asset: str
    subject: str
    box_css_px: tuple[float, float, float, float]
    present: bool
    visible_fraction: float
    raw: dict[str, JsonValue]


@dataclass(frozen=True)
class Render:
    group: Group
    number: int
    expected_identity: str | None
    targets: tuple[RenderTarget, ...]
    content: str
    raw: dict[str, JsonValue]
    sha256: str


@dataclass(frozen=True)
class Browser:
    path: Path
    sha256: str
    renders: dict[tuple[str, int], Render]
    codes: dict[bytes, tuple[Render, ...]]


@dataclass(frozen=True)
class RenderEvidence:
    render: Render
    first_render: int
    last_render: int
    exact: bool
    evidence: dict[str, JsonValue]


@dataclass(frozen=True)
class Photo:
    asset: str
    centre_px: Point
    bounds_px: Bounds
    photograph_corners_px: tuple[Point, ...]
    evidence_sha256: str


def direct_centre_qualification(
    policy: dict[str, JsonValue],
    validation: dict[str, JsonValue],
    source_labels: Path,
    locator_code_sha256: JsonValue,
) -> tuple[frozenset[str], dict[str, JsonValue]]:
    assets_value = policy.get("qualified_direct_centre_assets", [])
    if not isinstance(assets_value, list):
        raise ValueError("direct-centre qualified assets must be listed")
    assets = frozenset(locator.text_value(value, "direct-centre asset") for value in assets_value)
    if not assets:
        return assets, {}
    comparison_path = Path(
        locator.text_value(
            policy.get("direct_centre_validation_report"), "direct-centre comparison path"
        )
    )
    checks_path = Path(
        locator.text_value(policy.get("direct_centre_checks"), "direct-centre checks")
    )
    locations_path = Path(
        locator.text_value(policy.get("direct_centre_locations"), "direct-centre locations")
    )
    if digest(comparison_path) != policy.get("direct_centre_validation_sha256"):
        raise ValueError("direct-centre comparison hash mismatch")
    comparison = document(comparison_path)
    if comparison.get("checks_sha256") != digest(checks_path) or comparison.get(
        "locations_sha256"
    ) != digest(locations_path):
        raise ValueError("direct-centre checks/locations hash mismatch")
    checked = document(checks_path)
    located = document(locations_path)
    locator.text_value(validation.get("index_sha256"), "direct-centre reference index hash")
    if (
        located.get("source_labels_sha256") != digest(source_labels)
        or located.get("code_sha256") != locator_code_sha256
        or located.get("index_sha256") != validation.get("index_sha256")
    ):
        raise ValueError("direct-centre source index/labels/locator provenance mismatch")

    def keyed(values: JsonValue, name: str) -> dict[tuple[str, str], dict[str, JsonValue]]:
        if not isinstance(values, list):
            raise ValueError(f"{name} must be a list")
        result: dict[tuple[str, str], dict[str, JsonValue]] = {}
        for item in values:
            row = locator.mapping(item, name)
            key = (
                locator.text_value(row.get("image"), "checked image"),
                locator.text_value(row.get("asset_id"), "checked asset"),
            )
            if key in result:
                raise ValueError(f"duplicate {name} image/asset")
            result[key] = row
        return result

    checks = keyed(checked.get("checks"), "direct-centre checks")
    locations = keyed(located.get("results"), "direct-centre locations")
    results = keyed(comparison.get("results"), "direct-centre comparison")
    if checks.keys() != results.keys() or not checks.keys() <= locations.keys():
        raise ValueError("direct-centre comparison omits an annotated observation")
    source_entries = locator.mapping(document(source_labels), "source labels")["labels"]
    source_hashes = {
        locator.text_value(row.get("asset_id"), "source asset"): row.get("source_sha256")
        for value in cast(list[JsonValue], source_entries)
        for row in (locator.mapping(value, "source label"),)
    }
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for key, check in checks.items():
        location, result = locations[key], results[key]
        if (
            check.get("image_sha256") != location.get("image_sha256")
            or location.get("source_sha256") != source_hashes.get(key[1])
            or result.get("locator_status") != location.get("status")
        ):
            raise ValueError("direct-centre image/source/locator identity mismatch")
        if location.get("status") not in ("candidate", "camera_checked"):
            if result.get("status") != "rejected_by_locator":
                raise ValueError("direct-centre matcher rejection was not retained")
            rejected[key[1]] += 1
            continue
        left, top, right, bottom = locator.numbers(
            check.get("visible_bbox_camera_xyxy_px"), 4, "annotated visible bird box"
        )
        observed = ((left + right) / 2, (top + bottom) / 2)
        predicted = locator.numbers(
            location.get("visible_bbox_centre_camera_px"), 2, "direct-centre prediction"
        )
        allowance = locator.number(check.get("edge_allowance_px"), "camera edge allowance")
        error_px = float(
            numpy.linalg.norm(numpy.abs(numpy.asarray(predicted) - observed) + allowance)
        )
        if (
            result.get("status") != "within_budget"
            or result.get("registration_budget_px") != 7
            or result.get("annotation_edge_allowance_px") != allowance
            or allowance < 0
            or tuple(locator.numbers(result.get("observed_centre_px"), 2, "checked centre"))
            != observed
            or tuple(locator.numbers(result.get("predicted_centre_px"), 2, "predicted centre"))
            != predicted
            or abs(locator.number(result.get("worst_case_error_px"), "centre error") - error_px)
            > 1e-6
            or error_px > 7
        ):
            raise ValueError("direct-centre observation exceeds or misstates the fixed 7 px budget")
        accepted[key[1]] += 1
    if not assets <= accepted.keys():
        raise ValueError("direct-centre qualified asset has no passing observation")
    return assets, {
        "direct_centre_comparison_sha256": digest(comparison_path),
        "direct_centre_checks_sha256": digest(checks_path),
        "direct_centre_locations_sha256": digest(locations_path),
        "direct_centre_accepted_by_asset": dict(accepted),
        "direct_centre_locator_rejections_by_asset": dict(rejected),
        "direct_centre_scope": policy.get("direct_centre_scope"),
        "photo_corner_validation_claimed": False,
    }


def read_qualification(
    path: Path, source_labels: Path, locations: dict[str, JsonValue]
) -> Qualification:
    policy = document(path)
    if policy.get("source_labels_sha256") != digest(source_labels) or locations.get(
        "source_labels_sha256"
    ) != digest(source_labels):
        raise ValueError("source-label hash does not match qualification or local locations")
    if policy.get("settling_s") != 6 or policy.get("maximum_sample_gap_s") != 1:
        raise ValueError("qualification requires the predeclared6s settling and1s sample gap")
    validation_path = Path(
        locator.text_value(policy.get("camera_validation_report"), "camera validation path")
    )
    equivalence_path = Path(
        locator.text_value(policy.get("feature_reuse_equivalence_report"), "equivalence path")
    )
    for report_path, key in (
        (validation_path, "camera_validation_sha256"),
        (equivalence_path, "feature_reuse_equivalence_sha256"),
    ):
        if digest(report_path) != policy.get(key):
            raise ValueError(f"qualification evidence hash mismatch: {report_path}")
    validation, equivalence = document(validation_path), document(equivalence_path)
    code_hash = policy.get("qualified_locator_code_sha256")
    if locations.get("code_sha256") != code_hash or code_hash != digest(Path(locator.__file__)):
        raise ValueError("local-photo locator code is not qualified")
    if (
        equivalence.get("code_sha256_after") != code_hash
        or validation.get("code_sha256")
        not in (equivalence.get("code_sha256_before"), equivalence.get("code_sha256_after"))
        or equivalence.get("asset_image_results") != 30
        or equivalence.get("exact_result_matches") != 30
        or equivalence.get("mismatches") != []
        or equivalence.get("labels_sha256") != digest(source_labels)
    ):
        raise ValueError("cached locator lacks the pinned30-result equivalence proof")
    qualified_raw = policy.get("qualified_assets")
    if not isinstance(qualified_raw, list):
        raise ValueError("qualified assets must be listed")
    qualified = frozenset(locator.text_value(value, "qualified asset") for value in qualified_raw)
    checked = {
        locator.text_value(row.get("asset_id"), "checked asset")
        for item in cast(list[JsonValue], validation["results"])
        if (row := locator.mapping(item, "camera validation")).get("status") == "camera_checked"
    }
    direct_assets, direct_proof = direct_centre_qualification(
        policy, validation, source_labels, code_hash
    )
    if not qualified <= checked | direct_assets or validation.get("source_labels_sha256") != digest(
        source_labels
    ):
        raise ValueError("qualified assets lack pinned held-out camera checks")
    reach = locator.mapping(policy.get("reach"), "reach policy")
    for path_key, hash_key in (
        ("home_reference_image", "home_reference_image_sha256"),
        ("axis_measurement_report", "axis_measurement_sha256"),
    ):
        if digest(Path(locator.text_value(reach.get(path_key), path_key))) != reach.get(hash_key):
            raise ValueError("reach evidence hash mismatch")
    home_image = Path(cast(str, reach["home_reference_image"]))
    home_rows = optical.read_jsonl(home_image.parent / "index.jsonl")
    home_matches = [row for row in home_rows if row.get("filename") == home_image.name]
    if len(home_matches) != 1 or home_matches[0].get("sequence") != reach.get(
        "home_reference_source_sequence"
    ):
        raise ValueError("home reach image source sequence mismatch")

    def interval(key: str) -> tuple[float, float]:
        lower, upper = locator.numbers(reach.get(key), 2, key)
        if lower >= upper:
            raise ValueError("reach intervals must increase")
        return lower, upper

    allowance = locator.number(reach.get("home_mapping_allowance_camera_px"), "home allowance")
    if allowance != policy.get("registration_allowance_camera_px") or allowance != 7:
        raise ValueError("qualification requires the predeclared7px registration allowance")
    return Qualification(
        policy,
        qualified,
        matrix(reach.get("home_css_to_camera"), "home mapping"),
        allowance,
        interval("pan_usable_shift_px"),
        interval("tilt_usable_shift_px"),
        interval("pan_outer_shift_px"),
        interval("tilt_outer_shift_px"),
        codes.layout_geometry(locator.mapping(reach.get("layout_reference_frame"), "reach layout")),
        integer(reach.get("image_width_px"), "reach width"),
        integer(reach.get("image_height_px"), "reach height"),
        {
            "camera_validation_sha256": digest(validation_path),
            "feature_equivalence_sha256": digest(equivalence_path),
            "home_index_sha256": digest(home_image.parent / "index.jsonl"),
            "axis_measurement_sha256": reach["axis_measurement_sha256"],
            **direct_proof,
        },
    )


def expected_rule(
    manifest: dict[str, JsonValue], frame: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    scene_id = locator.text_value(frame.get("scene_id"), "scene_id")
    scenes = [locator.mapping(item, "scene") for item in cast(list[JsonValue], manifest["scenes"])]
    scene = [scene for scene in scenes if scene.get("scene_id") == scene_id]
    if len(scene) != 1:
        raise ValueError("browser scene is absent or ambiguous in session manifest")
    scene_frame = integer(frame.get("scene_frame"), "scene_frame")
    rules = [
        locator.mapping(item, "scene annotation")
        for item in cast(list[JsonValue], scene[0]["annotations"])
    ]
    applicable = [
        rule
        for rule in rules
        if integer(rule.get("start_frame"), "annotation start")
        <= scene_frame
        < integer(rule.get("end_frame"), "annotation end")
    ]
    if len(applicable) != 1:
        raise ValueError("browser render lacks one applicable expected-selection rule")
    return applicable[0]


def make_render(frame: dict[str, JsonValue], manifest: dict[str, JsonValue]) -> Render:
    group = (
        locator.text_value(frame.get("presentation_id"), "presentation"),
        locator.text_value(frame.get("optical_session"), "session"),
        integer(frame.get("run_number"), "run"),
        integer(frame.get("repeat_index"), "repeat"),
        locator.text_value(frame.get("scene_id"), "scene"),
    )
    rule = expected_rule(manifest, frame)
    selected = rule.get("selected_identity")
    expected = None if selected is None else locator.text_value(selected, "expected identity")
    targets: list[RenderTarget] = []
    for item in cast(list[JsonValue], frame["targets"]):
        target = locator.mapping(item, "render target")
        fraction = locator.number(target.get("visible_fraction"), "visible fraction")
        covered = locator.number(target.get("covered_fraction"), "covered fraction")
        if not 0 <= fraction <= 1 or not 0 <= covered <= 1:
            raise ValueError("render visibility fractions outside[0,1]")
        raw: dict[str, JsonValue] = {
            key: target.get(key)
            for key in (
                "intended_identity",
                "asset_id",
                "subject",
                "box_screen_css_px",
                "present",
                "visibility",
                "visible_fraction",
                "covered_fraction",
            )
        }
        targets.append(
            RenderTarget(
                locator.text_value(target.get("intended_identity"), "target identity"),
                locator.text_value(target.get("asset_id"), "asset"),
                locator.text_value(target.get("subject"), "subject"),
                rectangle(target.get("box_screen_css_px"), "target box"),
                boolean(target.get("present"), "target present"),
                fraction,
                raw,
            )
        )
    scenes = [locator.mapping(item, "scene") for item in cast(list[JsonValue], manifest["scenes"])]
    scene = next(item for item in scenes if item.get("scene_id") == group[4])
    intended = {
        locator.text_value(item.get("identity"), "manifest target identity"): locator.text_value(
            item.get("asset_id"), "manifest target asset"
        )
        for value in cast(list[JsonValue], scene["targets"])
        for item in (locator.mapping(value, "manifest target"),)
    }
    if intended != {target.identity: target.asset for target in targets}:
        raise ValueError("browser render omits or changes an intended target slot")
    identities = {target.identity for target in targets}
    if len(identities) != len(targets) or (expected is not None and expected not in identities):
        raise ValueError("render target inventory does not contain its unique expected identity")
    content: dict[str, JsonValue] = {
        "scene_id": group[4],
        "targets": [target.raw for target in targets],
        "expected_rule": rule,
    }
    return Render(
        group,
        integer(frame.get("render_number"), "render"),
        expected,
        tuple(targets),
        canonical(content),
        frame,
        value_hash(frame),
    )


def read_browser(path: Path, labels: dict[str, locator.Label]) -> Browser:
    rows = optical.read_jsonl(path)
    manifests: dict[tuple[str, str], dict[str, JsonValue]] = {}
    renders: dict[tuple[str, int], Render] = {}
    by_code: dict[bytes, list[Render]] = {}
    last_number: dict[str, int] = {}
    for row in rows:
        if row.get("kind") == "session":
            key = (
                locator.text_value(row.get("presentation_id"), "presentation"),
                locator.text_value(row.get("optical_session"), "session"),
            )
            if key in manifests:
                raise ValueError("duplicate browser session manifest")
            manifest = locator.mapping(row.get("manifest"), "session manifest")
            for item in cast(list[JsonValue], manifest["assets"]):
                asset = locator.mapping(item, "manifest asset")
                photo = locator.mapping(asset.get("photograph"), "manifest photograph")
                identifier = locator.text_value(photo.get("asset_id"), "manifest asset id")
                label = labels.get(identifier)
                if label is None or asset.get("source_sha256") != digest(
                    Path(label.photograph.source)
                ):
                    raise ValueError("browser asset does not match reviewed source labels")
                crop = label.photograph.crop
                expected_crop = {
                    "left_px": crop.left_px,
                    "top_px": crop.top_px,
                    "width_px": crop.width_px,
                    "height_px": crop.height_px,
                }
                if photo.get("crop") != expected_crop:
                    raise ValueError("browser source crop does not match reviewed labels")
            manifests[key] = manifest
        if row.get("kind") != "frame" or row.get("optical_code_screen_css_px") is None:
            continue
        key = (
            locator.text_value(row.get("presentation_id"), "presentation"),
            locator.text_value(row.get("optical_session"), "session"),
        )
        if key not in manifests:
            raise ValueError("coded browser frame lacks its session manifest")
        render = make_render(row, manifests[key])
        if render.number <= last_number.get(key[1], -1):
            raise ValueError("global render numbers must increase strictly within each session")
        last_number[key[1]] = render.number
        renders[(key[1], render.number)] = render
        by_code.setdefault(optical.frame_identity(row), []).append(render)
    if not renders:
        raise ValueError("browser contains no coded renders")
    return Browser(
        path.resolve(),
        digest(path),
        renders,
        {code: tuple(items) for code, items in by_code.items()},
    )


def bind_associations(
    path: Path, recording: scorer.Recording, browser: Browser
) -> dict[int, RenderEvidence]:
    report = document(path)
    inputs = locator.mapping(report.get("inputs"), "association inputs")
    source_index = locator.mapping(inputs.get("source_index"), "association index")
    source_browser = locator.mapping(inputs.get("source_browser"), "association browser")
    if (
        source_index.get("sha256") != recording.index_sha256
        or source_browser.get("sha256") != browser.sha256
    ):
        raise ValueError("association belongs to another source index or browser")
    if (
        Path(locator.text_value(source_index.get("path"), "association index path")).resolve()
        != recording.directory / "demo/index.jsonl"
    ):
        raise ValueError("association source path aliases another recording")
    by_name = {frame.filename: frame for frame in recording.frames}
    seen: set[str] = set()
    sequences: set[int] = set()
    anchors: dict[int, RenderEvidence] = {}
    for item in cast(list[JsonValue], report["frames"]):
        row = locator.mapping(item, "association row")
        name = locator.text_value(row.get("image"), "association image")
        source = by_name.get(name)
        sequence = integer(row.get("source_sequence"), "association source sequence")
        if (
            source is None
            or name in seen
            or sequence in sequences
            or sequence != source.sequence
            or row.get("image_sha256") != source.jpeg_sha256
            or Path(locator.text_value(row.get("image_path"), "association image path")).resolve()
            != recording.directory / "demo" / source.filename
        ):
            raise ValueError("association image/sequence/hash mismatch or duplicate")
        seen.add(name)
        sequences.add(sequence)
        if row.get("status") != "optically_associated":
            continue
        payload = bytes.fromhex(locator.text_value(row.get("optical_code_hex"), "optical payload"))
        matches = browser.codes.get(payload, ())
        if len(matches) != 1:
            raise ValueError("association payload does not identify one actual browser record")
        render = matches[0]
        associated = locator.mapping(row.get("association"), "association identity")
        if (
            any(
                associated.get(key) != render.raw.get(key)
                for key in (
                    "presentation_id",
                    "optical_session",
                    "run_number",
                    "render_number",
                    "scene_id",
                    "scene_frame",
                    "sequence_frame",
                )
            )
            or row.get("browser_frame_sha256") != render.sha256
        ):
            raise ValueError("association identity/browser-frame hash mismatch")
        anchors[source.sequence] = RenderEvidence(
            render,
            render.number,
            render.number,
            True,
            {
                "kind": "dual-CRC exact render",
                "association_row_sha256": value_hash(row),
                "browser_frame_sha256": render.sha256,
                "image_sha256": source.jpeg_sha256,
                "source_sequence": source.sequence,
            },
        )
    return anchors


def bracket_render(
    position: int,
    frames: tuple[scorer.RecordedFrame, ...],
    anchors: dict[int, RenderEvidence],
    browser: Browser,
) -> tuple[RenderEvidence | None, str]:
    current = frames[position]
    if current.sequence in anchors:
        return anchors[current.sequence], "exact"
    positions = [index for index, frame in enumerate(frames) if frame.sequence in anchors]
    insertion = bisect_left(positions, position)
    if insertion == 0 or insertion == len(positions):
        return None, "no two surrounding optical anchors"
    lower_position, upper_position = positions[insertion - 1], positions[insertion]
    lower_frame, upper_frame = frames[lower_position], frames[upper_position]
    if upper_frame.monotonic_s - lower_frame.monotonic_s > MAXIMUM_ANCHOR_SPAN_S:
        return None, "optical anchor span exceeds2s"
    lower, upper = anchors[lower_frame.sequence], anchors[upper_frame.sequence]
    if lower.render.group != upper.render.group or lower.render.number > upper.render.number:
        return None, "optical anchors differ in session/run/repeat/scene or render order"
    candidates = []
    for number in range(lower.render.number, upper.render.number + 1):
        render = browser.renders.get((lower.render.group[1], number))
        if render is None:
            return None, "missing browser render within optical interval"
        if render.group != lower.render.group or render.content != lower.render.content:
            return None, "target content or expected-selection rule changes within interval"
        candidates.append(render)
    evidence: dict[str, JsonValue] = {
        "kind": "invariant content between optical anchors",
        "lower_anchor": lower.evidence,
        "upper_anchor": upper.evidence,
        "anchor_span_s": upper_frame.monotonic_s - lower_frame.monotonic_s,
        "browser_frame_sha256": [render.sha256 for render in candidates],
        "canonical_content_sha256": hashlib.sha256(lower.render.content.encode()).hexdigest(),
        "exact_exposure_or_render_claimed": lower.render.number == upper.render.number,
    }
    return RenderEvidence(
        lower.render,
        lower.render.number,
        upper.render.number,
        lower.render.number == upper.render.number,
        evidence,
    ), "invariant_exact" if lower.render.number == upper.render.number else "invariant_interval"


def source_centre(label: locator.Label) -> Point:
    left, top, right, bottom = label.bounds
    return (left + right) / 2, (top + bottom) / 2


def projected_bounds(mapping: Matrix, centre: Point, allowance: Point, extra_px: float) -> Bounds:
    positions = [
        (centre[0] + x, centre[1] + y)
        for x in (-allowance[0], allowance[0])
        for y in (-allowance[1], allowance[1])
    ]
    denominators = [float(mapping[2] @ numpy.array((x, y, 1.0))) for x, y in positions]
    if min(denominators) <= 0 <= max(denominators):
        raise ValueError("uncertainty rectangle crosses mapping horizon")
    points = [project(mapping, position) for position in positions]
    return (
        min(point[0] for point in points) - extra_px,
        min(point[1] for point in points) - extra_px,
        max(point[0] for point in points) + extra_px,
        max(point[1] for point in points) + extra_px,
    )


def reach_bounds(
    target: RenderTarget, render: Render, label: locator.Label, policy: Qualification
) -> tuple[bool | None, dict[str, JsonValue]]:
    if codes.layout_geometry(render.raw) != policy.home_layout:
        return None, {"reason": "browser layout differs from reach reference"}
    left, top, width, height = target.box_css_px
    crop = label.photograph.crop
    source_x, source_y = source_centre(label)
    centre = (left + source_x / crop.width_px * width, top + source_y / crop.height_px * height)
    allowance = (
        label.edge_allowance_px / crop.width_px * width,
        label.edge_allowance_px / crop.height_px * height,
    )
    try:
        bounds = projected_bounds(policy.home_matrix, centre, allowance, policy.home_allowance_px)
    except ValueError:
        return None, {"reason": "home mapping cannot bound this source-centre rectangle"}
    axes = (
        (bounds[0], bounds[2], policy.usable_x_px, policy.outer_x_px, policy.width_px),
        (bounds[1], bounds[3], policy.usable_y_px, policy.outer_y_px, policy.height_px),
    )
    guaranteed = True
    impossible = False
    for lower, upper, usable, outer, dimension_px in axes:
        band_lower = dimension_px * (1 - CENTRAL_BAND_FRACTION) / 2
        band_upper = dimension_px * (1 + CENTRAL_BAND_FRACTION) / 2
        guaranteed &= lower + usable[1] >= band_lower and upper + usable[0] <= band_upper
        impossible |= upper + outer[1] < band_lower or lower + outer[0] > band_upper
    reachable = True if guaranteed else False if impossible else None
    return reachable, {
        "home_centre_bounds_camera_xyxy_px": list(bounds),
        "reachable": reachable,
        "definition": (
            "existence of a permitted pose within the central 20% band; "
            "empirical single-sweep rectangular-shift approximation"
        ),
    }


def local_photo(
    row: dict[str, JsonValue],
    label: locator.Label,
    frame: scorer.RecordedFrame,
    policy: Qualification,
) -> Photo | None:
    if row.get("status") not in ("candidate", "camera_checked"):
        return None
    gates = (
        locator.number(row.get("matches"), "matches") >= locator.MINIMUM_MATCHES,
        locator.number(row.get("fit_inliers"), "inliers") >= locator.MINIMUM_INLIERS,
        locator.number(row.get("held_out_count"), "held out") >= locator.MINIMUM_HELD_OUT,
        locator.number(row.get("held_out_within_2px_fraction"), "held fraction")
        >= locator.MINIMUM_HELD_OUT_FRACTION,
        locator.number(row.get("held_out_median_error_px"), "held median")
        < locator.MAXIMUM_MEDIAN_ERROR_PX,
        locator.number(row.get("source_hull_fraction"), "hull") >= locator.MINIMUM_HULL_FRACTION,
        row.get("centre_inside_source_hull") is True,
    )
    if not all(gates):
        return None
    matches = integer(row.get("matches"), "matches")
    fit_count = integer(row.get("fit_count"), "fit count")
    if (
        fit_count + integer(row.get("held_out_count"), "held out") != matches
        or integer(row.get("fit_inliers"), "inliers") > fit_count
    ):
        raise ValueError("local fit sample counts are inconsistent")
    mapping = matrix(row.get("source_to_camera_homography"), "local photo mapping")
    corners = numpy.asarray(row.get("photograph_corners_camera_px"), dtype=numpy.float32)
    if corners.shape != (4, 2) or not numpy.isfinite(corners).all():
        return None
    crop = label.photograph.crop
    expected_corners = numpy.array(
        [
            project(mapping, point)
            for point in (
                (0, 0),
                (crop.width_px, 0),
                (crop.width_px, crop.height_px),
                (0, crop.height_px),
            )
        ]
    )
    if float(numpy.max(numpy.abs(expected_corners - corners))) > RECOMPUTATION_TOLERANCE_CAMERA_PX:
        raise ValueError("local photograph rectangle does not match its source mapping")
    sides = numpy.linalg.norm(corners - numpy.roll(corners, -1, axis=0), axis=1)
    if (
        not cv2.isContourConvex(corners)
        or cv2.contourArea(corners, oriented=True) <= 100
        or min(sides) < 10
        or max(sides) > 4 * min(sides)
        or not numpy.all(corners[:, 0] >= 0)
        or not numpy.all(corners[:, 0] < frame.width_px)
        or not numpy.all(corners[:, 1] >= 0)
        or not numpy.all(corners[:, 1] < frame.height_px)
    ):
        return None
    centre = cast(
        Point, locator.numbers(row.get("visible_bbox_centre_camera_px"), 2, "bird centre")
    )
    bounds = cast(
        Bounds,
        locator.numbers(row.get("centre_allowance_bounds_camera_xyxy_px"), 4, "centre allowance"),
    )
    if (
        row.get("edge_allowance_source_px") != label.edge_allowance_px
        or row.get("registration_allowance_camera_px") != policy.home_allowance_px
        or label.registration_allowance_px != policy.home_allowance_px
    ):
        raise ValueError("local centre allowance differs from fixed source/registration policy")
    calculated = project(mapping, source_centre(label))
    calculated_bounds = projected_bounds(
        mapping,
        source_centre(label),
        (label.edge_allowance_px, label.edge_allowance_px),
        label.registration_allowance_px,
    )
    if (
        max(
            abs(actual - expected)
            for actual, expected in zip(
                centre + bounds, calculated + calculated_bounds, strict=True
            )
        )
        > RECOMPUTATION_TOLERANCE_CAMERA_PX
    ):
        raise ValueError("local centre/bounds do not match reviewed source bird box")
    if (
        not bounds[0] <= centre[0] <= bounds[2]
        or not bounds[1] <= centre[1] <= bounds[3]
        or not 0 <= centre[0] < frame.width_px
        or not 0 <= centre[1] < frame.height_px
    ):
        return None
    return Photo(
        label.photograph.asset_id,
        centre,
        bounds,
        tuple((float(point[0]), float(point[1])) for point in corners),
        value_hash(row),
    )


def bind_locations(
    locations: dict[str, JsonValue], recording: scorer.Recording, labels: dict[str, locator.Label]
) -> dict[str, dict[str, dict[str, JsonValue]]]:
    if locations.get("index_sha256") != recording.index_sha256:
        raise ValueError("local-photo locations belong to another recording")
    by_name = {frame.filename: frame for frame in recording.frames}
    result: dict[str, dict[str, dict[str, JsonValue]]] = {}
    for item in cast(list[JsonValue], locations["results"]):
        row = locator.mapping(item, "location row")
        name = locator.text_value(row.get("image"), "location image")
        asset = locator.text_value(row.get("asset_id"), "location asset")
        frame = by_name.get(name)
        label = labels.get(asset)
        if (
            frame is None
            or label is None
            or row.get("image_sha256") != frame.jpeg_sha256
            or row.get("source_sha256") != digest(Path(label.photograph.source))
        ):
            raise ValueError("local-photo image/source hash or asset ID mismatch")
        images = result.setdefault(name, {})
        if asset in images:
            raise ValueError("duplicate local-photo image/asset result")
        images[asset] = row
    return result


def intersection_over_union(first: Bounds, second: tuple[Point, ...]) -> float:
    left, top, right, bottom = first
    rectangle_points = numpy.array(
        ((left, top), (right, top), (right, bottom), (left, bottom)), dtype=numpy.float32
    )
    polygon = numpy.asarray(second, dtype=numpy.float32)
    area, _ = cv2.intersectConvexConvex(rectangle_points, polygon)
    total = (right - left) * (bottom - top) + abs(cv2.contourArea(polygon)) - area
    return float(area / total) if total > 0 else 0.0


def selection_mapping(
    selected_bbox: Bounds | None, targets: tuple[RenderTarget, ...], photos: dict[str, Photo]
) -> tuple[str | None, dict[str, JsonValue]]:
    matches = [
        target.identity
        for target in targets
        if selected_bbox is not None
        and target.asset in photos
        and intersection_over_union(selected_bbox, photos[target.asset].photograph_corners_px)
        >= SELECTION_MINIMUM_IOU
    ]
    return (matches[0] if len(matches) == 1 else None), {
        "candidate_identities": cast(JsonValue, matches),
        "minimum_iou": SELECTION_MINIMUM_IOU,
        "basis": (
            "selected bbox used only for identity mapping against independent photo rectangles"
        ),
    }


def prepare_label(
    evidence: RenderEvidence,
    frame: scorer.RecordedFrame,
    rows: dict[str, dict[str, JsonValue]],
    labels: dict[str, locator.Label],
    policy: Qualification,
    selected_bbox: Bounds | None,
) -> tuple[dict[str, JsonValue] | None, dict[str, JsonValue]]:
    render = evidence.render
    details: dict[str, JsonValue] = {
        "render_evidence": evidence.evidence,
        "targets": [],
        "expected_identity": render.expected_identity,
        "intended_target_inventory": [target.raw for target in render.targets],
    }
    photos: dict[str, Photo] = {}
    for asset, row in rows.items():
        photo = local_photo(row, labels[asset], frame, policy)
        if photo is not None:
            photos[asset] = photo
    details["local_photo_evidence"] = [
        {
            "asset_id": asset,
            "status": row.get("status"),
            "reason": row.get("reason"),
            "row_sha256": value_hash(row),
            "fit_gates_pass": asset in photos,
        }
        for asset, row in rows.items()
    ]
    visible_assets = {
        target.asset for target in render.targets if target.present and target.visible_fraction > 0
    }
    if any(asset not in visible_assets for asset in photos):
        return None, details | {
            "reason": "independent photograph evidence contradicts controlled absent inventory"
        }
    if len({target.asset for target in render.targets}) != len(render.targets):
        return None, details | {
            "reason": "repeated source asset makes photograph identity ambiguous"
        }
    truths: list[JsonValue] = []
    for target in render.targets:
        if target.asset not in labels:
            raise ValueError("render target lacks a reviewed source label")
        photo = photos.get(target.asset)
        controlled_visible = target.present and target.visible_fraction > 0
        visible: bool | None = (
            True
            if controlled_visible and photo is not None
            else None
            if controlled_visible
            else False
        )
        reachable, reach_evidence = reach_bounds(target, render, labels[target.asset], policy)
        qualified = (
            visible is True
            and target.visible_fraction == 1
            and photo is not None
            and target.asset in policy.qualified_assets
        )
        centre = photo.centre_px if qualified and photo is not None else None
        uncertainty = (
            (
                max(
                    photo.centre_px[0] - photo.bounds_px[0], photo.bounds_px[2] - photo.centre_px[0]
                ),
                max(
                    photo.centre_px[1] - photo.bounds_px[1], photo.bounds_px[3] - photo.centre_px[1]
                ),
            )
            if qualified and photo is not None
            else (0.0, 0.0)
        )
        truth_evidence: dict[str, JsonValue] = {
            "asset_id": target.asset,
            "subject": target.subject,
            "visible_evidence": (
                "independent local photo fit"
                if visible is True
                else "local-photo visibility unresolved"
                if visible is None
                else "controlled render absent or fully occluded"
            ),
            "local_fit_sha256": photo.evidence_sha256 if photo else None,
            "centre_qualified": qualified,
            "complete_source_crop_visible": target.visible_fraction == 1,
            "reach": reach_evidence,
        }
        truths.append(
            {
                "identity": target.identity,
                "visible": visible,
                "reachable": reachable,
                "registration_valid": qualified,
                "centre_method": "validated_local_photograph",
                "centre_px": {"x_px": centre[0], "y_px": centre[1]} if centre is not None else None,
                "centre_uncertainty_px": {"x_px": uncertainty[0], "y_px": uncertainty[1]},
                "evidence": canonical(truth_evidence),
            }
        )
    selected, selection_evidence = selection_mapping(selected_bbox, render.targets, photos)
    details["targets"] = truths
    details["selection_mapping"] = selection_evidence
    label: dict[str, JsonValue] = {
        "presentation_id": render.group[0],
        "run_number": render.group[2],
        "repeat_index": render.group[3],
        "scene_id": render.group[4],
        "render_number": evidence.first_render if evidence.exact else None,
        "inventory_complete": True,
        "expected_identity": render.expected_identity,
        "targets": truths,
        "selection": {"identity": selected, "evidence": canonical(selection_evidence)}
        if selected is not None and frame.selected_track_id is not None
        else None,
        "evidence": canonical(evidence.evidence),
    }
    if not evidence.exact:
        label["render_interval"] = {
            "first_render_number": evidence.first_render,
            "last_render_number": evidence.last_render,
            "equivalent_content_confirmed": True,
            "evidence": canonical(evidence.evidence),
        }
    return label, details


def selected_boxes(recording: scorer.Recording) -> dict[int, Bounds | None]:
    result: dict[int, Bounds | None] = {}
    for row in optical.read_jsonl(recording.directory / "demo/index.jsonl"):
        metadata = locator.mapping(row.get("metadata"), "recording metadata")
        target = locator.mapping(metadata.get("target"), "recorded selected target")
        track = target.get("track")
        bounds = None
        if isinstance(track, dict) and isinstance(track.get("detection"), dict):
            box = cast(dict[str, JsonValue], track["detection"]).get("bbox")
            if isinstance(box, list) and len(box) == 4:
                bounds = cast(Bounds, locator.numbers(box, 4, "selected bbox"))
        result[integer(row.get("sequence"), "recorded sequence")] = bounds
    return result


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("browser", type=Path)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    try:
        input_paths = (
            options.run_directory / "demo/index.jsonl",
            options.run_directory / "detections.jsonl",
            options.browser,
            options.associations,
            options.locations,
            options.source_labels,
            options.qualification,
        )
        input_hashes = {str(path.resolve()): digest(path) for path in input_paths}
        code_paths = (
            Path(__file__),
            Path(scorer.__file__),
            Path(locator.__file__),
            Path(codes.__file__),
        )
        code_hashes = {str(path): digest(path) for path in code_paths}
        recording = scorer.load_recording(options.run_directory)
        if any(
            (recording.directory / "demo" / frame.filename).is_symlink()
            for frame in recording.frames
        ):
            raise ValueError("recorded JPEGs must not alias other images through symlinks")
        labels = {
            label.photograph.asset_id: label for label in locator.load_labels(options.source_labels)
        }
        locations = document(options.locations)
        policy = read_qualification(options.qualification, options.source_labels, locations)
        if any(
            (frame.width_px, frame.height_px) != (policy.width_px, policy.height_px)
            for frame in recording.frames
        ):
            raise ValueError("recording image dimensions differ from reach qualification")
        browser = read_browser(options.browser, labels)
        anchors = bind_associations(options.associations, recording, browser)
        local = bind_locations(locations, recording, labels)
        boxes = selected_boxes(recording)
        annotations = scorer.make_review_template(
            recording, 6, 1, f"predeclared qualification SHA256{digest(options.qualification)}"
        )
        entries = cast(list[dict[str, JsonValue]], annotations["frames"])
        prepared: list[JsonValue] = []
        for position, (frame, entry) in enumerate(zip(recording.frames, entries, strict=True)):
            evidence, mode = bracket_render(position, recording.frames, anchors, browser)
            details: dict[str, JsonValue] = {
                "filename": frame.filename,
                "source_sequence": frame.sequence,
                "jpeg_sha256": frame.jpeg_sha256,
                "render_mode": mode,
            }
            if evidence is not None:
                label, label_details = prepare_label(
                    evidence,
                    frame,
                    local.get(frame.filename, {}),
                    labels,
                    policy,
                    boxes[frame.sequence],
                )
                entry["label"] = label
                details.update(label_details)
            else:
                details["reason"] = mode
            details["label_available"] = entry["label"] is not None
            prepared.append(details)
        scored = scorer.score_recording(recording, annotations)
        scored["global_verdict"] = "inconclusive"
        if any(digest(Path(path)) != checksum for path, checksum in input_hashes.items()):
            raise ValueError("an input changed during preparation; rerun with a frozen policy")
        if any(digest(Path(path)) != checksum for path, checksum in code_hashes.items()):
            raise ValueError("preparation/scoring code changed during preparation")
        options.output.mkdir(parents=True, exist_ok=False)
        (options.output / "qualification-snapshot.json").write_text(
            json.dumps(policy.raw, indent=2, allow_nan=False) + "\n"
        )
        annotation_path = options.output / "annotations.json"
        annotation_path.write_text(json.dumps(annotations, indent=2, allow_nan=False) + "\n")
        scored["annotations_sha256"] = digest(annotation_path)
        (options.output / "score.json").write_text(
            json.dumps(scored, indent=2, allow_nan=False) + "\n"
        )
        report: dict[str, JsonValue] = {
            "scope": (
                "conservative retained-recording preparation; global verdict remains inconclusive"
            ),
            "inputs_sha256": cast(JsonValue, input_hashes),
            "qualification_evidence": policy.proof_hashes,
            "qualification_snapshot_sha256": digest(options.output / "qualification-snapshot.json"),
            "qualification_bound_before_labels": True,
            "code_sha256": cast(JsonValue, code_hashes),
            "annotations_sha256": digest(annotation_path),
            "score_sha256": digest(options.output / "score.json"),
            "recording_issues": list(recording.issues),
            "counts": {
                "frames": len(prepared),
                "labels": sum(
                    cast(dict[str, JsonValue], row)["label_available"] is True for row in prepared
                ),
                "render_modes": dict(
                    Counter(
                        cast(str, cast(dict[str, JsonValue], row)["render_mode"])
                        for row in prepared
                    )
                ),
                "null_label_reasons": dict(
                    Counter(
                        cast(str, cast(dict[str, JsonValue], row).get("reason", "unknown"))
                        for row in prepared
                        if cast(dict[str, JsonValue], row)["label_available"] is False
                    )
                ),
            },
            "frames": prepared,
        }
        (options.output / "preparation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(2, str(error) + "\n")
    print(options.output / "preparation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
