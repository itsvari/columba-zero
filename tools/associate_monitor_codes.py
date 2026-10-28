#!/usr/bin/env python
"""Associate camera images by dual optical CRC, without claiming geometric registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.json_data import JsonValue
from tools import associate_monitor_frames as association
from tools import detect_monitor_markers, register_monitor_frames
from tools.register_monitor_frames import marker_screen_points, point, project, rectangle

TARGET_MARGIN_CAMERA_PX = 4
CODE_ROI_MARGIN_CAMERA_PX = 5
MARKER_ROI_RADIUS_CAMERA_PX = 22
SIFT_FEATURE_LIMIT = 2000
SIFT_CONTRAST_THRESHOLD = 0.02
SIFT_MAXIMUM_KEYPOINT_SIZE_PX = 12
LOWE_RATIO = 0.72
RANSAC_RESIDUAL_CAMERA_PX = 2.0
MINIMUM_INLIERS = 10
MINIMUM_INLIER_FRACTION = 0.45
MINIMUM_INLIER_WIDTH_CAMERA_PX = 50
MINIMUM_INLIER_HEIGHT_CAMERA_PX = 80
MINIMUM_QUADRILATERAL_AREA_RATIO = 0.5
MAXIMUM_QUADRILATERAL_AREA_RATIO = 2.0
MASKED_TARGET_INTENSITY = 127
STATIC_BOUNDARY_REGIONS_CAMERA_PX = ((0, 300, 800, 45), (775, 0, 25, 345))


@dataclass(frozen=True)
class Reference:
    name: str
    frame: dict[str, JsonValue]
    matrix: association.Matrix
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    marker_area_camera_px2: float
    provenance: dict[str, JsonValue]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectangle_polygon(
    frame: dict[str, JsonValue], key: str, matrix: association.Matrix
) -> np.ndarray[tuple[int, ...], np.dtype[np.int32]]:
    left, top, width, height = rectangle(frame[key], key)
    positions = (
        (left, top),
        (left + width, top),
        (left + width, top + height),
        (left, top + height),
    )
    return np.rint([project(matrix, position) for position in positions]).astype(np.int32)


def cell_quality(
    image: association.Image,
    frame: dict[str, JsonValue],
    matrix: association.Matrix,
    upper: bool,
    diagnostics: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    key = "optical_check_screen_css_px" if upper else "optical_code_screen_css_px"
    left, top, width, height = rectangle(frame[key], key)
    cell_width, cell_height = width / association.CODE_COLUMNS, height / association.CODE_ROWS
    centres = [
        (left + (column + 0.5) * cell_width, top + (row + 0.5) * cell_height)
        for row in range(association.CODE_ROWS)
        for column in range(association.CODE_COLUMNS)
    ]
    minimum_cell_camera_px = min(
        min(
            math.dist(
                project(matrix, centre), project(matrix, (centre[0] + cell_width, centre[1]))
            ),
            math.dist(
                project(matrix, centre), project(matrix, (centre[0], centre[1] + cell_height))
            ),
        )
        for centre in centres
    )
    positions = np.array(
        [
            [
                project(matrix, (x + offset_x * cell_width, y + offset_y * cell_height))
                for offset_x, offset_y in ((0, 0), (-0.15, 0), (0.15, 0), (0, -0.15), (0, 0.15))
            ]
            for x, y in centres
        ],
        dtype=np.float64,
    )
    translation = point(
        diagnostics.get("selected_shift_camera_px", [0.0, 0.0]), "sampling translation"
    )
    grey = cast(association.Image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    intensities = association.sample_code(grey, positions, translation)
    low, high = np.percentile(intensities, (10, 90))
    threshold = float((low + high) / 2)
    return {
        "minimum_cell_camera_px": minimum_cell_camera_px,
        "contrast": float(high - low),
        "minimum_class_margin": float(np.min(np.abs(intensities - threshold))),
        "ambiguous_cells": int(
            np.sum(np.abs(intensities - threshold) < association.MINIMUM_CLASS_MARGIN)
        ),
        "sampling_translation_camera_px": list(translation),
    }


def decode_pair(
    image: association.Image, frame: dict[str, JsonValue], matrix: association.Matrix
) -> tuple[dict[str, JsonValue], bytes | None, bool]:
    bands: dict[str, JsonValue] = {}
    payloads: dict[str, bytes] = {}
    veto = False
    for upper, name in ((False, "lower"), (True, "upper")):
        diagnostics: dict[str, JsonValue] = {}
        band: dict[str, JsonValue] = {}
        try:
            payload = association.decode_code(
                image, frame, matrix, upper=upper, diagnostics=diagnostics
            )
            payloads[name] = payload
            band.update({"crc_valid": True, "payload_hex": payload.hex(), "sampling": diagnostics})
        except (ValueError, cv2.error) as error:
            reason = str(error)
            band.update({"crc_valid": False, "reason": reason})
            veto |= "multiple CRC-checked" in reason
        try:
            band["cell_quality"] = cell_quality(image, frame, matrix, upper, diagnostics)
        except (ValueError, cv2.error, OverflowError) as error:
            band["cell_quality_error"] = str(error)
        bands[name] = band
    if len(payloads) == 2 and payloads["lower"] != payloads["upper"]:
        return {"bands": bands, "reason": "independently CRC-valid bands disagree"}, None, True
    matched_payload = payloads.get("lower") if len(payloads) == 2 else None
    return {"bands": bands}, matched_payload, veto


def features(
    image: association.Image, mask: association.Image | None = None
) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray[tuple[int, ...], np.dtype[np.float32]] | None]:
    sift = cv2.SIFT.create(nfeatures=SIFT_FEATURE_LIMIT, contrastThreshold=SIFT_CONTRAST_THRESHOLD)
    points, descriptors = sift.detectAndCompute(image, mask)
    if descriptors is None:
        return (), None
    retained = [
        index
        for index, keypoint in enumerate(points)
        if keypoint.size <= SIFT_MAXIMUM_KEYPOINT_SIZE_PX
    ]
    return tuple(points[index] for index in retained), cast(
        np.ndarray[tuple[int, ...], np.dtype[np.float32]], descriptors[retained]
    )


def sift_proposal(
    reference: Reference,
    current_points: tuple[cv2.KeyPoint, ...],
    current_descriptors: np.ndarray[tuple[int, ...], np.dtype[np.float32]] | None,
) -> tuple[dict[str, JsonValue], association.Matrix | None]:
    evidence: dict[str, JsonValue] = {
        "method": "masked-reference-camera SIFT/RANSAC",
        "reference_image": reference.name,
        "reference_keypoints": len(reference.keypoints),
        "current_keypoints": len(current_points),
    }
    if current_descriptors is None:
        return evidence | {"reason": "current image has no features"}, None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbours = matcher.knnMatch(reference.descriptors, current_descriptors, k=2)
    matches = [
        pair[0]
        for pair in neighbours
        if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance
    ]
    evidence["ratio_test_matches"] = len(matches)
    if len(matches) < MINIMUM_INLIERS:
        return evidence | {"reason": "too few ratio-test matches"}, None
    source = np.array(
        [reference.keypoints[match.queryIdx].pt for match in matches], dtype=np.float32
    )
    destination = np.array(
        [current_points[match.trainIdx].pt for match in matches], dtype=np.float32
    )
    transform, inlier_mask = cv2.findHomography(
        source, destination, cv2.RANSAC, RANSAC_RESIDUAL_CAMERA_PX, maxIters=2000, confidence=0.995
    )
    if transform is None or inlier_mask is None:
        return evidence | {"reason": "RANSAC homography unavailable"}, None
    inliers = inlier_mask.ravel().astype(bool)
    count = int(inliers.sum())
    evidence.update({"inliers": count, "inlier_fraction": count / len(matches)})
    if count < MINIMUM_INLIERS or count / len(matches) < MINIMUM_INLIER_FRACTION:
        return evidence | {"reason": "insufficient RANSAC inliers"}, None
    span = np.ptp(source[inliers], axis=0)
    evidence["inlier_span_reference_camera_px"] = cast(JsonValue, span.tolist())
    if span[0] < MINIMUM_INLIER_WIDTH_CAMERA_PX or span[1] < MINIMUM_INLIER_HEIGHT_CAMERA_PX:
        return evidence | {"reason": "inliers do not span both code-band levels"}, None
    matrix = transform @ reference.matrix
    corners = np.array(
        [project(matrix, position) for position in marker_screen_points(reference.frame)],
        dtype=np.float32,
    )
    area_ratio = abs(cv2.contourArea(corners)) / reference.marker_area_camera_px2
    upright = (
        corners[0, 0] < corners[1, 0]
        and corners[3, 0] < corners[2, 0]
        and corners[0, 1] < corners[3, 1]
        and corners[1, 1] < corners[2, 1]
    )
    evidence["marker_quadrilateral_area_ratio"] = area_ratio
    if (
        not cv2.isContourConvex(corners)
        or not upright
        or not MINIMUM_QUADRILATERAL_AREA_RATIO <= area_ratio <= MAXIMUM_QUADRILATERAL_AREA_RATIO
    ):
        return evidence | {"reason": "sampling proposal is folded or has implausible scale"}, None
    evidence["reference_camera_to_current_camera_homography"] = cast(JsonValue, transform.tolist())
    evidence["sampling_css_to_current_camera_homography"] = cast(JsonValue, matrix.tolist())
    evidence["inlier_reference_camera_px"] = cast(JsonValue, source[inliers].tolist())
    evidence["inlier_current_camera_px"] = cast(JsonValue, destination[inliers].tolist())
    return evidence, matrix


@dataclass(frozen=True)
class IndexedImage:
    name: str
    sequence: int
    path: Path


@dataclass(frozen=True)
class CaptureIndex:
    path: Path
    sha256: str
    images: tuple[IndexedImage, ...]


@dataclass(frozen=True)
class BrowserExport:
    path: Path
    sha256: str
    layout: str
    frame: dict[str, JsonValue]
    frames_by_code: dict[bytes, list[dict[str, JsonValue]]]


def read_index(path: Path) -> CaptureIndex:
    records = association.read_jsonl(path)
    images: list[IndexedImage] = []
    names: set[str] = set()
    sequences: set[int] = set()
    for record in records:
        name, sequence = record.get("filename"), record.get("sequence")
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ValueError("index image names must be unique basenames")
        if type(sequence) is not int or sequence < 0 or sequence in sequences:
            raise ValueError("index source sequences must be unique non-negative integers")
        image_path = path.resolve().parent / name
        if image_path.is_symlink() or not image_path.is_file():
            raise ValueError(f"index image is missing or a symlink: {name}")
        names.add(name)
        sequences.add(sequence)
        images.append(IndexedImage(name, sequence, image_path))
    if not images:
        raise ValueError("index contains no images")
    return CaptureIndex(path.resolve(), digest(path), tuple(images))


def layout_geometry(frame: dict[str, JsonValue]) -> str:
    geometry = [marker_screen_points(frame)] + [
        rectangle(frame.get(key), key)
        for key in (
            "target_region_screen_css_px",
            "optical_code_screen_css_px",
            "optical_check_screen_css_px",
        )
    ]
    return json.dumps(geometry, sort_keys=True, allow_nan=False)


def read_browser(path: Path) -> BrowserExport:
    frames = [
        frame
        for frame in association.read_jsonl(path)
        if frame.get("kind") == "frame" and frame.get("optical_code_screen_css_px") is not None
    ]
    if not frames:
        raise ValueError("browser export contains no optically coded frames")
    layouts = {layout_geometry(frame) for frame in frames}
    if len(layouts) != 1:
        raise ValueError("one fixed browser layout is required per export")
    by_code: dict[bytes, list[dict[str, JsonValue]]] = {}
    for frame in frames:
        by_code.setdefault(association.frame_identity(frame), []).append(frame)
    return BrowserExport(path.resolve(), digest(path), next(iter(layouts)), frames[0], by_code)


def read_image(path: Path) -> tuple[association.Image, str]:
    source = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode JPEG: {path.name}")
    return cast(association.Image, image), hashlib.sha256(source).hexdigest()


def valid_band_payloads(optical: dict[str, JsonValue]) -> set[bytes]:
    bands = cast(dict[str, dict[str, JsonValue]], optical["bands"])
    return {
        bytes.fromhex(cast(str, band["payload_hex"]))
        for band in bands.values()
        if band.get("crc_valid") is True
    }


def reference_features(
    image: association.Image,
    frame: dict[str, JsonValue],
    matrix: association.Matrix,
    markers: dict[str, JsonValue],
) -> tuple[
    tuple[cv2.KeyPoint, ...],
    np.ndarray[tuple[int, ...], np.dtype[np.float32]] | None,
    dict[str, JsonValue],
]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for upper in (False, True):
        region = np.zeros_like(mask)
        key = "optical_check_screen_css_px" if upper else "optical_code_screen_css_px"
        cv2.fillConvexPoly(region, rectangle_polygon(frame, key, matrix), 255)
        side = 2 * CODE_ROI_MARGIN_CAMERA_PX + 1
        mask |= cast(association.Image, cv2.dilate(region, np.ones((side, side), dtype=np.uint8)))
    for value in markers.values():
        position = point(value, "reference marker")
        cv2.circle(
            mask, (round(position[0]), round(position[1])), MARKER_ROI_RADIUS_CAMERA_PX, 255, -1
        )
    for left_px, top_px, width_px, height_px in STATIC_BOUNDARY_REGIONS_CAMERA_PX:
        mask[top_px : top_px + height_px, left_px : left_px + width_px] = 255
    excluded = np.zeros_like(mask)
    target = rectangle_polygon(frame, "target_region_screen_css_px", matrix)
    cv2.fillConvexPoly(excluded, target, 255)
    side = 2 * TARGET_MARGIN_CAMERA_PX + 1
    excluded = cast(association.Image, cv2.dilate(excluded, np.ones((side, side), dtype=np.uint8)))
    mask[excluded > 0] = 0
    grey = cast(association.Image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    grey[excluded > 0] = MASKED_TARGET_INTENSITY
    keypoints, descriptors = features(grey, mask)
    details: dict[str, JsonValue] = {
        "feature_count": len(keypoints),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "mask_dimensions_px": [mask.shape[1], mask.shape[0]],
        "target_polygon_reference_camera_px": cast(JsonValue, target.tolist()),
        "target_exclusion_margin_camera_px": TARGET_MARGIN_CAMERA_PX,
        "code_roi_margin_camera_px": CODE_ROI_MARGIN_CAMERA_PX,
        "marker_roi_radius_camera_px": MARKER_ROI_RADIUS_CAMERA_PX,
        "static_boundary_rois_camera_px": [
            list(region) for region in STATIC_BOUNDARY_REGIONS_CAMERA_PX
        ],
        "masked_target_intensity": MASKED_TARGET_INTENSITY,
    }
    return keypoints, descriptors, details


def prepare_references(
    capture: CaptureIndex, browser: BrowserExport, marker_path: Path
) -> tuple[tuple[Reference, ...], list[JsonValue]]:
    raw = cast(JsonValue, json.loads(marker_path.read_text()))
    if not isinstance(raw, dict) or not isinstance(raw.get("frames"), list):
        raise ValueError("reference marker file must contain frames")
    entries = cast(list[JsonValue], raw["frames"])
    by_name = {image.name: image for image in capture.images}
    references: list[Reference] = []
    results: list[JsonValue] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            raise ValueError("reference marker entries require image names")
        name = cast(str, item["image"])
        if name in seen or name not in by_name:
            raise ValueError("reference images must occur once in their index and marker file")
        seen.add(name)
        indexed = by_name[name]
        result: dict[str, JsonValue] = {
            "image": name,
            "image_path": str(indexed.path),
            "source_sequence": indexed.sequence,
            "status": "rejected",
            "marker_entry_sha256": hashlib.sha256(
                json.dumps(item, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "reference_index_sha256": capture.sha256,
            "reference_browser_sha256": browser.sha256,
            "review_evidence": item.get("review_evidence"),
        }
        results.append(result)
        try:
            result["image_sha256"] = digest(indexed.path)
            image, image_hash = read_image(indexed.path)
            result["image_sha256"] = image_hash
            if image.shape[:2] != (450, 800):
                raise ValueError("reference feature ROIs require 800x450 camera images")
            markers = item.get("marker_centres_camera_px")
            if not isinstance(markers, dict):
                raise ValueError("reference lacks marker sampling proposals")
            matrix = association.camera_matrix(browser.frame, markers)
            optical, payload, veto = decode_pair(image, browser.frame, matrix)
            result["optical"] = optical
            if veto or payload is None or len(browser.frames_by_code.get(payload, [])) != 1:
                raise ValueError(
                    "reference lacks an agreeing dual-CRC payload with unique export match"
                )
            frame = browser.frames_by_code[payload][0]
            points, descriptors, roi = reference_features(image, frame, matrix, markers)
            result["roi"] = roi
            if descriptors is None:
                raise ValueError("reference has no eligible SIFT features")
            corners = np.array(
                [project(matrix, position) for position in marker_screen_points(frame)],
                dtype=np.float32,
            )
            result.update(
                {
                    "status": "optically_associated",
                    "optical_code_hex": payload.hex(),
                    "browser_frame_sha256": hashlib.sha256(
                        json.dumps(frame, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest(),
                    "camera_image_dimensions_px": [image.shape[1], image.shape[0]],
                }
            )
            references.append(
                Reference(
                    name, frame, matrix, points, descriptors, abs(cv2.contourArea(corners)), result
                )
            )
        except (ValueError, cv2.error, OSError) as error:
            result["reason"] = str(error)
    return tuple(references), results


def associate_image(
    image: association.Image, browser: BrowserExport, references: tuple[Reference, ...]
) -> dict[str, JsonValue]:
    candidates, marker_status = detect_monitor_markers.detect(image)
    proposals: list[JsonValue] = []
    pair_payloads: set[bytes] = set()
    band_payloads: set[bytes] = set()
    veto = False
    direct_associated = False
    if marker_status == "candidate":
        markers = {
            candidate["identity"]: cast(JsonValue, candidate["centre"]) for candidate in candidates
        }
        try:
            matrix = association.camera_matrix(browser.frame, markers)
            optical, payload, veto = decode_pair(image, browser.frame, matrix)
            proposals.append({"method": "direct exact marker candidates", "optical": optical})
            band_payloads.update(valid_band_payloads(optical))
            if payload is not None:
                pair_payloads.add(payload)
                direct_associated = not veto and len(browser.frames_by_code.get(payload, [])) == 1
        except (ValueError, cv2.error) as error:
            proposals.append({"method": "direct exact marker candidates", "reason": str(error)})
    if not direct_associated:
        grey = cast(association.Image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        current_points, current_descriptors = features(grey)
        for reference in references:
            proposal, proposed_matrix = sift_proposal(
                reference, current_points, current_descriptors
            )
            if proposed_matrix is not None:
                optical, payload, ambiguous = decode_pair(image, reference.frame, proposed_matrix)
                proposal["optical"] = optical
                veto |= ambiguous
                band_payloads.update(valid_band_payloads(optical))
                if payload is not None:
                    pair_payloads.add(payload)
            proposals.append(proposal)
    result: dict[str, JsonValue] = {
        "status": "rejected",
        "direct_marker_status": marker_status,
        "direct_marker_candidate_count": len(candidates),
        "direct_marker_candidate_quality": [
            {
                "identity": candidate["identity"],
                "contrast": candidate["contrast"],
                "minimum_margin": candidate["minimum_margin"],
            }
            for candidate in candidates
        ],
        "direct_marker_candidate_identities": [candidate["identity"] for candidate in candidates],
        "direct_optical_association": direct_associated,
        "proposals": proposals,
        "distinct_dual_band_payloads": len(pair_payloads),
        "distinct_crc_valid_band_payloads": len(band_payloads),
        "optical_ambiguity_veto": veto,
    }
    if veto or len(band_payloads) > 1:
        result["reason"] = "ambiguous or disagreeing CRC-valid optical evidence"
        return result
    if len(pair_payloads) != 1:
        result["reason"] = "no unique agreeing dual-CRC payload"
        return result
    payload = next(iter(pair_payloads))
    matches = browser.frames_by_code.get(payload, [])
    result["browser_match_count"] = len(matches)
    if len(matches) != 1:
        result["reason"] = "decoded payload does not identify one browser frame"
        return result
    selected = matches[0]
    result.update(
        {
            "status": "optically_associated",
            "optical_code_hex": payload.hex(),
            "browser_frame_sha256": hashlib.sha256(
                json.dumps(selected, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "association": {
                key: selected.get(key)
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
    return result


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("browser", type=Path)
    parser.add_argument("--reference-index", type=Path, required=True)
    parser.add_argument("--reference-browser", type=Path, required=True)
    parser.add_argument("--reference-markers", type=Path, required=True)
    parser.add_argument(
        "--reference-report", type=Path, help="optional provenance only; never selects code bits"
    )
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    try:
        capture = read_index(options.index)
        browser = read_browser(options.browser)
        reference_capture = read_index(options.reference_index)
        reference_browser = read_browser(options.reference_browser)
        if browser.layout != reference_browser.layout:
            raise ValueError("target and reference browser layout geometry differ")
        options.output.mkdir(parents=True, exist_ok=False)
        cv2.setNumThreads(1)
        cv2.setRNGSeed(20261027)
        references, reference_results = prepare_references(
            reference_capture, reference_browser, options.reference_markers
        )
        inputs = {
            "source_index": {"path": str(capture.path), "sha256": capture.sha256},
            "source_browser": {"path": str(browser.path), "sha256": browser.sha256},
            "reference_index": {
                "path": str(reference_capture.path),
                "sha256": reference_capture.sha256,
            },
            "reference_browser": {
                "path": str(reference_browser.path),
                "sha256": reference_browser.sha256,
            },
            "reference_markers": {
                "path": str(options.reference_markers.resolve()),
                "sha256": digest(options.reference_markers),
            },
        }
        if options.reference_report is not None:
            inputs["reference_report"] = {
                "path": str(options.reference_report.resolve()),
                "sha256": digest(options.reference_report),
            }
        results: list[JsonValue] = []
        for indexed in capture.images:
            result: dict[str, JsonValue] = {
                "image": indexed.name,
                "image_path": str(indexed.path),
                "source_sequence": indexed.sequence,
                "status": "rejected",
            }
            try:
                result["image_sha256"] = digest(indexed.path)
                image, image_hash = read_image(indexed.path)
                result["image_sha256"] = image_hash
                result.update(associate_image(image, browser, references))
            except (ValueError, cv2.error, OSError) as error:
                result["reason"] = str(error)
            results.append(result)
        associated = [
            cast(dict[str, JsonValue], result)
            for result in results
            if cast(dict[str, JsonValue], result)["status"] == "optically_associated"
        ]
        scenes = Counter(
            cast(str, cast(dict[str, JsonValue], result["association"])["scene_id"])
            for result in associated
        )
        expected_scenes = {
            cast(str, frame["scene_id"])
            for frames_for_code in browser.frames_by_code.values()
            for frame in frames_for_code
        }
        rejection_reasons = Counter(
            cast(str, cast(dict[str, JsonValue], result).get("reason", "unspecified"))
            for result in results
            if cast(dict[str, JsonValue], result)["status"] == "rejected"
        )
        report: dict[str, JsonValue] = {
            "schema_version": 1,
            "scope": (
                "optical identity only; no geometric registration, "
                "projected photograph position or acceptance claim"
            ),
            "proposal_order": (
                "direct exact marker candidates first; on no association, "
                "every verified reference in marker-file order"
            ),
            "uniqueness": (
                "all individually CRC-valid band payloads across evaluated proposals "
                "must agree; both bands must pass and agree under at least one proposal; "
                "one export record required"
            ),
            "inputs": cast(JsonValue, inputs),
            "layout_sha256": hashlib.sha256(browser.layout.encode()).hexdigest(),
            "sampling_parameters": {
                "target_margin_camera_px": TARGET_MARGIN_CAMERA_PX,
                "code_roi_margin_camera_px": CODE_ROI_MARGIN_CAMERA_PX,
                "marker_roi_radius_camera_px": MARKER_ROI_RADIUS_CAMERA_PX,
                "sift_feature_limit": SIFT_FEATURE_LIMIT,
                "sift_contrast_threshold": SIFT_CONTRAST_THRESHOLD,
                "sift_maximum_keypoint_size_px": SIFT_MAXIMUM_KEYPOINT_SIZE_PX,
                "lowe_ratio": LOWE_RATIO,
                "ransac_residual_camera_px": RANSAC_RESIDUAL_CAMERA_PX,
                "minimum_inliers": MINIMUM_INLIERS,
                "minimum_inlier_fraction": MINIMUM_INLIER_FRACTION,
                "minimum_inlier_span_camera_px": [
                    MINIMUM_INLIER_WIDTH_CAMERA_PX,
                    MINIMUM_INLIER_HEIGHT_CAMERA_PX,
                ],
                "quadrilateral_area_ratio": [
                    MINIMUM_QUADRILATERAL_AREA_RATIO,
                    MAXIMUM_QUADRILATERAL_AREA_RATIO,
                ],
                "minimum_cell_camera_px": association.MINIMUM_CELL_CAMERA_PX,
                "minimum_code_contrast": association.MINIMUM_CODE_CONTRAST,
                "minimum_class_margin": association.MINIMUM_CLASS_MARGIN,
                "code_translation_radius_camera_px": association.CODE_SEARCH_RADIUS_CAMERA_PX,
                "code_translation_step_camera_px": association.CODE_SEARCH_STEP_CAMERA_PX,
            },
            "code_sha256": {
                str(path): digest(path)
                for path in (
                    Path(__file__),
                    Path(association.__file__),
                    Path(detect_monitor_markers.__file__),
                    Path(register_monitor_frames.__file__),
                )
            },
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "reference_results": reference_results,
            "reference_proposal_order": [reference.name for reference in references],
            "counts": {
                "images": len(results),
                "optically_associated": len(associated),
                "rejected": len(results) - len(associated),
                "direct_optically_associated": sum(
                    result.get("direct_optical_association") is True for result in associated
                ),
                "scenes_with_optical_identity": dict(scenes),
                "exported_scene_count": len(expected_scenes),
                "exported_scenes_without_optical_identity": cast(
                    JsonValue, sorted(expected_scenes - scenes.keys())
                ),
                "rejection_reasons": dict(rejection_reasons),
            },
            "frames": results,
        }
        path = options.output / "association.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (options.output / "association.sha256").write_text(digest(path) + "  association.json\n")
    except (OSError, ValueError, cv2.error, json.JSONDecodeError) as error:
        parser.exit(2, str(error) + "\n")
    print(path)
    return 0 if len(associated) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
