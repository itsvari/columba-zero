from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import cv2
import numpy
import pytest

from app.json_data import JsonValue
from tools.associate_monitor_frames import camera_matrix, crc64, decode_code, identity_bytes, main
from tools.monitor_stimulus import build_manifest, manifest_json, render_html
from tools.register_monitor_frames import MARKER_IDS, project

NODE_AVAILABLE = shutil.which("node") is not None


def generated_code(presentation: str, session: str, run: int, render: int) -> bytes:
    html = render_html(build_manifest("tuning", registration_grid=True, optical_code=True))
    source = html.split("function opticalBytes() {", 1)[1].split("\nfunction drawOpticalCode()", 1)[
        0
    ]
    javascript = (
        f"const manifest={{presentation_id:'{presentation}'}};"
        f"const opticalSession='{session}',runNumber={run},renderNumber={render};"
        "function opticalBytes() {"
        + source
        + "\nprocess.stdout.write(Buffer.from(opticalBytes()).toString('hex'));"
    )
    return bytes.fromhex(
        subprocess.run(
            ("node", "-e", javascript), capture_output=True, text=True, check=True
        ).stdout
    )


def generated_browser_records() -> list[dict[str, JsonValue]]:
    manifest = build_manifest("tuning", registration_grid=True, optical_code=True)
    html = render_html(manifest)
    source = html.split('<script>\n"use strict";', 1)[1].split("</script>", 1)[0]
    empty_offset = sum(
        scene.duration_frames
        for scene in manifest.scenes[
            : next(index for index, scene in enumerate(manifest.scenes) if scene.name == "empty")
        ]
    )
    source = source.replace("let sequenceFrame = 0;", f"let sequenceFrame = {empty_offset};")
    harness = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const manifest = input.manifest;
const drawing = new Proxy({}, {get: () => () => {}});
const canvas = {getContext: () => drawing, getBoundingClientRect: () =>
  ({left: 0, top: 0, width: 1600, height: 900})};
const button = {addEventListener: () => {}};
global.document = {getElementById: name => name === 'manifest' ?
  {textContent: JSON.stringify(manifest)} : name === 'display' ? canvas : button,
  addEventListener: () => {}, hidden: false, fullscreenElement: null};
global.window = {devicePixelRatio: 1, innerWidth: 1600, innerHeight: 900,
  addEventListener: () => {}};
global.navigator = {userAgent: 'offline test'};
global.crypto = require('crypto').webcrypto;
global.requestAnimationFrame = () => {};
global.Image = class {
  set src(value) { const asset = manifest.assets.find(item => item.data_url === value);
    this.naturalWidth = asset.photograph.source_width_px;
    this.naturalHeight = asset.photograph.source_height_px; }
  decode() { return Promise.resolve(); }
};
eval(input.source + '\nsetTimeout(() => process.stdout.write(records.join("\\n")), 10);');
"""
    result = subprocess.run(
        ("node", "-e", harness),
        input=json.dumps({"manifest": json.loads(manifest_json(manifest)), "source": source}),
        text=True,
        capture_output=True,
        check=True,
    )
    return [cast(dict[str, JsonValue], json.loads(line)) for line in result.stdout.splitlines()]


def synthetic_capture(
    directory: Path,
    *,
    curved: bool = False,
    occluded: bool = False,
    corrupt: bool = False,
    blur: bool = False,
    mixed: bool = False,
    missing_quadrant: bool = False,
    covered_centre_cross: bool = False,
    camera_scale: float = 0.36,
    generated_record: bool = False,
) -> tuple[Path, Path, Path]:
    manifest = build_manifest("tuning", registration_grid=True, optical_code=True)
    records = generated_browser_records() if generated_record else []
    selected = next((record for record in records if record.get("kind") == "frame"), None)
    presentation = manifest.presentation_id
    session = cast(str, selected["optical_session"]) if selected else "0123456789abcdef"
    run = cast(int, selected["run_number"]) if selected else 2
    render = cast(int, selected["render_number"]) if selected else 317
    code = identity_bytes(presentation, session, run, render)
    design = numpy.full((900, 1600, 3), 18, dtype=numpy.uint8)
    cv2.rectangle(design, (160, 180), (1440, 720), (184, 184, 184), -1)
    points: list[JsonValue] = []
    for cross in manifest.registration_points:
        x, y = round(cross.centre_x_px), round(cross.centre_y_px)
        if curved and cross.point_id == "row-3-column-3":
            y += 12
        in_central_region = (
            abs((cross.centre_x_px - 160) / 1280 - 0.5) <= 0.2
            and abs((cross.centre_y_px - 180) / 540 - 0.5) <= 0.2
        )
        missing = (
            occluded
            and in_central_region
            or covered_centre_cross
            and cross.point_id == "row-3-column-3"
        ) or (missing_quadrant and cross.centre_x_px < 800 and cross.centre_y_px < 450)
        if not missing:
            for colour, thickness in (((255, 255, 255), 7), ((0, 0, 0), 3)):
                cv2.line(design, (x - 14, y), (x + 14, y), colour, thickness)
                cv2.line(design, (x, y - 14), (x, y + 14), colour, thickness)
        points.append(
            {
                "point_id": cross.point_id,
                "screen_css_px": [cross.centre_x_px, cross.centre_y_px],
                "visible": not missing,
            }
        )
    for top in (96, 740):
        band_code = (
            identity_bytes(presentation, session, run, render + 1) if mixed and top == 96 else code
        )
        for bit_index in range(192):
            bit = (band_code[bit_index // 8] >> (7 - bit_index % 8)) & 1
            if corrupt and bit_index == 41:
                bit ^= 1
            colour = (0, 0, 0) if bit else (255, 255, 255)
            x, y = 1056 + bit_index % 32 * 12, top + bit_index // 32 * 12
            cv2.rectangle(design, (x, y), (x + 11, y + 11), colour, -1)
    matrix = numpy.array(
        ((camera_scale, 0.008, 70), (0.004, camera_scale, 40), (0.00001, 0.00001, 1.0))
    )
    image = cv2.warpPerspective(design, matrix, (800, 450), flags=cv2.INTER_AREA)
    if blur:
        image = cv2.GaussianBlur(image, (3, 3), 0.35)
        noise = numpy.random.default_rng(23).normal(0, 1, image.shape)
        image = numpy.clip(image.astype(numpy.float64) + noise, 0, 255).astype(numpy.uint8)
    filename = "frame_000000057.jpg"
    assert cv2.imwrite(str(directory / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    markers: list[JsonValue] = []
    marker_centres: dict[str, JsonValue] = {}
    for marker in manifest.registration_markers:
        bounds = marker.bounds
        markers.append(
            {
                "marker_id": marker.marker_id,
                "box_screen_css_px": {
                    "left_px": bounds.left_px,
                    "top_px": bounds.top_px,
                    "width_px": bounds.width_px,
                    "height_px": bounds.height_px,
                },
            }
        )
        marker_centres[marker.marker_id] = list(
            project(matrix, (bounds.left_px + 32, bounds.top_px + 32))
        )
    frame: dict[str, JsonValue] = {
        "schema_version": 2,
        "kind": "frame",
        "presentation_id": presentation,
        "optical_session": session,
        "run_number": run,
        "render_number": render,
        "scene_id": "synthetic",
        "scene_frame": 5,
        "sequence_frame": 7,
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
        "registration_points": points,
        "targets": [
            {
                "present": True,
                "intended_identity": "bird",
                "box_screen_css_px": {
                    "left_px": 750,
                    "top_px": 400,
                    "width_px": 100,
                    "height_px": 100,
                },
            }
        ],
    }
    index, browser, annotation = (
        directory / "index.jsonl",
        directory / "browser.jsonl",
        directory / "markers.json",
    )
    index.write_text(json.dumps({"filename": filename, "sequence": 340}) + "\n")
    browser.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
        if selected
        else json.dumps(frame) + "\n"
    )
    annotation.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "image": filename,
                        "marker_identities_confirmed": True,
                        "all_markers_fully_visible": True,
                        "marker_centres_camera_px": marker_centres,
                    }
                ]
            }
        )
    )
    return index, browser, annotation


def run_capture(directory: Path, **options: bool) -> tuple[int, dict[str, JsonValue]]:
    index, browser, annotation = synthetic_capture(directory, **options)
    output = directory / "result"
    status = main(
        [str(index), str(browser), str(annotation), "--output", str(output), "--max-error-px", "3"]
    )
    report = cast(dict[str, JsonValue], json.loads((output / "association.json").read_text()))
    return status, cast(dict[str, JsonValue], cast(list[JsonValue], report["frames"])[0])


def test_perspective_blur_noise_and_cli_round_trip(tmp_path: Path) -> None:
    status, frame = run_capture(tmp_path, blur=True)
    assert status == 0, frame
    assert frame["status"] == "validated"
    assert frame["image_sequence"] == 340
    assert cast(dict[str, JsonValue], frame["association"])["render_number"] == 317
    assert len(cast(list[JsonValue], frame["observed_crosses"])) >= 5
    assert len(cast(list[JsonValue], frame["projected_photograph_centres"])) == 1
    assert (tmp_path / "result" / "association.json").is_file()


@pytest.mark.parametrize("case", ["corrupt", "mixed", "curved", "occluded", "missing_quadrant"])
def test_corruption_curvature_and_missing_centre_fail_closed(tmp_path: Path, case: str) -> None:
    status, frame = run_capture(tmp_path, **{case: True})
    assert status == 1
    assert frame["status"] != "validated"
    assert not frame.get("projected_photograph_centres")
    if case == "curved":
        registration = cast(dict[str, JsonValue], frame["registration"])
        assert cast(float, registration["max_validation_error_px"]) > 3


def test_partial_marker_and_duplicate_browser_frame_are_rejected(tmp_path: Path) -> None:
    index, browser, annotation = synthetic_capture(tmp_path)
    data = json.loads(annotation.read_text())
    del data["frames"][0]["marker_centres_camera_px"][MARKER_IDS[0]]
    annotation.write_text(json.dumps(data))
    assert (
        main(
            [
                str(index),
                str(browser),
                str(annotation),
                "--output",
                str(tmp_path / "partial"),
                "--max-error-px",
                "3",
            ]
        )
        == 1
    )
    data = json.loads(annotation.read_text())
    data["frames"][0]["marker_centres_camera_px"][MARKER_IDS[0]] = [85, 55]
    annotation.write_text(json.dumps(data))
    browser.write_text(browser.read_text() * 2)
    assert (
        main(
            [
                str(index),
                str(browser),
                str(annotation),
                "--output",
                str(tmp_path / "duplicate"),
                "--max-error-px",
                "3",
            ]
        )
        == 1
    )


def test_code_at_approximately_real_camera_sampling(tmp_path: Path) -> None:
    _, browser, annotation = synthetic_capture(tmp_path, blur=True, camera_scale=0.20)
    frame = cast(dict[str, JsonValue], json.loads(browser.read_text()))
    markers = cast(
        dict[str, JsonValue],
        json.loads(annotation.read_text())["frames"][0]["marker_centres_camera_px"],
    )
    loaded = cv2.imread(str(tmp_path / "frame_000000057.jpg"))
    assert loaded is not None
    image = cast(numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]], loaded)
    matrix = camera_matrix(frame, markers)
    assert decode_code(image, frame, matrix) == decode_code(image, frame, matrix, upper=True)


def test_photo_can_cover_the_exact_centre_when_other_interior_points_are_observed(
    tmp_path: Path,
) -> None:
    status, frame = run_capture(tmp_path, covered_centre_cross=True)
    assert status == 0
    assert frame["status"] == "validated"


def test_crc64_ecma_check_value() -> None:
    assert crc64(b"123456789") == 0x6C40DF5F0B497347


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node is needed for JavaScript parity")
def test_javascript_code_matches_the_offline_protocol() -> None:
    presentation = build_manifest("tuning", optical_code=True).presentation_id
    session = "0123456789abcdef"
    assert generated_code(presentation, session, 2, 317) == identity_bytes(
        presentation, session, 2, 317
    )


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node is needed for the generated runtime")
def test_cli_accepts_frame_exported_by_generated_runtime(tmp_path: Path) -> None:
    index, browser, annotation = synthetic_capture(tmp_path, generated_record=True, blur=True)
    output = tmp_path / "actual-runtime"
    assert (
        main(
            [
                str(index),
                str(browser),
                str(annotation),
                "--output",
                str(output),
                "--max-error-px",
                "3",
            ]
        )
        == 0
    )
    report = json.loads((output / "association.json").read_text())
    assert report["frames"][0]["status"] == "validated"
    assert report["frames"][0]["association"]["render_number"] == 1


def test_blank_centre_and_projection_outside_hull(tmp_path: Path) -> None:
    index, browser, annotation = synthetic_capture(tmp_path)
    frame = json.loads(browser.read_text())
    frame["targets"][0]["box_screen_css_px"] = {
        "left_px": 165,
        "top_px": 185,
        "width_px": 10,
        "height_px": 10,
    }
    browser.write_text(json.dumps(frame) + "\n")
    output = tmp_path / "outside-hull"
    arguments = [str(index), str(browser), str(annotation), "--max-error-px", "3"]
    assert main([*arguments, "--output", str(output)]) == 0
    report = json.loads((output / "association.json").read_text())
    assert report["frames"][0]["projected_photograph_centres"] == []
    image_path = tmp_path / "frame_000000057.jpg"
    image = cv2.imread(str(image_path))
    assert image is not None
    matrix = numpy.array(((0.36, 0.008, 70), (0.004, 0.36, 40), (0.00001, 0.00001, 1.0)))
    manifest = build_manifest("tuning", registration_grid=True, optical_code=True)
    for point in manifest.registration_points:
        if (
            abs((point.centre_x_px - 160) / 1280 - 0.5) <= 0.2
            and abs((point.centre_y_px - 180) / 540 - 0.5) <= 0.2
        ):
            centre = project(matrix, (point.centre_x_px, point.centre_y_px))
            x, y = round(centre[0]), round(centre[1])
            cv2.rectangle(image, (x - 12, y - 12), (x + 12, y + 12), (184, 184, 184), -1)
    assert cv2.imwrite(str(image_path), image)
    assert main([*arguments, "--output", str(tmp_path / "blank-centre")]) == 1
    report = json.loads((tmp_path / "blank-centre" / "association.json").read_text())
    assert report["frames"][0]["status"] == "unvalidated"
