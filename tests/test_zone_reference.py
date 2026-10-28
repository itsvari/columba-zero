"""Local reference integrity, boundary validation and replacement behaviour."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.json_data import JsonValue
from app.zone_geometry import ReferencePoint, ReferencePolygon
from app.zone_reference import (
    CameraIdentity,
    ReferenceProvenance,
    UnqualifiedPolicy,
    ZoneReference,
    load_reference,
    save_reference,
)


@pytest.fixture
def reference(tmp_path: Path) -> ZoneReference:
    image_path = tmp_path / "surface.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    surface = polygon(((0, 0), (30, 0), (30, 20), (0, 20)))
    zone = polygon(((2, 2), (20, 2), (20, 8), (8, 8), (8, 18), (2, 18)))
    return ZoneReference(
        CameraIdentity("camera-one", 30, 20, 180, "a" * 64),
        image_path,
        hashlib.sha256(image_path.read_bytes()).hexdigest(),
        surface,
        zone,
        UnqualifiedPolicy("no independent planar qualification performed"),
        ReferenceProvenance("reference-one", datetime(2026, 10, 28, tzinfo=UTC), "offline fixture"),
    )


def polygon(vertices: tuple[tuple[float, float], ...]) -> ReferencePolygon:
    return ReferencePolygon(tuple(ReferencePoint(*point) for point in vertices), 30, 20)


def test_save_reload_and_relative_image_path(reference: ZoneReference, tmp_path: Path) -> None:
    folder = tmp_path / "metadata"
    folder.mkdir()
    artifact = folder / "reference.json"
    save_reference(reference, artifact)
    payload = cast(dict[str, JsonValue], json.loads(artifact.read_text()))
    assert payload["image_path"] == "../surface.png"
    policy = payload["policy"]
    assert isinstance(policy, dict)
    assert policy["status"] == "unqualified"
    assert load_reference(artifact, reference.camera) == reference


def test_camera_binding(reference: ZoneReference, tmp_path: Path) -> None:
    artifact = tmp_path / "reference.json"
    save_reference(reference, artifact)
    for camera in (
        replace(reference.camera, camera_id="different"),
        replace(reference.camera, rotation_deg=0),
        replace(reference.camera, width_px=31),
        replace(reference.camera, calibration_sha256="b" * 64),
    ):
        with pytest.raises(ValueError, match="mismatch"):
            load_reference(artifact, camera)


def test_tampered_image(reference: ZoneReference, tmp_path: Path) -> None:
    artifact = tmp_path / "reference.json"
    save_reference(reference, artifact)
    assert cv2.imwrite(str(reference.image_path), np.ones((20, 30, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_reference(artifact, reference.camera)


@pytest.mark.parametrize("image_contents", [b"invalid", b"\x89PNG\r\n\x1a\ncorrupt"])
def test_decode_required(reference: ZoneReference, image_contents: bytes) -> None:
    reference.image_path.write_bytes(image_contents)
    altered = replace(reference, image_sha256=hashlib.sha256(image_contents).hexdigest())
    with pytest.raises(ValueError):
        save_reference(altered, reference.image_path.parent / "reference.json")


def test_decoded_dimensions_required(reference: ZoneReference) -> None:
    assert cv2.imwrite(str(reference.image_path), np.zeros((5, 5, 3), dtype=np.uint8))
    altered = replace(
        reference, image_sha256=hashlib.sha256(reference.image_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="dimensions"):
        save_reference(altered, reference.image_path.parent / "reference.json")


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"unexpected": True},
        {"schema_version": True},
        {"image_sha256": "incorrect"},
        {"policy": {"status": "qualified", "reason": "fit"}},
        {"zone": [[0, 0], [2, 2], [0, 2], [2, 0]]},
        {"zone": [[False, 0], [2, 0], [0, 2]]},
    ],
)
def test_schema_rejection(
    reference: ZoneReference, tmp_path: Path, change: dict[str, JsonValue]
) -> None:
    artifact = tmp_path / "reference.json"
    save_reference(reference, artifact)
    payload = cast(dict[str, JsonValue], json.loads(artifact.read_text()))
    payload.update(change)
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_reference(artifact, reference.camera)


def test_missing_field_rejected(reference: ZoneReference, tmp_path: Path) -> None:
    artifact = tmp_path / "reference.json"
    save_reference(reference, artifact)
    payload = cast(dict[str, JsonValue], json.loads(artifact.read_text()))
    del payload["provenance"]
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_reference(artifact, reference.camera)


def test_zone_containment_and_concave_surface_rejection(reference: ZoneReference) -> None:
    with pytest.raises(ValueError, match="beyond"):
        replace(reference, surface=polygon(((0, 0), (10, 0), (10, 20), (0, 20))))
    with pytest.raises(ValueError, match="convex"):
        replace(reference, surface=reference.zone)
    assert replace(reference, zone=reference.surface).zone == reference.surface


def test_failed_replace_preserves_previous_artifact(
    reference: ZoneReference, tmp_path: Path
) -> None:
    artifact = tmp_path / "reference.json"
    save_reference(reference, artifact)
    original = artifact.read_bytes()
    replacement = replace(reference, policy=UnqualifiedPolicy("still pending independent checks"))
    with patch("app.zone_reference.os.replace", side_effect=OSError("write failed")):
        with pytest.raises(OSError, match="write failed"):
            save_reference(replacement, artifact)
    assert artifact.read_bytes() == original
    assert list(tmp_path.glob(".reference.json.*")) == []


def test_provenance_and_camera_construction(reference: ZoneReference) -> None:
    with pytest.raises(ValueError, match="timezone"):
        replace(reference.provenance, captured_at=datetime(2026, 10, 28))
    with pytest.raises(ValueError):
        replace(reference.camera, calibration_sha256="not a hash")
    with pytest.raises(ValueError):
        replace(reference.camera, width_px=True)


def test_duplicate_schema_fields_rejected(reference: ZoneReference, tmp_path: Path) -> None:
    artifact = tmp_path / "reference.json"
    save_reference(reference, artifact)
    artifact.write_text(
        artifact.read_text().replace(
            '"schema_version": 1', '"schema_version": 2, "schema_version": 1'
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_reference(artifact, reference.camera)


def test_image_cannot_be_overwritten_by_metadata(reference: ZoneReference) -> None:
    original = reference.image_path.read_bytes()
    with pytest.raises(ValueError, match="must differ"):
        save_reference(reference, reference.image_path)
    assert reference.image_path.read_bytes() == original
