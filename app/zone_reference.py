"""Local, unqualified zone-reference artefacts; no runtime activation authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from .json_data import JsonValue
from .zone_geometry import (
    ConvexImageSupport,
    ImagePoint,
    ReferencePoint,
    ReferencePolygon,
)

SCHEMA_VERSION = 1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
Rotation = Literal[0, 180]


def _text(value: JsonValue) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected non-empty text")
    return value


def _hash(value: JsonValue) -> str:
    result = _text(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError("expected lowercase SHA-256")
    return result


def _integer(value: JsonValue) -> int:
    if type(value) is not int:
        raise ValueError("expected integer")
    return value


def _fields(value: JsonValue, names: set[str]) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("invalid or incomplete reference schema")
    return value


@dataclass(frozen=True, slots=True)
class CameraIdentity:
    camera_id: str
    width_px: int
    height_px: int
    rotation_deg: Rotation
    calibration_sha256: str

    def __post_init__(self) -> None:
        _text(self.camera_id)
        _hash(self.calibration_sha256)
        if min(_integer(self.width_px), _integer(self.height_px)) <= 0:
            raise ValueError("camera dimensions must be positive")
        if _integer(self.rotation_deg) not in (0, 180):
            raise ValueError("unsupported camera rotation")


@dataclass(frozen=True, slots=True)
class UnqualifiedPolicy:
    reason: str

    def __post_init__(self) -> None:
        _text(self.reason)


@dataclass(frozen=True, slots=True)
class ReferenceProvenance:
    reference_id: str
    captured_at: datetime
    source: str

    def __post_init__(self) -> None:
        _text(self.reference_id)
        _text(self.source)
        if self.captured_at.utcoffset() is None:
            raise ValueError("capture timestamp requires timezone")


@dataclass(frozen=True, slots=True)
class ZoneReference:
    camera: CameraIdentity
    image_path: Path
    image_sha256: str
    surface: ReferencePolygon
    zone: ReferencePolygon
    policy: UnqualifiedPolicy
    provenance: ReferenceProvenance

    def __post_init__(self) -> None:
        _hash(self.image_sha256)
        if not self.image_path.is_absolute():
            raise ValueError("local reference image path must be absolute")
        for polygon in (self.surface, self.zone):
            if (polygon.width_px, polygon.height_px) != (
                self.camera.width_px,
                self.camera.height_px,
            ):
                raise ValueError("polygon dimensions differ from camera")
        # convex support makes vertex containment sufficient for the entire zone.
        support = ConvexImageSupport(
            tuple(ImagePoint(point.x_px, point.y_px) for point in self.surface.vertices)
        )
        if not all(
            support.contains(ImagePoint(point.x_px, point.y_px)) for point in self.zone.vertices
        ):
            raise ValueError("zone extends beyond registration surface")


def _validate_image(reference: ZoneReference) -> None:
    data = reference.image_path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("reference image must be PNG")
    if hashlib.sha256(data).hexdigest() != reference.image_sha256:
        raise ValueError("reference image hash mismatch")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[:2] != (reference.camera.height_px, reference.camera.width_px):
        raise ValueError("reference image cannot be decoded at declared dimensions")


def _polygon_payload(polygon: ReferencePolygon) -> list[JsonValue]:
    return [[point.x_px, point.y_px] for point in polygon.vertices]


def _payload(reference: ZoneReference, path: Path) -> dict[str, JsonValue]:
    return {
        "schema_version": SCHEMA_VERSION,
        "camera": {
            "camera_id": reference.camera.camera_id,
            "width_px": reference.camera.width_px,
            "height_px": reference.camera.height_px,
            "rotation_deg": reference.camera.rotation_deg,
            "calibration_sha256": reference.camera.calibration_sha256,
        },
        "image_path": os.path.relpath(reference.image_path, path.parent.resolve()),
        "image_sha256": reference.image_sha256,
        "surface": _polygon_payload(reference.surface),
        "zone": _polygon_payload(reference.zone),
        "policy": {"status": "unqualified", "reason": reference.policy.reason},
        "provenance": {
            "reference_id": reference.provenance.reference_id,
            "captured_at": reference.provenance.captured_at.isoformat(),
            "source": reference.provenance.source,
        },
    }


def save_reference(reference: ZoneReference, path: Path) -> None:
    """Replace metadata atomically after validating the separately held local image."""
    if path.resolve() == reference.image_path.resolve():
        raise ValueError("metadata path must differ from reference image")
    _validate_image(reference)
    serialised = json.dumps(_payload(reference, path), allow_nan=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(serialised)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _polygon(value: JsonValue, camera: CameraIdentity) -> ReferencePolygon:
    if not isinstance(value, list):
        raise ValueError("expected polygon vertices")
    return ReferencePolygon(
        tuple(_point(point) for point in value), camera.width_px, camera.height_px
    )


def _point(value: JsonValue) -> ReferencePoint:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("expected coordinate pair")
    if any(type(coordinate) not in (int, float) for coordinate in value):
        raise ValueError("expected numeric coordinates")
    return ReferencePoint(float(cast(int | float, value[0])), float(cast(int | float, value[1])))


def _unique_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate reference schema field")
        result[name] = value
    return result


def load_reference(path: Path, expected_camera: CameraIdentity) -> ZoneReference:
    payload = _fields(
        cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields),
        ),
        {
            "schema_version",
            "camera",
            "image_path",
            "image_sha256",
            "surface",
            "zone",
            "policy",
            "provenance",
        },
    )
    if _integer(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported reference schema version")
    camera_data = _fields(
        payload["camera"],
        {
            "camera_id",
            "width_px",
            "height_px",
            "rotation_deg",
            "calibration_sha256",
        },
    )
    camera = CameraIdentity(
        _text(camera_data["camera_id"]),
        _integer(camera_data["width_px"]),
        _integer(camera_data["height_px"]),
        cast(Rotation, _integer(camera_data["rotation_deg"])),
        _hash(camera_data["calibration_sha256"]),
    )
    if camera != expected_camera:
        raise ValueError("reference camera identity, orientation or calibration mismatch")
    policy = _fields(payload["policy"], {"status", "reason"})
    if policy["status"] != "unqualified":
        raise ValueError("this schema supports only unqualified reference policy")
    provenance = _fields(payload["provenance"], {"reference_id", "captured_at", "source"})
    reference = ZoneReference(
        camera,
        (path.parent / _text(payload["image_path"])).resolve(),
        _hash(payload["image_sha256"]),
        _polygon(payload["surface"], camera),
        _polygon(payload["zone"], camera),
        UnqualifiedPolicy(_text(policy["reason"])),
        ReferenceProvenance(
            _text(provenance["reference_id"]),
            datetime.fromisoformat(_text(provenance["captured_at"])),
            _text(provenance["source"]),
        ),
    )
    _validate_image(reference)
    return reference
