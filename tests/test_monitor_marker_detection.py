from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from tools.detect_monitor_markers import PATTERNS, Image, detect


def scene() -> Image:
    image = np.full((400, 650, 3), 30, dtype=np.uint8)
    for (name, pattern), (left, top) in zip(
        PATTERNS.items(), ((50, 40), (550, 40), (550, 300), (50, 300)), strict=True
    ):
        del name
        image[top : top + 64, left : left + 64] = 255
        for row, cells in enumerate(pattern):
            for column, cell in enumerate(cells):
                if cell == "1":
                    x, y = left + 8 + column * 10, top + 8 + row * 10
                    image[y : y + 10, x : x + 10] = 0
    return image


def test_detects_four_markers_through_perspective_blur_and_noise() -> None:
    image = scene()
    source = np.array([[0, 0], [649, 0], [649, 399], [0, 399]], dtype=np.float32)
    destination = np.array([[20, 10], [620, 25], [640, 380], [10, 390]], dtype=np.float32)
    image = cast(
        Image,
        cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (650, 400)),
    )
    image = cast(Image, cv2.GaussianBlur(image, (3, 3), 0.5))
    noise = np.random.default_rng(1).normal(0, 2, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    candidates, status = detect(image)
    assert status == "candidate"
    assert {item["identity"] for item in candidates} == set(PATTERNS)


@pytest.mark.parametrize(
    "damage", ["partial", "wrong_pattern", "rotated", "extra", "extra_unknown", "occluded"]
)
def test_rejects_incomplete_or_ambiguous_scene(damage: str) -> None:
    image = scene()
    if damage == "partial":
        image[40:104, 50:60] = 30
    elif damage == "wrong_pattern":
        image[68:78, 78:88] = 255
    elif damage == "rotated":
        image = cast(Image, cv2.rotate(image, cv2.ROTATE_180))
    elif damage == "occluded":
        image[40:55, 50:114] = 30
    else:
        image[170:234, 290:354] = image[40:104, 50:114]
        if damage == "extra_unknown":
            image[198:208, 318:328] = 255
    assert detect(image)[1] != "candidate"


def test_recorded_frame_372() -> None:
    path = Path(
        "outputs/validation/monitor-20261027/outputs/monitor-axis-calibration/20261027T105439Z-afa4fbc9/demo/frame_000000372.jpg"
    )
    if not path.exists():
        pytest.skip("optional recorded evidence is not present")
    image = cv2.imread(str(path))
    assert image is not None
    candidates, status = detect(cast(Image, image))
    assert status == "candidate"
    expected = {
        "top-left": (152, 118),
        "top-right": (486, 147),
        "bottom-right": (485, 315),
        "bottom-left": (137, 307),
    }
    for item in candidates:
        assert np.linalg.norm(np.array(item["centre"]) - expected[item["identity"]]) < 3
