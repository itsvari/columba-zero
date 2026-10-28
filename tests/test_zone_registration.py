"""Actual feature registration on a synthetic plane and adversarial screening."""

from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from app.zone_geometry import ImagePoint, ReferencePoint, ReferencePolygon
from app.zone_reference import CameraIdentity
from app.zone_registration import (
    BirdBox,
    InvalidRegistration,
    RegistrationCandidate,
    RegistrationFrame,
    RegistrationPolicy,
    estimate_registration,
)

WIDTH_PX = 640
HEIGHT_PX = 480


def policy() -> RegistrationPolicy:
    return RegistrationPolicy(
        maximum_features=3000,
        minimum_features=40,
        minimum_matches=40,
        minimum_training_inliers=25,
        minimum_inlier_fraction=0.7,
        minimum_heldout_matches=8,
        match_ratio=0.7,
        ransac_threshold_px=2,
        maximum_heldout_residual_px=3,
        minimum_reference_coverage=0.25,
        minimum_current_coverage=0.25,
        minimum_projected_area_ratio=0.5,
        maximum_projected_area_ratio=2,
        maximum_local_axis_ratio=2,
        maximum_age_s=0.25,
        future_tolerance_s=0.001,
        bird_margin_px=5,
    )


def surface() -> ReferencePolygon:
    return ReferencePolygon(
        tuple(
            ReferencePoint(*point)
            for point in ((0, 0), (WIDTH_PX, 0), (WIDTH_PX, HEIGHT_PX), (0, HEIGHT_PX))
        ),
        WIDTH_PX,
        HEIGHT_PX,
    )


def frame(image: NDArray[np.uint8], captured_at_s: float = 10) -> RegistrationFrame:
    return RegistrationFrame(
        image, CameraIdentity("camera", WIDTH_PX, HEIGHT_PX, 0, "a" * 64), captured_at_s, "BGR"
    )


def texture() -> NDArray[np.uint8]:
    generator = np.random.default_rng(7)
    grey = generator.integers(0, 256, (HEIGHT_PX, WIDTH_PX), dtype=np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def test_known_projectivity_with_bird_masks_and_distractors() -> None:
    reference_image = texture()
    forward = np.array(((1, 0.01, 9), (-0.01, 1, 7), (0.00002, -0.00001, 1)), np.float64)
    current_image = np.asarray(
        cv2.warpPerspective(reference_image, forward, (WIDTH_PX, HEIGHT_PX)), dtype=np.uint8
    )
    reference_image[180:250, 250:330] = 255
    current_image[100:180, 400:480] = 0
    reference = replace(frame(reference_image, 1), bird_boxes=(BirdBox(250, 180, 330, 250),))
    current = replace(frame(current_image), bird_boxes=(BirdBox(400, 100, 480, 180),))
    reference_before, current_before = reference_image.copy(), current_image.copy()
    result = estimate_registration(reference, current, surface(), policy(), 10.1)
    assert isinstance(result, RegistrationCandidate), result
    matrix = np.array(result.coefficients).reshape(3, 3)
    expected = np.linalg.inv(forward)
    for point in ((100, 100), (300, 300), (500, 350)):
        homogeneous = np.array((*point, 1))
        actual_point, expected_point = matrix @ homogeneous, expected @ homogeneous
        assert (
            np.linalg.norm(
                actual_point[:2] / actual_point[2] - expected_point[:2] / expected_point[2]
            )
            < 1
        )
    assert result.metrics.heldout_matches >= 8
    assert result.metrics.training_inliers >= 25
    assert result.support.contains(ImagePoint(300, 300))
    assert np.array_equal(reference_image, reference_before)
    assert np.array_equal(current_image, current_before)


def test_blank_and_unrelated_views_rejected() -> None:
    blank = frame(np.zeros((HEIGHT_PX, WIDTH_PX, 3), np.uint8))
    result = estimate_registration(blank, blank, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration) and result.reason == "features"
    unrelated = np.random.default_rng(99).integers(0, 256, (HEIGHT_PX, WIDTH_PX, 3), dtype=np.uint8)
    result = estimate_registration(frame(texture()), frame(unrelated), surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration)


@pytest.mark.parametrize("view", ["reference", "current"])
def test_all_features_masked_in_either_view(view: str) -> None:
    reference, current = frame(texture()), frame(texture())
    masked = replace(reference, exclusion_mask=np.full((HEIGHT_PX, WIDTH_PX), 255, np.uint8))
    if view == "reference":
        reference = masked
    else:
        current = masked
    result = estimate_registration(reference, current, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration) and result.reason == "features"


@pytest.mark.parametrize("time_s", [9.9, 10.251, float("nan")])
def test_time_rejection(time_s: float) -> None:
    image = frame(texture())
    result = estimate_registration(image, image, surface(), policy(), time_s)
    assert isinstance(result, InvalidRegistration) and result.reason == "input"


def test_mismatched_camera_and_mask_shape_rejected() -> None:
    reference = frame(texture())
    current = replace(reference, camera=replace(reference.camera, rotation_deg=180))
    result = estimate_registration(reference, current, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration) and result.reason == "input"
    current = replace(reference, exclusion_mask=np.zeros((2, 2), np.uint8))
    result = estimate_registration(reference, current, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration) and result.reason == "input"


def grid() -> NDArray[np.float64]:
    return np.array(
        [
            (horizontal, vertical)
            for horizontal in range(80, 600, 60)
            for vertical in range(80, 440, 60)
        ],
        np.float64,
    )


def test_corrupt_heldout_cannot_change_training_fit_or_pass() -> None:
    points = grid()
    corrupted = points.copy()
    corrupted[::5] += (15, -12)
    image = frame(texture())
    with patch("app.zone_registration._correspondences", return_value=(points, corrupted)):
        result = estimate_registration(image, image, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration) and result.reason == "quality"
    assert "held-out" in result.detail
    assert result.metrics.training_max_residual_px is not None
    assert result.metrics.training_max_residual_px < 0.001
    assert result.metrics.heldout_max_residual_px is not None
    assert result.metrics.heldout_max_residual_px > 15


def test_insufficient_spatial_support_rejected() -> None:
    points = grid() * 0.1 + 150
    image = frame(texture())
    with patch("app.zone_registration._correspondences", return_value=(points, points)):
        result = estimate_registration(image, image, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration)
    assert "spatial support" in result.detail


def test_horizon_crossing_candidate_rejected() -> None:
    points = grid()
    matrix = np.array(((1, 0, 0), (0, 1, 0), (0.004, 0, -1)), np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    projected = homogeneous[:, :2] / homogeneous[:, 2:]
    training_count = len(points) - len(points[::5])
    image = frame(texture())
    with (
        patch("app.zone_registration._correspondences", return_value=(points, projected)),
        patch(
            "app.zone_registration.cv2.findHomography",
            return_value=(matrix, np.ones((training_count, 1), np.uint8)),
        ),
    ):
        result = estimate_registration(image, image, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration)
    assert "horizon" in result.detail


@pytest.mark.parametrize("view", ["reference", "current"])
def test_masked_texture_cannot_supply_registration(view: str) -> None:
    image = np.zeros((HEIGHT_PX, WIDTH_PX, 3), np.uint8)
    image[160:300, 240:400] = texture()[160:300, 240:400]
    reference = replace(frame(image), bird_boxes=(BirdBox(240, 160, 400, 300),))
    current = frame(image)
    if view == "current":
        reference, current = current, reference
    result = estimate_registration(reference, current, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration) and result.reason == "features"


def test_implausible_axis_squash_rejected() -> None:
    points = grid()
    transformed = points * (4, 0.25)
    image = frame(texture())
    with patch("app.zone_registration._correspondences", return_value=(points, transformed)):
        result = estimate_registration(image, image, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration)
    assert "anisotropy" in result.detail


def test_few_training_inliers_rejected() -> None:
    points = grid()
    incorrect = np.random.default_rng(8).uniform((0, 0), (WIDTH_PX, HEIGHT_PX), points.shape)
    incorrect[:12] = points[:12]
    image = frame(texture())
    with patch("app.zone_registration._correspondences", return_value=(points, incorrect)):
        result = estimate_registration(image, image, surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration)
    assert result.reason in ("fit", "quality")


def test_texture_confined_to_small_unmasked_support_rejected() -> None:
    image = texture()
    excluded = np.full((HEIGHT_PX, WIDTH_PX), 255, np.uint8)
    excluded[100:350, 150:450] = 0
    reference = replace(frame(image), exclusion_mask=excluded)
    result = estimate_registration(reference, frame(image), surface(), policy(), 10.1)
    assert isinstance(result, InvalidRegistration)
    assert result.reason in ("features", "matches", "quality")


def test_rgb_conversion_and_capture_provenance() -> None:
    reference = frame(texture(), 1)
    current = replace(frame(reference.image.copy()), colour_order="RGB")
    result = estimate_registration(reference, current, surface(), policy(), 10.1)
    assert isinstance(result, RegistrationCandidate), result
    assert result.provenance.reference_captured_at_s == 1
    assert result.provenance.current_captured_at_s == 10
    assert result.provenance.reference_sha256 != result.provenance.current_sha256


def test_invalid_screening_policy_rejected() -> None:
    with pytest.raises(ValueError):
        replace(policy(), bird_margin_px=-1)
    with pytest.raises(ValueError):
        replace(policy(), minimum_inlier_fraction=float("nan"))
    with pytest.raises(ValueError):
        replace(policy(), maximum_age_s=0)
