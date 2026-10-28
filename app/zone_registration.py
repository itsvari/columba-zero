"""Offline planar registration candidates, never physical-error qualification.

All timestamps use one caller-owned monotonic clock. Screening thresholds and
held-out feature residuals provide algorithm diagnostics, not containment bounds.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .zone_geometry import ConvexImageSupport, ImagePoint, ReferencePolygon
from .zone_reference import CameraIdentity

Coefficients = tuple[float, float, float, float, float, float, float, float, float]
ORB_PATCH_SIZE_PX = 31
ORB_LEVEL_COUNT = 4
ORB_BLUR_RADIUS_PX = 3
ORB_RESAMPLING_MARGIN_PX = 4
ORB_SCALE_FACTOR = 1.2
DESCRIPTOR_RADIUS_PX = math.ceil(
    (ORB_PATCH_SIZE_PX / math.sqrt(2) + ORB_BLUR_RADIUS_PX)
    * ORB_SCALE_FACTOR ** (ORB_LEVEL_COUNT - 1)
    + ORB_RESAMPLING_MARGIN_PX
)
HOMOGRAPHY_MINIMUM_POINTS = 4
HELDOUT_STRIDE = 5
RANSAC_CONFIDENCE = 0.995
RANSAC_MAXIMUM_ITERATIONS = 2000


@dataclass(frozen=True, slots=True)
class BirdBox:
    left_px: float
    top_px: float
    right_px: float
    bottom_px: float

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (self.left_px, self.top_px, self.right_px, self.bottom_px)
            )
            or self.left_px >= self.right_px
            or self.top_px >= self.bottom_px
        ):
            raise ValueError("invalid bird box")


@dataclass(frozen=True, slots=True)
class RegistrationFrame:
    """Caller-owned image; nonzero mask pixels and bird boxes are excluded.

    Inputs are read without mutation; only immutable hashes and geometry survive
    estimation. The caller must not mutate frame buffers during the call.
    """

    image: NDArray[np.uint8]
    camera: CameraIdentity
    captured_at_s: float
    colour_order: Literal["RGB", "BGR"]
    bird_boxes: tuple[BirdBox, ...] = ()
    exclusion_mask: NDArray[np.uint8] | None = None


@dataclass(frozen=True, slots=True)
class RegistrationPolicy:
    maximum_features: int
    minimum_features: int
    minimum_matches: int
    minimum_training_inliers: int
    minimum_inlier_fraction: float
    minimum_heldout_matches: int
    match_ratio: float
    ransac_threshold_px: float
    maximum_heldout_residual_px: float
    minimum_reference_coverage: float
    minimum_current_coverage: float
    minimum_projected_area_ratio: float
    maximum_projected_area_ratio: float
    maximum_local_axis_ratio: float
    maximum_age_s: float
    future_tolerance_s: float
    bird_margin_px: int

    def __post_init__(self) -> None:
        counts = (
            self.maximum_features,
            self.minimum_features,
            self.minimum_matches,
            self.minimum_training_inliers,
            self.minimum_heldout_matches,
        )
        if any(type(count) is not int or count < HOMOGRAPHY_MINIMUM_POINTS for count in counts):
            raise ValueError("feature and correspondence counts must be integers at least four")
        if self.maximum_features < self.minimum_features:
            raise ValueError("feature budget below minimum")
        fractions = (
            self.minimum_inlier_fraction,
            self.match_ratio,
            self.minimum_reference_coverage,
            self.minimum_current_coverage,
        )
        if any(not math.isfinite(value) or not 0 < value <= 1 for value in fractions):
            raise ValueError("fractions must lie in (0, 1]")
        positive = (
            self.ransac_threshold_px,
            self.maximum_heldout_residual_px,
            self.minimum_projected_area_ratio,
            self.maximum_projected_area_ratio,
            self.maximum_local_axis_ratio,
            self.maximum_age_s,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("screening bounds must be finite and positive")
        if self.minimum_projected_area_ratio > self.maximum_projected_area_ratio:
            raise ValueError("projected area bounds reversed")
        if not math.isfinite(self.future_tolerance_s) or self.future_tolerance_s < 0:
            raise ValueError("future tolerance must be finite and non-negative")
        if type(self.bird_margin_px) is not int or self.bird_margin_px < 0:
            raise ValueError("bird margin must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RegistrationProvenance:
    reference_sha256: str
    current_sha256: str
    reference_captured_at_s: float
    current_captured_at_s: float
    camera: CameraIdentity


@dataclass(frozen=True, slots=True)
class RegistrationMetrics:
    reference_features: int = 0
    current_features: int = 0
    matches: int = 0
    training_matches: int = 0
    training_inliers: int = 0
    heldout_matches: int = 0
    training_max_residual_px: float | None = None
    heldout_max_residual_px: float | None = None
    reference_coverage: float | None = None
    current_coverage: float | None = None


@dataclass(frozen=True, slots=True)
class RegistrationCandidate:
    coefficients: Coefficients
    support: ConvexImageSupport
    provenance: RegistrationProvenance
    metrics: RegistrationMetrics


@dataclass(frozen=True, slots=True)
class InvalidRegistration:
    reason: Literal["input", "features", "matches", "fit", "quality"]
    detail: str
    provenance: RegistrationProvenance
    metrics: RegistrationMetrics


def _image_hash(frame: RegistrationFrame) -> str:
    digest = hashlib.sha256()
    digest.update(f"{frame.image.dtype}:{frame.image.shape}:{frame.colour_order}".encode())
    digest.update(frame.image.tobytes())
    return digest.hexdigest()


def _validate_inputs(
    reference: RegistrationFrame,
    current: RegistrationFrame,
    surface: ReferencePolygon,
    policy: RegistrationPolicy,
    now_s: float,
) -> None:
    if reference.camera != current.camera:
        raise ValueError("camera identity, calibration or orientation mismatch")
    times = (reference.captured_at_s, current.captured_at_s, now_s)
    if not all(math.isfinite(value) and value >= 0 for value in times):
        raise ValueError("timestamps must be finite non-negative monotonic times")
    if reference.captured_at_s > current.captured_at_s + policy.future_tolerance_s:
        raise ValueError("reference capture follows current capture")
    age_s = now_s - current.captured_at_s
    if age_s > policy.maximum_age_s or age_s < -policy.future_tolerance_s:
        raise ValueError("current frame is stale or from the future")
    if (surface.width_px, surface.height_px) != (
        reference.camera.width_px,
        reference.camera.height_px,
    ):
        raise ValueError("surface dimensions differ from camera")
    ConvexImageSupport(tuple(ImagePoint(point.x_px, point.y_px) for point in surface.vertices))
    for frame in (reference, current):
        if (
            frame.image.dtype != np.uint8
            or frame.image.shape != (frame.camera.height_px, frame.camera.width_px, 3)
            or frame.colour_order not in ("RGB", "BGR")
        ):
            raise ValueError("expected camera-sized RGB/BGR uint8 frame")
        if frame.exclusion_mask is not None and (
            frame.exclusion_mask.dtype != np.uint8
            or frame.exclusion_mask.shape != frame.image.shape[:2]
        ):
            raise ValueError("exclusion mask must be camera-sized uint8")


def _feature_mask(
    frame: RegistrationFrame, policy: RegistrationPolicy, surface: ReferencePolygon | None
) -> NDArray[np.uint8]:
    allowed = np.full(frame.image.shape[:2], 255, dtype=np.uint8)
    if surface is not None:
        allowed.fill(0)
        vertices = np.array([(point.x_px, point.y_px) for point in surface.vertices], np.int32)
        cv2.fillPoly(allowed, [vertices], 255)
        kernel = np.ones((2 * DESCRIPTOR_RADIUS_PX + 1,) * 2, np.uint8)
        allowed = np.asarray(
            cv2.erode(allowed, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0),
            dtype=np.uint8,
        )
    excluded = np.zeros_like(allowed)
    if frame.exclusion_mask is not None:
        excluded[frame.exclusion_mask != 0] = 255
    for box in frame.bird_boxes:
        left_px = max(0, math.floor(box.left_px))
        top_px = max(0, math.floor(box.top_px))
        right_px = min(frame.camera.width_px, math.ceil(box.right_px))
        bottom_px = min(frame.camera.height_px, math.ceil(box.bottom_px))
        excluded[top_px:bottom_px, left_px:right_px] = 255
    radius_px = DESCRIPTOR_RADIUS_PX + policy.bird_margin_px
    excluded = np.asarray(
        cv2.dilate(excluded, np.ones((2 * radius_px + 1,) * 2, np.uint8)), dtype=np.uint8
    )
    allowed[excluded != 0] = 0
    return allowed


def _features(
    frame: RegistrationFrame, mask: NDArray[np.uint8], policy: RegistrationPolicy
) -> tuple[tuple[cv2.KeyPoint, ...], NDArray[np.uint8] | None]:
    conversion = cv2.COLOR_RGB2GRAY if frame.colour_order == "RGB" else cv2.COLOR_BGR2GRAY
    grey = cv2.cvtColor(frame.image, conversion)
    detector = cv2.ORB.create(
        nfeatures=policy.maximum_features,
        nlevels=ORB_LEVEL_COUNT,
        scaleFactor=ORB_SCALE_FACTOR,
        patchSize=ORB_PATCH_SIZE_PX,
    )
    keypoints, descriptors = detector.detectAndCompute(grey, mask)
    return tuple(keypoints), (
        None if descriptors is None else np.asarray(descriptors, dtype=np.uint8)
    )


def _correspondences(
    reference_points: tuple[cv2.KeyPoint, ...],
    current_points: tuple[cv2.KeyPoint, ...],
    reference_descriptors: NDArray[np.uint8],
    current_descriptors: NDArray[np.uint8],
    ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(current_descriptors, reference_descriptors, k=2)
    accepted = sorted(
        (
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ),
        key=lambda match: (match.distance, match.queryIdx, match.trainIdx),
    )
    unique: list[cv2.DMatch] = []
    used_reference: set[int] = set()
    for match in accepted:
        if match.trainIdx not in used_reference:
            unique.append(match)
            used_reference.add(match.trainIdx)
    unique.sort(
        key=lambda match: (current_points[match.queryIdx].pt, reference_points[match.trainIdx].pt)
    )
    return (
        np.array([current_points[match.queryIdx].pt for match in unique], np.float64).reshape(
            -1, 2
        ),
        np.array([reference_points[match.trainIdx].pt for match in unique], np.float64).reshape(
            -1, 2
        ),
    )


def _project(matrix: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] == 0):
        raise ValueError("non-finite projection or projective horizon")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(cv2.convexHull(points.astype(np.float32)), np.float64).reshape(-1, 2)


def _area(points: NDArray[np.float64]) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def _screen_geometry(
    matrix: NDArray[np.float64],
    current_inliers: NDArray[np.float64],
    reference_inliers: NDArray[np.float64],
    surface: ReferencePolygon,
    camera: CameraIdentity,
    policy: RegistrationPolicy,
) -> tuple[ConvexImageSupport, float, float]:
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
        raise ValueError("singular or non-finite fit")
    current_hull = _hull(current_inliers)
    support = ConvexImageSupport(
        tuple(ImagePoint(float(point[0]), float(point[1])) for point in current_hull)
    )
    denominators = np.column_stack((current_hull, np.ones(len(current_hull)))) @ matrix[2]
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError("projective horizon crosses training-inlier support")
    surface_points = np.array([(point.x_px, point.y_px) for point in surface.vertices], np.float64)
    reference_coverage = _area(_hull(reference_inliers)) / _area(surface_points)
    current_area_px2 = _area(current_hull)
    current_coverage = current_area_px2 / (camera.width_px * camera.height_px)
    projected = _project(matrix, current_hull)
    area_ratio = _area(projected) / current_area_px2
    if reference_coverage < policy.minimum_reference_coverage or (
        current_coverage < policy.minimum_current_coverage
    ):
        raise ValueError("insufficient spatial support")
    if not policy.minimum_projected_area_ratio <= area_ratio <= policy.maximum_projected_area_ratio:
        raise ValueError("implausible projected area")
    if np.linalg.det(matrix) / float(denominators[0]) ** 3 <= 0:
        raise ValueError("orientation-reversing planar fit")
    for mapped_point, denominator in zip(projected, denominators, strict=True):
        jacobian = (matrix[:2, :2] - np.outer(mapped_point, matrix[2, :2])) / denominator
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if (
            not np.isfinite(singular_values).all()
            or singular_values[1] <= 0
            or (singular_values[0] / singular_values[1] > policy.maximum_local_axis_ratio)
        ):
            raise ValueError("implausible local anisotropy")
        local_area_ratio = float(singular_values[0] * singular_values[1])
        if (
            not policy.minimum_projected_area_ratio
            <= local_area_ratio
            <= policy.maximum_projected_area_ratio
        ):
            raise ValueError("implausible local projected area")
    return support, reference_coverage, current_coverage


def _fit(
    current_points: NDArray[np.float64],
    reference_points: NDArray[np.float64],
    surface: ReferencePolygon,
    current: RegistrationFrame,
    policy: RegistrationPolicy,
    provenance: RegistrationProvenance,
    metrics: RegistrationMetrics,
) -> RegistrationCandidate | InvalidRegistration:
    heldout = np.arange(len(current_points)) % HELDOUT_STRIDE == 0
    metrics = replace(
        metrics, training_matches=int(np.sum(~heldout)), heldout_matches=int(np.sum(heldout))
    )
    if metrics.heldout_matches < policy.minimum_heldout_matches or (
        metrics.training_matches < HOMOGRAPHY_MINIMUM_POINTS
    ):
        return InvalidRegistration(
            "matches", "too few independent held-out matches", provenance, metrics
        )
    matrix, inlier_mask = cv2.findHomography(
        current_points[~heldout],
        reference_points[~heldout],
        cv2.RANSAC,
        policy.ransac_threshold_px,
        maxIters=RANSAC_MAXIMUM_ITERATIONS,
        confidence=RANSAC_CONFIDENCE,
    )
    if matrix is None or inlier_mask is None:
        return InvalidRegistration("fit", "RANSAC could not fit a homography", provenance, metrics)
    inliers = inlier_mask.ravel() != 0
    metrics = replace(metrics, training_inliers=int(np.sum(inliers)))
    if metrics.training_inliers < policy.minimum_training_inliers or (
        metrics.training_inliers / metrics.training_matches < policy.minimum_inlier_fraction
    ):
        return InvalidRegistration("quality", "insufficient training inliers", provenance, metrics)
    try:
        training_current = current_points[~heldout][inliers]
        training_reference = reference_points[~heldout][inliers]
        training_residuals = np.linalg.norm(
            _project(matrix, training_current) - training_reference, axis=1
        )
        heldout_residuals = np.linalg.norm(
            _project(matrix, current_points[heldout]) - reference_points[heldout], axis=1
        )
        metrics = replace(
            metrics,
            training_max_residual_px=float(np.max(training_residuals)),
            heldout_max_residual_px=float(np.max(heldout_residuals)),
        )
        if not np.isfinite(heldout_residuals).all() or (
            float(np.max(heldout_residuals)) > policy.maximum_heldout_residual_px
        ):
            raise ValueError("held-out residual exceeds screening threshold")
        support, reference_coverage, current_coverage = _screen_geometry(
            matrix, training_current, training_reference, surface, current.camera, policy
        )
        metrics = replace(
            metrics, reference_coverage=reference_coverage, current_coverage=current_coverage
        )
    except ValueError as error:
        return InvalidRegistration("quality", str(error), provenance, metrics)
    values = tuple(float(value) for value in matrix.ravel())
    coefficients: Coefficients = (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
    )
    return RegistrationCandidate(coefficients, support, provenance, metrics)


def estimate_registration(
    reference: RegistrationFrame,
    current: RegistrationFrame,
    surface: ReferencePolygon,
    policy: RegistrationPolicy,
    now_s: float,
) -> RegistrationCandidate | InvalidRegistration:
    """Estimate afresh on a convex surface, with no eligibility or error authority.

    The returned support is the convex hull of training inliers only; held-out
    correspondences screen the fit without expanding its supported domain.
    """
    provenance = RegistrationProvenance(
        _image_hash(reference),
        _image_hash(current),
        reference.captured_at_s,
        current.captured_at_s,
        current.camera,
    )
    metrics = RegistrationMetrics()
    try:
        _validate_inputs(reference, current, surface, policy, now_s)
    except ValueError as error:
        return InvalidRegistration("input", str(error), provenance, metrics)
    reference_keypoints, reference_descriptors = _features(
        reference, _feature_mask(reference, policy, surface), policy
    )
    current_keypoints, current_descriptors = _features(
        current, _feature_mask(current, policy, None), policy
    )
    metrics = replace(
        metrics,
        reference_features=len(reference_keypoints),
        current_features=len(current_keypoints),
    )
    if min(metrics.reference_features, metrics.current_features) < policy.minimum_features or (
        reference_descriptors is None or current_descriptors is None
    ):
        return InvalidRegistration("features", "too few unmasked features", provenance, metrics)
    current_points, reference_points = _correspondences(
        reference_keypoints,
        current_keypoints,
        reference_descriptors,
        current_descriptors,
        policy.match_ratio,
    )
    metrics = replace(metrics, matches=len(current_points))
    if metrics.matches < policy.minimum_matches:
        return InvalidRegistration(
            "matches", "too few unique descriptor matches", provenance, metrics
        )
    return _fit(current_points, reference_points, surface, current, policy, provenance, metrics)
