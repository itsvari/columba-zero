"""Geometric examples independent of runtime registration and target selection."""

import math

import pytest

from app.zone_geometry import (
    ConvexImageSupport,
    ImagePoint,
    Membership,
    ProjectiveMapping,
    ReferencePoint,
    ReferencePolygon,
)


def support() -> ConvexImageSupport:
    return ConvexImageSupport(
        tuple(ImagePoint(*point) for point in ((0, 0), (10, 0), (10, 10), (0, 10)))
    )


def test_concave_polygon_and_boundary_uncertainty() -> None:
    polygon = ReferencePolygon(
        tuple(
            ReferencePoint(*point) for point in ((0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10))
        ),
        10,
        10,
    )
    assert polygon.classify(ReferencePoint(2, 8), 1) is Membership.INSIDE
    assert polygon.classify(ReferencePoint(8, 8), 1) is Membership.OUTSIDE
    assert polygon.classify(ReferencePoint(4, 8)) is Membership.UNCERTAIN
    assert polygon.classify(ReferencePoint(3, 8), 1) is Membership.UNCERTAIN
    assert polygon.classify(ReferencePoint(5, 8), 1) is Membership.UNCERTAIN
    assert polygon.classify(ReferencePoint(4, 4)) is Membership.UNCERTAIN


@pytest.mark.parametrize(
    "vertices",
    [
        ((0, 0), (1, 1)),
        ((0, 0), (1, 1), (2, 2)),
        ((0, 0), (4, 4), (0, 4), (4, 0)),
        ((0, 0), (4, 0), (4, 4), (0, 0)),
        ((0, 0), (11, 0), (0, 4)),
    ],
)
def test_invalid_polygons(vertices: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises(ValueError):
        ReferencePolygon(tuple(ReferencePoint(*point) for point in vertices), 10, 10)


def test_nonfinite_points_and_negative_bounds() -> None:
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            ImagePoint(invalid, 0)
        with pytest.raises(ValueError):
            ReferencePoint(0, invalid)
    polygon = ReferencePolygon(
        (ReferencePoint(0, 0), ReferencePoint(4, 0), ReferencePoint(0, 4)), 10, 10
    )
    with pytest.raises(ValueError):
        polygon.classify(ReferencePoint(1, 1), -1)
    with pytest.raises(ValueError):
        ReferencePolygon(polygon.vertices, 0, 10)


def test_affine_transform_propagates_rectangle_and_declared_error() -> None:
    mapping = ProjectiveMapping((2, 0, 3, 0, 3, -1, 0, 0, 1), support(), 0.5)
    result = mapping.map_point(ImagePoint(4, 5), 1, 2)
    assert result.point.x_px == pytest.approx(11)
    assert result.point.y_px == pytest.approx(14)
    assert result.uncertainty_radius_px == pytest.approx(math.sqrt(40) + 0.5)


def test_projective_transform_and_whole_rectangle_support() -> None:
    mapping = ProjectiveMapping((1, 0, 0, 0, 1, 0, 0.1, 0, 1), support(), 0)
    result = mapping.map_point(ImagePoint(5, 3), 1, 1)
    assert result.point.x_px == pytest.approx(10 / 3)
    assert result.point.y_px == pytest.approx(2)
    expected_radius_px = math.hypot(4 / 1.4 - 10 / 3, 4 / 1.4 - 2)
    assert result.uncertainty_radius_px == pytest.approx(expected_radius_px)
    with pytest.raises(ValueError):
        mapping.map_point(ImagePoint(0.5, 5), 1, 0)
    with pytest.raises(ValueError):
        mapping.map_point(ImagePoint(11, 5))
    with pytest.raises(ValueError):
        mapping.map_point(ImagePoint(5, 5), -1, 0)


@pytest.mark.parametrize(
    "coefficients",
    [
        (0,) * 9,
        (1, 0, 0, 0, 0, 0, 0, 0, 1),
        (1, 0, 0, 0, 1, 0, 1, 0, -5),
        (1, 0, 0, 0, 1, 0, 1, 0, 0),
        (1, 0, 0, 0, 1, 0, 0, 0, math.nan),
        (1, 0),
    ],
)
def test_invalid_mapping(coefficients: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        ProjectiveMapping(coefficients, support(), 0)


def test_concave_mapping_support_rejected() -> None:
    with pytest.raises(ValueError):
        ConvexImageSupport(
            tuple(ImagePoint(*point) for point in ((0, 0), (10, 0), (4, 4), (10, 10), (0, 10)))
        )


def test_clockwise_polygon_and_negative_homography_scale() -> None:
    polygon = ReferencePolygon(
        (ReferencePoint(0, 4), ReferencePoint(4, 4), ReferencePoint(4, 0), ReferencePoint(0, 0)),
        4,
        4,
    )
    assert polygon.classify(ReferencePoint(2, 2)) is Membership.INSIDE
    mapping = ProjectiveMapping((-1, 0, 0, 0, -1, 0, 0, 0, -1), support(), 0)
    assert mapping.map_point(ImagePoint(2, 2)).point == ReferencePoint(2, 2)
