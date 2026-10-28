"""Offline planar geometry; declared error bounds are not registration fit residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class _Point(Protocol):
    @property
    def x_px(self) -> float: ...

    @property
    def y_px(self) -> float: ...


def _finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coordinates and bounds must be finite")


@dataclass(frozen=True, slots=True)
class ImagePoint:
    x_px: float
    y_px: float

    def __post_init__(self) -> None:
        _finite(self.x_px, self.y_px)


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    x_px: float
    y_px: float

    def __post_init__(self) -> None:
        _finite(self.x_px, self.y_px)


class Membership(Enum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    UNCERTAIN = "uncertain"


def _cross(first: _Point, second: _Point, third: _Point) -> float:
    return (second.x_px - first.x_px) * (third.y_px - first.y_px) - (second.y_px - first.y_px) * (
        third.x_px - first.x_px
    )


def _distance_to_edge(point: _Point, first: _Point, second: _Point) -> float:
    horizontal = second.x_px - first.x_px
    vertical = second.y_px - first.y_px
    fraction = max(
        0.0,
        min(
            1.0,
            ((point.x_px - first.x_px) * horizontal + (point.y_px - first.y_px) * vertical)
            / (horizontal * horizontal + vertical * vertical),
        ),
    )
    return math.hypot(
        point.x_px - first.x_px - fraction * horizontal,
        point.y_px - first.y_px - fraction * vertical,
    )


def _intersects(first: _Point, second: _Point, third: _Point, fourth: _Point) -> bool:
    first_side, second_side = _cross(first, second, third), _cross(first, second, fourth)
    third_side, fourth_side = _cross(third, fourth, first), _cross(third, fourth, second)
    if first_side * second_side < 0 and third_side * fourth_side < 0:
        return True
    return any(
        side == 0 and _distance_to_edge(point, start, end) == 0
        for side, point, start, end in (
            (first_side, third, first, second),
            (second_side, fourth, first, second),
            (third_side, first, third, fourth),
            (fourth_side, second, third, fourth),
        )
    )


def _validate_polygon(vertices: tuple[_Point, ...]) -> None:
    if len(vertices) < 3 or len({(point.x_px, point.y_px) for point in vertices}) != len(vertices):
        raise ValueError("polygon requires at least three distinct vertices")
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        if _cross(vertices[index - 1], first, second) == 0:
            raise ValueError("polygon has a degenerate corner")
        for other_index in range(index + 2, len(vertices)):
            if index == 0 and other_index == len(vertices) - 1:
                continue
            if _intersects(
                first, second, vertices[other_index], vertices[(other_index + 1) % len(vertices)]
            ):
                raise ValueError("polygon intersects itself")


def _contains(vertices: tuple[_Point, ...], point: _Point) -> bool:
    inside = False
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        if (first.y_px > point.y_px) == (second.y_px > point.y_px):
            continue
        crossing = (second.x_px - first.x_px) * (point.y_px - first.y_px) / (
            second.y_px - first.y_px
        ) + first.x_px
        if point.x_px < crossing:
            inside = not inside
    return inside


@dataclass(frozen=True, slots=True)
class ReferencePolygon:
    vertices: tuple[ReferencePoint, ...]
    width_px: float
    height_px: float

    def __post_init__(self) -> None:
        _finite(self.width_px, self.height_px)
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("reference dimensions must be positive")
        _validate_polygon(self.vertices)
        if any(
            not (0 <= point.x_px <= self.width_px and 0 <= point.y_px <= self.height_px)
            for point in self.vertices
        ):
            raise ValueError("polygon exceeds reference bounds")

    def classify(self, point: ReferencePoint, uncertainty_radius_px: float = 0) -> Membership:
        _finite(uncertainty_radius_px)
        if uncertainty_radius_px < 0:
            raise ValueError("uncertainty must be non-negative")
        for index, first in enumerate(self.vertices):
            if _distance_to_edge(point, first, self.vertices[(index + 1) % len(self.vertices)]) <= (
                uncertainty_radius_px
            ):
                return Membership.UNCERTAIN
        return Membership.INSIDE if _contains(self.vertices, point) else Membership.OUTSIDE


@dataclass(frozen=True, slots=True)
class ConvexImageSupport:
    vertices: tuple[ImagePoint, ...]

    def __post_init__(self) -> None:
        _validate_polygon(self.vertices)
        turns = tuple(
            _cross(self.vertices[index - 1], point, self.vertices[(index + 1) % len(self.vertices)])
            for index, point in enumerate(self.vertices)
        )
        if not (all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)):
            raise ValueError("mapping support must be strictly convex")

    def contains(self, point: ImagePoint) -> bool:
        sides = tuple(
            _cross(first, self.vertices[(index + 1) % len(self.vertices)], point)
            for index, first in enumerate(self.vertices)
        )
        return all(side >= 0 for side in sides) or all(side <= 0 for side in sides)


@dataclass(frozen=True, slots=True)
class MappedPoint:
    point: ReferencePoint
    uncertainty_radius_px: float


@dataclass(frozen=True, slots=True)
class ProjectiveMapping:
    """A caller-qualified mapping on a convex domain.

    The radius combines geometric propagation of an axis-aligned image error box
    with a caller-declared reference-space bound. It is not statistical confidence
    and cannot be inferred from a homography's fitting residuals. A horizon-free
    projectivity maps the box to a convex quadrilateral, whose furthest point from
    the mapped centre lies at a vertex.
    """

    coefficients: tuple[float, ...]
    support: ConvexImageSupport
    declared_reference_error_px: float

    def __post_init__(self) -> None:
        if len(self.coefficients) != 9:
            raise ValueError("projective mapping requires nine coefficients")
        _finite(*self.coefficients, self.declared_reference_error_px)
        if self.declared_reference_error_px < 0:
            raise ValueError("declared error must be non-negative")
        scale = max(abs(value) for value in self.coefficients)
        if scale == 0:
            raise ValueError("singular mapping")
        normalised = tuple(value / scale for value in self.coefficients)
        object.__setattr__(self, "coefficients", normalised)
        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth = normalised
        determinant = (
            first * (fifth * ninth - sixth * eighth)
            - second * (fourth * ninth - sixth * seventh)
            + third * (fourth * eighth - fifth * seventh)
        )
        _finite(determinant)
        if determinant == 0:
            raise ValueError("singular mapping")
        denominators = tuple(self._denominator(point) for point in self.support.vertices)
        _finite(*denominators)
        if not (
            all(value > 0 for value in denominators) or all(value < 0 for value in denominators)
        ):
            raise ValueError("projective horizon crosses mapping support")

    def _denominator(self, point: ImagePoint) -> float:
        return (
            self.coefficients[6] * point.x_px
            + self.coefficients[7] * point.y_px
            + self.coefficients[8]
        )

    def _transform(self, point: ImagePoint) -> ReferencePoint:
        denominator = self._denominator(point)
        return ReferencePoint(
            (
                self.coefficients[0] * point.x_px
                + self.coefficients[1] * point.y_px
                + self.coefficients[2]
            )
            / denominator,
            (
                self.coefficients[3] * point.x_px
                + self.coefficients[4] * point.y_px
                + self.coefficients[5]
            )
            / denominator,
        )

    def map_point(
        self, point: ImagePoint, uncertainty_x_px: float = 0, uncertainty_y_px: float = 0
    ) -> MappedPoint:
        _finite(uncertainty_x_px, uncertainty_y_px)
        if min(uncertainty_x_px, uncertainty_y_px) < 0:
            raise ValueError("uncertainty must be non-negative")
        corners = tuple(
            ImagePoint(point.x_px + horizontal, point.y_px + vertical)
            for horizontal in (-uncertainty_x_px, uncertainty_x_px)
            for vertical in (-uncertainty_y_px, uncertainty_y_px)
        )
        if not all(self.support.contains(corner) for corner in corners):
            raise ValueError("point uncertainty exceeds declared mapping support")
        centre = self._transform(point)
        mapped_corners = tuple(self._transform(corner) for corner in corners)
        radius_px = (
            max(
                math.hypot(corner.x_px - centre.x_px, corner.y_px - centre.y_px)
                for corner in mapped_corners
            )
            + self.declared_reference_error_px
        )
        _finite(radius_px)
        return MappedPoint(centre, radius_px)
