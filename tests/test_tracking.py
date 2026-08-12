from __future__ import annotations

from app.models import Detection
from app.tracking import TemporalConfirmer, point_in_polygon, select_target


def detection(x: float, y: float, size: float = 10) -> Detection:
    return Detection("pigeon", 0.9, x, y, x + size, y + size)


def test_temporal_confirmation_requires_three_hits_and_releases() -> None:
    confirmer = TemporalConfirmer(window_frames=5, activation_frames=3, release_frames=1)
    assert not confirmer.update(True)
    assert not confirmer.update(False)
    assert not confirmer.update(True)
    assert confirmer.update(True)
    assert confirmer.latched
    assert not confirmer.update(True)
    for _ in range(4):
        assert not confirmer.update(False)
    assert not confirmer.latched
    assert not confirmer.update(True)
    assert not confirmer.update(True)
    assert confirmer.update(True)


def test_selects_largest_cluster_and_area_weighted_centroid() -> None:
    result = select_target(
        [detection(0, 0), detection(12, 0), detection(200, 200, size=30)],
        cluster_distance_px=30,
    )
    assert result is not None
    assert result.member_count == 2
    assert result.pixel_x == 11
    assert result.pixel_y == 5


def test_point_in_polygon() -> None:
    polygon = ((0, 0), (10, 0), (10, 10), (0, 10))
    assert point_in_polygon((5, 5), polygon)
    assert not point_in_polygon((11, 5), polygon)
