from __future__ import annotations

import math
from collections import deque

from .models import Detection, Target


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            crossing_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def select_target(detections: list[Detection], cluster_distance_px: float) -> Target | None:
    if not detections:
        return None
    remaining = set(range(len(detections)))
    clusters: list[list[Detection]] = []
    while remaining:
        seed = remaining.pop()
        cluster_indexes = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            cx, cy = detections[current].centroid
            neighbours = {
                index
                for index in remaining
                if math.dist((cx, cy), detections[index].centroid) <= cluster_distance_px
            }
            remaining -= neighbours
            cluster_indexes |= neighbours
            frontier.extend(neighbours)
        clusters.append([detections[index] for index in cluster_indexes])
    chosen = max(clusters, key=lambda group: (len(group), sum(item.area for item in group)))
    total_area = sum(item.area for item in chosen)
    weights = [max(item.area, 1.0) for item in chosen]
    weight_sum = sum(weights)
    pixel_x = (
        sum(item.centroid[0] * weight for item, weight in zip(chosen, weights, strict=True))
        / weight_sum
    )
    pixel_y = (
        sum(item.centroid[1] * weight for item, weight in zip(chosen, weights, strict=True))
        / weight_sum
    )
    return Target(pixel_x, pixel_y, len(chosen), total_area)


class TemporalConfirmer:
    def __init__(self, window_frames: int, activation_frames: int, release_frames: int) -> None:
        self.window_frames = window_frames
        self.activation_frames = activation_frames
        self.release_frames = release_frames
        self._history: deque[bool] = deque(maxlen=window_frames)
        self._latched = False

    @property
    def hit_count(self) -> int:
        return sum(self._history)

    @property
    def latched(self) -> bool:
        return self._latched

    def update(self, target_present: bool) -> bool:
        """Return true once on the rising confirmation edge."""
        self._history.append(target_present)
        hits = self.hit_count
        if self._latched:
            if hits <= self.release_frames:
                self._latched = False
            return False
        if len(self._history) >= self.activation_frames and hits >= self.activation_frames:
            self._latched = True
            return True
        return False

    def reset(self) -> None:
        self._history.clear()
        self._latched = False
