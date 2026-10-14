"""The CPU inference settings shared by deployment and export comparison."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import Frame

NMS_INTERSECTION_OVER_UNION = 0.7
COLOUR_CHANNEL_COUNT = 3


class ScalarTensor(Protocol):
    def item(self) -> float: ...


class Coordinates(Protocol):
    def tolist(self) -> list[float]: ...


class PredictionBox(Protocol):
    @property
    def cls(self) -> ScalarTensor: ...
    @property
    def conf(self) -> ScalarTensor: ...
    @property
    def xyxy(self) -> Sequence[Coordinates]: ...


class Prediction(Protocol):
    @property
    def names(self) -> dict[int, str]: ...
    @property
    def boxes(self) -> Sequence[PredictionBox] | None: ...
    @property
    def speed(self) -> dict[str, float | None]: ...


class PredictionModel(Protocol):
    @property
    def names(self) -> dict[int, str]: ...
    def predict(
        self,
        source: Frame | str,
        *,
        imgsz: int | list[int],
        conf: float,
        device: str,
        rect: bool,
        iou: float,
        verbose: bool,
        stream: bool = False,
    ) -> Sequence[Prediction]: ...
