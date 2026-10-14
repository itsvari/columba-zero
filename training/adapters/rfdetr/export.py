"""Attempt RF-DETR Nano export with fixed-size positional interpolation folded."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Protocol, cast

import torch

from training.adapters.export_graph import arguments, export


class DetectorModel(Protocol):
    model: torch.nn.Module


class Detector(Protocol):
    model: DetectorModel


class Package(Protocol):
    def RFDETRNano(self, *, pretrain_weights: str, device: str) -> Detector: ...


class Exportable(Protocol):
    def export(self) -> None: ...


class PositionEmbedding(Protocol):
    position_embeddings: torch.Tensor
    fixed_position: torch.Tensor

    interpolate_pos_encoding: Interpolator


class PackedQueries(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        boxes, logits, _ = cast(tuple[torch.Tensor, torch.Tensor, None], self.model(image))
        return torch.cat((boxes, logits), dim=-1).squeeze(0)


def fold_positions(model: torch.nn.Module, input_hw: tuple[int, int]) -> None:
    for module in model.modules():
        if not hasattr(module, "interpolate_pos_encoding"):
            continue
        embedding = cast(PositionEmbedding, module)
        with torch.no_grad():
            placeholder = torch.zeros(1, 1, embedding.position_embeddings.shape[-1])
            fixed = embedding.interpolate_pos_encoding(placeholder, *input_hw)
        module.register_buffer("fixed_position", fixed)
        embedding.interpolate_pos_encoding = fixed_interpolator(embedding)


class Interpolator(Protocol):
    def __call__(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor: ...


def fixed_interpolator(embedding: PositionEmbedding) -> Interpolator:
    def interpolate(embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return embedding.fixed_position

    return interpolate


def main() -> None:
    settings = arguments(__doc__ or "RF-DETR export")
    os.environ.setdefault("HF_HOME", str(settings.output.resolve() / "huggingface"))
    # the pinned PyPI distribution is used; repository is retained for licence/provenance.
    package = cast(Package, importlib.import_module("rfdetr"))
    detector = package.RFDETRNano(
        pretrain_weights=str(Path(settings.checkpoint).resolve()), device="cpu"
    )
    model = detector.model.model.eval()
    fold_positions(model, tuple(settings.input_hw))
    cast(Exportable, model).export()
    export(PackedQueries(model), settings.output, tuple(settings.input_hw))


if __name__ == "__main__":
    main()
