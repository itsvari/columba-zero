"""Export the upstream COCO NanoDet-Plus-m checkpoint in its raw-logit layout."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Mapping
from typing import Protocol, cast

import torch

from training.adapters.export_graph import Packed, arguments, export


class Configuration(Protocol):
    @property
    def model(self) -> Configuration: ...


class Logger(Protocol):
    def log(self, message: str) -> None: ...


class Utilities(Protocol):
    cfg: Configuration

    def load_config(self, configuration: Configuration, path: str) -> None: ...
    def load_model_weight(
        self,
        model: torch.nn.Module,
        checkpoint: Mapping[str, Mapping[str, torch.Tensor]],
        logger: Logger,
    ) -> None: ...
    def Logger(self, rank: int, directory: str, use_tensorboard: bool) -> Logger: ...


class Architecture(Protocol):
    def build_model(self, configuration: Configuration) -> torch.nn.Module: ...


def main() -> None:
    settings = arguments(__doc__ or "NanoDet export")
    os.environ.setdefault("TORCH_HOME", str(settings.output.resolve() / "torch-cache"))
    sys.path.insert(0, str(settings.repository.resolve()))
    utilities = cast(Utilities, importlib.import_module("nanodet.util"))
    architecture = cast(Architecture, importlib.import_module("nanodet.model.arch"))
    utilities.load_config(utilities.cfg, str(settings.repository / "config/nanodet-plus-m_320.yml"))
    model = architecture.build_model(utilities.cfg.model)
    checkpoint = cast(
        Mapping[str, Mapping[str, torch.Tensor]],
        torch.load(settings.checkpoint, map_location="cpu"),
    )
    logger = utilities.Logger(-1, str(settings.output), False)
    utilities.load_model_weight(model, checkpoint, logger)
    export(Packed(model.eval(), dictionary_output=False), settings.output, tuple(settings.input_hw))


if __name__ == "__main__":
    main()
