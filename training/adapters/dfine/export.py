"""Export COCO D-FINE-N with regenerated rectangular positional/anchor caches."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Mapping
from typing import Protocol, cast

import torch

from app.json_data import JsonValue
from training.adapters.export_graph import Packed, arguments, export

CACHE_KEYS = frozenset({"decoder.anchors", "decoder.valid_mask"})


class Configuration(Protocol):
    yaml_cfg: dict[str, JsonValue]

    @property
    def model(self) -> torch.nn.Module: ...


class Core(Protocol):
    def YAMLConfig(self, path: str) -> Configuration: ...


class Deployable(Protocol):
    def deploy(self) -> torch.nn.Module: ...


def main() -> None:
    settings = arguments(__doc__ or "D-FINE export")
    os.environ.setdefault("TORCH_HOME", str(settings.output.resolve() / "torch-cache"))
    sys.path.insert(0, str(settings.repository.resolve()))
    core = cast(Core, importlib.import_module("src.core"))
    configuration = core.YAMLConfig(
        str(settings.repository / "configs/dfine/dfine_hgnetv2_n_coco.yml")
    )
    backbone = configuration.yaml_cfg["HGNetv2"]
    if not isinstance(backbone, dict):
        raise ValueError("HGNetv2 configuration must be a mapping")
    backbone["pretrained"] = False
    configuration.yaml_cfg["eval_spatial_size"] = list(settings.input_hw)
    checkpoint = cast(
        Mapping[str, Mapping[str, torch.Tensor]],
        torch.load(settings.checkpoint, map_location="cpu", weights_only=False),
    )
    state = {key: value for key, value in checkpoint["model"].items() if key not in CACHE_KEYS}
    model = configuration.model.eval()
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != CACHE_KEYS or incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint incompatibilities: {incompatible}")
    deployed = cast(Deployable, model).deploy()
    export(Packed(deployed, dictionary_output=True), settings.output, tuple(settings.input_hw))


if __name__ == "__main__":
    main()
