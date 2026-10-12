"""Read the single-class dataset configuration used by training and auditing."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from .config import ConfigError, read_mapping
from .detector import DetectorError, validate_names
from .json_data import JsonValue


def load_dataset(path: Path) -> dict[str, JsonValue]:
    try:
        dataset = read_mapping(
            cast(JsonValue, yaml.safe_load(path.read_text(encoding="utf-8"))),
            "dataset",
        )
        validate_names(dataset.get("names"))
    except (OSError, yaml.YAMLError, ConfigError, DetectorError) as error:
        raise ValueError(f"invalid dataset {path}: {error}") from error
    return dataset
