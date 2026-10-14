"""Validate C5 recipes before any Slurm submission."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from app.json_data import JsonValue
from cluster.gpu_selection import GpuConstraint, selector

Lane = Literal["nv", "arm"]
SAFE_COMPONENT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")
TEACHER_REFERENCE = re.compile(r"([a-zA-Z0-9][a-zA-Z0-9_.-]*)/seed([0-9]+)\Z")
RUNNER_ARGUMENTS = frozenset(
    {
        "batch",
        "seed",
        "project",
        "name",
        "data",
        "model",
        "distill_model",
        "resume",
        "device",
        "save_dir",
    }
)
RECIPE_FIELDS = frozenset(
    {
        "id",
        "framework",
        "base_model",
        "dataset",
        "taxonomy",
        "seeds",
        "lane",
        "gpu_constraint",
        "seed_gpu_constraints",
        "est_gpu_hours",
        "teacher",
        "train",
        "deploy_sizes",
        "notes",
        "test_dataset",
        "allow_missing_evaluation",
        "batch_override",
    }
)


@dataclass(frozen=True)
class Recipe:
    identifier: str
    framework: str
    base_model: str
    dataset: str
    taxonomy: str
    seeds: tuple[int, ...]
    lane: Lane
    gpu_constraint: GpuConstraint | None
    seed_gpu_constraints: Mapping[int, GpuConstraint | None]
    estimated_gpu_hours: float
    teacher: str | None
    train: Mapping[str, JsonValue]
    deploy_sizes: tuple[tuple[int, int], ...]
    notes: str
    test_dataset: str = "columba-test-v2"
    allow_missing_evaluation: bool = False
    batch_override: int | None = None

    def constraint_for_seed(self, seed: int) -> GpuConstraint | None:
        if seed not in self.seeds:
            raise ValueError(f"seed {seed} is absent from {self.identifier}")
        return self.seed_gpu_constraints.get(seed, self.gpu_constraint)


def component(value: JsonValue, field: str) -> str:
    if not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must be a safe single path component")
    return value


def mapping(value: JsonValue, field: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a mapping with string keys")
    return value


def positive_integer(value: JsonValue, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def constraint(value: JsonValue) -> GpuConstraint | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("gpu_constraint must be a string or null")
    return selector(value)


def validate_recipe(value: JsonValue) -> Recipe:
    raw = mapping(value, "recipe")
    if unknown_fields := raw.keys() - RECIPE_FIELDS:
        raise ValueError(f"unknown recipe fields: {sorted(unknown_fields)}")
    identifier = component(raw.get("id"), "id")
    framework = component(raw.get("framework"), "framework")
    base_model = component(raw.get("base_model"), "base_model")
    dataset = component(raw.get("dataset"), "dataset")
    if dataset.startswith("columba-test"):
        raise ValueError("cannot train on a frozen test dataset")
    taxonomy = raw.get("taxonomy")
    if taxonomy not in ("single", "species"):
        raise ValueError("taxonomy must be single or species")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    checked_seeds = tuple(cast(list[int], seeds))
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("seeds must be unique")
    lane = raw.get("lane")
    if lane not in ("nv", "arm"):
        raise ValueError("lane must be nv or arm")
    default_constraint = constraint(raw.get("gpu_constraint"))
    overrides = cast(dict[int | str, JsonValue], raw.get("seed_gpu_constraints", {}))
    if not isinstance(overrides, dict):
        raise ValueError("seed_gpu_constraints must map seeds to GPU constraints")
    checked_overrides: dict[int, GpuConstraint | None] = {}
    for seed, gpu in overrides.items():
        if type(seed) is not int or seed not in checked_seeds:
            raise ValueError("each seed_gpu_constraints entry needs a listed seed")
        checked_overrides[seed] = constraint(gpu)
    if lane == "arm" and (default_constraint or checked_overrides):
        raise ValueError("arm recipes cannot request a GPU constraint")
    hours = raw.get("est_gpu_hours")
    if (
        type(hours) not in (int, float)
        or not math.isfinite(cast(float, hours))
        or cast(float, hours) < 0
    ):
        raise ValueError("est_gpu_hours must be a finite non-negative number")
    teacher = raw.get("teacher")
    if teacher is not None and (
        not isinstance(teacher, str) or not TEACHER_REFERENCE.fullmatch(teacher)
    ):
        raise ValueError("teacher must be null or <recipe_id>/seed<k>")
    train = mapping(raw.get("train"), "train")
    if reserved := train.keys() & RUNNER_ARGUMENTS:
        raise ValueError(f"runner owns train arguments: {sorted(reserved)}")
    positive_integer(train.get("imgsz"), "train.imgsz")
    positive_integer(train.get("epochs"), "train.epochs")
    if train.get("nbs", 64) != 64:
        raise ValueError("train.nbs must stay 64")
    if train.get("save", True) is not True or train.get("val", True) is not True:
        raise ValueError("train.save and train.val must be true")
    sizes = raw.get("deploy_sizes")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("deploy_sizes must contain [height, width] pairs")
    checked_sizes: list[tuple[int, int]] = []
    for size in sizes:
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("deploy_sizes must contain [height, width] pairs")
        height = positive_integer(size[0], "deploy height")
        width = positive_integer(size[1], "deploy width")
        if height % 32 or width % 32:
            raise ValueError("deploy sizes must be multiples of 32")
        checked_sizes.append((height, width))
    notes = raw.get("notes", "")
    allow_missing = raw.get("allow_missing_evaluation", False)
    if not isinstance(notes, str) or type(allow_missing) is not bool:
        raise ValueError("notes must be a string; allow_missing_evaluation must be boolean")
    if allow_missing and not identifier.startswith("smoke-"):
        raise ValueError("only smoke- recipes may allow missing evaluation")
    batch_override = raw.get("batch_override")
    if batch_override is not None:
        positive_integer(batch_override, "batch_override")
    return Recipe(
        identifier,
        framework,
        base_model,
        dataset,
        taxonomy,
        checked_seeds,
        cast(Lane, lane),
        default_constraint,
        checked_overrides,
        float(cast(float, hours)),
        teacher,
        train,
        tuple(checked_sizes),
        notes,
        component(raw.get("test_dataset", "columba-test-v2"), "test_dataset"),
        allow_missing,
        cast(int | None, batch_override),
    )


def load_recipe(path: Path) -> Recipe:
    import yaml

    return validate_recipe(yaml.safe_load(path.read_text(encoding="utf-8")))


def recipe_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_teacher(reference: str | None, runs_root: Path) -> Path | None:
    if reference is None:
        return None
    if not TEACHER_REFERENCE.fullmatch(reference):
        raise ValueError("teacher must be <recipe_id>/seed<k>")
    resolved = (runs_root / reference / "weights/best.pt").resolve()
    if not resolved.is_relative_to(runs_root.resolve()):
        raise ValueError("teacher path escapes the runs root")
    return resolved
