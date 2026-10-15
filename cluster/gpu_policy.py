"""Choose explicit GPU capacity without parking automatic runs in a busy queue."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from app.json_data import JsonValue
from cluster.gpu_selection import (
    GPU_CONSTRAINTS,
    GPU_PARTITIONS,
    GpuOffer,
    GpuRequest,
    request_for,
    resource_selection,
    selector,
)

QOS_GPU_LIMITS = {
    "nv": 8,
    "a100-40": 4,
    "a100-80": 2,
    "h100-47": 2,
    "h100-96": 1,
    "h200-141": 1,
    "h200-71": 1,
}
POLICY_PATH = Path(__file__).with_name("gpu_policy.yaml")


@dataclass(frozen=True)
class GpuPolicy:
    heavy_image_size_px: int
    heavy_est_gpu_hours: float
    allow_short_partition: bool
    light_preferences: tuple[str, ...]
    heavy_preferences: tuple[str, ...]
    type_limits: Mapping[str, int]
    nice: int
    max_submit_jobs: int

    def preferences(
        self, model: str, image_size_px: int, teacher: bool, est_gpu_hours: float
    ) -> tuple[str, ...]:
        heavy = (
            re.search(r"x(?:[-.]|$)", model) is not None
            or image_size_px >= self.heavy_image_size_px
            or teacher
            or est_gpu_hours >= self.heavy_est_gpu_hours
        )
        return self.heavy_preferences if heavy else self.light_preferences


def load_policy(path: Path = POLICY_PATH) -> GpuPolicy:
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("GPU policy must be a mapping")
    preferences: list[tuple[str, ...]] = []
    for key in ("light_preferences", "heavy_preferences"):
        choices = values.get(key)
        if (
            not isinstance(choices, list)
            or not choices
            or any(
                not isinstance(choice, str) or choice not in {*GPU_CONSTRAINTS, "xgpj0", "nv"}
                for choice in choices
            )
        ):
            raise ValueError(f"invalid GPU policy {key}")
        preferences.append(tuple(choices))
    image_size_px = values.get("heavy_image_size_px")
    est_gpu_hours = values.get("heavy_est_gpu_hours")
    allow_short = values.get("allow_short_partition")
    if type(image_size_px) is not int or image_size_px <= 0:
        raise ValueError("policy image size must be positive")
    if (
        not isinstance(est_gpu_hours, (int, float))
        or isinstance(est_gpu_hours, bool)
        or est_gpu_hours <= 0
    ):
        raise ValueError("policy hours must be positive")
    if type(allow_short) is not bool:
        raise ValueError("allow_short_partition must be boolean")
    limits = positive_limits(cast(JsonValue, values.get("type_limits")), "type_limits")
    if set(limits) != set(QOS_GPU_LIMITS) or any(
        value > QOS_GPU_LIMITS[key] for key, value in limits.items()
    ):
        raise ValueError("type_limits cannot raise the verified QoS caps")
    nice = values.get("nice", 0)
    if type(nice) is not int or not 0 <= nice <= 10000:
        raise ValueError("nice must be an integer within 0..10000")
    max_submit = values.get("max_submit_jobs")
    if type(max_submit) is not int or not 1 <= max_submit <= 32:
        raise ValueError("max_submit_jobs must be within verified association cap 32")
    return GpuPolicy(
        image_size_px,
        float(est_gpu_hours),
        allow_short,
        preferences[0],
        preferences[1],
        limits,
        nice,
        max_submit,
    )


def positive_limits(value: JsonValue, field: str) -> Mapping[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a nonempty resource/count mapping")
    if any(type(count) is not int or count <= 0 for count in value.values()):
        raise ValueError(f"{field} counts must be positive integers")
    return MappingProxyType(cast(dict[str, int], value))


def offer_choice(offer: GpuOffer) -> str:
    if offer.resource_type == "nv":
        return next(choice for choice in ("t4", "titanv", "titanrtx") if choice in offer.features)
    return "h200" if offer.resource_type == "h200-141" else offer.resource_type


def available_request(
    offers: Sequence[GpuOffer],
    preferences: Sequence[str],
    *,
    excluded_nodes: Sequence[str] = (),
    allow_short_partition: bool = True,
    avoid: tuple[str, str] | None = None,
) -> tuple[GpuRequest, str] | None:
    partitions = GPU_PARTITIONS if allow_short_partition else GPU_PARTITIONS[:1]
    for preference in preferences:
        choice = "a100-80" if preference == "xgpj0" else preference
        resource_type, feature = (
            ("nv", None) if choice == "nv" else resource_selection(selector(choice))
        )
        eligible = [
            offer
            for offer in offers
            if offer.free_units > 0
            and offer.node not in excluded_nodes
            and (preference != "xgpj0" or offer.node == "xgpj0")
            and (resource_type is None or offer.resource_type == resource_type)
            and (feature is None or feature in offer.features)
            and (
                avoid is None
                or (offer_choice(offer), offer.partition) != avoid
                and (offer.resource_type, offer.partition) != avoid
            )
        ]
        for partition in partitions:
            matched = [offer for offer in eligible if offer.partition == partition]
            if not matched:
                continue
            chosen_offer = max(matched, key=lambda offer: offer.free_units)
            chosen = offer_choice(chosen_offer)
            request = request_for(chosen, [chosen_offer])
            return request, chosen_offer.node
    return None
