"""Map approved GPU selectors to the cluster's current Slurm inventory."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

GpuConstraint = Literal[
    "t4",
    "titanrtx",
    "titanv",
    "a100-40",
    "a100-80",
    "h100-47",
    "h100-96",
    "h200",
    "a100",
    "h100",
]
GPU_CONSTRAINTS = (
    "t4",
    "titanrtx",
    "titanv",
    "a100-40",
    "a100-80",
    "h100-47",
    "h100-96",
    "h200",
    "a100",
    "h100",
)
ALLOWED_GRES_TYPES = frozenset({"nv", "a100-40", "a100-80", "h100-47", "h100-96", "h200-141"})
GPU_PARTITIONS = ("gpu-long", "gpu")
PARTITION_TIME_LIMIT_MINUTES = {"gpu-long": 72 * 60, "gpu": 3 * 60}
PARTITION_TIME_LIMITS = {"gpu-long": "3-00:00:00", "gpu": "03:00:00"}
GPU_NODE_PATTERN = re.compile(r"xgp[d-k][0-9]+\Z")


@dataclass(frozen=True)
class GpuOffer:
    node: str
    partition: str
    resource_type: str
    features: frozenset[str]
    free_units: int = 1
    state: str = "idle"


@dataclass(frozen=True)
class GpuRequest:
    partitions: tuple[str, ...]
    resources: str
    constraint: str | None
    time_limit: str

    def arguments(self) -> tuple[str, ...]:
        arguments = (
            f"--partition={','.join(self.partitions)}",
            f"--gres={self.resources}",
            f"--time={self.time_limit}",
        )
        return (*arguments, f"--constraint={self.constraint}") if self.constraint else arguments


def selector(value: str | None) -> GpuConstraint | None:
    if value is None:
        return None
    if value not in GPU_CONSTRAINTS:
        raise ValueError(f"GPU selector must be null or one of {', '.join(GPU_CONSTRAINTS)}")
    return cast(GpuConstraint, value)


def expand_nodes(expression: str) -> tuple[str, ...]:
    nodes: list[str] = []
    for match in re.finditer(r"([^,\[\]]+)(?:\[([^\]]+)\])?", expression):
        prefix, ranges = match.groups()
        if ranges is None:
            nodes.append(prefix)
            continue
        for span in ranges.split(","):
            bounds = span.split("-")
            first, last = int(bounds[0]), int(bounds[-1])
            padding = len(bounds[0]) if bounds[0].startswith("0") and len(bounds[0]) > 1 else 0
            nodes.extend(f"{prefix}{index:0{padding}d}" for index in range(first, last + 1))
    return tuple(nodes)


def parse_inventory(output: str) -> tuple[GpuOffer, ...]:
    offers: list[GpuOffer] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = line.strip().split("|") if "|" in line else line.split()
        if len(columns) not in {4, 6}:
            raise ValueError(f"unexpected sinfo row: {line}")
        if len(columns) == 6:
            partition, nodes, resources, used_resources, features, state = columns
        else:
            partition, nodes, resources, features = columns
            used_resources, state = "", "idle"
        partition = partition.rstrip("*")
        if partition not in GPU_PARTITIONS:
            continue
        for resource in re.finditer(r"(?:^|,)gpu:([^:(),]+):([0-9]+)", resources):
            resource_type = resource.group(1)
            if resource_type not in ALLOWED_GRES_TYPES:
                continue
            used = re.search(r"gpu:" + re.escape(resource_type) + r":([0-9]+)", used_resources)
            free_units = max(0, int(resource.group(2)) - (int(used.group(1)) if used else 0))
            if state not in {"idle", "mixed", "allocated"}:
                free_units = 0
            offers.extend(
                GpuOffer(
                    node,
                    partition,
                    resource_type,
                    frozenset(features.split(",")),
                    free_units,
                    state,
                )
                for node in expand_nodes(nodes)
            )
    return tuple(offers)


def gpu_inventory() -> tuple[GpuOffer, ...]:
    completed = subprocess.run(
        [
            "sinfo",
            "-N",
            "--noheader",
            "--partition=gpu-long,gpu",
            "--Format=Partition:12,NodeList:24,Gres:40,GresUsed:80,Features:100,StateLong:20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_inventory(completed.stdout)


def resource_selection(choice: GpuConstraint | None) -> tuple[str | None, str | None]:
    if choice in {"t4", "titanrtx", "titanv"}:
        return "nv", choice
    if choice == "h200":
        return "h200-141", None
    if choice in {"a100", "h100"}:
        return None, choice
    return choice, None


def request_for(
    choice: str | None,
    offers: Sequence[GpuOffer],
    *,
    required_node: str | None = None,
    excluded_nodes: Sequence[str] = (),
) -> GpuRequest:
    checked_choice = selector(choice)
    if checked_choice is None and required_node is None:
        raise ValueError("automatic GPU selection requires the configured capacity policy")
    resource_type, feature = resource_selection(checked_choice)
    if required_node and not GPU_NODE_PATTERN.fullmatch(required_node):
        raise ValueError("pinned GPU node must be an approved cluster GPU node")
    matched = [
        offer
        for offer in offers
        if offer.node not in excluded_nodes
        and (required_node is None or offer.node == required_node)
        and (resource_type is None or offer.resource_type == resource_type)
        and (feature is None or feature in offer.features)
    ]
    if not matched:
        raise ValueError(f"no approved GPU inventory matches {choice!r} on {required_node!r}")
    partitions = tuple(
        partition
        for partition in GPU_PARTITIONS
        if any(offer.partition == partition for offer in matched)
    )
    if checked_choice is not None or required_node is not None:
        partitions = partitions[:1]
    # Slurm rejects a 72-hour request spanning the 3-hour gpu partition.
    time_limit = PARTITION_TIME_LIMITS["gpu" if "gpu" in partitions else "gpu-long"]
    if checked_choice in {"a100", "h100"}:
        family_offers = [offer for offer in matched if offer.partition in partitions]
        resource_type = max(family_offers, key=lambda offer: offer.free_units).resource_type
        feature = None
    if checked_choice is None and required_node is not None:
        resource_type = matched[0].resource_type
        feature = (
            next(
                (
                    candidate
                    for candidate in ("t4", "titanrtx", "titanv")
                    if candidate in matched[0].features
                ),
                None,
            )
            if resource_type == "nv"
            else None
        )
    resources = f"gpu:{resource_type}:1" if resource_type else "gpu:1"
    constraint = feature
    return GpuRequest(partitions, resources, constraint, time_limit)
