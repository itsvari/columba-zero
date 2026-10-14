#!/usr/bin/env python
"""Hash images and group exact and near duplicates so that groups never cross splits.

    python -m tools.dedupe_images ~/columba-data/raw/roboflow-nea/provenance.jsonl \
        ~/columba-data/raw/roboflow-yolo1199/provenance.jsonl --max-distance 6 \
        --output outputs/ws2/groups.jsonl --clusters outputs/ws2/clusters.json

Each image gets a SHA-256 of its file bytes and a 64-bit perceptual hash (`imagehash.phash`
on the EXIF-oriented image). Two images are linked when their SHA-256 values are equal,
their phash Hamming distance is at most `--max-distance`, or they share a `scene_id`
(e.g. Roboflow copies `<stem>_jpg.rf.<hash>.jpg` of one original, or photos of one
iNaturalist observation). Linked images form a group (connected components);
the group id is `g-` plus a hash of the group's smallest image id, so it is stable
when unrelated images are added.

Hashes are cached per source in `raw/<source>/hashes.jsonl`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PHASH_BITS = 64
DEFAULT_MAX_DISTANCE = 6
HASH_CHUNK_BYTES = 1 << 20

Record = dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hashes(path: Path) -> tuple[str, str]:
    """phash of the image and of its left-right mirror (Roboflow sets contain flipped copies)."""
    import imagehash
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return str(imagehash.phash(image)), str(imagehash.phash(ImageOps.mirror(image)))


def hamming(left: str | int, right: str | int) -> int:
    left_value = int(left, 16) if isinstance(left, str) else left
    right_value = int(right, 16) if isinstance(right, str) else right
    return (left_value ^ right_value).bit_count()


def hash_source(provenance_path: Path) -> dict[str, Record]:
    """SHA-256 and phash for every record of one raw source, cached next to its provenance."""
    source_root = provenance_path.parent
    cache_path = source_root / "hashes.jsonl"
    cache: dict[str, Record] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as handle:
            cache = {item["id"]: item for item in map(json.loads, handle) if item}
    with open(provenance_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    changed = False
    for record in records:
        cached = cache.get(record["id"])
        if cached is not None and cached.get("file") == record["file"]:
            continue
        path = source_root / record["file"]
        phash, phash_mirror = perceptual_hashes(path)
        cache[record["id"]] = {
            "id": record["id"],
            "file": record["file"],
            "sha256": file_sha256(path),
            "phash": phash,
            "phash_mirror": phash_mirror,
        }
        changed = True
    if changed:
        with open(cache_path, "w", encoding="utf-8") as handle:
            for item in sorted(cache.values(), key=lambda item: str(item["id"])):
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    return {record["id"]: cache[record["id"]] for record in records}


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # The smaller id becomes the root so results don't depend on input order.
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


def band_masks(max_distance: int) -> list[tuple[int, int]]:
    """Split 64 bits into `max_distance + 1` bands (shift, mask). By pigeonhole, two hashes within
    `max_distance` bits of each other agree exactly on at least one band."""
    bands = max_distance + 1
    edges = [round(index * PHASH_BITS / bands) for index in range(bands + 1)]
    return [(low, (1 << (high - low)) - 1) for low, high in zip(edges, edges[1:], strict=False)]


def near_duplicate_pairs(
    hashes: dict[str, str], max_distance: int, mirrors: dict[str, str] | None = None
) -> list[tuple[str, str, int]]:
    """All pairs (a, b, distance) with a < b and phash distance ≤ max_distance.

    With `mirrors` (id → phash of the mirrored image), a pair also matches when one image is
    the mirror of the other; the reported distance is the smaller of the two.
    """
    values = {(item, False): int(value, 16) for item, value in hashes.items()}
    for item, value in (mirrors or {}).items():
        values[(item, True)] = int(value, 16)
    candidates: set[tuple[str, str]] = set()
    for shift, mask in band_masks(max_distance):
        buckets: dict[int, list[tuple[str, bool]]] = defaultdict(list)
        for key, value in values.items():
            buckets[(value >> shift) & mask].append(key)
        for members in buckets.values():
            for index, (left, left_mirrored) in enumerate(members):
                for right, right_mirrored in members[index + 1 :]:
                    # Comparing two mirrored hashes repeats the unmirrored comparison.
                    if left != right and not (left_mirrored and right_mirrored):
                        candidates.add((min(left, right), max(left, right)))
    pairs = []
    for left, right in sorted(candidates):
        distance = hamming(values[(left, False)], values[(right, False)])
        if mirrors is not None and right in mirrors:
            distance = min(distance, hamming(values[(left, False)], values[(right, True)]))
        if distance <= max_distance:
            pairs.append((left, right, distance))
    return pairs


def group_id(smallest_member: str) -> str:
    return "g-" + hashlib.sha256(smallest_member.encode()).hexdigest()[:12]


def assign_groups(items: Sequence[Record], max_distance: int) -> tuple[dict[str, str], list[Record]]:
    """Group records with keys `id`, `sha256`, `phash` and optional `scene_id`.

    Returns {id: group} and a list of near-duplicate clusters (groups with more than one image),
    each `{"group", "members", "links": [[a, b, reason], ...]}`.
    """
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("image ids must be unique")
    union = UnionFind(ids)
    links: list[tuple[str, str, str]] = []
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in items:
        by_key[("sha256", item["sha256"])].append(item["id"])
        if item.get("scene_id"):
            by_key[("scene", item["scene_id"])].append(item["id"])
    for (kind, _), members in sorted(by_key.items()):
        members.sort()
        for member in members[1:]:
            union.union(members[0], member)
            links.append((members[0], member, kind))
    mirrors = {item["id"]: item["phash_mirror"] for item in items if item.get("phash_mirror")}
    for left, right, distance in near_duplicate_pairs(
        {item["id"]: item["phash"] for item in items}, max_distance, mirrors
    ):
        union.union(left, right)
        links.append((left, right, f"phash:{distance}"))
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for item_id in ids:
        members_by_root[union.find(item_id)].append(item_id)
    groups: dict[str, str] = {}
    clusters: list[Record] = []
    links_by_root: dict[str, list[list[str]]] = defaultdict(list)
    for left, right, reason in links:
        links_by_root[union.find(left)].append([left, right, reason])
    for root, members in sorted(members_by_root.items()):
        group = group_id(min(members))
        for member in members:
            groups[member] = group
        if len(members) > 1:
            clusters.append(
                {"group": group, "members": sorted(members), "links": links_by_root[root]}
            )
    return groups, clusters


def nearest_distances(
    queries: dict[str, Sequence[str]], references: dict[str, Sequence[str]]
) -> dict[str, tuple[int, str | None]]:
    """For each query id, the smallest phash distance to any reference image and that image's id.

    Values are sequences of hex hashes per image (e.g. `[phash, phash_mirror]`); the distance
    between two images is the minimum over all hash pairs.
    """
    import numpy

    if not references:
        return {item: (PHASH_BITS + 1, None) for item in queries}
    reference_ids = [item for item, values in references.items() for _ in values]
    reference_values = numpy.array(
        [int(value, 16) for values in references.values() for value in values],
        dtype=numpy.uint64,
    )
    result: dict[str, tuple[int, str | None]] = {}
    for item, values in queries.items():
        best: tuple[int, str | None] = (PHASH_BITS + 1, None)
        for value in values:
            distances = numpy.bitwise_count(
                numpy.bitwise_xor(reference_values, numpy.uint64(int(value, 16)))
            )
            index = int(numpy.argmin(distances))
            if int(distances[index]) < best[0]:
                best = (int(distances[index]), reference_ids[index])
        result[item] = best
    return result


def load_items(provenance_paths: Sequence[Path]) -> list[Record]:
    items = []
    for provenance_path in provenance_paths:
        hashes = hash_source(provenance_path)
        with open(provenance_path, encoding="utf-8") as handle:
            for record in (json.loads(line) for line in handle if line.strip()):
                items.append({**record, **hashes[record["id"]]})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("provenance", nargs="+", type=Path, help="raw/<source>/provenance.jsonl")
    parser.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--output", type=Path, help="write {id, source, group, sha256, phash} lines")
    parser.add_argument("--clusters", type=Path, help="write the multi-image clusters as JSON")
    arguments = parser.parse_args()
    items = load_items(arguments.provenance)
    groups, clusters = assign_groups(items, arguments.max_distance)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with open(arguments.output, "w", encoding="utf-8") as handle:
            for item in items:
                line = {key: item[key] for key in ("id", "source", "sha256", "phash")}
                handle.write(json.dumps({**line, "group": groups[item["id"]]}) + "\n")
    if arguments.clusters:
        arguments.clusters.parent.mkdir(parents=True, exist_ok=True)
        arguments.clusters.write_text(json.dumps(clusters, indent=1) + "\n", encoding="utf-8")
    exact = sum(1 for cluster in clusters for link in cluster["links"] if link[2] == "sha256")
    near = sum(
        1 for cluster in clusters for link in cluster["links"] if link[2].startswith("phash")
    )
    print(
        f"{len(items)} images, {len(set(groups.values()))} groups, {len(clusters)} multi-image "
        f"groups, {exact} exact-duplicate links, {near} near-duplicate links "
        f"(phash distance <= {arguments.max_distance})"
    )


if __name__ == "__main__":
    main()
