#!/usr/bin/env python
"""Download a dataset source into `$COLUMBA_DATA_ROOT/raw/<source>/` with per-image provenance.

Run from the repository root:

    python -m tools.fetch_sources data_sources/roboflow-nea.yaml
    python -m tools.fetch_sources data_sources/inaturalist.yaml --limit-per-taxon 40

Each source writes `raw/<source>/provenance.jsonl`, one record per kept image:

    {"id": "coco-397133", "file": "images/000000397133.jpg", "source": "coco-birds",
     "source_id": "397133", "scene_id": "coco-397133", "license": "CC-BY-2.0",
     "attribution": "...", "url": "...", "width": 640, "height": 427,
     "boxes": [{"class": "bird", "xyxy": [x1, y1, x2, y2]}], "label_origin": "source",
     "classes_present": ["other_bird"], "split_in_source": "val2017", "extra": {...}}

`boxes` are in original image pixels and use the source's class names after the
source definition's `classes` mapping; `null` means the image has no labels at all.
`file` is relative to `raw/<source>/`. `scene_id` groups images of one scene (a
Roboflow original, a COCO image, an iNaturalist observation) for dedupe.
Images whose licence is not allowed (CC0, CC-BY, CC-BY-SA, or a Roboflow set's
stated licence) are not downloaded; counts go to `raw/<source>/licence_report.json`.

The Roboflow API key is read from `~/.config/columba/roboflow.env`
(`ROBOFLOW_API_KEY=...`) and is never printed or logged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import yaml

DEFAULT_DATA_ROOT = Path.home() / "columba-data"
ROBOFLOW_KEY_FILE = Path.home() / ".config/columba/roboflow.env"
ROBOFLOW_API = "https://api.roboflow.com"
INATURALIST_API = "https://api.inaturalist.org/v1"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
USER_AGENT = "columba-zero-data/1.0 (school research project; pigeon detector)"
HTTP_TIMEOUT_S = 120
DOWNLOAD_CHUNK_BYTES = 1 << 20
INATURALIST_MIN_INTERVAL_S = 1.1
INATURALIST_PAGE_SIZE = 200
ROBOFLOW_EXPORT_POLL_S = 5
ROBOFLOW_EXPORT_ATTEMPTS = 60
ROBOFLOW_STEM = re.compile(r"^(?P<original>.+)_(?P<ext>jpe?g|png|bmp|webp)\.rf\.[0-9a-f]+$", re.I)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# COCO 2017 `licenses` table ids → SPDX-style identifiers.
COCO_LICENSES = {
    1: "CC-BY-NC-SA-2.0",
    2: "CC-BY-NC-2.0",
    3: "CC-BY-NC-ND-2.0",
    4: "CC-BY-2.0",
    5: "CC-BY-SA-2.0",
    6: "CC-BY-ND-2.0",
    7: "PDM",  # "No known copyright restrictions" (public domain, CONTRACTS.md C1 v1.4)
    8: "PDM",  # "United States Government Work"
}
INATURALIST_LICENSES = {"cc0": "CC0-1.0", "cc-by": "CC-BY-4.0", "cc-by-sa": "CC-BY-SA-4.0"}
ALLOWED_LICENSE = re.compile(r"^(CC0-1\.0|PDM|CC-BY-\d\.\d|CC-BY-SA-\d\.\d)$")

Record = dict[str, Any]


def data_root() -> Path:
    return Path(os.environ.get("COLUMBA_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()


def license_allowed(license_id: str | None, extra_allowed: Iterable[str] = ()) -> bool:
    """Q2 default plus C1 v1.4: CC0, public domain (PDM), CC-BY and CC-BY-SA; NC, ND and unknown
    licences are excluded."""
    if not license_id:
        return False
    return bool(ALLOWED_LICENSE.match(license_id)) or license_id in set(extra_allowed)


def load_source(path: Path) -> Record:
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("name"), str):
        raise SystemExit(f"{path}: a source definition needs a `name`")
    if source.get("kind") not in FETCHERS:
        raise SystemExit(f"{path}: unknown source kind {source.get('kind')!r}")
    return cast(Record, source)


def read_provenance(path: Path) -> list[Record]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_roboflow_key(path: Path = ROBOFLOW_KEY_FILE) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ROBOFLOW_API_KEY" and value.strip():
            return value.strip().strip("'\"")
    raise SystemExit(f"ROBOFLOW_API_KEY not found in {path}")


def http_session() -> Any:
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def download(session: Any, url: str, target: Path, label: str) -> None:
    """Stream `url` to `target` atomically. `label` is what gets logged, never the URL."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with session.get(url, stream=True, timeout=HTTP_TIMEOUT_S) as response:
        if response.status_code != 200:
            raise RuntimeError(f"download of {label} failed: HTTP {response.status_code}")
        with open(temporary, "wb") as handle:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
    temporary.replace(target)


def safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in archive.infolist():
        destination = (target / member.filename).resolve()
        if root not in destination.parents and destination != root:
            raise RuntimeError(f"unsafe path in archive: {member.filename}")
    archive.extractall(target)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).size


# --- Roboflow -------------------------------------------------------------------------------


def roboflow_original_stem(stem: str) -> str:
    """`IMG_12_jpg.rf.0a1b...` → `IMG_12`: Roboflow copies of one upload share this stem."""
    match = ROBOFLOW_STEM.match(stem)
    return match.group("original") if match else stem


def scene_for(source: Record, original_stem: str) -> str:
    """Scene id of a Roboflow original: the first matching `scene_rules` entry (a regex on the
    original stem naming a capture session, e.g. frames of one video), else the stem itself."""
    for rule in source.get("scene_rules", []):
        if re.search(rule["pattern"], original_stem):
            return f"{source['name']}-{rule['scene']}"
    return f"{source['name']}-{original_stem}"


def yolo_to_xyxy(line: str, width: int, height: int) -> tuple[int, list[float]] | None:
    parts = line.split()
    if len(parts) != 5:  # polygons or keypoints are not used by these sources
        return None
    class_index = int(parts[0])
    centre_x, centre_y, box_width, box_height = (float(value) for value in parts[1:])
    x1 = max(0.0, (centre_x - box_width / 2) * width)
    y1 = max(0.0, (centre_y - box_height / 2) * height)
    x2 = min(float(width), (centre_x + box_width / 2) * width)
    y2 = min(float(height), (centre_y + box_height / 2) * height)
    return class_index, [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def roboflow_records(source: Record, export_root: Path) -> Iterator[Record]:
    data = yaml.safe_load((export_root / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)
    mapping: dict[str, str | None] = source.get("classes", {})
    for split in ("train", "valid", "test"):
        image_directory = export_root / split / "images"
        if not image_directory.is_dir():
            continue
        for image_path in sorted(image_directory.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            width, height = image_size(image_path)
            label_path = export_root / split / "labels" / f"{image_path.stem}.txt"
            boxes: list[Record] = []
            dropped: Counter[str] = Counter()
            lines = (
                label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            )
            for line in lines:
                if not line.strip():
                    continue
                parsed = yolo_to_xyxy(line, width, height)
                if parsed is None:
                    dropped["non-box"] += 1
                    continue
                class_name = names[parsed[0]]
                mapped = mapping.get(class_name, mapping.get("*"))
                if mapped is None:
                    dropped[class_name] += 1
                    continue
                boxes.append({"class": mapped, "xyxy": parsed[1]})
            original = roboflow_original_stem(image_path.stem)
            scene = scene_for(source, original)
            yield {
                "id": f"{source['name']}-{image_path.stem}",
                "file": image_path.relative_to(export_root.parent).as_posix(),
                "source": source["name"],
                "source_id": image_path.stem,
                "scene_id": scene,
                "license": source["license"],
                "attribution": source["attribution"],
                "url": source["url"],
                "width": width,
                "height": height,
                "boxes": boxes,
                "label_origin": "source",
                "classes_present": sorted({box["class"] for box in boxes}),
                "split_in_source": split,
                "extra": {"original_stem": original, "dropped_boxes": dict(dropped)},
            }


def fetch_roboflow(source: Record, out_dir: Path, arguments: argparse.Namespace) -> list[Record]:
    export_root = out_dir / "export"
    if not (export_root / "data.yaml").exists():
        key = read_roboflow_key()
        session = http_session()
        endpoint = (
            f"{ROBOFLOW_API}/{source['workspace']}/{source['project']}/"
            f"{source['version']}/{source.get('format', 'yolov8')}"
        )
        link = None
        for _ in range(ROBOFLOW_EXPORT_ATTEMPTS):
            response = session.get(endpoint, params={"api_key": key}, timeout=HTTP_TIMEOUT_S)
            if response.status_code != 200:
                raise SystemExit(
                    f"Roboflow export request for {source['name']} failed: "
                    f"HTTP {response.status_code}"
                )
            link = response.json().get("export", {}).get("link")
            if link:
                break
            print(f"{source['name']}: waiting for Roboflow to prepare the export")
            time.sleep(ROBOFLOW_EXPORT_POLL_S)
        if not link:
            raise SystemExit(f"Roboflow export for {source['name']} never became ready")
        archive_path = out_dir / "_download" / "export.zip"
        download(session, link, archive_path, f"{source['name']} export archive")
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, export_root)
        shutil.rmtree(archive_path.parent)
    return list(roboflow_records(source, export_root))


# --- COCO -----------------------------------------------------------------------------------


def coco_annotations(session: Any, root: Path, split: str) -> Record:
    annotation_dir = root / "raw" / "_coco-annotations"
    path = annotation_dir / "annotations" / f"instances_{split}.json"
    if not path.exists():
        archive_path = annotation_dir / "annotations_trainval2017.zip"
        if not archive_path.exists():
            download(session, COCO_ANNOTATIONS_URL, archive_path, "COCO 2017 annotations")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if member.startswith("annotations/instances_"):
                    archive.extract(member, annotation_dir)
    with open(path, encoding="utf-8") as handle:
        return cast(Record, json.load(handle))


def select_coco_images(
    source: Record, annotations: Record
) -> tuple[list[tuple[Record, list[Record]]], Counter[str]]:
    """Apply the source's category rules and licence filter.

    Returns the kept images with their boxes, and kept/dropped counts per licence.
    """
    categories = {category["name"]: category["id"] for category in annotations["categories"]}
    by_id = {category["id"]: category["name"] for category in annotations["categories"]}
    require = {categories[name] for name in source.get("require_any", [])}
    exclude = {categories[name] for name in source.get("exclude_any", [])}
    boxes_for = {categories[name] for name in source.get("box_categories", [])}
    per_image: dict[int, list[Record]] = {}
    for annotation in annotations["annotations"]:
        per_image.setdefault(annotation["image_id"], []).append(annotation)
    licences: Counter[str] = Counter()
    kept: list[tuple[Record, list[Record]]] = []
    for image in sorted(annotations["images"], key=lambda item: int(item["id"])):
        image_annotations = per_image.get(image["id"], [])
        present = {annotation["category_id"] for annotation in image_annotations}
        if require and not present & require:
            continue
        if present & exclude:
            continue
        if len(present & require) < source.get("min_required_categories", 1) and require:
            continue
        licence = COCO_LICENSES.get(image["license"], f"coco-license-{image['license']}")
        allowed = license_allowed(licence, source.get("extra_allowed_licenses", []))
        licences[f"{'kept' if allowed else 'dropped'}:{licence}"] += 1
        if not allowed:
            continue
        boxes = [
            {**annotation, "category_name": by_id[annotation["category_id"]]}
            for annotation in image_annotations
            if annotation["category_id"] in boxes_for
        ]
        kept.append((image, boxes))
    return kept, licences


def fetch_coco(source: Record, out_dir: Path, arguments: argparse.Namespace) -> list[Record]:
    root = data_root()
    session = http_session()
    split = source.get("split", "val2017")
    annotations = coco_annotations(session, root, split)
    licence_names = {item["id"]: item["name"] for item in annotations["licenses"]}
    kept, licences = select_coco_images(source, annotations)
    limit = arguments.limit or source.get("limit")
    if limit:
        kept = kept[: int(limit)]
    mapping: dict[str, str] = source.get("classes", {})
    records = []
    for index, (image, annotations_for_image) in enumerate(kept, 1):
        target = out_dir / "images" / image["file_name"]
        if not target.exists():
            download(session, image["coco_url"], target, f"COCO image {image['id']}")
        if index % 100 == 0:
            print(f"{source['name']}: {index}/{len(kept)} images")
        boxes: list[Record] = []
        for annotation in annotations_for_image:
            if annotation.get("iscrowd"):
                continue
            x, y, width, height = annotation["bbox"]
            boxes.append(
                {
                    "class": mapping.get(annotation["category_name"], annotation["category_name"]),
                    "xyxy": [round(x, 2), round(y, 2), round(x + width, 2), round(y + height, 2)],
                }
            )
        crowd = sum(1 for annotation in annotations_for_image if annotation.get("iscrowd"))
        licence = COCO_LICENSES[image["license"]]
        records.append(
            {
                "id": f"coco-{image['id']}",
                "file": target.relative_to(out_dir).as_posix(),
                "source": source["name"],
                "source_id": str(image["id"]),
                "scene_id": f"coco-{image['id']}",
                "license": licence,
                "attribution": (
                    f"Flickr photo {image.get('flickr_url', '')} "
                    f"({licence_names[image['license']]}), via COCO 2017 {split}"
                ),
                "url": image.get("flickr_url") or image["coco_url"],
                "width": image["width"],
                "height": image["height"],
                "boxes": boxes,
                "label_origin": "source",
                "classes_present": sorted({box["class"] for box in boxes}),
                "split_in_source": split,
                "extra": {"coco_url": image["coco_url"], "crowd_boxes": crowd},
            }
        )
    write_json(out_dir / "licence_report.json", dict(sorted(licences.items())))
    return records


# --- iNaturalist ----------------------------------------------------------------------------


class RateLimiter:
    def __init__(
        self,
        interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval_s = interval_s
        self.clock = clock
        self.sleep = sleep
        self.last = -interval_s

    def wait(self) -> None:
        delay = self.last + self.interval_s - self.clock()
        if delay > 0:
            self.sleep(delay)
        self.last = self.clock()


def inaturalist_get(
    session: Any, limiter: RateLimiter, path: str, params: dict[str, Any]
) -> Record:
    for attempt in range(5):
        limiter.wait()
        response = session.get(f"{INATURALIST_API}/{path}", params=params, timeout=HTTP_TIMEOUT_S)
        if response.status_code == 200:
            return cast(Record, response.json())
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(30 * (attempt + 1))  # back off well beyond their 1 request/s guidance
            continue
        raise RuntimeError(f"iNaturalist {path} failed: HTTP {response.status_code}")
    raise RuntimeError(f"iNaturalist {path} kept failing")


def photo_url(url: str, size: str) -> str:
    return re.sub(r"/(square|small|thumb|medium|large|original)\.", f"/{size}.", url)


def inaturalist_observations(
    session: Any, limiter: RateLimiter, taxon: Record, source: Record, limit: int
) -> Iterator[Record]:
    params: dict[str, Any] = {
        "taxon_id": taxon["taxon_id"],
        "quality_grade": "research",
        "photos": "true",
        "photo_license": ",".join(source["photo_licenses"]),
        "per_page": INATURALIST_PAGE_SIZE,
        "order_by": "id",
        "order": "desc",
        **source.get("query", {}),
    }
    if taxon.get("exclude_taxon_ids"):
        params["without_taxon_id"] = ",".join(str(item) for item in taxon["exclude_taxon_ids"])
    places = taxon.get("place_ids", source.get("place_ids", [None]))
    yielded = 0
    for place in places:
        page_params = dict(params)
        if place is not None:
            page_params["place_id"] = place
        while yielded < limit:
            page = inaturalist_get(session, limiter, "observations", page_params)
            results = page.get("results", [])
            if not results:
                break
            for observation in results:
                observation["_place_id"] = place
                yield observation
                yielded += 1
                if yielded >= limit:
                    return
            page_params["id_below"] = results[-1]["id"]


def fetch_inaturalist(source: Record, out_dir: Path, arguments: argparse.Namespace) -> list[Record]:
    session = http_session()
    limiter = RateLimiter(INATURALIST_MIN_INTERVAL_S)
    photo_limiter = RateLimiter(float(source.get("photo_interval_s", INATURALIST_MIN_INTERVAL_S)))
    size = source.get("photo_size", "large")
    default_limit = int(arguments.limit_per_taxon or source.get("limit_per_taxon", 50))
    photos_per_observation = int(source.get("photos_per_observation", 1))
    existing = {}
    provenance = out_dir / "provenance.jsonl"
    if provenance.exists():
        existing = {record["id"]: record for record in read_provenance(provenance)}
    records: dict[str, Record] = dict(existing)
    licences: Counter[str] = Counter()
    for taxon in source["taxa"]:
        if arguments.taxon and taxon["name"] not in arguments.taxon:
            continue
        per_taxon = int(arguments.limit_per_taxon or taxon.get("limit", default_limit))
        count = sum(1 for record in records.values() if record["extra"]["taxon"] == taxon["name"])
        if count >= per_taxon:
            continue
        for observation in inaturalist_observations(
            session, limiter, taxon, source, 10 * per_taxon
        ):
            if count >= per_taxon:
                break
            for photo in observation.get("photos", [])[:photos_per_observation]:
                licence = INATURALIST_LICENSES.get(photo.get("license_code") or "", None)
                licences[f"{'kept' if licence else 'dropped'}:{photo.get('license_code')}"] += 1
                if licence is None or not license_allowed(licence):
                    continue
                image_id = f"inat-{observation['id']}-{photo['id']}"
                if image_id in records:
                    continue
                target = out_dir / "images" / taxon["name"] / f"{image_id}.jpg"
                photo_limiter.wait()
                try:
                    download(session, photo_url(photo["url"], size), target, image_id)
                except RuntimeError as error:
                    print(f"skipping {image_id}: {error}")
                    continue
                width, height = image_size(target)
                user = observation.get("user") or {}
                observer = user.get("name") or user.get("login") or "unknown"
                observation_url = f"https://www.inaturalist.org/observations/{observation['id']}"
                records[image_id] = {
                    "id": image_id,
                    "file": target.relative_to(out_dir).as_posix(),
                    "source": source["name"],
                    "source_id": f"{observation['id']}/{photo['id']}",
                    "scene_id": f"inat-obs-{observation['id']}",
                    "license": licence,
                    "attribution": (
                        f"{photo.get('attribution') or f'(c) {observer}'} [{licence}], "
                        f"{observation_url}"
                    ),
                    "url": observation_url,
                    "width": width,
                    "height": height,
                    "boxes": None,
                    "label_origin": None,
                    "classes_present": [taxon["name"]],
                    "split_in_source": None,
                    "extra": {
                        "taxon": taxon["name"],
                        "taxon_id": (observation.get("taxon") or {}).get("id"),
                        "taxon_name": (observation.get("taxon") or {}).get("name"),
                        "photo_id": photo["id"],
                        "photo_url": photo_url(photo["url"], size),
                        "observer_login": user.get("login"),
                        "observed_on": observation.get("observed_on"),
                        "place_id": observation.get("_place_id"),
                        "place_guess": observation.get("place_guess"),
                    },
                }
                count += 1
                if count % 100 == 0:
                    print(f"{source['name']}: {taxon['name']} {count}/{per_taxon}")
                    write_inaturalist_state(out_dir, records, licences)
                if count >= per_taxon:
                    break
        print(f"{source['name']}: {taxon['name']} has {count} images")
        write_inaturalist_state(out_dir, records, licences)
    write_inaturalist_state(out_dir, records, licences)
    return sorted(records.values(), key=lambda record: record["id"])


def write_inaturalist_state(out_dir: Path, records: dict[str, Record], seen: Counter[str]) -> None:
    """Checkpoint provenance, and a licence report: kept images by licence (from provenance),
    and photo licence codes seen in this run (`seen_this_run:<code>`)."""
    write_jsonl(out_dir / "provenance.jsonl", sorted(records.values(), key=lambda r: r["id"]))
    report: dict[str, int] = dict(Counter(f"kept:{r['license']}" for r in records.values()))
    report.update({f"seen_this_run:{key.split(':', 1)[1]}": value for key, value in seen.items()})
    write_json(out_dir / "licence_report.json", dict(sorted(report.items())))


FETCHERS: dict[str, Callable[[Record, Path, argparse.Namespace], list[Record]]] = {
    "roboflow": fetch_roboflow,
    "coco": fetch_coco,
    "inaturalist": fetch_inaturalist,
}


def fetch(source_path: Path, arguments: argparse.Namespace) -> Path:
    source = load_source(source_path)
    out_dir = data_root() / "raw" / str(source["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    records = FETCHERS[str(source["kind"])](source, out_dir, arguments)
    for record in records:
        if not license_allowed(record["license"], source.get("extra_allowed_licenses", [])):
            raise SystemExit(f"{record['id']}: licence {record['license']!r} is not allowed")
        if not record.get("attribution"):
            raise SystemExit(f"{record['id']}: attribution missing")
    write_jsonl(out_dir / "provenance.jsonl", records)
    write_json(
        out_dir / "source.json",
        {
            "definition": source,
            "images": len(records),
            "fetched_at": time.strftime("%FT%TZ", time.gmtime()),
        },
    )
    print(f"{source['name']}: {len(records)} images in {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+", type=Path, help="data_sources/<name>.yaml files")
    parser.add_argument("--limit", type=int, help="COCO: keep at most this many images")
    parser.add_argument("--limit-per-taxon", type=int, help="iNaturalist: images per taxon")
    parser.add_argument("--taxon", action="append", help="iNaturalist: only these taxon names")
    arguments = parser.parse_args()
    for source_path in arguments.sources:
        fetch(source_path, arguments)


if __name__ == "__main__":
    main()
