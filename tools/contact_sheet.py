#!/usr/bin/env python
"""Render images with their boxes onto numbered contact sheets for review.

    python -m tools.contact_sheet ~/columba-data/columba-test-v2 \
        --output-dir ~/columba-data/contact_sheets/test-v2
    python -m tools.contact_sheet ~/columba-data/raw/coco-birds/provenance.jsonl --ids ids.txt \
        --output-dir ~/columba-data/contact_sheets/coco-birds

The input is either a C1 dataset directory (reads `manifest.jsonl` and YOLO labels) or a raw
source `provenance.jsonl` (reads its pixel boxes). Each tile shows the image letterboxed into
the tile, its boxes (red = pigeon, blue = other classes), and a caption with the tile number
and image id. Pigeon boxes whose short side is under 32 px at the 384x640 deploy letterbox
are drawn yellow. A `<sheet>.txt` next to each JPEG lists tile number → image id.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TILE_PX = 320
DEFAULT_COLUMNS = 5
DEFAULT_PER_SHEET = 30
CAPTION_PX = 14
DEPLOY_HW = (384, 640)
SMALL_BOX_PX = 32
COLOURS = {"pigeon": (230, 30, 30), "small": (255, 220, 0), "other": (40, 120, 255)}


@dataclass
class SheetEntry:
    image: Path
    caption: str
    boxes: list[tuple[str, list[float]]] = field(default_factory=list)  # (class, xyxy pixels)


def letterbox_scale(width: int, height: int, input_hw: tuple[int, int] = DEPLOY_HW) -> float:
    return min(input_hw[0] / height, input_hw[1] / width)


def is_small(xyxy: Sequence[float], width: int, height: int) -> bool:
    short_side = min(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    return short_side * letterbox_scale(width, height) < SMALL_BOX_PX


def render_tile(entry: SheetEntry, tile_px: int) -> Any:
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(entry.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    scale = min(tile_px / width, (tile_px - CAPTION_PX) / height)
    shown = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    tile = Image.new("RGB", (tile_px, tile_px), (40, 40, 40))
    offset_x = (tile_px - shown.width) // 2
    offset_y = CAPTION_PX + (tile_px - CAPTION_PX - shown.height) // 2
    tile.paste(shown, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    for class_name, xyxy in entry.boxes:
        if class_name == "pigeon":
            colour = COLOURS["small"] if is_small(xyxy, width, height) else COLOURS["pigeon"]
        else:
            colour = COLOURS["other"]
        draw.rectangle(
            [
                offset_x + xyxy[0] * scale,
                offset_y + xyxy[1] * scale,
                offset_x + xyxy[2] * scale,
                offset_y + xyxy[3] * scale,
            ],
            outline=colour,
            width=2,
        )
    draw.rectangle([0, 0, tile_px, CAPTION_PX], fill=(0, 0, 0))
    draw.text((2, 1), entry.caption[: tile_px // 6], fill=(255, 255, 255))
    return tile


def render_sheet(entries: Sequence[SheetEntry], output: Path, columns: int, tile_px: int) -> None:
    from PIL import Image

    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_px, max(1, rows) * tile_px), (0, 0, 0))
    for index, entry in enumerate(entries):
        sheet.paste(
            render_tile(entry, tile_px), ((index % columns) * tile_px, (index // columns) * tile_px)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=85)


def render_sheets(
    entries: Sequence[SheetEntry],
    output_dir: Path,
    prefix: str = "sheet",
    per_sheet: int = DEFAULT_PER_SHEET,
    columns: int = DEFAULT_COLUMNS,
    tile_px: int = DEFAULT_TILE_PX,
) -> list[Path]:
    outputs = []
    for start in range(0, len(entries), per_sheet):
        chunk = entries[start : start + per_sheet]
        output = output_dir / f"{prefix}-{start // per_sheet:03d}.jpg"
        render_sheet(chunk, output, columns, tile_px)
        output.with_suffix(".txt").write_text(
            "".join(
                f"{start + index} {entry.caption}\t{entry.image}\n"
                for index, entry in enumerate(chunk)
            ),
            encoding="utf-8",
        )
        outputs.append(output)
    return outputs


def yolo_boxes(
    label_path: Path, names: Sequence[str], width: int, height: int
) -> list[tuple[str, list[float]]]:
    boxes: list[tuple[str, list[float]]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(value) for value in parts[1:])
        boxes.append(
            (
                names[int(parts[0])],
                [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ],
            )
        )
    return boxes


def dataset_entries(dataset: Path, split: str | None) -> list[SheetEntry]:
    import yaml

    names = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = [names[key] for key in sorted(names)] if isinstance(names, dict) else names
    entries = []
    with open(dataset / "manifest.jsonl", encoding="utf-8") as handle:
        for record in (json.loads(line) for line in handle if line.strip()):
            if split and record["split"] != split:
                continue
            image = dataset / record["file"]
            label = dataset / "labels" / record["split"] / f"{Path(record['file']).stem}.txt"
            entries.append(
                SheetEntry(
                    image,
                    record["id"],
                    yolo_boxes(label, names, record["width"], record["height"]),
                )
            )
    return entries


def provenance_entries(provenance: Path) -> list[SheetEntry]:
    entries = []
    with open(provenance, encoding="utf-8") as handle:
        for record in (json.loads(line) for line in handle if line.strip()):
            boxes = [(box["class"], box["xyxy"]) for box in record.get("boxes") or []]
            entries.append(SheetEntry(provenance.parent / record["file"], record["id"], boxes))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="C1 dataset directory or provenance.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="sheet")
    parser.add_argument("--split", help="dataset input: only this split")
    parser.add_argument("--ids", type=Path, help="only these ids (one per line), in this order")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-sheet", type=int, default=DEFAULT_PER_SHEET)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE_PX)
    arguments = parser.parse_args()
    if arguments.input.is_dir():
        entries = dataset_entries(arguments.input, arguments.split)
    else:
        entries = provenance_entries(arguments.input)
    if arguments.ids:
        lines = arguments.ids.read_text(encoding="utf-8").splitlines()
        wanted = [line.split()[0] for line in lines if line.strip()]
        by_id = {entry.caption: entry for entry in entries}
        entries = [by_id[item] for item in wanted if item in by_id]
    if arguments.limit:
        entries = entries[: arguments.limit]
    outputs = render_sheets(
        entries,
        arguments.output_dir,
        arguments.prefix,
        arguments.per_sheet,
        arguments.columns,
        arguments.tile,
    )
    print(f"{len(entries)} images on {len(outputs)} sheets in {arguments.output_dir}")


if __name__ == "__main__":
    main()
