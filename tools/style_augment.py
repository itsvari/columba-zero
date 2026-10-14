#!/usr/bin/env python
"""Deterministic camera-style copies of labelled images (PLAN.md 5.1).

    python -m tools.style_augment demo ~/columba-data/columba-test-v2 \
        --output-dir ~/columba-data/contact_sheets/styles-demo --count 12

Styles (`apply_style`), each seeded by (seed, image id, style) so a rebuild gives identical
bytes:

- `noir_night`: NoIR camera at night with an 850 nm illuminator. Greyscale from a red/green
  weighted luminance (foliage reflects near-infrared), darker exposure (gamma), radial
  illuminator falloff around a jittered centre, short linear motion blur, and Gaussian +
  signal-dependent sensor noise. Boxes are unchanged.
- `noir_day`: NoIR camera in daylight without an IR-cut filter. Magenta cast, lower
  saturation, and foliage (green-dominant pixels) pushed towards bright pink-white.
  Boxes are unchanged.
- `wide`: Camera Module 3 Wide geometry. The image is shrunk (zoom-out, so birds become
  small) and pasted with a feathered edge onto a bird-free canvas scene (an empty scene from
  the same split) whose brightness is matched to the image, then barrel-distorted. Boxes are
  scaled, then remapped through the distortion by mapping points along each box edge and
  taking their bounding box, then clipped. Without a canvas only the distortion is applied.

Barrel model, in coordinates centred on the image and divided by the half-diagonal (corners
at r = 1): an output pixel at radius r_d samples the input at
r_s = r_d (1 + k r_d²) / (1 + k), k > 0. Corners stay fixed, the centre is magnified, and
content near the middle of the edges is pushed out of frame. Boxes go the other way:
r_d solves r_d + k r_d³ = r_s (1 + k) (Newton's method; monotonic for k ≥ 0).
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy

STYLES = ("noir_night", "noir_day", "wide")
EDGE_SAMPLES = 16
MIN_BOX_PX = 2.0
NEWTON_STEPS = 12
JPEG_QUALITY = 92
FEATHER_FRACTION = 0.12

Record = dict[str, Any]
FloatArray = numpy.ndarray[Any, numpy.dtype[numpy.float64]]


def style_rng(seed: int, image_id: str, style: str) -> numpy.random.Generator:
    digest = hashlib.sha256(f"{seed}:{image_id}:{style}".encode()).digest()
    return numpy.random.default_rng(int.from_bytes(digest[:8], "little"))


# --- geometry -------------------------------------------------------------------------------


def _normalise(width: int, height: int) -> tuple[float, float, float]:
    centre_x, centre_y = width / 2, height / 2
    return centre_x, centre_y, math.hypot(centre_x, centre_y)


def barrel_source_coordinates(
    x: FloatArray, y: FloatArray, width: int, height: int, k: float
) -> tuple[FloatArray, FloatArray]:
    """Output pixel coordinates → input coordinates they sample."""
    centre_x, centre_y, scale = _normalise(width, height)
    u, v = (x - centre_x) / scale, (y - centre_y) / scale
    factor = (1 + k * (u * u + v * v)) / (1 + k)
    return centre_x + u * factor * scale, centre_y + v * factor * scale


def barrel_destination_points(points: FloatArray, width: int, height: int, k: float) -> FloatArray:
    """Input points (N×2) → where they appear in the distorted output (inverse of the above)."""
    centre_x, centre_y, scale = _normalise(width, height)
    u = (points[:, 0] - centre_x) / scale
    v = (points[:, 1] - centre_y) / scale
    r_source = numpy.hypot(u, v)
    target = r_source * (1 + k)
    r_dest = r_source.copy()
    for _ in range(NEWTON_STEPS):
        r_dest -= (r_dest + k * r_dest**3 - target) / (1 + 3 * k * r_dest**2)
    ratio = numpy.divide(r_dest, r_source, out=numpy.ones_like(r_dest), where=r_source > 0)
    return numpy.stack([centre_x + u * ratio * scale, centre_y + v * ratio * scale], axis=1)


def remap_bilinear(image: FloatArray, source_x: FloatArray, source_y: FloatArray) -> FloatArray:
    """Sample an H×W×C float image at fractional coordinates (edge pixels repeat)."""
    height, width = image.shape[:2]
    x = numpy.clip(source_x, 0, width - 1)
    y = numpy.clip(source_y, 0, height - 1)
    x0 = numpy.floor(x).astype(numpy.int64)
    y0 = numpy.floor(y).astype(numpy.int64)
    x1 = numpy.minimum(x0 + 1, width - 1)
    y1 = numpy.minimum(y0 + 1, height - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    result: FloatArray = top * (1 - fy) + bottom * fy
    return result


def box_edge_points(xyxy: Sequence[float]) -> FloatArray:
    x1, y1, x2, y2 = xyxy
    steps = numpy.linspace(0.0, 1.0, EDGE_SAMPLES)
    xs = x1 + (x2 - x1) * steps
    ys = y1 + (y2 - y1) * steps
    return numpy.concatenate(
        [
            numpy.stack([xs, numpy.full_like(xs, y1)], axis=1),
            numpy.stack([xs, numpy.full_like(xs, y2)], axis=1),
            numpy.stack([numpy.full_like(ys, x1), ys], axis=1),
            numpy.stack([numpy.full_like(ys, x2), ys], axis=1),
        ]
    )


def remap_box(xyxy: Sequence[float], width: int, height: int, k: float) -> list[float] | None:
    """Box through the barrel distortion, clipped to the frame; None if (almost) out of frame."""
    mapped = barrel_destination_points(box_edge_points(xyxy), width, height, k)
    x1 = min(max(float(numpy.min(mapped[:, 0])), 0.0), float(width))
    x2 = min(max(float(numpy.max(mapped[:, 0])), 0.0), float(width))
    y1 = min(max(float(numpy.min(mapped[:, 1])), 0.0), float(height))
    y2 = min(max(float(numpy.max(mapped[:, 1])), 0.0), float(height))
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def barrel(image: FloatArray, k: float) -> FloatArray:
    height, width = image.shape[:2]
    y, x = numpy.mgrid[0:height, 0:width].astype(numpy.float64)
    source_x, source_y = barrel_source_coordinates(x + 0.5, y + 0.5, width, height, k)
    return remap_bilinear(image, source_x - 0.5, source_y - 0.5)


# --- styles ---------------------------------------------------------------------------------


def to_float(image: Any) -> FloatArray:
    return numpy.asarray(image.convert("RGB"), dtype=numpy.float64)


def to_image(array: FloatArray) -> Any:
    from PIL import Image

    return Image.fromarray(numpy.clip(numpy.rint(array), 0, 255).astype(numpy.uint8), "RGB")


def motion_blur(array: FloatArray, length: int, angle: float) -> FloatArray:
    if length <= 1:
        return array
    total = numpy.zeros_like(array)
    for step in range(length):
        offset = step - (length - 1) / 2
        dx, dy = round(offset * math.cos(angle)), round(offset * math.sin(angle))
        total += numpy.roll(array, (dy, dx), axis=(0, 1))
    result: FloatArray = total / length
    return result


def noir_night(array: FloatArray, rng: numpy.random.Generator) -> tuple[FloatArray, Record]:
    height, width = array.shape[:2]
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    grey = (0.45 * red + 0.45 * green + 0.10 * blue) / 255
    gamma = float(rng.uniform(1.3, 2.2))
    centre_x = width * float(rng.uniform(0.35, 0.65))
    centre_y = height * float(rng.uniform(0.35, 0.65))
    falloff = float(rng.uniform(1.0, 3.5))
    y, x = numpy.mgrid[0:height, 0:width].astype(numpy.float64)
    radius2 = ((x - centre_x) ** 2 + (y - centre_y) ** 2) / (width**2 + height**2) * 4
    illumination = 1 / (1 + falloff * radius2)
    gain = float(rng.uniform(0.7, 1.1))
    lit = grey**gamma * illumination * gain
    length = max(1, round(float(rng.uniform(0, 0.012)) * max(width, height)))
    angle = float(rng.uniform(0, math.pi))
    lit = motion_blur(lit[..., None], length, angle)[..., 0]
    sigma = float(rng.uniform(0.015, 0.05))
    noise = rng.normal(0.0, 1.0, lit.shape) * (sigma + 0.04 * numpy.sqrt(numpy.maximum(lit, 0)))
    value = numpy.clip(lit + noise, 0, 1) * 255
    params = {
        "gamma": round(gamma, 3),
        "falloff": round(falloff, 3),
        "gain": round(gain, 3),
        "centre": [round(centre_x, 1), round(centre_y, 1)],
        "blur_px": length,
        "noise_sigma": round(sigma, 4),
    }
    return numpy.repeat(value[..., None], 3, axis=2), params


def noir_day(array: FloatArray, rng: numpy.random.Generator) -> tuple[FloatArray, Record]:
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    foliage = numpy.clip((green - numpy.maximum(red, blue)) / 30, 0, 1)
    luminance = 0.3 * red + 0.59 * green + 0.11 * blue
    saturation = float(rng.uniform(0.35, 0.6))
    cast = numpy.array([rng.uniform(1.15, 1.3), rng.uniform(0.72, 0.85), rng.uniform(1.05, 1.2)])
    base = (luminance[..., None] + saturation * (array - luminance[..., None])) * cast
    glow = float(rng.uniform(0.75, 0.95))
    pink = numpy.array([255.0, 205.0, 235.0]) * (0.55 + 0.45 * luminance[..., None] / 255)
    result = base * (1 - glow * foliage[..., None]) + pink * glow * foliage[..., None]
    haze = float(rng.uniform(0.0, 0.12))
    result = result * (1 - haze) + 255 * haze
    params = {
        "saturation": round(saturation, 3),
        "cast": [round(float(value), 3) for value in cast],
        "foliage_glow": round(glow, 3),
        "haze": round(haze, 3),
        "foliage_fraction": round(float(foliage.mean()), 3),
    }
    return result, params


def zoom_out(
    array: FloatArray,
    boxes: Sequence[Record],
    canvas: FloatArray,
    rng: numpy.random.Generator,
    scale_range: tuple[float, float] = (0.3, 0.6),
) -> tuple[FloatArray, list[Record], Record]:
    from PIL import Image

    height, width = array.shape[:2]
    scale = float(rng.uniform(*scale_range))
    small_w, small_h = max(1, round(width * scale)), max(1, round(height * scale))
    small = to_float(to_image(array).resize((small_w, small_h), Image.Resampling.LANCZOS))
    background = to_float(to_image(canvas).resize((width, height), Image.Resampling.BILINEAR))
    background = background * (array.mean() / max(float(background.mean()), 1.0))
    left = int(rng.integers(0, width - small_w + 1))
    top = int(rng.integers(0, height - small_h + 1))
    feather = max(1.0, FEATHER_FRACTION * min(small_w, small_h))
    ys = numpy.arange(small_h, dtype=numpy.float64)
    xs = numpy.arange(small_w, dtype=numpy.float64)
    ramp_y = numpy.clip(numpy.minimum(ys + 0.5, small_h - ys - 0.5) / feather, 0, 1)
    ramp_x = numpy.clip(numpy.minimum(xs + 0.5, small_w - xs - 0.5) / feather, 0, 1)
    alpha = (ramp_y[:, None] * ramp_x[None, :])[..., None]
    result = background.copy()
    region = result[top : top + small_h, left : left + small_w]
    result[top : top + small_h, left : left + small_w] = alpha * small + (1 - alpha) * region
    sx, sy = small_w / width, small_h / height
    moved = [
        {
            **box,
            "xyxy": [
                left + box["xyxy"][0] * sx,
                top + box["xyxy"][1] * sy,
                left + box["xyxy"][2] * sx,
                top + box["xyxy"][3] * sy,
            ],
        }
        for box in boxes
    ]
    return result, moved, {"zoom": round(scale, 4), "offset": [left, top]}


def wide(
    array: FloatArray,
    boxes: Sequence[Record],
    rng: numpy.random.Generator,
    canvas: FloatArray | None,
) -> tuple[FloatArray, list[Record], Record]:
    params: Record = {}
    current = list(boxes)
    if canvas is not None:
        array, current, params = zoom_out(array, current, canvas, rng)
    height, width = array.shape[:2]
    k = float(rng.uniform(0.12, 0.3))
    distorted = barrel(array, k)
    remapped = []
    lost = 0
    for box in current:
        xyxy = remap_box(box["xyxy"], width, height, k)
        if xyxy is None:
            lost += 1
        else:
            remapped.append({**box, "xyxy": xyxy})
    params.update({"barrel_k": round(k, 4), "boxes_out_of_frame": lost})
    return distorted, remapped, params


def apply_style(
    image: Any,
    boxes: Sequence[Record],
    style: str,
    seed: int,
    image_id: str,
    canvas: Any | None = None,
) -> tuple[Any, list[Record], Record]:
    """Return (PIL image, boxes in pixels, parameters) for one style copy."""
    rng = style_rng(seed, image_id, style)
    array = to_float(image)
    if style == "noir_night":
        result, params = noir_night(array, rng)
        new_boxes = [dict(box) for box in boxes]
    elif style == "noir_day":
        result, params = noir_day(array, rng)
        new_boxes = [dict(box) for box in boxes]
    elif style == "wide":
        canvas_array = to_float(canvas) if canvas is not None else None
        result, new_boxes, params = wide(array, boxes, rng, canvas_array)
    else:
        raise ValueError(f"unknown style {style!r}")
    return to_image(result), new_boxes, {"style": style, "seed": seed, **params}


def choose_style(weights: dict[str, float], seed: int, image_id: str) -> str:
    """Deterministic weighted choice of one style for an image."""
    names = sorted(weights)
    rng = style_rng(seed, image_id, "choose")
    total = sum(weights[name] for name in names)
    pick = float(rng.uniform(0, total))
    for name in names:
        pick -= weights[name]
        if pick <= 0:
            return name
    return names[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="before/after contact sheets for a C1 dataset")
    demo.add_argument("dataset", type=Path)
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--count", type=int, default=12)
    demo.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    import json

    from PIL import Image, ImageOps

    from tools.contact_sheet import SheetEntry, dataset_entries, render_sheets

    manifest = arguments.dataset / "manifest.jsonl"
    empty_ids = {
        record["id"]
        for record in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())
        if not record["classes_present"]
    }
    entries = [entry for entry in dataset_entries(arguments.dataset, None) if entry.boxes]
    empties = [
        entry for entry in dataset_entries(arguments.dataset, None) if entry.caption in empty_ids
    ]
    chosen = entries[:: max(1, len(entries) // arguments.count)][: arguments.count]
    output = arguments.output_dir
    (output / "images").mkdir(parents=True, exist_ok=True)
    sheet: list[SheetEntry] = []
    for index, entry in enumerate(chosen):
        with Image.open(entry.image) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        boxes = [{"class": name, "xyxy": xyxy} for name, xyxy in entry.boxes]
        sheet.append(SheetEntry(entry.image, f"{index} original", entry.boxes))
        canvas = None
        if empties:
            with Image.open(empties[index % len(empties)].image) as opened:
                canvas = opened.convert("RGB")
        for style in STYLES:
            styled, new_boxes, params = apply_style(
                image, boxes, style, arguments.seed, entry.caption, canvas
            )
            path = output / "images" / f"{index:03d}-{style}.jpg"
            styled.save(path, quality=JPEG_QUALITY)
            sheet.append(
                SheetEntry(
                    path,
                    f"{index} {style}",
                    [(box["class"], box["xyxy"]) for box in new_boxes],
                )
            )
            print(json.dumps({"image": entry.caption, **params}))
    print(render_sheets(sheet, output, "styles", per_sheet=24, columns=4, tile_px=360))


if __name__ == "__main__":
    main()
