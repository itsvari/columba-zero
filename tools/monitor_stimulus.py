"""Build a reproducible, offline monitor presentation without application dependencies."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

REPOSITORY = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path(__file__).with_suffix(".html")
MANIFEST_PLACEHOLDER = "__STIMULUS_MANIFEST__"
DEFAULT_SEED = 20261027
DEFAULT_FRAME_RATE_HZ = 30
DEFAULT_STATIC_DURATION_S = 4.0
MAXIMUM_STATIC_DURATION_S = 60.0
MINIMUM_FRAME_RATE_HZ = 10
MAXIMUM_FRAME_RATE_HZ = 60
MAXIMUM_REPEAT_COUNT = 1000
MAXIMUM_LOG_RECORDS = 150_000
SCHEMA_VERSION = 2
MINIMUM_DISPLAY_SCALE = 0.4
MAXIMUM_DISPLAY_SCALE = 1.0
MINIMUM_TARGET_SCALE = 0.5
MAXIMUM_TARGET_SCALE = 2.0
CENTRED_ALIGNMENT_FRACTION = 0.5
REFERENCE_POINT_FRACTIONS = (0.1, 0.34, 0.5, 0.66, 0.9)
DESIGN_WIDTH_PX = 1600
DESIGN_HEIGHT_PX = 900
CENTRE_X_PX = DESIGN_WIDTH_PX / 2
CENTRE_Y_PX = DESIGN_HEIGHT_PX / 2
BASE_TARGET_HEIGHT_PX = 180.0
SIMILAR_AREA_RATIO = 1.08
CHALLENGER_AREA_RATIO = 1.6
BRIEF_OCCLUSION_S = 0.3
FULL_OCCLUSION_S = 1.5
IDENTITY_RETENTION_S = 0.75
PARTIAL_COVERED_FRACTION = 0.6
PLAYLISTS = ("tuning", "evaluation")
Playlist = Literal["tuning", "evaluation"]


@dataclass(frozen=True, slots=True)
class Rectangle:
    left_px: float
    top_px: float
    width_px: float
    height_px: float


TARGET_REGION = Rectangle(160, 180, 1280, 540)


@dataclass(frozen=True, slots=True)
class Photograph:
    asset_id: str
    source: str
    source_width_px: int
    source_height_px: int
    crop: Rectangle
    subject: Literal["pigeon", "other_bird"]
    review: str


TUNING_SHEET = "data_sources/contact_sheets/train-v2s-before-after.jpg"
PROFILE_SHEET = "data_sources/contact_sheets/train-v1clean-val-000.jpg"
EVALUATION_SHEET = "data_sources/contact_sheets/test-v2-styles-before-after.jpg"
NEGATIVE_SHEET = "data_sources/contact_sheets/test-v2-000.jpg"
PHOTOGRAPHS = (
    Photograph(
        "tuning-standing-pigeon",
        TUNING_SHEET,
        1000,
        1000,
        Rectangle(25, 777, 200, 204),
        "pigeon",
        "visually verified original photograph, bottom-left tile: grey pigeon on paving",
    ),
    Photograph(
        "tuning-profile-pigeon",
        PROFILE_SHEET,
        1200,
        750,
        Rectangle(168, 620, 121, 125),
        "pigeon",
        "visually verified bottom row, second tile: dark pigeon on paving",
    ),
    Photograph(
        "evaluation-branch-pigeon",
        EVALUATION_SHEET,
        500,
        1000,
        Rectangle(6, 134, 110, 107),
        "pigeon",
        "visually verified original photograph, second row, first tile: pigeon on branch",
    ),
    Photograph(
        "evaluation-ground-pigeon",
        EVALUATION_SHEET,
        500,
        1000,
        Rectangle(16, 264, 96, 96),
        "pigeon",
        "visually verified original photograph, third row, first tile: pigeon on ground",
    ),
    Photograph(
        "shorebird-confuser",
        NEGATIVE_SHEET,
        1200,
        750,
        Rectangle(614, 489, 125, 86),
        "other_bird",
        "coco-186938: visually checked small brown and white shorebird, not pigeon",
    ),
)


@dataclass(frozen=True, slots=True)
class EmbeddedPhotograph:
    photograph: Photograph
    source_sha256: str
    data_url: str


@dataclass(frozen=True, slots=True)
class Pose:
    frame: int
    centre_x_px: float
    centre_y_px: float
    width_px: float
    height_px: float


@dataclass(frozen=True, slots=True)
class Occlusion:
    start_frame: int
    end_frame: int
    covered_fraction: float


@dataclass(frozen=True, slots=True)
class Target:
    identity: str
    asset_id: str
    keyframes: tuple[Pose, ...]
    present_from_frame: int = 0
    present_until_frame: int | None = None
    occlusions: tuple[Occlusion, ...] = ()


@dataclass(frozen=True, slots=True)
class Annotation:
    start_frame: int
    end_frame: int
    selected_identity: str | None
    behaviour: str


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    seed: int
    name: str
    duration_frames: int
    background: Literal["plain", "textured"]
    targets: tuple[Target, ...]
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True, slots=True)
class RegistrationMarker:
    marker_id: str
    bounds: Rectangle
    pattern: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegistrationPoint:
    point_id: str
    centre_x_px: float
    centre_y_px: float


@dataclass(frozen=True, slots=True)
class Manifest:
    schema_version: int
    presentation_id: str
    playlist: Playlist
    seed: int
    frame_rate_hz: int
    repeat_count: int
    display_scale: float
    target_scale: float
    horizontal_position_fraction: float
    vertical_position_fraction: float
    design_width_px: int
    design_height_px: int
    target_region: Rectangle
    registration_markers: tuple[RegistrationMarker, ...]
    registration_points: tuple[RegistrationPoint, ...]
    optical_code: bool
    photo_registration_frame: bool
    maximum_log_records: int
    clock_domain: str
    coordinate_system: str
    scenes: tuple[Scene, ...]
    assets: tuple[EmbeddedPhotograph, ...]
    runtime_sha256: str


def frame_at(seconds: float, frame_rate_hz: int) -> int:
    return round(seconds * frame_rate_hz)


def photograph_for(asset_id: str) -> Photograph:
    return next(photograph for photograph in PHOTOGRAPHS if photograph.asset_id == asset_id)


def pose_at(
    seconds: float,
    frame_rate_hz: int,
    asset_id: str,
    centre_x_px: float = CENTRE_X_PX,
    centre_y_px: float = CENTRE_Y_PX,
    height_px: float = BASE_TARGET_HEIGHT_PX,
) -> Pose:
    crop = photograph_for(asset_id).crop
    return Pose(
        frame_at(seconds, frame_rate_hz),
        centre_x_px,
        centre_y_px,
        height_px * crop.width_px / crop.height_px,
        height_px,
    )


def build_scenes(
    playlist: Playlist,
    seed: int,
    frame_rate_hz: int,
    static_duration_s: float = DEFAULT_STATIC_DURATION_S,
) -> tuple[Scene, ...]:
    primary, challenger = (
        ("tuning-standing-pigeon", "tuning-profile-pigeon")
        if playlist == "tuning"
        else ("evaluation-branch-pigeon", "evaluation-ground-pigeon")
    )
    scenes: list[Scene] = []

    def add_scene(
        name: str,
        duration_s: float,
        targets: tuple[Target, ...],
        annotations: tuple[Annotation, ...],
    ) -> None:
        scene_seed = int.from_bytes(
            hashlib.sha256(f"{playlist}:{seed}:{name}".encode()).digest()[:4], "big"
        )
        scenes.append(
            Scene(
                f"{playlist}-{name}-{scene_seed:08x}",
                scene_seed,
                name,
                frame_at(duration_s, frame_rate_hz),
                "textured" if playlist == "evaluation" else "plain",
                targets,
                annotations,
            )
        )

    def label(duration_s: float, identity: str | None, behaviour: str) -> tuple[Annotation, ...]:
        return (Annotation(0, frame_at(duration_s, frame_rate_hz), identity, behaviour),)

    static_centres = (
        ("centre", 800, 450),
        ("left", 460, 450),
        ("right", 1140, 450),
        ("up", 800, 300),
        ("down", 800, 600),
        ("upper-left", 460, 300),
        ("upper-right", 1140, 300),
        ("lower-left", 460, 600),
        ("lower-right", 1140, 600),
    )
    for name, centre_x_px, centre_y_px in static_centres:
        target = Target(
            "incumbent", primary, (pose_at(0, frame_rate_hz, primary, centre_x_px, centre_y_px),)
        )
        add_scene(
            f"static-{name}",
            static_duration_s,
            (target,),
            label(static_duration_s, "incumbent", "acquire and settle"),
        )

    scene_random = random.Random(f"{playlist}:{seed}:paths")
    vertical_offset_px = scene_random.uniform(-35, 35)
    direction = 1 if playlist == "tuning" else -1
    movement_paths = (
        (
            "stop-reverse",
            (
                (0, 460, 450),
                (1, 460, 450),
                (4, 1140, 450),
                (5, 1140, 450),
                (7, 460, 450),
                (8, 460, 450),
            ),
        ),
        ("diagonal", ((0, 500, 320), (1, 500, 320), (7, 1100, 580), (8, 1100, 580))),
        (
            "curve",
            tuple(
                (
                    index / 2,
                    800 + direction * 300 * math.sin(index * math.pi / 8),
                    450 + 110 * math.sin(index * math.pi / 4) + vertical_offset_px,
                )
                for index in range(17)
            ),
        ),
    )
    for name, path in movement_paths:
        target = Target(
            "incumbent",
            primary,
            tuple(
                pose_at(seconds, frame_rate_hz, primary, centre_x_px, centre_y_px)
                for seconds, centre_x_px, centre_y_px in path
            ),
        )
        add_scene(name, 8, (target,), label(8, "incumbent", "retain through motion"))
    scale_target = Target(
        "incumbent",
        primary,
        tuple(
            pose_at(seconds, frame_rate_hz, primary, height_px=height_px)
            for seconds, height_px in ((0, 140), (1, 140), (4, 280), (5, 280), (8, 140))
        ),
    )
    add_scene("scale", 8, (scale_target,), label(8, "incumbent", "retain through scale changes"))

    incumbent = Target("incumbent", primary, (pose_at(0, frame_rate_hz, primary, 540),))
    primary_crop = photograph_for(primary).crop
    challenger_crop = photograph_for(challenger).crop
    aspect_correction = (primary_crop.width_px / primary_crop.height_px) / (
        challenger_crop.width_px / challenger_crop.height_px
    )
    for name, area_ratio, brief in (
        ("similar-size", SIMILAR_AREA_RATIO, False),
        ("brief-challenger", CHALLENGER_AREA_RATIO, True),
        ("sustained-challenger", CHALLENGER_AREA_RATIO, False),
    ):
        challenger_height_px = BASE_TARGET_HEIGHT_PX * math.sqrt(area_ratio * aspect_correction)
        second = Target(
            "challenger",
            challenger,
            (pose_at(0, frame_rate_hz, challenger, 1060, height_px=challenger_height_px),),
            frame_at(2, frame_rate_hz),
            frame_at(2 + BRIEF_OCCLUSION_S, frame_rate_hz) if brief else None,
        )
        annotations = label(8, "incumbent", "retain incumbent; advantage or duration insufficient")
        if name == "sustained-challenger":
            annotations = (
                Annotation(0, frame_at(2, frame_rate_hz), "incumbent", "acquire incumbent"),
                Annotation(
                    frame_at(2, frame_rate_hz),
                    frame_at(8, frame_rate_hz),
                    "challenger",
                    "switch after confirmation, smoothing and 0.5 s hold",
                ),
            )
        add_scene(name, 8, (incumbent, second), annotations)

    for name, duration_s, covered_fraction in (
        ("partial-occlusion", 0.6, PARTIAL_COVERED_FRACTION),
        ("brief-full-occlusion", BRIEF_OCCLUSION_S, 1.0),
        ("full-occlusion", FULL_OCCLUSION_S, 1.0),
    ):
        start_frame = frame_at(3, frame_rate_hz)
        end_frame = frame_at(3 + duration_s, frame_rate_hz)
        target = Target(
            "incumbent",
            primary,
            (pose_at(0, frame_rate_hz, primary),),
            occlusions=(Occlusion(start_frame, end_frame, covered_fraction),),
        )
        annotations = (
            Annotation(0, start_frame, "incumbent", "acquire incumbent"),
            Annotation(
                start_frame,
                end_frame,
                "incumbent" if duration_s < IDENTITY_RETENTION_S else None,
                "partial visibility"
                if covered_fraction < 1
                else "stale hold; retain briefly or lose after identity timeout",
            ),
            Annotation(end_frame, frame_at(7, frame_rate_hz), "incumbent", "reacquire or retain"),
        )
        add_scene(name, 7, (target,), annotations)

    departure = Target(
        "incumbent",
        primary,
        (
            pose_at(0, frame_rate_hz, primary),
            pose_at(2, frame_rate_hz, primary),
            pose_at(4, frame_rate_hz, primary, 1700),
            pose_at(7, frame_rate_hz, primary, 1700),
        ),
    )
    add_scene(
        "departure",
        7,
        (departure,),
        (
            Annotation(0, frame_at(2, frame_rate_hz), "incumbent", "acquire incumbent"),
            Annotation(
                frame_at(2, frame_rate_hz),
                frame_at(7, frame_rate_hz),
                None,
                "departure, stale hold, loss and home",
            ),
        ),
    )
    add_scene("empty", 4, (), label(4, None, "no pigeon acquisition"))
    confuser = Target(
        "confuser", "shorebird-confuser", (pose_at(0, frame_rate_hz, "shorebird-confuser"),)
    )
    add_scene("confuser", 5, (confuser,), label(5, None, "other bird must not acquire as pigeon"))
    return tuple(scenes)


def build_manifest(
    playlist: Playlist,
    seed: int = DEFAULT_SEED,
    frame_rate_hz: int = DEFAULT_FRAME_RATE_HZ,
    repeat_count: int = 1,
    display_scale: float = MAXIMUM_DISPLAY_SCALE,
    horizontal_position_fraction: float = CENTRED_ALIGNMENT_FRACTION,
    vertical_position_fraction: float = CENTRED_ALIGNMENT_FRACTION,
    registration_grid: bool = False,
    target_scale: float = 1.0,
    optical_code: bool = False,
    static_duration_s: float = DEFAULT_STATIC_DURATION_S,
    photo_registration_frame: bool = False,
) -> Manifest:
    if playlist not in PLAYLISTS:
        raise ValueError("playlist must be tuning or evaluation")
    if not MINIMUM_FRAME_RATE_HZ <= frame_rate_hz <= MAXIMUM_FRAME_RATE_HZ:
        raise ValueError(f"frame rate must be {MINIMUM_FRAME_RATE_HZ}–{MAXIMUM_FRAME_RATE_HZ} Hz")
    if not 1 <= repeat_count <= MAXIMUM_REPEAT_COUNT:
        raise ValueError(f"repeat count must be 1–{MAXIMUM_REPEAT_COUNT}")
    if not math.isfinite(display_scale) or not MINIMUM_DISPLAY_SCALE <= display_scale <= 1:
        raise ValueError("display scale must be finite and between 0.4 and 1")
    for position in (horizontal_position_fraction, vertical_position_fraction):
        if not math.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("display positions must be finite fractions between 0 and 1")
    if (
        not math.isfinite(target_scale)
        or not MINIMUM_TARGET_SCALE <= target_scale <= MAXIMUM_TARGET_SCALE
    ):
        raise ValueError("target scale must be finite and between 0.5 and 2")
    if (
        not math.isfinite(static_duration_s)
        or not 1 <= static_duration_s <= MAXIMUM_STATIC_DURATION_S
    ):
        raise ValueError("static duration must be finite and between 1 and 60 seconds")
    scenes = build_scenes(playlist, seed, frame_rate_hz, static_duration_s)
    scenes = tuple(
        replace(
            scene,
            targets=tuple(
                replace(
                    target,
                    keyframes=tuple(
                        replace(
                            pose,
                            width_px=pose.width_px * target_scale,
                            height_px=pose.height_px * target_scale,
                        )
                        for pose in target.keyframes
                    ),
                )
                for target in scene.targets
            ),
        )
        for scene in scenes
    )
    used_assets = {target.asset_id for scene in scenes for target in scene.targets}
    assets: list[EmbeddedPhotograph] = []
    for photograph in PHOTOGRAPHS:
        if photograph.asset_id not in used_assets:
            continue
        source = (REPOSITORY / photograph.source).read_bytes()
        if not source.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"not a JPEG photograph: {photograph.source}")
        assets.append(
            EmbeddedPhotograph(
                photograph,
                hashlib.sha256(source).hexdigest(),
                "data:image/jpeg;base64," + base64.b64encode(source).decode("ascii"),
            )
        )
    markers = tuple(
        RegistrationMarker(marker_id, Rectangle(left_px, top_px, 64, 64), pattern)
        for marker_id, left_px, top_px, pattern in (
            ("top-left", 24, 24, ("11111", "10001", "10101", "10001", "11111")),
            ("top-right", 1512, 24, ("11111", "11001", "10001", "10001", "11111")),
            ("bottom-left", 24, 812, ("11111", "10011", "10001", "10001", "11111")),
            ("bottom-right", 1512, 812, ("11111", "10001", "10001", "10101", "11111")),
        )
    )
    reference_points = (
        tuple(
            RegistrationPoint(
                f"row-{row + 1}-column-{column + 1}",
                TARGET_REGION.left_px + horizontal_fraction * TARGET_REGION.width_px,
                TARGET_REGION.top_px + vertical_fraction * TARGET_REGION.height_px,
            )
            for row, vertical_fraction in enumerate(REFERENCE_POINT_FRACTIONS)
            for column, horizontal_fraction in enumerate(REFERENCE_POINT_FRACTIONS)
        )
        if registration_grid
        else ()
    )
    manifest = Manifest(
        SCHEMA_VERSION,
        "",
        playlist,
        seed,
        frame_rate_hz,
        repeat_count,
        display_scale,
        target_scale,
        horizontal_position_fraction,
        vertical_position_fraction,
        DESIGN_WIDTH_PX,
        DESIGN_HEIGHT_PX,
        TARGET_REGION,
        markers,
        reference_points,
        optical_code,
        photo_registration_frame,
        MAXIMUM_LOG_RECORDS,
        "browser performance.now() milliseconds; never Pi exposure timestamps",
        "screen content in browser viewport CSS pixels; origin at viewport top-left",
        scenes,
        tuple(assets),
        hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(),
    )
    presentation_id = hashlib.sha256(manifest_json(manifest).encode()).hexdigest()
    return replace(manifest, presentation_id=presentation_id)


def manifest_json(manifest: Manifest) -> str:
    return json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def render_html(manifest: Manifest) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count(MANIFEST_PLACEHOLDER) != 1:
        raise ValueError("monitor template must contain exactly one manifest placeholder")
    return template.replace(MANIFEST_PLACEHOLDER, manifest_json(manifest).replace("<", "\\u003c"))


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="self-contained HTML destination"
    )
    parser.add_argument("--playlist", choices=PLAYLISTS, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frame-rate-hz", type=int, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--display-scale", type=float, default=MAXIMUM_DISPLAY_SCALE)
    parser.add_argument("--horizontal-position", type=float, default=CENTRED_ALIGNMENT_FRACTION)
    parser.add_argument("--vertical-position", type=float, default=CENTRED_ALIGNMENT_FRACTION)
    parser.add_argument("--registration-grid", action="store_true")
    parser.add_argument("--optical-code", action="store_true")
    parser.add_argument("--photo-registration-frame", action="store_true")
    parser.add_argument("--static-duration-s", type=float, default=DEFAULT_STATIC_DURATION_S)
    parser.add_argument("--target-scale", type=float, default=1.0)
    options = parser.parse_args(arguments)
    playlist: Playlist = "tuning" if options.playlist == "tuning" else "evaluation"
    output = Path(options.output)
    if output.suffix.lower() != ".html":
        parser.error("--output must end in .html")
    try:
        manifest = build_manifest(
            playlist,
            options.seed,
            options.frame_rate_hz,
            options.repeat_count,
            options.display_scale,
            options.horizontal_position,
            options.vertical_position,
            options.registration_grid,
            options.target_scale,
            options.optical_code,
            options.static_duration_s,
            options.photo_registration_frame,
        )
        html = render_html(manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        output.with_suffix(".manifest.json").write_text(
            manifest_json(manifest) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    duration_s = sum(scene.duration_frames for scene in manifest.scenes) / manifest.frame_rate_hz
    print(
        f"{output}: {playlist}, seed {manifest.seed}, {len(manifest.scenes)} scenes, "
        f"{duration_s * manifest.repeat_count:g} s; presentation {manifest.presentation_id[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
