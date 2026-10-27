from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.monitor_stimulus import (
    BRIEF_OCCLUSION_S,
    CHALLENGER_AREA_RATIO,
    FULL_OCCLUSION_S,
    IDENTITY_RETENTION_S,
    PHOTOGRAPHS,
    REPOSITORY,
    SIMILAR_AREA_RATIO,
    Manifest,
    Playlist,
    Scene,
    build_manifest,
    main,
    manifest_json,
    render_html,
)

PLAYLIST_NAMES: tuple[Playlist, ...] = ("tuning", "evaluation")
BROWSER_TIMEOUT_S = 40


def scene_named(manifest: Manifest, name: str) -> Scene:
    return next(scene for scene in manifest.scenes if scene.name == name)


class MonitorStimulusTests(unittest.TestCase):
    def test_registration_grid_has_known_interior_points_and_is_opt_in(self) -> None:
        original = build_manifest("tuning")
        calibrated = build_manifest("tuning", registration_grid=True)
        self.assertEqual(original.registration_points, ())
        self.assertEqual(len(calibrated.registration_points), 25)
        self.assertEqual(len({point.point_id for point in calibrated.registration_points}), 25)
        self.assertNotEqual(original.presentation_id, calibrated.presentation_id)
        self.assertEqual(original.scenes, calibrated.scenes)

    def test_target_scale_preserves_paths_and_area_advantage(self) -> None:
        original = build_manifest("tuning")
        enlarged = build_manifest("tuning", target_scale=1.25)
        for scene, scaled in zip(original.scenes, enlarged.scenes, strict=True):
            for target, larger in zip(scene.targets, scaled.targets, strict=True):
                for pose, resized in zip(target.keyframes, larger.keyframes, strict=True):
                    self.assertEqual(pose.centre_x_px, resized.centre_x_px)
                    self.assertEqual(pose.centre_y_px, resized.centre_y_px)
                    self.assertEqual(pose.frame, resized.frame)
                    self.assertAlmostEqual(pose.width_px * 1.25, resized.width_px)
                    self.assertAlmostEqual(pose.height_px * 1.25, resized.height_px)
        for value in (0, 0.49, 2.01, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "target scale"):
                build_manifest("tuning", target_scale=value)

    def test_display_scale_is_bounded_and_changes_the_presentation_identity(self) -> None:
        original = build_manifest("tuning")
        reduced = build_manifest("tuning", display_scale=0.65)
        self.assertNotEqual(original.presentation_id, reduced.presentation_id)
        self.assertEqual(reduced.display_scale, 0.65)
        self.assertEqual(original.scenes, reduced.scenes)
        for value in (0, 0.39, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "display scale"):
                build_manifest("tuning", display_scale=value)
        for position in (-0.1, 1.1, float("nan")):
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "positions"):
                build_manifest("tuning", horizontal_position_fraction=position)

    def test_identical_inputs_produce_identical_html_and_manifest(self) -> None:
        first = build_manifest("tuning", seed=42)
        second = build_manifest("tuning", seed=42)
        self.assertEqual(manifest_json(first), manifest_json(second))
        self.assertEqual(render_html(first), render_html(second))
        different = build_manifest("tuning", seed=43)
        self.assertNotEqual(first.presentation_id, different.presentation_id)
        self.assertNotEqual(scene_named(first, "curve"), scene_named(different, "curve"))

    def test_tuning_and_evaluation_have_distinct_positive_photographs_and_scene_ids(self) -> None:
        tuning, evaluation = (build_manifest(name) for name in PLAYLIST_NAMES)
        tuning_pigeons = {
            asset.photograph.asset_id
            for asset in tuning.assets
            if asset.photograph.subject == "pigeon"
        }
        evaluation_pigeons = {
            asset.photograph.asset_id
            for asset in evaluation.assets
            if asset.photograph.subject == "pigeon"
        }
        self.assertFalse(tuning_pigeons & evaluation_pigeons)
        self.assertFalse(
            {scene.scene_id for scene in tuning.scenes}
            & {scene.scene_id for scene in evaluation.scenes}
        )
        for manifest in (tuning, evaluation):
            self.assertEqual(
                {asset.photograph.subject for asset in manifest.assets}, {"pigeon", "other_bird"}
            )
            self.assertTrue(all(scene.annotations for scene in manifest.scenes))
            self.assertTrue(
                all(scene.scene_id.startswith(manifest.playlist) for scene in manifest.scenes)
            )
            self.assertEqual(
                len(set(scene.scene_id for scene in manifest.scenes)), len(manifest.scenes)
            )

    def test_photograph_crops_are_within_committed_images_and_record_visual_review(
        self,
    ) -> None:
        for photograph in PHOTOGRAPHS:
            with self.subTest(asset=photograph.asset_id):
                self.assertTrue((REPOSITORY / photograph.source).is_file())
                self.assertGreater(photograph.crop.width_px, 0)
                self.assertGreater(photograph.crop.height_px, 0)
                self.assertLessEqual(
                    photograph.crop.left_px + photograph.crop.width_px, photograph.source_width_px
                )
                self.assertLessEqual(
                    photograph.crop.top_px + photograph.crop.height_px, photograph.source_height_px
                )
                self.assertIn("visual", photograph.review)

    def test_scene_campaign_covers_offsets_motion_scale_negatives_and_bounded_annotations(
        self,
    ) -> None:
        for playlist in PLAYLIST_NAMES:
            manifest = build_manifest(playlist)
            names = {scene.name for scene in manifest.scenes}
            self.assertTrue(
                {
                    "static-centre",
                    "static-left",
                    "static-right",
                    "static-up",
                    "static-down",
                    "static-upper-left",
                    "static-lower-right",
                    "stop-reverse",
                    "diagonal",
                    "curve",
                    "scale",
                    "departure",
                    "empty",
                    "confuser",
                }
                <= names
            )
            self.assertEqual(scene_named(manifest, "empty").targets, ())
            for scene in manifest.scenes:
                for annotation in scene.annotations:
                    self.assertGreaterEqual(annotation.start_frame, 0)
                    self.assertLess(annotation.start_frame, annotation.end_frame)
                    self.assertLessEqual(annotation.end_frame, scene.duration_frames)
                for target in scene.targets:
                    frames = tuple(pose.frame for pose in target.keyframes)
                    self.assertEqual(frames, tuple(sorted(set(frames))))
                    self.assertTrue(
                        all(pose.width_px > 0 and pose.height_px > 0 for pose in target.keyframes)
                    )

    def test_challenger_advantage_and_duration_are_independent_of_photo_aspect_ratio(self) -> None:
        for playlist in PLAYLIST_NAMES:
            manifest = build_manifest(playlist)
            for name, expected_ratio in (
                ("similar-size", SIMILAR_AREA_RATIO),
                ("brief-challenger", CHALLENGER_AREA_RATIO),
                ("sustained-challenger", CHALLENGER_AREA_RATIO),
            ):
                scene = scene_named(manifest, name)
                primary, challenger = (target.keyframes[0] for target in scene.targets)
                ratio = (
                    challenger.width_px
                    * challenger.height_px
                    / (primary.width_px * primary.height_px)
                )
                self.assertAlmostEqual(ratio, expected_ratio)
            brief = scene_named(manifest, "brief-challenger").targets[1]
            assert brief.present_until_frame is not None
            self.assertLess(
                (brief.present_until_frame - brief.present_from_frame) / manifest.frame_rate_hz, 0.5
            )
            sustained = scene_named(manifest, "sustained-challenger").targets[1]
            self.assertIsNone(sustained.present_until_frame)

    def test_occlusion_cases_straddle_identity_retention_and_restore_same_target(self) -> None:
        manifest = build_manifest("tuning")
        brief = scene_named(manifest, "brief-full-occlusion").targets[0].occlusions[0]
        full = scene_named(manifest, "full-occlusion").targets[0].occlusions[0]
        self.assertEqual(
            (brief.end_frame - brief.start_frame) / manifest.frame_rate_hz, BRIEF_OCCLUSION_S
        )
        self.assertEqual(
            (full.end_frame - full.start_frame) / manifest.frame_rate_hz, FULL_OCCLUSION_S
        )
        self.assertLess(BRIEF_OCCLUSION_S, IDENTITY_RETENTION_S)
        self.assertGreater(FULL_OCCLUSION_S, IDENTITY_RETENTION_S)
        self.assertEqual((brief.covered_fraction, full.covered_fraction), (1, 1))

    def test_cli_runs_from_another_directory_and_writes_embedded_offline_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "monitor.html"
            process = subprocess.run(
                (
                    sys.executable,
                    str(REPOSITORY / "tools/monitor_stimulus.py"),
                    "--playlist",
                    "evaluation",
                    "--output",
                    str(output),
                ),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            html = output.read_text(encoding="utf-8")
            self.assertIn("data:image/jpeg;base64,", html)
            self.assertNotIn("__STIMULUS_MANIFEST__", html)
            self.assertNotIn("<script src=", html)
            self.assertNotIn("fetch(", html)
            self.assertEqual(
                output.with_suffix(".manifest.json").read_text(encoding="utf-8"),
                manifest_json(build_manifest("evaluation")) + "\n",
            )

    def test_cli_rejects_invalid_arguments_before_writing(self) -> None:
        for options in (
            ("--frame-rate-hz", "1"),
            ("--frame-rate-hz", "61"),
            ("--repeat-count", "0"),
            ("--repeat-count", "1001"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "test.html"
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(("--playlist", "tuning", "--output", str(output), *options))
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())

    @unittest.skipUnless(
        os.environ.get("COLUMBA_TEST_BROWSER") and shutil.which("node"),
        "set COLUMBA_TEST_BROWSER to a Chromium-compatible executable; needs Node",
    )
    def test_generated_html_behaviour_in_a_real_browser(self) -> None:
        driver = (REPOSITORY / "tests/monitor_stimulus_browser.js").read_text(encoding="utf-8")
        browser = os.environ["COLUMBA_TEST_BROWSER"]
        cases: tuple[tuple[Playlist, float], ...] = (
            ("tuning", 1.0),
            ("evaluation", 1.0),
            ("tuning", 0.6),
            ("evaluation", 0.6),
        )
        for playlist, display_scale in cases:
            with (
                self.subTest(playlist=playlist, display_scale=display_scale),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "monitor.html"
                manifest = build_manifest(
                    playlist,
                    repeat_count=2,
                    display_scale=display_scale,
                    horizontal_position_fraction=0.8,
                    vertical_position_fraction=0.1,
                    registration_grid=display_scale < 1,
                )
                html = render_html(manifest).replace("</html>", f"<script>{driver}</script></html>")
                output.write_text(html, encoding="utf-8")
                process = subprocess.run(
                    (
                        "node",
                        str(REPOSITORY / "tests/run_monitor_browser.mjs"),
                        browser,
                        str(output),
                        str(Path(directory) / "browser-profile"),
                    ),
                    capture_output=True,
                    text=True,
                    timeout=BROWSER_TIMEOUT_S,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn('"export_count":', process.stdout)
                self.assertIn('"checks":[', process.stdout)
                self.assertIn(json.dumps(playlist), process.stdout)


if __name__ == "__main__":
    unittest.main()
