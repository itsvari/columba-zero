from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

from app.json_data import JsonValue
from cluster.artifacts import LEADERBOARD_COLUMNS, REQUIRED_ARTIFACTS, mark_done, update_leaderboard
from cluster.hardware import select_batch
from cluster.recipe import load_recipe, resolve_teacher, validate_recipe
from cluster.run_job import (
    Trainer,
    TrainingContext,
    UltralyticsAdapter,
    evaluate,
    evaluation_commands,
    execute,
    publish_report,
    relocate_checkpoint,
)
from cluster.slurm import (
    Budget,
    Job,
    Submission,
    cancel_jobs,
    job_name,
    parse_queue,
    submit_pending,
)
from cluster.staging import stage_dataset
from cluster.sweep import DesiredRun, plan_runs, submission_for

REPOSITORY = Path(__file__).resolve().parents[1]
BASELINE_RECIPE = REPOSITORY / "training/recipes/baseline-repro.yaml"
SMOKE_RECIPE = REPOSITORY / "training/recipes/smoke-5ep.yaml"


def sample_recipe() -> dict[str, JsonValue]:
    return {
        "id": "example",
        "framework": "ultralytics",
        "base_model": "yolo11n.pt",
        "dataset": "columba-train-v1clean",
        "taxonomy": "single",
        "seeds": [0, 1, 2],
        "lane": "nv",
        "gpu_constraint": None,
        "est_gpu_hours": 3,
        "teacher": None,
        "train": {"imgsz": 640, "epochs": 5, "nbs": 64},
        "deploy_sizes": [[384, 640]],
    }


def context_for(root: Path) -> TrainingContext:
    return TrainingContext(
        load_recipe(SMOKE_RECIPE),
        0,
        root / "data/train",
        root / "weights/base.pt",
        None,
        root / "local",
        root / "persistent",
        32,
        "0",
    )


class ClusterTests(unittest.TestCase):
    def test_recipe_examples_and_gpu_mapping(self) -> None:
        recipe = load_recipe(BASELINE_RECIPE)
        self.assertEqual(
            [recipe.constraint_for_seed(seed) for seed in recipe.seeds],
            ["t4", "titanrtx", "titanv"],
        )
        self.assertEqual(recipe.train["patience"], 10)
        self.assertEqual(recipe.train["epochs"], 200)
        self.assertFalse(recipe.allow_missing_evaluation)
        self.assertTrue(load_recipe(SMOKE_RECIPE).allow_missing_evaluation)

    def test_recipe_rejects_bad_paths_and_runner_arguments(self) -> None:
        for field, value in (
            ("id", "../oops"),
            ("base_model", "/tmp/a.pt"),
            ("teacher", "../../best.pt"),
            ("dataset", "columba-test-v2"),
            ("lane", "a100"),
            ("gpu_constraint", "h100"),
            ("seeds", [True]),
            ("seeds", [0, 0]),
            ("allow_missing_evaluation", True),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_recipe({**sample_recipe(), field: cast(JsonValue, value)})
        for argument in ("batch", "data", "distill_model", "resume", "seed"):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                validate_recipe(
                    {**sample_recipe(), "train": {"imgsz": 640, "epochs": 5, argument: "forbidden"}}
                )

    def test_recipe_requires_square_training_and_fixed_nominal_batch(self) -> None:
        for changes in ({"imgsz": [384, 640]}, {"nbs": 32}, {"epochs": 0}, {"save": False}):
            train = {"imgsz": 640, "epochs": 5, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe({**sample_recipe(), "train": cast(dict[str, JsonValue], train)})

    def test_teacher_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                resolve_teacher("teacher/seed2", root),
                (root / "teacher/seed2/weights/best.pt").resolve(),
            )
            self.assertIsNone(resolve_teacher(None, root))
            with self.assertRaises(ValueError):
                resolve_teacher("teacher/../seed2", root)

    def test_memory_class_batch_selection(self) -> None:
        self.assertEqual(select_batch(12, "yolo11n.pt", 640), 32)
        self.assertEqual(select_batch(16, "yolo11n.pt", 640), 48)
        self.assertEqual(select_batch(24, "yolo11n.pt", 640), 64)
        self.assertEqual(select_batch(24, "yolo11x.pt", 960), 3)
        self.assertEqual(select_batch(24, "yolo11n.pt", 640, teacher=True), 32)

    def test_queue_parser_ignores_other_projects_for_budget(self) -> None:
        jobs = parse_queue(
            "1|moonshot-gate|RUNNING|gpu-long|gres/gpu:h100:8\n"
            "2|cz-a|RUNNING|gpu-long|gres/gpu:nv:1\n"
        )
        self.assertTrue(Budget().allows(jobs, "nv"))
        self.assertFalse(jobs[0].owned)

    def test_budget_caps_running_and_reserves_pending_slots(self) -> None:
        gpu_jobs = tuple(
            Job(str(index), f"cz-gpu-{index}", "RUNNING", "gpu-long", "gpu:nv:1")
            for index in range(4)
        )
        cpu_jobs = tuple(
            Job(str(index + 10), f"cz-cpu-{index}", "RUNNING", "normal", "N/A")
            for index in range(2)
        )
        self.assertFalse(Budget().allows(gpu_jobs, "nv"))
        self.assertFalse(Budget().allows(cpu_jobs, "arm"))
        self.assertTrue(Budget().allows(gpu_jobs[:3], "nv"))
        pending = replace(gpu_jobs[-1], state="PENDING")
        self.assertFalse(Budget().allows((*gpu_jobs[:3], pending), "nv"))
        self.assertFalse(Budget(3, 1).allows(gpu_jobs[:3], "nv"))
        all_pending = tuple(
            Job(str(index), f"cz-pending-{index}", "PENDING", "normal", "N/A")
            for index in range(12)
        )
        self.assertFalse(Budget().allows(all_pending, "nv"))
        for values in ((5, 2, 12), (4, 3, 12), (4, 2, 13)):
            with self.assertRaises(ValueError):
                Budget(*values)

    def test_submit_checks_squeue_before_every_sbatch_and_stops_at_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission: Submission = {
                "name": "cz-a",
                "lane": "nv",
                "constraint": "titanv",
                "script": "safe",
                "run_path": str(root / "columba/runs/a"),
            }
            second: Submission = {**submission, "name": "cz-b"}
            calls = [
                Mock(stdout=""),
                Mock(stdout="1001\n"),
                Mock(stdout="1001|cz-a|PENDING|gpu-long|gpu:nv:1\n"),
            ]
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.subprocess.run", side_effect=calls) as process,
            ):
                results = submit_pending([submission, second], Budget(1, 1))
            self.assertEqual([result["status"] for result in results], ["submitted", "budget full"])
            self.assertEqual(
                [call.args[0][0] for call in process.call_args_list], ["squeue", "sbatch", "squeue"]
            )
            self.assertIn("--nice=1000", process.call_args_list[1].args[0])
            self.assertIn("--constraint=titanv", process.call_args_list[1].args[0])

    def test_submit_skips_done_and_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "columba/runs/a"
            run.mkdir(parents=True)
            (run / "DONE").touch()
            submission: Submission = {
                "name": "cz-a",
                "lane": "nv",
                "constraint": None,
                "script": "safe",
                "run_path": str(run),
            }
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.subprocess.run", return_value=Mock(stdout="")) as process,
            ):
                self.assertEqual(submit_pending([submission], Budget())[0]["status"], "done")
                self.assertEqual(process.call_count, 1)

    def test_cancel_never_acts_on_other_jobs(self) -> None:
        jobs = (
            Job("1", "moonshot-gate", "PENDING", "gpu-long", "N/A"),
            Job("2", "cz-smoke", "PENDING", "gpu-long", "gpu:nv:1"),
        )
        with (
            patch("cluster.slurm.queue", return_value=jobs),
            patch("cluster.slurm.subprocess.run") as process,
        ):
            with self.assertRaises(ValueError):
                cancel_jobs(["1", "2"])
            process.assert_not_called()
            self.assertEqual(cancel_jobs(["2"]), ["2"])
            process.assert_called_once_with(["scancel", "2"], check=True)

    def test_plan_diff_and_slurm_templates(self) -> None:
        recipe = load_recipe(BASELINE_RECIPE)
        runs = tuple(DesiredRun(BASELINE_RECIPE, recipe, seed) for seed in recipe.seeds)
        jobs = (Job("1", runs[1].name, "RUNNING", "gpu-long", "gpu:nv:1"),)
        result = plan_runs(runs, {runs[0].relative_run}, jobs)
        self.assertEqual([row["state"] for row in result], ["done", "RUNNING", "missing"])
        submission = submission_for(
            runs[2], Path("/home/user/columba"), Path("/checkout"), "commit"
        )
        self.assertIn("--gres=gpu:nv:1", submission["script"])
        self.assertIn("--nice=1000", submission["script"])
        self.assertEqual(submission["constraint"], "titanv")
        self.assertNotEqual(job_name("x" * 100 + "a", 0), job_name("x" * 100 + "b", 0))

    def test_done_requires_all_artifacts_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(
                json.dumps({"evaluation": "skipped: tools missing", "exit_status": 0})
            )
            with self.assertRaises(RuntimeError):
                mark_done(root)
            self.assertFalse((root / "DONE").exists())
            for artifact in REQUIRED_ARTIFACTS:
                target = root / artifact
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            (root / "run.json").write_text(json.dumps({"evaluation": "complete", "exit_status": 0}))
            mark_done(root)
            self.assertTrue((root / "DONE").is_file())

    def test_missing_metrics_cannot_leave_a_successful_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report: dict[str, JsonValue] = {
                "status": "complete",
                "evaluation": "complete",
                "exit_status": 0,
                "epochs_done": 5,
            }
            status = publish_report(root / "local", root / "persistent", report)
            self.assertEqual(status, 1)
            saved = json.loads((root / "persistent/run.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertFalse((root / "persistent/DONE").exists())

    def test_evaluation_selects_on_val_and_reuses_on_separate_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = context_for(root)
            commands = evaluation_commands(context, root / "data/test", REPOSITORY)
            self.assertIn("--select-threshold", commands[1])
            self.assertNotIn("--select-threshold", commands[3])
            self.assertIn("--threshold-from", commands[3])
            self.assertIn(str(root / "data/test"), commands[2])
            self.assertIn(str(root / "data/train"), commands[0])

    def test_missing_tools_are_allowed_only_for_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = context_for(root)
            self.assertEqual(evaluate(context, root / "test", root), "skipped: tools missing")
            context = replace(context, recipe=load_recipe(BASELINE_RECIPE))
            with self.assertRaises(FileNotFoundError):
                evaluate(context, root / "test", root)

    def test_execute_records_failure_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            persistent = root / "persistent"
            persistent.mkdir()
            scratch = root / "cz-123"
            real_path = Path

            def redirected_path(value: str) -> Path:
                return root if value == "/tmp" else real_path(value)

            with (
                patch("cluster.run_job.Path", side_effect=redirected_path),
                patch(
                    "cluster.run_job.describe_hardware", side_effect=RuntimeError("gpu not allowed")
                ),
            ):
                result = execute(
                    load_recipe(SMOKE_RECIPE),
                    0,
                    root,
                    root,
                    persistent,
                    "checksum",
                    "123",
                )
            self.assertEqual(result, 1)
            report = json.loads((persistent / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(scratch.exists())
            self.assertFalse((persistent / "DONE").exists())

    def test_completed_training_is_not_restarted_for_missing_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = context_for(root)
            (context.persistent_run / "weights").mkdir(parents=True)
            (context.persistent_run / "weights/last.pt").touch()
            (context.persistent_run / "results.csv").write_text("epoch,time\n1,1\n2,2\n")
            report: dict[str, JsonValue] = {}
            with (
                patch("cluster.run_job.checkpoint_complete", return_value=True),
                patch("sys.modules", {"ultralytics": Mock()}),
            ):
                UltralyticsAdapter().train(context, report)
            self.assertEqual(report["training"], "already complete")

    def test_leaderboard_header_upsert_preserves_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaderboard.csv"
            incumbent = dict.fromkeys(LEADERBOARD_COLUMNS, "")
            incumbent.update({"recipe_id": "incumbent", "seed": "0", "status": "complete"})
            candidate = {**incumbent, "recipe_id": "candidate", "status": "failed"}
            update_leaderboard(path, [incumbent, candidate])
            update_leaderboard(path, [{**candidate, "status": "complete"}])
            with path.open() as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames, list(LEADERBOARD_COLUMNS))
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["recipe_id"], "incumbent")
            self.assertEqual(rows[1]["status"], "complete")

    def test_final_validation_callback_does_not_add_a_sixth_training_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = context_for(Path(directory))
            callbacks: dict[str, Callable[[Trainer], None]] = {}
            model = Mock()
            model.add_callback.side_effect = lambda event, callback: callbacks.update(
                {event: callback}
            )

            def train(**arguments: JsonValue) -> None:
                (context.local_run / "weights").mkdir(parents=True)
                (context.local_run / "weights/last.pt").touch()
                (context.local_run / "weights/best.pt").touch()
                (context.local_run / "results.csv").write_text("epoch,time\n1,1\n2,2\n")
                trainer = Mock(batch_size=32, epoch_time=1.0)
                for epoch in (0, 1):
                    trainer.epoch = epoch
                    callbacks["on_model_save"](trainer)
                    callbacks["on_fit_epoch_end"](trainer)
                trainer.epoch = 2
                callbacks["on_fit_epoch_end"](trainer)

            model.train.side_effect = train
            report: dict[str, JsonValue] = {}
            with patch.dict("sys.modules", {"ultralytics": Mock(YOLO=Mock(return_value=model))}):
                UltralyticsAdapter().train(context, report)
            self.assertEqual(report["epochs_done"], 2)
            self.assertEqual(report["seconds_per_epoch"], [1.0, 1.0])

    def test_resume_rewrites_staged_data_teacher_and_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = replace(context_for(root), teacher=root / "new-job/teacher.pt")
            arguments = {
                "data": "/tmp/cz-old/data/data.yaml",
                "distill_model": "/tmp/cz-old/teacher.pt",
            }
            saved = {"epoch": 1, "train_args": arguments, "optimizer": {"retained": True}}
            torch = Mock()
            torch.load.return_value = saved
            with patch.dict("sys.modules", {"torch": torch}):
                relocate_checkpoint(root / "last.pt", context)
            self.assertEqual(arguments["data"], str(context.dataset / "data.yaml"))
            self.assertEqual(arguments["distill_model"], str(context.teacher))
            self.assertEqual(arguments["save_dir"], str(context.local_run))
            self.assertEqual(saved["optimizer"], {"retained": True})
            torch.save.assert_called_once_with(saved, root / "last.pt")

    def test_real_c1_archive_stages_both_datasets_with_new_absolute_roots(self) -> None:
        from cluster.smoke_dataset import build_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dataset(root, "smoke-coco8", ("train", "val"))
            build_dataset(root, "smoke-coco8-test", ("test",))
            training, checksum = stage_dataset("smoke-coco8", root, root / "stage")
            test, _ = stage_dataset("smoke-coco8-test", root, root / "stage")
            self.assertEqual(len(checksum), 64)
            import yaml

            configuration = yaml.safe_load((training / "data.yaml").read_text())
            self.assertEqual(configuration["path"], str(training.resolve()))
            self.assertTrue((training / configuration["test"]).is_dir())
            self.assertEqual(test.parent, training.parent)

    def test_staging_checks_archive_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.tar.zst").write_bytes(b"not an archive")
            (root / "test.tar.zst.sha256").write_text("wrong  test.tar.zst\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                stage_dataset("test", root, root / "stage")


if __name__ == "__main__":
    unittest.main()
