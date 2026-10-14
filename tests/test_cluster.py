from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import yaml

from app.json_data import JsonValue
from cluster.arrays import (
    active_run_names,
    group_tasks,
    shares,
    submit_arrays,
    task_for,
    throttle_arrays,
)
from cluster.artifacts import (
    LEADERBOARD_COLUMNS,
    REQUIRED_ARTIFACTS,
    mark_done,
    repository_relative,
    update_leaderboard,
)
from cluster.deployment import READY_FILE, publish_archive
from cluster.environment import environment_for
from cluster.gpu_policy import available_request, load_policy
from cluster.gpu_selection import GPU_CONSTRAINTS, gpu_inventory, parse_inventory, request_for
from cluster.hardware import allowed_gpu, select_batch
from cluster.recipe import load_recipe, resolve_teacher, validate_recipe
from cluster.requeue import MAX_REQUEUES, update_ledger
from cluster.run_job import (
    Trainer,
    TrainingContext,
    UltralyticsAdapter,
    evaluate,
    evaluation_commands,
    execute,
    publish_report,
    relocate_checkpoint,
    run,
)
from cluster.slurm import (
    CPU_RUNNING_LIMIT,
    GPU_RUNNING_LIMIT,
    QUEUED_LIMIT,
    Budget,
    Job,
    Submission,
    cancel_jobs,
    gpu_resource_arguments,
    job_name,
    parse_queue,
    submit_pending,
)
from cluster.staging import stage_dataset
from cluster.state import snapshot
from cluster.sweep import (
    DesiredRun,
    excluded_nodes,
    fetch,
    plan_runs,
    probe_submission,
    submission_for,
)
from cluster.sweep import (
    main as sweep_main,
)
from cluster.transitions import (
    Candidate,
    HoldRequired,
    Rules,
    arm_transition,
    completed_metric,
    execute_transition,
    generate_recipes,
    load_rules,
    recover_interrupted_inputs,
    select_candidates,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BASELINE_RECIPE = REPOSITORY / "training/recipes/baseline-repro.yaml"
SMOKE_RECIPE = REPOSITORY / "training/recipes/smoke-5ep.yaml"

GPU_INVENTORY = """gpu-long|xgpd[2-3]|gpu:nv:2|cuda70,titanv
gpu-long|xgpe4|gpu:nv:1|cuda75,titanrtx
gpu-long|xgpf[6-11]|gpu:nv:1|cuda75,t4
gpu-long|xgpg[4-7]|gpu:a100-40:1|cuda80,a100
gpu-long|xgph[8-9]|gpu:a100-80:1|cuda80,a100
gpu-long|xgpi[11-12]|gpu:h100-47:4|cuda90,h100
gpu-long|xgpi[6-10]|gpu:h100-96:2|cuda90,h100
gpu|xgpg[4-7]|gpu:a100-40:1|cuda80,a100
gpu|xgpj0|gpu:a100-80:1|cuda80,a100,xgpj
gpu|xgpk0|gpu:h200-141:4|cuda80,h200
"""


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
            ("gpu_constraint", "v100"),
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

    def test_new_gpu_recipe_selectors_and_per_seed_overrides(self) -> None:
        for choice in GPU_CONSTRAINTS:
            with self.subTest(choice=choice):
                recipe = validate_recipe({**sample_recipe(), "gpu_constraint": choice})
                self.assertEqual(recipe.constraint_for_seed(0), choice)
        recipe = validate_recipe(
            {
                **sample_recipe(),
                "seed_gpu_constraints": cast(JsonValue, {0: "a100-80", 1: "h100-47", 2: "h200"}),
            }
        )
        self.assertEqual(recipe.constraint_for_seed(2), "h200")
        any_seed = validate_recipe(
            {
                **sample_recipe(),
                "gpu_constraint": "a100-80",
                "seed_gpu_constraints": cast(JsonValue, {0: None}),
            }
        )
        self.assertIsNone(any_seed.constraint_for_seed(0))
        self.assertEqual(any_seed.constraint_for_seed(1), "a100-80")

    def test_sinfo_inventory_and_every_gpu_request(self) -> None:
        with patch(
            "cluster.gpu_selection.subprocess.run", return_value=Mock(stdout=GPU_INVENTORY)
        ) as process:
            offers = gpu_inventory()
        self.assertEqual(process.call_args.args[0][0], "sinfo")
        expected = {
            "t4": ("gpu:nv:1", "t4"),
            "titanrtx": ("gpu:nv:1", "titanrtx"),
            "titanv": ("gpu:nv:1", "titanv"),
            "a100-40": ("gpu:a100-40:1", None),
            "a100-80": ("gpu:a100-80:1", None),
            "h100-47": ("gpu:h100-47:1", None),
            "h100-96": ("gpu:h100-96:1", None),
            "h200": ("gpu:h200-141:1", None),
            "a100": ("gpu:a100-40:1", None),
            "h100": ("gpu:h100-47:1", None),
        }
        for choice, (resources, feature) in expected.items():
            with self.subTest(choice=choice):
                request = request_for(choice, offers)
                self.assertEqual(request.resources, resources)
                self.assertEqual(request.constraint, feature)
                self.assertEqual(
                    request.partitions, ("gpu",) if choice == "h200" else ("gpu-long",)
                )
                self.assertEqual(
                    request.time_limit, "03:00:00" if choice == "h200" else "3-00:00:00"
                )
        self.assertTrue(any(offer.node == "xgpf10" for offer in offers))
        self.assertTrue(any(offer.node == "xgpj0" for offer in offers))

    def test_automatic_selection_requires_policy_and_pins_arm_gres(self) -> None:
        offers = parse_inventory(GPU_INVENTORY)
        with self.assertRaises(ValueError):
            request_for(None, offers)
        arm = request_for("a100-80", offers, required_node="xgpj0")
        self.assertEqual(arm.resources, "gpu:a100-80:1")
        self.assertEqual(arm.partitions, ("gpu",))
        self.assertEqual(arm.time_limit, "03:00:00")
        self.assertEqual(request_for(None, offers, required_node="xgpj0"), arm)
        with self.assertRaises(ValueError):
            request_for("h100", offers, required_node="xgpj0")
        with self.assertRaises(ValueError):
            request_for("a100-80", offers, required_node="xgpj0", excluded_nodes=["xgpj0"])

    def test_budget_counts_devices_across_types_and_nodes(self) -> None:
        jobs = parse_queue(
            "1|cz-a|RUNNING|gpu-long|gres/gpu:a100-80:2|2\n"
            "2|cz-b|PENDING|gpu|gres/gpu:h100-47:3|1\n"
            "3|moonshot-idle|RUNNING|gpu|gres/gpu:h200-141:8|1\n"
        )
        self.assertEqual(jobs[0].gpu_count, 4)
        self.assertEqual(jobs[1].gpu_count, 3)
        self.assertTrue(Budget().allows(jobs, "nv"))
        self.assertFalse(Budget().allows(jobs, "nv", gpus_requested=2))
        self.assertTrue(Budget().allows(jobs, "arm"))
        self.assertFalse(
            Budget().allows((*jobs, Job("4", "cz-c", "RUNNING", "gpu", "gpu:1")), "nv")
        )
        self.assertFalse(Budget().allows([Job("5", "cz-unknown", "RUNNING", "gpu", "N/A")], "nv"))

    def test_scheduler_emits_typed_and_untyped_requests_without_shell_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for choice in (*GPU_CONSTRAINTS, None):
                submission: Submission = {
                    "name": "cz-request",
                    "lane": "nv",
                    "constraint": choice,
                    "script": "unchanged script",
                    "run_path": str(root / "columba/runs/a"),
                }
                with (
                    self.subTest(choice=choice),
                    patch("cluster.slurm.Path.home", return_value=root),
                    patch(
                        "cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)
                    ),
                    patch(
                        "cluster.slurm.subprocess.run",
                        side_effect=[Mock(stdout=""), Mock(stdout="123\n")],
                    ) as process,
                ):
                    self.assertEqual(
                        submit_pending([submission], Budget())[0]["status"], "submitted"
                    )
                command = process.call_args.args[0]
                request = request_for(choice or "t4", parse_inventory(GPU_INVENTORY))
                self.assertTrue(set(request.arguments()).issubset(command))
                self.assertIn("--nice=1000", command)
                self.assertIn("--nodes=1", command)
                self.assertEqual(process.call_args.kwargs["input"], "unchanged script")

    def test_policy_prefers_free_capacity_and_keeps_arm_available(self) -> None:
        policy = load_policy()
        offers = parse_inventory(
            "gpu-long xgph8 gpu:a100-80:1 gpu:a100-80:1(IDX:0) cuda80,a100 mixed\n"
            "gpu-long xgpi19 gpu:h100-47:4 gpu:h100-47:3(IDX:0-2) cuda90,h100 mixed\n"
            "gpu xgpj0 gpu:a100-80:1 gpu:a100-80:0(IDX:N/A) cuda80,a100 mixed\n"
            "gpu xgpf2 gpu:nv:1 gpu:nv:0(IDX:N/A) cuda75,t4 idle\n"
            "gpu-long xgpe6 gpu:nv:1 gpu:nv:0(IDX:N/A) cuda75,titanrtx idle\n"
            "gpu xgpf0 gpu:nv:1 gpu:nv:0(IDX:N/A) cuda75,t4 reboot^\n"
        )
        light = available_request(
            offers, policy.preferences("yolo11n.pt", 640, False, 1), excluded_nodes=["xgpe6"]
        )
        self.assertIsNotNone(light)
        assert light is not None
        self.assertEqual(light[0].resources, "gpu:nv:1")
        self.assertEqual(light[0].constraint, "t4")
        self.assertEqual(light[0].time_limit, "03:00:00")
        self.assertEqual(light[1], "xgpf2")
        heavy = available_request(offers, policy.preferences("yolo11x.pt", 640, False, 1))
        assert heavy is not None
        self.assertEqual(heavy[0].resources, "gpu:h100-47:1")
        arm = available_request(offers, ["xgpj0"])
        assert arm is not None
        self.assertEqual(arm[1], "xgpj0")
        self.assertEqual(arm[0].time_limit, "03:00:00")
        self.assertIsNone(available_request(offers, ["a100-40"]))
        self.assertIsNone(available_request(offers, ["t4"], allow_short_partition=False))
        for model, size, teacher, hours in (
            ("yolo11s.pt", 800, False, 1),
            ("yolo11n.pt", 640, True, 1),
            ("yolo11n.pt", 640, False, 5),
        ):
            self.assertEqual(policy.preferences(model, size, teacher, hours)[0], "h100-47")
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "policy.json"
            invalid.write_text('{"light_preferences": ["v100"]}')
            with self.assertRaises(ValueError):
                load_policy(invalid)

    def test_automatic_multiple_free_nodes_selects_one_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission: Submission = {
                "name": "cz-one-node",
                "lane": "nv",
                "constraint": None,
                "script": "script",
                "run_path": str(root / "columba/runs/example"),
            }
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch("cluster.slurm.queue", return_value=()),
                patch("cluster.slurm.subprocess.run", return_value=Mock(stdout="123\n")) as process,
            ):
                submit_pending([submission], Budget())
            command = process.call_args.args[0]
            node = next(argument for argument in command if argument.startswith("--nodelist="))
            self.assertEqual(node, "--nodelist=xgpf6")
            self.assertNotIn(",", node)
            self.assertIn("--nodes=1", command)
            self.assertIn("--gres=gpu:nv:1", command)

    def test_automatic_batch_does_not_reuse_pending_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offers = parse_inventory("gpu-long xgpf6 gpu:nv:1 gpu:nv:0(IDX:N/A) t4 idle")
            submissions: list[Submission] = [
                {
                    "name": f"cz-{index}",
                    "lane": "nv",
                    "constraint": None,
                    "script": "script",
                    "run_path": str(root / f"columba/runs/{index}"),
                }
                for index in range(2)
            ]
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.gpu_inventory", return_value=offers),
                patch("cluster.slurm.queue", return_value=()),
                patch("cluster.slurm.subprocess.run", return_value=Mock(stdout="123\n")) as process,
            ):
                results = submit_pending(submissions, Budget())
            self.assertEqual(
                [result["status"] for result in results], ["submitted", "no free preferred GPU"]
            )
            self.assertEqual(process.call_count, 1)

    def test_pending_retarget_only_cancels_requested_expired_pending_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission: Submission = {
                "name": "cz-request",
                "lane": "nv",
                "constraint": "a100-80",
                "script": "script",
                "run_path": str(root / "columba/runs/example"),
                "base_model": "yolo11x.pt",
            }
            pending = Job(
                "123",
                "cz-request",
                "PENDING",
                "gpu-long",
                "gres/gpu:a100-80:1",
                1,
                datetime.now() - timedelta(hours=3),
            )
            unrelated = Job("999", "moonshot-other", "RUNNING", "gpu", "gpu:8")
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch("cluster.slurm.queue", side_effect=[(pending, unrelated), (unrelated,)]),
                patch(
                    "cluster.slurm.subprocess.run", side_effect=[Mock(), Mock(stdout="456\n")]
                ) as process,
            ):
                result = submit_pending([submission], Budget(), 2)[0]
            self.assertEqual(result["retargeted_from"], "123")
            self.assertEqual(
                process.call_args_list[0].args[0], ["scancel", "--state=PENDING", "123"]
            )
            self.assertIn("--gres=gpu:h100-47:1", process.call_args.args[0])
            for active in (
                replace(pending, state="RUNNING"),
                replace(pending, submitted_at=datetime.now()),
            ):
                with (
                    patch("cluster.slurm.Path.home", return_value=root),
                    patch(
                        "cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)
                    ),
                    patch("cluster.slurm.queue", return_value=(active,)),
                    patch("cluster.slurm.subprocess.run") as process,
                ):
                    self.assertEqual(
                        submit_pending([submission], Budget(), 2)[0]["status"], "active"
                    )
                    process.assert_not_called()
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch(
                    "cluster.slurm.queue",
                    side_effect=[(pending,), (replace(pending, state="RUNNING"),)],
                ),
                patch("cluster.slurm.subprocess.run") as process,
            ):
                self.assertEqual(submit_pending([submission], Budget(), 2)[0]["status"], "active")
                self.assertEqual(process.call_count, 1)
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.gpu_inventory", return_value=()),
                patch("cluster.slurm.queue", return_value=(pending,)),
                patch("cluster.slurm.subprocess.run") as process,
            ):
                self.assertEqual(submit_pending([submission], Budget(), 2)[0]["status"], "active")
                process.assert_not_called()

    def test_smoke_resources_are_bounded_before_any_mutation(self) -> None:
        recipe = load_recipe(SMOKE_RECIPE)
        submission = submission_for(
            DesiredRun(SMOKE_RECIPE, recipe, 0),
            Path.home() / "columba",
            Path("/checkout"),
            "commit",
        )
        self.assertEqual(submission["time_limit_minutes"], 10)
        self.assertEqual(gpu_resource_arguments(submission), ("--cpus-per-task=4", "--mem=16G"))
        for changes in ({"cpu_count": 9}, {"memory_limit_gb": 65}):
            invalid = cast(Submission, {**submission, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gpu_resource_arguments(invalid)

    def test_status_without_recipes_serialises_submission_time(self) -> None:
        submitted_at = datetime(2026, 10, 15, 1, 0)
        job = Job("123", "cz-status", "RUNNING", "gpu", "gpu:1", 1, submitted_at)
        with (
            patch("sys.argv", ["sweep", "status"]),
            patch("cluster.sweep.remote_root", return_value=Path("/columba")),
            patch("cluster.sweep.current_queue", return_value=(job,)),
            patch("cluster.sweep.inspect_runs", return_value={}),
            patch("cluster.sweep.remote", return_value='{"arrays": [], "transitions": []}'),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            sweep_main()
        self.assertEqual(json.loads(output.getvalue())["submitted_at"], submitted_at.isoformat())

    def test_sbatch_failure_surfaces_scheduler_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission: Submission = {
                "name": "cz-error",
                "lane": "nv",
                "constraint": "t4",
                "script": "script",
                "run_path": str(root / "columba/runs/example"),
            }
            failure = subprocess.CalledProcessError(
                1, ["sbatch"], stderr="Requested node configuration is not available"
            )
            with (
                patch("cluster.slurm.Path.home", return_value=root),
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch("cluster.slurm.queue", return_value=()),
                patch("cluster.slurm.subprocess.run", side_effect=failure),
            ):
                with self.assertRaisesRegex(RuntimeError, "Requested node configuration"):
                    submit_pending([submission], Budget())

    def test_gpu_guards_and_large_memory_batches(self) -> None:
        names = (
            "NVIDIA A100-PCIE-40GB",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA H100 NVL",
            "NVIDIA H100 80GB HBM3",
            "NVIDIA H200",
            "NVIDIA TITAN V",
            "Tesla T4",
        )
        self.assertTrue(all(allowed_gpu(name) for name in names))
        self.assertFalse(allowed_gpu("NVIDIA RTX 4090"))
        for memory_gb, expected in ((40, 12), (47, 16), (80, 24), (96, 32), (141, 48)):
            with self.subTest(memory_gb=memory_gb):
                self.assertEqual(select_batch(memory_gb, "yolo11x.pt", 640), expected)
                self.assertEqual(select_batch(memory_gb, "yolo11n.pt", 640), 64)

    def test_environment_selection_preserves_cpu_and_x86_pins(self) -> None:
        self.assertEqual(environment_for("x86_64", True).directory_name, "x86_64-cu126")
        self.assertEqual(environment_for("x86_64", True).torch_version, "2.8.0")
        self.assertEqual(environment_for("aarch64", True).directory_name, "aarch64-cu126")
        self.assertEqual(environment_for("aarch64", True).torch_version, "2.9.0")
        self.assertEqual(environment_for("aarch64", False).directory_name, "aarch64-cpu")
        self.assertEqual(environment_for("aarch64", False).torch_version, "2.8.0")

    def test_bootstrap_print_env_detects_gpu_unless_cpu_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binaries = Path(directory)
            combinations = product(
                ("x86_64", "aarch64"), (True, False), ((), ("--cpu",), ("--gpu",))
            )
            for architecture, gpu_available, flags in combinations:
                for name, output in (
                    ("uname", f"echo {architecture}"),
                    ("nvidia-smi", "exit 0" if gpu_available else "exit 1"),
                ):
                    executable = binaries / name
                    executable.write_text(f"#!/bin/sh\n{output}\n")
                    executable.chmod(0o755)
                environment = {
                    **os.environ,
                    "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                }
                expected_gpu = gpu_available if not flags else flags == ("--gpu",)
                with self.subTest(architecture=architecture, gpu=gpu_available, flags=flags):
                    result = subprocess.run(
                        ["bash", str(REPOSITORY / "cluster/bootstrap.sh"), *flags, "--print-env"],
                        env=environment,
                        check=True,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        Path(result.stdout.strip()).name,
                        environment_for(architecture, expected_gpu).directory_name,
                    )

    def test_atomic_checkout_publication_never_rewrites_a_live_checkout(self) -> None:
        buffer = io.BytesIO()
        payload = b"original bootstrap"
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            member = tarfile.TarInfo("cluster/bootstrap.sh")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            published = publish_archive(root, "a" * 16, "commit", io.BytesIO(buffer.getvalue()))
            script = published / "cluster/bootstrap.sh"
            original_inode = script.stat().st_ino
            self.assertTrue((published / READY_FILE).exists())
            self.assertEqual(
                publish_archive(root, "a" * 16, "commit", io.BytesIO(b"invalid")), published
            )
            self.assertEqual(script.stat().st_ino, original_inode)
            self.assertEqual(script.read_bytes(), payload)
            with self.assertRaises(tarfile.ReadError):
                publish_archive(root, "b" * 16, "commit", io.BytesIO(b"invalid"))
            self.assertFalse((root / ("b" * 16)).exists())
            self.assertFalse(any(path.name.startswith(".bbbb") for path in root.iterdir()))

    def test_recipe_batch_override(self) -> None:
        self.assertIsNone(validate_recipe(sample_recipe()).batch_override)
        recipe = validate_recipe({**sample_recipe(), "batch_override": 8})
        self.assertEqual(recipe.batch_override, 8)
        for value in (0, -1, "8", 2.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_recipe({**sample_recipe(), "batch_override": cast(JsonValue, value)})

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
            for index in range(GPU_RUNNING_LIMIT)
        )
        cpu_jobs = tuple(
            Job(str(index + 10), f"cz-cpu-{index}", "RUNNING", "normal", "N/A")
            for index in range(CPU_RUNNING_LIMIT)
        )
        self.assertFalse(Budget().allows(gpu_jobs, "nv"))
        self.assertFalse(Budget().allows(cpu_jobs, "arm"))
        self.assertTrue(Budget().allows(gpu_jobs[:-1], "nv"))
        pending = replace(gpu_jobs[-1], state="PENDING")
        self.assertFalse(Budget().allows((*gpu_jobs[:-1], pending), "nv"))
        self.assertFalse(Budget(3, 1).allows(gpu_jobs[:3], "nv"))
        all_pending = tuple(
            Job(str(index), f"cz-pending-{index}", "PENDING", "normal", "N/A")
            for index in range(QUEUED_LIMIT)
        )
        self.assertFalse(Budget().allows(all_pending, "nv"))
        for values in (
            (GPU_RUNNING_LIMIT + 1, CPU_RUNNING_LIMIT, QUEUED_LIMIT),
            (GPU_RUNNING_LIMIT, CPU_RUNNING_LIMIT + 1, QUEUED_LIMIT),
            (GPU_RUNNING_LIMIT, CPU_RUNNING_LIMIT, QUEUED_LIMIT + 1),
        ):
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
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
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
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
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
        self.assertIn("--gres=gpu:1", submission["script"])
        self.assertIn("--nice=1000", submission["script"])
        self.assertEqual(submission["constraint"], "titanv")
        self.assertNotEqual(job_name("x" * 100 + "a", 0), job_name("x" * 100 + "b", 0))

    def test_constraint_override_keeps_recipe_and_records_metadata(self) -> None:
        recipe = load_recipe(BASELINE_RECIPE)
        run = DesiredRun(BASELINE_RECIPE, recipe, 1)
        root, checkout = Path("/home/user/columba"), Path("/checkout")
        submission = submission_for(run, root, checkout, "commit", constraint="titanv")
        self.assertEqual(submission["constraint"], "titanv")
        self.assertIn("constraint_override", submission["script"])
        self.assertIsNone(
            submission_for(run, root, checkout, "commit", constraint="null")["constraint"]
        )
        self.assertEqual(submission_for(run, root, checkout, "commit")["constraint"], "titanrtx")
        with self.assertRaises(ValueError):
            submission_for(run, root, checkout, "commit", constraint="v100")

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

    def test_leaderboard_uses_lf_and_repository_relative_run_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaderboard.csv"
            row = dict.fromkeys(LEADERBOARD_COLUMNS, "")
            row.update({"recipe_id": "candidate", "seed": "0"})
            update_leaderboard(path, [row])
            self.assertNotIn(b"\r", path.read_bytes())
        self.assertEqual(
            repository_relative(Path("/worktrees/a/experiments/runs/recipe/seed0")),
            "experiments/runs/recipe/seed0",
        )
        self.assertEqual(repository_relative(Path("/elsewhere/seed0")), "/elsewhere/seed0")

    def test_fetch_excludes_smoke_from_real_leaderboard_but_allows_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            real_leaderboard = repository / "experiments/leaderboard.csv"
            real_leaderboard.parent.mkdir(parents=True)
            incumbent = dict.fromkeys(LEADERBOARD_COLUMNS, "")
            incumbent.update({"recipe_id": "incumbent", "seed": "0", "status": "complete"})
            update_leaderboard(real_leaderboard, [incumbent])
            original = real_leaderboard.read_bytes()
            desired = DesiredRun(SMOKE_RECIPE, load_recipe(SMOKE_RECIPE), 0)
            destination = root / "fetched"
            local = destination / desired.relative_run
            local.mkdir(parents=True)
            (local / "run.json").write_text(
                json.dumps({"recipe_id": "smoke-5ep", "seed": 0, "status": "complete"})
            )
            scratch = root / "leaderboard-smoke.csv"
            with (
                patch("cluster.sweep.REPOSITORY", repository),
                patch("cluster.sweep.subprocess.run", return_value=Mock(returncode=0)),
            ):
                fetch(root / "remote", [desired], destination, real_leaderboard)
                self.assertEqual(real_leaderboard.read_bytes(), original)
                other_leaderboard = root / "main/experiments/leaderboard.csv"
                update_leaderboard(other_leaderboard, [incumbent])
                other_original = other_leaderboard.read_bytes()
                fetch(root / "remote", [desired], destination, other_leaderboard)
                self.assertEqual(other_leaderboard.read_bytes(), other_original)
                fetch(root / "remote", [desired], destination, scratch)
            with scratch.open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["recipe_id"], "smoke-5ep")

    def test_smoke_interrupt_is_forbidden_for_production_recipes(self) -> None:
        with self.assertRaisesRegex(ValueError, "restricted"):
            run(BASELINE_RECIPE, 0, Path("/unused"), REPOSITORY, stop_after_epoch=1)
        desired = DesiredRun(BASELINE_RECIPE, load_recipe(BASELINE_RECIPE), 0)
        with self.assertRaisesRegex(ValueError, "smoke"):
            submission_for(desired, Path("/root"), Path("/checkout"), "commit", 1)

    def test_excluded_nodes_file_is_default_and_extras_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / "excluded_nodes.txt"
            listing.write_text("# header\nxgpe6  # broken\n\nxgpe2\n", encoding="utf-8")
            self.assertEqual(excluded_nodes(path=listing), ["xgpe6", "xgpe2"])
            self.assertEqual(
                excluded_nodes(["xgpd3", "xgpe6"], listing), ["xgpe6", "xgpe2", "xgpd3"]
            )
        self.assertIn("xgpe6", excluded_nodes())
        self.assertIn("xgpe2", excluded_nodes())
        self.assertNotIn("xgpj0", excluded_nodes())

    def test_gpu_probe_pins_only_an_allowed_nv_node(self) -> None:
        probe = probe_submission("xgpe6", Path("/home/user/columba"), Path("/checkout"))
        self.assertEqual(probe["required_node"], "xgpe6")
        self.assertIsNone(probe["constraint"])
        self.assertIn("--gres=gpu:1", probe["script"])
        with self.assertRaises(ValueError):
            probe_submission("xgpx6", Path("/root"), Path("/checkout"))

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


class OrchestrationTests(unittest.TestCase):
    def write_recipe(self, path: Path, identifier: str = "nano", model: str = "yolo11n.pt") -> Path:
        values = sample_recipe()
        values.update(id=identifier, base_model=model, seeds=[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(values))
        return path

    def rules(self) -> Rules:
        return replace(
            load_rules(REPOSITORY / "training/rounds/round2.yaml"),
            identifier="smoke-rules",
            approval="confirmed",
            top_k=1,
            next_rules=None,
            seeds=(0, 1, 2),
        )

    def test_qos_caps_and_predeclared_slots(self) -> None:
        policy = load_policy()
        self.assertEqual(policy.type_limits["a100-40"], 4)
        self.assertEqual(policy.type_limits["a100-80"], 2)
        self.assertEqual(policy.type_limits["h100-47"], 2)
        self.assertEqual(policy.type_limits["h100-96"], 1)
        self.assertEqual(policy.type_limits["h200-141"], 1)
        self.assertEqual(policy.max_submit_jobs, 32)
        self.assertLessEqual(sum(policy.array_gpu_slots.values()), 8)

    def test_array_throttles_total_and_type_limits(self) -> None:
        allocation = shares({"nv": 12, "h47": 6, "h96": 3}, {"nv": 8, "h47": 2, "h96": 1}, 8)
        self.assertEqual(sum(allocation.values()), 8)
        self.assertLessEqual(allocation["h47"], 2)
        self.assertEqual(allocation["h96"], 1)
        self.assertEqual(sum(shares({"nv": 12, "h47": 6}, {"nv": 8, "h47": 2}, 4).values()), 4)
        with self.assertRaises(ValueError):
            shares({}, {}, 9)

    def test_arrays_group_light_or_features_and_heavy_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            light = self.write_recipe(path / "light.yaml")
            heavy = self.write_recipe(path / "heavy.yaml", "teacher", "yolo11x.pt")
            offers = parse_inventory(GPU_INVENTORY)
            grouped = group_tasks([task_for(light, 0)], offers)
            request, _ = next(iter(grouped.values()))
            self.assertEqual(request.resources, "gpu:nv:1")
            self.assertEqual(request.constraint, "t4|titanv|titanrtx")
            self.assertEqual(request.partitions, ("gpu-long",))
            request, _ = next(iter(group_tasks([task_for(heavy, 0)], offers).values()))
            self.assertEqual(request.resources, "gpu:h100-47:1")
            request, _ = next(
                iter(group_tasks([task_for(light, 0)], offers, required_node="xgpj0").values())
            )
            self.assertEqual(request.resources, "gpu:a100-80:1")
            self.assertEqual(request.time_limit, "03:00:00")

    def test_array_submission_manifest_and_done_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            checkout = root / "checkouts/pinned"
            recipe = self.write_recipe(checkout / "training/recipes/light.yaml", "smoke-manifest")
            values = yaml.safe_load(recipe.read_text())
            values["seeds"] = [0, 1, 2]
            recipe.write_text(yaml.safe_dump(values))
            (checkout / "cluster/templates").mkdir(parents=True)
            (checkout / "cluster/templates/nv.sbatch").write_text(
                (REPOSITORY / "cluster/templates/nv.sbatch").read_text()
            )
            (checkout / "cluster/excluded_nodes.txt").write_text("xgpe2\nxgpe6\n")
            done = root / "runs/smoke-manifest/seed2/DONE"
            done.parent.mkdir(parents=True)
            done.write_text("complete")
            commands: list[tuple[Sequence[str], dict[str, JsonValue]]] = []

            def scheduler(
                command: Sequence[str], **keywords: JsonValue
            ) -> subprocess.CompletedProcess[str]:
                commands.append((command, keywords))
                return subprocess.CompletedProcess(
                    command, 0, "1 project\n" if command[0] == "du" else "700\n", ""
                )

            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.arrays.queue", return_value=()),
                patch("cluster.arrays.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch("cluster.arrays.subprocess.run", side_effect=scheduler),
            ):
                records = submit_arrays(
                    root, checkout, [task_for(recipe, seed) for seed in (0, 1, 2)], gpu_limit=1
                )
            self.assertEqual(records[0]["task_count"], 2)
            command, keywords = commands[-1]
            self.assertIn("--array=0-1%1", command)
            self.assertIn("--gres=gpu:nv:1", command)
            self.assertIn("--exclude=xgpe2,xgpe6", command)
            self.assertIn("--nice=1000", command)
            self.assertIn("--requeue", cast(str, keywords["input"]))
            manifest = json.loads(Path(cast(str, records[0]["manifest"])).read_text())
            self.assertEqual([item["seed"] for item in manifest["tasks"]], [0, 1])
            jobs = parse_queue("700_0|cz-array-test|RUNNING|gpu-long|gres/gpu:nv:1|1")
            self.assertEqual(active_run_names(root, jobs), {job_name("smoke-manifest", 0)})
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.arrays.queue", return_value=jobs),
                patch("cluster.arrays.subprocess.run") as control,
            ):
                throttle_arrays(root, 4)
            self.assertIn("ArrayTaskThrottle=2", control.call_args.args[0])
            state = cast(list[dict[str, JsonValue]], snapshot(root, {"700_0": "RUNNING"})["arrays"])
            self.assertEqual(state[0]["state"], "RUNNING")

    def test_array_tasks_count_against_submission_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            path = self.write_recipe(root / "checkouts/pinned/recipe.yaml")
            jobs = parse_queue(
                "\n".join(
                    f"{index}_0|cz-array-test|PENDING|gpu-long|gres/gpu:h100-96:1|1"
                    for index in range(24)
                )
            )
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.arrays.queue", return_value=jobs),
                patch("cluster.arrays.subprocess.run") as scheduler,
            ):
                with self.assertRaisesRegex(RuntimeError, "submitted-job cap"):
                    submit_arrays(root, root / "checkouts/pinned", [task_for(path, 0)])
            scheduler.assert_not_called()

    def test_requeue_progress_signal_and_restart_are_counted_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "weights").mkdir()
            (path / "weights/last.pt").write_bytes(b"checkpoint")
            (path / "run.json").write_text('{"epochs_done": 2}')
            self.assertFalse(update_ledger(path, "701", 0, "start"))
            self.assertTrue(update_ledger(path, "701", 0, "signal"))
            self.assertFalse(update_ledger(path, "701", 1, "start"))
            self.assertEqual(json.loads((path / "requeue.json").read_text())["count"], 1)
            with self.assertRaisesRegex(RuntimeError, "no saved epoch"):
                update_ledger(path, "701", 1, "signal")
            self.assertTrue((path / "HOLD").exists())

    def test_node_failure_restart_and_requeue_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "weights").mkdir()
            (path / "weights/last.pt").write_bytes(b"checkpoint")
            (path / "run.json").write_text('{"epochs_done": 3}')
            self.assertFalse(update_ledger(path, "702", 1, "start"))
            self.assertEqual(json.loads((path / "requeue.json").read_text())["count"], 1)
            (path / "requeue.json").write_text(json.dumps({"count": MAX_REQUEUES, "last_epoch": 2}))
            with self.assertRaisesRegex(RuntimeError, "limit"):
                update_ledger(path, "703", 0, "signal")

    def test_force_signal_cannot_target_production(self) -> None:
        with self.assertRaisesRegex(ValueError, "smoke epoch"):
            task_for(BASELINE_RECIPE, 0, 1)

    def test_transition_rank_small_gain_and_latency_tie(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nano_path = self.write_recipe(root / "nano.yaml", "nano")
            small_path = self.write_recipe(root / "small.yaml", "small", "yolo11s.pt")
            nano = Candidate(nano_path, load_recipe(nano_path), 0.70, 20.0)
            small = Candidate(small_path, load_recipe(small_path), 0.74, 40.0)
            self.assertEqual(select_candidates([nano, small], self.rules()), (nano,))
            small = replace(small, recall=0.741)
            self.assertEqual(select_candidates([nano, small], self.rules()), (small,))
            other_path = self.write_recipe(root / "other.yaml", "other", "yolo26n.pt")
            other = Candidate(other_path, load_recipe(other_path), 0.705, 21.0)
            self.assertEqual(select_candidates([nano, other], self.rules()), (nano,))
            with self.assertRaisesRegex(HoldRequired, "tie"):
                select_candidates([nano, replace(other, latency_ms=20.0)], self.rules())
            resolved = replace(self.rules(), tie_order=("other", "nano"))
            self.assertEqual(
                select_candidates([nano, replace(other, latency_ms=20.0)], resolved)[0],
                replace(other, latency_ms=20.0),
            )

    def test_generated_nano_800_and_small_deployment_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_recipe(root / "nano.yaml", "nano-640-r1")
            nano = Candidate(source, load_recipe(source), 0.7, 20.0)
            paths = generate_recipes([nano], self.rules(), root / "generated")
            self.assertEqual({load_recipe(path).train["imgsz"] for path in paths}, {640, 800})
            self.assertEqual(load_recipe(paths[0]).seeds, (0, 1, 2))
            self.assertEqual(load_recipe(paths[0]).train["epochs"], 300)
            self.assertEqual(generate_recipes([nano], self.rules(), root / "generated"), paths)
            source = self.write_recipe(root / "small.yaml", "small-640-r1", "yolo11s.pt")
            small = Candidate(source, load_recipe(source), 0.75, 40.0)
            path = generate_recipes(
                [small], self.rules(), root / "small-generated", "teacher/seed0"
            )[0]
            self.assertEqual(load_recipe(path).teacher, "teacher/seed0")
            self.assertEqual(load_recipe(path).deploy_sizes, ((352, 608), (384, 640)))

    def test_done_without_metrics_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_recipe(root / "recipe.yaml")
            run_path = root / "runs/nano/seed0"
            run_path.mkdir(parents=True)
            (run_path / "DONE").write_text("bad")
            with self.assertRaisesRegex(HoldRequired, "incomplete"):
                completed_metric(root, load_recipe(source), 0)

    def test_draft_arm_refused_and_prepare_does_not_submit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            checkout = root / "checkouts/pinned"
            checkout.mkdir(parents=True)
            source = checkout / "rules.yaml"
            source.write_text((REPOSITORY / "training/rounds/round2.yaml").read_text())
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.transitions.require_owned_dependencies"),
                patch("cluster.transitions.queue", return_value=()),
                patch("cluster.transitions.subprocess.run") as scheduler,
            ):
                with self.assertRaisesRegex(ValueError, "draft"):
                    arm_transition(root, checkout, source, ["704"], ["705", "706"])
                record = arm_transition(
                    root, checkout, source, ["704"], ["705", "706"], prepare_only=True
                )
            self.assertEqual(record["status"], "prepared, not armed")
            scheduler.assert_not_called()

    def test_checkpoint_recovery_resubmits_and_rearms_with_two_attempt_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_recipe(root / "source.yaml")
            rules = replace(self.rules(), inputs=(source,))
            run_path = root / "runs/nano/seed0"
            (run_path / "weights").mkdir(parents=True)
            (run_path / "weights/last.pt").write_bytes(b"checkpoint")
            (run_path / "run.json").write_text('{"epochs_done": 3, "job_id": "707"}')
            transition = root / "transitions/first"
            transition.mkdir(parents=True)
            plan: dict[str, JsonValue] = {"rules": str(source), "teacher_after": ["709", "710"]}
            with (
                patch.dict(os.environ, {"SLURM_JOB_ID": "708"}),
                patch("cluster.transitions.queue", return_value=()),
                patch("cluster.transitions.terminal_state", return_value="TIMEOUT"),
                patch(
                    "cluster.transitions.submit_arrays", return_value=[{"array_id": "711"}]
                ) as submission,
                patch("cluster.transitions.arm_transition", return_value={"job_id": "712"}) as arm,
            ):
                self.assertTrue(recover_interrupted_inputs(root, root, rules, plan, transition))
                self.assertEqual(arm.call_args.args[3], ["708", "711"])
                submission.assert_called_once()
                with self.assertRaisesRegex(HoldRequired, "no checkpoint progress"):
                    recover_interrupted_inputs(root, root, rules, plan, transition)
                (run_path / "run.json").write_text('{"epochs_done": 4, "job_id": "707"}')
                self.assertTrue(recover_interrupted_inputs(root, root, rules, plan, transition))
                (run_path / "run.json").write_text('{"epochs_done": 5, "job_id": "707"}')
                with self.assertRaisesRegex(HoldRequired, "recovery limit"):
                    recover_interrupted_inputs(root, root, rules, plan, transition)

    def test_genuine_failure_holds_without_submitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_recipe(root / "source.yaml")
            rules = replace(self.rules(), inputs=(source,))
            run_path = root / "runs/nano/seed0"
            run_path.mkdir(parents=True)
            (run_path / "run.json").write_text('{"epochs_done": 0, "job_id": "713"}')
            with (
                patch("cluster.transitions.terminal_state", return_value="FAILED"),
                patch("cluster.transitions.submit_arrays") as submission,
            ):
                with self.assertRaisesRegex(HoldRequired, "no mechanical recovery"):
                    recover_interrupted_inputs(root, root, rules, {}, root)
            submission.assert_not_called()

    def test_transition_malformed_plan_writes_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            plan = home / "columba/transitions/bad/plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}")
            with patch("pathlib.Path.home", return_value=home):
                execute_transition(plan)
            self.assertTrue((plan.parent / "HOLD").exists())
            self.assertEqual(json.loads((plan.parent / "run.json").read_text())["status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
