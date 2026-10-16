from __future__ import annotations

import csv
import io
import json
import os
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time
import unittest
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import yaml

from app.json_data import JsonValue
from cluster.arrays import (
    active_run_names,
    group_tasks,
    retarget_array,
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
from cluster.device_faults import (
    DeviceVisibilityError,
    active_bad_nodes,
    check_visibility,
    clear_bad_node,
    combined_exclusions,
    fault_evidence,
    handle_fault,
    repair,
    reserve_retry,
)
from cluster.environment import environment_for
from cluster.gpu_policy import available_request, load_policy
from cluster.gpu_probe import probe
from cluster.gpu_selection import GPU_CONSTRAINTS, gpu_inventory, parse_inventory, request_for
from cluster.hardware import allowed_gpu, select_batch
from cluster.recipe import load_recipe, resolve_teacher, validate_recipe
from cluster.requeue import MAX_REQUEUES, requeue_identifier, update_ledger
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
from cluster.staging import align_test_taxonomy, stage_dataset
from cluster.state import historical_states, snapshot
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
    best_teacher,
    collect_candidates,
    completed_metric,
    distill_selection,
    execute_transition,
    generate_recipes,
    load_rules,
    plain_evidence,
    recover_interrupted_inputs,
    require_owned_dependencies,
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
                request = request_for(choice or "titanv", parse_inventory(GPU_INVENTORY))
                self.assertTrue(set(request.arguments()).issubset(command))
                self.assertIn("--nice=0", command)
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
            self.assertEqual(node, "--nodelist=xgpd2")
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
            self.assertIn("--nice=0", process.call_args_list[1].args[0])
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
        self.assertIn("--nice=0", submission["script"])
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

    def test_species_recipe_aligns_only_staged_single_test_names(self) -> None:
        classes = REPOSITORY / "training/classes.yaml"
        species = yaml.safe_load(classes.read_text())["taxonomies"]["species"]["names"]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "data.yaml").write_text(
                yaml.safe_dump({"test": "images/test", "names": ["pigeon"]})
            )
            self.assertFalse(align_test_taxonomy(dataset, "single", classes))
            self.assertTrue(align_test_taxonomy(dataset, "species", classes))
            aligned = yaml.safe_load((dataset / "data.yaml").read_text())
            self.assertEqual(aligned["names"], species)
            self.assertEqual(aligned["test"], "images/test")
            self.assertFalse(align_test_taxonomy(dataset, "species", classes))
            (dataset / "data.yaml").write_text(yaml.safe_dump({"names": ["dove", "pigeon"]}))
            with self.assertRaises(ValueError):
                align_test_taxonomy(dataset, "species", classes)

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

    def test_qos_caps_are_independent_ceilings(self) -> None:
        policy = load_policy()
        self.assertEqual(policy.type_limits["a100-40"], 4)
        self.assertEqual(policy.type_limits["a100-80"], 2)
        self.assertEqual(policy.type_limits["h100-47"], 2)
        self.assertEqual(policy.type_limits["h100-96"], 1)
        self.assertEqual(policy.type_limits["h200-141"], 1)
        self.assertEqual(policy.max_submit_jobs, 32)
        self.assertEqual(policy.type_limits["nv"], 8)

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
            self.assertIn("--nice=0", command)
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
            self.assertIn("ArrayTaskThrottle=1", control.call_args.args[0])
            state = cast(list[dict[str, JsonValue]], snapshot(root, {"700_0": "RUNNING"})["arrays"])
            self.assertEqual(state[0]["state"], "RUNNING")

    def test_array_resubmission_returns_existing_dependency_and_global_throttles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            checkout = root / "checkouts/pinned"
            first = self.write_recipe(checkout / "first.yaml", "first")
            second = self.write_recipe(checkout / "second.yaml", "second")
            (checkout / "cluster/templates").mkdir(parents=True)
            (checkout / "cluster/templates/nv.sbatch").write_text("set -euo pipefail\n")
            (checkout / "cluster/excluded_nodes.txt").write_text("xgpe6\n")
            manifest_path = root / "arrays/first/manifest.json"
            manifest_path.parent.mkdir(parents=True)
            first_task = task_for(first, 0)
            from dataclasses import asdict

            manifest_path.write_text(
                json.dumps({"tasks": [asdict(first_task)], "directory": str(manifest_path.parent)})
            )
            record = {
                "array_id": "720",
                "throttle": 1,
                "resource_type": "nv",
                "task_count": 1,
                "manifest": str(manifest_path),
            }
            (manifest_path.parent / "submission.json").write_text(json.dumps(record))
            (root / "arrays/concurrency.json").write_text('{"gpu_limit": 4}')
            jobs = parse_queue("720_0|cz-array-first|PENDING|gpu-long|gres/gpu:nv:1|1")
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.arrays.queue", return_value=jobs),
                patch("cluster.arrays.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch(
                    "cluster.arrays.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, "1 project", ""),
                        subprocess.CompletedProcess([], 0, "", ""),
                        subprocess.CompletedProcess([], 0, "721", ""),
                    ],
                ) as scheduler,
            ):
                self.assertEqual(submit_arrays(root, checkout, [first_task]), [record])
                scheduler.assert_not_called()
                records = submit_arrays(root, checkout, [task_for(second, 0)])
            self.assertEqual(records[0]["array_id"], "721")
            self.assertLessEqual(
                records[0]["throttle"]
                + json.loads((manifest_path.parent / "submission.json").read_text())["throttle"],
                4,
            )

    def test_array_any_override_removes_recipe_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_recipe(Path(directory) / "recipe.yaml")
            values = yaml.safe_load(path.read_text())
            values["gpu_constraint"] = "h100-96"
            path.write_text(yaml.safe_dump(values))
            offers = parse_inventory(GPU_INVENTORY)
            pinned = next(iter(group_tasks([task_for(path, 0)], offers).values()))[0]
            automatic = next(iter(group_tasks([task_for(path, 0)], offers, "any").values()))[0]
            self.assertEqual(pinned.resources, "gpu:h100-96:1")
            self.assertEqual(automatic.resources, "gpu:nv:1")

    def test_transition_dependencies_reject_non_project_jobs(self) -> None:
        jobs = parse_queue("730|moonshot-other|RUNNING|gpu-long|gres/gpu:nv:1|1")
        with (
            patch("cluster.transitions.queue", return_value=jobs),
            patch("cluster.transitions.subprocess.run") as scheduler,
        ):
            with self.assertRaisesRegex(ValueError, "outside cz-"):
                require_owned_dependencies(["730"])
        scheduler.assert_not_called()

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

    def test_requeue_targets_one_array_task_even_when_its_job_id_is_the_array_root(self) -> None:
        self.assertEqual(requeue_identifier("700", "700", "1"), "700_1")
        self.assertEqual(requeue_identifier("701", "700", "0"), "700_0")
        self.assertEqual(requeue_identifier("702", None, None), "702")
        with self.assertRaises(ValueError):
            requeue_identifier("700", "700", None)

    def test_batch_shell_forwards_usr1_and_preserves_finished_child_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment"
            (environment / "bin").mkdir(parents=True)
            executable = environment / "bin/python"
            executable.write_text(
                "#!/usr/bin/env python\nimport os,signal,sys\nfrom pathlib import Path\n"
                "signal.signal(signal.SIGUSR1, lambda number,frame: sys.exit(75))\n"
                "Path(os.environ['COLUMBA_TEST_READY']).write_text(str(os.getpid()))\n"
                "signal.pause()\n"
            )
            executable.chmod(0o755)
            ready = root / "ready"
            template = (REPOSITORY / "cluster/templates/nv.sbatch").read_text()
            declarations = template[
                template.index('training_process_identifier=""') : template.index("finish_job()")
            ]
            supervisor = template[
                template.index("export COLUMBA_REQUEUE_REQUESTED=") : template.index(
                    '\nif [[ "$exit_status" == "$requeue_exit_status"'
                )
            ]
            script = (
                "set -euo pipefail\n"
                + declarations
                + "trap request_checkpoint USR1\n"
                + f"environment={shlex.quote(str(environment))}\nrun_arguments=()\n"
                + f"export COLUMBA_TEST_READY={shlex.quote(str(ready))}\n"
                + supervisor
                + "\nprintf '%s' \"$exit_status\"\n"
            )
            process = subprocess.Popen(
                ["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            try:
                deadline_s = time.monotonic() + 3
                while not ready.exists() and time.monotonic() < deadline_s:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                os.kill(process.pid, signal.SIGUSR1)
                stdout, stderr = process.communicate(timeout=3)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stdout, "75")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                if ready.exists():
                    try:
                        os.kill(int(ready.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

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

    def test_candidate_collection_requires_all_source_seeds_and_valid_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_recipe(root / "source.yaml")
            latency = root / "latency.json"
            latency.write_text(
                '{"input_hw": [384, 640], "estimated_pi5": {"end_to_end_p95_ms": 20}}'
            )
            rules = replace(self.rules(), inputs=(source,), latency_records={"yolo11n.pt": latency})
            with patch("cluster.transitions.completed_metric", return_value=0.7):
                self.assertEqual(collect_candidates(root, rules)[0].recall, 0.7)
                latency.write_text(
                    '{"input_hw": [640, 640], "estimated_pi5": {"end_to_end_p95_ms": 20}}'
                )
                with self.assertRaisesRegex(HoldRequired, "latency"):
                    collect_candidates(root, rules)
            with patch(
                "cluster.transitions.completed_metric", side_effect=HoldRequired("unfinished")
            ):
                with self.assertRaisesRegex(HoldRequired, "unfinished"):
                    collect_candidates(root, rules)

    def test_teacher_selection_requires_declared_tie_order(self) -> None:
        rules = load_rules(REPOSITORY / "training/rounds/round2-distill.yaml")
        identifiers = [load_recipe(path).identifier for path in rules.teachers]
        with patch("cluster.transitions.completed_metric", side_effect=[0.6, 0.7]):
            self.assertEqual(best_teacher(REPOSITORY, rules), identifiers[1] + "/seed0")
        with patch("cluster.transitions.completed_metric", side_effect=[0.7, 0.7]):
            with self.assertRaisesRegex(HoldRequired, "teacher tie"):
                best_teacher(REPOSITORY, replace(rules, teacher_tie_order=()))
        with patch("cluster.transitions.completed_metric", side_effect=[0.7, 0.7]):
            ordered = replace(rules, teacher_tie_order=(identifiers[1], identifiers[0]))
            self.assertEqual(best_teacher(REPOSITORY, ordered), identifiers[1] + "/seed0")

    def test_status_retains_cancelled_transition_and_filters_other_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "transitions/example/submission.json"
            record.parent.mkdir(parents=True)
            record.write_text('{"job_id": "740"}')
            output = (
                "740|CANCELLED by 123|cz-transition-example|\n"
                "740.batch|COMPLETED|batch|\n741|RUNNING|moonshot-other|\n"
            )
            with patch(
                "cluster.state.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, output, ""),
            ):
                self.assertEqual(historical_states(root), {"740": "CANCELLED"})

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
            draft = yaml.safe_load((REPOSITORY / "training/rounds/round2.yaml").read_text())
            draft.update({"approval": "draft", "confirmed_by": None})
            source.write_text(yaml.safe_dump(draft))
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
                patch("cluster.transitions.queue", return_value=()),
                patch("cluster.transitions.submit_arrays") as submission,
            ):
                with self.assertRaisesRegex(HoldRequired, "no mechanical recovery"):
                    recover_interrupted_inputs(root, root, rules, {}, root)
            submission.assert_not_called()

    def test_distill_checks_upstream_and_recovers_plain_and_teacher_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plain = self.write_recipe(root / "generated-recipes/round2-plain/plain.yaml", "plain")
            report_path = root / "transitions/parent/run.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {"round": "round2-plain", "status": "submitted", "recipes": [str(plain)]}
                )
            )
            rules = replace(self.rules(), stage="distill", selection_from="round2-plain", inputs=())
            report, paths = plain_evidence(root, rules, {"upstream": "parent"})
            self.assertEqual(paths, (plain,))
            self.assertEqual(report["status"], "submitted")
            run_path = root / "runs/plain/seed0"
            (run_path / "weights").mkdir(parents=True)
            (run_path / "weights/last.pt").write_bytes(b"checkpoint")
            (run_path / "run.json").write_text(
                '{"epochs_done": 2, "job_id": "750", "array_job_id": "749", "array_task_id": "0"}'
            )
            transition = root / "transitions/recovery-attempt"
            transition.mkdir(parents=True)
            with (
                patch.dict(os.environ, {"SLURM_JOB_ID": "751"}),
                patch("cluster.transitions.queue", return_value=()),
                patch("cluster.transitions.terminal_state", return_value="NODE_FAIL") as state,
                patch("cluster.transitions.submit_arrays", return_value=[{"array_id": "752"}]),
                patch("cluster.transitions.arm_transition", return_value={"job_id": "753"}),
            ):
                self.assertTrue(
                    recover_interrupted_inputs(
                        root,
                        root,
                        rules,
                        {"rules": str(plain), "upstream": "parent"},
                        transition,
                        paths,
                    )
                )
                state.assert_called_once_with("749_0")
            report_path.write_text('{"round": "other", "status": "submitted", "recipes": []}')
            with self.assertRaisesRegex(HoldRequired, "selection_from"):
                plain_evidence(root, rules, {"upstream": "parent"})

    def test_distill_selection_keeps_plain_rank_order_and_top_k(self) -> None:
        candidates = tuple(
            SimpleNamespace(recipe=SimpleNamespace(identifier=name)) for name in ("c", "a", "b")
        )
        typed = cast(Sequence[Candidate], candidates)
        self.assertEqual(
            [item.recipe.identifier for item in distill_selection(typed, ["a", "b", "c"], 1)],
            ["a"],
        )
        self.assertEqual(
            [item.recipe.identifier for item in distill_selection(typed, ["b", "c", "a"], 3)],
            ["b", "c", "a"],
        )
        with self.assertRaisesRegex(HoldRequired, "exceeds"):
            distill_selection(typed, ["a"], 2)
        with self.assertRaisesRegex(HoldRequired, "do not match"):
            distill_selection(typed, ["missing", "a"], 1)

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


class DeviceFaultTests(unittest.TestCase):
    def test_visibility_precedes_work_and_hardware_smoke_keeps_bootstrap_settings(self) -> None:
        script = (REPOSITORY / "cluster/templates/nv.sbatch").read_text()
        first = script.index("python -m cluster.device_faults check --nvidia-only")
        environment = script.index("bash cluster/bootstrap.sh --gpu\n")
        cuda = script.index('"$environment/bin/python" -m cluster.device_faults check')
        hardware = script.index("bash cluster/bootstrap.sh --gpu-smoke")
        training = script.index('"$environment/bin/python" -m cluster.run_job')
        self.assertLess(first, environment)
        self.assertLess(environment, cuda)
        self.assertLess(cuda, hardware)
        self.assertLess(hardware, training)
        self.assertNotIn('"$environment/bin/python" -m cluster.hardware --smoke', script)

    def test_visibility_requires_smi_success_expected_count_and_cuda(self) -> None:
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        with (
            patch(
                "cluster.device_faults.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "GPU-uuid\n", ""),
            ),
            patch.dict("sys.modules", {"torch": torch}),
        ):
            self.assertTrue(check_visibility(1)["cuda_available"])
            torch.cuda.is_available.return_value = False
            with self.assertRaises(DeviceVisibilityError):
                check_visibility(1)
            with self.assertRaises(DeviceVisibilityError):
                check_visibility(2, nvidia_only=True)
        with patch(
            "cluster.device_faults.subprocess.run",
            return_value=subprocess.CompletedProcess([], 6, "", "No devices were found"),
        ):
            with self.assertRaisesRegex(DeviceVisibilityError, "No devices"):
                check_visibility(1, nvidia_only=True)

    def test_driver_initialisation_failure_is_classified_but_not_oom(self) -> None:
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        with (
            patch(
                "cluster.device_faults.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "GPU-uuid\n", ""),
            ),
            patch.dict("sys.modules", {"torch": torch}),
        ):
            torch.cuda.init.side_effect = RuntimeError("CUDA driver initialization failed")
            with self.assertRaises(DeviceVisibilityError):
                check_visibility(1)
            torch.cuda.init.side_effect = RuntimeError("CUDA out of memory")
            with self.assertRaisesRegex(RuntimeError, "out of memory") as raised:
                check_visibility(1)
            self.assertNotIsInstance(raised.exception, DeviceVisibilityError)

    def test_quarantine_expiry_retry_budget_and_idempotent_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/device/seed0"
            entry = reserve_retry(root, run, "xgpd4", "800_0", "801", 0, "No devices were found")
            self.assertEqual(active_bad_nodes(root), ("xgpd4",))
            self.assertEqual(
                reserve_retry(root, run, "xgpd4", "800_0", "801", 0, "No devices were found"), entry
            )
            instant = datetime.fromisoformat(cast(str, entry["time"]))
            self.assertEqual(active_bad_nodes(root, instant + timedelta(hours=25)), ())
            for restart_count in (1, 2, 3):
                entry = reserve_retry(
                    root, run, "xgpd4", "800_0", "801", restart_count, "No devices were found"
                )
            self.assertEqual(entry["status"], "exhausted")
            self.assertEqual(json.loads((run / "device_faults.json").read_text())["retry_count"], 3)
            self.assertTrue((run / "HOLD").exists())
            self.assertEqual(
                json.loads((root / "state/bad_nodes.json").read_text())["nodes"]["xgpd4"]["count"],
                4,
            )

    def test_in_job_detection_records_excludes_and_requeues_one_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/device/seed0"
            fields = {"JobName": "cz-example", "ExcNodeList": "xgpe[2,6]", "ArrayTaskThrottle": "2"}
            environment = {
                "SLURM_JOB_ID": "801",
                "SLURM_ARRAY_JOB_ID": "800",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_RESTART_COUNT": "0",
            }
            with (
                patch.dict(os.environ, environment),
                patch("cluster.device_faults.platform.node", return_value="xgpd4"),
                patch("cluster.device_faults.owned_job", return_value=fields),
                patch(
                    "cluster.device_faults.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as scheduler,
            ):
                handle_fault(root, run, "No devices were found")
            command = scheduler.call_args_list[0].args[0]
            self.assertIn("JobId=800_0", command)
            self.assertIn("ReqNodeList=", command)
            self.assertIn(
                "xgpd4", next(value for value in command if value.startswith("ExcNodeList="))
            )
            self.assertEqual(scheduler.call_args_list[-1].args[0], ["scontrol", "requeue", "800_0"])
            self.assertFalse(update_ledger(run, "801", 1, "start"))
            self.assertEqual(json.loads((run / "requeue.json").read_text()).get("count", 0), 0)
            self.assertFalse((run / "HOLD").exists())

    def test_running_update_rejection_submits_finite_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/device/seed0"
            fields = {"JobName": "cz-example", "ExcNodeList": "(null)", "ArrayTaskThrottle": "2"}
            environment = {
                "SLURM_JOB_ID": "801",
                "SLURM_ARRAY_JOB_ID": "800",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_RESTART_COUNT": "0",
            }
            results = [
                subprocess.CompletedProcess([], 1, "", "Job is not pending"),
                subprocess.CompletedProcess([], 0, "802\n", ""),
            ]
            with (
                patch.dict(os.environ, environment),
                patch("cluster.device_faults.platform.node", return_value="xgpd4"),
                patch("cluster.device_faults.owned_job", return_value=fields),
                patch("cluster.slurm.queue", return_value=()),
                patch("cluster.device_faults.subprocess.run", side_effect=results) as scheduler,
            ):
                handle_fault(root, run, "No devices were found")
            command = scheduler.call_args_list[-1].args[0]
            self.assertIn("--time=1", command)
            self.assertIn("--nice=0", command)
            self.assertIn("--dependency=afterany:800_0", command)
            with (
                patch.dict(os.environ, {"SLURM_JOB_ID": "802"}),
                patch(
                    "cluster.device_faults.owned_job",
                    side_effect=[fields, {**fields, "ArrayTaskThrottle": "1"}],
                ),
                patch(
                    "cluster.device_faults.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as scheduler,
            ):
                repair(root, run, "800_0:0")
            commands = [call.args[0] for call in scheduler.call_args_list]
            self.assertEqual(commands[0], ["scontrol", "requeuehold", "800_0"])
            self.assertIn(["scontrol", "update", "JobId=800", "ArrayTaskThrottle=2"], commands)
            self.assertEqual(commands[-1], ["scontrol", "release", "800_0"])
            self.assertEqual(
                json.loads((run / "device_faults.json").read_text())["events"][0]["status"],
                "requeued",
            )

    def test_dynamic_nodes_are_combined_with_static_for_legacy_and_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reserve_retry(
                root, root / "runs/device/seed0", "xgpe4", "810", "810", 0, "No devices were found"
            )
            path = root / "recipe.yaml"
            recipe = sample_recipe()
            recipe.update(gpu_constraint="titanrtx", seeds=[0])
            path.write_text(yaml.safe_dump(recipe))
            with (
                patch("pathlib.Path.home", return_value=root.parent),
                patch("cluster.arrays.combined_exclusions", return_value=("xgpe4",)),
            ):
                with self.assertRaises(ValueError):
                    group_tasks([task_for(path, 0)], parse_inventory(GPU_INVENTORY))
            submission: Submission = {
                "name": "cz-example",
                "lane": "nv",
                "constraint": "titanrtx",
                "script": "",
                "run_path": str(root / "runs/test"),
            }
            from cluster.slurm import scheduling_request

            with patch("cluster.slurm.combined_exclusions", return_value=("xgpe4",)):
                with self.assertRaises(ValueError):
                    scheduling_request(submission, parse_inventory(GPU_INVENTORY))
            exclusions = combined_exclusions(root)
            self.assertIn("xgpe4", exclusions)
            self.assertTrue(set(excluded_nodes()) <= set(exclusions))

    def test_legacy_fault_log_recognition_and_genuine_failure_after_device_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/device/seed0"
            (root / "logs").mkdir()
            (root / "logs/cz-array-example-811_0.log").write_text("No devices were found")
            report: dict[str, JsonValue] = {
                "job_id": "812",
                "array_job_id": "811",
                "array_task_id": "0",
                "host": "xgpd2",
                "exit_status": 6,
                "error": "shell or bootstrap failed",
            }
            self.assertIsNotNone(fault_evidence(root, run, report))
            reserve_retry(root, run, "xgpd2", "811_0", "812", 0, "No devices were found")
            report.update(
                exit_status=1,
                error="ValueError: dataset annotation is invalid",
                slurm_restart_count=1,
            )
            self.assertIsNone(fault_evidence(root, run, report))
            report.update(error="FileNotFoundError: dataset/No devices were found/file.jpg")
            self.assertIsNone(fault_evidence(root, run, report))

    def test_new_restart_shell_failure_does_not_preserve_stale_device_failure(self) -> None:
        from cluster.job_failure import record_failure

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            run = home / "columba/runs/device/seed0"
            run.mkdir(parents=True)
            (run / "run.json").write_text(
                '{"job_id":"880", "status":"device_fault", "slurm_restart_count":0,'
                '"array_job_id":"880", "array_task_id":"0"}'
            )
            with (
                patch("pathlib.Path.home", return_value=home),
                patch.dict(
                    os.environ,
                    {
                        "SLURM_JOB_ID": "880",
                        "SLURM_RESTART_COUNT": "1",
                        "SLURM_ARRAY_JOB_ID": "880",
                        "SLURM_ARRAY_TASK_ID": "0",
                    },
                ),
            ):
                record_failure(run, {}, 1)
            report = json.loads((run / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["slurm_restart_count"], 1)

    def test_appended_old_device_output_does_not_classify_a_new_code_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            path = root / "logs/cz-array-example-881_0.log"
            path.write_text(
                "CZ_DEVICE_ATTEMPT 881_0:0\nNo devices were found\n"
                "CZ_DEVICE_ATTEMPT 881_0:1\nboot failed: missing package\n"
            )
            report: dict[str, JsonValue] = {
                "job_id": "881",
                "array_job_id": "881",
                "array_task_id": "0",
                "host": "xgpd4",
                "slurm_restart_count": 1,
                "error": "job shell or bootstrap failed",
                "exit_status": 1,
            }
            self.assertIsNone(fault_evidence(root, root / "runs/device/seed0", report))
            path.write_text(path.read_text() + "no CUDA-capable device\n")
            self.assertIsNotNone(fault_evidence(root, root / "runs/device/seed0", report))

    def test_transition_device_recovery_with_and_without_checkpoint(self) -> None:
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.yaml"
                values = sample_recipe()
                values.update(id="device", seeds=[0])
                source.write_text(yaml.safe_dump(values))
                run = root / "runs/device/seed0"
                run.mkdir(parents=True)
                (run / "run.json").write_text(
                    '{"job_id":"815", "host":"xgpd4", "error":"No devices were found",'
                    '"exit_status":6, "epochs_done":0}'
                )
                if checkpoint:
                    (run / "weights").mkdir()
                    (run / "weights/last.pt").touch()
                rules = replace(
                    load_rules(REPOSITORY / "training/rounds/round2.yaml"),
                    inputs=(source,),
                    next_rules=None,
                )
                target = root / "transitions/test"
                target.mkdir(parents=True)
                with (
                    patch.dict(os.environ, {"SLURM_JOB_ID": "816"}),
                    patch("cluster.transitions.queue", return_value=()),
                    patch("cluster.transitions.terminal_state") as terminal,
                    patch("cluster.transitions.submit_arrays", return_value=[{"array_id": "817"}]),
                    patch(
                        "cluster.transitions.arm_transition", return_value={"job_id": "818"}
                    ) as arm,
                ):
                    self.assertTrue(
                        recover_interrupted_inputs(
                            root, root, rules, {"rules": str(source)}, target
                        )
                    )
                terminal.assert_not_called()
                self.assertEqual(arm.call_args.args[3], ["816", "817"])
                self.assertEqual(
                    json.loads((run / "device_faults.json").read_text())["retry_count"], 1
                )
                self.assertEqual(
                    json.loads(
                        (root / "transitions/recovery" / f"{rules.identifier}.json").read_text()
                    )["runs"],
                    {},
                )

    def test_transition_device_budget_exhaustion_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.yaml"
            values = sample_recipe()
            values.update(id="device", seeds=[0])
            source.write_text(yaml.safe_dump(values))
            run = root / "runs/device/seed0"
            for index in range(4):
                reserve_retry(
                    root,
                    run,
                    "xgpd4",
                    str(820 + index),
                    str(820 + index),
                    0,
                    "No devices were found",
                )
            rules = replace(
                load_rules(REPOSITORY / "training/rounds/round2.yaml"), inputs=(source,)
            )
            with (
                patch("cluster.transitions.queue", return_value=()),
                patch("cluster.transitions.submit_arrays") as submit,
            ):
                with self.assertRaisesRegex(HoldRequired, "device-fault retry budget"):
                    recover_interrupted_inputs(root, root, rules, {}, root)
            submit.assert_not_called()

    def test_probe_checks_all_node_gpus_and_clears_only_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            environment = {
                "SLURM_JOB_NAME": "cz-ws3-probe-xgpd4",
                "SLURM_JOB_ID": "830",
                "SLURM_GPUS_ON_NODE": "2",
                "COLUMBA_RUN_PATH": str(root / "runs/gpu-probes/xgpd4"),
            }
            header = "name, driver_version, compute_cap, memory.total [MiB]\n"
            row = "NVIDIA TITAN V, 580.0, 7.0, 12050 MiB\n"
            with (
                patch.dict(os.environ, environment),
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.gpu_probe.platform.node", return_value="xgpd4"),
                patch("cluster.gpu_probe.node_resources", return_value=("nv", 2)),
                patch(
                    "cluster.gpu_probe.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, header + row * 2, ""),
                ),
                patch("cluster.gpu_probe.clear_bad_node") as clear,
            ):
                self.assertEqual(probe(True)["status"], "probe_complete")
                clear.assert_called_once()
            with (
                patch.dict(os.environ, environment),
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.gpu_probe.platform.node", return_value="xgpd4"),
                patch("cluster.gpu_probe.node_resources", return_value=("nv", 2)),
                patch(
                    "cluster.gpu_probe.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, header + row, ""),
                ),
                patch("cluster.gpu_probe.clear_bad_node") as clear,
            ):
                self.assertEqual(probe(True)["status"], "probe_failed")
                clear.assert_not_called()
            script = (REPOSITORY / "cluster/templates/gpu-probe.sbatch").read_text()
            self.assertIn("--time=00:01:00", script)
            self.assertNotIn("bootstrap", script)

    def test_probe_submission_requests_all_gpus_and_enforces_count_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            request = probe_submission("xgpd2", root, REPOSITORY)
            offers = parse_inventory(GPU_INVENTORY)
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.slurm.node_resources", return_value=("nv", 2)),
                patch("cluster.slurm.queue", return_value=()),
                patch("cluster.slurm.gpu_inventory", return_value=offers),
                patch(
                    "cluster.slurm.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "840\n", ""),
                ) as scheduler,
            ):
                self.assertEqual(
                    submit_pending([request], Budget(gpu_jobs=1))[0]["status"], "budget full"
                )
                scheduler.assert_not_called()
                self.assertEqual(
                    submit_pending([request], Budget(gpu_jobs=2))[0]["status"], "submitted"
                )
            self.assertIn("--gres=gpu:nv:2", scheduler.call_args.args[0])
            self.assertFalse(
                any(value.startswith("--exclude=") for value in scheduler.call_args.args[0])
            )

    def test_retarget_always_restores_array_throttle_after_gres_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = {"ArrayJobId": "850", "ArrayTaskThrottle": "2", "TimeLimit": "3-00:00:00"}
            jobs = parse_queue("850_0|cz-array-test|PENDING|gpu-long|gres/gpu:nv:1|1")
            with (
                patch("cluster.arrays.owned_job", return_value=fields),
                patch("cluster.arrays.queue", return_value=jobs),
                patch("cluster.arrays.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch(
                    "cluster.arrays.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as scheduler,
            ):
                result = retarget_array(root, "850", "h100-47")
            self.assertEqual(result["throttle"], 2)
            self.assertIn("TresPerNode=gres/gpu:h100-47:1", scheduler.call_args_list[0].args[0])
            self.assertEqual(
                scheduler.call_args_list[-1].args[0],
                ["scontrol", "update", "JobId=850", "ArrayTaskThrottle=2"],
            )
            with (
                patch("cluster.arrays.owned_job", return_value=fields),
                patch("cluster.arrays.queue", return_value=jobs),
                patch("cluster.arrays.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch(
                    "cluster.arrays.subprocess.run",
                    side_effect=[
                        subprocess.CalledProcessError(1, ["scontrol"]),
                        subprocess.CompletedProcess([], 0, "", ""),
                    ],
                ) as scheduler,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    retarget_array(root, "850", "h100-47")
            self.assertIn("ArrayTaskThrottle=2", scheduler.call_args_list[-1].args[0])

    def test_submit_commands_merge_the_actual_dynamic_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            checkout = root / "checkouts/source"
            (checkout / "cluster/templates").mkdir(parents=True)
            (checkout / "cluster/templates/nv.sbatch").write_text(
                (REPOSITORY / "cluster/templates/nv.sbatch").read_text()
            )
            (checkout / "cluster/excluded_nodes.txt").write_text("xgpd2\nxgpe2\nxgpe6\n")
            source = checkout / "recipe.yaml"
            values = sample_recipe()
            values.update(id="smoke-dynamic", seeds=[0])
            source.write_text(yaml.safe_dump(values))
            reserve_retry(
                root, root / "runs/bad/seed0", "xgpf6", "860", "860", 0, "No devices were found"
            )
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.arrays.queue", return_value=()),
                patch("cluster.arrays.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch(
                    "cluster.arrays.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, "1 project", ""),
                        subprocess.CompletedProcess([], 0, "861", ""),
                    ],
                ) as scheduler,
            ):
                submit_arrays(root, checkout, [task_for(source, 0)])
            self.assertIn(
                "xgpf6",
                next(
                    value for value in scheduler.call_args.args[0] if value.startswith("--exclude=")
                ),
            )
            request: Submission = {
                "name": "cz-legacy",
                "lane": "nv",
                "constraint": None,
                "script": "echo test",
                "run_path": str(root / "runs/legacy/seed0"),
            }
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.slurm.queue", return_value=()),
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch(
                    "cluster.slurm.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "862", ""),
                ) as scheduler,
            ):
                submit_pending([request], Budget())
            self.assertIn(
                "xgpf6",
                next(
                    value for value in scheduler.call_args.args[0] if value.startswith("--exclude=")
                ),
            )
            self.assertNotIn("--nodelist=xgpf6", scheduler.call_args.args[0])
            clear_bad_node(root, "xgpf6")
            self.assertNotIn("xgpf6", active_bad_nodes(root))
            self.assertTrue(set(excluded_nodes()) <= set(combined_exclusions(root)))

    def test_transition_waits_for_device_repair_and_requeued_task(self) -> None:
        for active_task in (False, True):
            with self.subTest(active_task=active_task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.yaml"
                values = sample_recipe()
                values.update(id="device", seeds=[0])
                source.write_text(yaml.safe_dump(values))
                run = root / "runs/device/seed0"
                reserve_retry(root, run, "xgpd4", "870_0", "870", 0, "No devices were found")
                ledger = json.loads((run / "device_faults.json").read_text())
                ledger["events"][0].update(status="repair_pending", repair_job_id="871")
                (run / "device_faults.json").write_text(json.dumps(ledger))
                target = root / "transitions/test"
                target.mkdir(parents=True)
                queue_text = (
                    "870_0|cz-array-test|RUNNING|gpu-long|gres/gpu:nv:1|1"
                    if active_task
                    else "871|cz-device-repair-test|PENDING|normal|N/A|1"
                )
                rules = replace(
                    load_rules(REPOSITORY / "training/rounds/round2.yaml"), inputs=(source,)
                )
                with (
                    patch.dict(os.environ, {"SLURM_JOB_ID": "872"}),
                    patch("cluster.transitions.queue", return_value=parse_queue(queue_text)),
                    patch("cluster.transitions.submit_arrays", return_value=[]),
                    patch(
                        "cluster.transitions.arm_transition", return_value={"job_id": "873"}
                    ) as arm,
                ):
                    self.assertTrue(
                        recover_interrupted_inputs(
                            root, root, rules, {"rules": str(source)}, target
                        )
                    )
                self.assertEqual(arm.call_args.args[3], ["872", "870_0" if active_task else "871"])
                self.assertEqual(
                    json.loads((run / "device_faults.json").read_text())["retry_count"], 1
                )

    def test_all_gpu_probe_refuses_an_impossible_type_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            request = probe_submission("xgpi11", home / "columba", REPOSITORY)
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.slurm.node_resources", return_value=("h100-47", 4)),
                patch("cluster.slurm.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch("cluster.slurm.subprocess.run") as scheduler,
            ):
                with self.assertRaisesRegex(ValueError, "per-type QoS cap"):
                    submit_pending([request], Budget())
            scheduler.assert_not_called()

    def test_heavy_array_uses_free_short_partition_then_long_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.yaml"
            values = sample_recipe()
            values.update(id="heavy", base_model="yolo11x.pt", seeds=[0], gpu_constraint="h100-47")
            path.write_text(yaml.safe_dump(values))
            offers = parse_inventory(
                "gpu-long|xgpi11|gpu:h100-47:4|cuda90,h100\ngpu|xgpi12|gpu:h100-47:4|cuda90,h100"
            )
            offers = tuple(
                replace(offer, free_units=0 if offer.partition == "gpu-long" else 4)
                for offer in offers
            )
            request = next(iter(group_tasks([task_for(path, 0)], offers).values()))[0]
            self.assertEqual(request.partitions, ("gpu",))
            self.assertEqual(request.time_limit, "03:00:00")
            request = next(
                iter(
                    group_tasks(
                        [task_for(path, 0)], tuple(replace(offer, free_units=4) for offer in offers)
                    ).values()
                )
            )[0]
            self.assertEqual(request.partitions, ("gpu-long",))


if __name__ == "__main__":
    unittest.main()


class FanoutTests(unittest.TestCase):
    def test_width_respects_expanded_project_and_association_capacity(self) -> None:
        from cluster.fanout import fanout_width

        self.assertEqual(fanout_width(3, 24, 5), 5)
        self.assertEqual(fanout_width(8, 24, 5), 3)
        self.assertEqual(fanout_width(12, 24, 5), 2)
        self.assertEqual(fanout_width(20, 24, 5), 1)
        self.assertEqual(fanout_width(2, 3, 5), 1)
        with self.assertRaisesRegex(RuntimeError, "submitted-task"):
            fanout_width(5, 4, 5)

    def test_nv_can_claim_all_eight_without_a_reserved_split(self) -> None:
        from cluster.array_budget import allocate_throttles

        self.assertEqual(allocate_throttles({"nv": 12}, {"nv": "nv"}, (), {}, 8), {"nv": 8})
        jobs = parse_queue("10|cz-other|PENDING|gpu|gres/gpu:h100-47:2|1")
        self.assertEqual(allocate_throttles({"nv": 12}, {"nv": "nv"}, jobs, {}, 8), {"nv": 6})
        jobs = parse_queue("10|moonshot-other|RUNNING|gpu|gres/gpu:h100-47:2|1")
        self.assertEqual(allocate_throttles({"nv": 12}, {"nv": "nv"}, jobs, {}, 8), {"nv": 8})

    def test_typed_and_generic_tres_are_not_double_charged(self) -> None:
        job = Job("10", "cz-double", "RUNNING", "gpu", "gres/gpu:h100-47=2,gres/gpu=2")
        self.assertEqual(job.gpu_count, 2)
        from cluster.slurm import resource_type_for_job

        self.assertEqual(resource_type_for_job(job), "h100-47")

    def test_claim_race_exactly_one_trains_and_logs_duplicate(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from cluster.run_claim import acquire

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/smoke-race/seed0"
            with (
                patch(
                    "cluster.run_claim.sibling_jobs",
                    return_value=(("10_0", "RUNNING"), ("11_0", "CONFIGURING")),
                ),
                patch("cluster.run_claim.subprocess.run") as cancellation,
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                futures = [
                    executor.submit(acquire, root, run, identifier, "cz-race")
                    for identifier in ("10_0", "11_0")
                ]
                self.assertEqual(sum(future.result() for future in futures), 1)
            cancellation.assert_not_called()
            events = json.loads(next((root / "claims").glob("*.json")).read_text())["events"]
            self.assertEqual(
                [event["event"] for event in events], ["claim_acquired", "duplicate_start"]
            )

    def test_stale_pending_owner_is_replaced_and_only_pending_siblings_cancelled(self) -> None:
        from cluster.run_claim import acquire

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/smoke-stale/seed0"
            with patch("cluster.run_claim.sibling_jobs", return_value=(("10_0", "RUNNING"),)):
                self.assertTrue(acquire(root, run, "10_0", "cz-stale"))
            with (
                patch(
                    "cluster.run_claim.sibling_jobs",
                    return_value=(("10_0", "PENDING"), ("11_0", "RUNNING"), ("12_0", "COMPLETED")),
                ),
                patch.dict(os.environ, {"USER": "owner"}),
                patch("cluster.run_claim.subprocess.run") as cancellation,
            ):
                self.assertTrue(acquire(root, run, "11_0", "cz-stale"))
            self.assertEqual(
                cancellation.call_args.args[0],
                ["scancel", "--state=PENDING", "--user=owner", "--name=cz-stale"],
            )
            report = json.loads((run / "run.json").read_text())
            self.assertEqual(report["fanout_claims"][-1]["cancelled_pending_siblings"], ["10_0"])

    def test_fanout_sets_are_typed_and_fit_the_short_partition(self) -> None:
        from cluster.fanout import requests_for

        with tempfile.TemporaryDirectory() as directory:
            light = OrchestrationTests().write_recipe(Path(directory) / "light.yaml")
            heavy = OrchestrationTests().write_recipe(
                Path(directory) / "heavy.yaml", "teacher", "yolo11x.pt"
            )
            offers = parse_inventory(GPU_INVENTORY)
            requests = requests_for(task_for(light, 0), offers)
            self.assertEqual(
                [request.resources for request in requests],
                ["gpu:nv:1", "gpu:a100-40:1", "gpu:h100-47:1"],
            )
            self.assertEqual(requests[0].constraint, "t4|titanv|titanrtx")
            heavy_requests = requests_for(task_for(heavy, 0), offers)
            self.assertEqual(
                [request.resources for request in heavy_requests],
                [
                    "gpu:h100-47:1",
                    "gpu:h100-96:1",
                    "gpu:a100-40:1",
                    "gpu:a100-80:1",
                    "gpu:h200-141:1",
                ],
            )
            self.assertEqual(heavy_requests[-1].partitions, ("gpu",))
            self.assertTrue(
                all(
                    request.time_limit == "03:00:00"
                    for request in heavy_requests
                    if "gpu" in request.partitions
                )
            )

    def test_fanout_submission_holds_until_copies_exist_and_preserves_shared_name(self) -> None:
        from cluster.fanout import submit_runs

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            checkout = root / "checkouts/pinned"
            path = OrchestrationTests().write_recipe(checkout / "recipe.yaml", "smoke-fanout")
            (checkout / "cluster/templates").mkdir(parents=True)
            (checkout / "cluster/templates/nv.sbatch").write_text(
                (REPOSITORY / "cluster/templates/nv.sbatch").read_text()
            )
            commands: list[Sequence[str]] = []

            def scheduler(
                command: Sequence[str], **keywords: JsonValue
            ) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                result = (
                    "1 project"
                    if command[0] == "du"
                    else str(100 + len(commands))
                    if command[0] == "sbatch"
                    else ""
                )
                return subprocess.CompletedProcess(command, 0, result, "")

            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.fanout.queue", return_value=()),
                patch("cluster.fanout.gpu_inventory", return_value=parse_inventory(GPU_INVENTORY)),
                patch("cluster.fanout.subprocess.run", side_effect=scheduler),
            ):
                records = submit_runs(root, checkout, [task_for(path, 0)])
            submitted = [command for command in commands if command[0] == "sbatch"]
            self.assertEqual(len(submitted), 3)
            self.assertEqual(
                len(
                    {
                        next(value for value in command if value.startswith("--job-name="))
                        for command in submitted
                    }
                ),
                1,
            )
            self.assertTrue(
                all(
                    "--hold" in command and "--nice=0" in command and "--time=00:01:00" in command
                    for command in submitted
                )
            )
            self.assertEqual(
                commands[-1],
                ["scontrol", "release", ",".join(str(record["array_id"]) for record in records)],
            )
            script = (REPOSITORY / "cluster/templates/nv.sbatch").read_text()
            self.assertLess(
                script.index("python -m cluster.run_claim"),
                script.index("cluster/bootstrap.sh --gpu"),
            )
            self.assertLess(
                script.index("python -m cluster.run_claim"),
                script.index("cluster.device_faults check"),
            )

    def test_targeted_throttle_cli_forwards_job_and_value(self) -> None:
        with (
            patch("sys.argv", ["sweep", "throttle-arrays", "--job", "951222", "--throttle", "2"]),
            patch("cluster.sweep.remote_root", return_value=Path("/columba")),
            patch(
                "cluster.sweep.deploy", return_value=(Path("/columba/checkouts/pinned"), "commit")
            ),
            patch("cluster.sweep.remote", return_value="[]") as remote,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            sweep_main()
        command = remote.call_args.args[0]
        payload = json.loads(shlex.split(command[-1])[-1])
        self.assertEqual(payload["job"], "951222")
        self.assertEqual(payload["throttle"], 2)

    def test_targeted_throttle_performs_scontrol_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "columba"
            manifest = root / "arrays/source/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"tasks": []}')
            record: dict[str, JsonValue] = {
                "array_id": "90",
                "task_count": 3,
                "throttle": 1,
                "resource_type": "nv",
                "manifest": str(manifest),
            }
            (manifest.parent / "submission.json").write_text(json.dumps(record))
            jobs = parse_queue("90_0|cz-old-array|RUNNING|gpu|gres/gpu:nv:1|1")
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("cluster.arrays.queue", return_value=jobs),
                patch("cluster.arrays.owned_job", return_value={"ArrayJobId": "90"}),
                patch("cluster.arrays.subprocess.run") as control,
            ):
                records = throttle_arrays(root, 8, "90", 2)
            self.assertEqual(records[0]["throttle"], 2)
            self.assertEqual(
                control.call_args.args[0], ["scontrol", "update", "JobId=90", "ArrayTaskThrottle=2"]
            )

    def test_requeue_refills_before_requesting_the_same_task_again(self) -> None:
        from cluster.requeue import main as requeue_main

        with (
            patch.dict(
                os.environ,
                {"SLURM_JOB_ID": "100", "SLURM_JOB_NAME": "cz-resume", "COLUMBA_FANOUT": "true"},
            ),
            patch(
                "sys.argv",
                ["requeue", "signal", "--run-path", str(Path.home() / "columba/runs/smoke/seed0")],
            ),
            patch("cluster.requeue.update_ledger", return_value=True),
            patch("cluster.fanout.refanout") as refill,
            patch("cluster.requeue.subprocess.run") as control,
        ):
            requeue_main()
        refill.assert_called_once()
        self.assertEqual(control.call_args.args[0], ["scontrol", "requeue", "100"])

    def test_nice_is_configurable_without_changing_the_qos_caps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            values = yaml.safe_load((REPOSITORY / "cluster/gpu_policy.yaml").read_text())
            values["nice"] = 1000
            path.write_text(yaml.safe_dump(values))
            self.assertEqual(load_policy(path).nice, 1000)
            values["nice"] = -1
            path.write_text(yaml.safe_dump(values))
            with self.assertRaises(ValueError):
                load_policy(path)

    def test_narrow_fanout_spreads_large_backlogs_across_free_type_caps(self) -> None:
        from cluster.fanout import narrow_requests, requests_for

        with tempfile.TemporaryDirectory() as directory:
            path = OrchestrationTests().write_recipe(
                Path(directory) / "heavy.yaml", "teacher", "yolo11x.pt"
            )
            offers = parse_inventory(GPU_INVENTORY)
            requests = requests_for(task_for(path, 0), offers)
            planned: dict[str, int] = {}
            choices = [
                narrow_requests(requests, 1, offers, (), planned)[0].resources for _ in range(8)
            ]
            self.assertGreaterEqual(len(set(choices)), 3)
            self.assertIn("gpu:a100-40:1", choices)
            self.assertIn("gpu:a100-80:1", choices)

    def test_transition_waits_for_a_new_winner_after_original_copies_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = OrchestrationTests().write_recipe(root / "source.yaml")
            rules = replace(OrchestrationTests().rules(), inputs=(source,))
            job = Job("900_0", job_name("nano", 0), "RUNNING", "gpu", "gres/gpu:a100-40:1")
            with (
                patch("cluster.transitions.queue", return_value=(job,)),
                patch("cluster.transitions.submit_arrays", return_value=[]),
                patch("cluster.transitions.arm_transition", return_value={}) as arm,
                patch("cluster.transitions.terminal_state") as terminal,
                patch.dict(os.environ, {"SLURM_JOB_ID": "999"}),
            ):
                self.assertTrue(
                    recover_interrupted_inputs(root, root, rules, {"rules": str(source)}, root)
                )
            terminal.assert_not_called()
            self.assertIn("900_0", arm.call_args.args[3])

    def test_claim_report_preserves_recipe_identity_before_runner_validation(self) -> None:
        from cluster.recipe import recipe_sha256
        from cluster.run_claim import acquire
        from cluster.run_job import validate_existing_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = OrchestrationTests().write_recipe(root / "recipe.yaml")
            recipe = load_recipe(path)
            run = root / "runs/nano/seed0"
            metadata: dict[str, JsonValue] = {
                "recipe_sha256": recipe_sha256(path),
                "dataset": recipe.dataset,
            }
            with patch("cluster.run_claim.sibling_jobs", return_value=(("10_0", "RUNNING"),)):
                self.assertTrue(acquire(root, run, "10_0", "cz-identity", metadata))
            validate_existing_run(run, recipe, recipe_sha256(path))

    def test_device_repair_replenishes_alternative_types_without_checkpoint_budget(self) -> None:
        from cluster.device_faults import refill_device_copies

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/smoke-device/seed0"
            event = reserve_retry(root, run, "xgpd3", "100_0", "100", 0, "No devices were found")
            with patch(
                "cluster.fanout.refanout_task", return_value=[{"array_id": "101"}]
            ) as refill:
                refill_device_copies(root, run, str(event["event_id"]), "100_0")
            refill.assert_called_once_with(root, "100_0")
            ledger = json.loads((run / "device_faults.json").read_text())
            self.assertEqual(ledger["retry_count"], 1)
            self.assertEqual(ledger["events"][0]["fanout_array_ids"], ["101"])
            self.assertFalse((run / "requeue.json").exists())
