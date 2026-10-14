# WS3 — Cluster infrastructure

- State: done
- Branch: `ws3/cluster`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14

## Done

1. Read the coordinator preamble, PLAN sections 0–6 and WS3, CONTRACTS v1
   and WS0 status. Read cluster etiquette twice. BatchMode SSH verified.
2. `cluster/bootstrap.sh`: architecture-aware uv, locked per-platform venvs,
   pinned Ultralytics/ncnn/pnnx, PyTorch 2.8.0 + torchvision 0.23.0, CUDA 12.6
   on x86_64, CPU on ARM. All temporary/cache paths stay in the project or the
   job's `/tmp/cz-*`. GPU architecture and YOLO forward/backward checks pass
   on T4, Titan RTX and Titan V.
3. C5 validator, schema and examples. `baseline-repro` preserves the v1-2
   args with runner-owned paths/batch/seed and periodic saves. Seeds map
   0=T4, 1=Titan RTX, 2=Titan V through `seed_gpu_constraints`.
4. Staged runner: verifies both archive SHA-256 sidecars, re-roots staged
   C1 YAML, caches base weights, stages the teacher, selects batch, persists
   checkpoints after every save and re-stages/re-writes checkpoint paths on
   resume. Completed training skips retraining when only evaluation is missing.
   Fixed WS1 val/test CLIs select the threshold on val and reuse it on the
   separate test dataset. Strict C6 completion writes DONE last.
5. nv, ARM and ARM bundle templates, all with nice 1000 and bounded resources.
   nv always requests `gpu:nv:1`; runtime rejects restricted GPU names.
6. `sweep` plan/submit/status/fetch/cancel plus ARM bundles. Submissions use
   immutable source snapshots and an NFS lock. Every sbatch rechecks squeue.
   Pending jobs reserve budget slots. Non-cz jobs cannot be cancelled.
   Fetch preserves incumbent rows and creates the C6 header when absent.
   No leaderboard is committed in this branch.
7. Three five-epoch GPU smoke runs completed on a tiny synthetic C1 fixture,
   followed by resubmission checks that re-staged data and retained completed
   checkpoints. All reports now show five epochs. Missing WS1 tools are
   explicitly recorded as evaluation_missing; DONE is absent.
8. Local checks: measured on the shared Mac M2 Pro, 66 unittest cases pass,
   Ruff check/format and strict mypy pass. Tests include mocked queue/sbatch
   budgets, cancellation, DONE diffing, validation, memory classes, teacher
   resolution, checkpoint relocation, final-validation callback regression,
   C1 extraction, incomplete-artifact failure and leaderboard upsert.

9. ARM NCNN fp16 export completed on xcng0, job 945206, input 384×640.
   Verified the copied param/bin/metadata and bundle DONE. No WS3 jobs remain
   queued/running. Final project footprint measured from xlogin1: 18 GiB,
   including the other workers' files under ~/columba, below the 80 GiB cap.

## In progress

Nothing. WS3 tasks 1–5 and the smoke acceptance criteria are complete.

## Blockers and requests

- WS1 evaluator and WS2 real datasets are not merged in this checkout. Their
  interfaces are implemented and unit-tested; real pigeon evaluation cannot
  be claimed. The synthetic smoke archives are named `smoke-coco8` and
  `smoke-coco8-test` for the requested bootstrap path, but contain CC0 circles,
  not COCO photographs. They make no accuracy claim.
- Coordinator: retain the `seed_gpu_constraints` C5 extension in CONTRACTS.
  Optional `test_dataset` defaults to columba-test-v2; only smoke- IDs may set
  `allow_missing_evaluation: true`. The schema documents both fields.
- `xgpe6` failed CUDA initialisation with torch 2.8.0+cu126. The same wheel
  succeeded on Titan RTX xgpe4. Use `--exclude-nodes xgpe6` until the node is
  checked. No restricted hardware was requested or used.
- Initial bootstrap imported Ultralytics before setting scratch settings and
  created `/home/a/ashelow/.config/Ultralytics/settings.json`. Fixed bootstrap
  so every import gets a job-local/project-local config with sync false. I
  have not modified or deleted that global file afterwards; the coordinator
  can review it. Initial scratch uv extraction on xlogin1 also failed in
  `/tmp`; subsequent installer temporary files stay under `~/columba/`.
- WS6 adapter extension is `training.adapters.<framework>.adapter.train` with
  `TrainingContext`. C6 evaluation currently uses WS1's Ultralytics predictor;
  non-Ultralytics checkpoint formats need a prediction adapter agreed with WS1.

## Results

All GPU timings below are measured training+validation wall times, including
checkpoint persistence, on four synthetic train and four val images at 640.
The five-epoch mean includes cold startup in the first epoch. These are
infrastructure smoke results, not throughput estimates for the real dataset
and not Pi latency results.

- T4, xgpf10, training job 945186, seed0, batch 32:
  epoch seconds `[11.11054, 0.96748, 0.95846, 0.92353, 1.21610]`;
  mean **3.03522 s/epoch**, epochs 2–5 mean **1.01639 s/epoch**.
- Titan RTX, xgpe4, training job 945187, seed1, batch 64:
  epoch seconds `[11.13228, 0.98078, 0.95798, 0.95924, 1.10647]`;
  mean **3.02735 s/epoch**, epochs 2–5 mean **1.00112 s/epoch**.
- Titan V, xgpd2, training job 945188, seed2, batch 32:
  epoch seconds `[12.49154, 1.22847, 1.22999, 1.11476, 1.11917]`;
  mean **3.43679 s/epoch**, epochs 2–5 mean **1.17310 s/epoch**.

The T4 exposed 14.56 GiB usable memory and initially selected batch 32. The
final heuristic recognises that usable-memory class and starts future T4 nano
640 runs at 48; V starts at 32 and RTX at 64. The measured smoke batches above
remain unchanged. nbs always stays 64.

Measured wheel architectures on all three GPU types:
`sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90`. Device architectures 7.0 and
7.5 are covered. Python 3.12.3, torch 2.8.0+cu126, CUDA runtime 12.6,
Ultralytics 8.4.175, NVIDIA driver 580.178.04.

Artifacts:

- Cluster C6 training folders: `~/columba/runs/smoke-5ep/seed{0,1,2}/`.
- Local fetched C6 files: `experiments/runs/smoke-5ep/seed{0,1,2}/`.
- Local smoke leaderboard: `outputs/ws3/leaderboard.csv` (not committed).
- Smoke archives/manifests: `outputs/ws3/data/`, uploaded to `~/columba/data/`.
- ARM bundle: `~/columba/runs/arm-bundles/b36aec4a8c50/`; local copy
  `outputs/ws3/arm-export/`, including `run.json`, DONE and exported weights.
- Queue/report snapshot: `outputs/ws3/status-final.jsonl`.
- Local checks: `outputs/ws3/unit-all.log`, `outputs/ws3/mypy.log`.
- Bootstrap/Slurm logs: `~/columba/logs/cz-smoke-*` and `bootstrap-*.json`.

Initial failed GPU jobs 945158/945160 exposed a standalone-loss hyperparameter
namespace bug, fixed before the successful runs; 945159 exposed xgpe6's CUDA
initialisation failure. Ultralytics fires on_fit_epoch_end again during final
validation with an incremented epoch; the runner now excludes that callback
and reconciles completed runs against results.csv. The first ARM job 945198
found that the NFS `~/.local/bin/uv` was x86_64. Bootstrap now uses
`~/columba/bin/aarch64/uv` on ARM without replacing the x86 binary.

Measured ARM export on xcng0: **37.24322 s** around the export call including
model load; Ultralytics reports **35.5 s** for NCNN conversion. fp16, input
384×640; torch 2.8.0+cpu, Python 3.12.3. No Pi inference claim.

Environment paths:

- `~/columba/venvs/x86_64-cu126`
- `~/columba/venvs/aarch64-cpu`

WS5 commands from the repository root, using the shared venv's Python:

```sh
python -m cluster.sweep plan --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep submit --recipes training/recipes/baseline-repro.yaml --exclude-nodes xgpe6
python -m cluster.sweep status --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep fetch --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep cancel --job-ids <cz-job-id>
```

Schema: `training/recipes/schema.md`; validator: `cluster.recipe.load_recipe`.
Teacher resolution: `cluster.recipe.resolve_teacher(reference, runs_root)`.
Runner: `python -m cluster.run_job --recipe <path> --seed <k>` inside a cz-
Slurm job. ARM bundle: `python -m cluster.sweep arm-bundle --models
smoke-5ep/seed0 --bundle-action export`.
