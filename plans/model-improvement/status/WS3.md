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

Nothing. Round 2 completed within the two-GPU/one-CPU WS3 limit.

## Blockers and requests

- Round 1 lacked WS1 tools; round 2 fast-forwarded to main 222ba8f and
  verified prediction/evaluation against their actual CLI. No WS1 change is
  required. Synthetic smoke results make no pigeon accuracy claim. The
  archives `smoke-coco8` and `smoke-coco8-test` contain CC0 circles, not COCO
  photographs; real dataset/model evaluation is outside this smoke check.
- C5 extensions were accepted in CONTRACTS v1.3. `test_dataset` defaults to
  columba-test-v2; only smoke- IDs may allow missing evaluation. Round-2
  recipes set `allow_missing_evaluation: false`.
- `xgpe6` failed CUDA initialisation with torch 2.8.0+cu126. The same wheel
  succeeded on Titan RTX xgpe4. Use `--exclude-nodes xgpe6` until the node is
  checked. No restricted hardware was requested or used.
- Initial bootstrap imported Ultralytics before setting scratch settings and
  created `/home/a/ashelow/.config/Ultralytics/settings.json`. Fixed bootstrap
  so every import gets a job-local/project-local config with sync false. I
  did not modify or delete that global file afterwards; the coordinator
  removed it before round 2. Initial scratch uv extraction on xlogin1 also failed in
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

## Round 2 verification

Fast-forwarded `ws3/cluster` to main **222ba8f**. WS1's actual predictor and
importable evaluator are present, and their CLI matches the runner's commands.
No change to WS1-owned files was needed. Code was deployed through immutable
checkout snapshots before submitting. The E2E run records code commit 710cc0b
and snapshot c090d8afa6e2ff70; distillation records 2ab7d29.

### Fresh end-to-end run

Measured on **xgpf11, Tesla T4**, job **945281**, recipe **smoke-e2e**, seed 0:
five fresh epochs at 640, batch **48**, nbs **64**, real val/test predictions
and evaluations, `status: complete`, `evaluation: complete`, exit status 0.
All required C6 files are non-empty and JSON parses; DONE was written last.
Local copy: `experiments/runs/smoke-e2e/seed0/`. Cluster copy:
`~/columba/runs/smoke-e2e/seed0/`.

```text
run.json
weights/best.pt
weights/last.pt
results.csv
predictions_val.json
predictions_val.meta.json
predictions_test.json
predictions_test.meta.json
metrics_val.json
metrics_test.json
TRAINED
DONE
```

Measured synthetic metrics excerpt from metrics_test.json:

```json
{
  "pigeon": {
    "ap50": 0.7613154172560117,
    "ap50_95": 0.7613154172560114,
    "recall_at_p90": 0.5,
    "precision_at_threshold": 1.0,
    "recall_at_threshold": 0.5
  },
  "operating_threshold": {
    "value": 0.7773585915565491,
    "source": "/tmp/cz-945281/runs/smoke-e2e/seed0/metrics_val.json"
  }
}
```

Val selected that same threshold, with `source: selected`, `attainable: true`
and `split: val`. Test inherited it. There are four images per split; empty
and confuser FAR are null because those subsets are absent, not measured zero.
Epoch seconds: `[10.11575, 0.86145, 0.83441, 0.83048, 0.98604]`; measured mean
**2.72562 s/epoch**, including cold startup and checkpoint persistence. This
is a pipeline check, not a model-quality or Pi result.

### Distillation and optimiser resume

Recipe `smoke-distill-resume`, seed 0, three epochs, `dis: 1.0`, batch **24**,
`teacher: smoke-e2e/seed0`. Two actual Slurm allocations:

- Job **945307**, **xgpf10 / T4**: intentionally interrupted after saved epoch
  one with `--stop-after-epoch 1`. `last.pt` has epoch 0, optimiser state, and
  absolute `distill_model: /tmp/cz-945307/weights/teacher.pt`. DONE absent.
- Job **945320**, **xgpf11 / T4**: resubmitted the unchanged recipe without
  the stop flag. The log says "Resuming training ... from epoch 2 to 3 total
  epochs". `run.json.resumed: true`, epochs_done 3, exit 0, full evaluation
  and DONE. results.csv contains epochs `[1, 2, 3]`, not restarted epochs.

The teacher SHA-256 matches the E2E best.pt. The node-staged teacher path is
absolute in both attempts. `checkpoint_relocation` records:

```json
{
  "previous_data": "/tmp/cz-945307/data/smoke-coco8/data.yaml",
  "previous_distill_model": "/tmp/cz-945307/weights/teacher.pt",
  "data": "/tmp/cz-945320/data/smoke-coco8/data.yaml",
  "distill_model": "/tmp/cz-945320/weights/teacher.pt"
}
```

Measured train/dis_loss: **0.77995, 0.77994, 0.79127**. The runner retains
both attempts, including the deliberate interruption, in `run.json.attempts`.
Cluster folder `~/columba/runs/smoke-distill-resume/seed0/`; local folder
`experiments/runs/smoke-distill-resume/seed0/`.

### Pinned node probe and leaderboard guard

Measured on **xgpe6**, probe job **945282**, with nv:1 and nodelist=xgpe6:
`nvidia-smi` returned **exit 6**, "No devices were found";
`torch.cuda.is_available()` returned **false**, CUDA_VISIBLE_DEVICES=0,
torch 2.8.0+cu126, CUDA runtime 12.6. No node changes were made. Training
submissions exclude xgpe6 by default; the explicit probe bypasses that list.
Report: `outputs/ws3/xgpe6-probe.json`, cluster
`~/columba/runs/gpu-probes/xgpe6/run.json`.

Both successful C6 runs were fetched with
`--leaderboard outputs/ws3/leaderboard-smoke.csv`. The fetch guard excludes
smoke-* from any experiments/leaderboard.csv, including another worktree's
path and a resolved symlink. Unit tests prove incumbent bytes stay unchanged
while scratch leaderboards accept smoke rows. No real leaderboard was written
or committed in this round.

### Checks and evidence

Measured on the shared Mac M2 Pro: **106 unittest cases pass**, Ruff check and
format check pass on app/tests/tools/cluster/training, strict mypy passes on
43 files. No network in unit tests. No WS3 jobs remain running or queued.

Evidence in this worktree:

- `outputs/ws3/e2e-artifact-check.json`
- `outputs/ws3/checkpoint-before-resume.json`
- `outputs/ws3/distill-before-resume/` (unstripped checkpoint snapshot)
- `outputs/ws3/distill-resume-evidence.json`
- `outputs/ws3/leaderboard-smoke.csv`
- `outputs/ws3/xgpe6-probe.json`
- `outputs/ws3/unit-round2-all.log`, `outputs/ws3/mypy-round2.log`

Re-run commands are documented in cluster/README.md. The smoke-only runtime
stop flag does not change recipe/checkpoint args and cannot be used with a
production recipe. No outstanding round-2 blocker or WS1 change request.
