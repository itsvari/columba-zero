# WS3 — Cluster infrastructure

- State: done
- Branch: `ws3/cluster`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-15

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

Nothing. Round 3 completed; no WS3 jobs remain queued or running.

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

## Round 3

Fast-forwarded to main a29e4aa. User-authorised all listed GPU types and xgpj0.
Implemented inventory-based requests, counted GPU budgets, atomic deployment
and an aarch64 CUDA environment. No xgpj0 exclusion will be added.
Initial queue on xlogin2 has all eight WS5 GPUs running; no WS5 job was changed.
GPU budget slots were freed under the later explicit cancellation authorisation.

### Round-3 implementation and cancellation ledger

- Inventory-driven selectors now cover nv, A100-40/80, H100-47/96 and H200-141,
  with a100/h100 family selectors. Explicit types prefer gpu-long; xgpj0 and
  H200 use gpu (three hours). xgpd0–1 and xgpf0–5 are also gpu only.
- Automatic null/CLI any selection reads cluster/gpu_policy.yaml and sinfo
  free units, chooses a concrete preferred type/partition and bounds its node
  list to one node. It skips submission when no preferred free capacity exists. PENDING
  allocations reserve the eight-GPU budget. Multi-node typed GRES are counted
  by device count. The CPU and total-job caps remain two and twenty-four.
- Light preferences: T4, Titan V, Titan RTX, H100-47. Heavy preferences:
  H100-47, H100-96, opportunistic xgpj0/A100-80, then nv. Heavy means x model,
  imgsz >= 800, a teacher/distillation run, or estimate >= four GPU-hours.
- --retarget-pending-after-hours opts into cancelling only requested expired
  PENDING cz- runs when alternate free preferred capacity exists. Slurm's
  state filter prevents cancellation if the run starts during the check.
- xgpe2/xgpe6 remain excluded. xgpj0 is allowed, not excluded. It advertises
  gpu and test, gpu:a100-80:1, 80 CPUs, 128000 MB, features xgpj/cuda80/a100.
  test requires the testusers account/test QoS and is not used by our scheduler.
- Atomic checkout publication stages a complete archive and ready marker,
  then renames; published checkouts are never re-extracted. Snapshot updates
  cannot invalidate a running job's bootstrap inode.
- x86_64 CUDA remains torch 2.8.0+cu126 / torchvision 0.23.0. ARM CUDA is
  torch 2.9.0+cu126 / torchvision 0.24.0 (2.8 CUDA SBSA wheels absent in the
  cu126 index). ARM CPU remains 2.8.0+cpu. Ultralytics/ncnn/pnnx pins remain
  8.4.175 / 1.0.20260526 / 20260526. Python 3.12 aarch64 ncnn and
  opencv-python-headless wheel availability was verified from PyPI metadata.
- ARM CUDA prewarm on xcng0, 946760: COMPLETED 0:0 in 12m55s, installed
  ~/columba/venvs/aarch64-cu126. No GPU was requested for prewarm.

Authorised cancellations (confirmed cz- names immediately before scancel;
reported saved epoch from the runner's on_model_save heartbeat):

| Job | Run | Last saved epoch | weights/last.pt |
|---|---|---:|---|
| 945884 | yolo11x-v2s-960-teacher/seed0 | 10 | exists |
| 945885 | yolo26x-v2s-960-teacher/seed0 | 4 | exists |
| 946639 | yolo11s-v2-640-r1/seed0 | 32 | exists |

No other WS5 or non-cz job was cancelled. WS5 reports still say running because
Slurm terminated the batch allocation; the queue is authoritative. Resubmission
rewrites the report and resumes last.pt. Other WS5 jobs 946641 and 946662–665
remain running. WS3 has not resubmitted WS5's work. Evidence:
outputs/ws3/cancellations-round3.json and cancelled-and-distill-reports.json.

### Round-3 measured results

Same synthetic four-train/four-val C1 fixture, YOLO11n at 640. Means include
cold startup and checkpoint persistence; warm means use epochs 2–5. These do
not measure real-dataset throughput or pigeon accuracy.

| Type / host | Job | Batch chosen | Mean s/epoch | Warm s/epoch |
|---|---|---:|---:|---:|
| H100-47 MIG / xgpi16 | 946777 | 64 | 2.28689 | 0.49123 |
| A100-40 / xgpg6 | 946782 | 64 | 1.56226 | 0.77724 |
| A100-80 aarch64 / xgpj0 | 946781 | 64 | 2.96276 | 1.56360 |
| T4 / xgpf10 (round 1) | 945186 | 32 | 3.03522 | 1.01639 |
| Titan V / xgpd2 (round 1) | 945188 | 32 | 3.43679 | 1.17310 |
| Titan RTX / xgpe4 (round 1) | 945187 | 64 | 3.02735 | 1.00112 |

H100 and A100 bootstrap forward/backward passed with torch 2.8.0+cu126,
sm_90 and sm_80 respectively. Both completed five epochs, real WS1 prediction
and evaluation on val/test, all C6 artifacts and DONE, fetched into
experiments/runs/ using outputs/ws3/leaderboard-smoke.csv. No production
leaderboard was changed. A100 queued about eight minutes, then started;
it did not need cancellation. Slurm's site submission hook moved its short
20-minute gpu-long request to gpu (SubmitLine retained gpu-long).

C6 listing for each successful seed0 folder: DONE, TRAINED, run.json,
results.csv, weights/{best,last}.pt, predictions_{val,test}.json,
predictions_{val,test}.meta.json, metrics_{val,test}.json.

Cross-GPU distillation/resume:
smoke-distill-gpu-change-r3/seed0, teacher smoke-h100-r3/seed0. Job 946798
on Titan V xgpd4 intentionally stopped after saved epoch one, batch 16.
946808 resumed on H100-47 xgpi16, completed epoch three with batch 64,
val/test metrics and DONE. The report retains both attempts and rewrites
absolute data/teacher paths from /tmp/cz-946798 to /tmp/cz-946808.
Ultralytics 8.4.175 explicitly allows batch and device overrides on resume;
the assertion that resume always keeps the old batch is incorrect for our pin.
The runner reapplies its heuristic, but a recipe's fixed batch_override wins.
The two teacher recipes currently fix batch 4 and 6, so their resumes retain
those values; changing recipe configuration under an existing ID is rejected.
A fresh recipe ID is required if WS5 changes those fixed overrides. Resuming
with the current batches on faster GPUs is supported and avoids losing epochs.

### Handoff commands and C5 request

After merge, WS5 can resume all missing round-1/teacher runs under live policy:

```sh
python -m cluster.sweep plan --recipes training/recipes/*-v2*-640-r1.yaml training/recipes/*teacher*.yaml
python -m cluster.sweep submit --recipes training/recipes/*-v2*-640-r1.yaml training/recipes/*teacher*.yaml --constraint any --retarget-pending-after-hours 2
```

Repeat submit as slots become free. Active runs and DONE markers are skipped.
No worker restarted the paused auto-submitter; the coordinator owns that.

C5 wording request: gpu_constraint and seed_gpu_constraints values may be
null, t4, titanrtx, titanv, a100-40, a100-80, h100-47, h100-96, h200,
a100 or h100. A per-seed null explicitly restores automatic selection.
lane: nv is the GPU lane for x86_64 and aarch64. Null means the live, configured
preference policy, rather than an untyped request that can queue anywhere.
run.json records architecture and the selected venv. Existing recipes and
sweep syntax remain compatible.

Completed: xgpj0 GPU probe and five-epoch CUDA smoke (946781).
A transient SSH tunnel reset paused cluster calls; the coordinator restored
it, and calls now use ConnectTimeout=20. No password or tunnel changes were
made by WS3. Final checks and ARM results are recorded below.

Live re-target test exposed an initial node-list bug: Slurm interprets every
entry in --nodelist as required, rather than alternatives. Own pending job
946824 (smoke-policy-retarget-r3/seed0, epoch 0, no last.pt) was cancelled by
the pending-state filter; its replacement 946836 erroneously requested six
nodes/GPUs and stayed PENDING/QOSMaxGRESPerUser. This transient request exceeded
the project pending-GPU budget; it never ran. Confirmed its cz- name and
immediately cancelled 946836 (same run, epoch 0, no last.pt). The correction
selects one free node and explicitly sends --nodes=1. A regression test checks
multiple free nodes still produce one GPU/node. The repeated live check passed.

Unmodified Slurm test-only probes verified that untyped gpu:1 requests pinned
to xgpj0 or constrained to a100/h100 are rejected (Requested node configuration
is not available); typed xgpj0 requests succeed. The earlier untyped NV test
only proved nv matching. Family selectors now resolve to a concrete family
variant, and pinned probes infer the node's typed GRES. Null must use policy;
no submit path emits untyped GPU requests.

Checkpoint audit job 946861 on xcng0 loaded all three cancelled WS5 last.pt
files on CPU: saved epochs exactly 10, 4 and 32, batches 4, 6 and 16, all with
optimiser state present. It also cleaned the prewarm job's /tmp/cz-946760.
Evidence: ~/columba/logs/checkpoint-audit-round3.json.

The corrected live pending re-target cancelled own 946856
(smoke-policy-retarget-r3/seed0, epoch 0, no last.pt) and submitted 946858
as one nv GPU on one node (T4 xgpf0, batch 48). It completed C6/DONE and
was fetched. Own xgpj0 probe 946872 and ARM smoke 946781 both completed.
All remaining job allocations count correctly under the eight-GPU budget,
including pending jobs.

### Round-3 final evidence

- xgpj0 smoke 946781: COMPLETED 0:0 in 1m23s. CUDA forward/backward passed,
  five epochs, both predictions/metas/metrics, DONE. Architecture aarch64;
  torch 2.9.0+cu126, CUDA runtime 12.6, driver 580.178.04; device sm_80,
  wheel architectures sm_80/sm_90. No ARM GPU-specific failure occurred.
- Formal xgpj0 probes 946872 and final 946894: COMPLETED 0:0 (16s and 14s).
  nvidia-smi exit 0, CUDA available true, A100 80GB PCIe. Final probe used
  committed implementation 2153f8a, cleaned its own job scratch plus the
  earlier probe's /tmp/cz-946872. The standard probe template now has EXIT
  cleanup. CPU audit cleaned the prewarm scratch as well.
- No cancellation of 946781 occurred: an attempted pending-only replacement
  check found it had started and refused to cancel. Its original request
  finished successfully. Future synthetic smoke submissions use four CPUs,
  16 GB and ten minutes to fit short scheduling gaps.
- H200 xgpk0 was still four-of-four allocated and planned for other work
  (mixed-); no H200 job was queued. H100-96 and x86 A100-80 were not separately
  benchmarked. Their mapping, guard names and memory classes are unit-tested.
- Local final checks on the shared Mac: 239 unittest cases run (seven
  dependency/hardware skips); all executed cases pass, Ruff check/format pass on app/tests/tools/
  cluster/training/native, strict mypy passes on 79 files. Unit tests have
  no network. All 45 cluster tests pass without skips.
- Final queue contains only the five continuing WS5 jobs 946641 and
  946662–946665. No WS3 jobs or local detached processes remain. The
  coordinator owns restarting the paused auto-submitter.

Synthetic ARM metrics excerpt (not a pigeon-accuracy result):

```json
{
  "val": {"ap50": 0.7613154172560117,
          "operating_threshold": {"value": 0.7790383100509644, "source": "selected"}},
  "test": {"operating_threshold": {"value": 0.7790383100509644,
           "source": "/tmp/cz-946781/runs/smoke-a100-arm-r3/seed0/metrics_val.json"}}
}
```

Fetched C6 artifacts: experiments/runs/{smoke-h100-r3,smoke-a100-r3,
smoke-a100-arm-r3,smoke-distill-gpu-change-r3,smoke-policy-retarget-r3}/seed0/.
All have DONE, TRAINED, run.json, results.csv, best/last weights, predictions
and metadata for both splits, and both metrics. Every test threshold equals
its val-selected threshold. Smoke leaderboard: outputs/ws3/leaderboard-smoke.csv.
No leaderboard file was committed or production leaderboard updated.

Evidence: outputs/ws3/xgpj0-probe.json, checkpoint-audit-round3.json,
policy-{pending,retarget}-repeat.json, final-deployment.json,
status-final-round3.jsonl, unit-round3.log, ruff-round3.log and mypy-round3.log.
The immutable snapshots used by earlier smoke jobs include their draft content
hash and then-current base commit; corrected re-target job used 13fa1db and
final typed probe used 2153f8a. Final source publication after the status commit
is recorded in final-deployment.json. Only CONTRACTS C5 wording remains for the
coordinator to update; no implementation blocker remains.
