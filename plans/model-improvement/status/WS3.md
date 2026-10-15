# WS3 — Cluster infrastructure

- State: in progress
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

## Round 4 in progress

Fast-forwarded to main 04e9980. Implementing finite Slurm arrays and dependency
transitions; no persistent login or compute orchestrator. Current jobs remain
untouched. Verified normal QoS typed GPU caps and association MaxSubmitJobs=32.
Round-2 rules will be drafts requiring WS5 confirmation before explicit arming.

## Round 4 complete — cluster-side orchestration

Implemented finite GPU arrays and finite arm/normal dependency jobs. No daemon,
login polling process, sleeping compute orchestrator, tmux, tunnel or remote-access
service was created. Branch remains ws3/cluster; main was fast-forwarded to
04e9980 before implementation. Production transitions remain **unarmed**.

### Implementation and limits

- Array manifests map task index to recipe/seed and pin recipe SHA-256 plus an
  immutable checkout. Active legacy/array runs and DONE runs are skipped.
  Resubmission returns existing active array receipts, so dependent jobs keep
  the correct dependencies after a partial submission/retry.
- Expanded squeue tasks count against the project submitted-task cap of 24.
  All account jobs count towards the **verified** association MaxSubmitJobs=32.
  Approved array backlog is held by Slurm; running GPU limits are eight total
  and the typed normal QoS caps. This differs from legacy single-job admission,
  which also limits pending GPU requests.
- **Verified** normal QoS caps: nv=8, A100-40=4, A100-80=2, H100-47=2,
  H100-96=1, H200-141=1, H200-71=1, CPU=256. Project CPU jobs remain capped at
  two. The data-driven array shares are nv=4, H100-47=2, H100-96=1, A100-80=1.
  Shares are recalculated across active and newly submitted arrays, respecting
  total/type caps. A persisted `throttle-arrays --gpu-limit 4` control also
  constrains future submissions. It does not cancel existing running work.
- Light arrays request typed nv with a t4|titanv|titanrtx feature OR-list. Heavy
  arrays rotate through the declared heavy shares. Explicit recipe/seed/CLI
  pins work; --constraint any removes a pin. A100-80 automatic groups accept
  gpu-long,gpu at three hours, retaining aarch64 xgpj0 support. Node exclusions
  remain xgpe2/xgpe6. The scheduler changed our twenty-minute gpu-long smoke
  requests to gpu; manifests and SubmitLine preserve the original request.
- GPU templates request --requeue and --signal=B:USR1@600. A signal finishes
  and persists the current saved epoch, then requeues the specific array task.
  Startup guards also cover scheduler node-failure restarts. A signal during
  bootstrap is retained for the trainer, and an interrupted bash wait always
  retrieves the child exit status even if the child has already finished. Per-run
  requeue.json permits ten lifetime requeues, requires checkpoint progress and
  records HOLD when a loop would occur. Batch/data/teacher relocation on resume
  still works; current memory selection or fixed batch_override is reapplied.
- Transition recovery covers required round-1 inputs, later plain arms and
  teachers: missing DONE after TIMEOUT/PREEMPTED/NODE_FAIL with last.pt and a
  progressed saved epoch is resubmitted through arrays. The transition re-arms
  after recovery. Its separate per-round recovery ledger allows two attempts
  per run, then HOLD. Genuine failures, missing C6 metrics after DONE, uncovered
  ties, bad latency evidence, insufficient eligible runs or budget failures HOLD.
- The confirmed plain transition consolidates leaderboard/ranking evidence,
  validates generated C5 recipes, submits arrays and declares a second
  transition after the plain arrays and both teachers. The distilled stage
  validates the declared upstream round, all plain seeds and both teachers.
  It also consolidates the finished plain rows. Round 3/WS7 stay manual.
- status includes expanded tasks, throttles, epochs, requeue counts, HOLD and
  transition outcomes, including terminal sacct state. fetch copies manifests,
  ledgers, transition records and generated YAML to outputs/ws3/cluster-state;
  --generated-round fetches their C6 runs. Smoke rows use a scratch leaderboard.

### Live verification (measured)

| Job/array | Work | Result |
| --- | --- | --- |
| 947089 | <=1-minute normal/xcng0 scheduler verification | COMPLETED 0:0; final allocation elapsed 2 s; one successful scontrol requeue, SLURM_RESTART_COUNT=1; in-job squeue and sbatch --test-only passed |
| 947128_0, 947128_1 | first two-task three-epoch smoke | Both DONE and COMPLETED 0:0; elapsed 59/56 s on final attempts, restart count 1 each |
| 947161_0, 947161_1 | corrected task-specific USR1/requeue smoke | Both DONE and COMPLETED 0:0; elapsed 55/51 s, restart count 1 each; saved epoch 1 -> resumed -> epoch 3 |
| 947181 -> 947182_0 | finite consolidation/generation -> one generated follow-up | CPU job COMPLETED in 30 s; generated GPU task DONE/COMPLETED in 50 s |
| 947213_0, 947213_1 -> 947214 -> 947219_0 | live dependency chain, two-task array -> arm CPU transition -> generated one-task array | All COMPLETED 0:0; parent tasks 51/52 s, CPU 38 s, follow-up 51 s; actual afterany:947213 dependency was recorded and waited before CPU execution |

The original smoke exposed that a numeric job ID can be the array root for one
of its tasks. `scontrol requeue <numeric root>` can affect the whole array. Fixed
this to always target <array>_<task>, and added regression coverage for both
root-sharing and separately numbered tasks. The corrected smoke proved one
requeue per task and resume on both. run.json retains both attempts and the
checkpoint relocation. Signal interruption, ledger updates and Slurm restart
counts are real; recovery failure/cap cases are mocked unit tests.

Measured synthetic timings on Titan V (infrastructure only): corrected 320-pixel
three-epoch runs averaged 7.952/7.670 s per epoch, batch 64. The final generated
640-pixel two-epoch run averaged 5.374 s per epoch, batch 32. These short runs
include warm-up effects and are not comparisons with production training or
Pi inference. No accuracy claims arise from synthetic smoke data.

Final generated C6 folder:
`~/columba/runs/smoke-chain-r4-640-smoke-chain-followup-r4/seed0/`.
Fetched to the matching experiments/runs directory. Artifacts: DONE, TRAINED,
run.json, results.csv, weights/best.pt, weights/last.pt, predictions_val.json,
predictions_val.meta.json, predictions_test.json, predictions_test.meta.json,
metrics_val.json and metrics_test.json. Both val and test synthetic R@P90 are
0.5; val selected threshold 0.7774906754493713, and test reused that exact value
from metrics_val.json. All eight requested smoke runs were fetched to the
scratch leaderboard outputs/ws3/leaderboard-smoke.csv. No production leaderboard
was updated or committed.

### Cancellation record and migration

Only manual cancellation in round 4: **947130**, the pending initial smoke
transition for smoke-followup-r4, cancelled after confirming its cz- name and
PENDING state, to replace the array-root requeue test. It was a CPU transition,
so recipe/seed, saved epoch and weights/last.pt are not applicable. No training
job was manually cancelled. The original array tasks had started between the
queue check and attempted pending-only retirement; the safety check refused
the cancellation and they completed from their preserved epoch-1 checkpoints.
Their scheduler-triggered requeues are recorded above and in each requeue.json.

947094 (yolo26s-v2s-640-r1) remains on the old path as approved. No running WS5
job was cancelled, resubmitted or modified. Current jobs at final verification:
946641, 946662, 946664, 946665, 947094, 946961 and 946962 all RUNNING. In
particular the H100-96 teacher 946961 has started. No WS3 job remains queued.
The Mac auto-submitter stays stopped; the coordinator's periodic check stays
paused. Neither was edited or restarted.

### Production hand-off — not armed

Prepared plan (no submission):
`~/columba/transitions/6e2e08c19c779b72/plan.json`.
Round-1 dependencies: 946641, 946662, 946663, 946664, 946665, 946963, 947094.
Teacher dependencies for the second stage: 946961, 946962. All eight declared
v2/v2s source combinations must have complete C6 evidence, including the already
finished yolo11n-v2 source. Completed dependency IDs no longer in squeue are
recorded but removed from the scheduler request; C6 checks still apply.

After WS5 confirms **both** training/rounds/round2.yaml and round2-distill.yaml
(approval: confirmed, named confirmed_by), the coordinator should run exactly:

```sh
python -m cluster.sweep arm-transition --rules training/rounds/round2.yaml \
  --after 946641 946662 946663 946664 946665 946963 947094 \
  --teacher-after 946961 946962
```

WS5 must confirm reference exclusion, latency bindings and tie orders. The draft
excludes the four v1clean references, selects top three eligible v2/v2s combos by
val R@P90 (within 0.01 use estimated Pi fp16 p95), gates small models at >0.04
over the best same-dataset nano, restricts small deployment sizes, and uses
three seeds / 300 epochs / patience 50 / cosine LR / nbs 64. Each selected plain
nano also gets an 800 train arm. Distillation waits for both teachers and all
plain seeds, then uses the best teacher by val R@P90. Empty tie_order fields
intentionally HOLD unresolved metric/latency ties; WS5 can declare an order
before arming. Teachers have a separate exact-score tie order.

Useful check-in commands:

```sh
python -m cluster.sweep status
python -m cluster.sweep fetch --generated-round round2-plain round2-distill
python -m cluster.sweep throttle-arrays --gpu-limit 4
```

Requests for coordinator documentation: clarify PLAN section 6's distinction
between the eight running-GPU cap and the 24 submitted array-task cap; document
C6's optional array_job_id, array_task_id, slurm_restart_count and adjacent
requeue/HOLD ledgers. No new C5 recipe field is required. WS5 confirmation and
explicit coordinator arming are the only production prerequisites.

### Checks and evidence

Final section-0 checks: **263** unittest cases, seven dependency/hardware skips;
all executed cases pass. All **69** cluster tests pass without skips and no
network. Ruff check and format pass on app/tests/tools/cluster/training/native
(87 Python files); strict mypy passes on 84 source files. git diff --check passes.
Tests cover manifests/throttles and global/type caps, active/DONE diffing,
root-sharing task requeue, real bash USR1 forwarding/child exit handling, node-failure and progress/cap guards, draft refusal,
HOLD cases, ranking/small gating/ties, generated nano/small/teacher recipes,
upstream validation, terminal recovery/re-arming and sacct state filtering.

Evidence in git-ignored outputs/ws3/: evidence-round4.json, status-round4.jsonl,
fetch-round4.log, unit-round4.log, ruff-round4.log, mypy-round4.log and
cluster-state/{arrays,transitions,generated-recipes}/. The CPU scheduler proof
is ~/columba/logs/ws3-injob-probe.txt. Live smoke snapshots were committed
c3103a4 (corrected requeue), fcf0103 (generation) and 162e7dd (actual dependency
chain). Final publication/preparation uses the final branch state without
changing any job's pinned checkout.

## Round 5 in progress

Branch ws3/cluster, GPT-6.1 via T3 delegate, 15 October 2026. Fast-forwarded to
main bd4ff06. Implementing device-visibility retry, expiring dynamic exclusions,
transition recovery without checkpoints, and throttle-preserving array retargets.
Running round-2 jobs, 947422 and the separate node-scanning agent's probes remain
untouched. Verification will simulate a fault in one brief allocation.

## Round 5 complete — device visibility recovery

Branch ws3/cluster, GPT-6.1 via T3 delegate, 15 October 2026. Fast-forwarded to
main bd4ff06 before work. Existing round-2 jobs, transition 947422 and the node
scanner's probes were neither cancelled nor modified. No package pins changed.

### Detection, quarantine and retry

New training wrappers check nvidia-smi UUID visibility before environment prep,
then CUDA availability, allocated device count and CUDA initialisation in the
selected CUDA venv. The physical count must cover the allocation; Torch must
see exactly SLURM_GPUS_ON_NODE. GPU work, hardware forward/backward and dataset
staging follow these checks. Hardware smoke still runs inside bootstrap's
scratch Ultralytics settings, avoiding a home-directory settings leak.

`~/columba/state/bad_nodes.json` is a version-1 mapping under `nodes`, keyed by
node. Entries contain node, first_seen, time, expires_at, job, error, count and
simulated. Times are UTC ISO strings. `cluster/device_policy.json` configures
24-hour quarantine and three device retries per run. Expired entries stop
excluding nodes automatically but remain visible for diagnosis. A new genuine
fault renews expiry and count. Simulated records never exclude production
placements or contaminate genuine-node counts. Static exclusions win over
expiry and clear-on-success.

Per-run `device_faults.json` records retry_count and events keyed by exact
array task plus restart count. Each event includes node/error/time, expected
restart, repair job, outcome and original array throttle. The fourth detected
fault writes HOLD after three retries. Device restarts bypass checkpoint
progress checks and consume neither the ten checkpoint-requeue allowance nor
the transition's two normal checkpoint recoveries. Early run.json records
exit 6, epochs_done, hardware error and device_fault_retries. The regular shell
failure handler preserves this evidence.

Dynamic exclusions join static/explicit exclusions in legacy submit, pending
retargets, arrays and transition-generated arrays. The status command shows
active/expired entries, expiry times, per-run device retries and repair IDs.
Fetch also transfers state JSON. CUDA OOM and ordinary data/code exceptions
remain genuine failures. A prior device event does not classify a later code
failure as a device fault. Failure reports retain the current restart identity,
and attempt markers prevent old appended log output from triggering a retry.

### Measured Slurm verification

- Simulation array **949633_0** requested one T4, one CPU and a one-minute limit.
  Its first allocation on **xgpf0** deliberately reported No devices were found.
  This was a simulated fault on healthy hardware, not a health finding.
- Slurm rejected the running task update with:
  `949633_[0]: Job is no longer pending execution`.
- A finite repair job **949634** ran on xcng0, afterany:949633_0, with one CPU,
  two GB, one minute and --nice=1000. It completed in under one reported second.
  It ran requeuehold, changed the held task's ExcNodeList, cleared ReqNodeList,
  restored ArrayTaskThrottle and released the same task. This is the working
  equivalent when a running-state exclusion update is rejected.
- 949633_0 restarted on **xgpf11**, with restart count 1, unchanged throttle 1,
  and `ExcNodeList=xgpd2,xgpe2,xgpe6,xgpf0`. It completed 0:0 in two reported
  seconds on its final allocation. Its checkpoint ledger contains only the
  recorded scheduler restart, with checkpoint retry count zero and no last.pt.
  Its device ledger has retry_count=1. Thus a fault before any epoch does not
  trip the no-progress guard. No actual broken node was used for this test.
- Initial setup job **949626_0** failed on xcng0 in one second because the test
  submission generated an invalid shell export. It never reached device code.
  Corrected the test and committed the reusable, syntax-checked
  cluster/templates/device-test.sbatch fixture before the successful T4 test.
  No additional dummy test was submitted to xcng0; 949634 did real repair work.
- Real all-GPU probe **949660** on **xgpj0** completed 0:0 in under one reported
  second, using system Python with a one-minute limit and no venv bootstrap.
  Configured/allocated/visible counts were all 1. nvidia-smi reported A100 80GB
  PCIe, driver 580.178.04, compute 8.0 and 81920 MiB. clear-on-success passed.
  This probe intentionally makes no Torch/forward-backward claim.
- All test allocations requested --nice=1000, recorded in SubmitLine. No WS3
  job remains queued. Removed only xgpf0's own completed simulated quarantine;
  per-run history and the captured global-ledger evidence remain available.

The helper is bounded useful work, not a persistent orchestrator. It checks
ownership and CPU/submission budgets before submission. It keeps Slurm task
identity so existing dependencies follow the restarted task. A transition
that races a queued repair defers to the repair and then the requeued task.
Transitions also recover recognised early device failures from owned cz- logs
or a matching device event, without a checkpoint. Replacement arrays exclude
bad nodes and re-arm the transition. An exhausted or unsuccessful repair HOLDs
with its reason; genuine failures do not enter device recovery.

### Probes, retargeting and placement

`gpu-probe` now requests every configured GPU on the selected node. Its limit
is 00:01:00, with no bootstrap or autorequeue. It runs the requested name/driver/
compute/memory query, validates the count against node GRES and the allocation,
and clears a dynamic entry only on complete success. Static entries remain.
Nodes whose full GPU count exceeds the per-type QoS cap are refused instead of
being parked indefinitely. In particular four-GPU H100-47 nodes need a QoS
exception or administrator assistance for an all-node probe under the two-GPU
per-user cap; H100-96/H200 nodes can have the same constraint.

New commands:

```sh
python -m cluster.sweep gpu-probe --probe-node xgpj0 --clear-on-success
python -m cluster.sweep retarget-array --job ARRAY_ID --constraint h100-47 --throttle 2
python -m cluster.sweep status
```

retarget-array accepts owned PENDING roots/tasks only, uses a typed GRES,
checks total/type throttle limits and reapplies the throttle after changing
GRES/partition. An explicit throttle increase is not applied if the request
update fails. Pending task IDs can be used when another task in the parent
array is running. No retarget command was exercised against production jobs.

Heavy arrays and array retargets now prefer gpu-long when that GPU type has
free units, otherwise gpu with a three-hour limit and normal checkpoint
self-requeue. Multi-partition requests fit the minimum partition limit.
Preference order, 4/2/1/1 shares and H100-96 limits are unchanged. xgpj0 remains
available. **The static exclusion file was not edited:** xgpd2, xgpe2 and xgpe6
remain. The coordinator can move xgpd2 to dynamic handling after this lands;
all-GPU probes and runtime guards now handle its flaky-device case.

### 947422 hand-off — not re-armed

Re-arming onto new code is safe and useful for future failures. Its pinned
round2-distill rules are byte-identical to the checked-in rules:
SHA-256 c795378651d0fa15869431b95567fa801eae1d877f5c9ce5be1b0dbbc0573b7c.
The rule still distils top_k=1, three seeds. The original plain selection is
upstream **4ddea01170a47126**. Current unsatisfied dependencies are **947419**
and **949457**, matching the coordinator's manual repair. Completed source and
teacher runs still undergo C6 validation.

An explicit arm now includes the source checkout in transition identity, so
using new code cannot silently return an old submission pinned to old code.
The coordinator can run:

```sh
python -m cluster.sweep arm-transition --rules training/rounds/round2-distill.yaml \
  --after 947419 949457 --upstream 4ddea01170a47126
```

Verify the replacement is pending with those dependencies and the new pinned
checkout, then retire old pending 947422. Do not leave both armed. WS3 has not
run this command, held 947422 or changed its dependencies. Current production
jobs 947419_2/3 and 949457_0 remain running, with 947419_4 pending at verification.
The coordinator's transition remains pending. No node-scanner job was touched.

### Checks, evidence and requests

Final section-0 checks: 283 unittest cases, seven dependency/hardware skips;
all executed cases pass. All 89 cluster cases pass without skips and without
network. Ruff check/format pass on app/tests/tools/cluster/training/native
(88 Python files); strict mypy passes on 85 files. git diff --check passes.
Tests cover visibility/count/CUDA init, OOM/data rejection, atomic expiry/counts,
independent retry/progress budgets, running-update fallback and held repair,
actual ledger exclusions in both submit paths, transition recovery with/without
checkpoints, repair deferral, exhaustion HOLD, all-node probe success/failure/
QoS refusal, retarget throttle restoration and free-partition placement.

Evidence: outputs/ws3/evidence-round5.json, status-round5.jsonl,
unit-round5.log, unit-round5-device.log, ruff-round5.log, mypy-round5.log,
device-live-final-submission-round5.json and probe-submission-round5.json.
Cluster proof: ~/columba/runs/smoke-device-r5-final/seed0/proof.json and
its device_faults.json/requeue.json, plus gpu-probes/xgpj0/run.json. Live
simulation used cc693ac; real all-GPU probe used 8b8dc19. Final changes are
published to a new immutable checkout; existing pinned jobs stay unchanged.

Coordinator requests: optionally document device_fault_retries and the adjacent
state/device_faults ledgers in C6; choose when to move xgpd2 out of the static
list. Full-node probes above normal QoS typed caps require administrator help.
No implementation blocker remains. 947422 replacement remains the coordinator's
explicit action.

## Round 6 — dynamic GPU budget (2026-10-15)

State: in progress. Branch ws3/cluster, GPT-6.1-sol via T3 delegate.
Started from main d52f3d1. User overrides the earlier nice=1000 default to 0.
Jobs 950409, 951220 and 951222 are protected from all mutations in this round.
Work: remove fixed GPU shares; account for shared total and typed ceilings;
live-capacity placement and finite automatic pending-array maintenance; targeted
throttle CLI; configurable Nice. Validation will use mocked Slurm and read-only
live cluster observations, with no production retargeting or job probes planned.
