# Cluster commands

Use the repository root and a Python environment with PyYAML. Every SSH
operation uses `ssh -o BatchMode=yes -o ConnectTimeout=20 columba-cluster`; authentication failure
stops the command. No password handling is implemented.

## Current state (17 Oct 2026)

The model-improvement project is closed except WS10. The only remaining cluster
work is WS10 step 4: a site fine-tune (a round-4 recipe based on
`training/recipes/generated/round2-plain/yolo26n-v2s-800-round2-plain.yaml`),
then re-selection with `tools/select_model.py`. After the WS7 pruning,
`~/columba` holds:

| Path | Contents |
|---|---|
| `runs/` | `yolo26n-v2s-800-round2-plain` (3 seeds) and `baseline-repro` |
| `data/` | `columba-train-v2s`, `columba-train-v2s-species`, `columba-test-v2`, `columba-test-v2-styles`, smoke-coco8 |
| `venvs/` | `x86_64-cu126`, `aarch64-cu126`, `aarch64-cpu` |
| `weights/`, `bin/`, `logs/`, `arrays/`, `transitions/`, `generated-recipes/`, `state/`, `claims/` | base weights, uv, logs and orchestration state |
| `scratch/ws8/venv`, `scratch/ws8/round3/repo/native/build/linux-aarch64`, `scratch/ws4/ncnn-xcng` | the ARM evaluation toolchain that `tools/select_model.py` hard-codes |

All other scratch, `checkouts/` (republished per job) and `cache/uv` were
deleted; jobs recreate them. The `columba-train-v1clean`, `columba-train-v2`
and `columba-train-v2-species` archives were deleted too. Recipes that use them
(`baseline-repro*.yaml`, the `*-v1clean-*` and `*-v2-*` round-1 recipes) need
the archive rebuilt with `tools/build_dataset.py` and uploaded first.

The sections below describe the WS3 tooling as built; job IDs in examples are
historical.

```sh
python -m cluster.sweep plan --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep submit --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep status --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep fetch --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep cancel --job-ids 123456
```

Pass several recipe files after `--recipes`. Use `--seeds 1` to select an
existing seed without changing the recipe hash. `plan` labels each recipe/seed as
missing, active or done; `status` adds its persisted status and epoch count.
`submit` takes a source snapshot into `~/columba/checkouts/<content-hash>/` so
new submissions cannot overwrite running jobs' code. Publication builds a
temporary directory, marks it ready and atomically renames it into place.
A previously published checkout is never re-extracted or modified. It never
pushes git.
It records the local commit in `run.json`; commit implementation changes
before production submissions. The scheduler serialises submissions with an
NFS lock, rechecks `squeue` before every `sbatch`, skips active runs and `DONE`,
and reserves pending jobs against the running limits. Other projects do not
count towards our caps and cannot be cancelled by this CLI.

Project caps are eight GPUs, two CPU jobs and twenty-four queued jobs including
running jobs. GPU devices, not allocations, are counted across every GPU type
and every allocated node. Legacy single-job submission reserves GPU slots for PENDING jobs. New fan-out
submission delegates total/type GPU admission to QoS and counts every copy
towards the submitted-task cap. `--gpu-limit`,
`--cpu-limit`, `--queue-limit` may lower those caps. These limits count all
active cz- jobs, so another workstream may temporarily fill the
chosen limit. Resubmit later to schedule runs omitted due to budget. Normal
submission requires WS1 tools and both WS2 dataset archives; smoke recipes
may explicitly allow missing evaluators; missing evaluation leaves `DONE` absent.

`gpu_policy.yaml` sets `nice: 0` while the high-priority project is idle.
Every submission passes that value explicitly; templates default to zero. `lane: nv` names the GPU training lane;
it supports all user-approved GPU types, including aarch64 `xgpj0`.
ARM CPU jobs use `normal` on xcng0. GPU templates use eight CPUs and 64 GB RAM;
ARM CPU templates use four CPUs, 16 GB and three hours. Synthetic smoke IDs
use four CPUs, 16 GB and ten minutes on the legacy submit path. New fan-out
smokes use two CPUs, 8 GB and at most one minute, with no checkpoint signal.

GPU requests from the verified 15 October inventory:

| Selector | GRES | Feature | Default partition / limit |
|---|---|---|---|
| t4 / titanrtx / titanv | gpu:nv:1 | corresponding selector | gpu-long / 72 h |
| a100-40 / a100-80 | gpu:a100-40:1 / gpu:a100-80:1 | none | gpu-long / 72 h |
| h100-47 / h100-96 | gpu:h100-47:1 / gpu:h100-96:1 | none | gpu-long / 72 h |
| h200 | gpu:h200-141:1 | none | gpu / 3 h |
| a100 / h100 | concrete matching family variant | none | gpu-long / 72 h |
| null / CLI any | explicit type chosen by policy | nv feature if needed | first free preferred type, gpu-long then gpu |

Pinned types keep working when occupied; they may queue. `--node xgpj0` selects
A100-80 in `gpu` with a three-hour limit. `xgpj0` also advertises `test`, which
this scheduler does not use. xgpd0–1 and xgpf0–5 are `gpu` only. H200 xgpk0 is
also `gpu` only. Live test-only requests with untyped gpu:1 and A100/H100
features, or pinned xgpj0, were rejected with "Requested node configuration
is not available". Typed requests succeeded. Family selectors therefore resolve
to a concrete variant, favouring free units within the preferred partition.
Slurm also rejects a 72-hour request spanning gpu-long and gpu.
Automatic placement therefore selects one explicit type/partition and restricts
its node list to one node with free units at submission time.

`cluster/gpu_policy.yaml` controls automatic selection. Light jobs prefer T4,
Titan V, Titan RTX, then H100-47. X models, images at least 800 px, distillation
teacher/distillation jobs, or estimates at least four GPU-hours prefer H100-47, H100-96, opportunistic
xgpj0, A100-80, then nv. For each preference, gpu-long is tried before gpu.
`allow_short_partition: true` permits three-hour allocations with checkpoint
resume. Free-unit counts use sinfo Gres minus GresUsed; unavailable, rebooting,
reserved and unresponsive nodes are ignored. Capacity and fair-share can change
between sinfo and sbatch, so free units do not guarantee an immediate start.
If no preferred capacity is free, an automatic run remains unsubmitted and
prints `no free preferred GPU`; it does not reserve a GPU budget slot.

The user snapshot on 15 October showed all A100/H200 units allocated, the
fewest pending jobs on nv, and the most new-GPU headroom on H100-47. These are
point-in-time observations, not permanent capacity assumptions. The live
inventory is checked on every submit invocation.

Opt-in re-targeting applies only to PENDING runs named by the current recipes:

```sh
python -m cluster.sweep submit --recipes training/recipes/example.yaml --constraint any --retarget-pending-after-hours 2
```

An expired pending run is cancelled only when an alternative preferred
type/partition has free capacity. Slurm's `scancel --state=PENDING` filter
protects jobs that start during the check; the queue is checked again before
resubmission. Running jobs and unrelated cz- runs are untouched. This option
explicitly permits overriding a recipe's pinned type/node for the retry.
Without this option, active runs are always skipped.

`fetch` copies C6 JSON, CSV and best/last weights to `experiments/runs/` and
upserts the C6 leaderboard by recipe and seed, preserving the incumbent.
Missing reports are not fabricated. The header is created when absent. `smoke-*` runs are excluded from the real
`experiments/leaderboard.csv`, including when that path is passed explicitly.
To record smoke rows, pass `--leaderboard outputs/ws3/leaderboard-smoke.csv`.
Failed and evaluation-missing reports get rows. Unsubmitted runs are skipped.
Bootstrap failures also leave a report through the template EXIT handler. The leaderboard is owned by
WS1; WS3 does not commit it.

Bootstrap installs uv in `~/.local/bin` when absent. If that shared NFS binary
belongs to another architecture, it installs a compatible copy in
`~/columba/bin/<arch>/uv`. It uses only `~/columba/` for caches
and environments. x86_64 CUDA uses torch 2.8.0 / torchvision 0.23.0 from the
CUDA 12.6 index. aarch64 CUDA uses torch 2.9.0 / torchvision 0.24.0: the cu126
index supplies Python 3.12 SBSA wheels for 2.9, but not 2.8. ARM CPU remains
2.8.0 / 0.23.0. Ultralytics 8.4.175, ncnn 1.0.20260526 and pnnx 20260526 remain
pinned. Python 3.12 aarch64 ncnn and opencv-python-headless wheels exist.
Bootstrap selects the environment from uname -m and the GPU allocation;
`--gpu-smoke` checks the actual device architecture and YOLO forward/backward.
Login installation is avoided because of its 1 GB virtual-memory limit.

Environment paths:

- `~/columba/venvs/x86_64-cu126`
- `~/columba/venvs/aarch64-cu126`
- `~/columba/venvs/aarch64-cpu`

The runner records `architecture`, `venv_path`, `python_executable`, CUDA,
PyTorch and device architecture in run.json. A run can resume on another GPU
or CPU architecture; its data and teacher paths are rewritten to the new
job-local staging directory. Ultralytics 8.4.175 permits batch and device
overrides on resume. The runner reapplies its memory-class batch, or the
recipe batch_override when set. A larger GPU can therefore change batch while
retaining optimiser state; a fixed recipe batch_override remains fixed.
Changing that recipe field under an existing ID is rejected by the recipe-hash
guard. Use a fresh recipe ID if you change its training configuration.
The actual resumed batch is recorded by the save callback and in attempts.

Every submission excludes the nodes listed in `cluster/excluded_nodes.txt`
(currently `xgpe6` and `xgpe2`, whose GPUs report "No devices were found").
`--exclude-nodes` adds nodes to that list; it cannot remove them. `submit
--constraint <selector>` overrides the recipe GPU type for one submission
(use with `--seeds`) without changing the recipe hash. Edit the file
to change the default. Exclusions accept only node names in the permitted lanes. They
preserve the approved GPU inventory. xgpj0 is explicitly allowed.

ARM export bundle:

```sh
python -m cluster.sweep arm-bundle --models smoke-5ep/seed0 --bundle-action export --cpu-limit 1
```

The bundle writes `~/columba/runs/arm-bundles/<hash>/run.json`, NCNN exports
under `<hash>/<recipe_id>/seed<k>/`, and a separate `DONE` after success.
To bundle WS4 benchmarks, use `--bundle-action benchmark` and provide
`~/columba/benchmark-commands.json`, a JSON list of argument lists starting
with `bench_ncnn.py`. Commands run sequentially in one job.

Keep project storage below 80 GB. Remove old project checkout snapshots and
cache entries only after their jobs finish and the coordinator agrees which
runs to retain. Dataset, weight and run paths outside `~/columba/` are rejected.

Round-2 E2E and resume checks:

```sh
python -m cluster.sweep submit --recipes training/recipes/smoke-e2e.yaml --gpu-limit 2 --cpu-limit 1
python -m cluster.sweep fetch --recipes training/recipes/smoke-e2e.yaml --leaderboard outputs/ws3/leaderboard-smoke.csv
python -m cluster.sweep submit --recipes training/recipes/smoke-distill-resume.yaml --stop-after-epoch 1 --gpu-limit 2 --cpu-limit 1
python -m cluster.sweep submit --recipes training/recipes/smoke-distill-resume.yaml --gpu-limit 2 --cpu-limit 1
python -m cluster.sweep gpu-probe --probe-node xgpe6 --gpu-limit 2 --cpu-limit 1
```

The first distillation submission intentionally exits after saving epoch one.
The second uses the unchanged recipe and last.pt, re-stages the teacher/data,
and resumes the remaining epochs. `--stop-after-epoch` is permitted only for
smoke IDs and must precede the final epoch. It does not become part of the
recipe or checkpoint args. `run.json.attempts` retains both attempts, and
`checkpoint_relocation` records the old and new data/teacher paths. The probe
pins an approved GPU node and explicitly bypasses the training exclusion list;
it queries all configured GPUs with nvidia-smi in at most one minute, using
system Python without bootstrapping a venv. Its report lives in `~/columba/runs/gpu-probes/<node>/run.json`.

## Unattended rounds on Slurm

Training uses finite arrays and short dependency jobs. There is no persistent
process on the login node, no polling compute job and no dependency on the Mac
remaining connected. Submit a declared set of recipes once:

```sh
python -m cluster.sweep submit-arrays --recipes training/recipes/example.yaml
python -m cluster.sweep status
python -m cluster.sweep fetch --generated-round round2-plain round2-distill
```

Each GPU request gets an array, with a checksum-protected task manifest mapping
its task index to recipe and seed. Recipes and code use immutable published
checkouts. Generated recipes stay under `~/columba/generated-recipes/<round>/`.
Arrays skip DONE and already active recipe/seed pairs, including legacy jobs;
legacy `submit` also recognises active array tasks. `squeue --array` expands every
task for the 24 submitted-task project cap. All account jobs count towards the
verified association MaxSubmitJobs=32; only cz- jobs count towards our project
budget. Array backlog is held by Slurm: unlike the older single-job admission
path, pending GPU array tasks may exceed eight, while array throttles and QoS
limit running GPUs to eight. This follows the approved upfront-array design.

The normal QoS per-user limits verified on 15 October are GPU=8, nv=8,
A100-40=4, A100-80=2, H100-47=2, H100-96=1, H200-141=1, H200-71=1 and CPU=256.
Project CPU concurrency remains two. These type limits are ceilings inside
one shared allowance of eight, never reserved shares. nv may use all eight.
The fixed `array_gpu_slots` setting has been removed.

`submit-arrays` and transition-generated runs now use per-run fan-out, encoded
as singleton arrays to preserve the manifest, status and dependency interfaces.
The light set is nv, A100-40 and H100-47. nv uses the OR constraint
`t4|titanv|titanrtx`. The heavy set is H100-47, H100-96, A100-40, A100-80 and
H200-141. Set these in `light_preferences` and `heavy_preferences`; `h200`
selects GRES h200-141. An explicit recipe/seed/CLI constraint remains pinned.
Requests span gpu-long and gpu where that type exists, with a three-hour
limit whenever gpu is included. This covers gpu-only A100/H100 nodes, xgpj0
and H200 without relying on a later retarget operation.

For legacy multi-task arrays, lower throttles and persist their limit:

```sh
python -m cluster.sweep throttle-arrays --gpu-limit 4
# restore the approved allowance when instructed:
python -m cluster.sweep throttle-arrays --gpu-limit 8
```

The command checks ownership through manifests and the current queue, then uses
`scontrol update JobId=<array> ArrayTaskThrottle=<share>`. Shares sum to the
requested limit or less and obey type caps. It refuses more active arrays than
available slots, since a zero Slurm throttle means unlimited. Already running
tasks finish; this does not cancel them or reduce legacy job allocations. Slurm
QoS still enforces the account's total/type limits across overlapping arrays.

### Checkpoint requeue

GPU jobs request `--requeue --signal=B:USR1@600`. The bash trap forwards USR1 to
the trainer, which finishes its current epoch, copies last.pt and results to
NFS, and exits with the dedicated requeue status. The script requeues exactly
`<array_id>_<task_id>`, including the task that shares the array root job ID.
Every restart stages data and teacher again and relocates checkpoint paths.
Slurm's automatic node-failure requeues use the same startup guard.

`requeue.json` in each run folder records saved epochs, scheduler restart counts
and a lifetime cap of ten requeues. A restart without a newly saved epoch, a
missing checkpoint or the cap writes HOLD and stops. A graceful signal does not
get counted again at startup. Last.pt is persisted after each saved epoch;
`save_period: 5` additionally creates periodic snapshots in Ultralytics. A run
whose training checkpoint is complete retries only prediction/evaluation.

### Explicitly armed transitions

`training/rounds/round2.yaml` and `round2-distill.yaml` now carry WS5 confirmation.
Future rules must carry named confirmation before arming. The following commands
show the original round-1 setup. Preparation validates ownership and writes a
plan without submitting a job:

```sh
python -m cluster.sweep prepare-transition --rules training/rounds/round2.yaml \
  --after 946641 946662 946663 946664 946665 946963 947094 \
  --teacher-after 946961 946962
python -m cluster.sweep arm-transition --rules training/rounds/round2.yaml \
  --after 946641 946662 946663 946664 946665 946963 947094 \
  --teacher-after 946961 946962
```

Arming a draft is rejected. Dependencies must belong to cz-. Completed jobs no
longer in squeue are retained in the plan but removed from the actual Slurm
request, avoiding expired dependency IDs. Their required DONE/artifacts are
still checked. The finite transition runs on xcng0/normal (four CPUs, 15 minutes),
consolidates C6 leaderboard rows and ranking JSON, applies the confirmed rules,
validates generated C5 recipes and submits their arrays. A second transition,
when declared, waits for plain arrays and both teachers before distilled arms.
Round 3 and WS7 remain manual.

The draft selects the top three eligible v2/v2s model/data combinations, excluding
v1clean references. It ranks measured val R@P90, breaks differences within 0.01
using estimated Pi fp16 p95 latency at 384×640, and requires a declared order
for remaining exact ties. Small models must improve over the best nano on the
same dataset by strictly more than 0.04 and deploy only up to 384×640. Plain
selected nanos get additional train-imgsz-800 arms. Both stages use three seeds,
300 epochs, patience 50, cosine LR and nbs 64. The second stage uses the best
teacher by measured val R@P90; a teacher tie also needs a declared order. Test
metrics are consolidated but never used to choose these arms.

Missing DONE in required source, plain or teacher runs after
TIMEOUT/PREEMPTED/NODE_FAIL, with last.pt and checkpoint progress, triggers mechanical resubmission under the array/self-requeue path.
The same transition is re-armed after the recovery arrays. A separate ledger
under `transitions/recovery/<round>.json` allows at most two such recoveries per
run and requires epoch progress between attempts. Genuine failures, missing
metrics after DONE, invalid latency, unhandled ties, insufficient eligible
runs or exhausted budgets write HOLD and submit no new training work.

State locations are `~/columba/{arrays,transitions,generated-recipes,runs,logs}`.
`status` shows array tasks, throttles, saved epochs, requeue counts, HOLD reasons
and transition outcomes. `fetch` copies orchestration JSON/HOLD files and generated
YAML to `outputs/ws3/cluster-state/` (change with `--state-destination`). Pass
`--generated-round <id> ...` to fetch those recipes' C6 artifacts and upsert the
local leaderboard. Smoke rows remain excluded from the production leaderboard.

The infrastructure smoke uses two three-epoch tasks, a real USR1 checkpoint
requeue, then one two-epoch follow-up task generated by a confirmed smoke-only
rule. Its predictions/evaluations use synthetic data and cannot establish
pigeon accuracy. Its round-4 smoke recipes and rules were removed in the
cleanup (git history at `6de04fb`). To repeat, write fresh smoke recipe and
rules IDs; existing DONE runs are intentionally skipped.


## Device visibility and dynamic exclusions

New GPU jobs check nvidia-smi UUID visibility before environment preparation.
After selecting the architecture's CUDA venv, they check
`torch.cuda.is_available()`, the allocated device count and CUDA initialisation.
Only then do they run the hardware smoke, checkpoint-start guard and training.
The physical nvidia-smi count must cover the allocation; the CUDA-visible count
must equal SLURM_GPUS_ON_NODE. Driver failures are recorded before datasets,
weights or training are staged. OOM, data and code failures keep their normal
failure path.

`cluster/device_policy.json` configures `bad_node_ttl_hours` (24) and
`max_retries_per_run` (3). The cluster-side dynamic ledger is
`~/columba/state/bad_nodes.json`, written atomically under the submission lock:

```json
{
  "version": 1,
  "nodes": {
    "xgpf0": {
      "node": "xgpf0",
      "first_seen": "2026-10-15T06:38:07.080563+00:00",
      "time": "2026-10-15T06:38:07.080563+00:00",
      "expires_at": "2026-10-16T06:38:07.080563+00:00",
      "job": "949633_0",
      "error": "simulated: No devices were found",
      "count": 1,
      "simulated": true
    }
  }
}
```

This example is the real simulation record, not a claim that xgpf0 is broken.
Normal failures set simulated=false. Simulated records are ignored by placement.
Expired entries remain visible for diagnosis but stop excluding the node.
Each new fault renews that node's expiry and increments its count. Static
`excluded_nodes.txt`, active dynamic entries and explicit exclusions are combined
for legacy submit, pending retargets, arrays and transition-generated arrays.
Static exclusions always win; a successful probe clears only the dynamic entry.
xgpd2/xgpe2/xgpe6 remain static. Moving xgpd2 to dynamic handling is supported once
the coordinator chooses to remove its static entry.

Each run has a separate `device_faults.json` with retry_count and an event list.
An event identifies the exact Slurm task and restart, node, time, error, expected
restart, repair job, outcome and saved array throttle. Reservations are idempotent
for a task/restart pair. Three device retries are allowed; the fourth detected
fault writes HOLD. This does not consume checkpoint/time-limit recovery retries,
and a device restart needs no saved epoch or checkpoint. `run.json` records the
early failure and device_fault_retries. Successful training retains that count.

The handler first tries to update the task's exclusions while it is running,
then requeues exactly <array>_<task>. This cluster rejects that running-state
update. Its working fallback submits a finite one-CPU normal/xcng0 repair job,
with a one-minute limit and afterany dependency on the failed task. The original
job exits 6. The repair job runs `scontrol requeuehold`, updates ExcNodeList,
clears stale ReqNodeList pins, restores ArrayTaskThrottle and releases the task.
Its preserved GPU type/features still constrain placement. The job restarts on
another allowed node with the same identity and checkpoint if one exists. CPU,
queued-job and retry budgets apply; an unsuccessful repair writes HOLD. These
operations use the documented [Slurm job controls](https://slurm.schedmd.com/scontrol.html).

Transition recovery recognises current-attempt device events and early shell
logs containing No devices were found, CUDA initialisation errors or no
CUDA-capable device. It can resubmit with no checkpoint and re-arm itself after
the replacement. It defers to an already pending repair or requeued task instead
of racing it. Device events and normal TIMEOUT/PREEMPTED/NODE_FAIL checkpoint
recovery have separate budgets. A previous device event does not turn a later
data/code failure into a hardware retry.

### One-minute probes and retargeting

```sh
python -m cluster.sweep gpu-probe --probe-node xgpj0 --clear-on-success
python -m cluster.sweep retarget-array --job ARRAY_ID --constraint h100-47 --throttle 2
python -m cluster.sweep status
```

The probe ignores static/dynamic exclusions, requests every configured GPU on
that node, and runs without venv bootstrap or PyTorch. It queries name, driver,
compute capability and memory, and checks the visible count against both node
GRES and the allocation. Only a complete, successful count check clears the
dynamic entry. It does not claim CUDA forward/backward validation. All-node
requests exceeding a per-type QoS cap are refused, rather than left pending
forever. For example, a four-GPU H100-47 node cannot be fully allocated under
this account's two-H100-47 cap without administrator help.

Retargeting accepts only owned PENDING arrays/tasks. It selects a typed GRES and
partition, clamps time to the partition limit and reapplies the saved or explicit
ArrayTaskThrottle after the GRES/partition update, even if that update fails.
It checks the destination type cap and total active-array throttle budget.
An array with running tasks must be targeted by a pending task ID rather than
its root. Running jobs are not retargeted.

Legacy heavy-array placement uses live free GPU units: prefer gpu-long
when that type has room, otherwise gpu with a three-hour request. This also
applies to retarget-array. Type caps remain unchanged; there are no reserved 4/2/1/1 shares. A multi-partition request must fit every partition under
EnforcePartLimits=ALL. Normal checkpoint requeue resumes across short slices.

status shows active/expired/simulated ledger entries, expiry times and per-run
device retries, including finite repair job IDs. fetch also copies state JSON.
The bounded fixture `cluster/templates/device-test.sbatch` simulates a first
failure only in a WS3 smoke allocation, then verifies the new node, exclusion,
restart and unchanged checkpoint counter. Use a fresh smoke run path to repeat.

### Re-arming a pinned transition

Transition identities now include the source checkout, so an explicit arm on
new code does not silently return a submission pinned to an older checkout.
The --upstream argument preserves the plain stage's selection for a distilled
transition. Round 2 re-armed its distilled transition (replacing 947422) with:

```sh
python -m cluster.sweep arm-transition --rules training/rounds/round2-distill.yaml \
  --after 947419 949457 --upstream 4ddea01170a47126
```

Verify the new job's dependencies and pinned checkout, then retire the old
pending transition. Do not leave both armed.


### Fan-out claims, submit caps and checkpoint segments

Each recipe/seed has one shared cz- job name. Its typed copies are submitted
held, recorded as immutable manifests, then released together. Under the NFS
submission lock, a copy claims `~/columba/claims/<run-hash>.json` before venv
bootstrap, GPU checks or data staging. It checks the exact-name jobs through
squeue. A different RUNNING, CONFIGURING or COMPLETING owner wins; the loser
logs duplicate_start and exits successfully within seconds. A missing,
terminal or PENDING owner is stale, so a replacement may claim it. The winner
uses scancel with the exact shared name, account owner and state=PENDING.
Running siblings are never cancelled. Claim history and the pending sibling
IDs are preserved as `fanout_claims` in run.json; status/fetch include claims.

Every expanded copy counts towards project MaxSubmit=24 and account
MaxSubmit=32, including other account jobs for the latter. The account also
has MaxJobs=16; GPU total/type QoS and the project CPU cap govern admission. At most eight new
runs receive the widest set that fits. Larger batches use one or two copies
per run. Narrow sets distribute their first choices across available typed
headroom instead of parking the entire backlog on H100-47. Extra copies are
held by QoS; there is no per-type array throttle split. If even one copy per
run does not fit, submission refuses before sbatch and asks for a smaller
recipe/seed batch. No idle slot or long-lived management job is created.

After a checkpoint signal, the runner replenishes the same recipe/seed's
fan-out and then requeues its existing task. New copies have BeginTime one
minute later, allowing the checkpoint owner to leave RUNNING before a new
copy checks the claim. The normal saved-epoch and ten-requeue guard remains
shared by the run. A refill blocked by the submit cap records
fanout_refill_error and keeps the existing self-requeue path. Device-fault
quarantine/retry remains separate, uses the same exclusions and rechecks the
claim on restart. Dependency transitions look up every active exact-name
copy before diagnosing a missing DONE, then re-arm against replacement jobs.

The existing commands remain available for old arrays:

```sh
python -m cluster.sweep throttle-arrays --job ARRAY_ID --throttle 2
python -m cluster.sweep retarget-array --job ARRAY_ID --constraint a100-40 --throttle 2
```

The targeted throttle command now forwards both options and actually writes
ArrayTaskThrottle. Manual retargeting still chooses the partition with live
free units and reapplies the throttle after a GRES update. Automatic pending
retargeting was replaced by fan-out; no hourly poller is installed.

When the high-priority project returns, set `nice: 1000` in gpu_policy.yaml,
restore nv-only preferences and the four-GPU allowance. Submit future work
with `submit-arrays --gpu-limit 4`, which uses the bounded typed-array path.
First retire the pending fan-out copies and allow their running owners to
finish; legacy throttle controls alone cannot constrain existing singleton
fan-out copies. Existing jobs remain pinned to their submitted policy/Nice;
changing policy does not rewrite them. Restore Nice=1000 on those allocations
through the coordinator. Default Nice=0 affects new submissions and segment
copies from new checkouts only.
