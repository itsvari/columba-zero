# Cluster commands

Use the repository root and a Python environment with PyYAML. Every SSH
operation uses `ssh -o BatchMode=yes -o ConnectTimeout=20 columba-cluster`; authentication failure
stops the command. No password handling is implemented.

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
and every allocated node. PENDING jobs reserve GPU slots. `--gpu-limit`,
`--cpu-limit`, `--queue-limit` may lower those caps. These limits count all
active cz- jobs, so another workstream may temporarily fill the
chosen limit. Resubmit later to schedule runs omitted due to budget. Normal
submission requires WS1 tools and both WS2 dataset archives; smoke recipes
may explicitly allow missing evaluators; missing evaluation leaves `DONE` absent.

Templates always request `--nice=1000`. `lane: nv` names the GPU training lane;
it supports all user-approved GPU types, including aarch64 `xgpj0`.
ARM CPU jobs use `normal` on xcng0. GPU templates use eight CPUs and 64 GB RAM;
ARM CPU templates use four CPUs, 16 GB and three hours. Synthetic smoke IDs
use four CPUs, 16 GB and ten minutes to fit short scheduling gaps.

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
To bundle WS4 benchmarks, use `--bundle-action benchmark` after WS4 merges its
tool and provides `~/columba/benchmark-commands.json`, a JSON list of argument
lists starting with `bench_ncnn.py`. Commands run sequentially in one job.

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
it prints nvidia-smi and CUDA availability without training or changing the
node. Its report lives in `~/columba/runs/gpu-probes/<node>/run.json`.
