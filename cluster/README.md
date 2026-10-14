# Cluster commands

Use the repository root and a Python environment with PyYAML. Every SSH
operation uses `ssh -o BatchMode=yes columba-cluster`; authentication failure
stops the command. No password handling is implemented.

```sh
python -m cluster.sweep plan --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep submit --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep status --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep fetch --recipes training/recipes/baseline-repro.yaml
python -m cluster.sweep cancel --job-ids 123456
```

Pass several recipe files after `--recipes`. `plan` labels each recipe/seed as
missing, active or done; `status` adds its persisted status and epoch count.
`submit` takes a source snapshot into `~/columba/checkouts/<content-hash>/` so
new submissions cannot overwrite running jobs' code. It never pushes git.
It records the local commit in `run.json`; commit implementation changes
before production submissions. The scheduler serialises submissions with an
NFS lock, rechecks `squeue` before every `sbatch`, skips active runs and `DONE`,
and reserves pending jobs against the running limits. Other projects do not
count towards our caps and cannot be cancelled by this CLI.

Project caps are four GPU jobs, two CPU jobs and twelve queued jobs including
running jobs. `--gpu-limit`, `--cpu-limit`, `--queue-limit` may lower those caps.
During WS3 Wave B use `--gpu-limit 3 --cpu-limit 1`. These conservative limits
count all active cz- jobs, so another workstream may temporarily fill the
chosen limit. Resubmit later to schedule runs omitted due to budget. Normal
submission requires WS1 tools and both WS2 dataset archives; smoke recipes
explicitly allow missing evaluators and do not write `DONE`.

Templates always request `--nice=1000`. GPU jobs request
`--partition=gpu-long --gres=gpu:nv:1` with the validated recipe constraint;
ARM jobs request `--partition=normal --nodelist=xcng0`. Templates allow at most
eight GPU-lane CPUs, 64 GB RAM, and three days; ARM uses four CPUs, 16 GB and
three hours. The runtime rejects GPUs other than T4, Titan RTX and Titan V.

`fetch` copies C6 JSON, CSV and best/last weights to `experiments/runs/` and
upserts the C6 leaderboard by recipe and seed, preserving the incumbent.
Missing reports are not fabricated. The header is created when absent.
Failed and evaluation-missing reports get rows. Unsubmitted runs are skipped.
Bootstrap failures also leave a report through the template EXIT handler. The leaderboard is owned by
WS1; WS3 does not commit it.

Bootstrap installs uv once in `~/.local/bin`, uses only `~/columba/` for caches
and environments, and pins PyTorch 2.8.0 / torchvision 0.23.0 from the official
CUDA 12.6 index for x86_64. CUDA 12.6 covers both permitted architecture
classes; bootstrap verifies the device's `sm_70` or `sm_75` and runs YOLO
forward/backward. ARM uses the CPU index. Login-node package installation may
fail under its 1 GB virtual-memory limit; templates bootstrap on compute
nodes. Ultralytics 8.4.175, ncnn 1.0.20260526 and pnnx 20260526 remain pinned.

Environment paths:

- `~/columba/venvs/x86_64-cu126`
- `~/columba/venvs/aarch64-cpu`

If an nv node has a CUDA driver failure, add `--exclude-nodes xgpe6` to a
submission. Exclusions accept only node names in the permitted lanes. They
cannot broaden the GPU request or select restricted hardware.

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
