# C5 recipes

Validate recipes with `cluster.recipe.load_recipe(path)` before submitting.
Run the CLI from the repository root with the shared Python environment.

```yaml
id: example-v1
framework: ultralytics
base_model: yolo11n.pt
dataset: columba-train-v1clean
taxonomy: single
seeds: [0, 1, 2]
lane: nv
gpu_constraint: null
seed_gpu_constraints: {0: t4, 1: titanrtx, 2: titanv}
est_gpu_hours: 3
teacher: null
train:
  epochs: 300
  patience: 50
  imgsz: 640
  nbs: 64
  save_period: 5
  cos_lr: true
deploy_sizes: [[352, 608], [384, 640], [448, 800]]
notes: ""
```

Required fields are `id`, `framework`, `base_model`, `dataset`, `taxonomy`,
`seeds`, `lane`, `est_gpu_hours`, `train` and `deploy_sizes`. Unknown top-level
fields fail validation. Identifiers, dataset names and base weights must be
safe single path components. Training on `columba-test*` is forbidden.
`train.imgsz` is one positive integer. Deployment dimensions are positive
multiples of 32. `nbs` stays 64. Seeds are unique non-negative integers.

`gpu_constraint` defaults to null. `seed_gpu_constraints` overrides it for
listed seeds using integer YAML keys; null explicitly restores automatic
selection. Allowed values are null, `t4`, `titanrtx`, `titanv`, `a100-40`,
`a100-80`, `h100-47`, `h100-96`, `h200`, `a100` and `h100`. `lane: nv`
is the GPU lane on both x86_64 and aarch64 (xgpj0). Automatic selection uses
cluster/gpu_policy.yaml and live free units; it never queues an untyped
any-GPU request. `sweep submit --constraint any` overrides existing pins
without changing recipe hashes. ARM recipes use `lane: arm` and cannot request GPU constraints.
`est_gpu_hours` is an estimate, never a measured runtime or a scheduling limit.
Pinned types prefer gpu-long (three days); H200 and pinned xgpj0 use gpu
(three hours). Automatic placement may also use short GPU nodes with resume.
ARM CPU jobs have a three-hour limit.

Optional `teacher: "teacher-recipe/seed0"` resolves to
`$COLUMBA_RUNS_ROOT/teacher-recipe/seed0/weights/best.pt`. The runner stages it
in the job's `/tmp/cz-$SLURM_JOB_ID/weights/teacher.pt` and passes that absolute
path as `distill_model`. `train.dis` controls the distillation loss weight.
Resubmission stages the teacher again and rewrites its checkpoint path.

`test_dataset` defaults to `columba-test-v2`. Both archives and their
`.tar.zst.sha256` sidecars must exist in `~/columba/data/`. Archives contain a
single top-level directory matching the dataset name, with the C1 manifest,
summary, YAML, images and labels. SHA-256 is checked before extraction. The
runner changes only the staged YAML's root and split paths.

`train` passes through to the framework adapter, except runner-owned `batch`,
`seed`, `project`, `name`, `data`, `model`, `device`, `distill_model`, `resume`
and `save_dir`, which recipes cannot contain. Saving and validation cannot be
disabled. `save_period` defaults to five epochs. The runner copies `last.pt`,
`best.pt`, results and its report to NFS after each saved epoch, and cleans
node-local storage on exit. A completed checkpoint skips training on a retry
that needs only evaluation. A changed recipe with an existing run ID fails;
create a new ID when changing a recipe.

Batch estimates at 640 for nano models are 32 on 12 GB, 48 on 16 GB and 64 on
24–141 GB GPUs (capped at nbs 64). X models start at 12, 16, 24, 32 and 48
on 40, 47, 80, 96 and 141 GB classes respectively. Larger models, larger images and teachers reduce the starting
batch. Ultralytics may lower it after an OOM; `run.json.batch_size` records the
actual selected batch. The pinned Ultralytics version permits batch changes
on resume; the runner reapplies its current memory estimate or batch_override
while keeping optimiser state. A fixed recipe override therefore stays fixed.
Changing the recipe requires a fresh ID. `seconds_per_epoch` contains measured training plus
validation times, and `attempt_wall_time_s` includes staging and evaluation.
Neither is a Pi latency result.

`deploy_sizes` records future export candidates. C6 evaluation uses the square
training size and val-selected threshold. WS5/WS7 evaluate exported candidates
at their deployment size separately.

`framework: ultralytics` is implemented. To add another framework, WS6 creates
`training/adapters/<framework>/__init__.py` exposing `adapter` with
`train(context: cluster.run_job.TrainingContext, report: dict[str, JsonValue])`.
The adapter writes the C6 weights/results into `context.local_run`. Evaluation
currently uses WS1's Ultralytics predictor; another adapter needs a compatible
checkpoint or a coordinator-approved prediction adapter before promotion.

Optional `batch_override: <positive int>` replaces the runner's memory-class
batch on the nv lane (e.g. x-size teachers at 960, where the conservative
heuristic picks 3). `nbs` stays 64, so gradient accumulation still targets an
effective batch of 64. Measure memory before raising it.

Only recipes whose ID begins `smoke-` may use
`allow_missing_evaluation: true`. Missing WS1 tools then set
`evaluation: "skipped: tools missing"`, `status: evaluation_missing`, and leave
`DONE` absent. A normal successful run writes `DONE` after all C6 artifacts.
`TRAINED` means only that training finished, not that evaluation is complete.

`baseline-repro.yaml` preserves the v1-2 source args: YOLO11n, 640, 200-epoch
budget, patience 10, cosine LR off, original augmentations and optimiser.
Its seed-to-GPU mapping is 0=T4, 1=Titan RTX, 2=Titan V. Batch and paths are
runner-owned and checkpoint saving is enabled every five epochs. This is a
reproduction of the recipe on cleaned data, not of the original batch-8 run.

`smoke-5ep.yaml` uses the same configuration for five epochs and permits
missing evaluation. Until WS2 publishes the real data, build its tiny CC0
synthetic fixtures with `python -m cluster.smoke_dataset --out outputs/ws3/data`.
The historical `smoke-coco8` archive name contains synthetic circles, not COCO
photographs. Never report its metrics as pigeon accuracy.
