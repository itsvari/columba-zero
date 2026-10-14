# WS5 — YOLO training and distillation

- State: in progress  <!-- not started | in progress | blocked | done -->
- Branch: `ws5/training`
- Agent: Claude Opus 5.5 via T3 delegate
- Last updated: 2026-10-14

## Done

1. `baseline-repro.yaml`: `workers` 0 → 6 and `cache` false → ram. The notes
   now record every deviation from v1-2:
   - Batch comes from the runner's memory class.
   - Effective batch is 48 on T4 (accumulate 1) and 64 elsewhere.
   - Data is v1clean.
   - The optimizer is still AdamW (optimizer auto).
2. Round 0 submitted. Seeds 0 (T4) and 2 (Titan V) are DONE for both
   `baseline-repro` and `baseline-repro-p50`. Both seed-1 (Titan RTX) jobs
   are pending.
3. Patience 10 made `baseline-repro` stop very early, so I added
   `baseline-repro-p50.yaml`: the same args, patience 50. All 3 seeds
   submitted.
4. Recipes committed. 12 round-1 screening recipes:
   `{yolo11n,yolo26n,yolo11s,yolo26s}-{v1clean,v2,v2s}-640-r1`. 2 teacher
   recipes: `yolo{11,26}x-v2s-960-teacher`, on Titan RTX. All pass
   `cluster.recipe.load_recipe`.
5. Base weights cached on the cluster: `~/columba/weights/yolo11{n,s,x}.pt`
   and `yolo26{n,s,x}.pt`, from the Ultralytics v8.4.0 asset release, the
   same source the runner uses.
6. `experiments/reports/WS5-plan.md`: rounds, recipes, GPU-hour estimates
   and selection rules.
7. Leaderboard rows for `baseline-repro` and `baseline-repro-p50` seeds 0
   and 2, added via `sweep.py fetch`.
8. Round 1: `yolo11n-v1clean-640-r1` (945490) and `yolo26n-v1clean-640-r1`
   (945491) submitted.

## In progress

- Pending/running jobs: `baseline-repro` s1 (945341) and
  `baseline-repro-p50` s1 (945409), both waiting for Titan RTX. Round 1:
  945490 and 945491.
- `yolo11s-v1clean-640-r1` and `yolo26s-v1clean-640-r1` were refused with
  "budget full": pending jobs count against the 4-GPU cap. Submit them when
  a slot frees.

## Blockers and requests

- **Coordinator decision:** make `baseline-repro-p50` the section-1
  baseline instead of `baseline-repro`. With patience 10, T4 seed 0 stopped
  at epoch 11 and kept its epoch-1 weights (measured test recall@p90 0.006).
  That would be a straw-man baseline.
- **WS3 bug:** `cluster/artifacts.py:143` `csv.DictWriter` writes CRLF, so
  every fetch rewrites every line of `experiments/leaderboard.csv`. I
  normalised the file back to LF by hand. Fix: `lineterminator="\n"`. Also,
  `run_path` holds an absolute path into the fetching worktree.
- **WS3 request:** `select_batch` gives batch 3 for x models at 960 on a 24 GB
  Titan RTX (accumulate 21). Batch 8 probably fits (estimated, not measured).
  Please allow a recipe batch override or a measured AutoBatch for teachers.
  Otherwise the teachers take ≈ 60 h each (estimated).
- Titan RTX capacity: usable nodes are xgpe2 and xgpe4 (xgpe3 down, xgpe6
  excluded). Teacher and seed-1 queueing will be slow.
- s-size Pi latency is not measured yet. Before round 2, I need one arm
  benchmark job for COCO yolo11s/yolo26s, or the coordinator bundles it.

## Results

Measured on the cluster, YOLO11n 640, `columba-train-v1clean`. Timings are
training + validation per epoch; epoch 1 (≈ 35 s, includes RAM caching and
start-up) is excluded. These are not Pi results.

| Run | Job | Host / GPU | Batch | s/epoch | Epochs (stop) | val R@P90 | test R@P90 | test AP50 | test empty FAR | test confuser FAR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-repro/seed0 | 945340 | xgpf11 / T4 | 48 | 7.0 | 11 (patience 10, best = epoch 1) | 0.071 | 0.006 | 0.131 | 0.024 | 0.068 |
| baseline-repro/seed2 | 945342 | xgpd2 / Titan V | 32 | 5.3 | 53 (patience 10) | 0.468 | 0.218 | 0.564 | 0.008 | 0.205 |
| baseline-repro-p50/seed0 | 945408 | xgpf11 / T4 | 48 | 7.0 | 151 (patience 50) | 0.529 | 0.283 | 0.571 | 0.000 | 0.219 |
| baseline-repro-p50/seed2 | 945410 | xgpd2 / Titan V | 32 | 5.4 | 137 (patience 50) | 0.489 | 0.215 | 0.633 | 0.000 | 0.356 |

Round 1 first epochs (measured, T4 xgpf10/xgpf11, batch 48):
yolo11n-v1clean ≈ 7.4 s/epoch, yolo26n-v1clean ≈ 8.3 s/epoch.

Every run in the table has test small-pigeon recall@P90 = 0.0 (measured).
v1clean val has no empty or confuser images. Its selected thresholds
(0.99 and 0.76) therefore ignore false alarms.

Artifacts:

- Cluster: `~/columba/runs/baseline-repro{,-p50}/seed{0,2}/`.
- Local: `experiments/runs/baseline-repro{,-p50}/seed{0,2}/` (git-ignored).
