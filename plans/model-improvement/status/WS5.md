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
2. Round 0: seeds 0 (T4) and 2 (Titan V) DONE for both `baseline-repro`
   and `baseline-repro-p50`. The coordinator made `baseline-repro-p50` the
   section-1 baseline. `baseline-repro` stays a recorded 2-seed
   reproduction: its seed 1 (945341) was cancelled by the coordinator.
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
8. Round 1 v1clean: all 4 DONE and fetched. `yolo11n` (945490), `yolo26n`
   (945491), `yolo11s` (945564), `yolo26s` (945565).
9. `cluster/excluded_nodes.txt` (xgpe6, xgpe2) is read by `cluster.sweep`
   for every submission except `gpu-probe`. `--exclude-nodes` now adds to
   the list instead of replacing it. Test:
   `tests/test_cluster.py::test_excluded_nodes_file_is_default_and_extras_are_added`.
10. Teacher memory probe **945775** (`cz-ws5-teacher-mem`, one manual
    Titan V job, xgpd2, done). Measured at 960 with AMP:
    - yolo11x: 2.1 GiB/sample.
    - yolo26x: 2.6 GiB/sample.
    - Batch 6 OOMs on 12 GB.

    Teachers set accordingly: yolo11x on Titan V with `batch_override: 4`,
    yolo26x on Titan RTX with `batch_override: 6`. Results in
    `~/columba/runs/ws5-probes/teacher-mem.json`; the script is
    `~/columba/scratch/ws5/teacher_mem.py` (local copy in `outputs/ws5/probe/`).
11. Round 0 complete. `baseline-repro-p50` seed 1 (945876, Titan V) DONE
    and fetched. Section-1 baseline, mean ± sd over 3 seeds: test R@P90
    **0.246 ± 0.035**, test AP50 0.604 ± 0.031, val R@P90 0.473 ± 0.066.
12. `yolo11n-v2-640-r1` (945886, T4) DONE and fetched: test R@P90
    **0.691**, AP50 0.788, confuser FAR 0.041, empty FAR 0.000.

## In progress

Check-in 16 Oct (main 88eabb0 merged).

- **Round 3 (species gate, PLAN 5.4): decided at training size, provisional
  until WS7.** Full table in `experiments/reports/WS5-round3.md`; rows are
  fetched to the leaderboard.

  | Clause | Species vs single, 3 seeds each |
  |---|---|
  | Recall | **pass**: test R@P90 0.791 ± 0.013 vs 0.780 ± 0.024; val 0.818 ± 0.004 vs 0.794 ± 0.006 |
  | Latency | **pass**: +1.9% est. Pi p95 at 448×800 |
  | False alarms | **fail** (strict, test): empty FAR 0.008 (1 street scene on every seed) vs 0.000; confuser 0.041 vs 0.032. No worse on val. |

  - **Decision:** do not ship species. Ship the single-class model, trained
    with the species images as unlabelled negatives (v2s single).
  - WS7's deploy-size gate (`select_model.py enforce_species_gate`) makes
    the final call. The seeds for WS7, chosen on val:
    `yolo26n-v2s-species-800-r3/seed2` and its single reference
    `yolo26n-v2s-800-round2-plain/seed1`.
  - **Proposed, not started:** a single-taxonomy control at
    `batch_override: 16` on Titan V (3 seeds, ≈ 17 nv GPU-h). It would
    separate the batch/GPU confound from the taxonomy effect.
- **Distill-640 complete** (`WS5-round2.md`):
  - val R@P90 0.807 ± 0.009 vs plain 0.788 ± 0.010, but test 0.733 ± 0.013
    vs 0.743 ± 0.003.
  - Test precision at the val threshold is < 0.90 on every seed.
  - Confuser FAR 0.050 (s2 breaches).
  - No improvement on test.
- **Distill-800** running (951222 s0/s2 on h100-47, 951220 s1 on A100-40).
  A placeholder is in `WS5-round2.md`.

## Blockers and requests

- Turn 3 SSH outage: resolved by turn 4.
- Bad nodes: **xgpe6** and **xgpe2** (Titan RTX, "No devices were found";
  probe 945282, job 945409). **xgpe3** is down ("Node unexpectedly
  rebooted"). Only **xgpe4** is a working Titan RTX.

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

Test precision/recall at the val threshold, and the round-1 table with all 4 v1clean runs, are in
`experiments/reports/WS5-plan.md`.

Round 1 (measured, T4 xgpf10/xgpf11, nano batch 48, 150-epoch budget):

| Run | Job | s/epoch | Epochs | val R@P90 | test R@P90 | test AP50 | test empty FAR | test confuser FAR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo11n-v1clean-640-r1/seed0 | 945490 | 7.4 | 117 | 0.507 | 0.238 | 0.591 | 0.008 | 0.219 |
| yolo26n-v1clean-640-r1/seed0 | 945491 | 8.3 | 136 | 0.664 | 0.006 | 0.643 | 0.000 | 0.301 |
| yolo11s-v1clean-640-r1/seed0 | 945564 | 12.0 (batch 24) | 126 | 0.475 | 0.365 | 0.646 | 0.000 | 0.110 |
| yolo26s-v1clean-640-r1/seed0 | 945565 | 14.6 (batch 24) | 114 | 0.525 | 0.190 | 0.618 | 0.000 | 0.274 |

yolo26n's test precision never reaches 0.9 (precision 0.866 and recall 0.496
at the val threshold), so test R@P90 collapses. See the caveat in
`experiments/reports/WS5-plan.md`.

Every run except yolo11s (0.039) has test small-pigeon recall@P90 = 0.0 (measured).
v1clean val has no empty or confuser images. Its selected thresholds
(0.99 and 0.76) therefore ignore false alarms.

Artifacts:

- Cluster: `~/columba/runs/<recipe_id>/seed<k>/`.
- Local: `experiments/runs/<recipe_id>/seed<k>/` (git-ignored).
