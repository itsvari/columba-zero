# WS5 final report: YOLO training and distillation

16 Oct 2026. All WS5 training runs are finished and on the leaderboard.

- **Best recipe at training size:** `yolo26n-v2s-800-round2-distill`, a
  YOLO26n student distilled from the yolo26x 960-px teacher, trained at
  800 px on `columba-train-v2s`.
- **Species decision (PLAN 5.4):** do not ship species.
- Both are provisional on WS7. Under the 15 Oct PLAN section 1 decision,
  promotion is judged on native exports at the 16:9 deploy size.

All accuracy and false-alarm numbers are **measured** by WS1's evaluator on
the cluster:
- square at each run's training size;
- val = the v2/v2s val split (1,089 images with negatives);
- test = frozen `columba-test-v2` (319 images: 125 empty, 73 confuser, 353
  pigeon boxes, 147 small).

The operating threshold is chosen on val and reused on test. Latency is
**estimated** (WS4 records on xcng0, `cal-v0-published`), not measured on a
Pi. Details per round are in `WS5-plan.md`, `WS5-round2.md` and
`WS5-round3.md`.

## Outcomes by round

| Round | What | Outcome |
|---|---|---|
| 0 | `baseline-repro` (v1-2 args, patience 10) and `baseline-repro-p50` (patience 50), YOLO11n 640 on v1clean | Patience 10 stopped one seed at epoch 11, so the coordinator made **`baseline-repro-p50` the section-1 baseline**: test R@P90 **0.246 ± 0.035**, AP50 0.604 ± 0.031. `baseline-repro` stays a 2-seed record; seed 1 was cancelled while pending and never ran, so it has no row. |
| Teachers | yolo11x / yolo26x at 960 on v2s (fresh H100 reruns `-b16` / `-b12`; first attempts `cancelled-superseded`) | yolo26x: val/test R@P90 0.845 / 0.861. yolo11x: 0.822 / 0.870. The distillation rules picked **yolo26x** (higher val). |
| 1 | 12 screening runs, {11n, 26n, 11s, 26s} × {v1clean, v2, v2s}, 150 ep | v1clean runs reach test R@P90 0.006–0.365, limited by confuser false alarms. v2/v2s jump to 0.69–0.79 with confuser FAR mostly ≤ 0.05: **data was the largest gain**. The mechanical selection took yolo26n-v2s, yolo26n-v2 and yolo11n-v2s. s models missed the +0.04 val margin over nanos (yolo26s-v2s val 0.809 vs nano 0.783) and would fail the 70 ms target (82–84 ms) anyway. |
| 2 plain | 3 combinations × {640, 800} × 3 seeds, 300 ep, patience 50 | All 18 runs beat the baseline by about 0.5 test R@P90. **800 px wins**: yolo26n-v2s-800 test 0.780 ± 0.024 and yolo26n-v2-800 0.779 ± 0.022, vs 0.73–0.74 at 640. yolo11n-v2s-800 sits at the confuser gate (2 of 3 seeds over). |
| 2 distill-640 | yolo26n-v2s-640 + yolo26x teacher (mechanical, top_k 1) | Val +0.019, but test −0.010 (0.733 ± 0.013 vs 0.743 ± 0.003). Test P at the val threshold < 0.90 on every seed. **No improvement.** |
| 2 distill-800 | Same at 800 px (coordinator addition), matched to plain-800 | **Section-1 pass vs plain-800**: test 0.811 ± 0.011 vs 0.780 ± 0.024 (Δ +0.031 > 0.024). Small recall 0.603 vs 0.537, AP50 0.893 vs 0.860, AP50-95 0.529 vs 0.511. Every seed within the FAR gates (empty 0, confuser 2–3 of 73). |
| 3 species | yolo26n-v2s-800, species taxonomy, 3 seeds (Titan V, batch 16) | Recall and latency pass (+1.9% at 448×800). The strict FAR clause fails (1 empty false alarm per seed vs 0). **Not shipped.** |
| 3 control | Single taxonomy, same batch 16 on Titan V | test 0.790 ± 0.014 vs species 0.791 ± 0.013. Within species' sd, so by the pre-registered rule **the species recall gain is attributed to batch size**. The control becomes a WS7 candidate. |

## Candidates for WS7

Seeds are chosen on **val only**: the highest val R@P90 among seeds whose
val empty FAR is ≤ 0.02 and val confuser FAR ≤ 0.05. Exact val ties break
in this order: lower val confuser FAR, lower val empty FAR, higher val AP50,
lower seed index.

Weights:
- cluster: `~/columba/runs/<recipe>/seed<k>/weights/best.pt`;
- local fetched copy: `experiments/runs/<recipe>/seed<k>/weights/best.pt`
  (git-ignored).

All four candidates are deploy candidates at 448×800, also exportable at
384×640 and 352×608.

| # | Recipe / seed | Taxonomy | best.pt sha256 (first 16) | Val R@P90 | Val thr | Test R@P90 / AP50 (context) | Test FAs empty / confuser | Training-size gates |
|---|---|---|---|---:|---:|---|---|---|
| 1 | `yolo26n-v2s-800-round2-distill` / **seed1** | single | 5aa95a43ad780124 | 0.815 | 0.5846 | 0.799 / 0.900 (arm 0.811 ± 0.011) | 0 / 3 | Section 1 vs baseline: pass. vs plain-800: pass (+0.031). FAR: pass. Est. Pi p95 ≈ 51.6 ms (COCO yolo26n record, 448×800): pass. |
| 2 | `yolo26n-v2s-800-round2-plain` / **seed1** | single | 4b753fef63b230c0 | 0.801 | 0.6340 | 0.805 / 0.858 (arm 0.780 ± 0.024) | 0 / 1 | Section 1: pass. FAR: pass. Latency: pass. Current provisional single-class reference; seed 0 is ineligible (val confuser 0.060). |
| 3 | `yolo26n-v2s-species-800-r3` / **seed2** | species (pigeon-only scoring) | ec8c3b8960c56731 | 0.822 | 0.6098 | 0.802 / 0.888 (arm 0.791 ± 0.013) | 1 / 3 | Section 1: pass. 5.4 recall: pass. 5.4 latency: pass (+1.9%). **5.4 FAR: fail** (1 empty false alarm per seed vs 0). |
| 4 | `yolo26n-v2s-800-r3-control-b16` / **seed2** | single | c978c4a153a19812 | 0.808 (tie with s0, broken on val confuser 0.032 < 0.041) | 0.5751 | 0.805 / 0.883 (arm 0.790 ± 0.014) | 0 / 3 | Section 1: pass. FAR on the chosen seed: pass; the arm's s1 has 4 of 73 confuser FAs. Qualifies under the pre-registered rule. |

Training commits (`run.json.git_commit`): fd026c4, 10c98b3, 82554e7,
f722a9b. Distill-800 s1 resumed across GPU types (A100-40 → h100-47 MIG)
with batch fixed at 64.

## Recommendation

1. **Primary candidate: `yolo26n-v2s-800-round2-distill/seed1`.**
   - It is the only arm that clearly beats plain-800 under the section-1
     rule.
   - Its gains are broad, not one lucky threshold: test R@P90, small
     recall, AP50 and AP50-95.
   - Its false alarms are within the gates on every seed.
   - The teacher is used only at training time. Deploy cost is the same as
     plain yolo26n at 448×800.
2. **Reference and fallback: `yolo26n-v2s-800-round2-plain/seed1`.** WS7 uses
   it as the matched single for the species gate. It is the fallback if the
   distilled export fails parity or a deploy-size gate.
3. **Species: do not ship.** With the control explaining its recall gain by
   batch size, species offers no recall advantage over a matched single
   model, and it adds 1 empty-scene false alarm per seed. WS7 can still run
   `enforce_species_gate` on `species/seed2` vs `plain/seed1` at deploy size
   for the record. The training-size evidence points to the single-class
   model trained with species images as unlabelled negatives.
4. **Control b16:** a valid candidate, but it is dominated by distill-800
   on every training-size accuracy metric. Score it in WS7 for
   completeness. Its main lesson is that batch 16 (Titan V) slightly helps
   single-class models. A future distill-800 at batch 16 is untested.

**WS7's deploy-size numbers make the final call.** WS7's provisional
deploy-size scoring (16:9, fp16) gives yolo26n-v2s-800 test R@P90 0.686,
against 0.78 square at training size. Deploy-size numbers can therefore
reorder candidates; re-rank them on WS7's numbers, by val. On `columba-test-v2-styles`, v2s keeps the advantage over
v2 (0.52 vs 0.36 for yolo26n-800), which supports v2s for all four
candidates.

## Known limits

- **One seed per GPU type** in places, and different GPUs between arms
  (H100/A100 at batch 64 vs Titan V at batch 16). Seed and hardware effects
  cannot be fully separated.
- **False-alarm resolution is coarse:** 1 confuser FA = 0.0137 and 1 empty
  FA = 0.008. Several decisions turn on single images.
- **Test-v2 limits** (PLAN section 1): positives come almost entirely from
  one Roboflow source, and 196 of 212 small boxes (at 384×640) come from one
  depot session. Site footage (WS10, `columba-test-site-v1`) should re-test
  the winner.
- Latency figures are estimates scaled from xcng0; no Pi measurements exist
  yet.
