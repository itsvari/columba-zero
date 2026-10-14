# WS1 — Evaluation harness

- State: done
- Scope: tasks 1–5, including coordinator follow-up contamination analysis
- Branch: `ws1/evaluator`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14

## Done

1. Tasks 1–4 were merged previously: C3 PyTorch/NCNN prediction, rectangular
   letterboxing, C4 pycocotools metrics, validation-only threshold selection and
   hand-built tests. Native NCNN compute defaults are retained; `--ncnn-fp32`
   remains an optional parity mode. Original smoke artifacts are in
   `outputs/ws1/`, as recorded in the prior status commit `4864e68`.
2. Fast-forwarded onto main `a15ed4c`, read WS2's publication notes and CONTRACTS
   v1.4, and verified `columba-test-v2` against its committed freeze file before
   prediction and on every reproduction run.
3. Re-scored `pigeon_baseline_v1-2`: PyTorch 640×640, deployed NCNN 640×640 and
   PyTorch 384×640. Each configuration selects its operating threshold on the
   full `columba-train-v1clean` val split and reuses it for full/unseen test.
4. Reconstructed original Roboflow and COCO membership from the notebook/raw
   exports. Conservative exposure uses WS2's original-stem/scene/SHA-256 and
   mirror-phash connected components, with train preceding validation. A
   visually false phash match is rejected in a committed review record.
   Every test and validation image has committed exposure evidence.
5. Added `--subset ids.txt` and `evaluate(..., subset=...)`. Unknown or empty
   subset IDs fail; predictions are validated against the full split before
   filtering. Added tests for subset behaviour, exposure precedence and the
   notebook's COCO splitting method.
6. Committed a reproducible driver, metric JSONs, input fingerprints, C3
   metadata and six C6 leaderboard rows. The driver uses WS3's
   `leaderboard_row` and `update_leaderboard`; recipe IDs distinguish input
   size and full/unseen scope. No weights or prediction lists enter git.

## Results

All metrics/counts are **measured on Mac M2 Pro**, host `vexcalibur`.
Exposure membership is reconstructed and conservative, not a recovered log of
actual Windows training membership. Full detail and all metric tables:
[`experiments/reports/WS1-incumbent.md`](../../../experiments/reports/WS1-incumbent.md).

- Full-test pigeon recall_at_p90: PyTorch 640×640 **0.592068**; deployed NCNN
  640×640 **0.592068**; PyTorch 384×640 **0.339943**.
- Unseen-test pigeon recall_at_p90: PyTorch 640×640 **0.000000**; deployed NCNN
  640×640 **0.000000**; PyTorch 384×640 **0.142857**.
- Test exposure: **117 seen_train, 15 seen_val, 187 unseen**. The unseen view
  contains only **3 pigeon-positive images / 7 pigeon boxes**, plus **125 empty
  and 59 confuser images**. Of 121 positive images, 118 are exposed by family.
- Validation exposure: **156 seen_train, 0 seen_val, 0 unseen** by conservative
  grouping. Direct membership is 113 train, 36 val and 7 old test. No unseen
  validation metric/threshold is possible; JSON records `not_evaluable`.
- Deployed NCNN at its selected validation threshold **0.576171875**:
  full empty/confuser FAR **0.112000 / 0.561644**; unseen FAR
  **0.112000 / 0.576271**. All three configurations fail the false-alarm gates.
- **Recommendation:** log deployed NCNN unseen recall_at_p90 **0.0** as the
  provisional original-incumbent number, with its tiny denominator and
  contaminated threshold calibration. Keep full **0.592068** as a contaminated
  historical diagnostic. Neither supports the section-1 three-seed promotion
  claim. Use WS5's clean three-seed `baseline-repro` on full frozen test-v2 as
  the promotion reference when published; add fresh positive scenes and
  negative-inclusive validation for the next dataset version.

Artifacts:

- `experiments/incumbent/{pt-640x640,ncnn-640x640,pt-384x640}/`: metrics, run
  records, C3 metadata and fingerprints.
- `experiments/incumbent/contamination.json`, `exposure_{test,val}.jsonl`,
  `coco_original_splits.json`, `exposure_review.json`, `unseen_{test,val}.txt`.
- `experiments/leaderboard.csv`: three full and three unseen rows, no duplicates.
- Scratch predictions/logs: this worktree's `outputs/ws1/incumbent/`.
- Reproduction: `python -m experiments.incumbent.rescore --data-root ~/columba-data`.
  Published datasets, raw exports and original model files are read only.
  No cluster jobs or package changes were needed.

## Verification

All 121 unit tests pass in 2.453 s, measured on Mac M2 Pro. Ruff lint and format
checks pass on `app tests tools cluster training experiments/incumbent`; strict
mypy passes for the project and explicitly for the new driver.

Section-0 checks and direct artifact verification are recorded in
`outputs/ws1/incumbent/{unittest,ruff-check,ruff-format,mypy,rescore-mypy}.log`.
The new driver is included explicitly in Ruff and strict mypy checks.
Subset CLI output was compared with the committed unseen metrics, and
re-running the driver confirmed stable values and six-row leaderboard upserts.

## In progress

Nothing.

## Blockers and requests

No implementation blocker. The coordinator owns PLAN.md and CONTRACTS.md.

1. **PLAN section 1:** adopt the comparison-baseline recommendation above.
   The original incumbent lacks independent validation and has only seven
   unseen positive boxes. Do not treat its full score as an honest promotion
   baseline or its unseen 0.0 as enough evidence for a promotion claim.
2. **C4 subset wording proposal:** "`--subset FILE` restricts evaluation to
   split image IDs listed one per line. All prediction image/category IDs are
   validated against the full split before filtering. Unknown or empty subsets
   are errors. Metrics record the subset path. An unseen test view reuses the
   full validation threshold unless an independently selected validation file
   is explicitly supplied. It never selects an operating threshold on test."
3. **C4 provenance wording proposal:** "Validation-only threshold selection
   establishes split provenance, not independence from historical training.
   Report calibration-set exposure to the evaluated checkpoint separately.
   `recall_at_p90` on test is a diagnostic PR statistic, not the deployed
   operating threshold. Zero small ground truths yields small recall 0 with
   the zero denominator retained in counts; do not interpret it as a miss rate."
4. Historical COCO `getImgIds` ordering was not saved; current reconstruction
   order and the annotation hash are committed. The original Windows dataset
   is absent, and 87 old COCO bird images lack local hashes because WS2's
   licence filter excluded them. These limits are explicit in the report.
