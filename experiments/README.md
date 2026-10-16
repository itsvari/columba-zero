# Experiments

Committed summaries from the model-improvement project, which is closed except
for WS10 (Pi hardware acceptance). Formats are defined in
[CONTRACTS.md](../plans/model-improvement/CONTRACTS.md). Weights, datasets, NCNN
exports and full run folders never go into git. Every number is labelled
**measured**, **estimated** or **simulated** and names its host; desktop and
cluster timings are not Pi results.

## Reports

| Report | Outcome |
|---|---|
| [WS1-incumbent](reports/WS1-incumbent.md) | `pigeon_baseline_v1-2` re-scored on `columba-test-v2`: its test recall is contaminated by training families, so `baseline-repro-p50` became the comparison baseline. |
| [WS5-plan](reports/WS5-plan.md), [round 2](reports/WS5-round2.md), [round 3](reports/WS5-round3.md), [final](reports/WS5-final.md) | YOLO training rounds 0–3. Best at square training size: distill-800; species not shipped. Both were then judged by WS7 at deploy size. |
| [WS6-latency-gate](reports/WS6-latency-gate.md) | Challenger latency gate (D-FINE-N, RF-DETR Nano, NanoDet-Plus-m). WS6 closed at the gate; no challenger was trained. |
| [WS7-provisional](reports/WS7-provisional.md) | First deploy-size selection over the plain arms; int8 and one-to-one diagnostics. |
| [WS7-aspect](reports/WS7-aspect.md) | Square vs 448×800 accuracy: the rectangular loss comes almost entirely from portrait test images. |
| [WS7-final](reports/WS7-final.md) | **Shipped:** `yolo26n-v2s-800-round2-plain` seed 1, yolo11 decode, fp16, 448×800, threshold 0.6265. Distill-800, species and the batch-16 control are held. |
| [WS7-quant](reports/WS7-quant.md) | Int8 PTQ study: no int8 or mixed configuration passes the accuracy policy; fp16 kept, QAT proposed and deferred. |
| [WS7-methods](reports/WS7-methods.md) | Commands to reproduce WS7 selection and packaging. |

The WS7 reports' JSON companions (`WS7-*-selection.json`, `WS7-final-policy.json`,
`WS7-quant-results.json` and others) hold the full numbers. The bulk
per-candidate evidence directories were removed in the cleanup; they are in
git history at `6de04fb`.

## Files

| Path | Contents | Contract |
|---|---|---|
| `leaderboard.csv` | One row per training run, failures included; historical rows are never edited. Updated by `cluster/sweep.py fetch`. | C6 |
| `latency/` | WS4 latency records per host/model/format/size, `calibration.json`, `summary.md` and the [reaction budget](latency/reaction_budget.md). See [latency/README.md](latency/README.md). | C7 |
| `incumbent/` | WS1 re-score of v1-2 and its contamination analysis (`rescore.py` reproduces it). | C4 |
| `reports/WS7-baseline/` | Baseline-repro-p50 test metrics that `tools/select_model.py select` reads by default. | C4 |
| `runs/<recipe_id>/seed<k>/` | Local copies of cluster run folders; git-ignored. | C6 |
