# Experiments

Small, committed summaries of training runs and benchmarks for the model
improvement plan. Formats are defined in
[plans/model-improvement/CONTRACTS.md](../plans/model-improvement/CONTRACTS.md).
Weights, datasets, NCNN exports and full run folders never go into git.

| Path | Contents | Contract | In git |
|---|---|---|---|
| `leaderboard.csv` | One row per training run, failures included. Updated by `cluster/sweep.py fetch`. | C6 | yes |
| `latency/<host>-<model_id>-<format>-<h>x<w>.json` | One latency benchmark result each. | C7 | yes |
| `latency/calibration.json` | Factors that turn `xcng` timings into estimated Pi 5 timings. | C7 | yes |
| `reports/WS<N>-*.md` | Workstream reports: decisions and the evidence behind them. | — | yes |
| `runs/<recipe_id>/seed<k>/` | Local copies of cluster run folders (`run.json`, weights, metrics, `DONE`). | C6 | no (ignored) |

Every number in these files is labelled **measured**, **estimated** or
**simulated**, and names the host it came from. Desktop and cluster timings are
not Pi results.
