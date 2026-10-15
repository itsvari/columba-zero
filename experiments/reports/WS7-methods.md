# WS7 reproduction and hand-off

Use the shared environment without changing its pinned packages.

```sh
export PATH="/Users/vari/Developer/columba-zero/.venv/bin:$PATH"
```

## Add another candidate

`add` requires three completed C6 runs. It fetches their checkpoints, exports
both heads at the supplied deploy size, computes CPU PyTorch references and
submits one guarded ARM job. That job calibrates int8 on 200 validation images,
scores every precision on val, test-v2 and test-v2-styles, and measures native
latency. It retains all three seeds. It never changes the training runs.

For the species arm, after the coordinator confirms the recipe is ready:

```sh
python -m tools.select_model add \
  --recipe yolo26n-v2s-species-800-r3 \
  --input-hw 448 800 \
  --taxonomy species \
  --dataset-name columba-train-v2s-species \
  --single-recipe yolo26n-v2s-800-round2-plain \
  --reference-data-root /Users/vari/columba-data \
  --output outputs/ws7-later
```

For a distilled or other single-taxonomy candidate, use its actual recipe ID,
omit the species flags and supply its deploy size. A 640-trained model needs
`--input-hw 384 640`. The `add` command does not wait for unfinished training.

After the submitted ARM bundle finishes:

```sh
python -m tools.select_model fetch \
  --recipe yolo26n-v2s-species-800-r3 --output outputs/ws7-later
python -m tools.select_model select \
  experiments/reports/WS7-candidates outputs/ws7-later/results \
  --output experiments/reports/WS7-selection.json \
  --report experiments/reports/WS7-provisional.md
```

Pass any number of additional candidate roots to `select`. Keep one copy of
each recipe/head/precision/seed in those roots. Duplicate seed records fail
the three-seed gate. C4 and C7 files, plus parity and accuracy-gate evidence,
drive selection; no inference or threshold fitting occurs during selection.

Species scoring uses an isolated taxonomy view of each single-taxonomy test
set. `cluster.staging.align_test_taxonomy` changes only that view's YAML.
Pigeon remains class 0. The original data archive and extracted dataset stay
unchanged. The selector also compares a species arm with its named matching
single-taxonomy arm under PLAN section 5.4.

## Replay individual stages

```sh
python tools/export_ncnn.py --help
python tools/quantize_ncnn.py --help
python -m tools.select_model --help
```

`quantize_ncnn.py` runs only on the xcng ARM lane. Its provenance records every
selected validation image and SHA-256, the source checkpoint, calibration
tools and frozen test manifest. It rejects test overlap. Both exported heads
use centred deployment letterbox, RGB /255 and padding 114.

`predict-native` runs the deployed `columba_native` path, with package-default
compute and four threads. `evaluate` selects a fresh operating threshold on
val for every export. Test and styles reuse only that export's val threshold.
`remeasure` measures native end-to-end time with `OMP_WAIT_POLICY=ACTIVE`,
retaining an older PASSIVE result as a diagnostic. Each seed-0 package also
has a WS4 `bench_ncnn` bridge measurement. The native measurement includes the
extension call and the Python return/decision overhead.

The initial matrix was already running when styles was requested. A second
ARM job added styles and ACTIVE timing to completed candidates without
restarting it. CPU references computed on the Mac were published only for
cases that had not started. The first seed's ARM CPU reference validated
all 1,408 val/test image comparisons against the Mac CPU reference. The
remaining references use the Mac CPU path. GPU references failed this check
and remain excluded diagnostics.

## Audit and packaging

The complete raw evidence and C8 exports are under `outputs/ws7/` locally and
`~/columba/scratch/ws7/` on the cluster. Raw prediction arrays and model
weights are deliberately absent from Git. The committed `WS7-candidates/`
contains the small C4/C7, provenance, parity and accuracy-gate files needed to
rerun selection. The audit replays parity from the raw predictions and checks
the actual package bytes and validation-only calibration.

```sh
python -m tools.select_model audit \
  --candidates outputs/ws7/results --exports outputs/ws7/exports \
  --dataset /Users/vari/columba-data/columba-train-v2 \
  --test-dataset /Users/vari/columba-data/columba-test-v2 \
  --out experiments/reports/WS7-audit.json
python -m tools.select_model package \
  --decision experiments/reports/WS7-selection.json \
  --exports outputs/ws7/exports \
  --report experiments/reports/WS7-provisional.md
```

The package command requires an admitted candidate. When selection is HOLD,
an explicit `--diagnostic` permits the C8/hash dry run with a parity-safe,
accuracy-safe candidate. That package remains blocked by the recorded gates.
It does not turn HOLD into a selection. The manifest entry is provisional and
`prepare_models.py --model-id ID` verifies its SHA-256s and byte counts. The
existing default model entry stays active. Final promotion belongs to the
coordinator after distilled and species results arrive.

The initial blocked diagnostic package can be fetched and verified in a
checkout containing this branch's manifest:

```sh
rsync -a -e 'ssh -o BatchMode=yes' \
  columba-cluster:columba/scratch/ws7/packages/ outputs/ws7/packages/
python tools/prepare_models.py \
  --model-id yolo26n-v2s-800-round2-plain-seed1-yolo11-fp32-provisional
```

The original matrix is job 950463 on xcng0. The supplemental timing/styles
job is 950885 on xcng1. Their results, exports and logs remain under the WS7
scratch root. A diagnostic T4 reference job, 950509, was cancelled by WS7
after 10 min 59 s because its reference parity failed. No WS5 job was changed
or cancelled. The initial matrix used PASSIVE accuracy; the controlled
same-package comparison reproduced identical predictions for fp32, fp16 and
int8 under ACTIVE on all 319 test images. The committed
[`WS7-platform/`](WS7-platform/) records those hashes, timing probes, CPU
reference comparisons and input-hash diagnostics.

## Policies requiring coordinator review

- Document optional `omp_wait_policy` in C7 and set `OMP_WAIT_POLICY=ACTIVE`
  in the deployed launcher. PASSIVE more than doubled NCNN inference time in
  a controlled same-allocation comparison. WS7 does not own that launcher.
- Confirm the explicit precision gate: each val/test R@P90, AP50 and operating
  recall may drop at most 0.01 versus the same-head rectangular PyTorch run.
  Strict export parity remains a separate gate.
- Confirm per-seed false-alarm/time/parity gates and the styles tie policy:
  otherwise matching v2/v2s configurations within 0.01 validation recall and
  5% latency use mean styles recall. Test-v2 supplies all promotion gates.
- Review the temporal approximation before deployment. Reaction estimates
  assume independent frame hits and proxy per-frame recall with the measured
  still-image operating recall. No measured Pi 5 or correlated-video result
  is claimed.
