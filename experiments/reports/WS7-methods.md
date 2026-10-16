# WS7 reproduction and hand-off

Use the shared environment without changing its pinned packages. Run commands
from the repository root. Cluster access is exclusively BatchMode SSH. If the connection fails, retry
`ssh -o BatchMode=yes columba-cluster true` every 60 seconds for up to ten
minutes before stopping and reporting. The coordinator's Mac watchdog normally
restores the tunnel within 30–60 seconds. Never supply a password.

```sh
export PATH="/Users/vari/Developer/columba-zero/.venv/bin:$PATH"
export YOLO_CONFIG_DIR="$PWD/outputs/ws7-final/settings"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=6
python -c 'from tools.predict_to_coco import configure_ultralytics; configure_ultralytics()'
```

## Add and fetch a candidate

`add` requires completed C6 runs for seeds 0, 1 and 2. It fetches immutable
checkpoints, exports the standard head in fp32 and fp16, and computes CPU
PyTorch references on val, test-v2 and styles. One guarded ARM bundle then
scores native exports at the deploy size, derives each export's threshold on
val, checks parity and precision accuracy, and measures native latency.
Int8 and one-to-one heads are excluded from final deployment. No GPU is needed.

The final distilled-arm command was:

```sh
python -m tools.select_model add \
  --recipe yolo26n-v2s-800-round2-distill --input-hw 448 800 \
  --decodes yolo11 --precisions fp32 fp16 \
  --dataset-name columba-train-v2s --node xcng0 \
  --output outputs/ws7-final
```

Use the new recipe ID when adding another single-taxonomy candidate. A
640-trained model needs `--input-hw 384 640`. For species, add
`--taxonomy species --dataset-name columba-train-v2s-species
--single-recipe yolo26n-v2s-800-round2-plain`. `--node xcng1` can use the other
ARM node. At most two project CPU jobs may be submitted or running; use
different nodes for simultaneous timing bundles. `add` checks the live queue
and uses `cluster.slurm`'s atomic budget guard. It never modifies training jobs.
Do not repeat `add` for an already completed bundle; fetch its evidence.

After its DONE marker exists:

```sh
python -m tools.select_model fetch \
  --recipe yolo26n-v2s-800-round2-distill --output outputs/ws7-final
python -m tools.select_model evidence \
  --candidates outputs/ws7-final/results --exports outputs/ws7-final/exports \
  --output experiments/reports/WS7-final-candidates
python -m tools.select_model select \
  experiments/reports/WS7-candidates experiments/reports/WS7-final-candidates \
  --final --output experiments/reports/WS7-final-selection.json \
  --report experiments/reports/WS7-final.md
```

Supply any number of additional candidate roots to `select`. Keep one copy of
each recipe/head/precision/seed. Duplicate seeds fail the three-seed gate.
Keep the fp32 sibling evidence: final deployment considers fp16, while its
same-graph fp32 strict comparison supplies the parity gate. Selection reads
C4/C7 and saved parity/accuracy evidence; it performs no inference or fitting.
The default baseline is native fp16 at 384x640, all three seeds, under
`experiments/reports/WS7-baseline/fp16`.

## Seed and evaluation policies

Contracts v1.9 and PLAN section 1 govern exports. The coordinator's 16 October
clarification, recorded in `WS7-final-policy.json`, confirms that deploy-size
val takes precedence over WS5's square-val seed nomination. Every arm uses
the same seed rule: highest val R@P90 among val FAR passes, then lower val
confuser FAR, lower val empty FAR, higher val AP50, lower seed index.
This rule was confirmed before inspecting deploy-size test results for any
switched seed. Test failure holds that arm; another seed is never retried.

Arm ranking uses three-seed mean val R@P90. The improvement rule compares
three-seed test mean against baseline by more than the larger sample standard
deviation. The chosen export alone supplies test FAR, latency, reaction,
precision and parity gates. Matched v2/v2s ties within 0.01 val and 5% latency
use styles recall. Styles never replaces test-v2 promotion gates.

Each fp16 export must lose at most 0.01 in each val/test R@P90, AP50 and
operating recall against the same-head rectangular CPU PyTorch reference.
Its same-graph fp32 export must pass strict comparison: equal counts,
IoU >=0.8 and score difference <=0.05 at confidence 0.25. Strict fp16 per-image
parity is a reported diagnostic. Historical int8 diagnostics use 0.7 / 0.10
and validation-only calibration on 200 images, with no test overlap.

Species test sets use a temporary copy with taxonomy names aligned by
`cluster.staging.align_test_taxonomy`. Files stay inside the copy, satisfying
the evaluator's manifest boundary. Source datasets stay immutable. Pigeon
remains class 0; non-pigeon classes do not trigger. PLAN 5.4 compares the
species arm with its named single-taxonomy reference.

## Evidence and timing

Raw C3 arrays and all experimental exports remain ignored under `outputs/ws7/`
and `outputs/ws7-final/`, with ARM copies in `~/columba/scratch/ws7/`.
`evidence` collects small C4/C7 files, C3 sidecars and provenance into Git.
`audit` reproduces saved strict parity and verifies actual package bytes:

```sh
python -m tools.select_model audit \
  --candidates outputs/ws7-final/results --exports outputs/ws7-final/exports \
  --dataset /Users/vari/columba-data/columba-train-v2s \
  --test-dataset /Users/vari/columba-data/columba-test-v2 \
  --out experiments/reports/WS7-final-audit.json
```

Accuracy is measured via deployed `columba_native` package-default compute on
allocated xcng cores. CPU PyTorch references use Mac vexcalibur; the original
first-reference comparison passed all 1,408 val/test images against xcng.
GPU references were excluded after that comparison failed.

C7 records four pinned cores, A76-compatible NCNN build flags, default compute
flags, OMP_WAIT_POLICY=ACTIVE, 20 warm-ups and 200 timed iterations. It includes
native letterbox, inference, decode/NMS, Python return and a bool decision.
Benchmark source frames are 480x640, recorded as `camera_hw`, resized from the
first 20 val photos. Pi p95 is an estimate using xcng x1.527110713, not a Pi
measurement. WS4 reaction is simulated at 30 FPS, k2/n3, 20,000 trials, seed42,
independent hits, with still-image operating recall as the frame-recall proxy.
The coordinator now authorises `camera.fps: 30`, leaving `recording_fps`
unchanged. The simulator already samples serial service durations and selects
the latest available capture after each service, dropping stale frames. Its
30 FPS result is throughput limited, not 30 detections per second. Final
selection reproduces the reaction gate and reports an effective detection rate
from a 20,000-frame stationary scheduling replay using the same Pi latency
samples. Do not replace capture FPS with detector FPS and count the service
limit twice. WS10 step 3 must measure real reaction time.

WS10 step 3 and steps 5–7 remain open: real reaction, site threshold/k-n, Pi OMP and on-Pi
geometry. The launcher stays unchanged. `columba-test-site-v1` must check the
real camera; frozen test-v2 has only nine approximately 16:9 images.

## Durable packaging

Final model files follow the incumbent's directly tracked convention under
`runs/detect/pigeon_detector/<model-id>/weights/best_ncnn_model/`. The existing
baseline files are unchanged. Git attributes preserve exact param/bin bytes
and LF metadata. There is no Git LFS filter. Only the shipped fp16 package and
the small plain-800 fallback are added; experimental weights remain ignored.

```sh
python -m tools.select_model package --final \
  --decision experiments/reports/WS7-final-selection.json \
  --exports outputs/ws7-final/exports \
  --fallback-candidate experiments/reports/WS7-candidates/yolo26n-v2s-800-round2-plain/seed1/yolo11/fp16 \
  --fallback-exports outputs/ws7/exports \
  --report experiments/reports/WS7-final.md
python tools/prepare_models.py
python tools/prepare_models.py --model-id yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16
```

Use the selected export's root for `--exports`. The manifest contains only
repository-relative durable source/runtime paths, checkpoint SHA-256,
val-derived threshold and each file's SHA-256/byte count. Default preparation
verifies and copies the winner; `--model-id` also selects the named fallback.
Clean-clone tests verify both packages with autocrlf false and true.

## Earlier diagnostics

The original 54-case matrix was job950463, supplementary styles/ACTIVE timing
job950885, and rectangular baseline job951221. WS7 cancelled only its own
failed GPU reference job950509. No training job was changed or cancelled.
`WS7-platform/` records ACTIVE/PASSIVE prediction equality, reference parity,
input hashes and timing probes. `WS7-provisional.md` retains int8 and one-to-one
diagnostics; the original audit is `WS7-audit.json`.

`WS7-aspect.{json,md}` compares plain-800 seed1 PyTorch at 800x800 and 448x800.
It separates shape-specific WS1 small-object cohorts from a fixed square-defined
cohort. Portrait images account for 38 of 39 lost matches at the full-test P90
cut. This analysis is diagnostic; it never changes a deployment threshold.
The full reproduction commands remain available in the prior committed version
of this methods report and `tools.select_model aspect --help`.
