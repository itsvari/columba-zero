# WS6 challenger latency gate

Step 1 only, 14 October 2026, branch `ws6/challengers`. No training, recipes,
shared-venv changes, WS4 edits, push or merge.

NanoDet-Plus-m is a **latency-only survivor** at 384×640 in fp16. It does not
have an unqualified combined correctness/latency pass: fp32 matches the
framework, but fp16 changes threshold crossings and NMS selections. Hold
challenger training until that precision issue is resolved. D-FINE-N and
RF-DETR Nano fail export with the pinned converter/runtime.

## Gate at 384×640

All latencies are end-to-end p95 in milliseconds. Mac values are **measured**
on shared M2 Pro host `vexcalibur`, a76-compatible NCNN. xcng values are
**measured** on Neoverse-N1 host `xcng0`, four threads pinned to cores 0–3.
Pi values are **estimated**, never measured on a Pi.

| COCO model | Route / correctness | Mac p95 | xcng p95 | Estimated Pi p95 | ≤100 ms | ≤70 ms | Training decision |
|---|---|---:|---:|---:|---|---|---|
| D-FINE-N | TorchScript and ONNX → PNNX; NCNN cannot load | unavailable | unavailable | unavailable | FAIL export | FAIL export | Drop |
| RF-DETR Nano | TorchScript and ONNX → PNNX; NCNN cannot load | unavailable | unavailable | unavailable | FAIL export | FAIL export | Drop |
| NanoDet-Plus-m, fp16 | TorchScript → PNNX; fp32 parity passes, timed fp16 parity fails | 28.72 | 62.00 | 94.68 | PASS latency only | FAIL | Hold |
| NanoDet-Plus-m, fp32 | Same graph, fp32 computation; parity passes | 33.86 | 202.31 | 308.96 | FAIL | FAIL | Drop this precision |
| YOLO11n, fp16, WS4 reference | Verified C8 export | 10.63 | 25.31 | 38.65 | PASS | PASS | Main line |
| YOLO26n, fp16, WS4 reference | Verified C8 export | 10.33 | 24.16 | 36.89 | PASS | PASS | Main line |

YOLO reference values come from the camera 640×480 section of
`experiments/latency/summary.md`. Challenger measurements use the real
Ultralytics bus image resized to the same camera size. NanoDet uses its own
0.05 score floor, IoU 0.6 class-aware NMS and maximum 100 detections; WS4 YOLO
uses confidence 0.5, IoU 0.7 and maximum 300. Thus this compares complete
configured pipelines at the same dimensions, not matched postprocessing cost
or accuracy. NanoDet fp16 has only **estimated 5.32 ms** hard-limit headroom
and is **estimated 2.57×** slower than YOLO26n.

## Export routes and operator failures

### D-FINE-N

COCO-only `dfine_n_coco.pth`, 4M parameters in the upstream model table.
Configuration `configs/dfine/dfine_hgnetv2_n_coco.yml`. Backbone pretraining
is disabled because the full checkpoint supplies its weights. Change the
shared `eval_spatial_size` to `[384, 640]` and regenerate only
`decoder.anchors` / `decoder.valid_mask`, whose saved square shapes differ.
The committed wrapper verifies these are the only missing checkpoint keys.
Deploy mode, batch 1, packed output `(300, 84)`: normalised cxcywh and 80 raw
class logits. Preprocessing follows upstream inference: PIL bilinear direct
RGB resize and `/255`, **without ImageNet normalisation or letterboxing**.
Decode sigmoid logits, top 300 query/class pairs and map boxes to original
pixels; no NMS.

PyTorch forward and ONNX opset 17 export succeed. TorchScript → PNNX emits an
unloadable graph containing `torch.topk`, `torch.gather` and `F.linear`.
NCNN reports `layer torch.topk not exists or registered`; WS4 Engine load
returns -1. ONNX → PNNX also fails: its constant-folding stage reports an
invalid decoder TopK node with one output instead of the required two.
D-FINE uses grid sampling for deformable attention, but the observed decisive
failure is the retained TopK/Gather graph, not a demonstrated GridSample
registration failure. No custom attention or TopK runtime was attempted.
No parity or valid NCNN timing exists. Export failure is sufficient to drop it.

### RF-DETR Nano

Pinned `rfdetr==1.3.0`, `RFDETRNano`, COCO nano checkpoint from the package's
own download map. Batch 1 at 384×640, packed `(300, 95)` output: normalised
cxcywh plus 91 logits in the framework's sparse COCO category index space.
PIL RGB direct resize, `/255`, ImageNet mean `[.485,.456,.406]` and standard
deviation `[.229,.224,.225]`. Sigmoid/top-300 query-class pairs, no NMS.

The package's standard rectangular exporter incorrectly requires dimensions
divisible by 14 despite Nano's patch size 16. The wrapper exports the underlying
model directly, preserving its weights. Original ONNX export rejects
`aten::_upsample_bicubic2d_aa`. For the fixed input size, precompute the unchanged
positional interpolation and register it as a buffer. This allows opset 17
ONNX export. TorchScript → PNNX still exits 245 during pass_level1, including
on the rerunnable committed wrapper. ONNX → PNNX finishes but leaves
`torch.tile`, `TopK`, `GatherElements`, `torch.where` and expressions;
conversion also warns about unsupported 5-rank reshape/permute operations.
NCNN reports `layer torch.tile not exists or registered`; Engine load returns
-1. Deformable attention is present, but unsupported tensor/index operations
already prevent a usable export. No parity or valid timing exists. Drop it.

### NanoDet-Plus-m

Smallest upstream Plus variant: ShuffleNetV2 1.0×, 320-trained COCO checkpoint.
The graph exports at all three requested rectangles. TorchScript output
`N×112` contains 80 **raw class logits** and four distributions with eight
bins each. The upstream ONNX-specific forward applies sigmoid, so this wrapper
uses normal TorchScript forward and applies sigmoid once in NumPy.

An initial `[0]` batch slice generated a non-convertible Slice and crashed
inference. The committed export uses `.squeeze(0)`; PNNX still warns about the
batch squeeze, but emits a native NCNN Squeeze and the graph runs correctly.
The warning is not accepted as proof; real-image parity below establishes it.

Preprocessing exactly matches the framework's validation Pipeline at each
rectangle: BGR perspective warp for direct resizing, no letterbox, normalise
with mean `[103.53,116.28,123.675]`, std `[57.375,57.12,58.395]`. Priors are
integer stride centres, **no half-cell offset**, strides 8/16/32/64 with ceil
feature sizes. Distribution softmax → expected distances → input-space clip
→ class-aware NMS at IoU .6, floor .05, max 100 → original pixels.

## Correctness evidence

Reference images: Ultralytics `bus.jpg` and `zidane.jpg`, used only as export
smoke inputs. NPZ references contain the upstream Pipeline tensor, raw PyTorch
head output and the framework's `head.get_bboxes` result. Our preprocessing
matches the Pipeline **exactly, measured maximum absolute error 0**.

Across both images and all three sizes, fp32 NCNN parity passes on Mac and
xcng0. Mac **measured** maximum raw error is 0.000064, decoded box error
0.02863 px, score error 0.00000124. NumPy decode versus framework decode has
maximum box error 0.02863 px and score error below 0.00000003. The committed
NanoDet export wrapper was rerun and its exported graph passed the same checks.
Full per-image ARM and Mac values are in
`experiments/latency/ws6-challenger-gate-evidence.json`.

The timed fp16 computation does **not** pass the same final-detection check
at floor .05. On bus, NMS selects a different low-score hypothesis; matching
final detections yields a 159.34 px maximum box difference despite a maximum
matched score difference of .00554. On zidane the final top-100 class counts
differ. At confidence .5, bus matches within .185 px / .00537 score, but zidane
has 3 detections instead of the fp32 reference's 2 because a score crosses the
threshold. This is precision-sensitive selection, not evidence that the
fp32 export has the wrong head layout. Nevertheless, the fast runtime cannot
be labelled parity-qualified. No looser tolerance or confidence threshold was
chosen to turn this into a pass, and no training follows this gate.

## Additional sizes and timing boundaries

| Size | Precision | Measured Mac p95 | Measured xcng p95 | Estimated Pi p95 | Hard / target |
|---|---|---:|---:|---:|---|
| 352×608 | fp16 | 26.52 | 56.44 | 86.18 | PASS / FAIL, latency only |
| 384×640 | fp16 | 28.72 | 62.00 | 94.68 | PASS / FAIL, latency only |
| 448×800 | fp16 | 36.21 | 77.21 | 117.90 | FAIL / FAIL |
| 352×608 | fp32 | 29.42 | 141.35 | 215.86 | FAIL / FAIL |
| 384×640 | fp32 | 33.86 | 202.31 | 308.96 | FAIL / FAIL |
| 448×800 | fp32 | 45.37 | 94.21 | 143.88 | FAIL / FAIL |

WS4's custom bridge is reused unchanged. All runs use 20 warm-up and 200 paired
iterations, four NCNN threads, OpenCV preprocessing threads 1. Both precision
modes use fp32 stored weights, as in WS4; `fp16` selects packed/storage/arithmetic
computation. Inference includes Mat copy and extractor work. End-to-end
includes preprocessing, bridge/output copy, distribution decode, NMS, original
pixel mapping and a boolean detection decision. It excludes image-file I/O,
capture, display, tracking and confirmation, matching WS4's per-frame scope.

At 384×640 fp16, xcng **measured** stage p95 values are preprocessing 6.87 ms,
inference 21.24 ms and postprocessing 33.89 ms. End-to-end p95 is calculated
from paired samples, never by adding stage percentiles. NumPy GFL/NMS is a
large part of the cost; reporting inference alone would miss the gate's purpose.

The fp32 352/384 runs have large inference outliers: p50/p95 **measured**
24.94/107.82 ms and 27.91/163.17 ms respectively. The 448 run's 40.54/44.61 ms
shows these p95s are noisy, not a reliable size-scaling law. All raw samples
are retained; no outliers were removed or successful run substituted. Mac
measurements also use a separate host factor and are noisy on a shared machine.

Pi estimates multiply xcng paired p95 by **1.527110713**, calibration
`cal-v0-published`. Mac estimates use **3.792166804**, never the xcng factor.
These factors were calibrated on YOLO11n inference, not ShuffleNet/GFL/NumPy
postprocessing or DETR operator mixes. They are estimates with no Pi confidence
interval; WS10 must replace them with Pi measurements.

## Isolation, versions and licences

| Framework | Source commit | Environment | Licence |
|---|---|---|---|
| D-FINE | `956d1709314c2c6a4df6f34de232054578a7449f` | Python 3.12.3, torch 2.5.1, torchvision .20.1, transformers 4.49.0 | Apache-2.0 |
| RF-DETR tag 1.3.0 | `03aa832eb07126b8bbd0fff45790b1f2ad45be2b` | Python 3.12.3, rfdetr 1.3.0, torch 2.5.1, torchvision .20.1, transformers 4.51.3, peft .15.2 | Apache-2.0 |
| NanoDet | `be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0` | Python 3.9.19, torch 1.13.1, torchvision .14.1, Lightning 1.9.5, torchmetrics .11.4 | Apache-2.0 |

All use NumPy 1.26.4 and PNNX 20260526. NCNN 1.0.20260526, commit
`e54f7b1f88434e1d844ea0551b880a1cfb079ce1`, comes from WS4's read-only
builds. Full installed package versions, exact checkpoint hashes and repo
commits are recorded in the evidence JSON. Framework environments are isolated
under this worktree's `outputs/ws6/venvs/{dfine,rfdetr,nanodet}`; the shared
venv was only used for checks and the NumPy/ctypes benchmark.

Primary licence sources:
[D-FINE Apache licence](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/LICENSE),
[RF-DETR 1.3.0 Apache licence](https://github.com/roboflow/rf-detr/blob/03aa832eb07126b8bbd0fff45790b1f2ad45be2b/LICENSE),
[NanoDet Apache licence](https://github.com/RangiLyu/nanodet/blob/be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0/LICENSE).
Selected checkpoints use the projects' Apache release terms, compatible with
this project's AGPL-3.0 terms. D-FINE's
[model documentation](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/README.md)
warns separately about Objects365 checkpoints; only COCO-only N was used.
[Roboflow's licence statement](https://roboflow.com/licensing) explicitly
includes RF-DETR Nano under Apache-2.0. No XL/2XL or platform-licensed model
was used. Checkpoint URLs are the official D-FINE/storage dfinev1.0 release,
NanoDet v1.0.0-alpha-1 320 checkpoint, and
`https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth`.

Scratch footprint was approximately **measured 5 GB on Mac** including three
venvs, checkpoints, graphs and staged copies. Downloads remained below 10 GB.
No dataset download or training was needed.

## Bundled ARM execution

One job **945309**, `cz-ws6-gate`, normal / xcng0, nice 1000, 32 requested CPUs,
4 GB RAM, 30-minute limit. **Measured wall time 159 s**, COMPLETED, exit 0:0.
The queue was checked before submission: one other cz- CPU job was running,
so WS6 used the second permitted slot and had at most one running job.
The job stages under `/tmp/cz-945309`, cleans it with an EXIT trap and uses
`taskset -c 0-3` inside `srun --cpu-bind=cores`. C7 records confirm affinity
`0,1,2,3`. It includes fp32 parity before each size's fp16/fp32 benchmark.

All writes stay under `~/columba/scratch/ws6`; the WS4 build
`~/columba/scratch/ws4/ncnn-xcng` is read-only. Six C7s and three parity results
were copied back. Slurm script is
`training/adapters/nanodet/cluster_gate.sbatch`. No WS6 job remains running.

## Commands and artefacts

From the repository root, run export with the corresponding isolated Python:

```sh
outputs/ws6/venvs/nanodet/bin/python -m training.adapters.nanodet.export \
  --repository outputs/ws6/repos/nanodet \
  --checkpoint outputs/ws6/weights/nanodet-plus-m_320_checkpoint.ckpt \
  --output outputs/ws6/models/nanodet-384x640 --input-hw 384 640
```

D-FINE and RF-DETR expose the same options through
`training.adapters.dfine.export` / `training.adapters.rfdetr.export` with
their own repo, checkpoint and isolated environment. These are attempts,
not a promise of converter support; D-FINE's emitted graph still cannot load,
and RF-DETR's converter failure is propagated as a nonzero exit.

```sh
python -m training.adapters.verify_challenger --framework nanodet \
  --model outputs/ws6/models/nanodet-384x640 \
  --prefix /Users/vari/.t3/worktrees/columba-zero/t3-206c9967/native/build/ncnn-mac-a76compat \
  --input-hw 384 640 --output outputs/ws6/parity.json \
  outputs/ws6/nanodet-384x640-bus.npz outputs/ws6/nanodet-384x640-zidane.npz
python -m training.adapters.bench_challenger --framework nanodet \
  --model outputs/ws6/models/nanodet-384x640 \
  --prefix /Users/vari/.t3/worktrees/columba-zero/t3-206c9967/native/build/ncnn-mac-a76compat \
  --host-profile mac-a76compat --format fp16 --input-hw 384 640 \
  --frame outputs/ws6/bus.jpg
```

Benchmark exit 0 means hard **latency** PASS, 1 means hard latency FAIL, 2
means invalid input/runtime. Parity must be checked separately; it exits 0
only when every supplied reference passes. Frameworks are optional dependencies;
runtime NumPy adapters need neither torch nor transformers. The export wrappers
need the pinned environments above. `tests/test_challengers.py` has eight
synthetic, network-free tests and skips missing optional dependencies.

C7s are `experiments/latency/{vexcalibur,xcng0}-nanodet-plus-m-coco-ncnn-*.json`.
`ws6-challenger-gate-evidence.json` is an **evidence record, not a C7 timing**.
Scratch reference/export scripts, NPZs, model binaries and logs remain under
this worktree's `outputs/ws6/`; they are git-ignored and never deployment
artefacts. The actual framework reference generator is
`outputs/ws6/reference_nanodet.py`. It calls upstream Pipeline/get_bboxes;
`verify_challenger` reads those references without framework dependencies.

## Recommendation and limits

Keep YOLO26n/YOLO11n as the training priority. Drop both DETR challengers for
this NCNN route. NanoDet can remain a possible follow-up because the specified
**fp32 export correctness** gate passes and its **fp16 latency** fits 100 ms.
It is **not ready for training or promotion**: it misses 70 ms, has little
estimated hard-limit margin, and its timed fp16 final detections fail strict
parity. There is no unqualified survivor when correctness and latency must
hold in the same runtime precision. Do not infer an accuracy advantage from
these COCO smoke checks.

A later round could investigate fp16 storage with fp32 arithmetic, reduce the
NumPy GFL/NMS cost while preserving semantics, or document acceptable fp16
selection tolerances using representative validation images. That is follow-up
work, not a pass granted here. No changes to WS4 or contracts are requested.

Validation: 89 repository unit tests pass, Ruff lint and format checks pass on
`app tests tools cluster training`, strict mypy passes. These checks are in
addition to real NCNN parity, converter/load failures and the allocated ARM job.
