# WS6 — Challenger architectures

- State: done — closed by coordinator after the latency gate (no challenger training)
- Branch: `ws6/challengers`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14

Cleanup note (17 Oct 2026): `training/adapters/` and `tests/test_challengers.py`
were removed from the repository; they are in git history at `6de04fb`.

## Done

Latency gate completed for COCO-pretrained D-FINE-N, RF-DETR Nano and
NanoDet-Plus-m. Read CONTRACTS v1.2 and reused WS4's pinned custom NCNN bridge
and calibration without editing WS4 files. All three frameworks used isolated
venvs under this worktree's `outputs/ws6/venvs/`; no shared dependency changes.

- D-FINE-N: TorchScript and ONNX attempts, rectangular caches corrected;
  unsupported TopK/Gather/F.linear graph cannot load. Drop.
- RF-DETR Nano: TorchScript and ONNX attempts, fixed positional interpolation
  allows ONNX export; PNNX leaves unsupported tile/TopK/GatherElements and
  warns about 5D operations. NCNN cannot load. Drop.
- NanoDet-Plus-m: export and **measured fp32 parity pass** on bus/zidane at
  352×608, 384×640 and 448×800 on Mac and xcng0. Maximum Mac box difference
  0.02863 px, score difference 0.00000124. Preprocessing matches upstream
  Pipeline exactly.
- NanoDet fp16 at 384×640 is a **latency-only survivor**, **estimated Pi p95
  94.68 ms**, from **measured xcng0 p95 62.00 ms**, calibration 1.527110713.
  Passes 100 ms, misses 70 ms. The timed fp16 path fails strict final-detection
  parity due to precision-sensitive threshold/NMS selection. Training held.
- NanoDet fp32 at 384×640 has **measured xcng0 p95 202.31 ms**, **estimated Pi
  308.96 ms**. Large retained outliers make this noisy; it fails the hard gate.
- All selected upstream code/checkpoint releases use Apache-2.0, compatible
  with the project's AGPL terms. Exact commits/versions/checkpoint hashes are
  in the evidence record.

One bundled ARM job **945309**, `cz-ws6-gate`, normal/xcng0, nice 1000,
32 requested CPUs, taskset 0–3, 4 GB RAM, 30-minute limit. **Measured 159 s**,
COMPLETED, exit 0:0. Queue caps were checked; no WS6 job or background process
remains. No push or merge.

Validation: 89 repository unit tests pass, including eight synthetic challenger
tests. Ruff lint/format checks pass on `app tests tools cluster training`;
strict mypy passes on 43 files. The NanoDet committed export was rerun and
passed real fp32 parity. D-FINE/RF-DETR committed attempts reproduce failures.
All 12 C7 p95 values were checked against their 200 paired samples.

## In progress

Nothing. Tasks 2–3, training recipes, runner integration, C3 predictions and
three-seed training were not started, as instructed.

## Blockers and requests

No requested changes to WS4, PLAN, contracts or dependencies.

No unqualified combined correctness/latency survivor is ready for training.
Before a later NanoDet round, resolve fp16 arithmetic/selection sensitivity
and rerun parity plus latency in the same precision. Its fp32 correctness and
fp16 latency passes must not be presented as a same-precision combined pass.
Keep YOLO26n/YOLO11n as the training priority. This is a follow-up decision,
not an outstanding step-1 export or benchmark task.

## Results

- Report: `experiments/reports/WS6-latency-gate.md`.
- 12 C7s: `experiments/latency/{vexcalibur,xcng0}-nanodet-plus-m-coco-ncnn-*.json`.
- Non-C7 evidence: `experiments/latency/ws6-challenger-gate-evidence.json`.
- NumPy preprocess/decode: `training/adapters/{dfine,rfdetr,nanodet}/__init__.py`.
- Export attempts: `python -m training.adapters.<framework>.export
  --repository REPO --checkpoint CHECKPOINT --output DIR --input-hw H W`,
  run with that framework's isolated Python.
- Benchmark: `python -m training.adapters.bench_challenger --framework nanodet
  --model DIR --prefix WS4_PREFIX --host-profile mac-a76compat|xcng
  --format fp16|fp32 --input-hw H W --frame IMAGE --output experiments/latency`.
  Exit 0/1 means hard latency PASS/FAIL; exit 2 means invalid input/runtime.
- Parity: `python -m training.adapters.verify_challenger --framework nanodet
  --model DIR --prefix WS4_PREFIX --input-hw H W --format fp32|fp16
  --output REPORT.json REFERENCE.npz ...`. Exit 0 requires all references pass;
  nonzero means failure. NPZ schema is documented in that module.
- Cluster bundle: `training/adapters/nanodet/cluster_gate.sbatch`.
- Tests: `tests/test_challengers.py`, no network, optional dependency skips.

Scratch models, images, references and logs remain in
`/Users/vari/.t3/worktrees/columba-zero/t3-7b538c13/outputs/ws6/` and cluster
`~/columba/scratch/ws6/`. Approximate **measured Mac footprint 5 GB**.
Pi estimates are based on YOLO calibration; challenger op mixes may differ.

## Coordinator decision (2026-10-14)

WS6 is closed under its time box ("stop after the latency gate if no
challenger survives"). D-FINE-N and RF-DETR Nano fail NCNN export.
NanoDet-Plus-m passes the 100 ms hard limit only in fp16, with ~5 ms estimated
headroom, fails the 70 ms target, fails fp16 parity, and is ~2.6x slower than
YOLO26n at 384x640. That is not a survivor worth 3-seed training: any accuracy
gain would have to come with a speed cost the YOLO main line doesn't pay. No
`chal-*` recipes will be trained. Revisit only if a later NCNN/PNNX release
supports TopK/Gather/Tile for the DETR models.
