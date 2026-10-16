# Validation status

## Shipped model — 16 October 2026

The deployed package is `yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`
(448×800, fp16, yolo11 decode, threshold 0.62646484375, native backend, k2/n3,
800×450 lores capture at 30 FPS). Selection evidence is in the
[WS7 report](../experiments/reports/WS7-final.md); highlights:

- Test R@P90 at deploy size 0.686 ± 0.008 (3-seed mean, **measured** on xcng
  with `columba_native`), against 0.222 ± 0.013 for the baseline reproduction.
  Shipped seed: test empty FAR 0, confuser FAR 3/73.
- Pi 5 frame p95 36.5 ms (**estimated** from xcng with the published
  calibration factor) and reaction p95 345 ms at 30 FPS, k2/n3 (**simulated**).
- fp16 passes the export gate: its fp32 sibling passes strict parity and fp16
  loses at most 0.01 of val/test R@P90, AP50 and operating recall. Int8 was
  studied and rejected ([WS7-quant](../experiments/reports/WS7-quant.md)).
- On Mac vexcalibur, `prepare_models.py` verified the package bytes and C8
  metadata, and a native replay processed 20 frames with one confirmed trigger.
  Unpaced replay proves loading and pipeline operation, not camera reaction time.

`pigeon_baseline_v1-2` stays in the manifest as the fallback and is verified by
the same command. Reproduce the automated checks from the repository root:

```bash
python -m pytest
python -m ruff check app tests tools cluster training native
python -m ruff format --check app tests tools cluster training native
python -m mypy
python tools/prepare_models.py
```

The sections below record the earlier validation of `pigeon_baseline_v1-2`.

## Baseline v1-2: completed on Windows desktop, 10 October 2026

- 25 automated unittest cases passed: configuration and old CLI rejection, model
  metadata/files/classes, inference filtering/warmup, sequential video completion,
  bounded live buffer and drops, capture errors/timeouts, GUI exit behavior,
  JSONL output, recording failure, startup/runtime cleanup, frame limit, and
  absence of GPIO imports, CLI overrides, SIGTERM handling/restoration, and
  cleanup failures.
- Runtime model copies match SHA-256 values in `model_manifest.json`.
- CPU PT/NCNN comparison: **63 test images, zero comparison failures** at confidence
  0.5, input 640, square preprocessing, NMS IoU 0.7. All counts matched and matched
  boxes met IoU >=0.8 and confidence difference <=0.05. See
  `export_comparison_summary.json`; raw predictions are in ignored
  `outputs/validation/export_comparison.json`.
- Authoritative `best.pt` evaluated on the existing **test** split: 63 images,
  183 labeled instances, zero background-only images. Precision **0.9049**,
  recall **0.8318**, mAP50 **0.9048**, mAP50–95 **0.5728**. These are test metrics,
  separate from the training log. Confidence floor 0.001 enables AP integration;
  precision/recall are curve-selected, not fixed-threshold 0.5 measurements.
  See `pt_test_metrics.json`. Split independence/session provenance was not audited.
- Real NCNN runtime processed a five-frame AVI made from held-out images in
  sequence, exited at EOF, logged all five frames with zero drops, and produced
  an annotated MJPEG AVI that decoded to five frames. GUI was disabled for this
  integration check; window behavior was tested with mocked OpenCV calls.

Environment: Windows 11, Intel Core i7-13700K, Python 3.12.10,
Ultralytics 8.4.175, NCNN 1.0.20260526, Torch 2.14.1+cu132 (CPU inference),
OpenCV 5.0.0.93, NumPy 2.5.3. Desktop timing was measured while two validation
processes ran concurrently; do not treat it as a performance benchmark. The
existing desktop venv contains both OpenCV pip variants; new installs must use
only one GUI-capable OpenCV provider.

The evaluator could not write its optional label cache outside the workspace;
validation completed successfully without it. Original labels were unchanged.

## Baseline v1-2: deployment fix and cleanup checks, 12 October 2026

The manifest originally hashed the Windows CRLF copy of NCNN `metadata.yaml`,
447 bytes with SHA-256
`16c589f246b1531c072a4c248629a694a6aa4f03d1f271a1957a9dcc38155f84`.
Git tracked LF bytes, so model preparation failed on the local LF checkout in
the same way as the reported clean Linux checkout. The canonical entry now
records 427 bytes with SHA-256
`180b7e93ae87e5fa4848da6a598c06001362c3b3c410c8bf9412cae8158bae65`.
`.gitattributes` enforces LF for this metadata and exact bytes for the other
four manifest files. Their checksums are unchanged.

- 33 automated unittest cases passed, including the original 25 runtime cases.
  Added checks cover tracked manifest bytes, clean checkouts with both
  `core.autocrlf=false` and `true`, source and copy corruption, export matching,
  streaming summaries, benchmark resource release, and final log failure.
- Model preparation passed and verified all five source and copied files.
  It now checks recorded sizes as well as SHA-256. A Linux execution was not
  repeated here because the local Docker daemon was unavailable.
- Ruff lint and formatting checks passed. Strict mypy passed on application
  code, tests, and tools. Ultralytics and Pi camera bindings are excluded from
  import analysis because their type information is incomplete; prediction
  protocols describe the detector boundary.
- Real NCNN replay processed a five-frame diagnostic AVI in order, logged zero
  drops, exited at EOF, and recorded an AVI that decoded to five frames. Counts
  were `0, 1, 0, 1, 0` on alternating blank frames and a pigeon tile from existing
  annotated validation imagery. GUI calls remain covered by mocks.
- Real CPU PT/NCNN comparison passed on three smoke images at confidence 0.5,
  input 640, square preprocessing, and NMS IoU 0.7. The positive tile had one
  detection in each model, matched IoU 0.99747, and confidence delta 0.000064.
  These derived images do not constitute independent accuracy validation or
  representative negative footage. The earlier 63-image test results remain
  the recorded dataset evidence and were not rerun.
- All eight supporting tools accepted `--help`. The video benchmark processed
  five frames, the summariser read the real replay log, and `pip check` passed.
  Training and export generation were not rerun, preserving the trained files.

Environment: macOS 27.0.1 ARM64, Python 3.14.8, Ultralytics 8.4.175,
NCNN 1.0.20260526, Torch 2.14.1, OpenCV 5.0.0, and NumPy 2.5.2.
Smoke outputs are in ignored `outputs/validation/style-cleanup/`, including
`positive-result.json`, `positive-comparison.json`, replay JSONL, and recording.
Desktop smoke timing is not a Pi performance result.

Reproduce the automated checks from the repository root with the desktop
development dependencies installed:

```bash
python -m unittest discover -s tests -v
python -m ruff check app tests tools
python -m ruff format --check app tests tools
python -m mypy
python tools/prepare_models.py
```

## Baseline v1-2: review fixes verified, 12 October 2026

- All 41 unittest cases passed. New regression cases reproduced the failures
  before the fixes: an incorrect post-training checkpoint path, an autofocus
  request on a camera without that control, order-dependent export matching,
  decoder failures reported as successful benchmarks, warm-up included in
  measured latency, and output cleanup failures missing from `run_end`.
- Training evaluation now reads the actual trainer checkpoint path, including
  incremented run names. The regression uses a simulated training run; training
  was not repeated and original weights were unchanged.
- Camera configuration tests cover advertised and absent autofocus support.
  These checks use mocks and do not establish physical camera compatibility.
- Export matching uses maximum-cardinality bipartite matching over same-class
  pairs meeting both configured tolerances. Regression checks cover detection
  permutations, reassignment of earlier matches, confidence constraints, and
  custom tolerances.
- Benchmarks warm up before timing and reopen the video to measure all decoded
  frames. Deterministic timing tests exclude a simulated 100-second warm-up.
  Real NCNN benchmarking processed the five-frame diagnostic AVI. A deliberately
  truncated copy exited nonzero at frame 2 of 5 and wrote no benchmark report.
- Recording and display cleanup failures now produce `cleanup_error` records,
  retain any previous failure, and leave summaries with `completed_cleanly: false`.
  Both resources are released once, and log closure is still attempted.
- Real NCNN replay processed and recorded five frames in order, with counts
  `0, 1, 0, 1, 0`, zero drops, and clean EOF completion. The recorded AVI decoded
  to five frames. Real CPU PT and NCNN predictions passed comparison on the
  three existing smoke images.
- Ruff lint and formatting checks and strict mypy passed. The desktop environment
  is the same as the earlier 12 October checks. New smoke outputs are in ignored
  `outputs/validation/review-fixes/`.

These smoke clips reuse existing derived imagery. They do not add independent
accuracy evidence or establish Pi performance. Reproduce automated checks with
the commands above; hardware and field-footage acceptance remains pending below.

## Pending on physical Pi 5 (WS10)

Pi OS installation, the NCNN and native extension builds, and ARM dependency
compatibility; measured latency calibration; 800×450 lores capture geometry,
CSI colours and box alignment; monitor interaction, camera restart/shutdown and
recording overhead; one-hour memory/thermal/throttling behaviour; reaction time
for both cameras by day and night; the OpenMP wait policy; and site footage
(`columba-test-site-v1`), the site fine-tune and the final threshold and k/n.
No Raspberry Pi or camera is connected to this workspace, so no Pi performance
or hardware acceptance is claimed. Follow the
[Pi acceptance checklist](../RASPBERRY_PI_PROTOTYPE_PLAN.md) and append actual
measurements here.
