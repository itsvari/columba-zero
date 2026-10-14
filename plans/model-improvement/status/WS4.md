# WS4 — Latency benchmarking and reaction-time model

- State: done
- Branch: `ws4/latency`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14

## Done

All WS4 tasks 1–6 and its "Done when" are complete. PLAN itself is coordinator-owned; the recomputed replacement table is ready below.

1. Pinned ncnn tag `20260526`, commit `e54f7b1f88434e1d844ea0551b880a1cfb079ce1`, matches pip `ncnn==1.0.20260526`. Build profiles `mac-a76compat`, `xcng`, `pi5`; Vulkan off, fp16 and dot-product on, all declared post-A76 ARM options off. Generated macros and compiled specialised kernel objects are recorded in `experiments/latency/ncnn-{mac,xcng}-build-evidence.txt`. The script checks generated flags after building. Mac builds explicitly select ARM64 because CMake itself runs under Rosetta.
2. Python drives `tools/bench_ncnn/bridge.cpp`, linked against the custom NCNN static library, never pip NCNN. C7 separates pre/inference/post and paired end-to-end measurements. 4 threads, 20 warm-up, 200 timed iterations. Letterbox from BGR uint8; C8 sigmoid scores are already activated; class-aware NumPy NMS, original-frame box mapping and decision are included.
3. **Measured** 144 C7 records: Mac M2 Pro host `vexcalibur` and Neoverse-N1 host `xcng0`; COCO80 YOLO11n/YOLO26n plus one-class `pigeon_baseline_v1-2`; 352×608, 384×640, 448×800, 640×640; fp32/fp16/int8; camera 640×480 and 1536×864. Primary camera JSONs live directly under `experiments/latency/`; larger-camera JSONs under `highres/`. Every end-to-end p95 was checked against its 200 paired samples.
4. **Estimated** calibration `cal-v0-published`: xcng factor **1.527110713** = published Pi5 94.03 ms / xcng mean inference 61.573793715 ms, matched COCO YOLO11n NCNN FP32 640×640. Mac has its own factor **3.792166804** = 94.03 / 24.795850200. Source: https://github.com/ultralytics/ultralytics/blob/v8.3.200/docs/en/guides/raspberry-pi.md, accessed 2026-10-14. Table uses Ultralytics 8.3.152 and excludes pre/post. The current guide now publishes YOLO26, so the official versioned source is used. Published NCNN version, thread count and thermals are unknown. Scaling the whole pipeline and other precisions is approximate, not a Pi confidence interval. See `experiments/latency/calibration.json`.
5. **Simulated** reaction budgets use calibrated paired samples, 30 FPS, uniform sensor latency 1–2 frame periods, recall 0.8, latest-frame policy, 20,000 trials and seed 42. The CLI accepts sensor ranges, recall, k/n, camera FPS, correlation and latest/FIFO policies. It models skipped captures and never reuses a frame. Markov correlation decays over the captured-frame gap. Summary-only input uses a labelled lognormal approximation. `experiments/latency/reaction_results.json` contains 36 starting model/size/precision rows with independent 2/3, independent 3/5 and correlated 2/3 scenarios. `reaction_budget.md` contains the replacement FP16 table.
6. Gate command implemented and exercised against the real custom library. It prints estimated Pi p95, hard PASS/FAIL at 100 ms and target PASS/FAIL at 70 ms; exit 0 for hard PASS, 1 for hard FAIL, 2 for invalid input/calibration. Actual Mac gate runs demonstrate both outcomes. `--help` and `experiments/latency/README.md` document it.

Validation: all 59 unit tests pass, Ruff lint and format checks pass on `app tests tools cluster training`, strict mypy passes on 28 files. Unit tests need no network or NCNN build. Build script and bundled Slurm script pass `bash -n`; the final Mac build was exercised. No background WS4 processes or running Slurm jobs remain.

## In progress

Nothing.

## Blockers and requests

No execution blocker remains.

- Resolved: coordinator updated PLAN section 4 using `experiments/latency/reaction_budget.md`. At recall 0.8, FP16 estimated Pi frame p95 is **29.14–55.99 ms**, simulated independent 2/3 reaction p95 **199.82–317.46 ms**. The rho=.6 sensitivity p95 is **299.36–414.67 ms**. These are assumptions on estimated Pi distributions, not measured Pi reaction times.
- Resolved: CONTRACTS v1.2, included in v1.6, accepts the C7 extensions: optional paired end-to-end `samples_ms`, camera dimensions, class count, confidence, model hashes, load averages, software versions, raw CPU feature report, Slurm allocation provenance, timing boundaries and latency-only int8 calibration provenance.
- WS6: the gate supports C8 `(4+nc,N)` YOLO outputs. Other challenger layouts need a model-specific decoder before an end-to-end gate is valid.
- WS7: int8 here uses 32 COCO128 images with NCNN KL calibration, direct RGB resize and 1/255 normalisation. It is **latency-only calibration**, not deployment calibration or an accuracy/parity claim. FP16 is faster than the round-1 nano int8 models on both measured hosts. Round-2 s models have different trade-offs, recorded below.
- Round 1 omitted the optional YOLO26 one-to-one probe; round 2 below completes it. `--nms-free` supports class-pair top-k but must only be used on a separately verified one-to-one export.

## Results

- Full C7 summary table: `experiments/latency/summary.md`, all 144 measurements, inference p50/p95, end-to-end p95 and estimated Pi p95.
- Recomputed reaction table: `experiments/latency/reaction_budget.md`; full scenario values in `reaction_results.json`.
- **Estimated Pi**, one-class incumbent FP16 384×640, camera 640×480: frame p95 **32.83 ms**; **simulated** 2/3 reaction p50/p95 **123.46/202.98 ms**; rho=.6 p95 **302.37 ms**.
- Mac results are explicitly labelled **measured, Mac M2 Pro a76compat build, shared machine**, with load averages. They are noisy relative measurements and have a separate host-matched factor.

### APIs for WS6, WS7 and WS8

```sh
python tools/bench_ncnn.py gate --model DIR --input-hw H W \
  --host-profile xcng --prefix ~/columba/scratch/ws4/ncnn-xcng --format fp16
python tools/bench_ncnn.py gate --model DIR --input-hw H W \
  --host-profile mac-a76compat --format fp16
python tools/reaction_budget.py C7.json --calibration experiments/latency/calibration.json \
  --camera-fps 30 --recall 0.8 --k 2 --n 3 --correlation 0 --seed 42
```

The xcng command must run in an allocated arm-lane job. `--format int8` needs actual quantised param/bin weights. `--camera-hw 864 1536` selects the larger camera and should use a separate `--output` directory.

WS8 install prefixes:

- Mac: `/Users/vari/.t3/worktrees/columba-zero/t3-206c9967/native/build/ncnn-mac-a76compat`.
- xcng: `/home/a/ashelow/columba/scratch/ws4/ncnn-xcng`.

Each has `include/ncnn/`, `lib/libncnn.a`, `lib/cmake/ncnn/ncnnConfig.cmake`, conversion tools in `bin/` and `build-evidence.txt`; the benchmark bridge is `lib/libcolumba_bench.dylib` on Mac and `.so` on Linux. Use `CMAKE_PREFIX_PATH`, `find_package(ncnn CONFIG REQUIRED)` and link target `ncnn`. Mac consumers need the libomp discovery flags documented in README.

### Cluster execution and provenance

One successful bundled job **945162**, `cz-ws4-bench`, normal/xcng0, nice 1000, requested 32 CPUs and 16 GB, limit 2 h 45 min. **Measured wall time 9 min 29 s**, exit 0. `srun --cpu-bind=cores` established the allocation affinity; `taskset` selected cores **0–3**, verified in all 72 xcng C7s. Node-local `/tmp/cz-$SLURM_JOB_ID` staging had an EXIT cleanup trap. Installed prefix and ARM venv remain under `~/columba/scratch/ws4/` for reuse. No torch was installed on ARM. The final bundle script is `experiments/latency/cluster_job.sbatch`.

The queue was checked before every submission; WS4 never had more than one running CPU job. Three failed setup attempts collected no timings: 945127 had an incorrect absolute Slurm log path; 945144 installed the custom library but found x86 NumPy wheels from login-host venv creation; 945153 installed ARM wheels but encountered macOS AppleDouble metadata during model enumeration. Fixed the log path, installed the venv on ARM and enumerated directories only. The successful retry reused the verified library and venv. Logs remain in the WS4 scratch directory.

Kernel evidence needs interpretation: generic bf16 storage routines can be present despite disabled ARM84 bf16 instructions; runtime bf16 storage is off. NCNN feature queries return booleans on Mac but raw HWCAP values on Linux. The recorded raw expression `fp16_query + 2 * dotprod_query` is 3 on Mac and 2098176 on N1; both support fp16 and dot-product. Generated macros and specialised object names establish the A76 build restrictions.


## Round 2 — s-models and NMS-free head

- State: done. Fast-forwarded `ws4/latency` to latest main `92d3849`, which contains coordinator commit `98af7a1` and CONTRACTS v1.6.
- Reusing both custom NCNN builds and `cal-v0-published` factors.
- Planned COCO YOLO11s/YOLO26s, four sizes, fp16 and latency-only int8, camera 640×480, on Mac and xcng. YOLO26n one-to-one and matched one-to-many at 384×640 FP16 will run in the same bundle for postprocessing comparison.
- Optional one-class s checkpoint was not DONE at the start; skipped as authorised. Its training job is still running.
- Initial queue check: two cz- GPU jobs running, no cz- CPU jobs running, within project caps.


### Round-2 results

- **Measured** 36 additional C7s, 18 on shared Mac M2 Pro `vexcalibur` and 18 on Neoverse-N1 `xcng0`, camera 640×480. COCO80 YOLO11s/YOLO26s, four sizes, fp16 and latency-only int8; YOLO26n standard one-to-many and swapped one-to-one at 384×640 FP16.
- One **bundled** job `945604`, `cz-ws4-bench-s`, normal/xcng0, nice 1000, requested 32 CPUs and 16 GB, limit 2 h 45 min. Completed exit 0 in **4 min 2 s measured wall time**. Reused existing NCNN and ARM venv. `srun --cpu-bind=cores` checked that allocated affinity included 0–3, then every benchmark used literal `taskset -c 0-3`; all 18 C7s record `0,1,2,3`. Node-local staging had EXIT cleanup. No WS4 processes or CPU jobs remain running.
- Every submission was preceded by `squeue -u $USER`; two cz- GPU jobs and no cz- CPU job were running before this single submission. No retries were needed.
- Calibration factors are unchanged: xcng **1.527110713**, shared Mac **3.792166804**. These remain **estimated** Pi times, not Pi measurements. COCO80 exports include a larger output head than the planned 1/11-class deploy models.
- **Estimated Pi frame p95**, s models: 352×608 and 384×640 pass the 100 ms hard gate; 448×800 and 640×640 fail in both precisions. Only YOLO11s 352×608 int8 meets the 70 ms target, **67.40 ms**. YOLO26s 352×608 int8 is **70.35 ms**, just above target. FP16 352×608 is **72.15/72.46 ms** for YOLO11s/YOLO26s. FP16 384×640 is **81.83/83.54 ms**.
- **Simulated**, recall .8, 30 FPS, sensor uniform 1–2 periods, latest-frame, k2/n3, 20k trials/seed42: all s cases at 352×608 and 384×640 pass independent-hit reaction p95 <=500 ms. At assumed rho=.6, 352×608 stays below 500 ms but with little margin for some cases; all 384×640 cases exceed 500 ms, **528.45–535.15 ms**. Larger inputs fail the independent-hit reaction gate as well. These are sensitivity estimates, not measured temporal recall.
- **Measured observed postprocessing p50 difference**, many minus one, YOLO26n 384×640 FP16: **0.125622 ms on xcng0**, **0.027521 ms on shared Mac**. ARM post p50 is **0.439249 ms with NMS vs 0.313627 ms with one-to-one top-k**. This is a comparison of different head predictions on one photograph, not an isolated NMS microbenchmark or a universal saving. Export verification found swapped cv2/cv3 weights exactly equal to the original one-to-one weights, maximum absolute difference 0.
- Int8 uses the same 32-image COCO128 RGB/direct-resize/1/255 KL calibration method as round 1. It is latency-only and still needs WS7's deployment calibration, accuracy and parity checks.

Artifacts: appended `experiments/latency/summary.md` and `reaction_budget.md`; `reaction_round2_results.json`, `round2_head_comparison.json`, `round2_head_export_verification.json`, `round2_manifest.json`, `round2_job.sbatch`, and 36 new C7 files directly under `experiments/latency/`. Logs, checkpoints, exports and runnable bundle remain in `outputs/ws4/round2/` and cluster `~/columba/scratch/ws4/round2/`.

### Round-2 completion and coordinator actions

All requested measurements and reports are complete. Optional one-class s was not DONE at the start and was skipped as authorised. No implementation or ownership blockers remain. WS5 should use the ARM table for latency selection; 384×640 is within the frame hard gate but needs stronger measured temporal recall or a different processing/confirmation budget to fit the correlated-miss reaction scenario. PLAN is coordinator-owned; the new s rows are ready to incorporate if desired. The earlier C7 extension and nano PLAN requests are resolved.

Section-0 checks on rebased main plus this work: 192 unit tests pass, Ruff lint and formatting pass for `app tests tools cluster training`, strict mypy passes on 66 files. C7 matrix membership, all 200-sample p95 values, host-specific calibration scaling, model hashes across hosts, head decoder flags and ARM affinity were verified against the real records.
