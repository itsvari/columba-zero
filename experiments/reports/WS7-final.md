# WS7 final turn: blocked before final selection

**No final winner is selected.** Main `47a9b8b` is rebased into `ws7/export`. Contracts v1.9 and the coordinator's deploy-val seed clarification are implemented. Final ARM evidence is unavailable because BatchMode SSH failed.

## Cluster blocker

Species preparation failed while publishing its checkout, before upload/submission, with `Connection timed out during banner exchange` and `Connection to 127.0.0.1 port 22022 timed out`. The subsequent BatchMode check returned `ssh: connect to host 127.0.0.1 port 22022: Connection refused`. Cluster work stopped under the coordinator's explicit instruction. No password was requested or used.

- Distilled bundle `958128` was submitted on xcng0. The last successful observation showed seed0 and seed1 fp32/fp16 complete; all-seed/styles completion is unknown after access failed. It was not cancelled.
- Species has all three checkpoints, six exports and complete CPU PyTorch val/test/styles references locally. Its failed submission produced no job receipt. The taxonomy-view symlink boundary bug was fixed and regression-tested.
- Batch-16 control has not been fetched, exported or submitted.
- No GPU was used and no other worker's job was modified or cancelled. CPU project cap remained two.

## Seed policy and prepared evidence

The coordinator confirmed that deploy-size val takes precedence over WS5's square-val nominations. Apply one identical selector to every arm: highest val R@P90 among val FAR passes, then lower val confuser FAR, lower val empty FAR, higher val AP50, lower seed index. Test gates apply to that seed only. The three-seed arm improvement rule is unchanged. `WS7-final-policy.json` records the decision before inspecting deploy-size test results for switched seeds.

| Arm | WS5 square-val nomination | Deploy native fp16 pick |
|---|---|---|
| distill-800 | seed1, val R@P90 0.815 | pending ARM evidence |
| plain-800 | seed1, val R@P90 0.801 | existing seed1, 0.796150 |
| species-800 | seed2, val R@P90 0.822 | pending ARM evidence |
| control-b16 | seed2, val R@P90 0.808 | not evaluated |

The final nomination-versus-deploy table cannot be completed until native fp16 evidence is fetched. Do not substitute the following CPU previews for deployed-path gates. Accuracy is **measured** on Mac vexcalibur using 448x800 PyTorch. Every reference independently derives its threshold on val.

| Prepared arm | PT val R@P90 seed0 / seed1 / seed2 |
|---|---|
| yolo26n-v2s-800-round2-distill | 0.810149 / 0.805774 / 0.797025 |
| yolo26n-v2s-species-800-r3 | 0.808399 / 0.819773 / 0.813648 |

Checkpoint SHA-256s and complete val preview values are in `WS7-final-preparation.json`. C4 and C3 sidecars for these CPU references are under `WS7-final-references/`. Int8 and one-to-one remain excluded. Final fp32 exports are strict parity companions, not deployment candidates.

## Validated plain fallback, not a new final winner

The existing native fp16 plain-800 seed1 remains the validated fallback. Its previously measured gates below are unchanged. Its 3-seed test mean is 0.685552 +/- 0.007495, compared with rectangular native fp16 baseline 0.221907 +/- 0.013084. The operating threshold is 0.62646484375. Test R@P90 is 0.691218.

| Gate | Value | Limit | Result |
|---|---:|---:|---|
| Three-seed test improvement | 0.463645 | >0.013084 | pass |
| Val empty / confuser FAR, measured xcng0 | 0.000000 / 0.032110 | <=0.02 / <=0.05 | pass |
| Test empty / confuser FAR, measured xcng0 | 0.000000 / 0.041096 | <=0.02 / <=0.05 | pass |
| Estimated Pi frame p95 | 36.543667 ms | <=100 ms | pass |
| Simulated reaction p95 at 30 FPS | 345.363965 ms | <=500 ms | pass |
| Precision maximum loss in six metrics | 0.000005 | <=0.01 | pass |
| Export parity | strict same-graph fp32 + fp16 accuracy | both | pass |

C7 uses ACTIVE, four pinned xcng cores, camera-shaped 480x640 benchmark frames, 20 warm-ups and 200 samples. Pi timing is **estimated** with x1.527110713, not measured on Pi. Reaction is **simulated**, k2/n3, 20,000 trials, seed42, independent hits and still-image operating recall as the frame-recall proxy. Strict fp16 per-image parity remains diagnostic.

The unchanged camera default is 15 FPS. A separate **simulated** fallback sensitivity at 15 FPS gives p95 **582.276430 ms**, exceeding 500 ms. This does not change the registered 30 FPS selection gate; WS10 must reconcile the actual capture rate and reaction budget. See `WS7-fallback-reaction-15fps.json`.

## Durable package and verification

Fallback source: `runs/detect/pigeon_detector/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16/weights/best_ncnn_model`.
Runtime copy: `models/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`.

The manifest now uses `fallback_models` instead of the ignored-output provisional entry. It has no absolute path or `outputs/` source. The incumbent remains the default until final selection. New files follow the existing directly tracked model convention; no Git LFS filter is configured. The fp16 binary is 4,885,148 bytes. Existing baseline files are unchanged.

| File | SHA-256 |
|---|---|
| model.ncnn.param | `bd6201ead99a645b200d84ec76222c24475324f8be2f90d9de9e5b5513fba8f0` |
| model.ncnn.bin | `639e90dc36ad34d1235221f4aa907fae5a434245a90883454c1ca8dd4d7ffb6e` |
| metadata.yaml | `051f3facc7f1bd9385221496329c99acc7fe22a250fe3cec339b4e3452d9bd1f` |

Both default and named fallback preparation verify hashes and byte counts. Clean-clone tests cover both packages with autocrlf false and true. The fallback passed `check_model` against C8 and a real native 800x450 replay of 20 frames with one confirmed k2/n3 trigger. The replay used a scratch configuration; `app/config.yaml` is unchanged. This is functional verification, not deployment accuracy or Pi performance.

Validation passed with the newly built current Mac extension: pytest 322 tests plus 228 subtests, unittest 322 tests, CTest one test, Ruff check and format on `app tests tools cluster training native`, and mypy on all 91 source files. Evidence is `WS7-final-verification.json`; raw logs remain in ignored `outputs/ws7-final/`.

## Geometry and remaining work

Deploy-size scores decide promotion. WS5 square-input results are context. The existing plain-800 PyTorch diagnostic falls from test R@P90 0.801700 at 800x800 to 0.691218 at 448x800. Portrait images account for 38 of 39 lost full-test P90 matches. Test-v2 has only nine approximately 16:9 images, two positive, so it cannot establish real-camera performance. `WS7-aspect.md` gives the complete buckets; `columba-test-site-v1` in WS10 is the real check.

After BatchMode SSH is restored, fetch distill, submit the cached species bundle, add the control, and run final select over both evidence roots. The existing `--final` report generator emits every gate and the square-val/deploy-val comparison uniformly. Select and package the actual winner, then apply only the authorised config keys and repeat the native replay and checks. Exact commands are in `WS7-methods.md`.

WS10 steps 5–7 remain open: site threshold/k-n, Pi OMP policy and on-Pi geometry. The launcher is unchanged. No contract change is requested. Seed policy needs no further decision; cluster connectivity is the immediate blocker.
