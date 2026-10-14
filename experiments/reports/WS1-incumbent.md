# Incumbent re-score and contamination analysis

All metrics and counts are **measured on Mac M2 Pro**, host `vexcalibur`, on
2026-10-14. Exposure assignments reconstruct the supplied notebook and use
conservative grouping; they are not recovered logs of the actual Windows
`pigeon_baseline_v1-2` training run. No Pi accuracy or latency is claimed.

## Recommendation for the section-1 comparison baseline

The deployed **NCNN 640×640 unseen recall_at_p90 is 0.0000**. Record that as the
provisional incumbent number, with the seven-box denominator and contaminated
threshold calibration attached. The full-test number **0.5921 is contaminated**
and should be a historical diagnostic, not the accuracy promotion baseline.

**Do not use either original-incumbent number to claim that the section-1
three-seed promotion test has passed.** There are only three independent unseen
pigeon-positive images, and no unseen validation images on which to calibrate
this incumbent. The original incumbent also has only one training seed. Replace
the promotion reference with WS5's `baseline-repro` three-seed result, trained
on `columba-train-v1clean` and evaluated on the full frozen `columba-test-v2`,
when it is available. That dataset excludes every test family from the new
training recipe. A future test version also needs fresh pigeon-positive scenes
and a validation set containing empty/confuser scenes.

The false-alarm result is already actionable: all 125 empty test images are
unseen, and deployed NCNN alarms on 14 of them at the validation threshold.
It also alarms on 34 of 59 unseen confuser images. The incumbent fails the
section-1 empty/confuser gates of 2% / 5% by a large margin.

## Inputs and scoring policy

- Branch fast-forwarded to main `a15ed4ce4d51ff94282b21e7769fffd6c8a563d1` before work.
- Frozen test: `~/columba-data/columba-test-v2`, 319 images, 353 pigeon boxes,
  125 empty images and 73 confuser images. The committed freeze file passed
  `python -m tools.build_dataset verify-freeze` before prediction and on reruns.
- Validation: `~/columba-data/columba-train-v1clean`, val, 156 images, 280 pigeon
  boxes, **zero empty and zero confuser images**.
- PyTorch checkpoint: `/Users/vari/Developer/columba-zero/runs/detect/pigeon_detector/pigeon_baseline_v1-2/weights/best.pt`.
- Deployed NCNN package: `/Users/vari/Developer/columba-zero/models/pigeon_baseline_v1-2/best_ncnn_model/`.
  Its wrapper directory is accepted by the predictor. The package and original
  checkpoint were read only. NCNN used its **runtime-default fp16 compute flags**,
  not the `--ncnn-fp32` parity option. Package weights are fp32.
- Predictions use confidence floor 0.001, class-aware NMS IoU 0.7, max_det=300,
  CPU, batch=1, four inference threads. Sizes below are **height×width**.
- A separate operating threshold was selected on the complete v1clean val
  split for each configuration. Both full and unseen test metrics reuse that
  same threshold. No test threshold was selected; test `recall_at_p90` remains
  the diagnostic PR-curve statistic defined by C4.
- AP uses C4's pycocotools method. Small boxes use each configuration's input
  size, so the full test has 178 small pigeon boxes at 640×640 and 212 at 384×640.

## Validation results and threshold contamination

| Configuration | Selected threshold | AP50 | AP50–95 | Recall at P90 | Small recall at P90 |
|---|---:|---:|---:|---:|---:|
| PyTorch 640×640 | 0.5794193745 | 0.9422 | 0.7192 | 0.8964 | 0.7647 |
| Deployed NCNN 640×640 | 0.5761718750 | 0.9419 | 0.7164 | 0.9000 | 0.7647 |
| PyTorch 384×640 | 0.4830672145 | 0.9470 | 0.7272 | 0.9107 | 0.7143 |

All 156 validation images are conservatively `seen_train` through an original
stem, capture scene, exact copy or mirror-aware phash family. Direct raw-file
membership alone gives 113 train, 36 validation and 7 old-test images. The
conservative group rule promotes every one of those families to train exposure.
There are no unseen validation images. Each `metrics_val_unseen.json` explicitly
records `status: not_evaluable` and a zero-image denominator.

These thresholds follow the requested validation-only protocol, but their
precision calibration is optimistic because this incumbent has already seen
those image families and the split has no negative-image examples.

## Test results

### Full frozen test, contaminated positives

| Configuration | AP50 | AP50–95 | Recall at P90 | Small recall at P90 | Precision at threshold | Recall at threshold | Empty FAR | Confuser FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch 640×640 | 0.8500 | 0.5825 | 0.5921 | 0.3820 | 0.8021 | 0.8499 | 0.1120 | 0.5616 |
| Deployed NCNN 640×640 | 0.8507 | 0.5809 | 0.5921 | 0.3820 | 0.7995 | 0.8584 | 0.1120 | 0.5616 |
| PyTorch 384×640 | 0.7434 | 0.4989 | 0.3399 | 0.0896 | 0.7177 | 0.7705 | 0.2000 | 0.6164 |

### Unseen test view

This view has 187 images: three pigeon-positive images with seven boxes,
125 empty images and 59 confuser images. It contains no small pigeon boxes at
640×640 and only one at 384×640. The C4 small-recall value 0.0 with no small ground
truths is a defined empty-denominator value, not evidence of missed small birds.

| Configuration | AP50 | AP50–95 | Recall at P90 | Small recall at P90 | Precision at threshold | Recall at threshold | Empty FAR | Confuser FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch 640×640 | 0.3143 | 0.2634 | 0.0000 | 0.0000 | 0.0968 | 0.8571 | 0.1120 | 0.5763 |
| Deployed NCNN 640×640 | 0.3151 | 0.2640 | 0.0000 | 0.0000 | 0.0938 | 0.8571 | 0.1120 | 0.5763 |
| PyTorch 384×640 | 0.3354 | 0.2436 | 0.1429 | 0.0000 | 0.0741 | 0.8571 | 0.2000 | 0.6271 |

All three configurations recover six of the seven pigeon boxes at their
operating threshold. Low precision comes from false detections on negatives,
not complete failure to detect the unseen pigeons. A one-box change is a
14.3-percentage-point change in recall on this view, so ranking sizes by these
unseen recall numbers would be unreliable.

## Contamination table

The primary assignment is conservative: any matching train member makes a
family `seen_train`; otherwise a validation member makes it `seen_val`;
otherwise it is `unseen`. The published manifest's `group` is retained in the
per-image evidence. `analysis_group` applies WS2's grouping to the expanded
historical source pool, so its IDs can differ from the selected-only test build.

| Test source | seen_train | seen_val | unseen | Total |
|---|---:|---:|---:|---:|
| roboflow-yolo1199 | 104 | 12 | 3 | 119 |
| coco-birds | 13 | 3 | 1 | 17 |
| coco-negatives | 0 | 0 | 125 | 125 |
| inaturalist | 0 | 0 | 58 | 58 |
| **Total** | **117** | **15** | **187** | **319** |
| Pigeon-positive images only | 106 | 12 | 3 | 121 |

| v1clean val source | seen_train | seen_val | unseen | Total |
|---|---:|---:|---:|---:|
| roboflow-nea | 28 | 0 | 0 | 28 |
| roboflow-yolo1199 | 125 | 0 | 0 | 125 |
| coco-birds | 3 | 0 | 0 | 3 |
| **Total** | **156** | **0** | **0** | **156** |

For comparison, direct raw-file membership without family propagation is:

| Split | Direct train | Direct validation | Direct old-test / absent |
|---|---:|---:|---:|
| test-v2 | 103 | 26 | 190 |
| v1clean val | 113 | 36 | 7 |

118 of 121 test pigeon-positive images are exposed at the family level.
The three remaining positives are original Roboflow **test** images, not
necessarily images never examined during the incumbent's past development:

- `roboflow-yolo1199-080_jpg.rf.f59bc0d9dd9c7013eb7452a366400918`
- `roboflow-yolo1199-Columba-livia257_jpg.rf.6a1f4991b026a3efe3b997463ed41331`
- `roboflow-yolo1199-Columba-livia45_jpg.rf.9da8cf8194f978e1b4fe39f5f062f0a3`

### Reconstruction details and limits

1. Notebook cells 12 and 17 preserve each Roboflow export's source split,
   remapping only pigeons. NEA references require a pigeon box, matching the
   notebook's 126/40/24 retained images. The notebook records all 895/241/63
   yolo1199 images as retained; all 1199 are therefore historical references.
   One current yolo1199 train label is empty, unlike that notebook count; keeping
   it as a reference is conservative and is documented in the driver.
2. COCO uses `COCO.getImgIds(catIds=[bird])`, `loadImgs` order, then
   `random.Random(42).shuffle`, with boundaries `int(n*0.8)` and `int(n*0.9)`.
   All **125** bird IDs are split before the WS2 licence filter. The result is
   100 train, 12 validation and 13 test images. Annotation SHA-256, the exact
   current `getImgIds` order and every reconstructed split are committed.
   `getImgIds` exposes set-derived ordering; the historical order was not saved
   by the notebook, so COCO membership is a reconstruction, not independently
   verified historical membership.
3. WS2's raw hashes and `scene_id` encode original stems before `_jpg.rf.` and
   `_jpeg.rf.`, capture-session rules, SHA-256 identity and mirror-aware phash
   links at Hamming distance <= 10. Connected components propagate exposure,
   including cross-source duplicates. Generic original stems can join different
   photos and transitive phash groups can grow; these assignments deliberately
   err towards excluding potentially seen positives.
4. The iNaturalist Javan myna `inat-404123303-742092805` initially linked at phash
   distance 10 to two noise-augmented NEA pigeon images. Visual inspection
   showed different birds, photos and backgrounds. `exposure_review.json`
   records the rejection for that entire three-copy NEA family; the driver
   refuses to apply it if the candidate group later changes. This myna is
   included among the 59 unseen confuser images.
5. Only the 38 licence-allowed COCO bird images have local image hashes. Direct
   COCO membership uses all 125 IDs, but unknown visual copies of the other 87
   original COCO images cannot be detected from the available raw images.
   Pretraining/checkpoint-selection history of the initial COCO backbone is
   outside this v1 fine-tuning audit. "Unseen" is relative to the reconstructed
   fine-tuning corpus and available duplicate evidence.
6. The incumbent's Windows data directory is absent. Its saved `args.yaml`
   points at `C:/Users/reksh/Documents/datasets/pigeons-source/data.yaml`,
   whereas the notebook demonstrates the source merge in Colab. This analysis
   follows the coordinator's stated v1-merge lineage; actual v1-2 file membership
   cannot be proven from the checkpoint alone.
7. WS2 notes that 196 of 212 rectangular small boxes come from one depot
   capture session. That session is exposed to the incumbent and absent from
   the unseen view, so the full small-recall statistic is also contaminated
   and dominated by one scene.

## Artifacts and reproduction

- [Leaderboard](../leaderboard.csv): three `incumbent-contaminated` full rows and
  three `incumbent-unseen-subset` rows, written through WS3's
  `leaderboard_row` / `update_leaderboard`. Input size and scope are in recipe IDs;
  `train_imgsz` remains the original training size 640.
- `experiments/incumbent/{pt-640x640,ncnn-640x640,pt-384x640}/`: full/unseen val/test
  metric JSONs, run records, C3 metadata and input fingerprints. No weights or
  prediction lists are committed. Each unseen leaderboard row shares the
  corresponding configuration's original validation threshold and val score.
- [Contamination summary](../incumbent/contamination.json),
  [test evidence](../incumbent/exposure_test.jsonl),
  [validation evidence](../incumbent/exposure_val.jsonl),
  [COCO splits](../incumbent/coco_original_splits.json),
  [reviewed rejection](../incumbent/exposure_review.json), and
  [unseen test IDs](../incumbent/unseen_test.txt).
- C3 predictions and logs are under this worktree's `outputs/ws1/incumbent/`.
  `rescore.log` is the final reproducible run; `score_full.log` records the initial
  full inference run. All jobs are bounded local CPU jobs; no cluster job was used.

```sh
python -m experiments.incumbent.rescore --data-root ~/columba-data
```

The driver verifies the freeze, reconstructs exposure, predicts all three
configurations, evaluates full/unseen populations and upserts the C6 leaderboard.
Scratch predictions are reused only with matching model, dataset and inference
fingerprints. Rerunning recomputes metrics and exposure; changed inputs force
fresh prediction. The calibration thresholds always come from the full val split.

The evaluator also exposes the subset operation directly:

```sh
python tools/evaluate_predictions.py --dataset ~/columba-data/columba-test-v2 \
  --split test --predictions outputs/ws1/incumbent/ncnn-640x640/predictions_test.json \
  --subset experiments/incumbent/unseen_test.txt \
  --threshold-from experiments/incumbent/ncnn-640x640/metrics_val.json \
  --out outputs/ws1/incumbent/ncnn-640x640/metrics_test_unseen_cli.json
```

Unknown subset IDs and empty subsets fail explicitly. All predictions are first
validated against the full dataset split, then restricted to selected IDs.

Verification on Mac M2 Pro: all 121 unit tests pass in 2.453 s. Ruff lint/format
and strict mypy pass, including the new driver. The subset CLI/API results
match the committed metrics. The C6 header, six unique rows and portable
run paths were read back and checked.
