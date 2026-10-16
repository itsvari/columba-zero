# WS7 final export decision

> Cleanup note (17 Oct 2026): the per-candidate evidence behind this report
> (`WS7-candidates/`, `WS7-final-candidates/`, `WS7-final-references/`) was
> removed in the repository cleanup. It is available in git history at
> `6de04fb`. `WS7-final-selection.json` is kept.

Selection: `yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800]`.
All training candidates are available; deploy-size evidence decides the final pick.
Coverage: 24 configurations and 72 seed/export cases.
Comparison baseline: `baseline-repro-p50`, measured native fp16 384x640 test R@P90 0.2219 +/- 0.0131, independently val-thresholded per seed.
Styles is a secondary measured column. Test-v2 still supplies every promotion
gate. Paired v2/v2s arms within 0.01 validation recall and 5% measured latency
use styles recall to break the tie. The 5% band treats host timing noise
as equal latency.

Primary accuracy is measured using columba_native package-default compute on xcng0/xcng1.
Native styles accuracy and ACTIVE latency use xcng0/xcng1; C3/C7 record each host.
Val/test CPU PyTorch references use xcng0 for the first seed's two heads, then Mac
vexcalibur. Mac versus xcng0 passes all 1,408 first-reference image comparisons.
All PyTorch styles references use Mac vexcalibur CPU.
Every export independently reproduces parity against its own saved CPU reference.
Styles scores are measured on WS2's synthetic camera-style copies.
These rectangular deploy-size measurements decide the section-1 and false-alarm
gates. WS5 square-input leaderboard scores are context, not deployment scores.
C7 latency uses OMP_WAIT_POLICY=ACTIVE; PASSIVE results remain diagnostics.
WS10 will decide the deployed wait policy on the Pi, including pipeline contention.
The launcher is unchanged.
Pi 5 latency is estimated with WS4 cal-v0-published, xcng times x 1.527110713.
Reaction is simulated by WS4 with 30 FPS, k2/n3, 20,000 trials, seed 42,
independent hits and per-frame recall proxied by test recall at each export's
fresh validation-selected threshold. Temporal correlation is unmeasured.

Arms rank by three-seed mean validation R@P90; test is a promotion gate.
Choose the highest-val-recall seed among val FAR passes, then apply test,
latency, reaction and export gates to that seed alone. No test-driven retry.
Every seed's FAR remains visible as robustness context.
FP16 and int8 each permit at most 0.01 loss in val/test R@P90, AP50 and
operating recall versus same-head rectangular PyTorch. This is an explicit
confirmed accuracy policy. FP16 parity requires strict fp32 parity of the same
graph plus that accuracy gate. Strict fp16 per-image parity stays diagnostic.
Strict comparison uses count equality, IoU >=0.8 / score delta <=0.05;
int8 diagnostics use IoU >=0.7 / score delta <=0.10, at confidence 0.25.
Coordinator exclusions: int8 and one-to-one heads are not deployment candidates.

Calibration uses 200 deterministic val images, centred deployment letterbox,
RGB /255 and padding 114. Selected SHA-256s are checked against frozen test.
No calibration image or operating threshold comes from test.

## Arm results

Accuracy values below are measured on xcng; frame p95 is estimated Pi 5;
reaction p95 is simulated. Timing uses the val-chosen seed, or worst seed
for diagnostic configurations with no val-eligible seed. Input sizes are H x W.

| Arm / decode / precision / input | Val R@P90 | Test R@P90 mean +/- sd | Test AP50 | Test operating R | PT styles R@P90 | Export styles R@P90 | Pi frame p95 ms | Reaction p95 ms | Shipped seed | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| yolo26n-v2-800-round2-plain/yolo11/fp16/[448, 800] | 0.7795 | 0.6648 +/- 0.0201 | 0.7271 | 0.6242 | 0.3573 | 0.3582 | 36.70 | 346.84 | 2 | no |
| yolo26n-v2-800-round2-plain/yolo11/fp32/[448, 800] | 0.7822 | 0.6638 +/- 0.0156 | 0.7243 | 0.6280 | 0.3573 | 0.3573 | 69.26 | 612.67 | 2 | no |
| yolo26n-v2-800-round2-plain/yolo11/int8/[448, 800] | 0.5089 | 0.2984 +/- 0.0114 | 0.3793 | 0.2871 | 0.3573 | 0.1999 | 33.71 | 1051.62 | 0 | no |
| yolo26n-v2-800-round2-plain/yolo26/fp16/[448, 800] | 0.7594 | 0.6364 +/- 0.0173 | 0.7212 | 0.5987 | 0.3321 | 0.3340 | 36.61 | 353.33 | 2 | no |
| yolo26n-v2-800-round2-plain/yolo26/fp32/[448, 800] | 0.7597 | 0.6421 +/- 0.0216 | 0.7242 | 0.6015 | 0.3321 | 0.3321 | 69.51 | 635.40 | 2 | no |
| yolo26n-v2-800-round2-plain/yolo26/int8/[448, 800] | 0.4721 | 0.2890 +/- 0.0057 | 0.3644 | 0.2833 | 0.3321 | 0.1731 | 33.70 | 1080.57 | 1 | no |
| yolo26n-v2s-640-round2-plain/yolo11/fp16/[384, 640] | 0.7728 | 0.6534 +/- 0.0114 | 0.7224 | 0.6280 | 0.5002 | 0.4976 | 25.16 | 289.76 | 1 | no |
| yolo26n-v2s-640-round2-plain/yolo11/fp32/[384, 640] | 0.7740 | 0.6563 +/- 0.0107 | 0.7241 | 0.6289 | 0.5002 | 0.5002 | 46.75 | 417.75 | 1 | no |
| yolo26n-v2s-640-round2-plain/yolo11/int8/[384, 640] | 0.7337 | 0.6412 +/- 0.0189 | 0.7087 | 0.6072 | 0.5002 | 0.4825 | 23.46 | 293.59 | 1 | no |
| yolo26n-v2s-640-round2-plain/yolo26/fp16/[384, 640] | 0.7521 | 0.6251 +/- 0.0114 | 0.7118 | 0.5873 | 0.4766 | 0.4712 | 25.14 | 321.11 | 1 | no |
| yolo26n-v2s-640-round2-plain/yolo26/fp32/[384, 640] | 0.7489 | 0.6298 +/- 0.0189 | 0.7126 | 0.5911 | 0.4766 | 0.4766 | 46.44 | 465.12 | 0 | no |
| yolo26n-v2s-640-round2-plain/yolo26/int8/[384, 640] | 0.6976 | 0.6043 +/- 0.0284 | 0.6893 | 0.5430 | 0.4766 | 0.4492 | 23.45 | 379.96 | 0 | no |
| yolo26n-v2s-800-r3-control-b16/yolo11/fp16/[448, 800] | 0.7868 | 0.6903 +/- 0.0114 | 0.7427 | 0.6601 | 0.5530 | 0.5530 | 36.56 | 303.38 | 0 | no |
| yolo26n-v2s-800-r3-control-b16/yolo11/fp32/[448, 800] | 0.7865 | 0.6969 +/- 0.0049 | 0.7479 | 0.6667 | 0.5530 | 0.5530 | 68.70 | 521.03 | 0 | no |
| yolo26n-v2s-800-round2-distill/yolo11/fp16/[448, 800] | 0.8049 | 0.6837 +/- 0.0033 | 0.7600 | 0.6582 | 0.5282 | 0.5297 | 36.46 | 307.85 | 0 | no |
| yolo26n-v2s-800-round2-distill/yolo11/fp32/[448, 800] | 0.8043 | 0.6874 +/- 0.0118 | 0.7602 | 0.6572 | 0.5282 | 0.5282 | 69.00 | 538.36 | 0 | no |
| yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800] | 0.7859 | 0.6856 +/- 0.0075 | 0.7416 | 0.6298 | 0.5187 | 0.5219 | 36.54 | 345.36 | 1 | yes |
| yolo26n-v2s-800-round2-plain/yolo11/fp32/[448, 800] | 0.7822 | 0.6884 +/- 0.0075 | 0.7446 | 0.6280 | 0.5187 | 0.5187 | 68.62 | 609.29 | 1 | no |
| yolo26n-v2s-800-round2-plain/yolo11/int8/[448, 800] | 0.5410 | 0.3258 +/- 0.0049 | 0.4137 | 0.3277 | 0.5187 | 0.3245 | 33.39 | 809.97 | 0 | no |
| yolo26n-v2s-800-round2-plain/yolo26/fp16/[448, 800] | 0.7562 | 0.6563 +/- 0.0215 | 0.7407 | 0.5996 | 0.4907 | 0.4939 | 36.53 | 360.40 | 1 | no |
| yolo26n-v2s-800-round2-plain/yolo26/fp32/[448, 800] | 0.7565 | 0.6563 +/- 0.0215 | 0.7384 | 0.6015 | 0.4907 | 0.4907 | 69.50 | 650.07 | 1 | no |
| yolo26n-v2s-800-round2-plain/yolo26/int8/[448, 800] | 0.5039 | 0.3144 +/- 0.0057 | 0.4002 | 0.3154 | 0.4907 | 0.3006 | 33.55 | 882.22 | 0 | no |
| yolo26n-v2s-species-800-r3/yolo11/fp16/[448, 800] | 0.8145 | 0.6856 +/- 0.0098 | 0.7559 | 0.6383 | 0.5137 | 0.5146 | 36.89 | 342.74 | 2 | no |
| yolo26n-v2s-species-800-r3/yolo11/fp32/[448, 800] | 0.8139 | 0.6903 +/- 0.0128 | 0.7579 | 0.6374 | 0.5137 | 0.5137 | 70.02 | 612.64 | 2 | no |

## Precision comparisons

Mean deltas are export minus same-head rectangular CPU PyTorch.
Accuracy and strict per-image diagnostic parity count passing seeds independently.

| Arm / decode / precision | Test R@P90 delta | Test AP50 delta | Accuracy gate | Strict parity diagnostic |
|---|---:|---:|---:|---:|
| yolo26n-v2-800-round2-plain/yolo11/fp16/[448, 800] | +0.0009 | +0.0028 | 2/3 | 0/3 |
| yolo26n-v2-800-round2-plain/yolo11/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2-800-round2-plain/yolo11/int8/[448, 800] | -0.3654 | -0.3450 | 0/3 | 0/3 |
| yolo26n-v2-800-round2-plain/yolo26/fp16/[448, 800] | -0.0057 | -0.0030 | 2/3 | 0/3 |
| yolo26n-v2-800-round2-plain/yolo26/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2-800-round2-plain/yolo26/int8/[448, 800] | -0.3532 | -0.3598 | 0/3 | 0/3 |
| yolo26n-v2s-640-round2-plain/yolo11/fp16/[384, 640] | -0.0028 | -0.0018 | 3/3 | 0/3 |
| yolo26n-v2s-640-round2-plain/yolo11/fp32/[384, 640] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-640-round2-plain/yolo11/int8/[384, 640] | -0.0151 | -0.0154 | 0/3 | 0/3 |
| yolo26n-v2s-640-round2-plain/yolo26/fp16/[384, 640] | -0.0047 | -0.0008 | 2/3 | 0/3 |
| yolo26n-v2s-640-round2-plain/yolo26/fp32/[384, 640] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-640-round2-plain/yolo26/int8/[384, 640] | -0.0255 | -0.0234 | 0/3 | 0/3 |
| yolo26n-v2s-800-r3-control-b16/yolo11/fp16/[448, 800] | -0.0066 | -0.0052 | 2/3 | 0/3 |
| yolo26n-v2s-800-r3-control-b16/yolo11/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-800-round2-distill/yolo11/fp16/[448, 800] | -0.0038 | -0.0002 | 2/3 | 0/3 |
| yolo26n-v2s-800-round2-distill/yolo11/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800] | -0.0028 | -0.0029 | 3/3 | 0/3 |
| yolo26n-v2s-800-round2-plain/yolo11/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-800-round2-plain/yolo11/int8/[448, 800] | -0.3626 | -0.3308 | 0/3 | 0/3 |
| yolo26n-v2s-800-round2-plain/yolo26/fp16/[448, 800] | +0.0000 | +0.0023 | 3/3 | 0/3 |
| yolo26n-v2s-800-round2-plain/yolo26/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-800-round2-plain/yolo26/int8/[448, 800] | -0.3418 | -0.3382 | 0/3 | 0/3 |
| yolo26n-v2s-species-800-r3/yolo11/fp16/[448, 800] | -0.0047 | -0.0020 | 2/3 | 0/3 |
| yolo26n-v2s-species-800-r3/yolo11/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |

## Head-swap comparison

Deltas are one-to-one minus standard NMS, matched by recipe, precision
and deploy size. Latency compares val-chosen-seed estimated Pi frame p95.

| Arm / precision | Val R@P90 delta | Test R@P90 delta | Styles R@P90 delta | Frame p95 delta ms |
|---|---:|---:|---:|---:|
| yolo26n-v2-800-round2-plain / fp16 | -0.0201 | -0.0283 | -0.0242 | -0.0920 |
| yolo26n-v2-800-round2-plain / fp32 | -0.0225 | -0.0217 | -0.0252 | +0.2440 |
| yolo26n-v2-800-round2-plain / int8 | -0.0367 | -0.0094 | -0.0268 | -0.0115 |
| yolo26n-v2s-640-round2-plain / fp16 | -0.0207 | -0.0283 | -0.0264 | -0.0233 |
| yolo26n-v2s-640-round2-plain / fp32 | -0.0251 | -0.0264 | -0.0236 | -0.3177 |
| yolo26n-v2s-640-round2-plain / int8 | -0.0362 | -0.0368 | -0.0334 | -0.0082 |
| yolo26n-v2s-800-round2-plain / fp16 | -0.0297 | -0.0293 | -0.0280 | -0.0123 |
| yolo26n-v2s-800-round2-plain / fp32 | -0.0257 | -0.0321 | -0.0280 | +0.8756 |
| yolo26n-v2s-800-round2-plain / int8 | -0.0370 | -0.0113 | -0.0239 | +0.1582 |

## Individual exports

Deltas are export minus same-head rectangular PyTorch, measured on xcng.
Each export reselects its threshold on val. No square-training metric
is substituted for rectangular inference accuracy.

| Candidate | Val R@P90 | Test R@P90 | Test R@P90 delta | Test AP50 delta | Val empty / confuser FAR | Test empty / confuser FAR | Val threshold | Strict parity failed images |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo26n-v2-800-round2-plain/seed0/yolo11/fp16 | 0.7760 | 0.6686 | -0.0028 | -0.0003 | 0.0 / 0.03669724770642202 | 0.0 / 0.0410958904109589 | 0.622559 | 5 |
| yolo26n-v2-800-round2-plain/seed1/yolo11/fp16 | 0.7752 | 0.6827 | +0.0085 | +0.0077 | 0.005249343832020997 / 0.03669724770642202 | 0.008 / 0.0547945205479452 | 0.612305 | 3 |
| yolo26n-v2-800-round2-plain/seed2/yolo11/fp16 | 0.7874 | 0.6431 | -0.0028 | +0.0010 | 0.0 / 0.022935779816513763 | 0.016 / 0.0821917808219178 | 0.518555 | 2 |
| yolo26n-v2-800-round2-plain/seed0/yolo11/fp32 | 0.7795 | 0.6714 | +0.0000 | +0.0000 | 0.0 / 0.03669724770642202 | 0.0 / 0.0410958904109589 | 0.617862 | 0 |
| yolo26n-v2-800-round2-plain/seed1/yolo11/fp32 | 0.7769 | 0.6742 | +0.0000 | +0.0000 | 0.005249343832020997 / 0.03669724770642202 | 0.008 / 0.0547945205479452 | 0.614362 | 0 |
| yolo26n-v2-800-round2-plain/seed2/yolo11/fp32 | 0.7900 | 0.6459 | +0.0000 | +0.0000 | 0.0 / 0.022935779816513763 | 0.016 / 0.0821917808219178 | 0.521970 | 0 |
| yolo26n-v2-800-round2-plain/seed0/yolo11/int8 | 0.5144 | 0.2918 | -0.3796 | -0.3407 | 0.0 / 0.013761467889908258 | 0.0 / 0.0 | 0.453125 | 64 |
| yolo26n-v2-800-round2-plain/seed1/yolo11/int8 | 0.5083 | 0.3116 | -0.3626 | -0.3476 | 0.0 / 0.01834862385321101 | 0.0 / 0.0136986301369863 | 0.595215 | 61 |
| yolo26n-v2-800-round2-plain/seed2/yolo11/int8 | 0.5039 | 0.2918 | -0.3541 | -0.3467 | 0.0 / 0.027522935779816515 | 0.0 / 0.0410958904109589 | 0.550781 | 72 |
| yolo26n-v2-800-round2-plain/seed0/yolo26/fp16 | 0.7524 | 0.6402 | +0.0028 | +0.0004 | 0.0 / 0.027522935779816515 | 0.0 / 0.0273972602739726 | 0.632324 | 9 |
| yolo26n-v2-800-round2-plain/seed1/yolo26/fp16 | 0.7585 | 0.6516 | -0.0142 | -0.0052 | 0.0026246719160104987 / 0.03669724770642202 | 0.008 / 0.0410958904109589 | 0.612305 | 10 |
| yolo26n-v2-800-round2-plain/seed2/yolo26/fp16 | 0.7673 | 0.6176 | -0.0057 | -0.0043 | 0.0 / 0.022935779816513763 | 0.008 / 0.0821917808219178 | 0.498047 | 7 |
| yolo26n-v2-800-round2-plain/seed0/yolo26/fp32 | 0.7568 | 0.6374 | +0.0000 | +0.0000 | 0.0 / 0.022935779816513763 | 0.0 / 0.0273972602739726 | 0.615507 | 0 |
| yolo26n-v2-800-round2-plain/seed1/yolo26/fp32 | 0.7533 | 0.6657 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.03669724770642202 | 0.008 / 0.0410958904109589 | 0.618938 | 0 |
| yolo26n-v2-800-round2-plain/seed2/yolo26/fp32 | 0.7690 | 0.6232 | +0.0000 | +0.0000 | 0.0 / 0.022935779816513763 | 0.008 / 0.0821917808219178 | 0.499722 | 0 |
| yolo26n-v2-800-round2-plain/seed0/yolo26/int8 | 0.4742 | 0.2946 | -0.3428 | -0.3411 | 0.0 / 0.013761467889908258 | 0.0 / 0.0 | 0.495117 | 69 |
| yolo26n-v2-800-round2-plain/seed1/yolo26/int8 | 0.4821 | 0.2833 | -0.3824 | -0.3814 | 0.0 / 0.013761467889908258 | 0.0 / 0.0136986301369863 | 0.593750 | 58 |
| yolo26n-v2-800-round2-plain/seed2/yolo26/int8 | 0.4602 | 0.2890 | -0.3343 | -0.3570 | 0.0 / 0.01834862385321101 | 0.0 / 0.0273972602739726 | 0.574219 | 68 |
| yolo26n-v2s-640-round2-plain/seed0/yolo11/fp16 | 0.7699 | 0.6516 | +0.0028 | +0.0005 | 0.0 / 0.04128440366972477 | 0.008 / 0.0821917808219178 | 0.512695 | 1 |
| yolo26n-v2s-640-round2-plain/seed1/yolo11/fp16 | 0.7769 | 0.6657 | -0.0028 | -0.0026 | 0.0026246719160104987 / 0.027522935779816515 | 0.0 / 0.0547945205479452 | 0.535645 | 1 |
| yolo26n-v2s-640-round2-plain/seed2/yolo11/fp16 | 0.7717 | 0.6431 | -0.0085 | -0.0032 | 0.0026246719160104987 / 0.03669724770642202 | 0.0 / 0.0547945205479452 | 0.536621 | 3 |
| yolo26n-v2s-640-round2-plain/seed0/yolo11/fp32 | 0.7743 | 0.6487 | +0.0000 | +0.0000 | 0.0 / 0.04128440366972477 | 0.008 / 0.0821917808219178 | 0.509636 | 0 |
| yolo26n-v2s-640-round2-plain/seed1/yolo11/fp32 | 0.7769 | 0.6686 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.027522935779816515 | 0.0 / 0.0547945205479452 | 0.537163 | 0 |
| yolo26n-v2s-640-round2-plain/seed2/yolo11/fp32 | 0.7708 | 0.6516 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.03669724770642202 | 0.0 / 0.0547945205479452 | 0.545927 | 0 |
| yolo26n-v2s-640-round2-plain/seed0/yolo11/int8 | 0.7323 | 0.6204 | -0.0283 | -0.0118 | 0.0026246719160104987 / 0.009174311926605505 | 0.0 / 0.0273972602739726 | 0.427979 | 27 |
| yolo26n-v2s-640-round2-plain/seed1/yolo11/int8 | 0.7437 | 0.6572 | -0.0113 | -0.0132 | 0.0026246719160104987 / 0.01834862385321101 | 0.0 / 0.0410958904109589 | 0.399902 | 25 |
| yolo26n-v2s-640-round2-plain/seed2/yolo11/int8 | 0.7253 | 0.6459 | -0.0057 | -0.0214 | 0.0026246719160104987 / 0.013761467889908258 | 0.008 / 0.0273972602739726 | 0.482178 | 28 |
| yolo26n-v2s-640-round2-plain/seed0/yolo26/fp16 | 0.7515 | 0.6119 | +0.0028 | +0.0036 | 0.0 / 0.04128440366972477 | 0.008 / 0.0684931506849315 | 0.524902 | 7 |
| yolo26n-v2s-640-round2-plain/seed1/yolo26/fp16 | 0.7568 | 0.6317 | -0.0142 | -0.0048 | 0.0026246719160104987 / 0.027522935779816515 | 0.0 / 0.0547945205479452 | 0.561523 | 6 |
| yolo26n-v2s-640-round2-plain/seed2/yolo26/fp16 | 0.7480 | 0.6317 | -0.0028 | -0.0013 | 0.0026246719160104987 / 0.03669724770642202 | 0.0 / 0.0410958904109589 | 0.552246 | 10 |
| yolo26n-v2s-640-round2-plain/seed0/yolo26/fp32 | 0.7515 | 0.6091 | +0.0000 | +0.0000 | 0.0 / 0.03669724770642202 | 0.008 / 0.0684931506849315 | 0.516671 | 0 |
| yolo26n-v2s-640-round2-plain/seed1/yolo26/fp32 | 0.7463 | 0.6459 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.027522935779816515 | 0.0 / 0.0547945205479452 | 0.579425 | 0 |
| yolo26n-v2s-640-round2-plain/seed2/yolo26/fp32 | 0.7489 | 0.6346 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.03669724770642202 | 0.0 / 0.0410958904109589 | 0.549575 | 0 |
| yolo26n-v2s-640-round2-plain/seed0/yolo26/int8 | 0.7034 | 0.5722 | -0.0368 | -0.0222 | 0.0 / 0.009174311926605505 | 0.0 / 0.0273972602739726 | 0.467041 | 45 |
| yolo26n-v2s-640-round2-plain/seed1/yolo26/int8 | 0.6903 | 0.6261 | -0.0198 | -0.0238 | 0.0026246719160104987 / 0.013761467889908258 | 0.0 / 0.0410958904109589 | 0.504395 | 38 |
| yolo26n-v2s-640-round2-plain/seed2/yolo26/int8 | 0.6990 | 0.6147 | -0.0198 | -0.0241 | 0.0 / 0.009174311926605505 | 0.008 / 0.0273972602739726 | 0.522949 | 37 |
| yolo26n-v2s-800-r3-control-b16/seed0/yolo11/fp16 | 0.8040 | 0.6799 | -0.0142 | -0.0036 | 0.0 / 0.04128440366972477 | 0.0 / 0.0684931506849315 | 0.500977 | 4 |
| yolo26n-v2s-800-r3-control-b16/seed1/yolo11/fp16 | 0.7752 | 0.6884 | -0.0057 | -0.0090 | 0.0 / 0.04128440366972477 | 0.0 / 0.0410958904109589 | 0.546875 | 3 |
| yolo26n-v2s-800-r3-control-b16/seed2/yolo11/fp16 | 0.7813 | 0.7025 | +0.0000 | -0.0031 | 0.0026246719160104987 / 0.03669724770642202 | 0.0 / 0.0684931506849315 | 0.603027 | 3 |
| yolo26n-v2s-800-r3-control-b16/seed0/yolo11/fp32 | 0.8023 | 0.6941 | +0.0000 | +0.0000 | 0.0 / 0.04128440366972477 | 0.0 / 0.0684931506849315 | 0.506970 | 0 |
| yolo26n-v2s-800-r3-control-b16/seed1/yolo11/fp32 | 0.7743 | 0.6941 | +0.0000 | +0.0000 | 0.0 / 0.04128440366972477 | 0.0 / 0.0410958904109589 | 0.543561 | 0 |
| yolo26n-v2s-800-r3-control-b16/seed2/yolo11/fp32 | 0.7830 | 0.7025 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.03669724770642202 | 0.0 / 0.0684931506849315 | 0.602363 | 0 |
| yolo26n-v2s-800-round2-distill/seed0/yolo11/fp16 | 0.8093 | 0.6856 | -0.0057 | -0.0020 | 0.0 / 0.03211009174311927 | 0.0 / 0.0547945205479452 | 0.558105 | 1 |
| yolo26n-v2s-800-round2-distill/seed1/yolo11/fp16 | 0.8075 | 0.6856 | +0.0113 | +0.0039 | 0.0 / 0.03211009174311927 | 0.0 / 0.0547945205479452 | 0.565918 | 2 |
| yolo26n-v2s-800-round2-distill/seed2/yolo11/fp16 | 0.7979 | 0.6799 | -0.0170 | -0.0023 | 0.0 / 0.022935779816513763 | 0.0 / 0.0547945205479452 | 0.583984 | 2 |
| yolo26n-v2s-800-round2-distill/seed0/yolo11/fp32 | 0.8101 | 0.6912 | +0.0000 | +0.0000 | 0.0 / 0.03211009174311927 | 0.0 / 0.0547945205479452 | 0.560291 | 0 |
| yolo26n-v2s-800-round2-distill/seed1/yolo11/fp32 | 0.8058 | 0.6742 | +0.0000 | +0.0000 | 0.0 / 0.03669724770642202 | 0.0 / 0.0547945205479452 | 0.570495 | 0 |
| yolo26n-v2s-800-round2-distill/seed2/yolo11/fp32 | 0.7970 | 0.6969 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.022935779816513763 | 0.0 / 0.0547945205479452 | 0.586435 | 0 |
| yolo26n-v2s-800-round2-plain/seed0/yolo11/fp16 | 0.7813 | 0.6884 | -0.0057 | -0.0014 | 0.0026246719160104987 / 0.045871559633027525 | 0.008 / 0.0821917808219178 | 0.566406 | 2 |
| yolo26n-v2s-800-round2-plain/seed1/yolo11/fp16 | 0.7962 | 0.6912 | +0.0000 | -0.0000 | 0.0 / 0.03211009174311927 | 0.0 / 0.0410958904109589 | 0.626465 | 2 |
| yolo26n-v2s-800-round2-plain/seed2/yolo11/fp16 | 0.7804 | 0.6771 | -0.0028 | -0.0074 | 0.005249343832020997 / 0.04128440366972477 | 0.0 / 0.0684931506849315 | 0.587891 | 1 |
| yolo26n-v2s-800-round2-plain/seed0/yolo11/fp32 | 0.7813 | 0.6941 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.045871559633027525 | 0.008 / 0.0821917808219178 | 0.567661 | 0 |
| yolo26n-v2s-800-round2-plain/seed1/yolo11/fp32 | 0.7892 | 0.6912 | +0.0000 | +0.0000 | 0.0 / 0.03211009174311927 | 0.0 / 0.0410958904109589 | 0.639187 | 0 |
| yolo26n-v2s-800-round2-plain/seed2/yolo11/fp32 | 0.7760 | 0.6799 | +0.0000 | +0.0000 | 0.005249343832020997 / 0.04128440366972477 | 0.0 / 0.0684931506849315 | 0.597301 | 0 |
| yolo26n-v2s-800-round2-plain/seed0/yolo11/int8 | 0.5468 | 0.3314 | -0.3626 | -0.3352 | 0.005249343832020997 / 0.013761467889908258 | 0.0 / 0.0547945205479452 | 0.462891 | 62 |
| yolo26n-v2s-800-round2-plain/seed1/yolo11/int8 | 0.5293 | 0.3229 | -0.3683 | -0.3371 | 0.005249343832020997 / 0.009174311926605505 | 0.0 / 0.0273972602739726 | 0.452393 | 53 |
| yolo26n-v2s-800-round2-plain/seed2/yolo11/int8 | 0.5468 | 0.3229 | -0.3569 | -0.3203 | 0.0026246719160104987 / 0.027522935779816515 | 0.0 / 0.0410958904109589 | 0.517090 | 54 |
| yolo26n-v2s-800-round2-plain/seed0/yolo26/fp16 | 0.7559 | 0.6317 | +0.0000 | +0.0065 | 0.0 / 0.04128440366972477 | 0.0 / 0.0547945205479452 | 0.579590 | 6 |
| yolo26n-v2s-800-round2-plain/seed1/yolo26/fp16 | 0.7743 | 0.6657 | +0.0000 | +0.0005 | 0.0 / 0.027522935779816515 | 0.0 / 0.0273972602739726 | 0.623535 | 5 |
| yolo26n-v2s-800-round2-plain/seed2/yolo26/fp16 | 0.7384 | 0.6714 | +0.0000 | -0.0002 | 0.005249343832020997 / 0.03669724770642202 | 0.0 / 0.0547945205479452 | 0.629883 | 7 |
| yolo26n-v2s-800-round2-plain/seed0/yolo26/fp32 | 0.7559 | 0.6317 | +0.0000 | +0.0000 | 0.0 / 0.05045871559633028 | 0.0 / 0.0547945205479452 | 0.578010 | 0 |
| yolo26n-v2s-800-round2-plain/seed1/yolo26/fp32 | 0.7725 | 0.6657 | +0.0000 | +0.0000 | 0.0 / 0.027522935779816515 | 0.0 / 0.0273972602739726 | 0.625657 | 0 |
| yolo26n-v2s-800-round2-plain/seed2/yolo26/fp32 | 0.7410 | 0.6714 | +0.0000 | +0.0000 | 0.005249343832020997 / 0.03211009174311927 | 0.0 / 0.0684931506849315 | 0.628161 | 0 |
| yolo26n-v2s-800-round2-plain/seed0/yolo26/int8 | 0.5144 | 0.3144 | -0.3173 | -0.3243 | 0.005249343832020997 / 0.009174311926605505 | 0.0 / 0.0547945205479452 | 0.509277 | 66 |
| yolo26n-v2s-800-round2-plain/seed1/yolo26/int8 | 0.4908 | 0.3088 | -0.3569 | -0.3573 | 0.0 / 0.009174311926605505 | 0.0 / 0.0273972602739726 | 0.468018 | 62 |
| yolo26n-v2s-800-round2-plain/seed2/yolo26/int8 | 0.5066 | 0.3201 | -0.3513 | -0.3329 | 0.0026246719160104987 / 0.022935779816513763 | 0.0 / 0.0273972602739726 | 0.577637 | 56 |
| yolo26n-v2s-species-800-r3/seed0/yolo11/fp16 | 0.8093 | 0.6912 | +0.0000 | -0.0009 | 0.0 / 0.05045871559633028 | 0.008 / 0.0547945205479452 | 0.575195 | 3 |
| yolo26n-v2s-species-800-r3/seed1/yolo11/fp16 | 0.8206 | 0.6912 | -0.0113 | -0.0048 | 0.0 / 0.05045871559633028 | 0.008 / 0.0547945205479452 | 0.548828 | 3 |
| yolo26n-v2s-species-800-r3/seed2/yolo11/fp16 | 0.8136 | 0.6742 | -0.0028 | -0.0003 | 0.0026246719160104987 / 0.045871559633027525 | 0.008 / 0.0410958904109589 | 0.604980 | 5 |
| yolo26n-v2s-species-800-r3/seed0/yolo11/fp32 | 0.8084 | 0.6912 | +0.0000 | +0.0000 | 0.0 / 0.05045871559633028 | 0.008 / 0.0547945205479452 | 0.574959 | 0 |
| yolo26n-v2s-species-800-r3/seed1/yolo11/fp32 | 0.8198 | 0.7025 | +0.0000 | +0.0000 | 0.0 / 0.05045871559633028 | 0.008 / 0.0547945205479452 | 0.554943 | 0 |
| yolo26n-v2s-species-800-r3/seed2/yolo11/fp32 | 0.8136 | 0.6771 | +0.0000 | +0.0000 | 0.0026246719160104987 / 0.045871559633027525 | 0.008 / 0.0410958904109589 | 0.609674 | 0 |

## Rejected gates

- `yolo26n-v2-800-round2-plain/yolo11/fp16/[448, 800]`: shipped seed2 fails confuser_far
- `yolo26n-v2-800-round2-plain/yolo11/fp32/[448, 800]`: fp32 is the strict parity companion; final deployment uses fp16; shipped seed2 fails confuser_far; shipped seed2 fails reaction_p95_ms
- `yolo26n-v2-800-round2-plain/yolo11/int8/[448, 800]`: int8 excluded by coordinator decision; shipped seed0 fails reaction_p95_ms; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2-800-round2-plain/yolo26/fp16/[448, 800]`: one-to-one head excluded by coordinator decision; shipped seed2 fails confuser_far
- `yolo26n-v2-800-round2-plain/yolo26/fp32/[448, 800]`: fp32 is the strict parity companion; final deployment uses fp16; one-to-one head excluded by coordinator decision; shipped seed2 fails confuser_far; shipped seed2 fails reaction_p95_ms
- `yolo26n-v2-800-round2-plain/yolo26/int8/[448, 800]`: int8 excluded by coordinator decision; one-to-one head excluded by coordinator decision; shipped seed1 fails reaction_p95_ms; shipped seed1 fails export parity; shipped seed1 fails deployed precision accuracy
- `yolo26n-v2s-640-round2-plain/yolo11/fp16/[384, 640]`: shipped seed1 fails confuser_far
- `yolo26n-v2s-640-round2-plain/yolo11/fp32/[384, 640]`: fp32 is the strict parity companion; final deployment uses fp16; shipped seed1 fails confuser_far
- `yolo26n-v2s-640-round2-plain/yolo11/int8/[384, 640]`: int8 excluded by coordinator decision; shipped seed1 fails export parity; shipped seed1 fails deployed precision accuracy
- `yolo26n-v2s-640-round2-plain/yolo26/fp16/[384, 640]`: one-to-one head excluded by coordinator decision; shipped seed1 fails confuser_far; shipped seed1 fails export parity; shipped seed1 fails deployed precision accuracy
- `yolo26n-v2s-640-round2-plain/yolo26/fp32/[384, 640]`: fp32 is the strict parity companion; final deployment uses fp16; one-to-one head excluded by coordinator decision; shipped seed0 fails confuser_far
- `yolo26n-v2s-640-round2-plain/yolo26/int8/[384, 640]`: int8 excluded by coordinator decision; one-to-one head excluded by coordinator decision; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2s-800-r3-control-b16/yolo11/fp16/[448, 800]`: shipped seed0 fails confuser_far; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2s-800-r3-control-b16/yolo11/fp32/[448, 800]`: fp32 is the strict parity companion; final deployment uses fp16; shipped seed0 fails confuser_far; shipped seed0 fails reaction_p95_ms
- `yolo26n-v2s-800-round2-distill/yolo11/fp16/[448, 800]`: shipped seed0 fails confuser_far
- `yolo26n-v2s-800-round2-distill/yolo11/fp32/[448, 800]`: fp32 is the strict parity companion; final deployment uses fp16; shipped seed0 fails confuser_far; shipped seed0 fails reaction_p95_ms
- `yolo26n-v2s-800-round2-plain/yolo11/fp32/[448, 800]`: fp32 is the strict parity companion; final deployment uses fp16; shipped seed1 fails reaction_p95_ms
- `yolo26n-v2s-800-round2-plain/yolo11/int8/[448, 800]`: int8 excluded by coordinator decision; shipped seed0 fails confuser_far; shipped seed0 fails reaction_p95_ms; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2s-800-round2-plain/yolo26/fp16/[448, 800]`: one-to-one head excluded by coordinator decision
- `yolo26n-v2s-800-round2-plain/yolo26/fp32/[448, 800]`: fp32 is the strict parity companion; final deployment uses fp16; one-to-one head excluded by coordinator decision; shipped seed1 fails reaction_p95_ms
- `yolo26n-v2s-800-round2-plain/yolo26/int8/[448, 800]`: int8 excluded by coordinator decision; one-to-one head excluded by coordinator decision; shipped seed0 fails confuser_far; shipped seed0 fails reaction_p95_ms; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2s-species-800-r3/yolo11/fp16/[448, 800]`: species shipped empty_far is worse than matched single
- `yolo26n-v2s-species-800-r3/yolo11/fp32/[448, 800]`: fp32 is the strict parity companion; final deployment uses fp16; shipped seed2 fails reaction_p95_ms; species shipped empty_far is worse than matched single

Admissible Pareto front: `['yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800]']`.

No measured Pi 5 result exists. The calibration factor and the mapping
from still-image recall to temporal hits remain estimates and assumptions.

Reproduction commands, candidate addition, resource handling and policy
requests are in [WS7-methods.md](WS7-methods.md).

## Deployment candidate ranking and section-1 gates

The six standard-head fp16 arms below are the deployment candidates. Each has the complete section-1 gate table for its validation-chosen seed, plus the three-seed accuracy rule. Rank is three-seed deploy-size val R@P90; held arms retain their val rank. FP32 companions, int8 and one-to-one configurations are excluded diagnostics; their complete gate tables are in the final appendix.

### 1. yolo26n-v2s-species-800-r3

Val mean 0.814523; val-chosen seed 2; HOLD.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.4636449480642115 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.0026246719160104987 | <= 0.02 | pass |
| Val confuser FAR | 0.045871559633027525 | <= 0.05 | pass |
| Test empty FAR | 0.008 | <= 0.02 | pass |
| Test confuser FAR | 0.0410958904109589 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 36.89069292924056 | <= 100.0 | pass |
| Simulated reaction p95 ms | 342.7398861968967 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.0028328611898016387 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp16 | standard head; int8 excluded | pass |
| Species pigeon three-seed recall | 0.6855524079320113 | >= 0.675552 | pass |
| Species shipped empty_far no worse than single | 0.008 | <= 0.0 | fail |
| Species shipped confuser_far no worse than single | 0.0410958904109589 | <= 0.0410958904109589 | pass |
| Species estimated Pi p95 within 5% of single | 36.89069292924056 | <= 38.370851 ms | pass |

### 2. yolo26n-v2s-800-round2-distill

Val mean 0.804899; val-chosen seed 0; HOLD.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.4617563739376771 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.0 | <= 0.02 | pass |
| Val confuser FAR | 0.03211009174311927 | <= 0.05 | pass |
| Test empty FAR | 0.0 | <= 0.02 | pass |
| Test confuser FAR | 0.0547945205479452 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 36.45666954209171 | <= 100.0 | pass |
| Simulated reaction p95 ms | 307.84920709793494 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.005665722379603388 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp16 | standard head; int8 excluded | pass |

### 3. yolo26n-v2s-800-r3-control-b16

Val mean 0.786818; val-chosen seed 0; HOLD.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.4683663833805477 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.0 | <= 0.02 | pass |
| Val confuser FAR | 0.04128440366972477 | <= 0.05 | pass |
| Test empty FAR | 0.0 | <= 0.02 | pass |
| Test confuser FAR | 0.0684931506849315 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 36.56229097655727 | <= 100.0 | pass |
| Simulated reaction p95 ms | 303.3776924979263 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.014164305949008527 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo11 / fp16 | standard head; int8 excluded | pass |

### 4. yolo26n-v2s-800-round2-plain

Val mean 0.785943; val-chosen seed 1; PASS.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.4636449480642115 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.0 | <= 0.02 | pass |
| Val confuser FAR | 0.03211009174311927 | <= 0.05 | pass |
| Test empty FAR | 0.0 | <= 0.02 | pass |
| Test confuser FAR | 0.0410958904109589 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 36.543667408497654 | <= 100.0 | pass |
| Simulated reaction p95 ms | 345.3639649591854 | <= 500.0 | pass |
| Precision accuracy maximum drop | 5.120128417135383e-06 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp16 | standard head; int8 excluded | pass |

### 5. yolo26n-v2-800-round2-plain

Val mean 0.779528; val-chosen seed 2; HOLD.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.4428706326723324 | > 0.020098, larger seed SD | pass |
| Val empty FAR | 0.0 | <= 0.02 | pass |
| Val confuser FAR | 0.022935779816513763 | <= 0.05 | pass |
| Test empty FAR | 0.016 | <= 0.02 | pass |
| Test confuser FAR | 0.0821917808219178 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 36.70146878222423 | <= 100.0 | pass |
| Simulated reaction p95 ms | 346.83590988286767 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.0028328611898016387 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp16 | standard head; int8 excluded | pass |

### 6. yolo26n-v2s-640-round2-plain

Val mean 0.772820; val-chosen seed 1; HOLD.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.4315391879131256 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.0026246719160104987 | <= 0.02 | pass |
| Val confuser FAR | 0.027522935779816515 | <= 0.05 | pass |
| Test empty FAR | 0.0 | <= 0.02 | pass |
| Test confuser FAR | 0.0547945205479452 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 25.162066024942284 | <= 100.0 | pass |
| Simulated reaction p95 ms | 289.7632858543775 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.0028328611898017497 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp16 | standard head; int8 excluded | pass |

## Square-val nominations versus deploy-val seed picks

Coordinator confirmed deploy-size val takes precedence over WS5 square-val seed nominations. Apply identical val eligibility, R@P90 and ties to every arm. Test gates apply chosen seed only; no test fallback. Three-seed arm rule unchanged.
Confirmed before inspecting deploy-size test metrics for any switched seed. First distilled ARM job 958128 just submitted; no new native test metrics inspected.
Policy recorded UTC: 2026-10-16T08:36:11.859661+00:00. WS5 values below are rounded square-input context; deploy values are measured native fp16.

| Arm | WS5 seed | Square val R@P90 | Deploy seed | Deploy val R@P90 | Nominated seed at deploy val | Fresh deploy threshold |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n-v2s-800-round2-distill | 1 | 0.815 | 0 | 0.8092738407699037 | 0.8075240594925635 | 0.55810546875 |
| yolo26n-v2s-800-round2-plain | 1 | 0.801 | 1 | 0.7961504811898513 | 0.7961504811898513 | 0.62646484375 |
| yolo26n-v2s-species-800-r3 | 2 | 0.822 | 2 | 0.8136482939632546 | 0.8136482939632546 | 0.60498046875 |
| yolo26n-v2s-800-r3-control-b16 | 2 | 0.808 | 0 | 0.8040244969378828 | 0.7812773403324584 | 0.5009765625 |

## Reaction scheduling audit

simulated using estimated Pi latency, not measured Pi reaction
WS4 serialises each sampled processing duration, then selects the newest available camera index. Stale captures are skipped. Confirmation counts processed observations, matching ConfirmationTracker.update. The existing 30 FPS capture result already includes throughput limits; reducing camera FPS to detector FPS would double-count them.

Capture rate: 30.0 FPS. Estimated effective detector rate: 27.446 FPS. Registered 30 FPS capture p95: 345.3639649591854 ms. Throughput-limited gate p95: 345.3639649591854 ms. These are the same correctly scheduled simulation, not separate gates.

**The 345.364 ms reaction p95 is simulated, not a Pi measurement. WS10 step 3 must confirm it on the physical Pi.** Sensor delivery, temporal correlation, track association, capture contention and thermal effects remain unmeasured.

## Why this model

`yolo26n-v2s-800-round2-plain` wins the registered ranking among arms passing every gate. Seed choice and the operating threshold use val alone. Test-v2 supplies promotion gates; styles is secondary and only breaks matched v2/v2s ties.
Distill-800 loses at deploy size: its validation-chosen seed0 has four test confuser false alarms out of 73, FAR 0.054795, above 0.05. Species seed2 fails its relative empty-FAR clause, 0.008 versus plain's zero. Control seed0 fails test confuser FAR, 0.068493, and loses 0.014164 test R@P90 under fp16, beyond the 0.01 policy. Its fp32 strict companion passes; its fp16 accuracy failure makes the combined deployment parity gate fail. No other seed was tried after a test failure.

## Deployment limits and WS10 hand-off

The plain-800 same-checkpoint geometry diagnostic below attributes 38 of 39 lost test matches to portrait images at the global P90 cut. Test-v2 has only nine approximately 16:9 images, two positive. This limits camera claims. WS10 columba-test-site-v1 is the real site check.

WS10 steps 5–7 remain open: site threshold and k/n, the Pi OMP wait policy, and on-Pi capture/model geometry. No Pi measurement is claimed. The launcher stays unchanged; authorised capture FPS is 30.

## Durable package

`yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`. Provisional: False. SHA-256 and byte counts verified.
Package source: `runs/detect/pigeon_detector/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16/weights/best_ncnn_model`. Verified copy: `models/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`.
Manifest: `deploy/model_manifest.json`. Eligible for promotion: `True`.

## Rectangular baseline

baseline-repro-p50, 384x640, own val-selected thresholds. PyTorch CPU on Mac vexcalibur; native package-default fp32/fp16 on xcng0. Primary comparison uses native fp16, matching the provisional export. Small-object recall is zero for all three baseline seeds.

| Seed | PyTorch R@P90 | Native fp32 R@P90 | Native fp16 R@P90 | FP16 AP50 | FP16 operating R | FP16 empty / confuser FAR |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.2323 | 0.2323 | 0.2295 | 0.4689 | 0.3229 | 0.000 / 0.3014 |
| 1 | 0.2295 | 0.2295 | 0.2295 | 0.4918 | 0.3428 | 0.000 / 0.3699 |
| 2 | 0.2068 | 0.2068 | 0.2068 | 0.5067 | 0.3541 | 0.008 / 0.4658 |

Three-seed test R@P90 mean +/- SD: pt 0.222852 +/- 0.013974; fp32 0.222852 +/- 0.013974; fp16 0.221907 +/- 0.013084.

## Selected export gates

Shipped seed 1 was chosen on val only. The three-seed means rank the arm; the rows below gate its shipped export.

| Gate | Value | Requirement | Pass |
|---|---:|---|---|
| Three-seed test improvement | 0.463645 | > 0.013084, larger seed SD | yes |
| Val empty FAR | 0.000000 | <= 0.02 | yes |
| Val confuser FAR | 0.032110 | <= 0.05 | yes |
| Test empty FAR | 0.000000 | <= 0.02 | yes |
| Test confuser FAR | 0.041096 | <= 0.05 | yes |
| Estimated Pi frame p95 ms | 36.543667 | <= 100.0 | yes |
| Simulated reaction p95 ms | 345.363965 | <= 500.0 | yes |
| Precision accuracy maximum drop | 0.000005 | <= 0.01 in each val/test R@P90, AP50, operating recall | yes |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | yes |
| Deployment policy | yolo11 / fp16 | standard head; int8 excluded | yes |

## Aspect-ratio diagnostic, PyTorch only

Measured on `vexcalibur`, CPU, same checkpoint and preprocessing implementation.
Inputs are 800x800 versus 448x800. The wide bucket includes ratios >=1.5396;
4:3 includes [1.1547,1.5396), square [1,1.1547), portrait <1.
Small means projected short side <32 px under WS1. Its cohort changes with input size.
The final column holds the square-defined small cohort fixed for both predictions.
Subset R@P90 can use a different diagnostic precision-90 score cut in each bucket.
No validation operating threshold or promotion gate is changed.

| Bucket | Images / pigeon images / boxes | Content scale ratio | R@P90 square / rect | Small R@P90 square / rect | Small boxes square / rect | Rect small recall, fixed cohort |
|---|---:|---:|---:|---:|---:|---:|
| all | 319 / 121 / 353 | 0.591 | 0.8017 / 0.6912 | 0.5850 / 0.4726 | 147 / 201 | 0.4082 |
| approx-16:9 | 9 / 2 / 17 | 0.957 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 7 / 7 | 1.0000 |
| 4:3 | 136 / 5 / 16 | 0.824 | 0.9375 / 0.8750 | n/a / n/a | 0 / 0 | n/a |
| square | 107 / 91 / 118 | 0.560 | 0.9915 / 0.9915 | n/a / 1.0000 | 0 / 1 | n/a |
| portrait | 67 / 23 / 202 | 0.560 | 0.5594 / 0.4158 | 0.4286 / 0.3938 | 140 / 193 | 0.3143 |

At each shape's full-test P90 cut, diagnostic matched pigeon boxes are:

| Bucket | Square matched | Rectangular matched | Lost matches |
|---|---:|---:|---:|
| all | 283 | 244 | 39 |
| approx-16:9 | 17 | 17 | 0 |
| 4:3 | 15 | 14 | 1 |
| square | 117 | 117 | 0 |
| portrait | 134 | 96 | 38 |

The geometry hypothesis is supported for this checkpoint. Square-to-rectangular content scale is 0.56 in portrait and square images, versus a pigeon-weighted 0.957 in the wide bucket. At the respective full-test P90 cuts, 38 of 39 lost matches are portrait pigeons, one is in the 4:3 bucket, and square/wide matches are unchanged. The fixed square-defined small cohort also loses recall, so this is more than a changing small-object denominator. It is an observational geometry comparison, not a controlled attribution of every error: canvas padding and sampling also change. The wide bucket contains only two positive images and 17 boxes, so unchanged recall does not establish real-camera performance. The square CPU result is 0.8017 for this seed; the approximately 0.78 WS5 value is a three-seed mean. Checked-in app/config.yaml and CameraConfig still default to 640x480. The scale-match argument requires an actual 16:9 capture stream; WS10 should verify the deployed stream geometry. No capture setting was changed.

## Quantisation diagnostic

A plausible cause of the 800-arm int8 collapse is full-graph KL post-training
quantisation sensitivity, including the detection head, to model activation ranges.
Both sizes used the same 200 val IDs and RGB /255 preprocessing. The 800 and
640 checkpoints also differ, so this is not a controlled input-shape comparison.
No specific cause is established. Int8 is excluded; no further investigation was run.

## Final package, config and verification

The winner is also the plain fallback, so one durable package serves both roles. No duplicate fallback manifest entry is needed. The source follows the incumbent's direct-Git convention; no Git LFS filter is configured. The manifest has no absolute worktree path or ignored-output source.

| Package file | Bytes | SHA-256 |
|---|---:|---|
| model.ncnn.param | 26392 | `bd6201ead99a645b200d84ec76222c24475324f8be2f90d9de9e5b5513fba8f0` |
| model.ncnn.bin | 4885148 | `639e90dc36ad34d1235221f4aa907fae5a434245a90883454c1ca8dd4d7ffb6e` |
| metadata.yaml | 845 | `051f3facc7f1bd9385221496329c99acc7fe22a250fe3cec339b4e3452d9bd1f` |

`app/config.yaml` now selects the native package at 448x800 with the independently val-selected threshold 0.62646484375. Confirmation is enabled at k2/n3. Capture is 800x450 lores at 30 FPS; recording FPS stays 15. All requested keys already existed in the schema. Other defaults and the launcher are unchanged.

Final verification is **measured on Mac vexcalibur**. Preparation rechecked actual model bytes and C8 metadata. A fresh native replay processed 20 frames and produced one confirmed trigger. This unpaced file replay proves loading and pipeline operation; it does not measure real camera reaction time.

| Check | Result |
|---|---|
| Full pytest | 325 passed, 223 subtests passed, zero skipped |
| Full unittest discovery | 325 passed |
| Native CTest | 1 passed |
| Ruff check, app/tests/tools/cluster/training/native | pass |
| Ruff format check, same directories | 94 files already formatted |
| Full mypy | 91 source files, no issues |

Two test fixtures were corrected after the deployment change: camera schema defaults now come from an empty config, and warm-up failure uses the backend-neutral detector factory. Neither change alters runtime behaviour. `WS7-final-verification.json` records current counts, replay evidence and log hashes. `WS7-final-preparation.json` records the completed jobs and checkpoint hashes. No cluster job was submitted during finalisation.

No selection or contract decision remains open. WS10 step 3 must measure real Pi reaction p95. WS10 steps 5–7 still cover site threshold/k-n, the Pi OMP wait policy and capture/model geometry.

## Section-1 gates for excluded diagnostic configurations

These 18 configurations are not deployment candidates. The rows retain their validation-chosen seed and measured metrics for completeness. Final admission is always false: fp32 is a strict-parity companion, while int8 and one-to-one are coordinator exclusions. These tables do not reopen selection.

### Diagnostic yolo26n-v2-800-round2-plain/yolo11/fp32/[448, 800]

Val-chosen seed 2; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.441926 | > 0.015602, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.022936 | <= 0.05 | pass |
| Test empty FAR | 0.016000 | <= 0.02 | pass |
| Test confuser FAR | 0.082192 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 69.263138 | <= 100.0 | pass |
| Simulated reaction p95 ms | 612.669799 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp32 | standard head; int8 excluded | pass |
| Final deployment admission | yolo11 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2-800-round2-plain/yolo11/int8/[448, 800]

Val-chosen seed 0; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.076487 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.013761 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.000000 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 33.709555 | <= 100.0 | pass |
| Simulated reaction p95 ms | 1051.617525 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.379603 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo11 / int8 | standard head; int8 excluded | fail |
| Final deployment admission | yolo11 / int8 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2-800-round2-plain/yolo26/fp16/[448, 800]

Val-chosen seed 2; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.414542 | > 0.017309, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.022936 | <= 0.05 | pass |
| Test empty FAR | 0.008000 | <= 0.02 | pass |
| Test confuser FAR | 0.082192 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 36.609518 | <= 100.0 | pass |
| Simulated reaction p95 ms | 353.334758 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.005666 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo26 / fp16 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / fp16 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2-800-round2-plain/yolo26/fp32/[448, 800]

Val-chosen seed 2; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.420208 | > 0.021636, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.022936 | <= 0.05 | pass |
| Test empty FAR | 0.008000 | <= 0.02 | pass |
| Test confuser FAR | 0.082192 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 69.507125 | <= 100.0 | pass |
| Simulated reaction p95 ms | 635.401273 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo26 / fp32 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2-800-round2-plain/yolo26/int8/[448, 800]

Val-chosen seed 1; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.067044 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.013761 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.013699 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 33.698084 | <= 100.0 | pass |
| Simulated reaction p95 ms | 1080.571984 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.382436 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo26 / int8 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / int8 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-640-round2-plain/yolo11/fp32/[384, 640]

Val-chosen seed 1; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.434372 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.002625 | <= 0.02 | pass |
| Val confuser FAR | 0.027523 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.054795 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 46.754263 | <= 100.0 | pass |
| Simulated reaction p95 ms | 417.747684 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp32 | standard head; int8 excluded | pass |
| Final deployment admission | yolo11 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-640-round2-plain/yolo11/int8/[384, 640]

Val-chosen seed 1; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.419263 | > 0.018862, larger seed SD | pass |
| Val empty FAR | 0.002625 | <= 0.02 | pass |
| Val confuser FAR | 0.018349 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.041096 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 23.456034 | <= 100.0 | pass |
| Simulated reaction p95 ms | 293.585550 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.033246 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo11 / int8 | standard head; int8 excluded | fail |
| Final deployment admission | yolo11 / int8 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-640-round2-plain/yolo26/fp16/[384, 640]

Val-chosen seed 1; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.403211 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.002625 | <= 0.02 | pass |
| Val confuser FAR | 0.027523 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.054795 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 25.138780 | <= 100.0 | pass |
| Simulated reaction p95 ms | 321.109365 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.014164 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo26 / fp16 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / fp16 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-640-round2-plain/yolo26/fp32/[384, 640]

Val-chosen seed 0; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.407932 | > 0.018862, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.036697 | <= 0.05 | pass |
| Test empty FAR | 0.008000 | <= 0.02 | pass |
| Test confuser FAR | 0.068493 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 46.436519 | <= 100.0 | pass |
| Simulated reaction p95 ms | 465.122550 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo26 / fp32 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-640-round2-plain/yolo26/int8/[384, 640]

Val-chosen seed 0; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.382436 | > 0.028376, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.009174 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.027397 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 23.447808 | <= 100.0 | pass |
| Simulated reaction p95 ms | 379.962839 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.048159 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo26 / int8 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / int8 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-800-r3-control-b16/yolo11/fp32/[448, 800]

Val-chosen seed 0; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.474976 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.041284 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.068493 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 68.701518 | <= 100.0 | pass |
| Simulated reaction p95 ms | 521.027137 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp32 | standard head; int8 excluded | pass |
| Final deployment admission | yolo11 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-800-round2-distill/yolo11/fp32/[448, 800]

Val-chosen seed 0; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.465534 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.032110 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.054795 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 69.003084 | <= 100.0 | pass |
| Simulated reaction p95 ms | 538.362825 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp32 | standard head; int8 excluded | pass |
| Final deployment admission | yolo11 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-800-round2-plain/yolo11/fp32/[448, 800]

Val-chosen seed 1; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.466478 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.032110 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.041096 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 68.620396 | <= 100.0 | pass |
| Simulated reaction p95 ms | 609.290558 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp32 | standard head; int8 excluded | pass |
| Final deployment admission | yolo11 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-800-round2-plain/yolo11/int8/[448, 800]

Val-chosen seed 0; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.103872 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.005249 | <= 0.02 | pass |
| Val confuser FAR | 0.013761 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.054795 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 33.389419 | <= 100.0 | pass |
| Simulated reaction p95 ms | 809.967002 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.362606 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo11 / int8 | standard head; int8 excluded | fail |
| Final deployment admission | yolo11 / int8 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-800-round2-plain/yolo26/fp16/[448, 800]

Val-chosen seed 1; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.434372 | > 0.021450, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.027523 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.027397 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 36.531362 | <= 100.0 | pass |
| Simulated reaction p95 ms | 360.400148 | <= 500.0 | pass |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo26 / fp16 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / fp16 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-800-round2-plain/yolo26/fp32/[448, 800]

Val-chosen seed 1; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.434372 | > 0.021450, larger seed SD | pass |
| Val empty FAR | 0.000000 | <= 0.02 | pass |
| Val confuser FAR | 0.027523 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.027397 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 69.496002 | <= 100.0 | pass |
| Simulated reaction p95 ms | 650.073765 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo26 / fp32 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / fp32 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-800-round2-plain/yolo26/int8/[448, 800]

Val-chosen seed 0; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.092540 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.005249 | <= 0.02 | pass |
| Val confuser FAR | 0.009174 | <= 0.05 | pass |
| Test empty FAR | 0.000000 | <= 0.02 | pass |
| Test confuser FAR | 0.054795 | <= 0.05 | fail |
| Estimated Pi frame p95 ms | 33.547623 | <= 100.0 | pass |
| Simulated reaction p95 ms | 882.218386 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.324273 | <= 0.01 in each val/test R@P90, AP50, operating recall | fail |
| Export parity | False | strict same-graph fp32 + accuracy for fp16 | fail |
| Deployment policy | yolo26 / int8 | standard head; int8 excluded | fail |
| Final deployment admission | yolo26 / int8 | standard yolo11 head and fp16 | excluded |

### Diagnostic yolo26n-v2s-species-800-r3/yolo11/fp32/[448, 800]

Val-chosen seed 2; excluded from final deployment.

| Gate | Observed | Requirement | Result |
|---|---|---|---|
| Three-seed test improvement | 0.468366 | > 0.013084, larger seed SD | pass |
| Val empty FAR | 0.002625 | <= 0.02 | pass |
| Val confuser FAR | 0.045872 | <= 0.05 | pass |
| Test empty FAR | 0.008000 | <= 0.02 | pass |
| Test confuser FAR | 0.041096 | <= 0.05 | pass |
| Estimated Pi frame p95 ms | 70.022148 | <= 100.0 | pass |
| Simulated reaction p95 ms | 612.642532 | <= 500.0 | fail |
| Precision accuracy maximum drop | 0.000000 | <= 0.01 in each val/test R@P90, AP50, operating recall | pass |
| Export parity | True | strict same-graph fp32 + accuracy for fp16 | pass |
| Deployment policy | yolo11 / fp32 | standard head; int8 excluded | pass |
| Species pigeon three-seed recall | 0.690274 | >= 0.678385 | pass |
| Species shipped empty_far no worse than single | 0.008000 | <= 0.0 | fail |
| Species shipped confuser_far no worse than single | 0.041096 | <= 0.0410958904109589 | pass |
| Species estimated Pi p95 within 5% of single | 70.022148 | <= 72.051416 ms | pass |
| Final deployment admission | yolo11 / fp32 | standard yolo11 head and fp16 | excluded |
