# WS7 provisional export decision

Provisional selection: `yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800]`.
Final selection waits for distilled and species candidates.
Coverage: 18 configurations and 54 seed/export cases.
Comparison baseline: `baseline-repro-p50`, measured native fp16 384x640 test R@P90 0.2219 +/- 0.0131, independently val-thresholded per seed.
Styles is a secondary measured column. Test-v2 still supplies every promotion
gate. Paired v2/v2s arms within 0.01 validation recall and 5% measured latency
use styles recall to break the tie. The 5% band treats host timing noise
as equal latency.

Primary accuracy is measured using columba_native package-default compute on xcng0.
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
| yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800] | 0.7859 | 0.6856 +/- 0.0075 | 0.7416 | 0.6298 | 0.5187 | 0.5219 | 36.54 | 345.36 | 1 | yes |
| yolo26n-v2s-800-round2-plain/yolo11/fp32/[448, 800] | 0.7822 | 0.6884 +/- 0.0075 | 0.7446 | 0.6280 | 0.5187 | 0.5187 | 68.62 | 609.29 | 1 | no |
| yolo26n-v2s-800-round2-plain/yolo11/int8/[448, 800] | 0.5410 | 0.3258 +/- 0.0049 | 0.4137 | 0.3277 | 0.5187 | 0.3245 | 33.39 | 809.97 | 0 | no |
| yolo26n-v2s-800-round2-plain/yolo26/fp16/[448, 800] | 0.7562 | 0.6563 +/- 0.0215 | 0.7407 | 0.5996 | 0.4907 | 0.4939 | 36.53 | 360.40 | 1 | no |
| yolo26n-v2s-800-round2-plain/yolo26/fp32/[448, 800] | 0.7565 | 0.6563 +/- 0.0215 | 0.7384 | 0.6015 | 0.4907 | 0.4907 | 69.50 | 650.07 | 1 | no |
| yolo26n-v2s-800-round2-plain/yolo26/int8/[448, 800] | 0.5039 | 0.3144 +/- 0.0057 | 0.4002 | 0.3154 | 0.4907 | 0.3006 | 33.55 | 882.22 | 0 | no |

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
| yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800] | -0.0028 | -0.0029 | 3/3 | 0/3 |
| yolo26n-v2s-800-round2-plain/yolo11/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-800-round2-plain/yolo11/int8/[448, 800] | -0.3626 | -0.3308 | 0/3 | 0/3 |
| yolo26n-v2s-800-round2-plain/yolo26/fp16/[448, 800] | +0.0000 | +0.0023 | 3/3 | 0/3 |
| yolo26n-v2s-800-round2-plain/yolo26/fp32/[448, 800] | +0.0000 | +0.0000 | 3/3 | 3/3 |
| yolo26n-v2s-800-round2-plain/yolo26/int8/[448, 800] | -0.3418 | -0.3382 | 0/3 | 0/3 |

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

## Rejected gates

- `yolo26n-v2-800-round2-plain/yolo11/fp16/[448, 800]`: shipped seed2 fails confuser_far
- `yolo26n-v2-800-round2-plain/yolo11/fp32/[448, 800]`: shipped seed2 fails confuser_far; shipped seed2 fails reaction_p95_ms
- `yolo26n-v2-800-round2-plain/yolo11/int8/[448, 800]`: int8 excluded by coordinator decision; shipped seed0 fails reaction_p95_ms; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2-800-round2-plain/yolo26/fp16/[448, 800]`: one-to-one head excluded by coordinator decision; shipped seed2 fails confuser_far
- `yolo26n-v2-800-round2-plain/yolo26/fp32/[448, 800]`: one-to-one head excluded by coordinator decision; shipped seed2 fails confuser_far; shipped seed2 fails reaction_p95_ms
- `yolo26n-v2-800-round2-plain/yolo26/int8/[448, 800]`: int8 excluded by coordinator decision; one-to-one head excluded by coordinator decision; shipped seed1 fails reaction_p95_ms; shipped seed1 fails export parity; shipped seed1 fails deployed precision accuracy
- `yolo26n-v2s-640-round2-plain/yolo11/fp16/[384, 640]`: shipped seed1 fails confuser_far
- `yolo26n-v2s-640-round2-plain/yolo11/fp32/[384, 640]`: shipped seed1 fails confuser_far
- `yolo26n-v2s-640-round2-plain/yolo11/int8/[384, 640]`: int8 excluded by coordinator decision; shipped seed1 fails export parity; shipped seed1 fails deployed precision accuracy
- `yolo26n-v2s-640-round2-plain/yolo26/fp16/[384, 640]`: one-to-one head excluded by coordinator decision; shipped seed1 fails confuser_far; shipped seed1 fails export parity; shipped seed1 fails deployed precision accuracy
- `yolo26n-v2s-640-round2-plain/yolo26/fp32/[384, 640]`: one-to-one head excluded by coordinator decision; shipped seed0 fails confuser_far
- `yolo26n-v2s-640-round2-plain/yolo26/int8/[384, 640]`: int8 excluded by coordinator decision; one-to-one head excluded by coordinator decision; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2s-800-round2-plain/yolo11/fp32/[448, 800]`: shipped seed1 fails reaction_p95_ms
- `yolo26n-v2s-800-round2-plain/yolo11/int8/[448, 800]`: int8 excluded by coordinator decision; shipped seed0 fails confuser_far; shipped seed0 fails reaction_p95_ms; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy
- `yolo26n-v2s-800-round2-plain/yolo26/fp16/[448, 800]`: one-to-one head excluded by coordinator decision
- `yolo26n-v2s-800-round2-plain/yolo26/fp32/[448, 800]`: one-to-one head excluded by coordinator decision; shipped seed1 fails reaction_p95_ms
- `yolo26n-v2s-800-round2-plain/yolo26/int8/[448, 800]`: int8 excluded by coordinator decision; one-to-one head excluded by coordinator decision; shipped seed0 fails confuser_far; shipped seed0 fails reaction_p95_ms; shipped seed0 fails export parity; shipped seed0 fails deployed precision accuracy

Admissible Pareto front: `['yolo26n-v2s-800-round2-plain/yolo11/fp16/[448, 800]']`.

No measured Pi 5 result exists. The calibration factor and the mapping
from still-image recall to temporal hits remain estimates and assumptions.

Reproduction commands, candidate addition, resource handling and policy
requests are in [WS7-methods.md](WS7-methods.md).

## Package dry run

`yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16-provisional` is provisional. SHA-256 and byte counts verified.
Package source: `outputs/ws7/packages/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16-provisional`. Verified copy: `models/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16-provisional`.
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
