# WS5 round 3: species gate (PLAN section 5.4)

Status on 16 Oct 2026. All 3 species seeds are DONE (array 950409, Titan V).

**Training-size decision (provisional): do not ship species.** The species
model passes the recall and latency clauses but fails the strict "false-alarm
rates no worse" clause by 1–2 images on test-v2. WS7 makes the final call on
deploy-size native exports, with the same rule (`tools/select_model.py`,
`enforce_species_gate`). The seed WS7 should export is given below.

## Setup

- **Single arm:** `yolo26n-v2s-800-round2-plain`. 3 seeds on H100 NVL / MIG
  3g.47gb / A100-80, batch 64 × accumulate 1.
- **Species arm:** `yolo26n-v2s-species-800-r3`. The same recipe on
  `columba-train-v2s-species` (same images, 11-class labels), taxonomy
  `species`, 3 seeds on Titan V. `batch_override: 16` × accumulate 4 keeps
  the effective batch (64) and weight-decay scaling equal; the per-step
  BatchNorm batch and GPU type differ.
- **Scoring:** pigeon-only on the frozen `columba-test-v2`. The runner
  aligns the staged test names to the species list
  (`align_test_taxonomy`; `run.json.test_taxonomy_aligned`). Pigeon metrics
  and both false-alarm rates use only class-0 predictions plus the manifest,
  so they are directly comparable. The operating threshold is chosen on val
  and reused on test.
- All numbers are **measured** by WS1's evaluator on the cluster, square at
  800 px. They are context for WS7's deploy-size scoring, not promotion
  numbers.

## Accuracy and false alarms (3-seed mean ± sd)

| Metric | Single (yolo26n-v2s-800) | Species (yolo26n-v2s-species-800) | Δ species − single |
|---|---|---|---:|
| val R@P90 | 0.794 ± 0.006 | **0.818 ± 0.004** | +0.024 |
| test R@P90 | 0.780 ± 0.024 | **0.791 ± 0.013** | +0.011 |
| test small R@P90 | 0.537 ± 0.051 | 0.551 ± 0.025 | +0.014 |
| test AP50 | 0.860 ± 0.017 | 0.875 ± 0.012 | +0.015 |
| test P / R at val threshold | 0.924 / 0.748 | 0.923 / 0.766 | – / +0.018 |
| val empty FAR | 0.003 ± 0.003 | 0.000 ± 0.000 | −0.003 |
| val confuser FAR | 0.038 ± 0.019 | 0.037 ± 0.009 | −0.001 |
| test empty FAR (FAs of 125) | 0.000 (0, 0, 0) | **0.008 (1, 1, 1)** | +0.008 |
| test confuser FAR (FAs of 73) | 0.032 (3, 1, 3) | **0.041 (3, 3, 3)** | +0.009 |

Per seed:

| Seed | Single val R | Single test R / AP50 | Single test FAs empty / confuser | Species val R | Species test R / AP50 | Species test FAs empty / confuser |
|---|---:|---|---|---:|---|---|
| 0 | 0.791 (val confuser 0.060) | 0.779 / 0.879 | 0 / 3 | 0.816 | 0.796 / 0.863 | 1 / 3 |
| 1 | **0.801** | 0.805 / 0.858 | 0 / 1 | 0.816 | 0.776 / 0.873 | 1 / 3 |
| 2 | 0.790 | 0.756 / 0.845 | 0 / 3 | **0.822** | 0.802 / 0.888 | 1 / 3 |

## Section 5.4 clauses (training size)

| Clause | Result |
|---|---|
| 3-seed mean pigeon R@P90 ≥ single − 0.01 | **pass**: test 0.791 vs 0.780 (−0.01 bar 0.770); val 0.818 vs 0.794 |
| False-alarm rates no worse | **fail** on test (strict). The mean empty FAR is 0.008 vs 0.000 (1 street scene per seed), and the mean confuser FAR is 0.041 vs 0.032. Shipped-seed view (WS7's rule): species s2 has 1 / 3 against single s1's 0 / 1. On **val**, both FARs are no worse (0.000 vs 0.003; 0.037 vs 0.038). |
| Estimated Pi 5 p95 ≤ 5% slower | **pass**: +1.9% (47.7 → 48.6 ms at 448×800; xcng0 job 950426, `experiments/latency/ws5-species/`) |
| Section 1 vs `baseline-repro-p50` | pass: 0.791 vs 0.246 ± 0.035 |

**Decision recorded:** species is **not shipped** at training size. Ship the
single-class model, which already uses the species images as unlabelled
negatives (that is what v2s single is). This is provisional until WS7's
deploy-size gate. If WS7's exported species seed meets the FAR clause at
448×800, WS7's rule decides.

### What the false alarms are

Images found by thresholding each seed's test predictions at its val
threshold:
- **Empty scenes:** species fires on 1 COCO street scene per seed
  (`coco-152465` in s0 and s2, `coco-499768` in s1). Single fires on none.
- **Confusers:**
  - Single's false alarms are true pigeon look-alikes: pink-necked green
    pigeon twice in s0, spotted dove in s0 and s2.
  - Species never fires on the pink-necked green pigeon. Its false alarms
    are a sparrow (all seeds), a Javan myna (all seeds) and a dove (one per
    seed).
  - So explicit species labels remove the green-pigeon confusion but not
    the small-brown-bird ones.

The FAR differences are 1–2 images, at the resolution limit of the gate (1
confuser false alarm = 0.0137; 1 empty false alarm = 0.008). The strict
clause still decides.

## Seed WS7 should export (val only)

Rule (PLAN section 1, 15 Oct decision): the highest val R@P90 among seeds
whose val FARs pass (empty ≤ 0.02, confuser ≤ 0.05).

- **Species:** `yolo26n-v2s-species-800-r3/seed2`. Val R@P90 0.822; val
  empty 0.000, confuser 0.037.
- **Matched single reference:** `yolo26n-v2s-800-round2-plain/seed1`. Val
  R@P90 0.801; val empty 0.000, confuser 0.032. Seed 0 is ineligible (val
  confuser 0.060).

Use these with `select_model.py --taxonomy species --single-recipe
yolo26n-v2s-800-round2-plain`.

## Confound and a possible follow-up (proposal, not started)

The species gain (+0.024 val, every species seed above every single seed)
comes with a different per-step batch and GPU: 16 on Titan V vs 64 on
H100/A100. Part of the gain may come from the smaller BatchNorm batch rather
than the taxonomy. A control run would separate the two:
- `yolo26n-v2s-800-round2-plain` with `batch_override: 16` on Titan V,
  3 seeds;
- ≈ 160 s/epoch × 110–150 epochs ≈ 5–7 h each, so ≈ 17 nv GPU-h on
  otherwise idle Titan Vs.

If single@16 matches species on recall, smaller batches help the single
model without the species FAR cost. I recommend it but have **not**
submitted it; it needs coordinator approval.
