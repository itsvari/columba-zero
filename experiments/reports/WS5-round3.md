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
model without the species FAR cost.

**Control (coordinator-approved and submitted 16 Oct; result below):**
`training/recipes/yolo26n-v2s-800-r3-control-b16.yaml`.
- **Why:** batch size, BatchNorm statistics and GPU type confound the
  species-vs-single comparison. The control removes them.
- **Recipe:** identical to `yolo26n-v2s-800-round2-plain` except
  `batch_override: 16`. It differs from the species arm only in
  dataset/taxonomy (`columba-train-v2s`, single).
- **Jobs:** seeds 0/1/2 as arrays 952063 / 952064 / 952065, pinned to Titan
  V.
- **Interpretation, fixed before results:**
  - If the control's 3-seed mean test R@P90 is within the larger seed sd
    of species, the species recall gain is attributed to batch size. The
    single b16 arm then becomes a WS7 candidate at deploy size, under the
    same rules.
  - Otherwise species keeps its recall advantage, and the decision rests
    on WS7's deploy-size false-alarm results.

## Control result (16 Oct): species recall gain attributed to batch size

`yolo26n-v2s-800-r3-control-b16`. 3 seeds on Titan V, batch 16 ×
accumulate 4, single taxonomy on `columba-train-v2s`. Jobs 952063–952065,
refilled after requeue as 954413 / 954417 / 954418 (and 955061 / 955062).
Measured, training size, 800 px.

| Metric | Control b16 (single) | Species b16 | Plain b64 (single) |
|---|---|---|---|
| val R@P90 | 0.798 ± 0.018 (0.808, 0.778, 0.808) | 0.818 ± 0.004 | 0.794 ± 0.006 |
| test R@P90 | **0.790 ± 0.014** (0.790, 0.776, 0.805) | **0.791 ± 0.013** | 0.780 ± 0.024 |
| test small R@P90 | 0.549 ± 0.024 | 0.551 ± 0.025 | 0.537 ± 0.051 |
| test AP50 | 0.878 ± 0.011 | 0.875 ± 0.012 | 0.860 ± 0.017 |
| test AP50-95 | 0.520 ± 0.005 | 0.519 ± 0.009 | 0.511 ± 0.008 |
| test P / R at val threshold | 0.916 / 0.777 | 0.923 / 0.766 | 0.924 / 0.748 |
| test empty FAs (of 125) | 0, 0, 0 | 1, 1, 1 | 0, 0, 0 |
| test confuser FAs (of 73) | 3, **4**, 3 | 3, 3, 3 | 3, 1, 3 |

**Pre-registered rule, applied verbatim:**
- **Rule:** "If the control's 3-seed mean test R@P90 is within the larger
  seed sd of species, the species recall gain is attributed to batch size."
- **Applied:** |0.790 − 0.791| = 0.001 ≤ max(0.014, 0.013) = 0.014, so
  **within**. The species gain is **attributed to batch size**, and
  single-class b16 **becomes a WS7 candidate** under the same rules.
- On val, species stays about 0.02 higher (0.818 vs 0.798). Control seed
  1 is a low outlier on val (0.778). The rule is defined on test, though,
  and test agrees with the control on every accuracy metric.

**Control seed for WS7 (val only):** `yolo26n-v2s-800-r3-control-b16/seed2`.
- Seeds 0 and 2 tie exactly on val R@P90 (0.808399, the same box count).
- Deterministic val-only tie-break, in order:
  1. higher val R@P90;
  2. lower val confuser FAR;
  3. lower val empty FAR;
  4. higher val AP50;
  5. lower seed index.
- Step 2 decides it: s2 val confuser 0.032 vs s0 0.041. s2's val empty FAR
  is 0.000, its val operating threshold 0.575.
- Test, for reference only: R@P90 0.805, 0 / 3 false alarms.
- Control seed 1 breaches the test confuser gate (4 of 73). It is not the
  val-chosen seed.

**Effect on the species decision:** with the recall gain explained by batch
size, species no longer has a recall advantage over a matched single model.
It still has the extra false alarms (1 empty scene per seed). So the
training-size decision stands: **do not ship species**. WS7's deploy-size
scoring of `species/seed2` against its single reference makes the final
call.

The training-size data also says batch 16 (on Titan V) helps single-class
recall slightly (+0.010 test, within noise) and AP50 (+0.018). The cause
cannot be separated further: batch, BatchNorm and GPU type all change
together.
