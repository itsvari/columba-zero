# WS7 int8 quantisation study

Date: 16 October 2026. Branch: `ws7/quant`, based on local main `e6e12be`.
Status: blocked before experiments.

## Connection failure

The required initial check failed with exit status 255:

```text
$ ssh -o BatchMode=yes columba-cluster true
ssh: connect to host 127.0.0.1 port 22022: Connection refused
```

The coordinator explicitly instructed WS7 to stop if this check fails.
No retry or alternative connection was attempted. No jobs were submitted,
and no password was requested, used or stored.

## Results available from this study

None. KL reproduction, calibration preprocessing inspection, per-layer
quantisation inspection and all accuracy and latency measurements remain
outstanding. The historical fp16 and KL int8 Pi p95 estimates supplied in
the brief are 36.54 ms and 33.39 ms respectively. These are reference values,
not new measurements or evidence that an alternative PTQ configuration works.

No recovery claim or QAT recommendation can yet be supported. QAT has not
started. The deploy manifest, application config and coordinator's int8
exclusion remain unchanged. The `ws7/export` branch was not modified.

## Resume order

1. Restore the tunnel, then run
   `ssh -o BatchMode=yes columba-cluster true`. Stop again if it fails.
2. Read the current PLAN sections 0 and 6, contracts, WS7 quantisation tool,
   WS4 latency service and existing plain-800 seed1 fp32/KL evidence.
3. Reproduce KL on xcng ARM CPUs and audit preprocessing and quantised layers
   before comparing calibration methods or mixed precision.
4. Compare KL, ACIQ and EQ with 200 and approximately 1,000 val images.
   Preserve the frozen-test SHA exclusion and audit calibration provenance.
5. Run a sensitivity scan and a few mixed-precision configurations. Score
   each with a fresh val threshold and measure ACTIVE-policy latency using
   the WS4 service. Stop tuning variants without a clear speed advantage
   over fp16. Use no GPUs and no compute on login nodes.
6. Report val/test R@P90, AP50, operating recall, empty/confuser FAR and
   estimated Pi p95 against the same-head rectangular PyTorch reference.
   Document NCNN granularity limits without adding custom kernels.
7. If the best faster PTQ variant still loses more than 0.01 R@P90 against
   fp16, write the requested QAT proposal only. Do not start QAT.

No study jobs are in flight. Resuming requires a working BatchMode connection;
there is no pending model-selection or deployment decision from this study.

## Verification

This commit changes documentation only. `git diff --check` passes.
Tests and lint were not rerun; no code, model or deployment config changed.
