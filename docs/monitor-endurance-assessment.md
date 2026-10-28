# Offline monitor endurance assessment

`tools/assess_monitor_endurance.py` reads a completed run and writes an independent timing and stability assessment to a new directory. It does not contact the Pi, operate a browser, run inference or score centring. Global acceptance remains inconclusive. The pinned policy SHA-256 is checked before assessment; the policy's frozen truth files are hashed as a prerequisite, not run.

Run from the repository root with the project's environment active:

```sh
python tools/assess_monitor_endurance.py \
  --run-directory /path/to/app/run \
  --telemetry /path/to/telemetry.jsonl \
  --supervisor /path/to/manifest.json \
  --full-off /path/to/full-off.json \
  --browser /path/to/case-browser.jsonl \
  --baseline /path/to/pre-run-voltage.json \
  --voltage-final /path/to/final-voltage.json \
  --video /path/to/unannotated.mp4 \
  --decode-proof /path/to/decode-proof.json \
  --policy-sha256 HASH_FROM_THE_FROZEN_RUN_PLAN \
  --output-directory /path/to/new-assessment
```

The baseline and final voltage receipts each contain `monotonic_s`, `boot_id` and `throttled` (an integer or the `throttled=0x...` string). A clear pre-capture baseline and clear final latch, on the same boot, cover the interval between them. An asserted current or latch bit observed in capture fails. Missing baseline, mismatched boots or missing final receipt cannot establish that the latch remained clear. The final receipt must be at or after the last processed frame; retain the raw receipts as evidence.

The decode proof is a separately produced JSON record with `success`, `independent`, and `video_sha256`. It must record an actual independent full decode of the supplied video; the tool checks the video hash but cannot establish the truth of the proof's claims. An absent or incomplete proof does not establish a pass. The manifest hashes all supplied inputs and the generated assessment; the manifest itself is not self-hashed. Input files are never changed.

The first and last `captured_at` values define the capture span. The supervisor start clock pair converts Unix capture timestamps into the Pi monotonic domain for comparison with controller ticks and telemetry. The conversion assumes a stable relationship between those clocks; review clock changes separately. Controller first-use deduplication reconstructs `tick_started_at_monotonic_s - observation_age_ms / 1000`, with a one-microsecond tolerance. Uncensored age includes tracking and stale ticks; stale time uses the preceding state over the actual tick interval. Frames are counted in half-open intervals. The warm-up is 120 seconds; 600-second windows follow. A final window shorter than 60 seconds is reported without its own verdict, while its samples remain in whole-hot and whole-capture checks.

RSS uses capture samples in the first and last 600 seconds after warm-up; export samples are excluded. Temperature and clock are descriptive capture-only ranges, not thermal certification. A sampling gap over twice the sidecar’s declared interval makes hot telemetry coverage inconclusive; this is an evidence-completeness rule, not an additional hardware performance gate. It cannot prove that every scheduled sample was taken. A short run cannot establish the memory comparison. A read-only matrix archive is useful for format checking but does not satisfy the one-hour or 18-repeat policy. Missing or malformed input files prevent an assessment; absent measurements and failed integrity checks prevent a pass.
