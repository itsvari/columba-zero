# WS4 — Latency benchmarking and reaction-time model

- State: in progress
- Branch: `ws4/latency`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14

## Done

- Read coordinator preamble, PLAN sections 0–6 and WS4, CONTRACTS v1, WS0 status and existing benchmarks.
- Cluster BatchMode connection succeeds; initial queue has no cz- jobs.

## In progress

- Pinned custom NCNN build, Python timing driver and reaction simulation.

## Blockers and requests

- Request C7 optional `samples_ms` for paired end-to-end measurements, build provenance, camera frame dimensions and load average.
- Current Raspberry Pi guide now publishes YOLO26 instead of YOLO11; looking up a versioned official YOLO11 table.

## Results

Pending measurements.

### Execution notes

- Custom Mac build verified as ARM64. CMake itself is an x86 binary, so the build script explicitly sets `CMAKE_OSX_ARCHITECTURES=arm64`.
- Published official YOLO11n Pi5 NCNN FP32 640×640 mean inference is 94.03 ms, excluding pre/post. Versioned source: https://github.com/ultralytics/ultralytics/blob/v8.3.200/docs/en/guides/raspberry-pi.md, accessed 2026-10-14. Uses Ultralytics 8.3.152; NCNN version and thread count unspecified.
- Initial bundled job 945127 failed before running, signal 53, because its Slurm output path assumed `/home/ashelow` instead of the actual `/home/a/ashelow`. Corrected to a submission-directory-relative output path and resubmitted after checking the queue. No benchmark ran in that failed job.
