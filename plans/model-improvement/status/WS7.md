# WS7 — Export, quantisation and selection

- State: in progress
- Branch: `ws7/export`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-15

## Done

## In progress

Contracts v1.7 read. Building rectangular C8 export, validation-only NCNN
calibration and repeatable selection/evaluation commands. Initial matrix uses
all three seeds of yolo26n-v2s-800, yolo26n-v2-800 and yolo26n-v2s-640,
round2-plain. Both YOLO26 heads and fp16/int8 are included. Accuracy uses
columba_native package-default compute on xcng, with fresh val thresholds.
Final selection waits for distilled and species candidates. Packaging is a
provisional dry run only. WS5 jobs are read-only and no GPU is planned.

## Blockers and requests

## Results

<!-- Label numbers as measured / estimated / simulated, and name the host. Link artifacts by path. -->
