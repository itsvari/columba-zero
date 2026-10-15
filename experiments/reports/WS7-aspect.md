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
