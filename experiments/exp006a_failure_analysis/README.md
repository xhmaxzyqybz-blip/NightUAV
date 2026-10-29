# EXP006A: frozen E15 failure-mode diagnostics

Completed analysis of the8 fixed dev inputs and saved EXP005B E15 PNG outputs. No model load, inference, training, network/loss/weight change or split change. E15 is frozen by user instruction, not selected by this analysis.

Checkpoint SHA256: `73bd2f1a50acc17aebb8c2fbc6e7b585cf8b38cc6b62573e2a2de0058a3f39d0`. Matches EXP005B checkpoint validation record. Input JPEG hashes match split records; regenerated LANCZOS RGB arrays exactly match saved baseline input pixels and recorded RGB hashes. Output PNG hashes are frozen here. Their association with E15 relies on the recorded baseline export; no model re-inference was performed. Mean output Y agrees with baseline float records within the8-bit rounding bound.

## Definitions

All pixel statistics are on1280x720 RGB arrays, Yprime=.2126R+.7152G+.0722B in encoded RGB/255. This is the baseline diagnostic definition, not physical linear luminance. Regions are determined exclusively by input: dark<.2, middle[.2,.7], bright>.7.

Primary saturation is ANY RGB channel equal to255 in saved PNG. Input-existing, output total, newly saturated (~input_sat & output_sat), retained and released saturation are separate counts. New saturation has two denominators: all region pixels and input-unsaturated region pixels. All-white, any-zero, all-black and channelwise endpoint statistics are also recorded separately. PNG255 can reflect rounding as well as clamping; do not label these counts as exact floating preclamp clipping.

Gain distributions include mean/std/min/max/p01/p05/p10/p25/median/p75/p90/p95/p99 and fixed[-1,1] histogram bins. Color metrics normalize RGB by R+G+B and report L1 and signed deltas; exact-zero sums excluded with counts, no epsilon. This removes common positive multiplicative RGB scaling, but is not a perceptual quality metric. HSV S reports color saturation, distinct from numeric endpoint clipping. Near-black color ratios remain sensitive to quantization.

## Pooled region summary

Pixel-weighted across all8 dev images, not per-image arithmetic mean. High saturation means any channel255.

|Region|Pixels|Input Y|Output Y|Gain|Input high %|Output high %|New high / region %|
|---|---:|---:|---:|---:|---:|---:|---:|
|dark|5503469|0.068156|0.244856|0.176700|0.0000|0.0004|0.0004|
|middle|1490772|0.364815|0.542050|0.177235|0.0001|5.1342|5.1341|
|bright|378559|0.859264|0.971289|0.112025|4.3164|97.5716|93.2552|

## Artifacts

- config/run.json: all definitions, frozen checkpoint hash and protected source hashes.
- config/sources.json: original, saved-input, E15-output hashes and float/PNG cross-check.
- metrics/regions.json: all24 image-region records, distributions and transitions.
- metrics/color_changes.json: all24 color-region records.
- metrics/pooled_regions.json: pooled descriptive summary.
- CROP_COORDINATES.md and config/crop_coordinates.json: input-only selection and coordinate mapping.
- [Contact sheet](outputs/index.html):12 input/E15/gain comparisons and4 input overviews.
- REVIEW_QUESTIONS.md: pending human questions, no automatic hypothesis confirmation.

Crops were not selected for output severity. A changed endpoint fraction, gain or color statistic alone does not establish visual quality loss, causal mechanism, H1, or a theoretical defect of DI-Retinex. No innovation module or loss is proposed. Images remain under Git-ignored outputs/. No commit/push. Stop for human review.
