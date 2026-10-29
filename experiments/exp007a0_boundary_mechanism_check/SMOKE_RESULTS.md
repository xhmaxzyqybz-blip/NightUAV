# Approved 200-step smoke test

PASS: 200 effective updates, fresh initialization seed1149; no checkpoint loaded or saved. No dev evaluation or lambda selection. Lambda fixed to 0.053192434320470425. No continuation.

Original network/RD/VS unchanged; observed original clamp input z used by the only additional squared upper penalty. Every step clamp equality verified; lambda0 forward/backward equality was established in the preceding gradient audit. Positive lambda intentionally changes the total gradient and trajectory.

Upper gradients measured every step; baseline comparison at1,10,...200 before clipping. For the same parameter count, the L2 norm ratio equals per-parameter-element RMS ratio. Per-step ratios and pooled-across-samples RMS ratio are distinct. autograd.grad monitoring does not accumulate into .grad. Total backward occurs once per update.

Timing separates data wait, transfer, forward/backward/optimizer, extra gradient-monitor overhead; elapsed also contains finite checks and logging. Memory includes monitoring tensors.

All200 different images come from the770-entry train manifest; no dev/quarantine/official val/test. Crop protocol is reused unchanged. A zero resample count would mean no rejection occurred, not new proof of the retry branch.

See config/smoke_run.json, metrics/smoke_steps.jsonl, metrics/smoke_summary.json and metrics/smoke_crop_records.json.

## 50-step descriptive windows

|Steps|Total|RD|VS|Upper|Weighted upper|z>1 channels|
|---|---:|---:|---:|---:|---:|---:|
|[1, 50]|0.02391187|0.02331594|0.00049087|0.00197508|0.00010506|3.399994%|
|[51, 100]|0.03703224|0.03666571|0.00034924|0.00032508|0.00001729|2.384115%|
|[101, 150]|0.01825370|0.01762771|0.00049498|0.00246299|0.00013101|2.956818%|
|[151, 200]|0.02521484|0.02439053|0.00066981|0.00290442|0.00015449|3.557800%|

Different windows contain different training images/crops; changes do not establish a fixed-input improvement or superiority over baseline. Loss fluctuations do not establish convergence. Gradient ratios may drift substantially from the initialization calibration.

Engineering smoke passed only. No formal15-epoch run, best selection, commit or push. Await human review.
