# EXP005A-0: DI-Retinex-U3D 10-epoch adapted baseline pilot

Completed exactly 10 epochs, 770 images per epoch, 7700 updates. STOPPED for human research review; no extension, early stopping or best checkpoint selection.

Official DI-Retinex dataloader uses full images. U3D originals are 3840x2160; this project uses 256x256 paired random patches after the original full-image Gaussian processing and conversion. Therefore this is an adapted baseline, not strict paper reproduction and not an innovation experiment.

Original network and RD+VS 1:1, random official initialization, seed1149, Adam lr0.001, weight decay0.0001, clipping0.1, exposure0.5 (original Normal std0.001), batch1. No old/pretrained checkpoint loaded. All 770 train paths traversed once every epoch; dev8 and quarantine22 excluded. Loader settings and source/split/script hashes: config/run.json. Benchmark candidates, skips, equivalence and throughput: metrics/loader_benchmark.json. Only tested configurations qualify for selection; skipped RAM-heavy configurations are not claimed slower.

Dev0..10: full eight RGB images resized with LANCZOS to 1280x720, no crop. Eval/no_grad, parameters and training RNG unchanged; train mode restored. Raw output reconstructed and checked against official clamp. Ratios count channels. Luminance is encoded RGB Yprime, not linear-light. Unsupervised dev objective uses original RD/VS on resized inputs and their full Gaussian blur, fixed per-image exposure sample via isolated RNG for epoch comparability. It is NOT an image quality score or GT metric.

Epoch0/1/2/5/10 images and eight input|epoch0|epoch1|epoch2|epoch5|epoch10 montages are Git ignored under results/demos/di_retinex_exp005a0/. No automatic preference/ranking.

Checkpoints epoch001/002/005/010 under checkpoints/di_retinex/exp005a0/ are pilot-analysis-only; weights_only and strict state loading plus exact forward match verified. No best.pth.

Metrics include per-epoch average total/RD/VS, global rolling50 step losses, per-step image/crop, finite gradients and parameter updates. Synchronized GPU-phase wall timing excludes separate finite/parameter-delta checks. Loader wait is main-process blocking time; with workers it is not total CPU work. Timing scopes and startup included in benchmark are explicit; do not infer whole-training speedup from loader-only throughput. Reserved CUDA memory includes cached dev allocations. No training objective adaptation or augmentation changes were made.

This small dev set and limited scene metadata cannot establish universal scene independence or final generalization. Ten epochs do not establish convergence. Human review is required before choosing any longer protocol. No commit/push.

## Actual epoch summaries

|Epoch|Mean total|Mean RD|Mean VS|Train seconds|Mean wait ms|
|---|---:|---:|---:|---:|---:|
|1|0.02591477|0.02567744|0.00023733|279.073|327.981|
|2|0.02225981|0.02211409|0.00014572|268.226|314.931|
|3|0.02433569|0.02425749|0.00007821|279.012|328.946|
|4|0.02654704|0.02647659|0.00007044|293.039|347.021|
|5|0.02191821|0.02188753|0.00003067|927.185|1101.303|
|6|0.02596639|0.02592935|0.00003704|965.172|1160.964|
|7|0.02522386|0.02519830|0.00002556|18115.472|1793.219|
|8|0.02481236|0.02479027|0.00002208|132622.972|2054.842|
|9|0.02997650|0.02995451|0.00002198|5687.933|1279.919|
|10|0.02788645|0.02786874|0.00001771|888.167|1068.540|

Final execution summary: metrics/summary.json. No best epoch conclusion.

## Post-run timing audit

IMPORTANT: Long stalls contaminate wall-clock phase timing; the recorded approximately 44.54 hours is not an estimate of continuous active training cost. See [completion review](COMPLETION_REVIEW.md) and metrics/completion_audit.json. Raw records are unchanged.
