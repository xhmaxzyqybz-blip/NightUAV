# EXP005B: DI-Retinex-U3D 15-epoch adapted baseline

Completed exactly 15 epochs, 770 images per epoch, 11550 updates. STOPPED for human research review; no extension, early stopping or best checkpoint selection.

Official DI-Retinex dataloader uses full images. U3D originals are 3840x2160; this project uses 256x256 paired random patches after the original full-image Gaussian processing and conversion. Therefore this is an adapted baseline, not strict paper reproduction and not an innovation experiment.

Original network and RD+VS 1:1, random official initialization, seed1149, Adam lr0.001, weight decay0.0001, clipping0.1, exposure0.5 (original Normal std0.001), batch1. No old/pretrained checkpoint loaded. All 770 train paths traversed once every epoch; dev8 and quarantine22 excluded. Loader settings and source/split/script hashes: config/run.json. Loader configuration is inherited from the pilot benchmark referenced by hash in config/run.json; no prior weights are used. Only tested configurations qualify for selection; skipped RAM-heavy configurations are not claimed slower.

Dev0..15: full eight RGB images resized with LANCZOS to 1280x720, no crop. Eval/no_grad, parameters and training RNG unchanged; train mode restored. Raw output reconstructed and checked against official clamp. Ratios count channels. Luminance is encoded RGB Yprime, not linear-light. Unsupervised dev objective uses original RD/VS on resized inputs and their full Gaussian blur, fixed per-image exposure sample via isolated RNG for epoch comparability. It is NOT an image quality score or GT metric.

Epoch0/1/2/5/10/15 images and eight input|epoch0|epoch1|epoch2|epoch5|epoch10|epoch15 montages are Git ignored under results/demos/di_retinex_exp005b/. No automatic preference/ranking.

Checkpoints epoch001/002/005/010/015 under checkpoints/di_retinex/exp005b/ are unranked adapted baseline checkpoints; weights_only and strict state loading plus exact forward match verified. No best.pth.

Metrics include per-epoch average total/RD/VS, global rolling50 step losses, per-step image/crop, finite gradients and parameter updates. Synchronized GPU-phase wall timing excludes separate finite/parameter-delta checks. Loader wait is main-process blocking time; with workers it is not total CPU work. Timing scopes and startup included in benchmark are explicit; do not infer whole-training speedup from loader-only throughput. Reserved CUDA memory includes cached dev allocations. No training objective adaptation or augmentation changes were made.

This small dev set and limited scene metadata cannot establish universal scene independence or final generalization. Fifteen epochs do not establish convergence. Human review is required before choosing any longer protocol. No commit/push.

## Actual epoch summaries

|Epoch|Mean total|Mean RD|Mean VS|Train seconds|Mean wait ms|
|---|---:|---:|---:|---:|---:|
|1|0.02591477|0.02567744|0.00023733|286.746|338.284|
|2|0.02225981|0.02211409|0.00014572|438.766|535.795|
|3|0.02433569|0.02425749|0.00007821|267.231|314.250|
|4|0.02654704|0.02647659|0.00007044|283.054|334.348|
|5|0.02191821|0.02188753|0.00003067|272.241|320.409|
|6|0.02596639|0.02592935|0.00003704|272.818|321.486|
|7|0.02522386|0.02519830|0.00002556|276.330|325.998|
|8|0.02481236|0.02479027|0.00002208|267.429|314.295|
|9|0.02997650|0.02995451|0.00002198|615.307|765.612|
|10|0.02788645|0.02786874|0.00001771|339.966|405.230|
|11|0.02457921|0.02453504|0.00004417|378.741|454.822|
|12|0.02394671|0.02388429|0.00006242|365.086|437.502|
|13|0.02831725|0.02828884|0.00002841|362.602|434.347|
|14|0.02879205|0.02874111|0.00005094|361.659|433.019|
|15|0.02638525|0.02633893|0.00004632|273.600|322.229|

Strict-zero crop protocol: initial crop plus at most 50 same-image replacement draws via unchanged scripts/nightuav_crop_adapter.py. Only exactly all-zero RGB is rejected; no threshold or epsilon. Crop/Gaussian coordinates stay paired. Separate controlled replacement RNG, no sample skip or substitution. See metrics/crop_resampling.json and per-step records for every invalid coordinate and retry count. This data-adaptation rule does not change network or losses.

Dark/bright gain diagnostics use fixed resized input masks Y_in<0.2 and Y_in>0.7, with clamped output Y_out. Counts and mean/median/p10/p90 gains are recorded per image and epoch; empty masks yield null gains. These preregistered thresholds are diagnostics only, not model/loss changes. Fixed input RGB hashes and mask counts are in config/fixed_dev_inputs.json.

Final execution summary: metrics/summary.json. No best epoch conclusion.
