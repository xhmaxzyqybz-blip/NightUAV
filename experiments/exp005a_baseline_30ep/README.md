# EXP005A: 15-epoch adapted baseline (STOPPED INCOMPLETE)

Target: 15 epochs. Actual: 12 complete epochs plus 154 steps of E13 (9,394 logical optimizer updates). No E15 checkpoint or E15 result exists. Training stopped at E13 step155 before forward/backward. Do not treat this run as a completed 15-epoch baseline.

## Failure

The predetermined crop from datasets/U3D/lowlight/train/0069.jpg at x=68,y=1850,w=h=256 is entirely zero in RGB and in the official Gaussian image. The input-positive-mean guard stopped execution. Original RD uses exposure / input_mean, which has a zero denominator on this crop. No epsilon, skip, resampling, loss change or retry was applied. Source SHA256 matches the split manifest. See metrics/failure_diagnosis.json and failure.json.

## Fixed protocol

Original DI-Retinex network and RD:VS=1:1; fresh random initialization, seed1149, Adam lr0.001, weight_decay0.0001, clipping0.1, exposure0.5 (official Normal std0.001), batch1. Train770/dev8/quarantine22, U3D v2.1. Original official loader processes full images; this project applies paired 256x256 random crops after full-image Gaussian processing. This is an adapted baseline, not strict paper reproduction. No pilot or pretrained weights were used.

## Budget change and traceability

The user reduced the running budget from 30 to 15. Directory name and original config/run.json are preserved for history; config/budget_amendment_15ep.json is the effective cap. The original process stopped after validated E10 checkpoint and two logged E11 steps. The continuation restored this same experiment E10 model/Adam and reconstructed sampler/crop/RNG states. Both replayed E11 steps matched exactly in source, crop, loss, gradient norm and parameter update. See config/continuation_replay_verified.json. No pilot state was loaded. Original metrics/train_steps.jsonl is preserved; train_steps_15ep.jsonl contains the canonical 9,394 completed updates.

## Dev and outputs

E0 through E12: all 8 fixed dev RGB inputs resized with LANCZOS to1280x720, eval/no_grad, no parameter or training RNG update. No dev or quarantine sample trained. Record raw min/max/<0/>1 channel ratios, clamp saturation, encoded luminance mean/median/p95, and original unsupervised dev objective (not quality or GT metric). Fixed input Y<0.2 / Y>0.7 masks define diagnostic gains: count, mean, median, p10,p90, on clamped output. All definitions unchanged.

E0/E1/E2/E5/E10 images and eight partial montages exist under results/demos/di_retinex_exp005a/. E15 is absent, not fabricated. Checkpoints E1/E2/E5/E10 independently passed weights_only=True, strict=True, exact state and finite GPU dev forward. E15 is absent. No best checkpoint selected.

## Actual full-epoch statistics

|Epoch|Total|RD|VS|Train seconds|
|---|---:|---:|---:|---:|
|1|0.02591477|0.02567744|0.00023733|548.74|
|2|0.02225981|0.02211409|0.00014572|277.96|
|3|0.02433569|0.02425749|0.00007821|266.24|
|4|0.02654704|0.02647659|0.00007044|272.44|
|5|0.02191821|0.02188753|0.00003067|274.99|
|6|0.02596639|0.02592935|0.00003704|289.31|
|7|0.02522386|0.02519830|0.00002556|276.78|
|8|0.02481236|0.02479027|0.00002208|303.72|
|9|0.02997650|0.02995451|0.00002198|278.01|
|10|0.02788645|0.02786874|0.00001771|345.86|
|11|0.02457921|0.02453504|0.00004417|281.09|
|12|0.02394671|0.02388429|0.00006242|276.56|

See metrics/stopped_summary.json for timing, memory and checks; curves/README.md for measured full-epoch diagnostic plots. E13 partial records are retained but excluded from complete-epoch means and dev curves. No NaN/Inf was recorded in completed losses/gradients/dev outputs; the failed step stopped before its objective was calculated. The input guard failure is not an observed optimizer NaN or OOM.

Human protocol review is needed for all-black training crops. No training is currently running. No automatic retry, extension, best selection, commit or push. The small dev set and incomplete scene metadata remain limitations.
