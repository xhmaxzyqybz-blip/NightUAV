# EXP004C train/dev integration smoke test

{
  "status": "EXP004C: PASS",
  "steps": 200,
  "unique_train_images": 200,
  "dev_evaluations": 5,
  "dev_forwards": 40,
  "saved_images": [
    "results/demos/di_retinex_exp004c/step000/0049.png",
    "results/demos/di_retinex_exp004c/step000/0114.png",
    "results/demos/di_retinex_exp004c/step000/0166.png",
    "results/demos/di_retinex_exp004c/step000/0179.png",
    "results/demos/di_retinex_exp004c/step000/0216.png",
    "results/demos/di_retinex_exp004c/step000/0343.png",
    "results/demos/di_retinex_exp004c/step000/0758.png",
    "results/demos/di_retinex_exp004c/step000/0792.png",
    "results/demos/di_retinex_exp004c/step200/0049.png",
    "results/demos/di_retinex_exp004c/step200/0114.png",
    "results/demos/di_retinex_exp004c/step200/0166.png",
    "results/demos/di_retinex_exp004c/step200/0179.png",
    "results/demos/di_retinex_exp004c/step200/0216.png",
    "results/demos/di_retinex_exp004c/step200/0343.png",
    "results/demos/di_retinex_exp004c/step200/0758.png",
    "results/demos/di_retinex_exp004c/step200/0792.png"
  ],
  "checkpoint_safe_strict_reload": true,
  "reload_forward_exactly_matches": true,
  "checkpoint_sha256": "fad60b2bfa80b8a5ae60abe144c40167c69d3b771ed1b6f8856ba377c6be581d",
  "train_dev_mixing": false,
  "nan_inf": false,
  "oom": false,
  "input_and_source_hashes_unchanged": true,
  "loop_seconds_including_dev50_200": 294.6012750000227,
  "train_compute_seconds": 17.080021598841995,
  "train_data_seconds": 244.55920969927683,
  "mean_data_preprocess_ms": 1222.7960484963842,
  "mean_transfer_ms": 1.5410320018418133,
  "first_train_compute_ms": 3107.322000199929,
  "remaining_mean_compute_ms": 70.21457084744758,
  "train_peak_allocated_mib": 1205.72705078125,
  "train_peak_reserved_mib": 2490.0,
  "dev_peak_allocated_mib": 689.19580078125,
  "dev_peak_reserved_mib": 2490.0,
  "loss_first": 0.0026075474452227354,
  "loss_last": 0.0003375462838448584,
  "loss_first20_mean": 0.016139983441098592,
  "loss_last20_mean": 0.01629590475349687
}

U3D v2.1: train 770, dev 8, quarantine 22. 200 steps, batch 1, train crop 256x256, random initialization seed 1149. Official network and AST-extracted RD:VS=1:1 unchanged. Adam lr .001, wd .0001, clip .1. Full-image original loader preprocessing precedes paired random crop; no caching optimization. All 770 entries are eligible; a shuffled permutation supplies 200 different source images.

Dev is evaluation only at 0/50/100/150/200, eval plus no_grad, no backward or optimizer calls, no crop or random operations; model parameters and CPU/CUDA RNG verified unchanged, then train mode restored. Full dev images are RGB LANCZOS resized to 1280x720, NOT native 4K evaluation. Raw preclamp expression reconstructed from official returned k/b and verified against clamped official output. Channel fractions are not pixel fractions. Luminance statistics are encoded RGB Yprime, not linear-light. No GT, PSNR, SSIM, or image-quality score. No unsupervised dev objective computed.

Only step000 and step200 images saved (8 each); Git ignored. Checkpoint: checkpoints/di_retinex/exp004c/step200.pth, independent integration-only storage. Rebuilt fresh model, weights_only=True, strict=True, exact state and CUDA dev output match. NEVER inherit this checkpoint for formal baseline training: reinitialize randomly.

Timing: synchronized wall times for forward+loss, backward, clip+optimizer separately; diagnostic finite checks/parameter differences are outside these times. Data timing covers full decode, blur, conversion and crop; GPU transfer is separate. Loader workers=0. Train reserved memory can include cache retained from preceding dev evaluation; allocator peaks are not process-wide GPU usage. First training step is separate from remaining steps. Loop time includes four in-loop dev evaluations and logging, excludes initial step0 evaluation, startup and checkpoint verification.

This is NOT formal baseline training, NOT full paper reproduction, NOT an innovation experiment and cannot support final generalization conclusions. Official val/test and quarantine never enter loaders. Current dev split remains limited by 8 representatives and incomplete scene metadata. Technical integration success does not determine full training schedule, early stopping or convergence.

Reproduction command and exact source/split/script hashes: config/run.json. Per-step samples, losses, gradients, update checks and timing: metrics/train_history.json; dev raw ranges and luminance: metrics/dev_history.json. No commit or push.
