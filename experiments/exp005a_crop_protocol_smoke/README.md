# EXP005A strict-zero crop protocol smoke

PASS: 200 steps only, fresh random official initialization. No previous checkpoint loaded. No dev evaluation or model selection. No checkpoint saved or formal 15-epoch training started.

## Locked protocol

256x256 paired crops AFTER original full-image Gaussian processing. Reject ONLY if every RGB channel is exactly zero; draw replacements from the SAME image using controlled RNG. Initial attempt plus at most 50 replacement attempts. A valid 50th replacement is accepted; 50 invalid replacements raise BlackCropRetryError with source path and all coordinates. No epsilon, brightness threshold, alternate image or skipped sample. Gaussian crop uses the same accepted coordinates.

The adapter is scripts/nightuav_crop_adapter.py. Initial-coordinate, image-order and replacement RNGs are separate Random(1149) streams. Single-process loading avoids worker-dependent RNG. Old training scripts are preserved; future baseline must explicitly integrate this adapter and record the protocol version.

## Training

Original network and AST-extracted official RD+VS 1:1; Adam lr0.001, weight_decay0.0001, clip0.1, exposure0.5, batch1, seed1149. First 200 images of a shuffled permutation of train770; dev8/quarantine22 unchanged and unused. GPU, sources, hashes, commands and definitions are in config/run.json. Per-step loss, gradient, update and timing: metrics/train_steps.jsonl.

## Tests

Each high-risk image received 1000 independent requested crops with same-image retries, plus an exact deterministic replay. All returned crops were nonzero. Synthetic patches with a single 1/255 or 1e-12 channel value were accepted immediately. An all-zero image exhausted exactly 50 replacements and raised with its path; a controlled sequence succeeded on replacement50. The previously failing 0069 crop also passed through the actual official preprocessing and replacement adapter. All source hashes and original network/loss files remained unchanged.

Training statistics:

```json
{
  "status": "PASS",
  "steps": 200,
  "unique_train_images": 200,
  "training_crop_stats": {
    "invalid_crop_resample_count": 0,
    "affected_image_count": 0,
    "affected_image_paths": [],
    "affected_sample_count": 0,
    "max_sample_resample_count": 0
  },
  "unit_crop_stats": {
    "invalid_crop_resample_count": 282,
    "affected_image_count": 4,
    "affected_image_paths": [
      "datasets/U3D/lowlight/train/0069.jpg",
      "datasets/U3D/lowlight/train/0323.jpg",
      "datasets/U3D/lowlight/train/0455.jpg",
      "datasets/U3D/lowlight/train/0651.jpg"
    ],
    "affected_sample_count": 262,
    "max_sample_resample_count": 5
  },
  "seconds": 81.686721800128,
  "mean_data_ms": 367.57504699984565,
  "mean_compute_ms": 36.186224984703586,
  "peak_allocated_mib": 1205.72705078125,
  "peak_reserved_mib": 1864.0,
  "all_gradients_finite": true,
  "all_parameters_updated": true,
  "nan_inf": false,
  "oom": false,
  "original_network_loss_files_unchanged": true,
  "all_image_hashes_unchanged": true,
  "splits_unchanged": true,
  "no_dev_or_quarantine_training": true
}
```

High-risk sampler results:

```json
[
  {
    "source": "datasets/U3D/lowlight/train/0323.jpg",
    "samples": 1000,
    "returned_nonzero": 1000,
    "deterministic_replay": true,
    "invalid_crop_resample_count": 109,
    "affected_image_count": 1,
    "affected_image_paths": [
      "datasets/U3D/lowlight/train/0323.jpg"
    ],
    "affected_sample_count": 98,
    "max_sample_resample_count": 5
  },
  {
    "source": "datasets/U3D/lowlight/train/0651.jpg",
    "samples": 1000,
    "returned_nonzero": 1000,
    "deterministic_replay": true,
    "invalid_crop_resample_count": 95,
    "affected_image_count": 1,
    "affected_image_paths": [
      "datasets/U3D/lowlight/train/0651.jpg"
    ],
    "affected_sample_count": 91,
    "max_sample_resample_count": 2
  },
  {
    "source": "datasets/U3D/lowlight/train/0455.jpg",
    "samples": 1000,
    "returned_nonzero": 1000,
    "deterministic_replay": true,
    "invalid_crop_resample_count": 76,
    "affected_image_count": 1,
    "affected_image_paths": [
      "datasets/U3D/lowlight/train/0455.jpg"
    ],
    "affected_sample_count": 71,
    "max_sample_resample_count": 2
  },
  {
    "source": "datasets/U3D/lowlight/train/0069.jpg",
    "samples": 1000,
    "returned_nonzero": 1000,
    "deterministic_replay": true,
    "invalid_crop_resample_count": 2,
    "affected_image_count": 1,
    "affected_image_paths": [
      "datasets/U3D/lowlight/train/0069.jpg"
    ],
    "affected_sample_count": 2,
    "max_sample_resample_count": 1
  }
]
```

Every invalid coordinate and retry number is retained in metrics/sampler_trials.json and train history. Unit-test counts and training counts are separate; synthetic exhaustion/forced integration tests are not added to stochastic rates. This proves technical smoke compatibility, not convergence, quality or generalization. No commit/push. Stop for human review.
