# EXP004B support: loading profile and archival review

## Scope and safeguards

No training, environment changes, dataset writes or model/loss changes. Existing human decisions and blocked v1 split remain unchanged. Raw measurements: [metrics](../experiments/exp004b_loading_profile/metrics.json). Review snapshot: [status](../experiments/exp004b_loading_profile/review_status.json).

## Loader measurements

Three original train images from the EXP004A manifest, three repetitions each (9 samples), 3840x2160, fixed 256x256 crop. Pillow full RGB decode, official 9x9 sigma=1.7 full-image Gaussian blur, original float64 division followed by float32 tensors. CUDA synchronized for transfer. No model execution. OS file cache may be warm; these are not cold disk measurements. Process state differs from EXP004A, so the component total is not expected to equal its 555.04 ms.

| Component | Mean ms |
|---|---:|
| decode_rgb_ms | 112.346 |
| numpy_copy_ms | 79.886 |
| blur_ms | 14.477 |
| full_normalize_tensor_ms | 475.738 |
| crop_contiguous_ms | 2.232 |
| h2d_ms | 1.786 |
| total_components_ms | 686.466 |
| cached_uint8_crop_convert_ms | 4.154 |

Full-resolution normalization/tensor conversion is the largest measured component, followed by decode and CPU copies. Current smoke code indexes Dataset directly: no DataLoader workers, prefetch or persistent workers. Every reuse reopens/redecodes the same file. Crop happens AFTER full decode, blur and tensor conversion. Only the cropped pair transfers to GPU; no full-image GPU transfer. There are full-image float64 intermediates followed by float32 copies.

### Evidence-based optimization candidate (not integrated)

Keep original RGB and full-image blurred RGB as bounded uint8 cache entries, crop both at identical coordinates, then normalize only the crop using the SAME arithmetic. All 9 tested crops are bitwise equal to the existing path. This covers three images at one fixed interior crop only, not all borders/crops. Before production integration, test random/border crops, full batches and loss/gradient equivalence.

Three uint8 image pairs consume 149,299,200 bytes (142.383 MiB); each pair 47.461 MiB. Cache lookup/miss/eviction policy and end-to-end training speedup were not benchmarked. The cached crop conversion timing excludes cache construction, H2D and validation. Do not multiply its speedup into a training throughput claim. Cache by full original path and preprocessing parameters, bounded per process; do not resize/recompress data or mix splits. Caching all 800 images is inappropriate without a RAM budget.

Windows multiprocessing requires a top-level pickleable Dataset and a guarded main entry. Consider 1-2 CPU workers, persistent_workers only when num_workers>0, bounded prefetch, and pinned memory only after measurement. Workers must return CPU tensors; CUDA transfer stays in the main process. Full 4K float tensors in worker queues multiply memory use. These options were NOT benchmarked here and no speed benefit is claimed. No formal training loader was changed.

## Read-only review support

Run `python -B scripts/check_exp004b_review.py` in nightuav. Optional `--output` must name a new file. It checks allowed decisions, duplicate rows, priority membership, contradictory cross-table decisions, same_scene transitive closure versus different_scene, and decodability of every contact image. It never modifies CSVs or publishes splits. Synthetic direct and transitive conflict tests passed. Hashes before/after reading match.

40 candidates pending; 671 unique relation decisions pending, including 60 priority pairs. All 78 linked PNG contacts verified. This is zero completed human decisions in the on-disk CSVs; work outside those files is unknown. Open `experiments/exp004b_dev_review/outputs/index.html`; review the 17 priority candidates first. Decisions may be entered in the priority sheet or full relationship sheet; differing non-pending entries are reported as conflicts, not silently overwritten. Pending/uncertain associations do not establish independence. Contact images remain local ignored outputs.

## Git and portability

82 files in the previous submission inventory matched their recorded byte hashes before edits. JSON/Python files parse; no files exceed 1 MiB and no common token/private-key patterns found. The only Users path hit is the public upstream placeholder C:/Users/username in dependency_plan.json; the email there is public upstream author metadata. That verbose pip plan remains local, untracked. Pattern scans cannot guarantee absence of every kind of secret.

The old report's 20 absolute-path files included URL false positives (e.g. https:/). Reinspection below distinguishes runtime dependence from historical records. Original experimental sources were archived first, then portability maintenance is a separate commit. Historical timing, paths and hashes are untouched: they describe the old executed scripts, not the revised ones. Git may normalize line endings, so the historical raw-byte hash is not necessarily the Git blob hash.

Fixes: EXP003 interpreter requirement now checks nightuav/Python 3.10 rather than a drive-specific executable; existing torch/CUDA requirements remain. EXP002 maps historical dataset/results tails onto the current root, rejects ambiguous/escaping paths, and retains input hash checks. Report JSON reads specify UTF-8; generated command is a portable template, not a falsely claimed executed command. No historical reports were regenerated. Syntax, relocation and path traversal rejection tests passed; GPU training was not rerun.

| Previously flagged file | Treatment |
|---|---|
| `docs/BASELINE_SELECTION.md` | Historical machine path preserved |
| `docs/DI_RETINEX_METHOD_STUDY.md` | URL/schema false positive; no machine path |
| `docs/DI_RETINEX_RESEARCH_HYPOTHESES.md` | URL/schema false positive; no machine path |
| `experiments/exp001_di_retinex/README.md` | Historical machine path preserved |
| `experiments/exp001_di_retinex/config/run.json` | Historical machine path preserved |
| `experiments/exp001_di_retinex/metrics/inference.json` | Historical machine path preserved |
| `experiments/exp002_baseline_diagnosis/README.md` | Historical machine path preserved |
| `experiments/exp002_baseline_diagnosis/REPRODUCE.md` | Historical machine path preserved |
| `experiments/exp002_baseline_diagnosis/metrics/exp001_histogram.svg` | URL/schema false positive; no machine path |
| `experiments/exp002_baseline_diagnosis/metrics/provenance.json` | Historical machine path preserved |
| `experiments/exp003_di_retinex_smoke/README.md` | Historical machine path preserved |
| `experiments/exp003_di_retinex_smoke/dependency_plan.json` | Public package metadata; left untracked |
| `experiments/exp003_di_retinex_smoke/training/config.json` | Historical machine path preserved |
| `experiments/exp003_di_retinex_smoke/training/summary.json` | Historical machine path preserved |
| `experiments/exp004_di_retinex_baseline/README.md` | Historical machine path preserved |
| `experiments/exp004a_multisample_smoke/README.md` | Historical machine path preserved |
| `scripts/analyze_exp002.py` | Runtime portability repaired; original version retained in archival commit |
| `scripts/infer_di_retinex.py` | URL/schema false positive; no machine path |
| `scripts/report_exp002.py` | Runtime portability repaired; original version retained in archival commit |
| `scripts/train_di_retinex_smoke.py` | Runtime portability repaired; original version retained in archival commit |

## Remaining limitations

No approved train/dev v2; v1 stays 0/0/800. No formal validation or full training readiness claim. Review decisions remain entirely human. The benchmark does not establish worker scaling, cold-storage throughput, or full-training throughput. Existing method hypotheses remain hypotheses. Data, external nested repositories, weights, caches and large contact sheets are excluded from Git.
