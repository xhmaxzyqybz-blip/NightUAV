# U3D train/dev v2 candidate split

This is a development split with human review and isolation of known scene relations. Missing acquisition-sequence metadata prevents claiming that every train/dev image is scene-independent. No training or model selection was performed.

## Review and counts

60 priority pairs: 23 same, 37 different, 0 uncertain/pending. CSV columns are consistent; no format repairs. Candidate decisions: 8 keep, 9 exclude, 23 defer. Original train: 800. Final train: 776; dev: 8; quarantine: 16. Total removed from training: 24 (dev plus quarantine).

## Complete known manual same groups

| ID | Dev representative | Group size | All original-train members |
|---|---|---:|---|
| D07 | datasets/U3D/lowlight/train/0179.jpg | 1 | 0179.jpg |
| D08 | datasets/U3D/lowlight/train/0114.jpg | 3 | 0027.jpg, 0114.jpg, 0127.jpg |
| D09 | datasets/U3D/lowlight/train/0049.jpg | 2 | 0049.jpg, 0381.jpg |
| D10 | datasets/U3D/lowlight/train/0166.jpg | 5 | 0166.jpg, 0210.jpg, 0223.jpg, 0546.jpg, 0594.jpg |
| D19 | datasets/U3D/lowlight/train/0792.jpg | 2 | 0792.jpg, 0795.jpg |
| D28 | datasets/U3D/lowlight/train/0758.jpg | 5 | 0240.jpg, 0368.jpg, 0551.jpg, 0646.jpg, 0758.jpg |
| D29 | datasets/U3D/lowlight/train/0343.jpg | 3 | 0343.jpg, 0359.jpg, 0525.jpg |
| D31 | datasets/U3D/lowlight/train/0216.jpg | 3 | 0009.jpg, 0131.jpg, 0216.jpg |

Group size includes the representative. Group membership is complete for the supplied same graph, not a claim of complete physical-scene retrieval. Evidence edges and original paths are preserved in JSON. All 9 exclude representatives are linked to kept scenes and quarantined. Defer alone does not exclude a file: only connections to kept groups do.

## Checks

```json
{
  "csv_column_counts_valid": true,
  "csv_format_repairs": [],
  "same_different_conflicts": [],
  "train_dev_path_intersection": 0,
  "train_quarantine_path_intersection": 0,
  "dev_quarantine_path_intersection": 0,
  "train_dev_sha256_intersection": 0,
  "train_quarantine_sha256_intersection": 0,
  "train_dev_confirmed_same_group_intersection": 0,
  "dev_group_count": 8,
  "duplicate_entries": 0,
  "all_paths_exist": true,
  "original_train_coverage": 800,
  "official_val_test_included": false,
  "all_checks_passed": true
}
```

SHA256 was computed from all 800 actual original train files, not assumed from cached metadata. Existing train-internal duplicate groups remain within train, not across dev. Original val/test are never split inputs. Old v1 manifests and all original images remain unchanged.

## Outstanding review risk

42 directed retrieval relations connect kept representatives to proposed train images without a priority judgment; 0 previous assistant visual records cross the known isolated-scene/train boundary. These are disclosed in the JSON, not new human same/different decisions. The 671-row full association CSV remains a historical pending template; the 60 priority decisions are authoritative here.

- No complete acquisition sequence metadata: known manual scene isolation does not prove universal scene independence.
- Only 60 priority relations adjudicated. Remaining full-review template rows are not different decisions.
- Eight dev images give a small, nonrepresentative development estimate, not an independent final test.
- Unreviewed retrieval associations are disclosed, not automatically merged or classified different.
- Future independent training must use fresh random initialization, never EXP004A smoke weights.
- Original val/test are excluded. The historical train/0209 versus val/0880 duplicate cannot cross this train/dev split because val is not included.

The requested automatic known-relation checks pass. This candidate can support development under the stated limitations; it is not an independent benchmark split. Resolve any subsequently confirmed cross-boundary relation by revising the split before further selection.

## Provenance and reproduction

Run `python -B scripts/build_u3d_train_dev_v2.py` from this repository using the project environment. Refuses overwrite and aborts on a failed assertion before writing split files. No dependencies installed. Input CSVs are not rewritten. The JSON contains review SHA256, per-image SHA256, group evidence and pending relations. Text lists contain only project-relative original paths. No commit or push.
