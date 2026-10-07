# U3D train/dev v2.1

Two rounds of targeted human review have produced this development split. U3D lacks complete acquisition-sequence metadata; this does not prove that all potential scene relationships have been excluded.

## Change from v2

| Version | Train | Dev | Quarantine |
|---|---:|---:|---:|
| v2 |776|8|16|
| v2.1 |770|8|22|

42 review rows: 36 different, 5 same, 1 uncertain, no pending or duplicate pair. CSV column counts are consistent. Decisions and notes are preserved verbatim in JSON; no missing notes were inferred. Old v2 is unchanged. All six new quarantine files were original v2 train members.

## Newly quarantined

| Pair | Dev | Image | Human decision | Handling |
|---|---|---|---|---|
| P004 | D07 / 0179.jpg | datasets/U3D/lowlight/train/0373.jpg | same | quarantine |
| P005 | D07 / 0179.jpg | datasets/U3D/lowlight/train/0593.jpg | same | quarantine |
| P016 | D09 / 0049.jpg | datasets/U3D/lowlight/train/0207.jpg | uncertain | quarantine |
| P027 | D28 / 0758.jpg | datasets/U3D/lowlight/train/0033.jpg | same | quarantine |
| P028 | D28 / 0758.jpg | datasets/U3D/lowlight/train/0358.jpg | same | quarantine |
| P029 | D28 / 0758.jpg | datasets/U3D/lowlight/train/0462.jpg | same | quarantine |

0207.jpg is quarantined conservatively; uncertain is NOT a same edge and does not enlarge the confirmed D09 scene group. All 36 different targets remain in train.

## Confirmed same connected components

| ID | Representative | Size including dev | Complete known members |
|---|---|---:|---|
| D07 | 0179.jpg | 3 | 0179.jpg, 0373.jpg, 0593.jpg |
| D08 | 0114.jpg | 3 | 0027.jpg, 0114.jpg, 0127.jpg |
| D09 | 0049.jpg | 2 | 0049.jpg, 0381.jpg |
| D10 | 0166.jpg | 5 | 0166.jpg, 0210.jpg, 0223.jpg, 0546.jpg, 0594.jpg |
| D19 | 0792.jpg | 2 | 0792.jpg, 0795.jpg |
| D28 | 0758.jpg | 8 | 0033.jpg, 0240.jpg, 0358.jpg, 0368.jpg, 0462.jpg, 0551.jpg, 0646.jpg, 0758.jpg |
| D29 | 0343.jpg | 3 | 0343.jpg, 0359.jpg, 0525.jpg |
| D31 | 0216.jpg | 3 | 0009.jpg, 0131.jpg, 0216.jpg |

Every confirmed same edge from both rounds is retained in the JSON with its source round and original note. Transitive closure has no different edge inside a same component. All eight representatives remain in separate confirmed components.

## Leakage checks

```json
{
  "all_passed": true,
  "csv_rows": 42,
  "csv_columns_valid": true,
  "duplicate_review_pairs": 0,
  "pending_user": 0,
  "same_different_conflicts": 0,
  "train_dev_path_intersection": 0,
  "train_quarantine_path_intersection": 0,
  "dev_quarantine_path_intersection": 0,
  "coverage_original_train": 800,
  "duplicate_entries_each_set": 0,
  "all_files_exist": true,
  "train_dev_sha256_intersection": 0,
  "train_quarantine_sha256_intersection": 0,
  "train_dev_confirmed_same_group_intersection": 0,
  "all_uncertain_targets_quarantined": true,
  "dev_representatives_unchanged": true,
  "original_images_match_v2_sha256": true
}
```

SHA256 was recomputed from all 800 actual images and matched v2. The three disjoint lists cover exactly the original official train. Official val/test are not used. Historical train/0209 versus val/0880 duplication cannot cross this train/dev boundary because official val is excluded.

## Conditions and limitations

The requested checks pass: v2.1 is ready for development train/dev integration, not a claim of full benchmark independence. No training or model loading was performed.

- Two rounds of targeted manual review do not prove absolute scene independence without acquisition sequence metadata.
- The 42 selected dev-to-train relations are resolved for split handling, not an exhaustive new similarity search.
- One uncertain pair remains semantically unresolved; its target is conservatively quarantined, not declared same.
- Eight dev representatives form a small development set, not a final independent benchmark.
- Original val/test remain excluded; fresh random initialization is required for independent experiments, without reusing smoke checkpoints.

## Reproduction and provenance

`python -B scripts/build_u3d_train_dev_v2_1.py` uses standard-library code and refuses overwriting outputs. JSON includes input CSV/parent split hashes, all image hashes, exact review notes, same edges and quarantine reasons. Old v2 and human CSVs are never rewritten. The four new manifests use project-relative image paths. No dependencies, environment changes, commit or push.
