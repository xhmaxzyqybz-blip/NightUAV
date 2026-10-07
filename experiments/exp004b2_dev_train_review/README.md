# EXP004B-2 manual review

42 unique dev-to-train pairs extracted from the v2 candidate JSON; no new retrieval or automatic same/different decisions.

Open [contact sheet](outputs/index.html), optionally filter by candidate. Pair IDs correspond exactly to [review CSV](review_dev_train.csv). Fill decision with `same`, `different`, or `uncertain`; record your reasoning in note. Preserve paths and pair IDs. The HTML is static and does not save edits.

Comparisons use RGB thumbnails with preserved aspect ratio and unchanged exposure; click original links for detail. Similarity metrics are retrieval evidence, not scene labels. Outputs are ignored by Git. No v2 changes, training, commit or push.

Generator: `python -B scripts/prepare_exp004b2_review.py`; refuses to overwrite an existing review directory. Source hashes, per-candidate counts and verification results: [manifest](generation_manifest.json).
