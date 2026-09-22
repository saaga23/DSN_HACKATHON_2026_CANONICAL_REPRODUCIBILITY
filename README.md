# DSN Bootcamp Qualification Hackathon 2026 ML Track
# Canonical Submission and Reproducibility Archive

## Overview

This repository contains the canonical competition submission and the
complete reproducibility archive for the DSN Bootcamp Qualification Hackathon
2026 ML Track.

- **Competition:** DSN Bootcamp Qualification Hackathon 2026 ML Track
- **Task:** Predict `total_sales` for each row in `test.csv` (regression, RMSE)
- **Candidate:** OPTION_1_CANONICAL
- **Status:** FROZEN — final submission, no further modifications authorized

## Final Submission

| Field | Value |
|-------|-------|
| File | `submission/FINAL_SUBMISSION.csv` |
| Public Score | **1073.18868** |
| Submission SHA256 | `55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4` |
| Rows | 1,705 |
| Columns | `id`, `total_sales` |
| Reproducibility | BYTE-IDENTICAL |

The public score is historical/observed evidence from the competition
public leaderboard. Private leaderboard performance is **not** guaranteed
and is unknown.

## Reproducibility

[Reproduce the canonical submission](code/canonical/reproduce_canonical.py)

[Open the reproduction notebook](notebooks/canonical_reproduction.ipynb)

[View the frozen submission](submission/FINAL_SUBMISSION.csv)

The canonical submission is an optimized blend of five models with frozen
weights, trained on the 6,818-row training set using 5-fold cross-validation
(seed 42):

| Component | Model | Weight |
|-----------|-------|--------|
| A_direct | CatBoostRegressor | 0.3150392234808109 |
| B_store_norm | CatBoostRegressor | 0.5635523048051151 |
| C_two_stage | CatBoostRegressor (two-stage) | 0.08326192124539074 |
| D_ridge | Ridge | 0.038146550468683295 |
| E_catboost_ridge | CatBoost + Ridge residual | 0.0 |

CatBoost parameters: `iterations=600`, `learning_rate=0.05`, `depth=6`,
`l2_leaf_reg=3.0`, `random_seed=42`.

### Requirements

```bash
pip install -r environment/requirements.txt
```

- numpy==1.26.4
- pandas==2.1.4
- scikit-learn==1.4.2
- catboost==1.2.10
- scipy==1.12.0

### Reproduce

```bash
python code/canonical/reproduce_canonical.py \
    --data-dir /path/to/competition_data \
    --output-dir submission
```

The script requires the competition data files (`train.csv`, `test.csv`,
`sample_submission.csv`) in a `competition_data/` directory. These files are
**not** included in this repository and must be supplied separately from the
competition platform.

The script asserts that the regenerated `FINAL_SUBMISSION.csv` matches the
canonical SHA256 `55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4`.

## Repository Structure

```
DSN_HACKATHON_2026_CANONICAL_REPRODUCIBILITY/
├── README.md
├── LICENSE
├── SUBMISSION_METADATA.md
├── REPRODUCIBILITY_CERTIFICATE.md
├── FINAL_PACKAGE_AUDIT.md
├── RELEASE_MANIFEST.csv
├── HASH_MANIFEST.sha256
├── SUBMISSION_CHECKLIST.md
│
├── submission/
│   ├── FINAL_SUBMISSION.csv
│   ├── hash.txt
│   └── provenance.json
│
├── code/
│   └── canonical/
│       ├── reproduce_canonical.py
│       └── _exec_notebook.py
│
├── notebooks/
│   └── canonical_reproduction.ipynb
│
├── reports/
│   ├── FINAL_RESEARCH_REPORT.pdf
│   ├── REPRODUCIBILITY_REPORT.pdf
│   └── PHASE23_CLOSURE.pdf
│
├── documentation/
│   ├── DECISION_LOG.md
│   ├── EXPERIMENT_INDEX.md
│   ├── FINAL_STATE.md
│   ├── README_FINAL.md
│   ├── REPRODUCTION_GUIDE.md
│   └── RULES_AND_DETAILS.md
│
├── environment/
│   ├── requirements.txt
│   ├── environment.yml
│   └── python_version.txt
│
└── archive/
    └── DSN_HACKATHON_2026_CANONICAL_SUBMISSION_PACKAGE.zip
```

## Environment

See `environment/requirements.txt` and `environment/environment.yml` for the
pinned package versions. Python >= 3.8 is required; Python 3.9-3.12 is
recommended.

## Submission Integrity

The canonical submission is immutable. Its SHA256 is:

```
55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4
```

Verify at any time with:

```bash
Get-FileHash -Algorithm SHA256 submission/FINAL_SUBMISSION.csv
sha256sum submission/FINAL_SUBMISSION.csv
```

## Research Closure

Research was conducted through Phase 18 (Ensemble Complementarity Audit). The
final result was **OUTCOME C**: no useful ensemble complementarity was found;
all 48 candidate blends were worse than the canonical CatBoost pipeline. The
canonical submission is the final, frozen artifact.

See `reports/FINAL_RESEARCH_REPORT.pdf` and `reports/PHASE23_CLOSURE.pdf` for
the complete research record.

## Important Limitations

1. **Competition data is not included.** `train.csv`, `test.csv`, and
   `sample_submission.csv` must be supplied separately from the competition
   platform. The repository does not contain competition labels.
2. **No credentials are required.** The reproduction path uses only local
   data files. No API keys, tokens, or authentication secrets are needed.
3. **Public score is historical evidence only.** The public score
   1073.18868 was observed on the public leaderboard. Private leaderboard
   performance is unknown and not guaranteed.
4. **CatBoost version sensitivity.** Byte-identical reproduction requires
   CatBoost 1.2.10. Different versions may produce slightly different
   predictions.
5. **No new submissions are authorized.** This repository is an archive, not
   an active submission pipeline.

## Citation

If you use this repository, please cite it appropriately and note that the
competition data is subject to the competition platform's terms of use.

---