# SUBMISSION_METADATA.md

## DSN Bootcamp Qualification Hackathon 2026 ML Track
## OPTION_1_CANONICAL — Canonical Submission Metadata

| Field | Value |
|-------|-------|
| Candidate identity | OPTION_1_CANONICAL |
| Competition | DSN Bootcamp Qualification Hackathon 2026 ML Track |
| Track | Machine Learning (ML) |
| Task | Regression — predict `total_sales` |
| Evaluation metric | Root Mean Squared Error (RMSE) |
| Submission filename | `submission/FINAL_SUBMISSION.csv` |
| Public Score | 1073.18868 |
| Row count | 1,705 |
| Column count | 2 (`id`, `total_sales`) |
| Submission SHA256 | `55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4` |
| Status | FROZEN / AUTHORITATIVE |
| Reproducibility | BYTE-IDENTICAL |
| Model type | Optimized Blend (frozen weights) |
| Training data | 6,818 rows (`train.csv`) |
| Test data | 1,705 rows (`test.csv`) |
| Cross-validation | KFold(n_splits=5, shuffle=True, random_state=42) |
| Features | 19 (13 numeric + 6 categorical) |
| Random seed | 42 |
| Generated at | 2026-09-10T10:01:30+01:00 |
| Archive date | 2026-09-22 |

## Reproduction

- **Script:** `code/canonical/reproduce_canonical.py`
- **Notebook:** `notebooks/canonical_reproduction.ipynb`
- **Authoritative source:** `code/canonical/_exec_notebook.py`
- **Environment:** `environment/requirements.txt`

## Competition Data Notice

The competition data (`train.csv`, `test.csv`, `sample_submission.csv`) is
**not** included in this repository. It must be supplied separately from the
competition platform. Reproduction is not possible without this data.

## Immutable

This submission is frozen. Do not modify `submission/FINAL_SUBMISSION.csv`.
Any regenerated copy must match the canonical SHA256 listed above.

---