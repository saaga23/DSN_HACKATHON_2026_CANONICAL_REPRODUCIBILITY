# Final State

**Date:** 2026-09-18

## Canonical Submission

- **File:** `submission/FINAL_SUBMISSION.csv`
- **SHA256:** `55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4`
- **Public Score:** 1073.18868
- **Row Count:** 1,705
- **Columns:** `id`, `total_sales`

## Baseline Model

CatBoost canonical pipeline.
- 7-seed mean RMSE: 1075.0968
- Seeds: 42, 123, 999, 2024, 7, 31, 1234

## Final Research Phase

Phase 18 — Ensemble Complementarity Audit
- **Result:** OUTCOME C — No useful ensemble complementarity found
- All 48 blends worse than CatBoost
- Residual correlations 0.95-0.98 — no complementary errors

## Current Decision

- Submission frozen
- No new submission authorized
- No Kaggle submission
- Phase 18 ensemble branch closed

## Known Historical Caveats

1. Phase 15 Product-Store Holdout 2570 RMSE was caused by an RMSE calculation bug and is INVALID.
2. Phase 17 invalid model-diversity experiment (P17-S2-02/03) labeled INVALID due to model identity error (CatBoost predictions mislabeled as LGBM/XGBoost).
3. E3 (public RMSE 1084.00044) was REJECTED by the public leaderboard. It is not included in this package.

## Reproduction Path

1. `competition_data/train.csv` + `test.csv` (supplied separately)
2. `code/canonical/reproduce_canonical.py`
3. `code/canonical/_exec_notebook.py`
4. `notebooks/canonical_reproduction.ipynb`

## Archive Path

All historical research is under `ARCHIVE/PHASES/`.

## Last Verification

2026-09-22 — SHA256 verified, file unchanged.

---