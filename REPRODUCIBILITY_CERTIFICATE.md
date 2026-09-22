# REPRODUCIBILITY_CERTIFICATE.md

## DSN Bootcamp Qualification Hackathon 2026 ML Track
## Final Reproducibility Certificate — OPTION_1_CANONICAL

**Certificate Date:** 2026-09-22
**Auditor:** Kilo (automated release audit)

---

## Summary

| Submission | Reproducibility Status |
|------------|------------------------|
| OPTION_1_CANONICAL (`FINAL_SUBMISSION.csv`) | **BYTE-IDENTICAL** |

The canonical submission can be regenerated to produce byte-identical
output given the exact code, data, and environment described below.

---

## 1. Can the canonical submission be regenerated?

**YES.** The canonical submission `FINAL_SUBMISSION.csv` can be regenerated
to byte-identical output.

**Authoritative script:** `code/canonical/reproduce_canonical.py`
**Alternate authoritative source:** `code/canonical/_exec_notebook.py`

The script uses hardcoded frozen blend weights and deterministic CatBoost
training with fixed seeds.

---

## 2. What exact code generated it?

| File | Role |
|------|------|
| `code/canonical/reproduce_canonical.py` | **Primary reproduction script** — writes `FINAL_SUBMISSION.csv` and asserts the canonical SHA256 |
| `code/canonical/_exec_notebook.py` | **Authoritative source** — the original notebook-style script that generated the submission |

---

## 3. What exact data files are required?

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 6,818 | Training data with target `total_sales` |
| `test.csv` | 1,705 | Test data (no target) |
| `sample_submission.csv` | 1,705 | Submission schema reference |

These files are **not** included in this repository. They must be supplied
separately from the competition platform.

`folds_seed42.csv` is optional — the reproduction script regenerates it
deterministically if it is missing.

---

## 4. What exact Python version is required?

**Not explicitly pinned.** The package versions indicate compatibility with
Python >= 3.8. The codebase uses f-strings and pandas 2.1.4, which require
Python >= 3.8.

**Recommended:** Python 3.9 – 3.12 (consistent with package availability).

---

## 5. What package versions are required?

```
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.4.2
catboost==1.2.10
scipy==1.12.0
```

`lightgbm==4.6.0` and `xgboost==3.2.0` are listed in `requirements.txt`
but are **not** used in the canonical reproduction script.

---

## 6. What OS/environment assumptions exist?

| Assumption | Detail |
|------------|--------|
| **OS** | Windows, Linux, or macOS. No OS-specific code. |
| **Working directory** | The script resolves paths relative to its own location or the `--data-dir` argument. |
| **File paths** | Uses `os.path.join(BASE_DIR, ...)` where `BASE_DIR` is the `--data-dir` argument. |
| **Memory** | Sufficient to load 6,818-row train + 1,705-row test into memory (~50 MB). |
| **Disk** | Minimal; no large artifacts written beyond the submission CSV. |

---

## 7. What random seeds are required?

| Component | Seed | Notes |
|-----------|------|-------|
| KFold split | 42 | `KFold(n_splits=5, shuffle=True, random_state=42)` |
| CatBoost (Models A, B, C) | 42 | `random_seed=42` |
| Ridge (Models D, E residual) | 42 | `random_state=42` |
| LabelEncoder | Deterministic | Fit on sorted unique values from combined train+test |

**All randomness is fully controlled.** No stochastic variation between
runs on the same data with the same package versions.

---

## 8. Are any paths hardcoded?

| Path | Hardcoded? | Detail |
|------|-----------|--------|
| `DATA_DIR` | Configurable | `--data-dir` argument, defaults to `DATA/competition_data` |
| `ARTIFACTS_DIR` | Configurable | `--artifacts-dir` argument, defaults to `artifacts` |
| `OUTPUT_DIR` | Configurable | `--output-dir` argument, defaults to `submission` |

Paths are resolved relative to the script location or the supplied
arguments, not the working directory.

---

## 9. Are any external downloads required?

**NO.** The reproduction path uses only local data files. No internet access
or external API calls are required.

---

## 10. Are any credentials/API keys required?

**NO.** The reproduction path does not require credentials, API keys, or
authentication tokens.

---

## 11. Are any generated artifacts required rather than source generation?

| Artifact | Required? | Detail |
|----------|-----------|--------|
| `folds_seed42.csv` | Optional | Script regenerates deterministically |
| `provenance.json` | No | Generated at runtime, not required for reproduction |

---

## 12. Is byte-identical reproduction possible?

**YES.** Byte-identical reproduction is guaranteed given:

1. Pinned package versions (Section 5)
2. Fixed random seeds (Section 7)
3. Exact code from `code/canonical/reproduce_canonical.py`
4. Exact data files (Section 3)

The reproduction script computes the SHA256 of the regenerated submission
and asserts it matches the canonical hash
`55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4`.

If environment drift occurs (e.g., a different CatBoost version), prediction
equivalence can be checked numerically by:

1. Computing correlation between regenerated predictions and canonical predictions
2. Computing max absolute difference
3. Computing mean absolute difference

---

## 13. What are the known limitations?

| Limitation | Severity | Detail |
|------------|----------|--------|
| **Competition data not included** | HIGH | Reproduction requires `train.csv`, `test.csv`, and `sample_submission.csv` supplied separately |
| **CatBoost version sensitivity** | MEDIUM | Byte-identical reproduction requires CatBoost 1.2.10. Newer versions may produce different predictions. |
| **No private leaderboard validation** | HIGH | Public score is verified. Private leaderboard performance is UNKNOWN. |
| **Python version not pinned** | LOW | Package versions imply Python >= 3.8, but exact minor version is not locked. |

---

## Reproducibility Status Summary

| Submission | Status | Evidence |
|------------|--------|----------|
| OPTION_1_CANONICAL | **BYTE-IDENTICAL** | `reproduce_canonical.py` SHA256 assertion, canonical hash match |

---

*Certificate issued as part of the canonical reproducibility archive.*  
*This certificate does not authorize any new Kaggle submission.*  
*Do not modify frozen submission files.*

---