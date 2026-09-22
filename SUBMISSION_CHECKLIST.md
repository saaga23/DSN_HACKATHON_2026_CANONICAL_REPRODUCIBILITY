# SUBMISSION_CHECKLIST.md

## Pre-Release Verification

| # | Check | Status |
|---|-------|--------|
| 1 | Canonical ZIP opens and passes archive integrity | PASS |
| 2 | Submission SHA256 = `55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4` | PASS |
| 3 | Submission row count = 1,705 | PASS |
| 4 | Schema = `id,total_sales` | PASS |
| 5 | No NaN / Inf values | PASS |
| 6 | No duplicate IDs | PASS |
| 7 | ID order matches sample submission | PASS |
| 8 | PDFs exist (FINAL_RESEARCH_REPORT, REPRODUCIBILITY_REPORT, PHASE23_CLOSURE) | PASS |
| 9 | Notebooks exist (canonical_reproduction) | PASS |
| 10 | Python scripts exist (reproduce_canonical, _exec_notebook) | PASS |
| 11 | Manifests exist (RELEASE_MANIFEST, HASH_MANIFEST) | PASS |
| 12 | Provenance exists (provenance.json) | PASS |
| 13 | Final package audit exists | PASS |
| 14 | No E3 active submission contamination | PASS |
| 15 | No secrets or credentials in repository | PASS |

## Repository Verification

| # | Check | Status |
|---|-------|--------|
| 1 | Public repository exists | PASS |
| 2 | Repository is PUBLIC | PASS |
| 3 | README renders correctly | PASS |
| 4 | Canonical script is accessible | PASS |
| 5 | Notebook is accessible | PASS |
| 6 | Submission file is accessible | PASS |
| 7 | PDFs are accessible | PASS |
| 8 | Environment files are accessible | PASS |
| 9 | Provenance is accessible | PASS |
| 10 | All README links resolve | PASS |
| 11 | No broken links | PASS |
| 12 | Git tag `v1.0.0-canonical` created | PASS |

## Final Status

**READY** — canonical submission is frozen, verified, and published.

---