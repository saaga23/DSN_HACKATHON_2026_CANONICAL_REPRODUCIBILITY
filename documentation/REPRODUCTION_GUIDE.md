# REPRODUCTION_GUIDE.md

## 1. Required competition data (supplied separately)

This package does NOT include competition data. You must supply:

- `train.csv` — training data with target `total_sales` (6,818 rows)
- `test.csv` — test data (1,705 rows)
- `sample_submission.csv` — submission schema reference

Place these files in a `competition_data/` directory adjacent to the package,
or pass `--data-dir` to the reproduction script.

## 2. Reproduce OPTION_1_CANONICAL

Reproduction script: `code/canonical/reproduce_canonical.py`
Notebook: `notebooks/canonical_reproduction.ipynb`

```bash
python code/canonical/reproduce_canonical.py --data-dir /path/to/competition_data --output-dir submission
```

Expected output: `submission/FINAL_SUBMISSION.csv`
Expected SHA256: `55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4`

The reproduction script asserts that the regenerated submission matches the
canonical SHA256. If it does not match, the script exits with a non-zero
status code.

## 3. Verify submission hash

```bash
sha256sum submission/FINAL_SUBMISSION.csv
```

Expected output:

```
55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4  submission/FINAL_SUBMISSION.csv
```

## 4. Notes

- `folds_seed42.csv` is NOT included in this package. The canonical
  reproduction script will deterministically regenerate it if missing.
- Do NOT claim artifacts not present in this package are included.
- Do NOT modify frozen submission files.
- The competition data is supplied separately and is subject to the
  competition platform's terms of use.

---