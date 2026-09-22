import os, sys, json, hashlib, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# Detect environment
if os.path.exists("/kaggle/input"):
    INPUT_DIR = "/kaggle/input"
    comp_dirs = [d for d in os.listdir(INPUT_DIR) if "dsn" in d.lower() or "bootcamp" in d.lower() or "hackathon" in d.lower()]
    if comp_dirs:
        DATA_DIR = os.path.join(INPUT_DIR, comp_dirs[0])
    else:
        DATA_DIR = INPUT_DIR
    OUTPUT_DIR = "/kaggle/working"
    ARTIFACTS_DIR = "/kaggle/working/artifacts"
    RELEASE_DIR = "/kaggle/working/release"
else:
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "competition_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    RELEASE_DIR = os.path.join(BASE_DIR, "release")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RELEASE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Cell 2 ---
# =============================================================================
# SECTION 2: PATHS (relative, Kaggle-compatible)
# =============================================================================

# Detect environment
if os.path.exists('/kaggle/input'):
    INPUT_DIR = '/kaggle/input'
    comp_dirs = [d for d in os.listdir(INPUT_DIR) if 'dsn' in d.lower() or 'bootcamp' in d.lower() or 'hackathon' in d.lower()]
    if comp_dirs:
        DATA_DIR = os.path.join(INPUT_DIR, comp_dirs[0])
    else:
        DATA_DIR = INPUT_DIR
    OUTPUT_DIR = '/kaggle/working'
    ARTIFACTS_DIR = '/kaggle/working/artifacts'
    RELEASE_DIR = '/kaggle/working/release'
else:
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'competition_data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
    RELEASE_DIR = os.path.join(BASE_DIR, 'release')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RELEASE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print(f'DATA_DIR: {DATA_DIR}')
print(f'OUTPUT_DIR: {OUTPUT_DIR}')
print(f'ARTIFACTS_DIR: {ARTIFACTS_DIR}')
print(f'RELEASE_DIR: {RELEASE_DIR}')

# --- Cell 3 ---
# =============================================================================
# SECTION 3: LOAD DATA AND GENERATE FOLDS
# =============================================================================
print('\n--- Loading data and generating folds ---')

# Load data
train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_raw = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

# Generate canonical folds
folds_df = pd.DataFrame({'row_idx': range(len(train_raw))})
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds_df['fold'] = -1
for fi, (tr, va) in enumerate(kf.split(folds_df)):
    folds_df.loc[va, 'fold'] = fi

# Save folds for reproducibility
folds_df.to_csv(os.path.join(ARTIFACTS_DIR, 'folds_seed42.csv'), index=False)

# Build fold list
folds = []
for fi in range(5):
    va = folds_df[folds_df['fold'] == fi]['row_idx'].values
    tr = folds_df[folds_df['fold'] != fi]['row_idx'].values
    folds.append((tr, va))

print(f'Training rows: {len(train_raw)}')
print(f'Test rows: {len(test_raw)}')
print(f'Folds: {len(folds)} (KFold shuffle=True, random_state=42)')

# --- Cell 4 ---
# =============================================================================
# SECTION 4: PREPROCESSING
# =============================================================================
print('\n--- Preprocessing ---')

def clean_category(cat):
    if pd.isna(cat):
        return cat
    return cat.strip().title()

def preprocess(train_df, test_df):
    df = pd.concat([train_df.copy(), test_df.copy()], ignore_index=True)
    df['product_category_clean'] = df['product_category'].apply(clean_category)
    df['fat_content_clean'] = df['fat_content'].str.strip().str.title()
    df['missing_weight'] = df['product_weight_kg'].isna().astype(int)
    df['missing_size'] = df['store_size'].isna().astype(int)
    ssm = df.groupby('store_code')['store_size'].first().to_dict()
    df['store_size_imputed'] = df['store_size'].fillna(df['store_code'].map(ssm))
    df['store_missing_size'] = df.groupby('store_code')['store_size'].transform(lambda x: x.isna().all()).astype(int)
    pw = df.groupby('product_code')['product_weight_kg'].transform('median')
    df['product_weight_kg_imputed'] = df['product_weight_kg'].fillna(pw).fillna(df['product_weight_kg'].fillna(pw).median())
    df['log_price'] = np.log1p(df['product_price'])
    df['price_sq'] = df['product_price'] ** 2
    df['log_visibility'] = np.log1p(df['shelf_visibility'])
    df['visibility_zero'] = (df['shelf_visibility'] == 0).astype(int)
    df['log_weight'] = np.log1p(df['product_weight_kg_imputed'])
    df['store_size_encoded'] = df['store_size_imputed'].map({'Small': 0, 'Medium': 1, 'Large': 2, 'MISSING': -1}).fillna(-1)
    df['tier_encoded'] = df['store_location_tier'].map({'Tier_1': 0, 'Tier_2': 1, 'Tier_3': 2}).fillna(1)
    df['fat_encoded'] = (df['fat_content_clean'] == 'Low Fat').astype(int)
    gp = df['product_price'].mean()
    df['price_vs_global_mean'] = df['product_price'] - gp
    df['price_ratio_global_mean'] = df['product_price'] / (gp + 1e-6)
    sm = df.groupby('store_code')['product_price'].transform('mean')
    df['price_vs_store_mean'] = df['product_price'] - sm
    df['price_ratio_store_mean'] = df['product_price'] / (sm + 1e-6)
    cm = df.groupby('product_category_clean')['product_price'].transform('mean')
    df['price_vs_cat_mean'] = df['product_price'] - cm
    df['price_ratio_cat_mean'] = df['product_price'] / (cm + 1e-6)
    df['price_x_visibility'] = df['product_price'] * df['shelf_visibility']
    df['price_x_store_age'] = df['product_price'] * df['store_age_years']
    df['visibility_x_store_age'] = df['shelf_visibility'] * df['store_age_years']
    return df

df = preprocess(train_raw, test_raw)
train_df = df[df['id'].isin(train_raw['id'])].reset_index(drop=True)
test_df = df[df['id'].isin(test_raw['id'])].reset_index(drop=True)
y_train = train_df['total_sales'].values


# --- Cell 5 ---
# =============================================================================
# SECTION 5: FEATURES
# =============================================================================
CATEGORICAL_FEATURES = ['product_code', 'store_code', 'product_category_clean', 'store_format', 'store_location_tier', 'fat_content_clean']
BASE13 = [
    'product_weight_kg_imputed', 'shelf_visibility', 'product_price', 'store_age_years',
    'missing_weight', 'missing_size', 'visibility_zero', 'log_price', 'log_weight', 'log_visibility',
    'store_size_encoded', 'tier_encoded', 'fat_encoded'
]
FEATS19 = BASE13 + CATEGORICAL_FEATURES

# Label encoders (fit on train+test combined)
le_map = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col].astype(str), test_df[col].astype(str)], ignore_index=True))
    le_map[col] = le

def encode(d, feats):
    for col in CATEGORICAL_FEATURES:
        if col in feats:
            d[col] = le_map[col].transform(d[col].astype(str))
    for c in feats:
        if c in d.columns and d[c].dtype in ['float64', 'int64']:
            d[c] = d[c].fillna(d[c].median())
    return d


# --- Cell 6 ---
# =============================================================================
# SECTION 6: MODEL A - DIRECT CATBOOST
# =============================================================================
print('\n--- Model A: Direct CatBoost ---')
oof_a = np.zeros(len(train_df))
test_a = np.zeros(len(test_raw))

for fi, (tr_idx, va_idx) in enumerate(folds):
    X_tr = encode(train_df[FEATS19].copy().iloc[tr_idx].reset_index(drop=True), FEATS19)
    X_val = encode(train_df[FEATS19].copy().iloc[va_idx].reset_index(drop=True), FEATS19)
    X_test = encode(test_df[FEATS19].copy(), FEATS19)
    cat_idx = [X_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    m = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', allow_writing_files=False,
        random_seed=42, verbose=False
    )
    m.fit(X_tr, y_train[tr_idx], cat_features=cat_idx, eval_set=(X_val, y_train[va_idx]), early_stopping_rounds=100)
    oof_a[va_idx] = m.predict(X_val)
    test_a += m.predict(X_test) / 5

cv_a = np.sqrt(mean_squared_error(y_train, oof_a))
print(f'A_direct: CV RMSE = {cv_a:.4f}')

# --- Cell 7 ---
# =============================================================================
# SECTION 7: MODEL B - STORE NORMALIZED CATBOOST
# =============================================================================
print('\n--- Model B: Store Normalized CatBoost ---')
store_scale = train_df.groupby('store_code')['total_sales'].mean().to_dict()
train_df['store_scale'] = train_df['store_code'].map(store_scale)
test_df['store_scale'] = test_df['store_code'].map(store_scale).fillna(y_train.mean())
train_df['y_norm'] = y_train / (train_df['store_scale'] + 1e-6)

oof_b = np.zeros(len(train_df))
test_b = np.zeros(len(test_raw))

for fi, (tr_idx, va_idx) in enumerate(folds):
    X_tr = encode(train_df[FEATS19].copy().iloc[tr_idx].reset_index(drop=True), FEATS19)
    X_val = encode(train_df[FEATS19].copy().iloc[va_idx].reset_index(drop=True), FEATS19)
    X_test = encode(test_df[FEATS19].copy(), FEATS19)
    cat_idx = [X_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    m = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', allow_writing_files=False,
        random_seed=42, verbose=False
    )
    m.fit(X_tr, train_df['y_norm'].iloc[tr_idx].values, cat_features=cat_idx,
          eval_set=(X_val, train_df['y_norm'].iloc[va_idx].values), early_stopping_rounds=100)
    p_norm = m.predict(X_val)
    p_raw = p_norm * (train_df['store_scale'].iloc[va_idx].values + 1e-6)
    oof_b[va_idx] = p_raw
    test_b += (m.predict(X_test) * (test_df['store_scale'].values + 1e-6)) / 5

cv_b = np.sqrt(mean_squared_error(y_train, oof_b))
print(f'B_store_norm: CV RMSE = {cv_b:.4f}')

# --- Cell 8 ---
# =============================================================================
# SECTION 8: MODEL C - TWO-STAGE REGRESSION
# =============================================================================
print('\n--- Model C: Two-Stage Regression ---')
store_scale_pred = np.zeros(len(train_df))
test_store_scale_pred = np.zeros(len(test_raw))

for fi, (tr_idx, va_idx) in enumerate(folds):
    X_tr = encode(train_df[FEATS19].copy().iloc[tr_idx].reset_index(drop=True), FEATS19)
    X_val = encode(train_df[FEATS19].copy().iloc[va_idx].reset_index(drop=True), FEATS19)
    X_test = encode(test_df[FEATS19].copy(), FEATS19)
    cat_idx = [X_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    m = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', allow_writing_files=False,
        random_seed=42, verbose=False
    )
    m.fit(X_tr, train_df['store_scale'].iloc[tr_idx].values, cat_features=cat_idx,
          eval_set=(X_val, train_df['store_scale'].iloc[va_idx].values), early_stopping_rounds=100)
    store_scale_pred[va_idx] = m.predict(X_val)
    test_store_scale_pred += m.predict(X_test) / 5

train_df['store_scale_pred'] = store_scale_pred
train_df['product_multiplier'] = y_train / (train_df['store_scale_pred'] + 1e-6)

oof_c = np.zeros(len(train_df))
test_c = np.zeros(len(test_raw))

for fi, (tr_idx, va_idx) in enumerate(folds):
    X_tr = encode(train_df[FEATS19].copy().iloc[tr_idx].reset_index(drop=True), FEATS19)
    X_val = encode(train_df[FEATS19].copy().iloc[va_idx].reset_index(drop=True), FEATS19)
    X_test = encode(test_df[FEATS19].copy(), FEATS19)
    cat_idx = [X_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    m = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', allow_writing_files=False,
        random_seed=42, verbose=False
    )
    m.fit(X_tr, train_df['product_multiplier'].iloc[tr_idx].values, cat_features=cat_idx,
          eval_set=(X_val, train_df['product_multiplier'].iloc[va_idx].values), early_stopping_rounds=100)
    p_mult = m.predict(X_val)
    p_raw = p_mult * (train_df['store_scale_pred'].iloc[va_idx].values + 1e-6)
    oof_c[va_idx] = p_raw
    test_c += (m.predict(X_test) * (test_store_scale_pred + 1e-6)) / 5

cv_c = np.sqrt(mean_squared_error(y_train, oof_c))
print(f'C_two_stage: CV RMSE = {cv_c:.4f}')

# --- Cell 9 ---
# =============================================================================
# SECTION 9: MODEL D - RIDGE
# =============================================================================
print('\n--- Model D: Ridge ---')
train_linear = pd.get_dummies(train_df[['product_code', 'store_code', 'product_category_clean', 'store_format', 'store_location_tier', 'fat_content_clean']], drop_first=True)
test_linear = pd.get_dummies(test_df[['product_code', 'store_code', 'product_category_clean', 'store_format', 'store_location_tier', 'fat_content_clean']], drop_first=True)

missing_in_test = set(train_linear.columns) - set(test_linear.columns)
missing_in_train = set(test_linear.columns) - set(train_linear.columns)
for col in missing_in_test:
    test_linear[col] = 0
for col in missing_in_train:
    train_linear[col] = 0
test_linear = test_linear[train_linear.columns]

numeric_cols = BASE13
X_linear_train = pd.concat([train_df[numeric_cols].reset_index(drop=True), train_linear.reset_index(drop=True)], axis=1)
X_linear_test = pd.concat([test_df[numeric_cols].reset_index(drop=True), test_linear.reset_index(drop=True)], axis=1)

oof_d = np.zeros(len(train_df))
test_d = np.zeros(len(test_raw))

for fi, (tr_idx, va_idx) in enumerate(folds):
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_linear_train.iloc[tr_idx], y_train[tr_idx])
    p = ridge.predict(X_linear_train.iloc[va_idx])
    oof_d[va_idx] = p
    test_d += ridge.predict(X_linear_test) / 5

cv_d = np.sqrt(mean_squared_error(y_train, oof_d))
print(f'D_ridge: CV RMSE = {cv_d:.4f}')

# --- Cell 10 ---
# =============================================================================
# SECTION 10: MODEL E - CATBOOST + RIDGE RESIDUAL
# =============================================================================
print('\n--- Model E: CatBoost + Ridge Residual ---')
oof_e = np.zeros(len(train_df))
test_e = np.zeros(len(test_raw))
oof_base = oof_a
residuals = y_train - oof_base

for fi, (tr_idx, va_idx) in enumerate(folds):
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_linear_train.iloc[tr_idx], residuals[tr_idx])
    p_resid = ridge.predict(X_linear_train.iloc[va_idx])
    p_combined = oof_base[va_idx] + p_resid
    oof_e[va_idx] = p_combined

    X_tr = encode(train_df[FEATS19].copy().iloc[tr_idx].reset_index(drop=True), FEATS19)
    X_val = encode(train_df[FEATS19].copy().iloc[va_idx].reset_index(drop=True), FEATS19)
    X_test = encode(test_df[FEATS19].copy(), FEATS19)
    cat_idx = [X_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    m = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', eval_metric='RMSE', allow_writing_files=False,
        random_seed=42, verbose=False
    )
    m.fit(X_tr, y_train[tr_idx], cat_features=cat_idx, eval_set=(X_val, y_train[va_idx]), early_stopping_rounds=100)
    p_base_test = m.predict(X_test)
    p_resid_test = ridge.predict(X_linear_test)
    test_e += (p_base_test + p_resid_test) / 5

cv_e = np.sqrt(mean_squared_error(y_train, oof_e))
print(f'E_catboost_ridge: CV RMSE = {cv_e:.4f}')

# --- Cell 11 ---
# =============================================================================
# SECTION 11: APPLY FROZEN WEIGHTS
# =============================================================================
print('\n--- Applying frozen blend weights ---')

FROZEN_WEIGHTS = {
    'A_direct': 0.3150392234808109,
    'B_store_norm': 0.5635523048051151,
    'C_two_stage': 0.08326192124539074,
    'D_ridge': 0.038146550468683295,
    'E_catboost_ridge': 0.0
}

# Compute historical blend OOF RMSE (for reference only)
oof_blend_hist = (
    FROZEN_WEIGHTS['A_direct'] * oof_a +
    FROZEN_WEIGHTS['B_store_norm'] * oof_b +
    FROZEN_WEIGHTS['C_two_stage'] * oof_c +
    FROZEN_WEIGHTS['D_ridge'] * oof_d +
    FROZEN_WEIGHTS['E_catboost_ridge'] * oof_e
)
cv_blend_hist = np.sqrt(mean_squared_error(y_train, oof_blend_hist))
print(f'Historical blend (frozen weights) OOF RMSE: {cv_blend_hist:.10f}')
print(f'Classification: HISTORICAL_OPTIMISTIC_OOF (B and C use transductive target-derived statistics)')

# --- Cell 12 ---
# =============================================================================
# SECTION 12: FINAL TEST PREDICTIONS
# =============================================================================
print('\n--- Generating final test predictions ---')

# Clip D_ridge before blending (MOOT: no negative predictions, but follow pipeline)
test_d_clipped = np.clip(test_d, 0, None)

test_preds = {
    'A_direct': test_a,
    'B_store_norm': test_b,
    'C_two_stage': test_c,
    'D_ridge': test_d_clipped,
    'E_catboost_ridge': test_e
}

# Save component predictions
for name, pred in test_preds.items():
    pd.DataFrame({'id': test_raw['id'], 'total_sales': pred}).to_csv(
        os.path.join(OUTPUT_DIR, f'test_pred_{name}.csv'), index=False
    )

# Final blend with frozen weights
final_test = (
    FROZEN_WEIGHTS['A_direct'] * test_a +
    FROZEN_WEIGHTS['B_store_norm'] * test_b +
    FROZEN_WEIGHTS['C_two_stage'] * test_c +
    FROZEN_WEIGHTS['D_ridge'] * test_d_clipped +
    FROZEN_WEIGHTS['E_catboost_ridge'] * test_e
)
final_test = np.clip(final_test, 0, None)

# --- Cell 13 ---
# =============================================================================
# SECTION 13: SAVE SUBMISSION AND PROVENANCE
# =============================================================================
print('\n--- Saving submission ---')
submission = pd.DataFrame({'id': test_raw['id'], 'total_sales': final_test})
output_path = os.path.join(RELEASE_DIR, 'FINAL_SUBMISSION.csv')
submission.to_csv(output_path, index=False)

# SHA256
with open(output_path, 'rb') as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

print(f'Submission saved: {output_path}')
print(f'Row count: {len(submission)}')
print(f'SHA256: {sha256}')
print(f'Min prediction: {final_test.min():.4f}')
print(f'Max prediction: {final_test.max():.4f}')
print(f'Mean prediction: {final_test.mean():.4f}')

# Provenance
provenance = {
    'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'model': 'Frozen Historical Blend',
    'components': {
        'A_direct': {'model': 'CatBoostRegressor', 'weight': 0.3150392234808109, 'cv_rmse': float(cv_a)},
        'B_store_norm': {'model': 'CatBoostRegressor', 'weight': 0.5635523048051151, 'cv_rmse': float(cv_b)},
        'C_two_stage': {'model': 'CatBoostRegressor (two-stage)', 'weight': 0.08326192124539074, 'cv_rmse': float(cv_c)},
        'D_ridge': {'model': 'Ridge', 'weight': 0.038146550468683295, 'cv_rmse': float(cv_d)},
        'E_catboost_ridge': {'model': 'CatBoost+Ridge', 'weight': 0.0, 'cv_rmse': float(cv_e)}
    },
    'blend_weights': FROZEN_WEIGHTS,
    'oof_rmse_historical': float(cv_blend_hist),
    'oof_rmse_classification': 'HISTORICAL_OPTIMISTIC_OOF',
    'output_file': 'FINAL_SUBMISSION.csv',
    'sha256': sha256,
    'random_seed': 42,
    'folds': 'KFold(n_splits=5, shuffle=True, random_state=42)',
    'features': FEATS19
}

with open(os.path.join(RELEASE_DIR, 'PROVENANCE.json'), 'w') as f:
    json.dump(provenance, f, indent=2)
print(f"Provenance saved: {os.path.join(RELEASE_DIR, 'PROVENANCE.json')}")

# --- Cell 14 ---
# =============================================================================
# SECTION 14: VALIDATION
# =============================================================================
print('\n--- Validating submission ---')
assert len(submission) == len(sample_sub), f'Row count mismatch: {len(submission)} vs {len(sample_sub)}'
assert list(submission.columns) == list(sample_sub.columns), 'Column mismatch'
assert (submission['id'] == sample_sub['id']).all(), 'ID mismatch'
assert submission['total_sales'].notna().all(), 'NaN predictions'
assert np.isfinite(submission['total_sales']).all(), 'Infinite predictions'
assert (submission['total_sales'] >= 0).all(), 'Negative predictions'

print('\n' + '=' * 70)
print('REPRODUCTION COMPLETE')
print('=' * 70)
print(f'OOF RMSE (historical): {cv_blend_hist:.10f}')
print(f'Classification: HISTORICAL_OPTIMISTIC_OOF')
print(f'Submission: {output_path}')
print(f'SHA256: {sha256}')
print(f'Expected SHA256: 55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4')
print(f"Match: {sha256 == '55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4'}")
