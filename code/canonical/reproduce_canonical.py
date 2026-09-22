#!/usr/bin/env python3
"""
DSN Bootcamp Qualification Hackathon 2026 ML Track
OPTION_1_CANONICAL REPRODUCTION SCRIPT
=========================================

Reproduces FINAL_SUBMISSION.csv with exact frozen blend weights.
Byte-identical reproduction verified.

Requirements:
- numpy==1.26.4
- pandas==2.1.4
- scikit-learn==1.4.2
- catboost==1.2.10
- scipy==1.12.0

Usage:
    python reproduce_canonical.py --output-dir submission
"""

import os
import sys
import hashlib
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce OPTION_1_CANONICAL submission")
    parser.add_argument("--data-dir", default="DATA/competition_data", help="Path to competition data")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Path to artifacts")
    parser.add_argument("--output-dir", default="submission", help="Output directory")
    return parser.parse_args()

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

def main():
    args = parse_args()
    print("=" * 70)
    print("OPTION_1_CANONICAL REPRODUCTION")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {__import__('sklearn').__version__}")
    print(f"CatBoost: {__import__('catboost').__version__}")
    print(f"SciPy: {__import__('scipy').__version__}")

    # Load data
    print("\n--- Loading data ---")
    train_raw = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    test_raw = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    sample_sub = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))

    print(f"Train: {len(train_raw)} rows")
    print(f"Test: {len(test_raw)} rows")
    print(f"Sample submission: {len(sample_sub)} rows")

    # Load or generate canonical folds
    folds_path = os.path.join(args.artifacts_dir, 'folds_seed42.csv')
    if os.path.exists(folds_path):
        print(f"Loading folds from {folds_path}")
        folds_df = pd.read_csv(folds_path)
    else:
        print(f"folds_seed42.csv not found at {folds_path}. Generating deterministic folds with KFold(n_splits=5, shuffle=True, random_state=42).")
        folds_df = pd.DataFrame({'row_idx': range(len(train_raw))})
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds_df['fold'] = -1
        for fi, (tr, va) in enumerate(kf.split(folds_df)):
            folds_df.loc[va, 'fold'] = fi
        os.makedirs(args.artifacts_dir, exist_ok=True)
        folds_df.to_csv(folds_path, index=False)
        print(f"Generated and saved folds to {folds_path}")

    # Fixed folds
    folds = []
    for fi in range(5):
        va = folds_df[folds_df['fold'] == fi]['row_idx'].values
        tr = folds_df[folds_df['fold'] != fi]['row_idx'].values
        folds.append((tr, va))
    print(f"Folds: {len(folds)}")

    # Preprocess
    print("\n--- Preprocessing ---")
    df = preprocess(train_raw, test_raw)
    train_df = df[df['id'].isin(train_raw['id'])].reset_index(drop=True)
    test_df = df[df['id'].isin(test_raw['id'])].reset_index(drop=True)
    y_train = train_df['total_sales'].values

    CATEGORICAL_FEATURES = ['product_code', 'store_code', 'product_category_clean', 'store_format', 'store_location_tier', 'fat_content_clean']
    BASE13 = [
        'product_weight_kg_imputed', 'shelf_visibility', 'product_price', 'store_age_years',
        'missing_weight', 'missing_size', 'visibility_zero', 'log_price', 'log_weight', 'log_visibility',
        'store_size_encoded', 'tier_encoded', 'fat_encoded'
    ]
    FEATS19 = BASE13 + CATEGORICAL_FEATURES

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

    # Model A
    print("\n--- Model A: Direct CatBoost ---")
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
    print(f"A_direct: CV RMSE = {cv_a:.4f}")

    # Model B
    print("\n--- Model B: Store Normalized CatBoost ---")
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
    print(f"B_store_norm: CV RMSE = {cv_b:.4f}")

    # Model C
    print("\n--- Model C: Two-Stage Regression ---")
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
    print(f"C_two_stage: CV RMSE = {cv_c:.4f}")

    # Model D
    print("\n--- Model D: Ridge ---")
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
    print(f"D_ridge: CV RMSE = {cv_d:.4f}")

    # Model E
    print("\n--- Model E: CatBoost + Ridge Residual ---")
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
    print(f"E_catboost_ridge: CV RMSE = {cv_e:.4f}")

    # FROZEN weights (canonical)
    print("\n--- Applying frozen blend weights ---")
    opt_w = np.array([0.3150392234808109, 0.5635523048051151, 0.08326192124539074, 0.038146550468683295, 0.0])
    blend_names = ['A_direct', 'B_store_norm', 'C_two_stage', 'D_ridge', 'E_catboost_ridge']
    for name, w in zip(blend_names, opt_w):
        print(f"  {name}: {w:.15f}")

    test_preds = {
        'A_direct': test_a,
        'B_store_norm': test_b,
        'C_two_stage': test_c,
        'D_ridge': np.clip(test_d, 0, None),
        'E_catboost_ridge': test_e
    }
    final_test = sum(opt_w[i] * test_preds[name] for i, name in enumerate(blend_names))
    final_test = np.clip(final_test, 0, None)

    # Validation
    print("\n--- Validating ---")
    assert len(sample_sub) == len(test_raw), "Row count mismatch"
    assert list(sample_sub.columns) == ['id', 'total_sales'], "Column mismatch"
    assert (sample_sub['id'] == test_raw['id']).all(), "ID mismatch"
    assert np.isfinite(final_test).all(), "Infinite predictions"
    assert (final_test >= 0).all(), "Negative predictions"

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'FINAL_SUBMISSION.csv')
    submission = pd.DataFrame({'id': test_raw['id'], 'total_sales': final_test})
    submission.to_csv(output_path, index=False)

    with open(output_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"\nSubmission saved: {output_path}")
    print(f"Row count: {len(submission)}")
    print(f"SHA256: {sha256}")
    print(f"Min prediction: {final_test.min():.4f}")
    print(f"Max prediction: {final_test.max():.4f}")
    print(f"Mean prediction: {final_test.mean():.4f}")

    # Provenance
    provenance = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": "Optimized Blend (Frozen Weights)",
        "components": {name: {"model": "CatBoostRegressor" if name != 'D_ridge' and name != 'E_catboost_ridge' else ("Ridge" if name == 'D_ridge' else "CatBoost+Ridge"), "weight": float(w)} for name, w in zip(blend_names, opt_w)},
        "blend_weights": {name: float(w) for name, w in zip(blend_names, opt_w)},
        "oof_rmse": {
            "A_direct": float(cv_a),
            "B_store_norm": float(cv_b),
            "C_two_stage": float(cv_c),
            "D_ridge": float(cv_d),
            "E_catboost_ridge": float(cv_e)
        },
        "output_file": "FINAL_SUBMISSION.csv",
        "sha256": sha256,
        "random_seed": 42,
        "folds": "KFold(n_splits=5, shuffle=True, random_state=42)",
        "features": FEATS19
    }
    prov_path = os.path.join(args.output_dir, 'provenance.json')
    with open(prov_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    print(f"Provenance saved: {prov_path}")

    print("\n" + "=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)
    print(f"Expected SHA256: 55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4")
    print(f"Actual SHA256:   {sha256}")
    print(f"Match: {sha256 == '55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4'}")

    if sha256 != '55f346a2d7a94e10f83cf5e393013a113d74210ab128ce95dd89501277c354e4':
        print("\nWARNING: SHA256 mismatch! Reproduction did not produce canonical submission.")
        sys.exit(1)

if __name__ == '__main__':
    main()
