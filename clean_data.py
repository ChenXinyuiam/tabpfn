"""
Data cleaning script for triangular geometry FEM/flow data.

File naming: "{base_mm}-{height_mm}-FL.xlsx"
  - base_mm  : bottom edge length of the triangle (mm)
  - height_mm: triangle height (mm)

Columns (same across all files, Chinese headers kept as-is):
  [0]  base length  (mm, constant per file)
  [1]  X            (coordinate, m)
  [2]  Z            (coordinate, m)
  [3]  n_param      (mesh/layer count, constant=6)
  [4]  total_length (m)
  [5]  area         (m²)
  [6]  height       (mm, constant per file)
  [7]  angle_1      (°)
  [8]  angle_2      (°)
  [9]  angle_3      (°)
  [10] disp_X       (displacement X)
  [11] disp_Y       (displacement Y)
  [12] dist_left    (distance to left edge, m)   ← 20 NaN in 690-340 only
  [13] dist_top     (distance to top edge, m)    ← 20 NaN in 690-340 only
  [14] dist_right   (distance to right edge, m)  ← 20 NaN in 690-340 only
  [15] dist_bottom  (distance to bottom/center edge, m)
  [16] FL           (target variable – missing in all files)

Missing-value strategy
─────────────────────
1. 690-340-FL  dist_left / dist_top / dist_right  (20 rows)
   These are exact linear functions of (X, Z) with R²=1.000.
   → Fit on complete rows, then predict the 20 missing values.

2. 830-342-FL  10 "skeleton" rows
   Rows where almost every column is NaN (only angle values survive).
   → Drop these rows before any further processing.

3. FL column  (target variable)
   Missing in every file (counts vary from 2 to 63).
   Rows with NaN FL cannot be used for modelling.
   → Drop them and save the cleaned file to data/cleaned/.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

RAW_DIR = "data/rawdata"
CLEAN_DIR = "data/cleaned"
os.makedirs(CLEAN_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fill missing distance columns by fitting a linear model on (X, Z)
# ─────────────────────────────────────────────────────────────────────────────
def impute_distance_cols_by_geometry(df: pd.DataFrame, dist_cols: list[str]) -> pd.DataFrame:
    """
    Each of the three distance columns is an *exact* linear function of X and Z
    (verified: R² = 1.0000 on complete rows).  Fit on non-missing rows and
    predict for rows where the distances are NaN.
    """
    df = df.copy()
    missing_mask = df[dist_cols[0]].isna()  # all three columns share the same missing rows
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return df

    X_feat = df[["X", "Z"]].values
    for col in dist_cols:
        train_mask = ~df[col].isna()
        lr = LinearRegression()
        lr.fit(X_feat[train_mask], df.loc[train_mask, col].values)
        df.loc[missing_mask, col] = lr.predict(X_feat[missing_mask])

    print(f"  → Imputed {n_missing} rows in distance columns {dist_cols} via linear regression on (X, Z)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main cleaning loop
# ─────────────────────────────────────────────────────────────────────────────
files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".xlsx") and not f.startswith("~$"))

summary_rows = []

for fname in files:
    path = os.path.join(RAW_DIR, fname)
    df = pd.read_excel(path, engine="openpyxl")
    original_shape = df.shape
    cols = list(df.columns)

    print(f"\n{'='*60}")
    print(f"File : {fname}")
    print(f"Shape: {original_shape[0]} rows × {original_shape[1]} cols")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    # ── Step 1: Drop entirely-null columns  ──────────────────────────────────
    all_null_cols = [c for c in cols if df[c].isna().all()]
    if all_null_cols:
        df.drop(columns=all_null_cols, inplace=True)
        print(f"  → Dropped fully-null column(s): {all_null_cols}")

    # ── Step 2: Drop skeleton rows (almost all NaN, only angles survive)  ───
    #    Threshold: >8 NaN values in a row (angles + dist_bottom = 4 non-null at most)
    null_per_row = df.isnull().sum(axis=1)
    skeleton_mask = null_per_row > 8
    n_skeleton = skeleton_mask.sum()
    if n_skeleton > 0:
        df = df[~skeleton_mask].reset_index(drop=True)
        print(f"  → Dropped {n_skeleton} skeleton rows (only angle values present)")

    # ── Step 3: Impute missing distance columns in 690-340-FL  ──────────────
    #    Columns 12-14 (dist_left, dist_top, dist_right) — exact linear fit.
    dist_cols = cols[12:15]  # left, top, right  (bottom is col 15, never missing)
    if df[dist_cols].isnull().any().any():
        df = impute_distance_cols_by_geometry(df, dist_cols)

    # ── Step 4: Drop rows with missing FL (target variable)  ─────────────────
    fl_col = "FL"
    n_missing_fl = df[fl_col].isna().sum()
    df_clean = df.dropna(subset=[fl_col]).reset_index(drop=True)
    print(f"  → Dropped {n_missing_fl} rows with NaN FL  →  {len(df_clean)} rows kept")

    # ── Step 5: Sanity-check no remaining NaN  ───────────────────────────────
    remaining_null = df_clean.isnull().sum().sum()
    if remaining_null > 0:
        print(f"  ⚠ WARNING: {remaining_null} NaN values still present after cleaning:")
        print(f"    {df_clean.isnull().sum()[df_clean.isnull().sum() > 0].to_string()}")

    # ── Step 6: Save  ─────────────────────────────────────────────────────────
    out_path = os.path.join(CLEAN_DIR, fname)
    df_clean.to_excel(out_path, index=False, engine="openpyxl")
    print(f"  → Saved to {out_path}  ({len(df_clean)} rows × {len(df_clean.columns)} cols)")

    summary_rows.append({
        "file": fname,
        "rows_original": original_shape[0],
        "rows_kept": len(df_clean),
        "rows_dropped": original_shape[0] - len(df_clean),
        "cols_original": original_shape[1],
        "cols_kept": len(df_clean.columns),
        "remaining_null": remaining_null,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
print(f"\nCleaned files saved to: {os.path.abspath(CLEAN_DIR)}/")
