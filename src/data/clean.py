"""
Data cleaning pipeline for triangular geometry / flow-loss data.

File naming convention: "{base_mm}-{height_mm}-FL.xlsx"

Cleaning steps:
  1. Drop entirely-null columns
  2. Drop skeleton rows (all NaN except angle columns)
  3. Impute missing distance columns in 690-340-FL via linear regression on (X, Z)
  4. Drop rows with NaN FL (target variable)

Usage (run from project root):
    python -m src.data.clean
"""

import argparse
import sys
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..config import (
    RAW_DIR,
    CLEANED_DIR,
    BASE_COL,
    ANGLE_A_COL,
    ANGLE_B_COL,
    ANGLE_C_COL,
    D_BASE_COL,
    D_LEFT_COL,
    D_RIGHT_COL,
)
from utils.paths import ensure_dir, iter_excel_files_for_batch, select_batch_dir

CENTROID_DISTANCE_COL = "点到重心距离"
CENTROID_X_COL = "重心坐标X"
CENTROID_Y_COL = "重心坐标Y"

RAW_COLUMN_ALIASES = {
    "BC": BASE_COL,
    "角度A": ANGLE_A_COL,
    "角度B": ANGLE_B_COL,
    "角度C": ANGLE_C_COL,
    "点到BC的垂直距离": D_BASE_COL,
    "到BC垂直距离": D_BASE_COL,
    "点到AB的垂直距离": D_LEFT_COL,
    "到AB垂直距离": D_LEFT_COL,
    "点到AC的垂直距离": D_RIGHT_COL,
    "到AC垂直距离": D_RIGHT_COL,
    "Fly": "FL",
    "FLy": "FL",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {src: dst for src, dst in RAW_COLUMN_ALIASES.items() if src in df.columns and dst not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  -> Renamed columns: {rename_map}")
    return df


def normalized_output_name(filename: str) -> str:
    match = re.search(r"(\d+-\d+)", filename)
    if not match:
        return filename

    prefix = match.group(1)
    suffix = "-sym.xlsx" if "反" in filename else "-FL.xlsx"
    return f"{prefix}{suffix}"


def impute_distance_cols(df: pd.DataFrame, dist_cols: list[str]) -> pd.DataFrame:
    """
    Each distance column is an exact linear function of X and Z (R²≈1).
    Fit on complete rows, predict for rows where the distances are NaN.
    """
    df = df.copy()
    missing_mask = df[dist_cols[0]].isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return df

    X_feat = df[["X", "Z"]].values
    for col in dist_cols:
        train_mask = ~df[col].isna()
        lr = LinearRegression()
        lr.fit(X_feat[train_mask], df.loc[train_mask, col].values)
        df.loc[missing_mask, col] = lr.predict(X_feat[missing_mask])

    print(f"  -> Imputed {n_missing} rows in {dist_cols} via linear regression on (X, Z)")
    return df


def impute_centroid_distance(df: pd.DataFrame) -> pd.DataFrame:
    required = {"X", "Z", CENTROID_X_COL, CENTROID_Y_COL, CENTROID_DISTANCE_COL}
    if not required.issubset(df.columns):
        return df

    df = df.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_mask = df[CENTROID_DISTANCE_COL].isna()
    computable_mask = missing_mask & df[["X", "Z", CENTROID_X_COL, CENTROID_Y_COL]].notna().all(axis=1)
    n_imputed = int(computable_mask.sum())
    if n_imputed == 0:
        return df

    dx = df.loc[computable_mask, "X"] - df.loc[computable_mask, CENTROID_X_COL]
    dz = df.loc[computable_mask, "Z"] - df.loc[computable_mask, CENTROID_Y_COL]
    df.loc[computable_mask, CENTROID_DISTANCE_COL] = np.sqrt(dx**2 + dz**2)
    print(f"  -> Imputed {n_imputed} rows in {CENTROID_DISTANCE_COL} via centroid distance formula")
    return df


def clean_file(src: os.PathLike, dst: os.PathLike, max_abs_fl: float | None = None) -> dict:
    df = pd.read_excel(src, engine="openpyxl")
    df = normalize_columns(df)
    original_shape = df.shape
    cols = list(df.columns)

    # Step 1: drop entirely-null columns
    null_cols = [c for c in cols if df[c].isna().all()]
    if null_cols:
        df.drop(columns=null_cols, inplace=True)
        print(f"  -> Dropped fully-null columns: {null_cols}")

    # Step 2: drop skeleton rows (>8 NaN per row)
    skeleton_mask = df.isnull().sum(axis=1) > 8
    n_skeleton = skeleton_mask.sum()
    if n_skeleton:
        df = df[~skeleton_mask].reset_index(drop=True)
        print(f"  -> Dropped {n_skeleton} skeleton rows")

    # Step 3: impute missing distance columns (columns 12-14: left, top, right)
    dist_cols = [col for col in [D_BASE_COL, D_LEFT_COL, D_RIGHT_COL] if col in df.columns]
    if len(dist_cols) == 3 and df[dist_cols].isnull().any().any():
        df = impute_distance_cols(df, dist_cols)

    df = impute_centroid_distance(df)

    # Step 4: coerce FL to numeric and drop rows with missing/non-numeric FL
    df["FL"] = pd.to_numeric(df["FL"], errors="coerce")
    n_outlier_fl = 0
    if max_abs_fl is not None:
        outlier_mask = df["FL"].abs() > max_abs_fl
        n_outlier_fl = int(outlier_mask.sum())
        if n_outlier_fl:
            df = df[~outlier_mask].reset_index(drop=True)
            print(f"  -> Dropped {n_outlier_fl} rows with |FL| > {max_abs_fl:g}")

    n_missing_fl = df["FL"].isna().sum()
    df = df.dropna(subset=["FL"]).reset_index(drop=True)
    print(f"  -> Dropped {n_missing_fl} rows with missing/non-numeric FL -> {len(df)} rows kept")

    remaining_null = df.isnull().sum().sum()
    if remaining_null:
        print(f"  WARNING: {remaining_null} NaN values remain after cleaning")

    df.to_excel(dst, index=False, engine="openpyxl")
    return {
        "file": src.name if hasattr(src, "name") else str(src),
        "rows_original": original_shape[0],
        "rows_kept": len(df),
        "rows_dropped": original_shape[0] - len(df),
        "cols_original": original_shape[1],
        "cols_kept": len(df.columns),
        "remaining_null": remaining_null,
        "rows_dropped_fl_outlier": n_outlier_fl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean one raw-data batch.")
    parser.add_argument("--batch", default=None, help="Batch folder under data/rawdata. Defaults to the latest batch.")
    parser.add_argument("--max-abs-fl", type=float, default=None, help="Drop rows whose absolute raw FL exceeds this value.")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        sys.exit(f"Raw data directory not found: {RAW_DIR}")

    batch_dir = select_batch_dir(RAW_DIR, args.batch)
    batch_name = batch_dir.name if batch_dir != RAW_DIR else ""
    xlsx_files = iter_excel_files_for_batch(batch_dir)
    if not xlsx_files:
        sys.exit(f"No .xlsx files found in {batch_dir}")

    dst_root = CLEANED_DIR / batch_name if batch_name else CLEANED_DIR
    ensure_dir(dst_root)
    print(f"Cleaning batch {batch_name or '(root)'}: {len(xlsx_files)} file(s) -> {dst_root}/")
    summary_rows = []
    for src in xlsx_files:
        print(f"\n{'='*60}\n{src.name}")
        relative_parent = src.relative_to(batch_dir).parent
        dst_name = normalized_output_name(src.name)
        if dst_name != src.name:
            print(f"  -> Normalized filename to {dst_name}")
        dst_dir = ensure_dir(dst_root / relative_parent)
        stats = clean_file(src, dst_dir / dst_name, max_abs_fl=args.max_abs_fl)
        stats["batch"] = batch_name
        stats["group"] = relative_parent.as_posix()
        summary_rows.append(stats)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
