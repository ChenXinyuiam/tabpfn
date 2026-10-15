"""
Compute geometric features for each data point:
  - Recalculated triangle area
  - Angle sum validation (geometry_valid flag)
  - Side lengths via the sine rule (左边长度, 右边长度)
  - Barycentric coordinates (lambda_A, lambda_B, lambda_C)
  - Normalised barycentric coordinates

Input:  data/interim/scaled/*.xlsx
Output: data/interim/features/*.xlsx

Usage (run from project root):
    python -m src.data.calculate
"""

import argparse
import sys

import numpy as np
import pandas as pd

from utils.paths import ensure_dir, iter_excel_files_for_batch, select_batch_dir

from ..config import (
    ANGLE_A_COL,
    ANGLE_B_COL,
    ANGLE_C_COL,
    AREA_COL,
    BASE_COL,
    D_BASE_COL,
    D_LEFT_COL,
    D_RIGHT_COL,
    HEIGHT_COL,
    LEFT_SIDE_COL,
    PROCESSED_DIR,
    RIGHT_SIDE_COL,
    SCALED_DIR,
)

# Output column names
AREA_CALC_COL     = "面积_calc"
ANGLE_SUM_COL     = "角和"
VALID_GEOMETRY_COL= "geometry_valid"
LAMBDA_A_COL      = "lambda_A"
LAMBDA_B_COL      = "lambda_B"
LAMBDA_C_COL      = "lambda_C"
LAMBDA_SUM_COL    = "lambda_sum"
LAMBDA_A_NORM_COL = "lambda_A_norm"
LAMBDA_B_NORM_COL = "lambda_B_norm"
LAMBDA_C_NORM_COL = "lambda_C_norm"


def _sin_deg(series: pd.Series) -> pd.Series:
    return np.sin(np.deg2rad(series.astype(float)))


def compute_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        BASE_COL, HEIGHT_COL,
        ANGLE_A_COL, ANGLE_B_COL, ANGLE_C_COL,
        D_BASE_COL, D_LEFT_COL, D_RIGHT_COL,
    }
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] missing geometry columns, skipping barycentric features: {missing}")
        return df

    df = df.copy()

    numeric_cols = [
        BASE_COL, HEIGHT_COL, AREA_COL,
        ANGLE_A_COL, ANGLE_B_COL, ANGLE_C_COL,
        D_BASE_COL, D_LEFT_COL, D_RIGHT_COL,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recalculate area and angle sum
    df[AREA_CALC_COL] = df[BASE_COL] * df[HEIGHT_COL] / 2.0
    df[ANGLE_SUM_COL] = df[ANGLE_A_COL] + df[ANGLE_B_COL] + df[ANGLE_C_COL]

    sin_A = _sin_deg(df[ANGLE_A_COL])

    df[VALID_GEOMETRY_COL] = (
        (df[BASE_COL]      > 0) &
        (df[HEIGHT_COL]    > 0) &
        (df[AREA_CALC_COL] > 0) &
        (np.abs(df[ANGLE_SUM_COL] - 180.0) < 1.0) &
        (np.abs(sin_A) > 1e-15) &
        (df[D_BASE_COL]  >= 0) &
        (df[D_LEFT_COL]  >= 0) &
        (df[D_RIGHT_COL] >= 0)
    )

    valid = df[VALID_GEOMETRY_COL]

    # Side lengths via sine rule: b = a*sin(B)/sin(A), c = a*sin(C)/sin(A)
    df[RIGHT_SIDE_COL] = np.nan
    df[LEFT_SIDE_COL]  = np.nan
    df.loc[valid, RIGHT_SIDE_COL] = (
        df.loc[valid, BASE_COL] * _sin_deg(df.loc[valid, ANGLE_B_COL]) / sin_A.loc[valid]
    )
    df.loc[valid, LEFT_SIDE_COL] = (
        df.loc[valid, BASE_COL] * _sin_deg(df.loc[valid, ANGLE_C_COL]) / sin_A.loc[valid]
    )

    # Barycentric coordinates: lambda = edge * distance / (2 * area)
    denom = 2.0 * df[AREA_CALC_COL]
    df[LAMBDA_A_COL] = np.nan
    df[LAMBDA_B_COL] = np.nan
    df[LAMBDA_C_COL] = np.nan

    df.loc[valid, LAMBDA_A_COL] = (
        df.loc[valid, BASE_COL]        * df.loc[valid, D_BASE_COL]  / denom.loc[valid]
    )
    df.loc[valid, LAMBDA_B_COL] = (
        df.loc[valid, RIGHT_SIDE_COL]  * df.loc[valid, D_RIGHT_COL] / denom.loc[valid]
    )
    df.loc[valid, LAMBDA_C_COL] = (
        df.loc[valid, LEFT_SIDE_COL]   * df.loc[valid, D_LEFT_COL]  / denom.loc[valid]
    )

    df[LAMBDA_SUM_COL] = df[LAMBDA_A_COL] + df[LAMBDA_B_COL] + df[LAMBDA_C_COL]

    # Normalised barycentric coordinates
    df[LAMBDA_A_NORM_COL] = np.nan
    df[LAMBDA_B_NORM_COL] = np.nan
    df[LAMBDA_C_NORM_COL] = np.nan

    valid_sum = valid & df[LAMBDA_SUM_COL].notna() & (np.abs(df[LAMBDA_SUM_COL]) > 1e-15)
    for raw, norm in [
        (LAMBDA_A_COL, LAMBDA_A_NORM_COL),
        (LAMBDA_B_COL, LAMBDA_B_NORM_COL),
        (LAMBDA_C_COL, LAMBDA_C_NORM_COL),
    ]:
        df.loc[valid_sum, norm] = df.loc[valid_sum, raw] / df.loc[valid_sum, LAMBDA_SUM_COL]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate geometry features for one scaled-data batch.")
    parser.add_argument("--batch", default=None, help="Batch folder under data/interim/scaled. Defaults to the latest batch.")
    args = parser.parse_args()

    if not SCALED_DIR.exists():
        sys.exit(f"Scaled data directory not found: {SCALED_DIR}")

    batch_dir = select_batch_dir(SCALED_DIR, args.batch)
    batch_name = batch_dir.name if batch_dir != SCALED_DIR else ""

    xlsx_files = iter_excel_files_for_batch(batch_dir)
    if not xlsx_files:
        sys.exit(f"No .xlsx files found in {batch_dir}")

    dst_root = PROCESSED_DIR / batch_name if batch_name else PROCESSED_DIR
    ensure_dir(dst_root)
    print(f"Processing batch {batch_name or '(root)'}: {len(xlsx_files)} file(s) -> {dst_root}/")
    for src in xlsx_files:
        relative_path = src.relative_to(batch_dir)
        dst = ensure_dir(dst_root / relative_path.parent) / src.name
        print(f"  {relative_path} ...", end=" ", flush=True)
        df = pd.read_excel(src, engine="openpyxl")
        df = compute_geometry_features(df)
        df.to_excel(dst, index=False, engine="openpyxl")
        print("done")

    print("All files processed.")


if __name__ == "__main__":
    main()
