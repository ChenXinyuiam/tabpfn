"""
Exploratory data analysis for processed Excel files.

Usage (run from project root):
    python -m src.data.explore                        # default: data/processedv2
    python -m src.data.explore --dir data/cleaned
    python -m src.data.explore --dir data/scaled --verbose
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def eda_file(file_path: Path, verbose: bool = False) -> None:
    print(f"\n{'='*100}")
    print(f"FILE: {file_path.name}")
    print(f"{'='*100}")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return

    print(f"\nCOLUMN NAMES ({len(df.columns)} columns):")
    print(", ".join(df.columns.tolist()))

    print(f"\nSHAPE: {df.shape[0]} rows x {df.shape[1]} columns")

    if verbose:
        print(f"\nDTYPES:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")

        print(f"\nFIRST 10 ROWS:")
        print(df.head(10).to_string())

        print(f"\nDESCRIPTIVE STATISTICS:")
        print(df.describe().to_string())

    # NaN values
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\nNaN VALUES:")
        for col in df.columns:
            if nan_counts[col] > 0:
                pct = 100 * nan_counts[col] / len(df)
                print(f"  {col}: {nan_counts[col]} ({pct:.1f}%)")
    else:
        print(f"\nNaN VALUES: none")

    # Negative values
    print(f"\nNEGATIVE VALUES:")
    has_neg = False
    for col in df.select_dtypes(include=[np.number]).columns:
        if (df[col] < 0).any():
            neg_count = (df[col] < 0).sum()
            print(f"  {col}: {neg_count} (min: {df[col].min():.4g})")
            has_neg = True
    if not has_neg:
        print("  none")

    # geometry_valid
    if "geometry_valid" in df.columns:
        false_count = (df["geometry_valid"] == False).sum()
        print(f"\ngeometry_valid: {false_count} False rows out of {len(df)}")

    # Outliers (IQR)
    print(f"\nOUTLIERS (IQR x1.5):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_found = False
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            if n > 0:
                print(f"  {col}: {n}")
                outlier_found = True
    if not outlier_found:
        print("  none")

    # Zero values
    print(f"\nZERO VALUES:")
    zero_found = False
    for col in numeric_cols:
        n = (df[col] == 0).sum()
        if n > 0:
            print(f"  {col}: {n}")
            zero_found = True
    if not zero_found:
        print("  none")

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDUPLICATE ROWS: {dup_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory data analysis for Excel files")
    parser.add_argument("--dir", type=Path, default=Path("data/processedv2"),
                        help="Directory containing .xlsx files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print dtypes, first rows, and descriptive stats")
    args = parser.parse_args()

    xlsx_files = sorted(args.dir.glob("*.xlsx"))
    if not xlsx_files:
        print(f"No .xlsx files found in {args.dir}")
        return

    print(f"Found {len(xlsx_files)} Excel file(s) in {args.dir}")
    for fp in xlsx_files:
        eda_file(fp, verbose=args.verbose)

    print(f"\n{'='*100}")
    print("EXPLORATION COMPLETE")


if __name__ == "__main__":
    main()
