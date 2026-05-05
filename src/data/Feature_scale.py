"""
Feature scaling pipeline: apply physical unit scale factors to cleaned data.

Scale factors:
  1e12 : 面积
  1e6  : X, Z, 周长, 重心坐标X/Y, distance columns, FL

Usage (run from project root):
    python -m src.data.scale
"""

import argparse
import sys

import pandas as pd

from ..config import CLEANED_DIR, SCALED_DIR, SCALE_1E12, SCALE_1E6
from utils.paths import ensure_dir, iter_excel_files, select_batch_dir


def scale_file(src, dst) -> None:
    '''
    对每个文件的目标列进行放大
    '''
    
    df = pd.read_excel(src, engine="openpyxl")

    missing = (SCALE_1E12 | SCALE_1E6) - set(df.columns)
    if missing:
        print(f"  [WARN] columns not found in {src.name}: {missing}")

    for col in SCALE_1E12:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col] * 1e12

    for col in SCALE_1E6:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col] * 1e6

    df.to_excel(dst, index=False, engine="openpyxl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale one cleaned-data batch.")
    parser.add_argument("--batch", default=None, help="Batch folder under data/cleaned. Defaults to the latest batch.")
    args = parser.parse_args()

    if not CLEANED_DIR.exists():
        sys.exit(f"Cleaned data directory not found: {CLEANED_DIR}")

    batch_dir = select_batch_dir(CLEANED_DIR, args.batch)
    batch_name = batch_dir.name if batch_dir != CLEANED_DIR else ""

    xlsx_files = iter_excel_files(batch_dir)
    if not xlsx_files:
        sys.exit(f"No .xlsx files found in {batch_dir}")

    dst_root = SCALED_DIR / batch_name if batch_name else SCALED_DIR
    ensure_dir(dst_root)
    print(f"Scaling batch {batch_name or '(root)'}: {len(xlsx_files)} file(s) -> {dst_root}/")
    for src in xlsx_files:
        dst = dst_root / src.name
        print(f"  {src.name} ...", end=" ", flush=True)
        scale_file(src, dst)
        print("done")

    print("All files scaled.")


if __name__ == "__main__":
    main()
