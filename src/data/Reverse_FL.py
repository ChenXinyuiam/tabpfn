"""Publish one processed batch.

``*-FL.xlsx`` files are copied unchanged.
``*-sym.xlsx`` files have the FL sign reversed before being published.
"""

import argparse
import shutil
import sys

import pandas as pd

from src.config import PROCESSED_DIR, PUBLISHED_PROCESSED_ROOT
from utils.paths import ensure_dir, iter_excel_files_for_batch, select_batch_dir

INPUT_DIR = PROCESSED_DIR
OUTPUT_DIR = PUBLISHED_PROCESSED_ROOT
FL_COL = "FL"


def process_file(src, dst) -> None:
    df = pd.read_excel(src, engine="openpyxl")

    if FL_COL not in df.columns:
        print(f"  [WARN] 'FL' column not found in {src.name}, skip")
        return

    df[FL_COL] = -df[FL_COL]
    df.to_excel(dst, index=False, engine="openpyxl")


def copy_file(src, dst) -> None:
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse FL signs for one processed-data batch.")
    parser.add_argument("--batch", default=None, help="Batch folder under data/interim/features. Defaults to the latest batch.")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR}")

    batch_dir = select_batch_dir(INPUT_DIR, args.batch)
    batch_name = batch_dir.name if batch_dir != INPUT_DIR else ""

    xlsx_files = [
        path
        for path in iter_excel_files_for_batch(batch_dir)
        if path.name.endswith("-FL.xlsx") or path.name.endswith("-sym.xlsx")
    ]

    if not xlsx_files:
        sys.exit(f"No *-FL.xlsx or *-sym.xlsx files found in {batch_dir}")

    dst_root = PUBLISHED_PROCESSED_ROOT / batch_name if batch_name else OUTPUT_DIR
    ensure_dir(dst_root)
    print(f"Processing batch {batch_name or '(root)'}: {len(xlsx_files)} file(s) -> {dst_root}/")
    for src in xlsx_files:
        relative_path = src.relative_to(batch_dir)
        dst = ensure_dir(dst_root / relative_path.parent) / src.name
        print(f"  {relative_path}", end=" ... ", flush=True)
        if src.name.endswith("-sym.xlsx"):
            process_file(src, dst)
        else:
            copy_file(src, dst)
        print("done")

    print("All files processed.")


if __name__ == "__main__":
    main()
