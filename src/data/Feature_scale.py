"""Feature scaling pipeline with safe per-column unit normalization.

Length-like columns are normalized to um and area-like columns to um^2. The
decision is made per column, so columns that are already in um are not scaled
again. Targets such as FL are kept in their original force units.
"""

import argparse
import sys

import pandas as pd

from ..config import CLEANED_DIR, SCALED_DIR
from src.data.units import normalize_units
from utils.paths import ensure_dir, iter_excel_files_for_batch, select_batch_dir


def scale_file(src, dst) -> pd.DataFrame:
    df = pd.read_excel(src, engine="openpyxl")
    df, unit_report = normalize_units(df)
    df.to_excel(dst, index=False, engine="openpyxl")
    return unit_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale one cleaned-data batch.")
    parser.add_argument("--batch", default=None, help="Batch folder under data/interim/cleaned. Defaults to the latest batch.")
    args = parser.parse_args()

    if not CLEANED_DIR.exists():
        sys.exit(f"Cleaned data directory not found: {CLEANED_DIR}")

    batch_dir = select_batch_dir(CLEANED_DIR, args.batch)
    batch_name = batch_dir.name if batch_dir != CLEANED_DIR else ""

    xlsx_files = iter_excel_files_for_batch(batch_dir)
    if not xlsx_files:
        sys.exit(f"No .xlsx files found in {batch_dir}")

    dst_root = SCALED_DIR / batch_name if batch_name else SCALED_DIR
    ensure_dir(dst_root)
    print(f"Scaling batch {batch_name or '(root)'}: {len(xlsx_files)} file(s) -> {dst_root}/")
    unit_reports = []
    for src in xlsx_files:
        relative_path = src.relative_to(batch_dir)
        dst = ensure_dir(dst_root / relative_path.parent) / src.name
        print(f"  {relative_path} ...", end=" ", flush=True)
        unit_report = scale_file(src, dst)
        if not unit_report.empty:
            unit_report.insert(0, "source_file", relative_path.as_posix())
            unit_reports.append(unit_report)
        print("done")

    if unit_reports:
        pd.concat(unit_reports, ignore_index=True).to_csv(
            dst_root / "unit_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print("All files scaled.")


if __name__ == "__main__":
    main()
