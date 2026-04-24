"""
Scale physical feature columns in all .xlsx files under data/cleaned/.
Output is written to data/scaled/ (created if absent).

Scale factors:
  1e12 : 面积
  1e6  : X, Z, 周长, 重心坐标X, 重心坐标Y,
         点到底边距离, 点到左边距离, 点到右边距离
"""

import sys
import pathlib
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_DIR  = pathlib.Path("data/cleaned")
OUTPUT_DIR = pathlib.Path("data/scaled")

SCALE_1E12 = {"面积"}
SCALE_1E6  = {
    "X", "Z", "周长",
    "重心坐标X", "重心坐标Y",
    "点到底边距离", "点到左边距离", "点到右边距离", "点到重心距离",
    "FL",
}
# ------------------------------------------------------------------


def scale_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    df = pd.read_excel(src, engine="openpyxl")

    missing = (SCALE_1E12 | SCALE_1E6) - set(df.columns)
    if missing:
        print(f"  [WARN] columns not found in {src.name}: {missing}")

    for col in SCALE_1E12:
        if col in df.columns:
            df[col] = df[col] * 1e12

    for col in SCALE_1E6:
        if col in df.columns:
            df[col] = df[col] * 1e6

    df.to_excel(dst, index=False, engine="openpyxl")


def main() -> None:
    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(
        f for f in INPUT_DIR.glob("*.xlsx")
        if not f.name.startswith("~$")   # skip Excel lock files
    )

    if not xlsx_files:
        sys.exit(f"No .xlsx files found in {INPUT_DIR}")

    print(f"Processing {len(xlsx_files)} file(s)  →  {OUTPUT_DIR}/")
    for src in xlsx_files:
        dst = OUTPUT_DIR / src.name
        print(f"  {src.name}", end=" ... ", flush=True)
        scale_file(src, dst)
        print("done")

    print("All files processed.")


if __name__ == "__main__":
    main()
