"""
Data cleaning for processed FL datasets (2026-04-22).

Bad values removed:
  - Rows where geometry_valid == False   (10 rows in 750-530)
  - Rows where FL is NaN                 (1 row in 770-330)
  - Spurious trailing column Unnamed: 29 (690-340 only)

Output: {label}_cleaned.xlsx  →  same directory as input
"""

import os
import pandas as pd

DATA_DIR = os.path.join("data", "processed", "2026-04-22")


def clean_file(fname: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_excel(path)
    label = fname.replace("-FL.xlsx", "")
    n_raw = len(df)

    # ── drop trailing empty column ────────────────────────────────────────
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        print(f"  [{label}] dropped columns: {unnamed}")

    # ── drop invalid geometry rows ────────────────────────────────────────
    if "geometry_valid" in df.columns:
        bad_geom = (~df["geometry_valid"]).sum()
        if bad_geom:
            bad_rows = df[~df["geometry_valid"]][["X", "Z"]].drop_duplicates()
            print(f"  [{label}] removing {bad_geom} rows with geometry_valid=False "
                  f"at (X,Z): {list(bad_rows.itertuples(index=False, name=None))}")
            df = df[df["geometry_valid"]].copy()

    # ── drop rows with NaN FL ─────────────────────────────────────────────
    null_fl = df["FL"].isna().sum()
    if null_fl:
        nan_rows = df[df["FL"].isna()][["X", "Z"]].drop_duplicates()
        print(f"  [{label}] removing {null_fl} rows with FL=NaN "
              f"at (X,Z): {list(nan_rows.itertuples(index=False, name=None))}")
        df = df[df["FL"].notna()].copy()

    n_clean = len(df)
    print(f"  [{label}] rows: {n_raw} → {n_clean}  (removed {n_raw - n_clean})")
    return df


def main():
    fl_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("-FL.xlsx"))
    print(f"Found {len(fl_files)} FL files in {DATA_DIR}\n")

    for fname in fl_files:
        label = fname.replace("-FL.xlsx", "")
        df_clean = clean_file(fname)

        out_path = os.path.join(DATA_DIR, f"{label}_cleaned.xlsx")
        df_clean.to_excel(out_path, index=False)
        print(f"  [{label}] saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
