import sys
import pathlib
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_DIR  = pathlib.Path("data/processedv2")
OUTPUT_DIR = pathlib.Path("data/processed/2026-04-22")

FL_COL = "FL"
# ------------------------------------------------------------------


def process_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    df = pd.read_excel(src, engine="openpyxl")

    if FL_COL not in df.columns:
        print(f"  [WARN] 'FL' column not found in {src.name}, skip")
        return

    # 核心操作：取反
    df[FL_COL] = -df[FL_COL]

    df.to_excel(dst, index=False, engine="openpyxl")


def main() -> None:
    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 只处理 *_sym.xlsx
    xlsx_files = sorted(
        f for f in INPUT_DIR.glob("*-sym.xlsx")
        if not f.name.startswith("~$")
    )

    if not xlsx_files:
        sys.exit(f"No *_sym.xlsx files found in {INPUT_DIR}")

    print(f"Processing {len(xlsx_files)} file(s) → {OUTPUT_DIR}/")

    for src in xlsx_files:
        dst = OUTPUT_DIR / src.name
        print(f"  {src.name}", end=" ... ", flush=True)

        process_file(src, dst)

        print("done")

    print("All files processed.")


if __name__ == "__main__":
    main()