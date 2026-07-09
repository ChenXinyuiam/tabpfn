"""Build the 2026-05-25 FD batch by adding distance features to 2026-05-22.

For each workbook under data/rawdata/2026-05-22, the script looks for a
geometry-matching workbook under data/rawdata/2026-05-24 and inserts
dist_AB_horizontal and dist_AC_horizontal. Files without a matching distance
workbook are skipped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DISTANCE_COLUMNS = ["dist_AB_horizontal", "dist_AC_horizontal"]


def workbook_key(path: Path) -> str:
    match = re.match(r"^(\d+-\d+[^-]*)", path.stem)
    return match.group(1) if match else path.stem


def distance_lookup(distance_root: Path, decimals: int) -> dict[str, tuple[Path, pd.DataFrame, pd.DataFrame]]:
    lookup: dict[str, tuple[Path, pd.DataFrame, pd.DataFrame]] = {}
    for path in sorted(distance_root.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue

        df = pd.read_excel(
            path,
            usecols=["point_x", "point_y", *DISTANCE_COLUMNS],
            engine="openpyxl",
        )
        for col in ["point_x", "point_y", *DISTANCE_COLUMNS]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["_join_x"] = df["point_x"].round(decimals)
        df["_join_z"] = df["point_y"].round(decimals)

        grouped = df.groupby(["_join_x", "_join_z"], dropna=False)[DISTANCE_COLUMNS].nunique(dropna=False)
        inconsistent = grouped[(grouped[DISTANCE_COLUMNS[0]] > 1) | (grouped[DISTANCE_COLUMNS[1]] > 1)]
        if not inconsistent.empty:
            raise RuntimeError(
                f"{path.name} has {len(inconsistent)} coordinate keys with inconsistent distance values."
            )

        deduped = (
            df[["_join_x", "_join_z", *DISTANCE_COLUMNS]]
            .drop_duplicates(["_join_x", "_join_z"])
            .reset_index(drop=True)
        )
        nearest_df = (
            df[["point_x", "point_y", *DISTANCE_COLUMNS]]
            .drop_duplicates(["point_x", "point_y"])
            .reset_index(drop=True)
        )
        lookup[workbook_key(path)] = (path, deduped, nearest_df)
    return lookup


def insert_or_replace(df: pd.DataFrame, col: str, values: pd.Series, after: str) -> None:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
    loc = df.columns.get_loc(after) + 1 if after in df.columns else len(df.columns)
    df.insert(loc, col, values.to_numpy())


def merge_one_file(
    source_path: Path,
    source_root: Path,
    output_root: Path,
    distance_df: pd.DataFrame,
    nearest_df: pd.DataFrame,
    decimals: int,
) -> dict[str, object]:
    df = pd.read_excel(source_path, engine="openpyxl")
    required = ["X [ m ]", "Z [ m ]"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{source_path} is missing required coordinate columns: {missing}")

    df["_join_x"] = pd.to_numeric(df["X [ m ]"], errors="coerce").round(decimals)
    df["_join_z"] = pd.to_numeric(df["Z [ m ]"], errors="coerce").round(decimals)
    merged = df.merge(distance_df, on=["_join_x", "_join_z"], how="left", validate="many_to_one")

    unmatched_mask = merged[DISTANCE_COLUMNS].isna().any(axis=1)
    unmatched = int(unmatched_mask.sum())
    nearest_distance_um = np.full(len(merged), np.nan, dtype=float)
    if unmatched:
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn.fit(nearest_df[["point_x", "point_y"]].to_numpy(dtype=float))
        query = df.loc[unmatched_mask, ["X [ m ]", "Z [ m ]"]].apply(pd.to_numeric, errors="coerce")
        dist, indices = nn.kneighbors(query.to_numpy(dtype=float))
        nearest_values = nearest_df.iloc[indices[:, 0]][DISTANCE_COLUMNS].reset_index(drop=True)
        merged.loc[unmatched_mask, DISTANCE_COLUMNS] = nearest_values.to_numpy()
        nearest_distance_um[np.flatnonzero(unmatched_mask.to_numpy())] = dist[:, 0] * 1e6

    insert_after = "z/H" if "z/H" in df.columns else "Z [ m ]"
    insert_or_replace(df, DISTANCE_COLUMNS[1], merged[DISTANCE_COLUMNS[1]], insert_after)
    insert_or_replace(df, DISTANCE_COLUMNS[0], merged[DISTANCE_COLUMNS[0]], insert_after)
    df.drop(columns=["_join_x", "_join_z"], inplace=True)

    relative = source_path.relative_to(source_root)
    output_path = output_root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")

    return {
        "source_file": source_path.as_posix(),
        "output_file": output_path.as_posix(),
        "rows": int(len(df)),
        "unmatched_rows": unmatched,
        "nearest_filled_rows": unmatched,
        "nearest_distance_um_mean": float(np.nanmean(nearest_distance_um)) if unmatched else 0.0,
        "nearest_distance_um_p95": float(np.nanpercentile(nearest_distance_um, 95)) if unmatched else 0.0,
        "nearest_distance_um_max": float(np.nanmax(nearest_distance_um)) if unmatched else 0.0,
        "dist_AB_missing_after_merge": int(df[DISTANCE_COLUMNS[0]].isna().sum()),
        "dist_AC_missing_after_merge": int(df[DISTANCE_COLUMNS[1]].isna().sum()),
    }


def build_dataset(source_root: Path, distance_root: Path, output_root: Path, decimals: int) -> pd.DataFrame:
    lookup = distance_lookup(distance_root, decimals)
    rows: list[dict[str, object]] = []

    for source_path in sorted(source_root.rglob("*.xlsx")):
        if source_path.name.startswith("~$"):
            continue

        key = workbook_key(source_path)
        matched = lookup.get(key)
        if matched is None:
            rows.append(
                {
                    "source_file": source_path.as_posix(),
                    "output_file": "",
                    "rows": np.nan,
                    "unmatched_rows": np.nan,
                    "nearest_filled_rows": np.nan,
                    "nearest_distance_um_mean": np.nan,
                    "nearest_distance_um_p95": np.nan,
                    "nearest_distance_um_max": np.nan,
                    "dist_AB_missing_after_merge": np.nan,
                    "dist_AC_missing_after_merge": np.nan,
                    "status": "skipped_no_matching_2026_05_24_file",
                    "distance_file": "",
                }
            )
            continue

        distance_path, distance_df, nearest_df = matched
        result = merge_one_file(source_path, source_root, output_root, distance_df, nearest_df, decimals)
        result["status"] = "written"
        result["distance_file"] = distance_path.as_posix()
        rows.append(result)

    manifest = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_root / "manifest_2026-05-25.csv", index=False, encoding="utf-8-sig")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data/rawdata/2026-05-25 with distance features.")
    parser.add_argument("--source-root", type=Path, default=Path("data/rawdata/2026-05-22"))
    parser.add_argument("--distance-root", type=Path, default=Path("data/rawdata/2026-05-24"))
    parser.add_argument("--output-root", type=Path, default=Path("data/rawdata/2026-05-25"))
    parser.add_argument("--decimals", type=int, default=12)
    args = parser.parse_args()

    manifest = build_dataset(args.source_root, args.distance_root, args.output_root, args.decimals)
    written = manifest[manifest["status"] == "written"]
    skipped = manifest[manifest["status"] != "written"]
    print(f"Written files: {len(written)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"Rows written: {int(written['rows'].fillna(0).sum())}")
    print(f"Unmatched rows: {int(written['unmatched_rows'].fillna(0).sum())}")
    print(f"Nearest-filled rows: {int(written['nearest_filled_rows'].fillna(0).sum())}")
    print(f"Missing dist_AB after merge: {int(written['dist_AB_missing_after_merge'].fillna(0).sum())}")
    print(f"Missing dist_AC after merge: {int(written['dist_AC_missing_after_merge'].fillna(0).sum())}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
