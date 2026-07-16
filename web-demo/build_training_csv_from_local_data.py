"""Build server training.csv from local paired FD/FL workbooks.

This script targets the local 2026-06-20 data layout:

  data/rawdata/2026-06-20/FL/*-FL.xlsx
  data/rawdata/2026-06-20/FD/*-FD-*.xlsx

The current public demo API expects four directional targets:
  fl_x_pred, fl_y_pred, fd_x_pred, fd_y_pred

The available local files are single-projection data:
  FL target: fl or flx
  FD target: u

So this builder maps:
  fl_x_pred = fl/flx
  fd_x_pred = nearest same-Re FD u
  fl_y_pred = 0
  fd_y_pred = 0

That keeps the server interface compatible while honestly using the available
single-direction local data.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from tabpfn_service import FEATURE_COLUMNS, TARGET_COLUMNS


def normalize_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"-FL$", "", stem)
    stem = re.sub(r"-FD-.*$", "", stem)
    return stem


def to_meters(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    median = series.abs().median()
    if pd.notna(median) and median > 10:
        return series * 1e-6
    return series


def area_to_m2(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    median = series.abs().median()
    if pd.notna(median) and median > 1:
        return series * 1e-12
    return series


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["re"] = pd.to_numeric(df["re"], errors="coerce")
    out["x_m"] = to_meters(df["x"])
    out["z_m"] = to_meters(df["z"])
    out["bc_m"] = to_meters(df["bc"])
    out["ab_m"] = to_meters(df["ab"])
    out["ac_m"] = to_meters(df["ac"])
    out["h_m"] = to_meters(df["h"])
    out["angle_a"] = pd.to_numeric(df["angle_a"], errors="coerce")
    out["angle_b"] = pd.to_numeric(df["angle_b"], errors="coerce")
    out["angle_c"] = pd.to_numeric(df["angle_c"], errors="coerce")
    out["area_m2"] = area_to_m2(df["area"])
    out["perimeter_m"] = to_meters(df["perimeter"])
    out["x_over_h"] = pd.to_numeric(df.get("x/H", df["x"] / df["h"]), errors="coerce")
    out["z_over_h"] = pd.to_numeric(df.get("z/H", df["z"] / df["h"]), errors="coerce")
    out["dist_ab_m"] = to_meters(df["prep_dist_to_ab"])
    out["dist_ac_m"] = to_meters(df["prep_dist_to_ac"])
    out["dist_bc_m"] = to_meters(df.get("prep_dist_to_bc", df["z"]))
    centroid_x = (pd.to_numeric(df["bc"], errors="coerce") + pd.to_numeric(df["x"], errors="coerce") * 0) / 2
    centroid_y = pd.to_numeric(df["h"], errors="coerce") / 3
    out["centroid_x_m"] = to_meters(centroid_x)
    out["centroid_y_m"] = to_meters(centroid_y)
    out["dist_centroid_m"] = np.sqrt((out["x_m"] - out["centroid_x_m"]) ** 2 + (out["z_m"] - out["centroid_y_m"]) ** 2)
    return out


def load_fd_lookup(fd_path: Path) -> dict[float, tuple[NearestNeighbors, pd.DataFrame]]:
    df = pd.read_excel(fd_path, engine="openpyxl")
    required = ["re", "x", "z", "u"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{fd_path} missing FD columns: {missing}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    lookup = {}
    for re_value, group in df.groupby("re"):
        if len(group) < 1:
            continue
        nn = NearestNeighbors(n_neighbors=1).fit(group[["x", "z"]].to_numpy(dtype=float))
        lookup[float(re_value)] = (nn, group.reset_index(drop=True))
    return lookup


def attach_nearest_fd_u(fl_df: pd.DataFrame, fd_lookup: dict[float, tuple[NearestNeighbors, pd.DataFrame]]) -> tuple[np.ndarray, np.ndarray]:
    fd_values = np.full(len(fl_df), np.nan, dtype=float)
    distances = np.full(len(fl_df), np.nan, dtype=float)
    for re_value, index in fl_df.groupby("re").groups.items():
        key = float(re_value)
        if key not in fd_lookup:
            continue
        nn, fd_group = fd_lookup[key]
        points = fl_df.loc[index, ["x", "z"]].to_numpy(dtype=float)
        dist, nearest = nn.kneighbors(points)
        fd_values[index] = fd_group.iloc[nearest[:, 0]]["u"].to_numpy(dtype=float)
        distances[index] = dist[:, 0]
    return fd_values, distances


def build_training_csv(fl_dir: Path, fd_dir: Path, output_csv: Path, max_rows_per_file: int | None) -> dict[str, object]:
    fd_by_key: dict[str, list[Path]] = defaultdict(list)
    for fd_path in sorted(fd_dir.glob("*FD*.xlsx")):
        if not fd_path.name.startswith("~$"):
            fd_by_key[normalize_key(fd_path)].append(fd_path)

    chunks: list[pd.DataFrame] = []
    reports = []
    for fl_path in sorted(fl_dir.glob("*FL*.xlsx")):
        if fl_path.name.startswith("~$"):
            continue
        key = normalize_key(fl_path)
        fd_paths = fd_by_key.get(key)
        if not fd_paths:
            reports.append({"fl_file": fl_path.name, "status": "skipped_no_fd_match", "key": key})
            continue
        fd_path = sorted(fd_paths)[0]
        fl_df = pd.read_excel(fl_path, engine="openpyxl")
        target_col = "fl" if "fl" in fl_df.columns else "flx" if "flx" in fl_df.columns else None
        if target_col is None:
            reports.append({"fl_file": fl_path.name, "status": "skipped_no_fl_target", "key": key})
            continue
        required = ["re", "x", "z", "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c", "area", "perimeter", "prep_dist_to_ab", "prep_dist_to_ac"]
        missing = [column for column in required if column not in fl_df.columns]
        if missing:
            reports.append({"fl_file": fl_path.name, "status": "skipped_missing_columns", "key": key, "missing": missing})
            continue
        for column in [*required, target_col]:
            fl_df[column] = pd.to_numeric(fl_df[column], errors="coerce")
        fl_df = fl_df.dropna(subset=[*required, target_col]).reset_index(drop=True)
        if fl_df.empty:
            reports.append({"fl_file": fl_path.name, "status": "skipped_empty", "key": key})
            continue

        fd_lookup = load_fd_lookup(fd_path)
        fd_u, nearest_distance = attach_nearest_fd_u(fl_df, fd_lookup)
        features = feature_frame(fl_df)
        out = features.copy()
        out["fl_x_pred"] = fl_df[target_col].to_numpy(dtype=float)
        out["fl_y_pred"] = 0.0
        out["fd_x_pred"] = fd_u
        out["fd_y_pred"] = 0.0
        out["source_fl_file"] = fl_path.name
        out["source_fd_file"] = fd_path.name
        out["nearest_fd_distance_um"] = nearest_distance
        out = out.dropna(subset=FEATURE_COLUMNS + list(TARGET_COLUMNS.values())).reset_index(drop=True)
        if max_rows_per_file is not None and len(out) > max_rows_per_file:
            out = out.sample(n=max_rows_per_file, random_state=42).reset_index(drop=True)
        chunks.append(out)
        reports.append(
            {
                "fl_file": fl_path.name,
                "fd_file": fd_path.name,
                "key": key,
                "status": "ok",
                "rows": int(len(out)),
                "nearest_fd_distance_um_mean": float(out["nearest_fd_distance_um"].mean()) if len(out) else None,
                "nearest_fd_distance_um_max": float(out["nearest_fd_distance_um"].max()) if len(out) else None,
            }
        )

    if not chunks:
        raise RuntimeError("No paired FD/FL data could be converted into training rows.")
    training = pd.concat(chunks, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    training[FEATURE_COLUMNS + list(TARGET_COLUMNS.values())].to_csv(output_csv, index=False, encoding="utf-8")
    manifest = {
        "output_csv": output_csv.as_posix(),
        "rows": int(len(training)),
        "files_used": int(sum(1 for report in reports if report.get("status") == "ok")),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "target_mapping": {
            "fl_x_pred": "FL single projection from fl/flx",
            "fd_x_pred": "nearest same-Re FD u",
            "fl_y_pred": "0 placeholder because local FL-y labels are not available",
            "fd_y_pred": "0 placeholder because local FD-y labels are not available in 2026-06-20 projection files",
        },
        "reports": reports,
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TriFocus server training.csv from paired local FD/FL files.")
    parser.add_argument("--fl-dir", type=Path, default=Path("data/rawdata/2026-06-20/FL"))
    parser.add_argument("--fd-dir", type=Path, default=Path("data/rawdata/2026-06-20/FD"))
    parser.add_argument("--output-csv", type=Path, default=Path("models/tabpfn_server/training.csv"))
    parser.add_argument("--max-rows-per-file", type=int, default=800)
    args = parser.parse_args()

    manifest = build_training_csv(args.fl_dir, args.fd_dir, args.output_csv, args.max_rows_per_file)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
