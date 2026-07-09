"""Interpolate each predicted FL workbook independently within each Re slice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.predict_fl_at_fd_points_tabpfn import augment_fl_points, interpolate_values


REQUIRED_COLUMNS = [
    "bc", "h", "angle_a", "angle_b", "angle_c", "re", "x", "z", "fl", "u_pred",
]


def valid_rows(df: pd.DataFrame) -> pd.Series:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required predicted FL columns: {missing}")
    mask = df[REQUIRED_COLUMNS].notna().all(axis=1)
    mask &= (df["bc"] > 0) & (df["h"] > 0) & (df["re"] > 0)
    return mask


def interpolate_u_pred(original: pd.DataFrame, augmented: pd.DataFrame) -> pd.DataFrame:
    out = augmented.copy()
    out["u_pred"] = np.nan
    out["u_pred_fill_method"] = "missing_re"

    for re_value, target_group in out.groupby("re", sort=True):
        source_group = original[original["re"] == re_value].dropna(subset=["x", "z", "u_pred"])
        if source_group.empty:
            continue
        source_points = source_group[["x", "z"]].to_numpy(dtype=float)
        target_points = target_group[["x", "z"]].to_numpy(dtype=float)
        values, methods = interpolate_values(
            source_points,
            source_group["u_pred"].to_numpy(dtype=float),
            target_points,
        )

        original_values = source_group.groupby(["x", "z"], as_index=False)["u_pred"].mean()
        original_lookup = {
            (round(float(row.x), 12), round(float(row.z), 12)): float(row.u_pred)
            for row in original_values.itertuples(index=False)
        }
        for index, point in enumerate(target_points):
            key = (round(float(point[0]), 12), round(float(point[1]), 12))
            if key in original_lookup:
                values[index] = original_lookup[key]
                methods[index] = "original"

        out.loc[target_group.index, "u_pred"] = values
        out.loc[target_group.index, "u_pred_fill_method"] = methods

    if out["u_pred"].isna().any():
        raise ValueError(f"Interpolation left {int(out['u_pred'].isna().sum())} rows without u_pred")
    return out


def process_file(input_path: Path, output_path: Path) -> dict[str, object]:
    source = pd.read_excel(input_path, engine="openpyxl")
    mask = valid_rows(source)
    valid = source.loc[mask].reset_index(drop=True)
    augmented = augment_fl_points(valid)
    augmented = interpolate_u_pred(valid, augmented)
    augmented["source_prediction_file"] = input_path.name
    if "fd_training_file" in source.columns and source["fd_training_file"].notna().any():
        augmented["fd_training_file"] = source["fd_training_file"].dropna().iloc[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_excel(output_path, index=False, engine="openpyxl")

    duplicate_positions = int(len(valid) - len(valid.drop_duplicates(["re", "x", "z"])))
    return {
        "source_file": input_path.as_posix(),
        "output_file": output_path.as_posix(),
        "rows_in": int(len(source)),
        "valid_rows_in": int(len(valid)),
        "invalid_rows_excluded": int((~mask).sum()),
        "duplicate_position_rows_aggregated": duplicate_positions,
        "rows_out": int(len(augmented)),
        "rows_added": int(augmented["generated"].sum()),
        "original_rows_out": int((~augmented["generated"]).sum()),
        "re_count": int(augmented["re"].nunique()),
        "fl_method_counts": augmented["fl_fill_method"].value_counts().to_dict(),
        "u_pred_method_counts": augmented["u_pred_fill_method"].value_counts().to_dict(),
        "u_pred_min": float(augmented["u_pred"].min()),
        "u_pred_mean": float(augmented["u_pred"].mean()),
        "u_pred_max": float(augmented["u_pred"].max()),
    }


def run_batch(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    paths = sorted(
        path for path in input_dir.glob("*-FL-with-u-pred.xlsx")
        if not path.name.startswith("~$")
    )
    if not paths:
        raise ValueError(f"No *-FL-with-u-pred.xlsx files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        output_path = output_dir / path.name.replace("-FL-with-u-pred.xlsx", "-FL-with-u-pred-interpolated.xlsx")
        print(f"[interpolate {index + 1}/{len(paths)}] {path.name}", flush=True)
        reports.append(process_file(path, output_path))
    summary = pd.DataFrame(reports)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "design": "each file interpolated independently; each Re interpolated independently",
        "interpolated_targets": ["fl", "u_pred"],
        "file_count": len(paths),
        "reports": reports,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate predicted FL files independently.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/fd_to_matching_fl_2026-06-20"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fd_to_matching_fl_2026-06-20-interpolated"),
    )
    args = parser.parse_args()
    summary = run_batch(args.input_dir, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
