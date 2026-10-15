"""Insert horizontal and vertical neighbor midpoints into raw FLy workbooks.

Each workbook is processed independently.  Only original samples are used as
endpoints: generated rows never become endpoints for a second interpolation
pass.  This keeps the output to one midpoint between each pair of neighboring
observations along a horizontal or vertical line in the same Reynolds slice.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("re", "x", "z", "fly")
COORD_COLUMNS = ("re", "x", "z")


def midpoint_key(re_value: float, x_value: float, z_value: float) -> tuple[float, float, float]:
    """Use rounded coordinates so equivalent floating-point midpoints merge."""
    return (round(float(re_value), 12), round(float(x_value), 12), round(float(z_value), 12))


def normalize_source(source: pd.DataFrame, input_path: Path) -> tuple[pd.DataFrame, int]:
    """Keep usable rows and reduce any repeated physical position to one value."""
    source = source.loc[:, ~source.columns.astype(str).str.startswith("Unnamed")].copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError(f"{input_path.name} is missing required columns: {missing}")

    for column in REQUIRED_COLUMNS:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    valid = source.dropna(subset=REQUIRED_COLUMNS).copy()
    duplicate_rows = int(valid.duplicated(list(COORD_COLUMNS)).sum())
    if duplicate_rows == 0:
        return valid.reset_index(drop=True), 0

    aggregations: dict[str, str] = {}
    for column in valid.columns:
        if column not in COORD_COLUMNS:
            aggregations[column] = "mean" if pd.api.types.is_numeric_dtype(valid[column]) else "first"
    collapsed = valid.groupby(list(COORD_COLUMNS), as_index=False).agg(aggregations)
    return collapsed[valid.columns].reset_index(drop=True), duplicate_rows


def interpolate_row(
    first: pd.Series,
    second: pd.Series,
    numeric_columns: list[str],
    direction: str,
) -> dict[str, object]:
    """Linearly interpolate every numeric field at the pair midpoint."""
    row: dict[str, object] = {}
    for column in first.index:
        if column == "re":
            row[column] = float(first[column])
        elif column == "x":
            row[column] = (float(first[column]) + float(second[column])) / 2.0
        elif column == "z":
            row[column] = (float(first[column]) + float(second[column])) / 2.0
        elif column in numeric_columns:
            left = pd.to_numeric(pd.Series([first[column]]), errors="coerce").iloc[0]
            right = pd.to_numeric(pd.Series([second[column]]), errors="coerce").iloc[0]
            row[column] = (left + right) / 2.0 if pd.notna(left) and pd.notna(right) else np.nan
        else:
            # The raw FLy workbooks are numeric, but retain a deterministic value
            # if a future file includes labels or notes.
            row[column] = first[column] if first[column] == second[column] else first[column]
    row["interpolation_direction"] = direction
    row["source_pair_count"] = 1
    return row


def interpolate_cell(
    corners: list[pd.Series],
    numeric_columns: list[str],
) -> dict[str, object]:
    """Create a cell center only when all four original corner points exist."""
    row: dict[str, object] = {}
    for column in corners[0].index:
        if column == "re":
            row[column] = float(corners[0][column])
        elif column == "x":
            row[column] = float(np.mean([float(corner[column]) for corner in corners]))
        elif column == "z":
            row[column] = float(np.mean([float(corner[column]) for corner in corners]))
        elif column in numeric_columns:
            values = pd.to_numeric(pd.Series([corner[column] for corner in corners]), errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else np.nan
        else:
            row[column] = corners[0][column]
    row["interpolation_direction"] = "cell_center"
    row["source_pair_count"] = 4
    return row


def merge_generated_rows(rows: list[dict[str, object]], numeric_columns: list[str], columns: list[str]) -> dict[str, object]:
    """Merge the rare coordinate obtained from both horizontal and vertical pairs."""
    if len(rows) == 1:
        return rows[0]

    merged = rows[0].copy()
    for column in numeric_columns:
        values = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce")
        merged[column] = float(values.mean()) if values.notna().any() else np.nan
    directions = sorted({str(row["interpolation_direction"]) for row in rows})
    merged["interpolation_direction"] = "+".join(directions)
    merged["source_pair_count"] = len(rows)
    for column in columns:
        if column not in merged:
            merged[column] = rows[0].get(column)
    return merged


def build_interpolated_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add axis midpoints and centers of valid four-corner original cells."""
    numeric_columns = [column for column in source.columns if pd.api.types.is_numeric_dtype(source[column])]
    candidates: dict[tuple[float, float, float], list[dict[str, object]]] = defaultdict(list)
    horizontal_pairs = 0
    vertical_pairs = 0
    cell_centers = 0

    for _, re_group in source.groupby("re", sort=True):
        point_lookup = {
            (round(float(row.x), 12), round(float(row.z), 12)): row
            for row in re_group.itertuples(index=False)
        }
        horizontal_edges: set[tuple[float, float, float]] = set()
        vertical_edges: set[tuple[float, float, float]] = set()

        for _, line in re_group.groupby("z", sort=True):
            ordered = line.sort_values("x").reset_index(drop=True)
            for index in range(len(ordered) - 1):
                row = interpolate_row(ordered.iloc[index], ordered.iloc[index + 1], numeric_columns, "horizontal")
                candidates[midpoint_key(row["re"], row["x"], row["z"])].append(row)
                horizontal_edges.add((
                    round(float(ordered.iloc[index]["z"]), 12),
                    round(float(ordered.iloc[index]["x"]), 12),
                    round(float(ordered.iloc[index + 1]["x"]), 12),
                ))
                horizontal_pairs += 1

        for _, line in re_group.groupby("x", sort=True):
            ordered = line.sort_values("z").reset_index(drop=True)
            for index in range(len(ordered) - 1):
                row = interpolate_row(ordered.iloc[index], ordered.iloc[index + 1], numeric_columns, "vertical")
                candidates[midpoint_key(row["re"], row["x"], row["z"])].append(row)
                vertical_edges.add((
                    round(float(ordered.iloc[index]["x"]), 12),
                    round(float(ordered.iloc[index]["z"]), 12),
                    round(float(ordered.iloc[index + 1]["z"]), 12),
                ))
                vertical_pairs += 1

        vertical_next = {
            (x_value, z_low): z_high
            for x_value, z_low, z_high in vertical_edges
        }
        # A cell is valid only if each of its four sides joins neighboring
        # original points, so a center never bridges a missing part of the mesh.
        for z_low, x_low, x_high in horizontal_edges:
            z_high = vertical_next.get((x_low, z_low))
            if z_high is None:
                continue
            if (z_high, x_low, x_high) not in horizontal_edges:
                continue
            if (x_high, z_low, z_high) not in vertical_edges:
                continue
            corner_keys = [
                (x_low, z_low),
                (x_high, z_low),
                (x_low, z_high),
                (x_high, z_high),
            ]
            if not all(key in point_lookup for key in corner_keys):
                continue
            corners = [pd.Series(point_lookup[key]._asdict()) for key in corner_keys]
            cell = interpolate_cell(corners, numeric_columns)
            candidates[midpoint_key(cell["re"], cell["x"], cell["z"])].append(cell)
            cell_centers += 1

    original_keys = {midpoint_key(row.re, row.x, row.z) for row in source[["re", "x", "z"]].itertuples(index=False)}
    generated_rows = [
        merge_generated_rows(rows, numeric_columns, source.columns.tolist())
        for key, rows in candidates.items()
        if key not in original_keys
    ]
    generated = pd.DataFrame(generated_rows, columns=[*source.columns, "interpolation_direction", "source_pair_count"])
    original = source.copy()
    original["interpolation_direction"] = "original"
    original["source_pair_count"] = 0
    original["generated"] = False
    generated["generated"] = True

    result = pd.concat([original, generated], ignore_index=True)
    result = result.sort_values(["re", "z", "x", "generated"], kind="stable").reset_index(drop=True)
    return result, {
        "horizontal_source_pairs": horizontal_pairs,
        "vertical_source_pairs": vertical_pairs,
        "cell_center_source_quads": cell_centers,
        "candidate_coordinate_count": len(candidates),
        "generated_coordinate_count": len(generated),
        "overlapping_axis_midpoint_coordinate_count": int(
            generated["interpolation_direction"].str.contains("\\+", regex=True).sum()
        ),
    }


def interpolate_file(input_path: Path, output_path: Path) -> dict[str, object]:
    source_file = pd.read_excel(input_path, sheet_name=0, engine="openpyxl")
    source, duplicate_rows = normalize_source(source_file, input_path)
    result, counts = build_interpolated_frame(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False, sheet_name="FLy_interpolated", engine="openpyxl")

    generated_fly = pd.to_numeric(result.loc[result["generated"], "fly"], errors="coerce")
    source_fly = pd.to_numeric(source["fly"], errors="coerce")
    return {
        "source_file": input_path.as_posix(),
        "output_file": output_path.as_posix(),
        "source_rows_total": int(len(source_file)),
        "valid_original_rows": int(len(source)),
        "invalid_rows_excluded": int(len(source_file) - len(source)),
        "duplicate_position_rows_aggregated": duplicate_rows,
        "rows_out": int(len(result)),
        "rows_added": int(result["generated"].sum()),
        "horizontal_rows_added": int((result["interpolation_direction"] == "horizontal").sum()),
        "vertical_rows_added": int((result["interpolation_direction"] == "vertical").sum()),
        "both_direction_rows_added": int((result["source_pair_count"] > 1).sum()),
        "generated_fly_nan_rows": int(generated_fly.isna().sum()),
        "source_fly_min": float(source_fly.min()),
        "source_fly_max": float(source_fly.max()),
        "generated_fly_min": float(generated_fly.min()),
        "generated_fly_max": float(generated_fly.max()),
        **counts,
    }


def run_batch(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    paths = sorted(path for path in input_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if not paths:
        raise ValueError(f"No .xlsx files found under {input_dir}")

    reports: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        output_path = output_dir / f"{path.stem}-neighbor-midpoints.xlsx"
        print(f"[neighbor midpoint {index}/{len(paths)}] {path.name}", flush=True)
        reports.append(interpolate_file(path, output_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(reports)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_dir": input_dir.as_posix(),
                "output_dir": output_dir.as_posix(),
                "method": "one linear midpoint per original horizontal or vertical neighbor pair within each re",
                "source_sheet": "first worksheet only",
                "file_count": len(paths),
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert horizontal and vertical FLy neighbor midpoints.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/rawdata/2026-09-26/FLy"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpolated_2026-09-26-FLy-neighbor-midpoints"))
    args = parser.parse_args()

    summary = run_batch(args.input_dir, args.output_dir)
    print(
        summary[
            [
                "source_file",
                "valid_original_rows",
                "rows_out",
                "rows_added",
                "horizontal_rows_added",
                "vertical_rows_added",
                "both_direction_rows_added",
                "generated_fly_nan_rows",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
