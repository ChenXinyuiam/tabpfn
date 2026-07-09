"""Adapters for the English-column FD/FL templates under docs/templates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LENGTH_COLS = [
    "bc", "ab", "ac", "h", "perimeter", "x", "y", "z",
    "prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc",
    "line_dist_to_ab", "line_dist_to_ac",
]
AREA_COLS = ["area"]

FL_COLUMNS = [
    "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c",
    "perimeter", "area", "re", "x", "z", "prep_dist_to_ab",
    "prep_dist_to_ac", "prep_dist_to_bc", "line_dist_to_ab",
    "line_dist_to_ac", "fl",
]
FD_COLUMNS = [
    "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c",
    "perimeter", "area", "re", "x", "y", "z", "x/H", "z/H",
    "prep_dist_to_ab", "prep_dist_to_ac", "dean_velocity_x",
    "dean_velocity_y",
]

ALIASES = {
    "BC": "bc", "三角形底边": "bc", "底边": "bc",
    "AB": "ab", "左边长度": "ab",
    "AC": "ac", "右边长度": "ac",
    "高": "h",
    "角度A": "angle_a", "角度1": "angle_a",
    "角度B": "angle_b", "角度2": "angle_b",
    "角度C": "angle_c", "角度3": "angle_c",
    "周长": "perimeter", "面积": "area",
    "Re": "re", "雷诺数": "re",
    "X": "x", "X [ m ]": "x", "point_x": "x",
    "Y": "y", "Y [ m ]": "y",
    "Z": "z", "Z [ m ]": "z", "point_y": "z",
    "到AB垂直距离": "prep_dist_to_ab", "点到左边距离": "prep_dist_to_ab",
    "点到AB的垂直距离": "prep_dist_to_ab",
    "dist_AB_horizontal": "prep_dist_to_ab",
    "到AC垂直距离": "prep_dist_to_ac", "点到右边距离": "prep_dist_to_ac",
    "点到AC的垂直距离": "prep_dist_to_ac",
    "dist_AC_horizontal": "prep_dist_to_ac",
    "到BC垂直距离": "prep_dist_to_bc", "点到底边距离": "prep_dist_to_bc",
    "点到BC的垂直距离": "prep_dist_to_bc",
    "到AB直线距离": "line_dist_to_ab",
    "点到AB的直线距离": "line_dist_to_ab",
    "到AC直线距离": "line_dist_to_ac",
    "点到AC的直线距离": "line_dist_to_ac",
    "FL": "fl", "Fly": "fl", "FLy": "fl", "target_fl_y": "fl",
    "target_fd_x": "dean_velocity_x", "target_fd_y": "dean_velocity_y",
}


def read_data_workbook(path: Path) -> pd.DataFrame:
    """Read a submitted workbook and reject the three-row documentation template."""
    df = pd.read_excel(path, engine="openpyxl")
    if len(df) <= 2 and str(df.columns[0]).strip() == "列名":
        raise ValueError(
            f"{path} is a blank documentation template. Submit a workbook whose first "
            "row contains the English field names and whose following rows contain data."
        )
    return df


def _numeric_multiplier(series: pd.Series, source_unit: str, *, area: bool = False) -> float:
    if source_unit == "um":
        return 1.0
    if source_unit == "m":
        return 1e12 if area else 1e6
    if source_unit != "auto":
        raise ValueError(f"Unsupported length unit: {source_unit}")
    values = pd.to_numeric(series, errors="coerce").dropna().abs()
    values = values[values > 0]
    median = float(values.median()) if len(values) else np.nan
    threshold = 1.0 if area else 0.1
    if np.isfinite(median) and median < threshold:
        return 1e12 if area else 1e6
    return 1.0


def standardize_template_frame(
    df: pd.DataFrame,
    *,
    kind: str,
    source_length_unit: str = "auto",
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map current and historical columns to the English template schema in um."""
    if kind not in {"fl", "fd"}:
        raise ValueError("kind must be 'fl' or 'fd'")
    original_columns = list(df.columns)
    rename = {
        source: target for source, target in ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    out = df.rename(columns=rename).copy()
    expected = FL_COLUMNS if kind == "fl" else FD_COLUMNS
    if target_column and target_column not in expected:
        expected = [*expected, target_column]
    numeric_columns = [column for column in expected if column in out.columns]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    unit_multipliers: dict[str, float] = {}
    for column in LENGTH_COLS:
        if column in out.columns:
            unit_multipliers[column] = _numeric_multiplier(out[column], source_length_unit)
            out[column] = out[column] * unit_multipliers[column]
    for column in AREA_COLS:
        if column in out.columns:
            unit_multipliers[column] = _numeric_multiplier(out[column], source_length_unit, area=True)
            out[column] = out[column] * unit_multipliers[column]

    # Some transitional FD files already use English names but omit values
    # that can be reconstructed from the supplied triangle geometry.
    if "bc" not in out.columns and {"area", "h"}.issubset(out.columns):
        out["bc"] = 2.0 * out["area"] / out["h"].replace(0, np.nan)
    if "area" not in out.columns and {"bc", "h"}.issubset(out.columns):
        out["area"] = out["bc"] * out["h"] / 2.0
    if {"bc", "angle_a", "angle_b", "angle_c"}.issubset(out.columns):
        sin_a = np.sin(np.deg2rad(out["angle_a"])).replace(0, np.nan)
        if "ab" not in out.columns:
            out["ab"] = out["bc"] * np.sin(np.deg2rad(out["angle_c"])) / sin_a
        if "ac" not in out.columns:
            out["ac"] = out["bc"] * np.sin(np.deg2rad(out["angle_b"])) / sin_a
    if "perimeter" not in out.columns and {"bc", "ab", "ac"}.issubset(out.columns):
        out["perimeter"] = out["bc"] + out["ab"] + out["ac"]

    if "h" in out.columns:
        positive_h = out["h"].where(out["h"].abs() > 1e-12)
        if "x" in out.columns:
            out["x/H"] = out["x"] / positive_h
        if "z" in out.columns:
            out["z/H"] = out["z"] / positive_h

    required = ["bc", "h", "angle_a", "angle_b", "angle_c", "re", "x", "z"]
    required.append(target_column or ("fl" if kind == "fl" else "dean_velocity_x"))
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Missing required {kind.upper()} columns after template mapping: {missing}")

    report = {
        "kind": kind,
        "target_column": target_column or ("fl" if kind == "fl" else "dean_velocity_x"),
        "rows": int(len(out)),
        "source_columns": original_columns,
        "renamed_columns": rename,
        "unit_multipliers": unit_multipliers,
        "missing_template_columns": [column for column in expected if column not in out.columns],
    }
    return out, report


def validate_template_frame(
    df: pd.DataFrame,
    *,
    kind: str,
    target_column: str | None = None,
) -> list[str]:
    errors: list[str] = []
    for column in ["bc", "h", "re"]:
        if column in df.columns and ((df[column].isna()) | (df[column] <= 0)).any():
            errors.append(f"{column} contains missing or non-positive values")
    angles = ["angle_a", "angle_b", "angle_c"]
    if all(column in df.columns for column in angles):
        angle_sum = df[angles].sum(axis=1)
        if (angle_sum.sub(180.0).abs() > 1.0).any():
            errors.append("triangle angle sum differs from 180 degrees by more than 1 degree")
    target = target_column or ("fl" if kind == "fl" else "dean_velocity_x")
    if target in df.columns and df[target].notna().sum() == 0:
        errors.append(f"{target} has no numeric values")
    return errors
