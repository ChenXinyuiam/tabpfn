"""Unit normalization helpers for FD/FL tabular data.
   将FD/FL中的所有数据做单位统一。 

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LENGTH_SCALE = 1e6
AREA_SCALE = 1e12

DEFAULT_LENGTH_COLUMNS = {
    "X",
    "Y",
    "Z",
    "X [ m ]",
    "Y [ m ]",
    "Z [ m ]",
    "BC",
    "AB",
    "AC",
    "底边",
    "三角形底边",
    "高",
    "周长",
    "重心坐标X",
    "重心坐标Y",
    "点到底边距离",
    "点到左边距离",
    "点到右边距离",
    "点到重心距离",
    "到BC垂直距离",
    "到AB垂直距离",
    "到AC垂直距离",
    "点到BC的垂直距离",
    "点到AB的垂直距离",
    "点到AC的垂直距离",
    "到AB直线距离",
    "到AC直线距离",
    "点到AB的直线距离",
    "点到AC的直线距离",
    "dist_AB_horizontal",
    "dist_AC_horizontal",
    "左边长度",
    "右边长度",
}

DEFAULT_AREA_COLUMNS = {
    "面积",
    "面积_calc",
}


@dataclass(frozen=True)
class UnitDecision:
    column: str
    kind: str
    median_abs_before: float | None
    scale: float
    inferred_input_unit: str
    output_unit: str


def numeric_median_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    return float(values.abs().median())


def infer_length_scale(column: str, median_abs: float | None) -> tuple[float, str]:
    """Infer whether a length-like column is meters or already micrometers."""
    if median_abs is None:
        return 1.0, "unknown"
    if "[ m ]" in column:
        return LENGTH_SCALE, "m"
    if median_abs == 0:
        return 1.0, "um_or_zero"
    if median_abs < 0.01:
        return LENGTH_SCALE, "m"
    return 1.0, "um"


def infer_area_scale(column: str, median_abs: float | None) -> tuple[float, str]:
    """Infer whether an area-like column is square meters or already square micrometers."""
    if median_abs is None:
        return 1.0, "unknown"
    if median_abs == 0:
        return 1.0, "um2_or_zero"
    if median_abs < 1e-3:
        return AREA_SCALE, "m2"
    return 1.0, "um2"


def normalize_units(
    df: pd.DataFrame,
    *,
    length_columns: set[str] | None = None,
    area_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize length columns to um and area columns to um^2.

    The current datasets are mixed: some columns are still in meters, while
    others are already in micrometers. Decisions are made per column from the
    column name and numeric scale, so already-normalized data is not multiplied
    a second time.
    """
    df = df.copy()
    length_columns = length_columns or DEFAULT_LENGTH_COLUMNS
    area_columns = area_columns or DEFAULT_AREA_COLUMNS
    decisions: list[UnitDecision] = []

    for column in df.columns:
        if column in length_columns:
            median_abs = numeric_median_abs(df[column])
            scale, input_unit = infer_length_scale(column, median_abs)
            df[column] = pd.to_numeric(df[column], errors="coerce") * scale
            decisions.append(
                UnitDecision(column, "length", median_abs, scale, input_unit, "um")
            )
        elif column in area_columns:
            median_abs = numeric_median_abs(df[column])
            scale, input_unit = infer_area_scale(column, median_abs)
            df[column] = pd.to_numeric(df[column], errors="coerce") * scale
            decisions.append(
                UnitDecision(column, "area", median_abs, scale, input_unit, "um2")
            )

    report = pd.DataFrame([decision.__dict__ for decision in decisions])
    return df, report

