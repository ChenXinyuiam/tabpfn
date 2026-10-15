"""Extract a smoothed red/blue boundary feature from one Fzong sample.

One sample is one source Excel file + one curvature radius R + one Reynolds
number re. Smoothing is performed independently on each horizontal z row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_COL_CANDIDATES = ("Fzong", "Fzong2")
RADIUS_COL_CANDIDATES = ("R", "r", "radius", "curvature_radius", "曲率半径")
RE_COL_CANDIDATES = ("re", "Re", "RE", "雷诺数")
X_COL_CANDIDATES = ("x", "X")
Z_COL_CANDIDATES = ("z", "Z")


@dataclass(frozen=True)
class SourceColumns:
    x: str
    z: str
    radius: str
    re_value: str
    target: str


def _find_column(columns: list[object], candidates: tuple[str, ...], *, required_name: str) -> str:
    exact = {str(column): str(column) for column in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]

    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lowered:
            return lowered[key]

    raise ValueError(f"Missing required {required_name} column; expected one of {list(candidates)}")


def resolve_source_columns(df: pd.DataFrame) -> SourceColumns:
    columns = list(df.columns)
    return SourceColumns(
        x=_find_column(columns, X_COL_CANDIDATES, required_name="x-coordinate"),
        z=_find_column(columns, Z_COL_CANDIDATES, required_name="z-coordinate"),
        radius=_find_column(columns, RADIUS_COL_CANDIDATES, required_name="curvature radius"),
        re_value=_find_column(columns, RE_COL_CANDIDATES, required_name="Reynolds number"),
        target=_find_column(columns, TARGET_COL_CANDIDATES, required_name="Fzong target"),
    )


def numeric_source_rows(df: pd.DataFrame, columns: SourceColumns) -> pd.DataFrame:
    """Return standardized numeric x/z/R/re/force rows without changing input."""
    out = df[[columns.x, columns.z, columns.radius, columns.re_value, columns.target]].copy()
    out.columns = ["x", "z", "R", "re", "force"]
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def collapse_duplicate_coordinates(sample: pd.DataFrame) -> pd.DataFrame:
    """Represent each physical coordinate once by its mean force value."""
    required = {"x", "z", "force"}
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"Sample is missing required columns: {missing}")

    signed = sample.assign(
        positive_source_rows=sample["force"].gt(0).astype(int),
        negative_source_rows=sample["force"].lt(0).astype(int),
        zero_source_rows=sample["force"].eq(0).astype(int),
    )
    points = (
        signed.groupby(["x", "z"], as_index=False, sort=True)
        .agg(
            force_raw=("force", "mean"),
            source_rows=("force", "size"),
            positive_source_rows=("positive_source_rows", "sum"),
            negative_source_rows=("negative_source_rows", "sum"),
            zero_source_rows=("zero_source_rows", "sum"),
        )
        .sort_values(["z", "x"], kind="stable")
        .reset_index(drop=True)
    )
    points["duplicate_sign_conflict"] = (points["positive_source_rows"] > 0) & (points["negative_source_rows"] > 0)
    points["sign_raw"] = np.sign(points["force_raw"].to_numpy(dtype=float)).astype(int)
    return points


def _validate_window_size(window_size: int) -> None:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer greater than or equal to 3")


def smooth_horizontal_rows(points: pd.DataFrame, *, window_size: int = 3) -> pd.DataFrame:
    """Apply a centered median to force values along x within each z row.

    Coordinates without a complete centered window, including row endpoints for
    window_size=3, retain their original force value.
    """
    _validate_window_size(window_size)
    required = {"x", "z", "force_raw", "sign_raw"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"Point table is missing required columns: {missing}")

    half_window = window_size // 2
    rows: list[pd.DataFrame] = []
    for _, row in points.groupby("z", sort=True):
        ordered = row.sort_values("x", kind="stable").copy()
        raw_force = ordered["force_raw"].to_numpy(dtype=float)
        smoothed_force = raw_force.copy()
        if len(raw_force) >= window_size:
            windows = np.lib.stride_tricks.sliding_window_view(raw_force, window_size)
            smoothed_force[half_window:-half_window] = np.median(windows, axis=1)
        ordered["force_smoothed"] = smoothed_force
        ordered["sign_smoothed"] = np.sign(smoothed_force).astype(int)
        ordered["sign_changed"] = ordered["sign_raw"] != ordered["sign_smoothed"]
        rows.append(ordered)

    if not rows:
        empty = points.copy()
        empty["force_smoothed"] = pd.Series(dtype=float)
        empty["sign_smoothed"] = pd.Series(dtype=int)
        empty["sign_changed"] = pd.Series(dtype=bool)
        return empty
    return pd.concat(rows, ignore_index=True)


def _compressed_nonzero_signs(signs: np.ndarray) -> list[int]:
    nonzero = [int(value) for value in signs if value != 0]
    if not nonzero:
        return []
    compressed = [nonzero[0]]
    for value in nonzero[1:]:
        if value != compressed[-1]:
            compressed.append(value)
    return compressed


def _pattern_counts(points: pd.DataFrame, sign_column: str) -> dict[str, int]:
    immediate_rbr = 0
    immediate_brb = 0
    broad_rbr_rows = 0
    broad_brb_rows = 0

    for _, row in points.groupby("z", sort=False):
        signs = row.sort_values("x", kind="stable")[sign_column].to_numpy(dtype=int)
        for index in range(1, len(signs) - 1):
            triple = tuple(signs[index - 1 : index + 2])
            immediate_rbr += int(triple == (1, -1, 1))
            immediate_brb += int(triple == (-1, 1, -1))

        runs = _compressed_nonzero_signs(signs)
        broad_rbr_rows += int(any(runs[index : index + 3] == [1, -1, 1] for index in range(len(runs) - 2)))
        broad_brb_rows += int(any(runs[index : index + 3] == [-1, 1, -1] for index in range(len(runs) - 2)))

    return {
        "immediate_rbr": immediate_rbr,
        "immediate_brb": immediate_brb,
        "broad_rbr_rows": broad_rbr_rows,
        "broad_brb_rows": broad_brb_rows,
    }


def _boundary_values(points: pd.DataFrame, sign_column: str) -> dict[str, float | bool]:
    red = points[points[sign_column] > 0]
    blue = points[points[sign_column] < 0]
    has_red = not red.empty
    has_blue = not blue.empty
    has_both = has_red and has_blue

    if has_both:
        rightmost_red_x = float(red["x"].max())
        leftmost_blue_x = float(blue["x"].min())
        midpoint_x = (rightmost_red_x + leftmost_blue_x) / 2.0
        distance_abs = abs(rightmost_red_x - leftmost_blue_x)
        gap_signed = leftmost_blue_x - rightmost_red_x
    else:
        rightmost_red_x = np.nan
        leftmost_blue_x = np.nan
        midpoint_x = np.nan
        distance_abs = np.nan
        gap_signed = np.nan

    return {
        "has_red": has_red,
        "has_blue": has_blue,
        "has_both_signs": has_both,
        "rightmost_red_x": rightmost_red_x,
        "leftmost_blue_x": leftmost_blue_x,
        "boundary_midpoint_x": midpoint_x,
        "boundary_distance_abs": distance_abs,
        "boundary_gap_signed": gap_signed,
    }


def spatial_distribution_features(points: pd.DataFrame) -> dict[str, float]:
    """Dimensionless within-sample descriptors, computed without labels."""
    x_min, x_max = points["x"].min(), points["x"].max()
    span = x_max - x_min
    boundary = _boundary_values(points, "sign_smoothed")
    red = points.loc[points["sign_smoothed"] > 0].groupby("z")["x"].max()
    blue = points.loc[points["sign_smoothed"] < 0].groupby("z")["x"].min()
    rows = pd.concat([red.rename("red"), blue.rename("blue")], axis=1).dropna()
    midpoints = (rows["red"] + rows["blue"]) / 2
    gaps = rows["blue"] - rows["red"]
    valid_span = np.isfinite(span) and span > 0
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "boundary_midpoint_norm": float((boundary["boundary_midpoint_x"] - x_min) / span) if valid_span else np.nan,
        "boundary_gap_norm": float(boundary["boundary_gap_signed"] / span) if valid_span else np.nan,
        "red_fraction": float((points["sign_smoothed"] > 0).mean()) if len(points) else np.nan,
        "row_midpoint_iqr_norm": float((midpoints.quantile(0.75) - midpoints.quantile(0.25)) / span)
        if valid_span and len(rows)
        else np.nan,
        "row_gap_median_norm": float(gaps.median() / span) if valid_span and len(rows) else np.nan,
        "row_gap_q90_norm": float(gaps.quantile(0.9) / span) if valid_span and len(rows) else np.nan,
    }


def extract_focus_boundary_feature(
    sample: pd.DataFrame,
    *,
    window_size: int = 3,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Calculate boundary features and return the auditable point-level table."""
    points = collapse_duplicate_coordinates(sample)
    smoothed = smooth_horizontal_rows(points, window_size=window_size)

    raw_boundary = _boundary_values(smoothed, "sign_raw")
    smooth_boundary = _boundary_values(smoothed, "sign_smoothed")
    raw_patterns = _pattern_counts(smoothed, "sign_raw")
    smooth_patterns = _pattern_counts(smoothed, "sign_smoothed")

    x_span = float(smoothed["x"].max() - smoothed["x"].min()) if not smoothed.empty else np.nan
    distance_abs = float(smooth_boundary["boundary_distance_abs"])
    distance_norm = distance_abs / x_span if np.isfinite(distance_abs) and x_span > 0 else np.nan

    duplicate_groups = int((smoothed["source_rows"] > 1).sum())
    duplicate_rows = int(smoothed.loc[smoothed["source_rows"] > 1, "source_rows"].sum())
    feature = {
        "raw_rows": int(len(sample)),
        "unique_coordinate_points": int(len(smoothed)),
        "z_row_count": int(smoothed["z"].nunique()),
        "x_span": x_span,
        "smoothing_method": f"centered_median_{window_size}",
        "smoothed_sign_changes": int(smoothed["sign_changed"].sum()),
        "duplicate_coordinate_groups": duplicate_groups,
        "duplicate_coordinate_rows": duplicate_rows,
        "duplicate_sign_conflicts": int(smoothed["duplicate_sign_conflict"].sum()),
        "raw_red_points": int((smoothed["sign_raw"] > 0).sum()),
        "raw_blue_points": int((smoothed["sign_raw"] < 0).sum()),
        "raw_zero_points": int((smoothed["sign_raw"] == 0).sum()),
        "smoothed_red_points": int((smoothed["sign_smoothed"] > 0).sum()),
        "smoothed_blue_points": int((smoothed["sign_smoothed"] < 0).sum()),
        "smoothed_zero_points": int((smoothed["sign_smoothed"] == 0).sum()),
        "rightmost_red_x_raw": raw_boundary["rightmost_red_x"],
        "leftmost_blue_x_raw": raw_boundary["leftmost_blue_x"],
        "boundary_midpoint_x_raw": raw_boundary["boundary_midpoint_x"],
        "boundary_distance_abs_raw": raw_boundary["boundary_distance_abs"],
        "boundary_gap_signed_raw": raw_boundary["boundary_gap_signed"],
        "rightmost_red_x": smooth_boundary["rightmost_red_x"],
        "leftmost_blue_x": smooth_boundary["leftmost_blue_x"],
        "boundary_midpoint_x": smooth_boundary["boundary_midpoint_x"],
        "boundary_distance_abs": smooth_boundary["boundary_distance_abs"],
        "boundary_gap_signed": smooth_boundary["boundary_gap_signed"],
        "boundary_distance_norm": distance_norm,
        "has_red": smooth_boundary["has_red"],
        "has_blue": smooth_boundary["has_blue"],
        "has_both_signs": smooth_boundary["has_both_signs"],
        "immediate_rbr_before": raw_patterns["immediate_rbr"],
        "immediate_rbr_after": smooth_patterns["immediate_rbr"],
        "immediate_brb_before": raw_patterns["immediate_brb"],
        "immediate_brb_after": smooth_patterns["immediate_brb"],
        "broad_rbr_rows_before": raw_patterns["broad_rbr_rows"],
        "broad_rbr_rows_after": smooth_patterns["broad_rbr_rows"],
        "broad_brb_rows_before": raw_patterns["broad_brb_rows"],
        "broad_brb_rows_after": smooth_patterns["broad_brb_rows"],
    }
    feature.update(spatial_distribution_features(smoothed))
    return feature, smoothed
