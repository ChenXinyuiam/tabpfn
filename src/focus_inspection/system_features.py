"""Task-neutral wrapper around the maintained smoothed boundary feature extractor."""

import numpy as np
import pandas as pd

from .focus_boundary_feature import extract_focus_boundary_feature

FOCUS_FEATURES = [
    "ab",
    "ac",
    "bc",
    "h",
    "re",
    "radius_um",
    "boundary_distance_norm",
    "boundary_midpoint_norm",
    "boundary_gap_norm",
    "red_fraction",
    "row_midpoint_iqr_norm",
    "row_gap_median_norm",
    "row_gap_q90_norm",
]


def section_features(points, force_column="target", window_size=3):
    rows = []
    if points.empty or not np.isfinite(points[force_column].to_numpy(dtype=float)).all():
        raise ValueError("Focus features require a complete finite point field")
    for sample_id, part in points.groupby("sample_id", sort=True):
        for column in ["geometry_family_id", "experiment_id", "re", "radius_um", "ab", "ac", "bc", "h"]:
            if part[column].nunique() != 1:
                raise ValueError("Sample contains inconsistent " + column)
        if part.duplicated(["x", "z"]).any():
            raise ValueError("Duplicate coordinates in focus sample")
        features, _ = extract_focus_boundary_feature(
            part[["x", "z", force_column]].rename(columns={force_column: "force"}), window_size=window_size
        )
        first = part.iloc[0]
        context = {
            c: first[c] for c in ["geometry_family_id", "experiment_id", "re", "radius_um", "ab", "ac", "bc", "h"]
        }
        rows.append({"sample_id": sample_id, **context, **features})
    return pd.DataFrame(rows)


def labeled_sections(points, samples, classes, *, force_column="target", window_size=3):
    features = section_features(points, force_column, window_size)
    if "label" not in samples:
        raise ValueError("Independent focus labels are required")
    out = features.merge(samples[["sample_id", "label"]], on="sample_id", validate="one_to_one", how="left")
    if out.label.isna().any() or not out.label.isin(classes).all():
        raise ValueError("Every focus sample must have a label in the declared class set")
    return out
