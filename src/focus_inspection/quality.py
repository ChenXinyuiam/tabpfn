"""Explicit feature settings and input-quality checks shared by fit and predict."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.provenance import file_sha256

from .focus_boundary_feature import numeric_source_rows

FEATURE_SCHEMA_VERSION = 3
LENGTH_TO_UM = {"um": 1.0, "mm": 1000.0, "m": 1_000_000.0}


@dataclass(frozen=True)
class FeatureConfig:
    window_size: int = 3
    length_unit: str = "um"
    invalid_rows: str = "error"

    def __post_init__(self):
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be odd and at least 3")
        if self.length_unit not in LENGTH_TO_UM:
            raise ValueError("length_unit must be um, mm or m; units are never inferred")
        if self.invalid_rows not in ("error", "drop"):
            raise ValueError("invalid_rows must be error or drop")

    def to_dict(self):
        return asdict(self)


def feature_code_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        name: file_sha256(root / name)
        for name in (
            "focus_boundary_feature.py",
            "quality.py",
            "dataset.py",
            "build_focus_boundary_features.py",
            "geometry_integrity.py",
        )
    }


def checked_source_rows(frame, columns, config: FeatureConfig):
    clean = numeric_source_rows(frame, columns)
    dropped = len(frame) - len(clean)
    if dropped and config.invalid_rows == "error":
        raise ValueError(
            f"{dropped}/{len(frame)} invalid point rows; correct the workbook or explicitly allow dropping"
        )
    if clean.empty:
        raise ValueError("No valid point rows")
    if (clean[["R", "re"]] <= 0).any().any():
        raise ValueError("R and Re must be positive")
    clean[["x", "z"]] *= LENGTH_TO_UM[config.length_unit]
    return clean, {"input_rows": len(frame), "valid_rows": len(clean), "dropped_rows": dropped}


def quality_reasons(row) -> list[str]:
    reasons = []
    if not row.get("geometry_consistent", True):
        reasons.append("inconsistent_triangle_geometry")
    if row.get("dropped_input_rows", 0) > 0:
        reasons.append("invalid_rows_dropped")
    if row.get("duplicate_sign_conflicts", 0) > 0:
        reasons.append("conflicting_duplicate_points")
    if row.get("broad_rbr_rows_after", 0) > 0 or row.get("broad_brb_rows_after", 0) > 0:
        reasons.append("multiple_sign_transitions")
    if not row.get("has_both_signs", True):
        reasons.append("undefined_boundary")
    if row.get("relative_coordinate_coverage", 1.0) < 1 - 1e-9:
        reasons.append("coordinate_grid_differs_within_file")
    return reasons


def constant_geometry_value(values: pd.Series, *, source_file: str, column: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not len(numeric) or not np.isfinite(numeric).all():
        raise ValueError(f"{source_file}: invalid geometry values in {column}")
    median = float(np.median(numeric))
    if median <= 0 or not np.allclose(numeric, median, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{source_file}: {column} must be positive and constant within each R/Re sample")
    return median
