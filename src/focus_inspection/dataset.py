"""Sample keys, geometry features and strict label alignment."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.provenance import file_sha256

from .focus_boundary_feature import resolve_source_columns
from .geometry_integrity import attach_geometry_audit
from .quality import LENGTH_TO_UM, constant_geometry_value
from .settings import BOUNDARY_FEATURE, BOUNDARY_MIDPOINT_FEATURE, CONTEXT_FEATURES, GEOMETRY_FEATURES


def _numbers_from_name(name: str) -> list[int]:
    numbers = [int(value) for value in re.findall(r"\d+", Path(name).stem)]
    if len(numbers) < 3:
        raise ValueError(f"Cannot parse geometry and radius from name: {name}")
    return numbers


def summary_id_from_source_file(source_file: str, radius: float | None = None) -> str:
    """Map raw file naming variants to the identifier used in the label sheet."""
    numbers = _numbers_from_name(source_file)
    reverse = "反" in Path(source_file).stem
    orientation = "反" if reverse else ""
    radius_text = str(numbers[2]) if radius is None else f"{float(radius):g}"
    return f"{numbers[0]}-{numbers[1]}{orientation}-{radius_text}"


def geometry_group_from_source_file(source_file: str) -> str:
    """Group all radii and orientations of one geometric size together."""
    numbers = _numbers_from_name(source_file)
    return f"{numbers[0]}-{numbers[1]}"


def load_focus_labels(label_file: Path) -> pd.DataFrame:
    wide = pd.read_excel(label_file, engine="openpyxl")
    if wide.empty or len(wide.columns) < 2:
        raise ValueError(f"Label workbook has no usable label matrix: {label_file}")

    wide = wide.rename(columns={wide.columns[0]: "summary_id"})
    labels = wide.melt(id_vars="summary_id", var_name="re", value_name="focus_label_name")
    labels["summary_id"] = labels["summary_id"].astype(str).str.strip()
    labels["re"] = pd.to_numeric(labels["re"], errors="raise")
    labels["focus_label_name"] = labels["focus_label_name"].astype(str).str.strip().str.upper()
    allowed = {"F", "UF"}
    unsupported = sorted(set(labels["focus_label_name"]) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported focus labels {unsupported}; expected F or UF")
    labels["focus_label"] = labels["focus_label_name"].map({"UF": 0, "F": 1}).astype(int)

    duplicate_mask = labels.duplicated(["summary_id", "re"], keep=False)
    if duplicate_mask.any():
        duplicates = labels.loc[duplicate_mask, ["summary_id", "re"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicated label keys: {duplicates[:10]}")
    return labels.sort_values(["summary_id", "re"]).reset_index(drop=True)


def load_geometry_features(raw_dir: Path, source_files: list[str], *, length_unit: str = "um") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_file in sorted(source_files):
        path = raw_dir / source_file
        if not path.exists():
            raise FileNotFoundError(f"Raw source file referenced by boundary features does not exist: {path}")
        frame = pd.read_excel(path, engine="openpyxl")
        columns = resolve_source_columns(frame)
        frame = frame.rename(columns={columns.radius: "R", columns.re_value: "re"})
        required = {"R", "re", *GEOMETRY_FEATURES}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{source_file} is missing geometry columns: {missing}")

        numeric = frame[["R", "re", *GEOMETRY_FEATURES]].copy()
        for column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        numeric = numeric.dropna(subset=["R", "re"])
        for (radius, re_value), sample in numeric.groupby(["R", "re"], sort=True):
            row: dict[str, object] = {
                "source_file": source_file,
                "R": float(radius),
                "re": float(re_value),
                "orientation_reverse": int("反" in Path(source_file).stem),
            }
            for column in GEOMETRY_FEATURES:
                value = constant_geometry_value(sample[column], source_file=source_file, column=column)
                scale = LENGTH_TO_UM[length_unit]
                if column == "area":
                    value *= scale**2
                elif not column.startswith("angle_"):
                    value *= scale
                row[column] = value
            rows.append(row)
    geometry = pd.DataFrame(rows)
    duplicate_mask = geometry.duplicated(["source_file", "R", "re"], keep=False)
    if duplicate_mask.any():
        raise ValueError("Geometry features contain duplicate source_file/R/re keys")
    return geometry


def build_feature_dataset(raw_dir: Path, boundary_features_file: Path) -> pd.DataFrame:
    boundary = pd.read_csv(boundary_features_file, encoding="utf-8-sig")
    required_boundary = {"source_file", "R", "re", BOUNDARY_FEATURE, BOUNDARY_MIDPOINT_FEATURE}
    missing_boundary = sorted(required_boundary - set(boundary.columns))
    if missing_boundary:
        raise ValueError(f"Boundary feature table is missing columns: {missing_boundary}")
    if boundary.duplicated(["source_file", "R", "re"]).any():
        raise ValueError("Boundary feature table contains duplicate source_file/R/re keys")

    for source_file, rows in boundary.groupby("source_file"):
        if Path(source_file).name != source_file or "\\" in source_file or "/" in source_file:
            raise ValueError(f"Expected a source filename, got {source_file}")
        if not np.isfinite(rows[["R", "re"]]).all().all() or (rows[["R", "re"]] <= 0).any().any():
            raise ValueError(f"Invalid R/Re keys: {source_file}")
        if "source_sha256" in rows:
            if not rows["source_sha256"].eq(file_sha256(raw_dir / source_file)).all():
                raise ValueError(f"Stale boundary features: {source_file} changed; rebuild features first")

    boundary["summary_id"] = [
        summary_id_from_source_file(name, radius)
        for name, radius in zip(boundary["source_file"], boundary["R"], strict=True)
    ]
    boundary["geometry_group"] = boundary["source_file"].map(geometry_group_from_source_file)
    units = boundary["source_length_unit"].unique() if "source_length_unit" in boundary else ["um"]
    if len(units) != 1 or units[0] not in LENGTH_TO_UM:
        raise ValueError("Feature table must declare one supported source length unit")
    geometry = load_geometry_features(raw_dir, boundary["source_file"].drop_duplicates().tolist(), length_unit=units[0])
    features = boundary.merge(
        geometry, on=["source_file", "R", "re"], how="left", validate="one_to_one", indicator=True
    )
    if features["_merge"].ne("both").any():
        raise ValueError("Missing raw geometry for one or more feature samples")
    features = features.drop(columns="_merge")
    return attach_geometry_audit(features)


def build_labeled_dataset(
    raw_dir: Path, label_file: Path, boundary_features_file: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    features = build_feature_dataset(raw_dir, boundary_features_file)
    labels = load_focus_labels(label_file)
    merged = features.merge(labels, on=["summary_id", "re"], how="outer", indicator=True, validate="one_to_one")

    unmatched_features = merged.loc[merged["_merge"] == "left_only", ["source_file", "R", "re"]]
    unmatched_labels = merged.loc[merged["_merge"] == "right_only", ["summary_id", "re"]]
    if not unmatched_features.empty or not unmatched_labels.empty:
        raise ValueError(
            "Feature/label matching is incomplete: "
            f"unmatched features={unmatched_features.to_dict('records')[:10]}, "
            f"unmatched labels={unmatched_labels.to_dict('records')[:10]}"
        )

    data = merged.drop(columns="_merge").sort_values(["source_file", "R", "re"]).reset_index(drop=True)
    missing_model_values = data[[*CONTEXT_FEATURES, BOUNDARY_FEATURE, BOUNDARY_MIDPOINT_FEATURE]].isna().sum()
    audit = {
        "sample_count": int(len(data)),
        "source_file_count": int(data["source_file"].nunique()),
        "geometry_group_count": int(data["geometry_group"].nunique()),
        "physical_geometry_count": int(data["physical_geometry_id"].nunique()),
        "geometry_warning_samples": int((~data["geometry_consistent"]).sum()),
        "geometry_warning_files": sorted(data.loc[~data["geometry_consistent"], "source_file"].unique().tolist()),
        "label_counts": {str(key): int(value) for key, value in data["focus_label_name"].value_counts().items()},
        "duplicate_feature_keys": int(data.duplicated(["source_file", "R", "re"]).sum()),
        "duplicate_label_keys": int(labels.duplicated(["summary_id", "re"]).sum()),
        "unmatched_feature_rows": int(len(unmatched_features)),
        "unmatched_label_rows": int(len(unmatched_labels)),
        "source_hashes_verified": "source_sha256" in features,
        "missing_model_values": {key: int(value) for key, value in missing_model_values.items()},
    }
    return data, audit
