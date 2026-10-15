"""Descriptive prediction counts with explicit denominators and no accuracy claims."""

import numpy as np
import pandas as pd

KEYS = ["source_file", "R", "re"]


def validate_prediction_table(data):
    required = [
        *KEYS,
        "focus_probability",
        "threshold",
        "focus_prediction",
        "decision",
        "needs_review",
        "known_training_sample",
        "data_quality_warning",
        "overlap_status",
        "review_reasons",
        "boundary_distance_abs",
        "boundary_midpoint_x",
    ]
    missing = sorted(set(required) - set(data))
    if missing:
        raise ValueError(f"Missing prediction report columns: {missing}")
    if data.empty or data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError("Prediction statistics require unique, nonempty Excel/R/Re samples")
    probabilities = data.focus_probability.to_numpy(dtype=float)
    threshold = data.threshold.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid prediction probabilities")
    if not np.isfinite(threshold).all() or not ((threshold > 0) & (threshold < 1)).all() or len(set(threshold)) != 1:
        raise ValueError("A prediction report requires one valid fixed threshold")
    for column in ["needs_review", "known_training_sample", "data_quality_warning"]:
        if not data[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"Prediction flag must be boolean: {column}")
    expected = np.where(probabilities > threshold, "F", "UF")
    if not np.array_equal(data.focus_prediction, expected):
        raise ValueError("Predicted classes do not match the saved threshold")
    if not np.array_equal(data.decision, np.where(data.needs_review, "REVIEW", expected)):
        raise ValueError("Review decisions do not match the saved flags")


def count_group(group):
    n = len(group)
    f = int(group.focus_prediction.eq("F").sum())
    return {
        "samples": n,
        "predicted_f": f,
        "predicted_uf": n - f,
        "predicted_f_rate": f / n,
        "review_samples": int(group.needs_review.sum()),
        "review_rate": float(group.needs_review.mean()),
        "no_review_flag_samples": int((~group.needs_review).sum()),
        "known_training_samples": int(group.known_training_sample.sum()),
        "data_quality_warning_samples": int(group.data_quality_warning.sum()),
        "probability_mean": float(group.focus_probability.mean()),
        "width_available_samples": int(np.isfinite(group.boundary_distance_abs).sum()),
        "width_median_um": float(group.boundary_distance_abs.replace([np.inf, -np.inf], np.nan).median()),
        "midpoint_available_samples": int(np.isfinite(group.boundary_midpoint_x).sum()),
        "midpoint_median_um": float(group.boundary_midpoint_x.replace([np.inf, -np.inf], np.nan).median()),
    }


def grouped_counts(data, columns):
    rows = []
    for key, group in data.groupby(columns, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(columns, key, strict=True)), **count_group(group)})
    return pd.DataFrame(rows)


def summarize_predictions(data):
    validate_prediction_table(data)
    summary = {
        **count_group(data),
        "files": int(data.source_file.nunique()),
        "radii": int(data.R.nunique()),
        "reynolds_numbers": int(data.re.nunique()),
        "threshold": float(data.threshold.iloc[0]),
        "evaluation": False,
        "accuracy": None,
        "sample_unit": "Excel x R x Re; same-geometry samples are not statistically independent",
        "prediction_rate_is_accuracy": False,
    }
    if "source_sha256" in data:
        valid = data.source_sha256.notna() & data.source_sha256.ne("")
        summary["verified_content_sample_units"] = int(
            data.loc[valid, ["source_sha256", "R", "re"]].drop_duplicates().shape[0]
        )
        summary["unverified_content_samples"] = int((~valid).sum())
        summary["duplicate_content_extra_samples"] = int(valid.sum()) - summary["verified_content_sample_units"]
    reasons = data.review_reasons.fillna("").map(lambda s: sorted(set(str(s).split(";")) - {""})).explode().dropna()
    counts = reasons.value_counts().rename_axis("reason").rename("samples").reset_index()
    counts["fraction_of_all_samples"] = counts.samples / len(data)
    summary["review_reason_counts_overlap"] = True
    tables = {
        "by_file": grouped_counts(data, ["source_file"]),
        "by_radius": grouped_counts(data, ["R"]),
        "by_reynolds": grouped_counts(data, ["re"]),
        "by_file_radius": grouped_counts(data, ["source_file", "R"]),
        "by_overlap": grouped_counts(data, ["overlap_status"]),
        "review_reasons": counts,
    }
    if "geometry_group" in data:
        tables["by_geometry"] = grouped_counts(data, ["geometry_group"])
    features = []
    for name in ["boundary_distance_abs", "boundary_midpoint_x"]:
        values = data[name].replace([np.inf, -np.inf], np.nan).dropna()
        features.append(
            {
                "feature": name,
                "unit": "um",
                "available": len(values),
                "missing": len(data) - len(values),
                "min": values.min(),
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "max": values.max(),
            }
        )
    tables["feature_statistics"] = pd.DataFrame(features)
    finite_boundary = np.isfinite(data.boundary_distance_abs) & np.isfinite(data.boundary_midpoint_x)
    tables["boundary_plot_counts"] = (
        data.loc[finite_boundary]
        .groupby(["focus_prediction", "boundary_distance_abs", "boundary_midpoint_x"])
        .size()
        .rename("samples")
        .reset_index()
    )
    summary["boundary_plot_samples"] = int(finite_boundary.sum())
    if "rule_available" in data:
        available = int(data.rule_available.sum())
        disagreements = int(data.rule_disagrees_with_model.sum())
        summary.update(
            rule_available_samples=available,
            rule_model_disagreements=disagreements,
            rule_model_disagreement_rate=disagreements / available if available else None,
        )
    return summary, tables
