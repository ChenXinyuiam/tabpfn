"""Non-calibrated review flags and content-aware training overlap checks."""

import numpy as np
import pandas as pd

from .quality import quality_reasons


def prediction_diagnostics(train, test, probabilities, threshold, feature_columns, *, legacy_recipe=False):
    keys = ["source_file", "R", "re"]
    train_keys = pd.MultiIndex.from_frame(train[keys])
    test_keys = pd.MultiIndex.from_frame(test[keys])
    if not train_keys.is_unique or not test_keys.is_unique:
        raise ValueError("Training or prediction data contains duplicate sample keys")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(test),) or not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must align one-to-one with prediction samples")
    result = pd.DataFrame(index=test.index)
    result["seen_sample_key"] = test_keys.isin(train_keys)
    result["known_training_sample"] = False
    result["duplicate_input_content"] = False
    if "source_sha256" in train and "source_sha256" in test:
        content_keys = ["source_sha256", "R", "re"]
        valid_train = train.source_sha256.notna() & train.source_sha256.ne("")
        valid_test = test.source_sha256.notna() & test.source_sha256.ne("")
        train_content = pd.MultiIndex.from_frame(train.loc[valid_train, content_keys])
        test_content = pd.MultiIndex.from_frame(test[content_keys])
        result["known_training_sample"] = test_content.isin(train_content) & valid_test.to_numpy()
        result["duplicate_input_content"] = test_content.duplicated(keep=False) & valid_test.to_numpy()
    geometry_key = (
        "physical_geometry_id"
        if "physical_geometry_id" in test and "physical_geometry_id" in train
        else "geometry_group"
    )
    result["seen_geometry"] = test[geometry_key].isin(train[geometry_key])
    limits = train[feature_columns].replace([np.inf, -np.inf], np.nan).agg(["min", "max"])
    flags, outside_columns, missing_columns, quality_flags = [], [], [], []
    for position, (_, row) in enumerate(test.iterrows()):
        reasons = quality_reasons(row)
        quality_flags.append(bool(reasons))
        outside = [
            column
            for column in feature_columns
            if pd.notna(row[column])
            and (row[column] < limits.loc["min", column] or row[column] > limits.loc["max", column])
        ]
        missing = [column for column in feature_columns if not np.isfinite(row[column])]
        if missing:
            reasons.append("missing_model_features")
        if outside:
            reasons.append("outside_training_range")
        if not result["seen_geometry"].iloc[position]:
            reasons.append("unseen_geometry")
        if result["known_training_sample"].iloc[position]:
            reasons.append("training_sample_not_independent")
        elif result["seen_sample_key"].iloc[position]:
            reasons.append("same_key_changed_or_unverified_content")
        if abs(float(probabilities[position]) - threshold) <= 0.1:
            reasons.append("near_decision_threshold")
        if legacy_recipe:
            reasons.append("legacy_recipe_config_unverified")
        if result["duplicate_input_content"].iloc[position]:
            reasons.append("duplicate_input_content")
        flags.append(";".join(reasons))
        outside_columns.append(";".join(outside))
        missing_columns.append(";".join(missing))
    result["outside_range_features"] = outside_columns
    result["missing_features"] = missing_columns
    result["data_quality_warning"] = quality_flags
    result["overlap_status"] = np.select(
        [result.known_training_sample, result.seen_sample_key, ~result.seen_geometry],
        ["training_content", "changed_or_unverified_key", "unseen_geometry"],
        default="new_sample_known_geometry",
    )
    result["review_reasons"] = flags
    result["needs_review"] = result["review_reasons"].ne("")
    result["decision"] = np.where(result["needs_review"], "REVIEW", np.where(probabilities > threshold, "F", "UF"))
    return result
