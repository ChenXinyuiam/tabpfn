"""Grouped splits and classification metrics shared by evaluation workflows."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold


def make_grouped_folds(
    data: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
    canonical_order: bool = False,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame, list[dict[str, object]]]:
    y = data["focus_label"].to_numpy(dtype=int)
    groups = data["geometry_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The complete dataset must contain both F and UF")
    if n_splits > len(unique_groups):
        raise ValueError(f"n_splits={n_splits} exceeds geometry groups={len(unique_groups)}")

    splitter = (
        LeaveOneGroupOut()
        if canonical_order and n_splits == len(unique_groups)
        else StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    folds = list(splitter.split(np.zeros((len(data), 1)), y, groups))
    assignments = data[["source_file", "summary_id", "geometry_group", "R", "re", "focus_label_name"]].copy()
    assignments["cv_fold"] = -1
    audit_rows: list[dict[str, object]] = []
    for fold_id, (train_index, test_index) in enumerate(folds, start=1):
        assignments.iloc[test_index, assignments.columns.get_loc("cv_fold")] = fold_id
        train_groups = sorted(set(groups[train_index]))
        test_groups = sorted(set(groups[test_index]))
        train_files = set(data.iloc[train_index]["source_file"])
        test_files = set(data.iloc[test_index]["source_file"])
        group_overlap = sorted(set(train_groups) & set(test_groups))
        file_overlap = sorted(train_files & test_files)
        if "physical_geometry_id" in data:
            physical_overlap = set(data.iloc[train_index]["physical_geometry_id"]) & set(
                data.iloc[test_index]["physical_geometry_id"]
            )
            if physical_overlap:
                raise RuntimeError(f"Physical geometry leakage in fold {fold_id}: {sorted(physical_overlap)}")
        if group_overlap or file_overlap:
            raise RuntimeError(f"Leakage detected in fold {fold_id}: groups={group_overlap}, files={file_overlap}")
        if len(np.unique(y[train_index])) < 2:
            raise RuntimeError(f"Fold {fold_id} training data does not contain both classes")
        audit_rows.append(
            {
                "fold": fold_id,
                "train_samples": int(len(train_index)),
                "test_samples": int(len(test_index)),
                "train_geometry_groups": train_groups,
                "test_geometry_groups": test_groups,
                "group_overlap": group_overlap,
                "source_file_overlap": file_overlap,
                "train_focused": int(y[train_index].sum()),
                "test_focused": int(y[test_index].sum()),
            }
        )
    if (assignments["cv_fold"] < 1).any():
        raise RuntimeError("Some samples were not assigned to a cross-validation fold")
    return folds, assignments, audit_rows


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray | None) -> dict[str, object]:
    if not len(y_true):
        if len(y_pred) or (probability is not None and len(probability)):
            raise ValueError("Prediction lengths differ from the empty label array")
        return {
            **{
                key: np.nan
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                    "mcc",
                    "roc_auc",
                    "average_precision",
                    "brier_score",
                    "log_loss",
                )
            },
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2
        else float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(np.r_[y_true, y_pred])) > 1 else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm.tolist(),
    }
    if probability is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
        metrics["average_precision"] = float(average_precision_score(y_true, probability))
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    if probability is not None:
        metrics["brier_score"] = float(brier_score_loss(y_true, probability))
        metrics["log_loss"] = float(log_loss(y_true, probability, labels=[0, 1]))
    else:
        metrics["brier_score"] = np.nan
        metrics["log_loss"] = np.nan
    return metrics


def metrics_by_geometry(data: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for geometry_group, group in data.groupby("geometry_group", sort=True):
        y_true = group["focus_label"].to_numpy(dtype=int)
        y_pred = group[prediction_column].to_numpy(dtype=int)
        rows.append(
            {
                "geometry_group": geometry_group,
                "samples": int(len(group)),
                "focused_samples": int(y_true.sum()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": evaluate_predictions(y_true, y_pred, None)["balanced_accuracy"],
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)
