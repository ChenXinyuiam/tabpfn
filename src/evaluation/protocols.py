"""Deterministic fold manifests, reused by every task and derived field."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def fold_manifest(points, n_splits=5, seed=42, protocol="geometry"):
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    out = points[["geometry_family_id", "sample_id", "point_id", "task"]].copy()
    if protocol == "geometry":
        out["split_group"] = out.geometry_family_id
    elif protocol == "spatial":
        # All Re, tasks and duplicate exports of an oriented position stay together.
        out["split_group"] = points.geometry_id + "|" + points.point_id
    else:
        raise ValueError("protocol must be geometry or spatial")
    groups = np.array(sorted(out.split_group.unique()))
    if len(groups) < n_splits:
        raise ValueError(f"{n_splits} folds require at least {n_splits} independent groups; found {len(groups)}")
    assignment = {}
    for fold, (_, test) in enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(groups), 1):
        assignment.update({group: fold for group in groups[test]})
    out["fold"] = out.split_group.map(assignment)
    return out.reset_index(drop=True)


def regression_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if not len(truth) or truth.shape != prediction.shape or not np.isfinite([truth, prediction]).all():
        raise ValueError("Metrics require aligned finite independent truth and predictions")
    error = prediction - truth
    denominator = float(((truth - truth.mean()) ** 2).sum())
    return {
        "n": len(truth),
        "r2": float(1 - (error**2).sum() / denominator) if len(truth) > 1 and denominator > 0 else None,
        "mae": float(abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "sign_error_rate": float((np.sign(truth) != np.sign(prediction)).mean()),
    }


def classification_metrics(truth, probability, classes):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss, recall_score

    truth = np.asarray(truth)
    predicted = np.asarray(classes)[np.argmax(probability, axis=1)]
    one_hot = np.array([[int(label == name) for name in classes] for label in truth])
    # log_loss orders string labels lexicographically, independently of the supplied list.
    ordered = sorted(classes)
    return {
        "n": len(truth),
        "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy_present_classes": float(
            recall_score(truth, predicted, labels=sorted(set(truth)), average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(truth, probability[:, [classes.index(c) for c in ordered]], labels=ordered)),
        "multiclass_brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "class_order": classes,
        "per_class": classification_report(truth, predicted, labels=classes, output_dict=True, zero_division=0),
    }


def regression_tables(predictions):
    rows = []
    for keys, part in predictions.groupby(["task", "fold", "geometry_family_id", "re"], sort=True):
        rows.append(
            dict(zip(["task", "fold", "geometry_family_id", "re"], keys, strict=True))
            | regression_metrics(part.truth, part.prediction)
        )
    return pd.DataFrame(rows)
