"""Outer grouped evaluation of fields, observed focus, or a fully cross-fitted cascade."""

import numpy as np
import pandas as pd

from src.focus_inspection.system_features import labeled_sections
from src.physics.force import reference_force
from src.workflows.engine import (
    composed_predictions,
    fit_system,
    predict_classifier,
)

from .protocols import classification_metrics, fold_manifest, regression_metrics, regression_tables


def evaluate_system(points, samples, meta, config, *, task="fields", n_splits=5, protocol="geometry"):
    if task not in {"fields", "FD", "FL", "focus", "cascade"}:
        raise ValueError("Unknown evaluation task")
    if task in {"focus", "cascade"} and protocol != "geometry":
        raise ValueError("Focus/cascade evaluation requires complete geometry holdouts")
    assignments = fold_manifest(points, n_splits, config.get("seed", 42), protocol)
    field_outputs, focus_outputs, metrics, inner_audits = [], [], [], []
    for fold in sorted(assignments.fold.unique()):
        print(f"Outer fold {fold}/{n_splits}: {task}", flush=True)
        mask = assignments.fold.eq(fold).to_numpy()
        train, test = points.loc[~mask].copy(), points.loc[mask].copy()
        if task in {"FD", "FL"}:
            train = train.loc[train.task.eq(task)]
        fitted = fit_system(train, samples, meta, config, "fields" if task in {"fields", "FD", "FL"} else task)
        if task in {"fields", "FD", "FL", "cascade"}:
            for field, model in fitted["fields"].items():
                query = test.loc[test.task.eq(field)].copy()
                if query.empty:
                    continue
                query["prediction"], query["truth"], query["fold"] = model.predict(query), query.target, int(fold)
                field_outputs.append(query)
                metrics.append({"task": field, "fold": int(fold), **regression_metrics(query.truth, query.prediction)})
        if task in {"focus", "cascade"}:
            query = test.loc[test.task.eq("Fzong")].copy()
            if query.empty:
                raise ValueError("Every focus outer fold needs reference samples; revise group allocation")
            if task == "cascade":
                query = composed_predictions(fitted["fields"], query, meta, config)
                query["prediction"] = query.Fzong_pred
                query["truth"] = reference_force(query.target, meta["targets"]["Fzong"], config["physics"])
                query["fold"] = int(fold)
                field_outputs.append(query)
                metrics.append(
                    {"task": "Fzong", "fold": int(fold), **regression_metrics(query.truth, query.prediction)}
                )
                force_column = "Fzong_pred"
                inner_audits.append(
                    {
                        "outer_fold": int(fold),
                        "outer_test_groups": sorted(test.geometry_family_id.unique()),
                        "inner_splits": fitted["crossfit_audit"],
                    }
                )
            else:
                force_column = "target"
            features = labeled_sections(
                query, samples, meta["classes"], force_column=force_column, window_size=config.get("window_size", 3)
            )
            prediction = predict_classifier(fitted["classifier"], features, meta["classes"])
            prediction["fold"] = int(fold)
            focus_outputs.append(prediction)
            probability = prediction[["probability_" + c for c in meta["classes"]]].to_numpy()
            metrics.append(
                {
                    "task": "focus",
                    "fold": int(fold),
                    **classification_metrics(prediction.label, probability, meta["classes"]),
                }
            )
    fields = pd.concat(field_outputs, ignore_index=True) if field_outputs else pd.DataFrame()
    focus = pd.concat(focus_outputs, ignore_index=True) if focus_outputs else pd.DataFrame()
    pooled = {}
    if not fields.empty:
        if fields.duplicated(["task", "sample_id", "point_id"]).any():
            raise RuntimeError("Repeated outer test predictions")
        for field, part in fields.groupby("task"):
            expected = points.loc[points.task.eq(field)]
            if len(part) != len(expected):
                raise RuntimeError("Incomplete out-of-fold coverage for " + field)
            per_geometry = [regression_metrics(g.truth, g.prediction) for _, g in part.groupby("geometry_family_id")]
            pooled[field] = {
                "pooled_oof": regression_metrics(part.truth, part.prediction),
                "geometry_equal_weight": {
                    key: float(np.mean([g[key] for g in per_geometry if g[key] is not None]))
                    if any(g[key] is not None for g in per_geometry)
                    else None
                    for key in ("r2", "mae", "rmse", "sign_error_rate")
                },
            }
    if not focus.empty:
        if (
            focus.sample_id.duplicated().any()
            or len(focus) != points.loc[points.task.eq("Fzong"), "sample_id"].nunique()
        ):
            raise RuntimeError("Incomplete focus out-of-fold coverage")
        pooled["focus"] = classification_metrics(
            focus.label, focus[["probability_" + c for c in meta["classes"]]].to_numpy(), meta["classes"]
        )
    if not pooled:
        raise ValueError("No evaluated samples")
    return {
        "fold_manifest": assignments,
        "point_oof": fields,
        "focus_oof": focus,
        "by_geometry_re": regression_tables(fields) if not fields.empty else pd.DataFrame(),
        "metrics": {
            "protocol": protocol,
            "n_splits": n_splits,
            "task": task,
            "folds": metrics,
            "summary": pooled,
            "configuration_selection": "fixed in advance; no test-set tuning",
            "inner_audits": inner_audits,
        },
    }
