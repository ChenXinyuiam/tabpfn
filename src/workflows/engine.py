"""Shared numerical path for independent modules and the cross-fitted cascade."""

import numpy as np
import pandas as pd

from src.evaluation.protocols import fold_manifest
from src.focus_inspection.system_features import FOCUS_FEATURES, labeled_sections, section_features
from src.geometry.triangle import FEATURES
from src.physics.force import PhysicsUnavailable, compose_force, validate_physics
from src.regression.models import FittedModel, ModelSpec
from src.regression.predict_fields import spatial_diagnostics


def fit_fields(points, config):
    models = {}
    for task in ("FD", "FL"):
        frame = points.loc[points.task.eq(task)]
        if not frame.empty:
            models[task] = FittedModel(ModelSpec(**config.get("regression", {})), FEATURES).fit(frame, frame.target)
    if not models:
        raise ValueError("No original FD/FL training data")
    return models


def field_predictions(models, query):
    out = query.copy()
    for task, model in models.items():
        out[task + "_pred"] = model.predict(query)
    return out


def composed_predictions(models, query, meta, config):
    if set(models) != {"FD", "FL"}:
        raise PhysicsUnavailable("Both FD and FL models are required for Fzong")
    out = field_predictions(models, query)
    out["Fzong_pred"] = compose_force(out.FD_pred, out.FL_pred, meta["targets"], config.get("physics"))
    return out


def crossfit_fields(points, meta, config):
    validate_physics(meta["targets"], config.get("physics"))
    query = points.loc[points.task.eq("Fzong")]
    if query.empty:
        raise ValueError("Cascade training needs independent Fzong sample positions linked to focus labels")
    assignments = fold_manifest(points, config.get("inner_splits", 3), config.get("seed", 42))
    outputs, audit = [], []
    for fold in sorted(assignments.fold.unique()):
        held = set(assignments.loc[assignments.fold.eq(fold), "geometry_family_id"])
        train = points.loc[~points.geometry_family_id.isin(held)]
        test = query.loc[query.geometry_family_id.isin(held)]
        if test.empty:
            continue
        model = fit_fields(train, config)
        output = composed_predictions(model, test, meta, config)
        output["inner_fold"] = int(fold)
        outputs.append(output)
        audit.append(
            {
                "inner_fold": int(fold),
                "train_groups": sorted(train.geometry_family_id.unique()),
                "test_groups": sorted(held),
            }
        )
    result = pd.concat(outputs, ignore_index=True)
    if len(result) != len(query) or result.duplicated(["sample_id", "point_id"]).any():
        raise RuntimeError("Each cascade training point must have exactly one grouped out-of-fold prediction")
    return result, audit


def fit_classifier(features, meta, config):
    classes = meta["classes"]
    if set(features.label) != set(classes):
        raise ValueError("Training fold must contain every declared focus class: " + str(classes))
    return FittedModel(ModelSpec(**config.get("classifier", {})), FOCUS_FEATURES, classification=True).fit(
        features, features.label
    )


def predict_classifier(model, features, classes):
    probability = model.probabilities(features)
    actual_order = list(model.estimator.classes_)
    probability = probability[:, [actual_order.index(name) for name in classes]]
    out = features.copy()
    for i, name in enumerate(classes):
        out["probability_" + name] = probability[:, i]
    out["predicted_label"] = np.asarray(classes)[probability.argmax(axis=1)]
    out["review_status"] = np.where(out.has_both_signs, "OK", "REVIEW")
    out["review_reason"] = np.where(out.has_both_signs, "", "undefined_boundary")
    return out


def fit_system(points, samples, meta, config, mode):
    if mode not in {"fields", "focus", "cascade"}:
        raise ValueError("mode must be fields, focus or cascade")
    result = {
        "mode": mode,
        "fields": {},
        "classifier": None,
        "focus_features": None,
        "crossfit": None,
        "crossfit_audit": [],
    }
    if mode in {"fields", "cascade"}:
        result["fields"] = fit_fields(points, config)
    if mode in {"focus", "cascade"}:
        if mode == "cascade":
            force_points, audit = crossfit_fields(points, meta, config)
            result["crossfit"], result["crossfit_audit"] = force_points, audit
            force_column = "Fzong_pred"
        else:
            force_points, force_column = points.loc[points.task.eq("Fzong")], "target"
        features = labeled_sections(
            force_points, samples, meta["classes"], force_column=force_column, window_size=config.get("window_size", 3)
        )
        result["focus_features"] = features
        result["classifier"] = fit_classifier(features, meta, config)
    return result


def predict_system(fitted, query, meta, config):
    out = field_predictions(fitted["fields"], query)
    for task in fitted["fields"]:
        for name in ("quantity", "unit", "component"):
            out[task + "_" + name] = meta["targets"][task][name]
    status = {
        "fields": "complete" if fitted["fields"] else "not_requested",
        "Fzong": "not_executed",
        "focus": "not_executed",
    }
    if "provided_force" in query:
        if fitted["mode"] != "focus":
            raise ValueError("Provided Fzong is only accepted by an observed-field focus bundle")
        out["Fzong_pred"] = query.provided_force
        out["Fzong_unit"] = meta["targets"]["Fzong"]["unit"]
        out["Fzong_component"] = meta["targets"]["Fzong"]["component"]
        status["Fzong"] = "provided_reference_field"
    elif fitted["fields"]:
        try:
            if set(fitted["fields"]) != {"FD", "FL"}:
                raise PhysicsUnavailable("Both FD and FL models are needed")
            out["Fzong_pred"] = compose_force(out.FD_pred, out.FL_pred, meta["targets"], config.get("physics"))
            out["Fzong_unit"] = config["physics"]["output_unit"]
            out["Fzong_component"] = config["physics"]["component"]
            status["Fzong"] = "complete"
        except PhysicsUnavailable as error:
            status["reason"] = str(error)
    focus = pd.DataFrame()
    if fitted["classifier"] is not None and "Fzong_pred" in out:
        features = section_features(out, "Fzong_pred", config.get("window_size", 3))
        focus = predict_classifier(fitted["classifier"], features, meta["classes"])
        status["focus"] = "complete"
    elif fitted["classifier"] is not None:
        status.setdefault("reason", "Observed-field focus model requires provided_fzong values")
    return out, focus, status


def diagnostics(training, query, field_models):
    records = []
    for task in field_models:
        source = training.loc[training.task.eq(task)]
        flags = (
            query[["sample_id", "point_id", "geometry_family_id", "geometry_id", "re"]].copy().reset_index(drop=True)
        )
        flags["task"] = task
        flags["new_geometry"] = ~flags.geometry_family_id.isin(source.geometry_family_id)
        flags["outside_feature_range"] = (
            (query[FEATURES].to_numpy() < source[FEATURES].min().to_numpy())
            | (query[FEATURES].to_numpy() > source[FEATURES].max().to_numpy())
        ).any(axis=1)
        flags["spatial_relation"] = "new_oriented_geometry"
        flags["nearest_training_distance_um"] = np.nan
        flags["training_position_overlap"] = False
        for (geometry, radius), group in query.reset_index(drop=True).groupby(["geometry_id", "radius_um"]):
            known = source.loc[source.geometry_id.eq(geometry) & source.radius_um.eq(radius)]
            if known.empty:
                continue
            spatial = spatial_diagnostics(known, group, FEATURES)
            flags.loc[group.index, "spatial_relation"] = spatial.spatial_relation.to_numpy()
            flags.loc[group.index, "nearest_training_distance_um"] = spatial.nearest_training_distance_um.to_numpy()
            flags.loc[group.index, "training_position_overlap"] = spatial.training_position_overlap.to_numpy()
        flags["needs_review"] = (
            flags.new_geometry | flags.outside_feature_range | flags.spatial_relation.ne("inside_training_hull")
        )
        records.append(flags)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
