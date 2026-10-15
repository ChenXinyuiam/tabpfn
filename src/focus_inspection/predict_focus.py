"""Classify new Excel/R/Re samples from a validated recipe and training snapshot."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

from .build_focus_boundary_features import run as build_features
from .cache import PredictionCache
from .calibration import apply_calibration
from .dataset import build_feature_dataset
from .modeling import predict_probability, resolve_device
from .prediction_safety import prediction_diagnostics
from .quality import FEATURE_SCHEMA_VERSION, FeatureConfig, feature_code_hashes


def _run(recipe_path, raw_dir, out_dir, *, device="prefer-cuda", model_path=None, cache_enabled=False, cache_from=()):
    started = time.perf_counter()
    recipe_path, raw_dir, out_dir = Path(recipe_path), Path(raw_dir), Path(out_dir)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    training_file = (recipe_path.parent / recipe["training_features"]).resolve()
    if not training_file.is_relative_to(recipe_path.parent.resolve()):
        raise ValueError("Training snapshot must reside inside the recipe directory")
    if file_sha256(training_file) != recipe["training_features_sha256"]:
        raise ValueError("Training snapshot changed since recipe selection")
    model_path = model_path or Path(recipe["model_path"])
    if file_sha256(model_path) != recipe["checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint differs from the evaluated recipe")
    legacy_recipe = "feature_config" not in recipe
    config = FeatureConfig(**recipe.get("feature_config", {}))
    if not legacy_recipe:
        if recipe.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Unsupported feature schema; regenerate the evaluated recipe")
        if recipe.get("feature_code_hashes") != feature_code_hashes():
            raise ValueError("Feature implementation changed since evaluation; regenerate the recipe")
    identity = recipe.get("model_identity")
    protocol_file = recipe_path.parent / "protocol.json"
    if identity is None and protocol_file.is_file():
        identity = json.loads(protocol_file.read_text(encoding="utf-8")).get("model_identity")
    if identity:
        if identity["modeling_sha256"] != file_sha256(Path(__file__).with_name("modeling.py")):
            raise ValueError("Model preprocessing changed since evaluation")
        if any(version(name) != expected for name, expected in identity["versions"].items()):
            raise ValueError("Model dependency versions differ from the evaluated environment")
    else:
        legacy_recipe = True
    threshold = float(recipe["threshold"])
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Recipe threshold must be strictly between 0 and 1")
    calibration = recipe.get("calibration")
    if calibration is not None and recipe.get("calibration_code_sha256") != file_sha256(
        Path(__file__).with_name("calibration.py")
    ):
        raise ValueError("Calibration implementation changed since evaluation")
    apply_calibration(np.array([0.5]), calibration)
    candidate = recipe["candidate"]
    names = list(recipe["feature_sets"]) if candidate == "blend" else [candidate]
    if not names or any(name not in recipe["feature_sets"] for name in names):
        raise ValueError("Recipe candidate must reference nonempty feature sets")
    for name in names:
        columns = recipe["feature_sets"][name]
        if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
            raise ValueError("Recipe feature sets must be nonempty lists of distinct columns")
        if {"focus_label", "focus_label_name"} & set(columns):
            raise ValueError("Labels cannot be model features")
    resolved_device = resolve_device(device)
    build_features(raw_dir, out_dir / "features", make_plots=False, config=config)
    test = build_feature_dataset(raw_dir, out_dir / "features/features.csv").reset_index(drop=True)
    train = pd.read_csv(training_file, encoding="utf-8-sig")
    keys = ["source_file", "R", "re"]
    if train.duplicated(keys).any() or test.duplicated(keys).any() or test.empty:
        raise ValueError("Prediction requires nonempty, unique Excel/R/Re samples")
    feature_columns = list(dict.fromkeys(column for name in names for column in recipe["feature_sets"][name]))
    if set(feature_columns) - set(train) or set(feature_columns) - set(test):
        raise ValueError("Recipe features are missing from the training or prediction dataset")
    cache = (
        PredictionCache(
            out_dir / "prediction_cache", model_path, predictor=predict_probability, read_through=cache_from
        )
        if cache_enabled or cache_from
        else None
    )
    predictor = cache if cache is not None else predict_probability
    inference_started = time.perf_counter()
    raw_probability = np.mean(
        [
            np.asarray(
                predictor(
                    train,
                    test,
                    feature_columns=recipe["feature_sets"][name],
                    model_path=model_path,
                    device=resolved_device,
                    random_state=recipe["seed"],
                    n_estimators=recipe["n_estimators"],
                ),
                dtype=float,
            )
            for name in names
        ],
        axis=0,
    ).astype(float)
    probability = apply_calibration(raw_probability, calibration)
    inference_seconds = time.perf_counter() - inference_started
    metadata = ["source_sha256", "geometry_group", "physical_geometry_id", "geometry_consistent", "geometry_issues"]
    prediction = test[
        keys + ["boundary_distance_abs", "boundary_midpoint_x"] + [c for c in metadata if c in test]
    ].copy()
    prediction["focus_probability"] = probability
    prediction["raw_focus_probability"] = raw_probability
    prediction["calibration_applied"] = calibration is not None
    prediction["threshold"] = recipe["threshold"]
    prediction["focus_prediction"] = np.where(probability > recipe["threshold"], "F", "UF")
    diagnostics = prediction_diagnostics(
        train,
        test,
        probability,
        threshold,
        feature_columns,
        legacy_recipe=legacy_recipe,
    )
    prediction = pd.concat([prediction, diagnostics], axis=1)
    width = prediction.boundary_distance_abs
    prediction["rule_available"] = np.isfinite(width)
    prediction["rule_prediction"] = np.where(prediction.rule_available, np.where(width < 100, "F", "UF"), "undefined")
    prediction["rule_disagrees_with_model"] = prediction.rule_available & prediction.rule_prediction.ne(
        prediction.focus_prediction
    )
    prediction.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    write_json(
        out_dir / "prediction_run.json",
        {
            "recipe": str(recipe_path.resolve()),
            "candidate": candidate,
            "device": resolved_device,
            "cache_hits": cache.hits if cache is not None else 0,
            "cache_misses": cache.misses if cache is not None else 0,
            "cache_enabled": cache is not None,
            "inference_seconds": inference_seconds,
            "elapsed_before_report_seconds": time.perf_counter() - started,
            "samples": len(prediction),
            "known_training_samples": int(prediction["known_training_sample"].sum()),
            "evaluation": False,
            "raw_dir": str(raw_dir.resolve()),
            "threshold": recipe["threshold"],
            "feature_config": config.to_dict(),
            "recipe_sha256": file_sha256(recipe_path),
            "legacy_recipe": legacy_recipe,
            "review_samples": int(prediction["needs_review"].sum()),
            "probability_is_calibrated": calibration is not None,
            "calibration_applied": calibration is not None,
            "calibration_guarantee": False,
            "geometry_warning_samples": int((~test.geometry_consistent).sum())
            if "geometry_consistent" in test
            else None,
            "model_identity_verified": identity is not None,
            "review_policy": "Heuristic flags, not a validated error guarantee; threshold margin=0.1",
        },
    )
    from .prediction_report import write_prediction_report

    write_prediction_report(out_dir, prediction)
    print(f"Saved {len(prediction)} predictions to {out_dir / 'predictions.csv'}")
    return prediction


def run(recipe_path, raw_dir, out_dir, *, device="prefer-cuda", model_path=None, cache_enabled=False, cache_from=()):
    with experiment_run(out_dir, "focus-predict") as run_dir:
        result = _run(
            recipe_path,
            raw_dir,
            run_dir,
            device=device,
            model_path=model_path,
            cache_enabled=cache_enabled,
            cache_from=cache_from,
        )
        result.attrs["output_dir"] = str(run_dir)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cache", action="store_true", help="Save exact-content inference cache")
    parser.add_argument(
        "--cache-from", type=Path, action="append", default=[], help="Reuse a compatible prediction_cache directory"
    )
    args = parser.parse_args()
    run(
        args.recipe,
        args.raw_dir,
        args.out_dir,
        device=args.device,
        model_path=args.model_path,
        cache_enabled=args.cache,
        cache_from=args.cache_from,
    )


if __name__ == "__main__":
    main()
