"""Nested geometry-held-out evaluation of a fixed width model's probabilities."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

from .build_focus_boundary_features import run as build_features
from .cache import PredictionCache
from .calibration import apply_calibration, fit_sigmoid
from .dataset import build_labeled_dataset
from .evaluation import evaluate_predictions, make_grouped_folds
from .geometry_integrity import write_geometry_audit
from .modeling import predict_probability, resolve_device
from .optimize_focus import assert_disjoint
from .quality import FEATURE_SCHEMA_VERSION, FeatureConfig, feature_code_hashes
from .settings import CONTEXT_FEATURES, DEFAULT_LABEL_FILE, DEFAULT_MODEL_PATH, DEFAULT_RAW_DIR, ROOT

FEATURES = [*CONTEXT_FEATURES, "boundary_distance_abs"]
DEFAULT_OUTPUT = ROOT / "results/focus_inspection/2026-09-04/focus_calibration"


def nested_calibration(
    data,
    *,
    model_path,
    device,
    seeds=(42, 2026, 3407),
    n_estimators=4,
    inner_splits=3,
    predictor=predict_probability,
    calibrator=fit_sigmoid,
):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct predeclared seeds")
    folds, assignments, outer_audit = make_grouped_folds(
        data, n_splits=data["geometry_group"].nunique(), random_state=0, canonical_order=True
    )
    results, metrics, calibration_rows, inner_audit = [], [], [], []
    for seed in seeds:
        raw = np.full(len(data), np.nan)
        calibrated = np.full(len(data), np.nan)
        for fold_id, (train_index, test_index) in enumerate(folds, 1):
            train, test = data.iloc[train_index], data.iloc[test_index]
            assert_disjoint(train, test)
            if not 2 <= inner_splits <= train.geometry_group.nunique():
                raise ValueError("Not enough outer-training geometries for inner calibration folds")
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold_id)
            oof = np.full(len(train), np.nan)
            coverage = np.zeros(len(train), dtype=int)
            for inner_id, (fit_index, valid_index) in enumerate(
                inner.split(train, train.focus_label, train.geometry_group), 1
            ):
                fit, valid = train.iloc[fit_index], train.iloc[valid_index]
                assert_disjoint(fit, valid)
                assert_disjoint(fit, test)
                assert_disjoint(valid, test)
                oof[valid_index] = predictor(
                    fit,
                    valid.drop(columns=["focus_label", "focus_label_name"], errors="ignore"),
                    feature_columns=FEATURES,
                    model_path=model_path,
                    device=device,
                    random_state=seed + fold_id * 100 + inner_id,
                    n_estimators=n_estimators,
                )
                coverage[valid_index] += 1
                inner_audit.append(
                    {
                        "seed": seed,
                        "outer_fold": fold_id,
                        "inner_fold": inner_id,
                        "fit_groups": sorted(fit.geometry_group.unique().tolist()),
                        "validation_groups": sorted(valid.geometry_group.unique().tolist()),
                        "test_groups": sorted(test.geometry_group.unique().tolist()),
                    }
                )
            if not (coverage == 1).all() or not np.isfinite(oof).all():
                raise RuntimeError("Calibration requires one complete OOF probability per training sample")
            state = calibrator(oof, train.focus_label.to_numpy())
            raw[test_index] = predictor(
                train,
                test.drop(columns=["focus_label", "focus_label_name"], errors="ignore"),
                feature_columns=FEATURES,
                model_path=model_path,
                device=device,
                random_state=seed + fold_id * 100,
                n_estimators=n_estimators,
            )
            calibrated[test_index] = apply_calibration(raw[test_index], state)
            calibration_rows.append({"seed": seed, "outer_fold": fold_id, "training_samples": len(train), **state})
            print(f"Seed {seed}: calibrated geometry fold {fold_id}/{len(folds)} complete", flush=True)
        if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
            raise RuntimeError("Incomplete outer probability predictions")
        result = data.copy()
        result["seed"] = seed
        result["cv_fold"] = assignments.cv_fold.to_numpy()
        for name, probabilities in (("raw", raw), ("sigmoid", calibrated)):
            result[f"proba_{name}"] = probabilities
            result[f"pred_{name}"] = (probabilities > 0.5).astype(int)
            metrics.append(
                {
                    "seed": seed,
                    "model": name,
                    **evaluate_predictions(
                        data.focus_label.to_numpy(), result[f"pred_{name}"].to_numpy(), probabilities
                    ),
                }
            )
        results.append(result)
    return (
        pd.concat(results, ignore_index=True),
        pd.DataFrame(metrics),
        {"outer_folds": outer_audit, "inner_folds": inner_audit, "calibrators": calibration_rows},
    )


def paired_probability_effect(data, *, repeats=2000):
    effects = []
    for _, group in data.groupby("geometry_group"):
        old = (group.proba_raw - group.focus_label) ** 2
        new = (group.proba_sigmoid - group.focus_label) ** 2
        effects.append([(new - old).sum(), len(group)])
    effects = np.asarray(effects)
    draws = np.random.default_rng(42).integers(len(effects), size=(repeats, len(effects)))
    totals = effects[draws].sum(axis=1)
    return {
        "brier_delta": float(effects[:, 0].sum() / effects[:, 1].sum()),
        "ci95": np.quantile(totals[:, 0] / totals[:, 1], [0.025, 0.975]).tolist(),
        "unit": "geometry; all seeds stay together",
        "repeats": repeats,
    }


def run(
    raw_dir,
    label_file,
    out_dir,
    model_path,
    device,
    *,
    seeds=(42, 2026, 3407),
    n_estimators=4,
    inner_splits=3,
    config=None,
    resume=False,
    cache_from=(),
    strict_geometry=False,
):
    from .calibration_report import write_report

    config = config or FeatureConfig()
    with experiment_run(out_dir, "focus-calibrate", resume=resume) as directory:
        started = time.perf_counter()
        cache = PredictionCache(directory / "prediction_cache", model_path, read_through=cache_from)
        protocol = {
            "features": FEATURES,
            "seeds": list(seeds),
            "n_estimators": n_estimators,
            "inner_splits": inner_splits,
            "threshold": 0.5,
            "calibrator": "LogisticRegression on clipped logit; C=1",
            "feature_config": config.to_dict(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_code_hashes": feature_code_hashes(),
            "model_identity": cache.identity,
            "calibration_code_sha256": file_sha256(Path(__file__).with_name("calibration.py")),
            "evaluation_code_sha256": file_sha256(Path(__file__)),
            "fold_code_sha256": file_sha256(Path(__file__).with_name("evaluation.py")),
            "label_sha256": file_sha256(label_file),
            "strict_geometry": strict_geometry,
            "source_hashes": {
                p.name: file_sha256(p) for p in sorted(raw_dir.glob("*Fzong*.xlsx")) if not p.name.startswith("~$")
            },
            "primary_metric": "Brier score; log loss, calibration bins and classification errors are also reported",
            "selection": "No model, threshold or seed selection. Existing development data, not external validation.",
        }
        path = directory / "protocol.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Cannot resume: protocol, data or implementation changed")
        write_json(path, protocol)
        build_features(raw_dir, directory / "features", config=config, make_plots=False)
        data, data_audit = build_labeled_dataset(raw_dir, label_file, directory / "features/features.csv")
        write_geometry_audit(data, directory / "geometry_audit.csv")
        if strict_geometry and (~data.geometry_consistent).any():
            raise ValueError("Inconsistent triangle geometry; see geometry_audit.csv. No raw values were changed")
        data.to_csv(directory / "training_features.csv", index=False, encoding="utf-8-sig")
        result, metrics, audit = nested_calibration(
            data,
            model_path=model_path,
            device=device,
            seeds=seeds,
            n_estimators=n_estimators,
            inner_splits=inner_splits,
            predictor=cache,
        )
        result.to_csv(directory / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        metrics.drop(columns="confusion_matrix").to_csv(
            directory / "metrics_by_seed.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(audit["calibrators"]).to_csv(directory / "fold_calibrators.csv", index=False, encoding="utf-8-sig")
        summary = {
            "data": data_audit,
            "protocol": protocol,
            "leakage_audit": audit,
            "paired_brier_effect": paired_probability_effect(result),
            "duration_seconds": time.perf_counter() - started,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
        }
        write_json(directory / "evaluation.json", summary)
        first_seed = result[result.seed.eq(seeds[0])]
        final_state = fit_sigmoid(first_seed.proba_raw.to_numpy(), first_seed.focus_label.to_numpy())
        common = {
            "candidate": "width",
            "feature_sets": {"width": FEATURES},
            "threshold": 0.5,
            "n_estimators": n_estimators,
            "seed": seeds[0],
            "model_path": str(model_path.resolve()),
            "checkpoint_sha256": cache.identity["checkpoint_sha256"],
            "model_identity": cache.identity,
            "training_features": "training_features.csv",
            "training_features_sha256": file_sha256(directory / "training_features.csv"),
            "feature_config": config.to_dict(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_code_hashes": feature_code_hashes(),
            "selection": "First predeclared seed. Full-data OOF calibration is for future use, not a test score.",
        }
        write_json(directory / "fixed_width_recipe.json", common)
        write_json(
            directory / "sigmoid_width_recipe.json",
            {**common, "calibration": final_state, "calibration_code_sha256": protocol["calibration_code_sha256"]},
        )
        write_report(directory, result, metrics, summary)
        print(metrics[["seed", "model", "accuracy", "brier_score", "log_loss"]].to_string(index=False))
        return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--length-unit", choices=("um", "mm", "m"), default="um")
    parser.add_argument("--invalid-rows", choices=("error", "drop"), default="error")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-from", type=Path, action="append", default=[])
    parser.add_argument("--strict-geometry", action="store_true")
    args = parser.parse_args()
    run(
        args.raw_dir,
        args.label_file,
        args.out_dir,
        args.model_path,
        resolve_device(args.device),
        seeds=tuple(args.seeds),
        n_estimators=args.n_estimators,
        inner_splits=args.inner_splits,
        config=FeatureConfig(args.window_size, args.length_unit, args.invalid_rows),
        resume=args.resume,
        cache_from=args.cache_from,
        strict_geometry=args.strict_geometry,
    )


if __name__ == "__main__":
    main()
