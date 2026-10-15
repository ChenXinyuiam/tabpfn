"""Predeclared, equal-budget feature ablation with paired seeds and geometry folds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

from .build_focus_boundary_features import run as build_features
from .cache import PredictionCache
from .dataset import build_labeled_dataset
from .evaluation import evaluate_predictions, make_grouped_folds
from .modeling import predict_probability, resolve_device
from .quality import FEATURE_SCHEMA_VERSION, FeatureConfig, feature_code_hashes
from .settings import CONTEXT_FEATURES, DEFAULT_LABEL_FILE, DEFAULT_MODEL_PATH, DEFAULT_RAW_DIR, ROOT

FEATURE_SETS = {
    "context": CONTEXT_FEATURES,
    "width": [*CONTEXT_FEATURES, "boundary_distance_abs"],
    "midpoint": [*CONTEXT_FEATURES, "boundary_distance_abs", "boundary_midpoint_x"],
}
DEFAULT_OUTPUT = ROOT / "results/focus_inspection/2026-09-04/focus_ablation"


def evaluate_ablation(
    data, *, model_path, device, seeds=(42, 2026, 3407), n_estimators=4, predictor=predict_probability
):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct, predeclared seeds")
    folds, assignments, audit = make_grouped_folds(
        data, n_splits=data["geometry_group"].nunique(), random_state=0, canonical_order=True
    )
    records, metrics = [], []
    for seed in seeds:
        result = data.copy()
        result["seed"] = seed
        result["cv_fold"] = assignments["cv_fold"].to_numpy()
        majority = np.full(len(data), -1, dtype=int)
        for name, columns in FEATURE_SETS.items():
            probability = np.full(len(data), np.nan)
            for fold_id, (fit_index, test_index) in enumerate(folds, 1):
                train = data.iloc[fit_index]
                test = data.iloc[test_index].drop(columns=["focus_label", "focus_label_name"], errors="ignore")
                probability[test_index] = predictor(
                    train,
                    test,
                    feature_columns=columns,
                    model_path=model_path,
                    device=device,
                    random_state=seed + fold_id * 100,
                    n_estimators=n_estimators,
                )
                majority[test_index] = int(train["focus_label"].mean() > 0.5)
                print(f"Seed {seed}: {name}, geometry fold {fold_id}/{len(folds)} complete", flush=True)
            if not np.isfinite(probability).all():
                raise RuntimeError("Incomplete ablation OOF probabilities")
            result[f"proba_{name}"] = probability
            result[f"pred_{name}"] = (probability > 0.5).astype(int)
        result["pred_majority"] = majority
        result["pred_rule_lt_100"] = np.where(
            data["boundary_distance_abs"].notna(), (data["boundary_distance_abs"] < 100).astype(int), np.nan
        )
        for name in [*FEATURE_SETS, "majority", "rule_lt_100"]:
            valid = result[f"pred_{name}"].notna()
            probability = result.loc[valid, f"proba_{name}"].to_numpy() if name in FEATURE_SETS else None
            metrics.append(
                {
                    "seed": seed,
                    "model": name,
                    "evaluated_samples": int(valid.sum()),
                    **evaluate_predictions(
                        result.loc[valid, "focus_label"].to_numpy(),
                        result.loc[valid, f"pred_{name}"].to_numpy(dtype=int),
                        probability,
                    ),
                }
            )
        records.append(result)
    return pd.concat(records, ignore_index=True), pd.DataFrame(metrics), audit


def paired_effect(data, *, candidate, reference, repeats=2000):
    effects = []
    for _, group in data.groupby("geometry_group"):
        difference = (
            group[f"pred_{candidate}"].eq(group.focus_label).sum()
            - group[f"pred_{reference}"].eq(group.focus_label).sum()
        )
        effects.append([difference, len(group)])
    effects = np.asarray(effects)
    indices = np.random.default_rng(42).integers(len(effects), size=(repeats, len(effects)))
    sums = effects[indices].sum(axis=1)
    return {
        "delta": float(effects[:, 0].sum() / effects[:, 1].sum()),
        "ci95": np.quantile(sums[:, 0] / sums[:, 1], [0.025, 0.975]).tolist(),
        "unit": "geometry group; paired seeds kept together",
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
    config=None,
    resume=False,
):
    from .ablation_report import write_report

    config = config or FeatureConfig()
    with experiment_run(out_dir, "focus-ablate", resume=resume) as directory:
        start = time.perf_counter()
        cache = PredictionCache(directory / "prediction_cache", model_path)
        protocol = {
            "feature_sets": FEATURE_SETS,
            "seeds": list(seeds),
            "n_estimators": n_estimators,
            "threshold": 0.5,
            "feature_config": config.to_dict(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_code_hashes": feature_code_hashes(),
            "evaluation_code_sha256": file_sha256(Path(__file__)),
            "fold_code_sha256": file_sha256(Path(__file__).with_name("evaluation.py")),
            "label_sha256": file_sha256(label_file),
            "model_identity": cache.identity,
            "source_hashes": {
                p.name: file_sha256(p) for p in sorted(raw_dir.glob("*Fzong*.xlsx")) if not p.name.startswith("~$")
            },
            "selection": "No feature, threshold or seed selection. Development comparison, not a fresh external test.",
        }
        path = directory / "protocol.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Cannot resume: inputs, parameters, implementation or environment changed")
        write_json(path, protocol)
        build_features(raw_dir, directory / "features", config=config, make_plots=False)
        data, data_audit = build_labeled_dataset(raw_dir, label_file, directory / "features/features.csv")
        data.to_csv(directory / "training_features.csv", index=False, encoding="utf-8-sig")
        result, metrics, audit = evaluate_ablation(
            data, model_path=model_path, device=device, seeds=seeds, n_estimators=n_estimators, predictor=cache
        )
        result.to_csv(directory / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        metrics.drop(columns="confusion_matrix").to_csv(
            directory / "metrics_by_seed.csv", index=False, encoding="utf-8-sig"
        )
        summary = {
            "data": data_audit,
            "folds": audit,
            "protocol": protocol,
            "width_minus_context": paired_effect(result, candidate="width", reference="context"),
            "midpoint_minus_width": paired_effect(result, candidate="midpoint", reference="width"),
            "duration_seconds": time.perf_counter() - start,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
        }
        write_json(directory / "evaluation.json", summary)
        for name, columns in FEATURE_SETS.items():
            write_json(
                directory / f"fixed_{name}_recipe.json",
                {
                    "candidate": name,
                    "feature_sets": {name: columns},
                    "threshold": 0.5,
                    "n_estimators": n_estimators,
                    "seed": seeds[0],
                    "model_path": str(model_path.resolve()),
                    "checkpoint_sha256": file_sha256(model_path),
                    "model_identity": cache.identity,
                    "training_features": "training_features.csv",
                    "training_features_sha256": file_sha256(directory / "training_features.csv"),
                    "feature_config": config.to_dict(),
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_code_hashes": feature_code_hashes(),
                    "selection": "Fixed predeclared recipe; first predeclared seed, not the best observed seed",
                },
            )
        write_report(directory, result, metrics, summary)
        print(metrics[["seed", "model", "accuracy", "balanced_accuracy", "specificity"]].to_string(index=False))
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
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--length-unit", choices=("um", "mm", "m"), default="um")
    parser.add_argument("--invalid-rows", choices=("error", "drop"), default="error")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.raw_dir,
        args.label_file,
        args.out_dir,
        args.model_path,
        resolve_device(args.device),
        seeds=tuple(args.seeds),
        n_estimators=args.n_estimators,
        resume=args.resume,
        config=FeatureConfig(args.window_size, args.length_unit, args.invalid_rows),
    )


if __name__ == "__main__":
    main()
