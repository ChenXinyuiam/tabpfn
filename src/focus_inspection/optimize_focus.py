"""Compare fixed TabPFN recipes with nested geometry-group validation."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

from .build_focus_boundary_features import run as build_features
from .cache import PredictionCache
from .dataset import build_labeled_dataset
from .evaluation import evaluate_predictions, make_grouped_folds
from .modeling import predict_probability, resolve_device
from .quality import FEATURE_SCHEMA_VERSION, FeatureConfig, feature_code_hashes
from .settings import CONTEXT_FEATURES, DEFAULT_LABEL_FILE, DEFAULT_MODEL_PATH, DEFAULT_RAW_DIR, ROOT

CANDIDATES = {
    "context": CONTEXT_FEATURES,
    "midpoint": [*CONTEXT_FEATURES, "boundary_distance_abs", "boundary_midpoint_x"],
    "spatial": [
        "re",
        "R",
        "bc",
        "h",
        "angle_a",
        "angle_b",
        "orientation_reverse",
        "boundary_distance_norm",
        "boundary_midpoint_norm",
        "boundary_gap_norm",
        "red_fraction",
        "row_midpoint_iqr_norm",
        "row_gap_median_norm",
        "row_gap_q90_norm",
    ],
}
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
DEFAULT_OUTPUT = ROOT / "results/focus_inspection/2026-09-04/focus_tabpfn_optimized"


def select_recipe(y: np.ndarray, probabilities: dict[str, np.ndarray]) -> dict[str, object]:
    """Selection sees only predictions for the current training partition."""
    best = None
    best_key = None
    for name, probability in probabilities.items():
        if len(probability) != len(y) or not np.isfinite(probability).all():
            raise ValueError("Recipe selection requires complete OOF probabilities")
        for threshold in THRESHOLDS:
            prediction = (probability > threshold).astype(int)
            accuracy = float(accuracy_score(y, prediction))
            balanced = float(balanced_accuracy_score(y, prediction))
            key = (accuracy, balanced, -abs(threshold - 0.5))
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "candidate": name,
                    "threshold": threshold,
                    "inner_accuracy": accuracy,
                    "inner_balanced_accuracy": balanced,
                }
    if best is None:
        raise ValueError("At least one candidate is required")
    return best


def assert_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    for column in ("geometry_group", "source_file"):
        overlap = set(train[column]) & set(test[column])
        if overlap:
            raise ValueError(f"Leakage in {column}: {sorted(overlap)}")


def candidate_probabilities(train, test, *, predictor, model_path, device, seed, n_estimators):
    # Test labels never cross the model interface, including during inner CV.
    unlabelled = test.drop(columns=["focus_label", "focus_label_name"], errors="ignore")
    predictions = {
        name: predictor(
            train,
            unlabelled,
            feature_columns=columns,
            model_path=model_path,
            device=device,
            random_state=seed,
            n_estimators=n_estimators,
        )
        for name, columns in CANDIDATES.items()
    }
    predictions["blend"] = np.mean(list(predictions.values()), axis=0)
    return predictions


def nested_evaluate(
    data, *, model_path, device, seed=42, n_estimators=4, inner_splits=3, predictor=predict_probability
):
    folds, assignments, outer_audit = make_grouped_folds(
        data,
        n_splits=data["geometry_group"].nunique(),
        random_state=seed,
    )
    names = [*CANDIDATES, "blend"]
    fixed = {name: np.full(len(data), np.nan) for name in names}
    baseline = np.full(len(data), np.nan)
    selected = np.full(len(data), np.nan)
    predictions = np.full(len(data), -1, dtype=int)
    choices, inner_audit = [], []
    for fold_id, (train_index, test_index) in enumerate(folds, start=1):
        train = data.iloc[train_index].reset_index(drop=True)
        test = data.iloc[test_index].reset_index(drop=True)
        assert_disjoint(train, test)
        if inner_splits < 2 or inner_splits > train["geometry_group"].nunique():
            raise ValueError("inner_splits must be between 2 and the number of outer-training geometry groups")
        splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold_id)
        inner = {name: np.full(len(train), np.nan) for name in names}
        coverage = np.zeros(len(train), dtype=int)
        for inner_id, (fit_index, valid_index) in enumerate(
            splitter.split(train, train["focus_label"], train["geometry_group"]), start=1
        ):
            fit, valid = train.iloc[fit_index], train.iloc[valid_index]
            assert_disjoint(fit, valid)
            assert_disjoint(fit, test)
            assert_disjoint(valid, test)
            probability = candidate_probabilities(
                fit,
                valid,
                predictor=predictor,
                model_path=model_path,
                device=device,
                seed=seed + fold_id * 100 + inner_id,
                n_estimators=n_estimators,
            )
            for name in names:
                inner[name][valid_index] = probability[name]
            coverage[valid_index] += 1
            inner_audit.append(
                {
                    "outer_fold": fold_id,
                    "inner_fold": inner_id,
                    "fit_groups": sorted(fit["geometry_group"].unique().tolist()),
                    "validation_groups": sorted(valid["geometry_group"].unique().tolist()),
                    "outer_test_groups": sorted(test["geometry_group"].unique().tolist()),
                }
            )
            print(f"Outer fold {fold_id}/{len(folds)}: inner fold {inner_id}/{inner_splits} complete", flush=True)
        if not (coverage == 1).all():
            raise RuntimeError("Incomplete inner OOF coverage")
        choice = select_recipe(train["focus_label"].to_numpy(), inner)
        probability = candidate_probabilities(
            train,
            test,
            predictor=predictor,
            model_path=model_path,
            device=device,
            seed=seed + fold_id * 100,
            n_estimators=n_estimators,
        )
        for name in names:
            fixed[name][test_index] = probability[name]
        selected[test_index] = probability[choice["candidate"]]
        predictions[test_index] = (selected[test_index] > choice["threshold"]).astype(int)
        baseline[test_index] = predictor(
            train,
            test.drop(columns=["focus_label", "focus_label_name"], errors="ignore"),
            feature_columns=[*CONTEXT_FEATURES, "boundary_distance_abs"],
            model_path=model_path,
            device=device,
            random_state=seed + 300 + fold_id,
            n_estimators=1,
        )
        choices.append(
            {"outer_fold": fold_id, "test_groups": sorted(test["geometry_group"].unique().tolist()), **choice}
        )
        print(
            f"Outer fold {fold_id}/{len(folds)} complete; selected {choice['candidate']} at {choice['threshold']}",
            flush=True,
        )
    if not np.isfinite(selected).all() or not np.isfinite(baseline).all() or (predictions < 0).any():
        raise RuntimeError("Incomplete outer OOF predictions")
    result = data.copy()
    result["cv_fold"] = assignments["cv_fold"].to_numpy()
    result["proba_baseline"] = baseline
    result["pred_baseline"] = (baseline > 0.5).astype(int)
    for name, probability in fixed.items():
        result[f"proba_{name}"] = probability
        result[f"pred_{name}"] = (probability > 0.5).astype(int)
    result["proba_nested"] = selected
    result["pred_nested"] = predictions
    y = data["focus_label"].to_numpy()
    metrics = pd.DataFrame(
        [
            {
                "model": name,
                **evaluate_predictions(y, result[f"pred_{name}"].to_numpy(), result[f"proba_{name}"].to_numpy()),
            }
            for name in ["baseline", *names, "nested"]
        ]
    )
    return result, metrics, {"outer_folds": outer_audit, "inner_folds": inner_audit, "choices": choices}


def paired_group_bootstrap(
    data: pd.DataFrame, *, repeats: int = 2000, seed: int = 42, prediction_column: str = "pred_nested"
) -> dict:
    group_effects = []
    for _, group in data.groupby("geometry_group"):
        base = group["pred_baseline"].eq(group["focus_label"]).sum()
        new = group[prediction_column].eq(group["focus_label"]).sum()
        group_effects.append([int(new - base), len(group)])
    effects = np.asarray(group_effects)
    samples = np.random.default_rng(seed).integers(len(effects), size=(repeats, len(effects)))
    totals = effects[samples].sum(axis=1)
    deltas = totals[:, 0] / totals[:, 1]
    return {
        "unit": "geometry_group",
        "repeats": repeats,
        "accuracy_delta": float(effects[:, 0].sum() / effects[:, 1].sum()),
        "accuracy_delta_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
    }


def save_fixed_recipes(out_dir: Path, selected_recipe: dict) -> None:
    for name, columns, estimators in [
        ("baseline", [*CONTEXT_FEATURES, "boundary_distance_abs"], 1),
        ("midpoint", CANDIDATES["midpoint"], selected_recipe["n_estimators"]),
    ]:
        write_json(
            out_dir / f"fixed_{name}_recipe.json",
            {
                **selected_recipe,
                "candidate": name,
                "threshold": 0.5,
                "feature_sets": {name: columns},
                "n_estimators": estimators,
                "selection": "Fixed benchmark recipe; future predictions are not independent validation scores",
            },
        )


def _run(
    raw_dir: Path,
    label_file: Path,
    out_dir: Path,
    model_path: Path,
    device: str,
    seed: int = 42,
    n_estimators: int = 4,
    inner_splits: int = 3,
    feature_config: FeatureConfig | None = None,
):
    from .optimization_report import write_report

    start = time.perf_counter()
    feature_config = feature_config or FeatureConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PredictionCache(out_dir / "prediction_cache", model_path)
    protocol = {
        "candidates": CANDIDATES,
        "blend": "equal probability mean of all three candidates",
        "thresholds": list(THRESHOLDS),
        "selection_metric": "accuracy; balanced accuracy breaks ties",
        "n_estimators": n_estimators,
        "inner_splits": inner_splits,
        "seed": seed,
        "feature_config": feature_config.to_dict(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_code_hashes": feature_code_hashes(),
        "model_identity": cache.identity,
        "evaluation_code_sha256": file_sha256(Path(__file__)),
        "source_hashes": {
            p.name: file_sha256(p) for p in sorted(raw_dir.glob("*Fzong*.xlsx")) if not p.name.startswith("~$")
        },
        "outer_protocol": "leave one geometry group out",
        "label_sha256": file_sha256(label_file),
        "checkpoint_sha256": file_sha256(model_path),
        "versions": {key: version(key) for key in ["tabpfn", "torch", "numpy", "pandas", "scikit-learn"]},
    }
    protocol_path = out_dir / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in protocol.items()):
            raise ValueError("Cannot resume: inputs, parameters, code or environment changed")
    write_json(protocol_path, protocol)
    build_features(raw_dir, out_dir / "features", make_plots=False, config=feature_config)
    feature_file = out_dir / "features/features.csv"
    data, data_audit = build_labeled_dataset(raw_dir, label_file, feature_file)
    protocol["features_sha256"] = file_sha256(feature_file)
    write_json(out_dir / "protocol.json", protocol)
    data.to_csv(out_dir / "training_features.csv", index=False, encoding="utf-8-sig")
    result, metrics, audit = nested_evaluate(
        data,
        model_path=model_path,
        device=device,
        seed=seed,
        n_estimators=n_estimators,
        inner_splits=inner_splits,
        predictor=cache,
    )
    result.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.drop(columns="confusion_matrix").to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit["choices"]).to_csv(out_dir / "fold_choices.csv", index=False, encoding="utf-8-sig")
    result.loc[result["pred_nested"] != result["focus_label"]].to_csv(
        out_dir / "errors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_group = (
        result.groupby("geometry_group")
        .apply(
            lambda g: pd.Series(
                {
                    "samples": len(g),
                    "baseline_accuracy": g["pred_baseline"].eq(g["focus_label"]).mean(),
                    "midpoint_accuracy": g["pred_midpoint"].eq(g["focus_label"]).mean(),
                    "nested_accuracy": g["pred_nested"].eq(g["focus_label"]).mean(),
                }
            ),
        )
        .reset_index()
    )
    per_group.to_csv(out_dir / "metrics_by_geometry.csv", index=False, encoding="utf-8-sig")
    deployment = select_recipe(
        data["focus_label"].to_numpy(), {name: result[f"proba_{name}"].to_numpy() for name in [*CANDIDATES, "blend"]}
    )
    recipe = {
        "candidate": deployment["candidate"],
        "threshold": deployment["threshold"],
        "feature_sets": CANDIDATES,
        "n_estimators": n_estimators,
        "seed": seed,
        "model_path": str(model_path.resolve()),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "training_features": "training_features.csv",
        "training_features_sha256": file_sha256(out_dir / "training_features.csv"),
        "model_identity": cache.identity,
        "feature_config": feature_config.to_dict(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_code_hashes": feature_code_hashes(),
        "selection": "All-data grouped OOF for future deployment; evaluation is the nested row in metrics.csv",
    }
    write_json(out_dir / "selected_recipe.json", recipe)
    save_fixed_recipes(out_dir, recipe)
    summary = {
        "data": data_audit,
        "protocol": protocol,
        "leakage_audit": audit,
        "paired_uncertainty": paired_group_bootstrap(result, seed=seed),
        "midpoint_paired_uncertainty": paired_group_bootstrap(result, seed=seed, prediction_column="pred_midpoint"),
        "duration_seconds": time.perf_counter() - start,
        "cache": {"hits": cache.hits, "misses": cache.misses},
        "metrics": metrics.to_dict("records"),
    }
    write_json(out_dir / "evaluation.json", summary)
    write_report(out_dir, result, metrics, per_group, summary)
    print(metrics[["model", "accuracy", "balanced_accuracy", "f1"]].to_string(index=False), flush=True)
    return result, metrics, summary


def run(
    raw_dir,
    label_file,
    out_dir,
    model_path,
    device,
    seed=42,
    n_estimators=4,
    inner_splits=3,
    *,
    resume=False,
    feature_config=None,
):
    with experiment_run(out_dir, "focus-optimize", resume=resume) as run_dir:
        result, metrics, summary = _run(
            raw_dir, label_file, run_dir, model_path, device, seed, n_estimators, inner_splits, feature_config
        )
        summary["output_dir"] = str(run_dir)
        return result, metrics, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Use --out-dir as the exact interrupted run directory")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--length-unit", choices=("um", "mm", "m"), default="um")
    parser.add_argument("--invalid-rows", choices=("error", "drop"), default="error")
    args = parser.parse_args()
    run(
        args.raw_dir,
        args.label_file,
        args.out_dir,
        args.model_path,
        resolve_device(args.device),
        args.seed,
        args.n_estimators,
        args.inner_splits,
        resume=args.resume,
        feature_config=FeatureConfig(args.window_size, args.length_unit, args.invalid_rows),
    )


if __name__ == "__main__":
    main()
