"""Native FD-u research benchmark with nested leave-one-geometry-out selection."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

from src.regression.evaluation import (
    BASE_REQUIRED,
    FEATURE_COLUMNS,
    balanced_sample,
    geometry_signature,
    metric_row,
    parse_radius,
    sample_complete_positions,
    stable_seed,
)
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run


@dataclass(frozen=True)
class Recipe:
    name: str
    backend: str
    relative: bool = False
    re_power: int = 0


# Freeze this menu before observing outer-fold outcomes. Scaling is empirical,
# not a claim that u is a force or follows a particular physical law.
RECIPES = (
    Recipe("raw_extra_trees", "extra-trees"),
    Recipe("relative_extra_trees", "extra-trees", True),
    Recipe("relative_re1_extra_trees", "extra-trees", True, 1),
    Recipe("relative_re2_extra_trees", "extra-trees", True, 2),
    Recipe("raw_tabpfn", "tabpfn"),
    Recipe("relative_re1_tabpfn", "tabpfn", True, 1),
)


def load_native_fd(folder: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Fail on invalid rows; preserve questionable but finite source geometry."""
    parts, inventory = [], []
    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        data = pd.read_excel(path, sheet_name=0)
        required = [*BASE_REQUIRED, "u"]
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError(f"{path.name}: missing {missing}")
        data[required] = data[required].apply(pd.to_numeric, errors="coerce")
        if data.empty or not np.isfinite(data[required].to_numpy()).all():
            raise ValueError(f"{path.name}: empty, missing or non-finite numeric data")
        if (data[["bc", "ab", "ac", "h", "re"]] <= 0).any().any():
            raise ValueError(f"{path.name}: nonpositive geometry/Re")
        for column in ("bc", "ab", "ac", "h"):
            if not np.allclose(data[column], data[column].iloc[0], rtol=1e-8):
                raise ValueError(f"{path.name}: mixed geometry in {column}")
        data["radius"] = parse_radius(path)
        data["x_over_h"] = data.x / data.h
        data["z_over_h"] = data.z / data.h
        data["target"] = data.u
        data["source_file"] = path.name
        data["source_row"] = np.arange(len(data)) + 2
        data["geometry_group"] = geometry_signature(data)
        data["position_group"] = path.name + "|" + data.x.round(8).astype(str) + "|" + data.z.round(8).astype(str)
        sides = np.sort(data[["ab", "ac", "bc"]].iloc[0].to_numpy())
        if sides[0] + sides[1] <= sides[2]:
            raise ValueError(f"{path.name}: invalid triangle sides")
        s = sides.sum() / 2
        heron = float(np.sqrt(s * np.prod(s - sides)))
        inventory.append(
            {
                "source_file": path.name,
                "sha256": file_sha256(path),
                "rows": len(data),
                "geometry_group": data.geometry_group.iloc[0],
                "re_min": float(data.re.min()),
                "re_max": float(data.re.max()),
                "radius_label": int(data.radius.iloc[0]),
                "area_mismatch_rows": int((~np.isclose(data.area, data.bc * data.h / 2, rtol=0.02)).sum()),
                "height_vs_sides_mismatch": not bool(
                    np.isclose(heron, data.bc.iloc[0] * data.h.iloc[0] / 2, rtol=0.02)
                ),
                "z_outside_0_h_rows": int(((data.z < 0) | (data.z > data.h)).sum()),
                "duplicate_re_x_z_rows": int(data.duplicated(["re", "x", "z"]).sum()),
                "negative_u_rows": int((data.u < 0).sum()),
            }
        )
        parts.append(data)
    if not parts:
        raise ValueError(f"No FD workbooks: {folder}")
    return pd.concat(parts, ignore_index=True), inventory


def features(data: pd.DataFrame, recipe: Recipe) -> pd.DataFrame:
    if not recipe.relative:
        return data[FEATURE_COLUMNS].copy()
    # Keep absolute scales, but do not trust the supplied area/perimeter fields.
    result = data[["bc", "h", "radius", "re"]].copy()
    for column in ("ab", "ac", "h", "radius", "x"):
        result[column + "_over_bc"] = data[column] / data.bc
    result["z_over_h"] = data.z / data.h
    for column in ("prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc", "line_dist_to_ab", "line_dist_to_ac"):
        result[column + "_over_h"] = data[column] / data.h
    result["edge_product"] = (
        result.prep_dist_to_ab_over_h * result.prep_dist_to_ac_over_h * result.prep_dist_to_bc_over_h
    )
    return result


def target_factor(data: pd.DataFrame, recipe: Recipe) -> np.ndarray:
    if recipe.re_power == 0:
        return np.ones(len(data))
    return ((data.re.to_numpy() / 30) ** recipe.re_power) * (20 / data.radius.to_numpy())


def predict(train: pd.DataFrame, test: pd.DataFrame, recipe: Recipe, args) -> np.ndarray:
    if set(train.geometry_group) & set(test.geometry_group):
        raise ValueError("Geometry leakage")
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(features(train, recipe))
    x_test = scaler.transform(features(test, recipe))
    y = train.target.to_numpy() / target_factor(train, recipe)
    # Relative recipes standardize tiny native targets for numerical stability.
    center = float(y.mean()) if recipe.relative else 0.0
    scale = float(y.std()) if recipe.relative else 1.0
    scale = scale if scale > 0 else 1.0
    if recipe.backend == "extra-trees":
        model = ExtraTreesRegressor(n_estimators=args.trees, n_jobs=-1, random_state=args.seed)
    else:
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            model_path=args.model_path,
            device=args.device,
            n_estimators=1,
            ignore_pretraining_limits=True,
            random_state=args.seed,
            show_progress_bar=False,
        )
    try:
        model.fit(x_train, (y - center) / scale)
        pred = np.concatenate(
            [
                np.asarray(model.predict(x_test[start : start + args.chunk_size])).reshape(-1)
                for start in range(0, len(test), args.chunk_size)
            ]
        )
        pred = (pred * scale + center) * target_factor(test, recipe)
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite predictions: {recipe.name}")
        return pred
    finally:
        del model
        gc.collect()
        if recipe.backend == "tabpfn":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def split(data: pd.DataFrame, held_out: str, args, *, validation=False):
    train = data[data.geometry_group != held_out]
    test = data[data.geometry_group == held_out]
    key = "|".join(sorted(train.geometry_group.unique()))
    train = balanced_sample(train, args.max_train_rows, stable_seed(key, args.seed))
    test = sample_complete_positions(
        test,
        args.validation_rows if validation else args.test_rows,
        stable_seed(held_out, args.seed),
    )
    if train.empty or test.empty or set(train.geometry_group) & set(test.geometry_group):
        raise ValueError("Invalid geometry split")
    return train, test


def select_recipe(inner_scores: pd.DataFrame, recipes: tuple[Recipe, ...], groups: set[str]) -> Recipe:
    scores = {}
    for recipe in recipes:
        rows = inner_scores[inner_scores.recipe == recipe.name]
        if len(rows) != len(groups) or set(rows.validation_geometry) != groups:
            raise ValueError("Incomplete inner-fold coverage")
        if not np.isfinite(rows.r2.to_numpy()).all():
            raise ValueError("Undefined inner R2; cannot select a recipe")
        scores[recipe.name] = float(rows.r2.mean())
    # Stable menu order resolves ties, with no access to outer labels.
    return max(recipes, key=lambda recipe: scores[recipe.name])


def evaluate(data: pd.DataFrame, args, out: Path, recipes=RECIPES, predictor=predict):
    groups = sorted(data.geometry_group.unique())
    if len(groups) < 4:
        raise ValueError("Need at least four independent geometry groups for nested evaluation")
    all_inner, selections, folds, predictions, split_records = [], [], [], [], []
    baseline_names = {"raw_extra_trees", "raw_tabpfn"}
    for outer in groups:
        pool = data[data.geometry_group != outer]
        inner = []
        for validation in sorted(pool.geometry_group.unique()):
            train, test = split(pool, validation, args, validation=True)
            split_records.append(
                {
                    "outer": outer,
                    "validation": validation,
                    "train_geometries": sorted(train.geometry_group.unique()),
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
            for recipe in recipes:
                pred = predictor(train, test, recipe, args)
                row = {
                    "outer_geometry": outer,
                    "validation_geometry": validation,
                    "recipe": recipe.name,
                    **metric_row(test.target.to_numpy(), pred),
                }
                inner.append(row)
                print(f"inner {outer} / {validation} / {recipe.name}: R2={row['r2']:.4f}", flush=True)
        all_inner.extend(inner)
        chosen = select_recipe(pd.DataFrame(inner), recipes, set(pool.geometry_group))
        selections.append({"outer_geometry": outer, "selected_recipe": chosen.name})
        train, test = split(data, outer, args)
        split_records.append(
            {
                "outer": outer,
                "validation": None,
                "train_geometries": sorted(train.geometry_group.unique()),
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        for recipe in recipes:
            if recipe.name not in baseline_names and recipe != chosen:
                continue
            pred = predictor(train, test, recipe, args)
            methods = [recipe.name] if recipe.name in baseline_names else []
            if recipe == chosen:
                methods.append("nested_selected")
            for method in methods:
                folds.append(
                    {
                        "outer_geometry": outer,
                        "method": method,
                        "recipe": recipe.name,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "test_rows_available": int((data.geometry_group == outer).sum()),
                        **metric_row(test.target.to_numpy(), pred),
                    }
                )
                output = test[
                    ["source_file", "source_row", "geometry_group", "radius", "re", "x", "z", "target"]
                ].copy()
                output["prediction"], output["method"], output["recipe"] = pred, method, recipe.name
                predictions.append(output)
                print(f"OUTER {outer} / {method}: R2={folds[-1]['r2']:.4f}", flush=True)
        # Persist partial progress, but experiment_run marks only the full run complete.
        pd.DataFrame(all_inner).to_csv(out / "inner_metrics.csv", index=False)
        pd.DataFrame(folds).to_csv(out / "fold_metrics.csv", index=False)
        write_json(out / "selections.json", selections)
        write_json(out / "splits.json", split_records)
    folds = pd.DataFrame(folds)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    summary = []
    for method, part in predictions.groupby("method", sort=True):
        per_fold = folds[folds.method == method]
        if set(per_fold.outer_geometry) != set(groups):
            raise ValueError("Incomplete outer-fold coverage")
        summary.append(
            {
                "method": method,
                "geometry_mean_r2": float(per_fold.r2.mean()),
                "geometry_worst_r2": float(per_fold.r2.min()),
                "rows": len(part),
                "geometry_groups": len(groups),
                **{
                    "pooled_" + key: value
                    for key, value in metric_row(part.target.to_numpy(), part.prediction.to_numpy()).items()
                },
            }
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    return folds, summary


def report(out: Path, folds: pd.DataFrame, summary: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = folds.pivot(index="outer_geometry", columns="method", values="r2")
    ax = pivot.plot.bar(figsize=(12, 5), rot=15, color=["#248f79", "#527ac9", "#d36a54"])
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set(
        xlabel="Held-out geometry (sorted side lengths)",
        ylabel="R2 on native FD-u",
        title="Nested unseen-geometry validation",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(out / "outer_r2.png", dpi=150)
    plt.close(ax.figure)
    text = (
        """# FD-u Nested Geometry Evaluation

This is a native-data research benchmark, not approval by the system audit gate.
Four independent geometries imply four outer folds, not five independent folds.
Mirrored versions and all Re/radius files of a shape stay in the same fold.
Recipes are selected by mean inner held-out-geometry R2; outer labels never select them.
Relative recipes also standardize the training target; this is a combined change, not a feature-only ablation.
Re/radius normalization is an empirical hypothesis. Native u meaning and units are unconfirmed.
Data integrity warnings are retained in data_inventory.csv; no source values are corrected.
Test positions are sampled with all their Re rows. Scores do not cover every source row.
Historical geometry outcomes already informed development: this is a nested research estimate, not a fresh prospective test.
No production model is replaced. Four shapes and one seed cannot certify distant extrapolation.

## Summary

```text
"""
        + summary.to_string(index=False)
        + "\n```\n\n## Outer Folds\n\n```text\n"
        + folds.to_string(index=False)
        + "\n```\n"
    )
    (out / "report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/rawdata/2026-06-20"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_extrapolation_optimization"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/models/tabpfn-v2.5-regressor-v2.5_default.ckpt")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extra-trees-only", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=3000)
    parser.add_argument("--validation-rows", type=int, default=200)
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("max_train_rows", "validation_rows", "test_rows", "chunk_size", "trees"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    recipes = tuple(r for r in RECIPES if not args.extra_trees_only or r.backend == "extra-trees")
    if not args.extra_trees_only and not args.model_path.is_file():
        parser.error(f"Checkpoint not found: {args.model_path}")
    with experiment_run(
        args.output_dir, "fd-nested-extrapolation", extra_sources=[__file__, Path(__file__).with_name("evaluation.py")]
    ) as out:
        write_json(
            out / "config.json",
            {
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "recipes": [asdict(r) for r in recipes],
                "selection": "mean_inner_geometry_r2",
                "checkpoint_sha256": file_sha256(args.model_path) if not args.extra_trees_only else None,
            },
        )
        data, inventory = load_native_fd(args.data_root / "FD")
        if args.max_train_rows < data.source_file.nunique():
            raise ValueError("Training budget must cover source files")
        pd.DataFrame(inventory).to_csv(out / "data_inventory.csv", index=False)
        print(
            f"Loaded {len(data)} rows, {data.source_file.nunique()} files, {data.geometry_group.nunique()} geometries",
            flush=True,
        )
        folds, summary = evaluate(data, args, out, recipes)
        report(out, folds, summary)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
