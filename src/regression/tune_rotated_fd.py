"""Grouped TabPFN benchmark for verified two-component rotation data."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.regression.evaluation import metric_row, stable_seed
from src.regression.rotation_dataset import ANGLES, load_snapshot
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

RAW = ["bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c", "perimeter", "area", "radius", "re", "x", "z"]
INNER_ANGLES = (0, 45, 90, 135)


@dataclass(frozen=True)
class RotRecipe:
    name: str
    mode: str
    estimators: int
    canonical: bool = False


RECIPES = (
    RotRecipe("direct_e1", "angle", 1),
    RotRecipe("direct_trig_e4", "trig", 4),
    RotRecipe("components_e1", "components", 1),
    RotRecipe("components_e4", "components", 4),
    RotRecipe("components_canonical_e4", "components", 4, True),
)


def geometry_folds(groups, count, seed):
    names = np.array(sorted(set(groups)))
    if len(names) < count:
        raise ValueError("Insufficient independent geometries")
    return [(set(names[a]), set(names[b])) for a, b in KFold(count, shuffle=True, random_state=seed).split(names)]


def sample_context(data, budget, seed):
    """Equal geometry quota, then source/Re quotas, without inspecting targets."""
    groups = list(data.groupby("geometry_group", sort=True))
    if budget < len(groups):
        raise ValueError("Budget must cover every geometry")
    chunks = []
    for i, (geometry, group) in enumerate(groups):
        quota = min(len(group), budget // len(groups) + int(i < budget % len(groups)))
        strata = list(group.groupby(["source_file", "re"], sort=True))
        selected = []
        for j, (key, rows) in enumerate(strata):
            take = min(len(rows), quota // len(strata) + int(j < quota % len(strata)))
            if take:
                selected.append(rows.sample(take, random_state=stable_seed(f"{geometry}|{key}", seed)))
        selected = pd.concat(selected)
        extra = quota - len(selected)
        if extra:
            remaining = group[~group.base_id.isin(selected.base_id)]
            selected = pd.concat([selected, remaining.sample(extra, random_state=seed)])
        chunks.append(selected)
    result = pd.concat(chunks).sample(frac=1, random_state=seed).reset_index(drop=True)
    if result.base_id.duplicated().any() or len(result) > budget:
        raise ValueError("Invalid context sample")
    return result


def features(data, recipe):
    if recipe.canonical:
        result = data[["bc", "h", "radius", "re", "angle_a", "angle_b", "angle_c", "bary_a", "bary_b", "bary_c"]].copy()
        for column in ("ab", "ac", "h", "x", "radius"):
            result[column + "_over_bc"] = data[column] / data.bc
        result["z_over_h"] = data.z / data.h
        result["bary_ab"] = data.bary_a * data.bary_b
        result["bary_ac"] = data.bary_a * data.bary_c
        result["bary_bc"] = data.bary_b * data.bary_c
    else:
        result = data[RAW].copy()
    if recipe.mode == "angle":
        result["rotation_deg"] = data.rotation_deg
    elif recipe.mode == "trig":
        theta = np.deg2rad(data.rotation_deg)
        result["rotation_sin"], result["rotation_cos"] = np.sin(theta), np.cos(theta)
    return result


def direct_training(context, seed):
    # Two orthogonal projections contain the same two labels per base point as
    # the component models. The full source dataset has been checked beforehand.
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(ANGLES), len(context)) * 15
    theta = np.column_stack([first, (first + 90) % 360]).ravel()
    rows = context.iloc[np.repeat(np.arange(len(context)), 2)].reset_index(drop=True)
    rows["rotation_deg"] = theta
    rows["target"] = rows.u0 * np.cos(np.deg2rad(theta)) + rows.u90 * np.sin(np.deg2rad(theta))
    return rows


def queries(base, angles):
    rows = base.iloc[np.repeat(np.arange(len(base)), len(angles))].reset_index(drop=True)
    rows["rotation_deg"] = np.tile(angles, len(base))
    return rows


def fit_predict(train_x, y, query_x, recipe, args):
    import torch
    from tabpfn import TabPFNRegressor

    preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), MinMaxScaler())
    x = preprocessor.fit_transform(train_x)
    q = preprocessor.transform(query_x)
    model = TabPFNRegressor(
        model_path=args.model_path,
        device=args.device,
        n_estimators=recipe.estimators,
        random_state=args.seed,
        ignore_pretraining_limits=True,
        show_progress_bar=False,
    )
    try:
        model.fit(x, np.asarray(y, dtype=float))
        return np.concatenate(
            [
                np.asarray(model.predict(q[start : start + args.chunk_size])).reshape(-1)
                for start in range(0, len(q), args.chunk_size)
            ]
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def predict(context, test, angles, recipe, args, fitter=fit_predict):
    if set(context.geometry_group) & set(test.geometry_group):
        raise ValueError("Geometry leakage")
    if recipe.mode == "components":
        train_x, query_x = features(context, recipe), features(test, recipe)
        p0 = fitter(train_x, context.u0, query_x, recipe, args)
        p90 = fitter(train_x, context.u90, query_x, recipe, args)
        theta = np.deg2rad(angles)
        result = (p0[:, None] * np.cos(theta) + p90[:, None] * np.sin(theta)).ravel()
    else:
        training = direct_training(context, args.seed)
        result = fitter(
            features(training, recipe), training.target, features(queries(test, angles), recipe), recipe, args
        )
    if not np.isfinite(result).all() or len(result) != len(test) * len(angles):
        raise ValueError("Invalid predictions")
    return result


def observed(test, angles):
    return test[[f"observed_{angle:03d}" for angle in angles]].to_numpy().ravel()


def geometry_metrics(test, angles, prediction):
    labels = observed(test, angles)
    groups = np.repeat(test.geometry_group.to_numpy(), len(angles))
    return [
        {"geometry_group": group, **metric_row(labels[groups == group], prediction[groups == group])}
        for group in sorted(set(groups))
    ]


def selection_metrics(test, angles, prediction):
    rows = geometry_metrics(test, angles, prediction)
    matrix = np.asarray(prediction).reshape(len(test), len(angles))
    for row in rows:
        mask = test.geometry_group.to_numpy() == row["geometry_group"]
        row["r2_all_inner_angles"] = row["r2"]
        scores = []
        for angle in (0, 90):
            score = metric_row(test.loc[mask, f"observed_{angle:03d}"].to_numpy(), matrix[mask, angles.index(angle)])[
                "r2"
            ]
            row[f"r2_{angle:03d}"] = score
            scores.append(score)
        row["r2"] = float(np.mean(scores))
    return rows


def choose_recipe(rows, expected_groups, recipes=RECIPES):
    scores = {}
    for recipe in recipes:
        selected = rows[rows.recipe == recipe.name]
        if len(selected) != len(expected_groups) or set(selected.geometry_group) != set(expected_groups):
            raise ValueError("Incomplete inner geometry coverage")
        if not np.isfinite(selected.r2).all():
            raise ValueError("Undefined selection scores")
        scores[recipe.name] = float(selected.r2.mean())
    return max(recipes, key=lambda recipe: scores[recipe.name])


def evaluate(base, test_pool, args, out, recipes=RECIPES, predictor=predict):
    inner_rows, folds, outputs, manifests, selections, timings = [], [], [], [], [], []
    for fold, (train_groups, test_groups) in enumerate(
        geometry_folds(base.geometry_group, args.outer_folds, args.seed), 1
    ):
        candidates = []
        for inner, (fit_groups, val_groups) in enumerate(
            geometry_folds(train_groups, 3, stable_seed(str(fold), args.seed)), 1
        ):
            context = sample_context(base[base.geometry_group.isin(fit_groups)], args.label_budget // 2, args.seed)
            validation = test_pool[test_pool.geometry_group.isin(val_groups)]
            manifests.append(
                {
                    "outer_fold": fold,
                    "inner_fold": inner,
                    "fit_groups": sorted(fit_groups),
                    "validation_groups": sorted(val_groups),
                    "outer_test_groups": sorted(test_groups),
                    "context_base_ids": context.base_id.tolist(),
                }
            )
            for recipe in recipes:
                start = time.perf_counter()
                prediction = predictor(context, validation, INNER_ANGLES, recipe, args)
                timings.append(
                    {"outer": fold, "inner": inner, "recipe": recipe.name, "seconds": time.perf_counter() - start}
                )
                rows = [
                    {"outer_fold": fold, "inner_fold": inner, "recipe": recipe.name, **row}
                    for row in selection_metrics(validation, INNER_ANGLES, prediction)
                ]
                candidates.extend(rows)
                print(
                    f"outer={fold} inner={inner} {recipe.name}: geometry R2={np.mean([r['r2'] for r in rows]):.4f}",
                    flush=True,
                )
                pd.DataFrame([*inner_rows, *candidates]).to_csv(out / "inner_metrics.csv", index=False)
        chosen = choose_recipe(pd.DataFrame(candidates), train_groups, recipes)
        selections.append({"fold": fold, "recipe": chosen.name, "test_groups": sorted(test_groups)})
        inner_rows.extend(candidates)
        context = sample_context(base[base.geometry_group.isin(train_groups)], args.label_budget // 2, args.seed)
        test = test_pool[test_pool.geometry_group.isin(test_groups)]
        manifests.append(
            {
                "outer_fold": fold,
                "inner_fold": None,
                "fit_groups": sorted(train_groups),
                "outer_test_groups": sorted(test_groups),
                "context_base_ids": context.base_id.tolist(),
            }
        )
        for recipe in recipes:
            if recipe not in (recipes[0], chosen):
                continue
            start = time.perf_counter()
            prediction = predictor(context, test, ANGLES, recipe, args)
            timings.append(
                {"outer": fold, "inner": None, "recipe": recipe.name, "seconds": time.perf_counter() - start}
            )
            methods = (["direct_baseline"] if recipe == recipes[0] else []) + (
                ["nested_tabpfn"] if recipe == chosen else []
            )
            for method in methods:
                for row in geometry_metrics(test, ANGLES, prediction):
                    folds.append({"fold": fold, "method": method, "recipe": recipe.name, **row})
                result = queries(test, ANGLES)[
                    ["base_id", "source_file", "source_row", "geometry_group", "radius", "re", "x", "z", "rotation_deg"]
                ].copy()
                result["target"], result["prediction"] = observed(test, ANGLES), prediction
                result["method"], result["recipe"], result["fold"] = method, recipe.name, fold
                outputs.append(result)
                print(
                    f"OUTER {fold} {method}: R2={metric_row(result.target.to_numpy(), prediction)['r2']:.4f}",
                    flush=True,
                )
        pd.DataFrame(folds).to_csv(out / "geometry_metrics.csv", index=False)
        pd.concat(outputs).to_csv(out / "predictions.csv", index=False)
        write_json(out / "splits.json", manifests)
        write_json(out / "selections.json", selections)
        write_json(out / "timings.json", timings)
    predictions = pd.concat(outputs, ignore_index=True)
    geometry = pd.DataFrame(folds)
    summary, angle_rows = [], []
    for method, part in predictions.groupby("method"):
        rows = geometry[geometry.method == method]
        if set(rows.geometry_group) != set(base.geometry_group) or len(rows) != base.geometry_group.nunique():
            raise ValueError("Incomplete outer geometry coverage")
        summary.append(
            {
                "method": method,
                "geometry_mean_r2": float(rows.r2.mean()),
                "geometry_worst_r2": float(rows.r2.min()),
                "rows": len(part),
                **metric_row(part.target.to_numpy(), part.prediction.to_numpy()),
            }
        )
        for (group, angle), subset in part.groupby(["geometry_group", "rotation_deg"]):
            angle_rows.append(
                {
                    "method": method,
                    "geometry_group": group,
                    "rotation_deg": int(angle),
                    **metric_row(subset.target.to_numpy(), subset.prediction.to_numpy()),
                }
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(angle_rows).to_csv(out / "angle_metrics.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_tabpfn_rotation/2026-09-29"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/models/tabpfn-v2.5-regressor-v2.5_default.ckpt")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label-budget", type=int, default=3000)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.label_budget < 2 or args.label_budget % 2 or args.chunk_size <= 0:
        parser.error("Positive even label budget and positive chunk size required")
    if not args.model_path.is_file():
        parser.error("Local checkpoint required")
    sources = [__file__, Path(__file__).with_name("rotation_dataset.py"), Path(__file__).with_name("evaluation.py")]
    with experiment_run(args.output_dir, "tabpfn-rotated-fd", extra_sources=sources) as out:
        base, test, info = load_snapshot(args.snapshot)
        write_json(
            out / "config.json",
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "recipes": [asdict(r) for r in RECIPES],
                "snapshot": info,
                "inner_angles": INNER_ANGLES,
                "outer_angles": ANGLES,
                "checkpoint_sha256": file_sha256(args.model_path),
                "selection": "inner geometry-equal and 0/90-reference-angle-equal R2",
                "target": "native rotated u",
            },
        )
        print(evaluate(base, test, args, out).to_string(index=False), flush=True)
        from src.regression.rotation_report import write_report

        write_report(out)


if __name__ == "__main__":
    main()
