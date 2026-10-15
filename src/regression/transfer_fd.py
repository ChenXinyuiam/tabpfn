"""Nested FD transfer experiments with coordinate-only mesh features and scaling."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.regression.evaluation import metric_row, stable_seed
from src.regression.rotation_dataset import ANGLES, load_snapshot
from src.regression.transfer_diagnostics import identical_components, repeated_conditions
from src.regression.tune_rotated_fd import (
    RECIPES as OLD_RECIPES,
)
from src.regression.tune_rotated_fd import (
    features as old_features,
)
from src.regression.tune_rotated_fd import (
    fit_predict,
    geometry_folds,
    observed,
    queries,
    sample_context,
)
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run


@dataclass(frozen=True)
class TransferRecipe:
    name: str
    coordinates: str = "mesh"
    powers: tuple[int, int] = (2, 1)
    rows: int = 1500
    estimators: int = 4
    checkpoint: str = "v25"
    boundary: bool = False
    symmetric: bool = False
    median: bool = False
    local_re: bool = False
    guarded: bool = False


RECIPES = (
    TransferRecipe("provided_raw_1500", "provided", (0, 0)),
    TransferRecipe("provided_scaled_1500", "provided"),
    TransferRecipe("mesh_raw_1500", powers=(0, 0)),
    TransferRecipe("mesh_scaled_1500"),
    TransferRecipe("mesh_scaled_4000", rows=4000),
    TransferRecipe("mesh_re2_4000", powers=(2, 2), rows=4000),
    TransferRecipe("mesh_boundary_4000", rows=4000, boundary=True),
    TransferRecipe("mesh_v3_4000", rows=4000, checkpoint="v3"),
    TransferRecipe("mesh_raw_4000", powers=(0, 0), rows=4000),
    TransferRecipe("mesh_raw_8000", powers=(0, 0), rows=8000),
    TransferRecipe("mesh_v3_raw_4000", powers=(0, 0), rows=4000, checkpoint="v3"),
    TransferRecipe("mesh_symmetric_raw_4000", powers=(0, 0), rows=4000, symmetric=True),
    TransferRecipe("mesh_symmetric_scaled_4000", rows=4000, symmetric=True),
    TransferRecipe("mesh_scaled_median_4000", rows=4000, median=True),
    TransferRecipe("mesh_local_re_24000", powers=(0, 0), rows=24000, local_re=True),
    TransferRecipe("guarded_mesh_raw_4000", powers=(0, 0), rows=4000, guarded=True),
    TransferRecipe("guarded_mesh_scaled_4000", rows=4000, guarded=True),
    TransferRecipe("guarded_mesh_v3_4000", rows=4000, checkpoint="v3", guarded=True),
)


def mesh_features(base: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Infer a horizontal-base triangle from the supplied full unlabeled mesh.

    Only x/z and file identity determine vertices. Future isolated query points
    need explicit vertices; their extrema must never be used to infer geometry.
    """
    result = base.copy()
    records = []
    for source, group in base.groupby("source_file", sort=True):
        points = group[["x", "z"]].drop_duplicates().to_numpy(dtype=float)
        z0, z1 = points[:, 1].min(), points[:, 1].max()
        bottom = points[np.isclose(points[:, 1], z0, rtol=0, atol=max(1e-5, (z1 - z0) * 1e-4))]
        left, right = float(bottom[:, 0].min()), float(bottom[:, 0].max())
        top = points[points[:, 1].argmax()]
        width, height = right - left, z1 - z0
        if width <= 0 or height <= 0:
            raise ValueError(f"Full triangular mesh required: {source}")
        apex = float((top[0] - left) / width)
        a = (group.z.to_numpy() - z0) / height
        c = (group.x.to_numpy() - left) / width - apex * a
        b = 1 - a - c
        bary = np.column_stack([a, b, c])
        if bary.min() < -0.005 or bary.max() > 1.005:
            raise ValueError(f"Mesh is not an aligned full triangle: {source}")
        side_ab = float(np.hypot(top[0] - left, height))
        side_ac = float(np.hypot(right - top[0], height))
        values = {
            "mesh_a": a, "mesh_b": b, "mesh_c": c,
            "mesh_width": width, "mesh_height": height,
            "mesh_apex": apex, "mesh_aspect": height / width,
            "mesh_ab": side_ab / width, "mesh_ac": side_ac / width,
            "mesh_x": (group.x.to_numpy() - left) / width,
            "mesh_z": a,
        }
        for name, value in values.items():
            result.loc[group.index, name] = value
        records.append({
            "source_file": source, "rows": len(group), "left": left, "right": right,
            "base_z": z0, "apex_x": float(top[0]), "apex_z": z1,
            "mesh_width": width, "mesh_height": height,
            "provided_bc": float(group.bc.iloc[0]), "provided_h": float(group.h.iloc[0]),
            "bary_min": float(bary.min()), "bary_max": float(bary.max()),
            "provided_bary_max_delta": float(np.max(np.abs(
                bary - group[["bary_a", "bary_b", "bary_c"]].to_numpy()
            ))),
        })
    return result, records


def guard_training(pool):
    """Quarantine ambiguous training conditions without identifying a correct copy."""
    flagged = set()
    for row in repeated_conditions(pool).to_dict("records"):
        flagged.update([(row["source_file"], row["re_a"]), (row["source_file"], row["re_b"])])
    for row in identical_components(pool).to_dict("records"):
        flagged.add((row["source_file"], row["re"]))
    mask = np.array([(source, float(reynolds)) in flagged for source, reynolds in zip(pool.source_file, pool.re)])
    accepted = pool.loc[~mask]
    if set(accepted.geometry_group) != set(pool.geometry_group):
        raise ValueError("Training guard would remove an entire geometry")
    report = [{"source_file": source, "re": reynolds,
               "removed_rows": int(((pool.source_file == source) & (pool.re == reynolds)).sum())}
              for source, reynolds in sorted(flagged)]
    return accepted, report


def features(data, recipe):
    if recipe.coordinates == "provided":
        return old_features(data, OLD_RECIPES[-1])
    if recipe.symmetric:
        data = data.copy()
        reflect = data.mesh_apex > 0.5
        b = data.mesh_b.copy()
        data.loc[reflect, "mesh_b"] = data.loc[reflect, "mesh_c"]
        data.loc[reflect, "mesh_c"] = b[reflect]
        ab = data.mesh_ab.copy()
        data.loc[reflect, "mesh_ab"] = data.loc[reflect, "mesh_ac"]
        data.loc[reflect, "mesh_ac"] = ab[reflect]
        data.loc[reflect, "mesh_x"] = 1 - data.loc[reflect, "mesh_x"]
        data.loc[reflect, "mesh_apex"] = 1 - data.loc[reflect, "mesh_apex"]
    result = data[[
        "mesh_width", "mesh_height", "mesh_aspect", "mesh_apex", "mesh_ab", "mesh_ac",
        "mesh_a", "mesh_b", "mesh_c", "mesh_x", "mesh_z", "re", "radius",
    ]].copy()
    result["radius_over_height"] = data.radius / data.mesh_height
    result["mesh_ab_product"] = data.mesh_a * data.mesh_b
    result["mesh_ac_product"] = data.mesh_a * data.mesh_c
    result["mesh_bc_product"] = data.mesh_b * data.mesh_c
    result["mesh_bubble"] = data.mesh_a * data.mesh_b * data.mesh_c
    result["nearest_edge"] = data[["mesh_a", "mesh_b", "mesh_c"]].min(axis=1)
    if recipe.symmetric:
        result["orientation"] = np.where(reflect, -1., 1.)
    return result


def factor(data, recipe, component):
    power = recipe.powers[component]
    scale = np.ones(len(data)) if power == 0 else (data.re.to_numpy() / 30) ** power * 10 / data.radius.to_numpy()
    if recipe.boundary:
        bubble = (data.mesh_a * data.mesh_b * data.mesh_c).to_numpy()
        scale = scale * np.maximum(27 * bubble, 0.05)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid target scaling")
    return scale


def predict_components(context, query, recipe, args, fitter=fit_predict):
    if set(context.geometry_group) & set(query.geometry_group):
        raise ValueError("Geometry leakage")
    if recipe.local_re:
        result = np.empty((len(query), 2))
        levels = np.sort(context.re.unique())
        for reynolds in sorted(query.re.unique()):
            nearest = levels[np.argsort(abs(levels - reynolds))]
            count = 1 if np.isclose(nearest[0], reynolds) else min(2, len(levels))
            local = context[context.re.isin(nearest[:count])]
            mask = query.re.to_numpy() == reynolds
            result[mask] = predict_components(local, query.loc[mask], replace(recipe, local_re=False), args, fitter)
        return result
    train_x, query_x = features(context, recipe), features(query, recipe)
    fit_args = SimpleNamespace(**vars(args))
    fit_args.model_path = args.model_v3 if recipe.checkpoint == "v3" else args.model_v25
    predictions = []
    for component, label in enumerate(("u0", "u90")):
        normalized = context[label].to_numpy() / factor(context, recipe, component)
        predictor = median_fit_predict if recipe.median and fitter is fit_predict else fitter
        prediction = predictor(train_x, normalized, query_x, recipe, fit_args)
        predictions.append(prediction * factor(query, recipe, component))
    result = np.column_stack(predictions)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions")
    return result


def median_fit_predict(train_x, y, query_x, recipe, args):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from tabpfn import TabPFNRegressor

    preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), MinMaxScaler())
    x = preprocessor.fit_transform(train_x)
    q = preprocessor.transform(query_x)
    model = TabPFNRegressor(model_path=args.model_path, device=args.device, n_estimators=recipe.estimators,
                           random_state=args.seed, ignore_pretraining_limits=True, show_progress_bar=False)
    try:
        model.fit(x, np.asarray(y, dtype=float))
        return np.concatenate([np.asarray(model.predict(q[start:start + args.chunk_size], output_type="median"))
                               for start in range(0, len(q), args.chunk_size)])
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def component_scores(query, prediction):
    rows = []
    for group, part in query.groupby("geometry_group"):
        mask = query.geometry_group.to_numpy() == group
        for component, angle in enumerate((0, 90)):
            rows.append({
                "geometry_group": group, "component": component,
                "r2": float(r2_score(part[f"observed_{angle:03d}"], prediction[mask, component])),
            })
    return rows


def choose_components(scores, groups, recipes):
    selected = []
    for component in (0, 1):
        means = {}
        for recipe in recipes:
            part = scores[(scores.recipe == recipe.name) & (scores.component == component)]
            if len(part) != len(groups) or set(part.geometry_group) != set(groups) or not np.isfinite(part.r2).all():
                raise ValueError("Incomplete/nonfinite inner geometry coverage")
            means[recipe.name] = float(part.r2.mean())
        selected.append(max(recipes, key=lambda recipe: means[recipe.name]))
    return selected


def save_metrics(predictions, out):
    summaries, geometry, angles = [], [], []
    for method, data in predictions.groupby("method"):
        for angle in (None, 0, 90):
            part = data if angle is None else data[data.rotation_deg == angle]
            scores = []
            for group, chunk in part.groupby("geometry_group"):
                row = {"method": method, "geometry_group": group,
                       "scope": "all_24_angles" if angle is None else f"angle_{angle:03d}",
                       **metric_row(chunk.target.to_numpy(), chunk.prediction.to_numpy())}
                geometry.append(row)
                scores.append(row["r2"])
            summaries.append({"method": method, "scope": row["scope"],
                              "geometry_mean_r2": float(np.mean(scores)),
                              "geometry_worst_r2": float(np.min(scores)), "rows": len(part),
                              **metric_row(part.target.to_numpy(), part.prediction.to_numpy())})
        for (group, angle), part in data.groupby(["geometry_group", "rotation_deg"]):
            angles.append({"method": method, "geometry_group": group, "rotation_deg": angle,
                           **metric_row(part.target.to_numpy(), part.prediction.to_numpy())})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "reference_summary.csv", index=False)
    pd.DataFrame(geometry).to_csv(out / "geometry_metrics.csv", index=False)
    pd.DataFrame(angles).to_csv(out / "angle_metrics.csv", index=False)
    return summary


def evaluate(base, test, args, out, recipes=RECIPES, predictor=predict_components):
    splits = geometry_folds(base.geometry_group, 5, args.seed)
    scores, manifests, selections, timings = [], [], [], []
    # Finish all inner selection before computing any new outer score.
    for fold, (train_groups, held_groups) in enumerate(splits, 1):
        candidates = []
        for inner, (fit_groups, val_groups) in enumerate(
            geometry_folds(train_groups, 3, stable_seed(str(fold), args.seed)), 1
        ):
            train = base[base.geometry_group.isin(fit_groups)]
            query = test[test.geometry_group.isin(val_groups)]
            guarded, guard_report = guard_training(train) if any(r.guarded for r in recipes) else (train, [])
            contexts = {(n, use_guard): sample_context(guarded if use_guard else train, n, args.seed)
                        for n, use_guard in sorted({(r.rows, r.guarded) for r in recipes})}
            for (n, use_guard), context in contexts.items():
                manifests.append({"outer_fold": fold, "inner_fold": inner, "budget": n,
                                  "guarded": use_guard, "quarantine": guard_report if use_guard else [],
                                  "fit_groups": sorted(fit_groups), "validation_groups": sorted(val_groups),
                                  "outer_test_groups": sorted(held_groups), "base_ids": context.base_id.tolist()})
            for recipe in recipes:
                start = time.perf_counter()
                prediction = predictor(contexts[(recipe.rows, recipe.guarded)], query, recipe, args)
                rows = [{"outer_fold": fold, "inner_fold": inner, "recipe": recipe.name, **row}
                        for row in component_scores(query, prediction)]
                candidates.extend(rows)
                timings.append({"outer": fold, "inner": inner, "recipe": recipe.name,
                                "seconds": time.perf_counter() - start})
                print(f"inner outer={fold} split={inner} {recipe.name}: "
                      f"R2={np.mean([r['r2'] for r in rows]):.4f}", flush=True)
                pd.DataFrame([*scores, *candidates]).to_csv(out / "inner_metrics.csv", index=False)
        chosen = choose_components(pd.DataFrame(candidates), train_groups, recipes)
        selections.append({"fold": fold, "u0": chosen[0].name, "u90": chosen[1].name,
                           "test_groups": sorted(held_groups)})
        scores.extend(candidates)
        write_json(out / "selections.json", selections)
        write_json(out / "splits.json", manifests)
        write_json(out / "timings.json", timings)
    pd.DataFrame(scores).groupby(["recipe", "component"]).r2.agg(["mean", "min", "count"]).to_csv(
        out / "inner_summary.csv"
    )
    if args.inner_only:
        return pd.DataFrame(scores)
    outputs = []
    for fold, (train_groups, held_groups) in enumerate(splits, 1):
        selection = selections[fold - 1]
        chosen = [next(r for r in recipes if r.name == selection[label]) for label in ("u0", "u90")]
        query = test[test.geometry_group.isin(held_groups)]
        pool = base[base.geometry_group.isin(train_groups)]
        guarded, guard_report = guard_training(pool) if any(r.guarded for r in [recipes[0], *chosen]) else (pool, [])
        predictions = {}
        for recipe in dict.fromkeys([recipes[0], *chosen]):
            context = sample_context(guarded if recipe.guarded else pool, recipe.rows, args.seed)
            manifests.append({"outer_fold": fold, "inner_fold": None, "budget": recipe.rows,
                              "fit_groups": sorted(train_groups), "outer_test_groups": sorted(held_groups),
                              "recipe": recipe.name, "base_ids": context.base_id.tolist()})
            manifests[-1]["quarantine"] = guard_report if recipe.guarded else []
            start = time.perf_counter()
            predictions[recipe.name] = predictor(context, query, recipe, args)
            timings.append({"outer": fold, "inner": None, "recipe": recipe.name,
                            "seconds": time.perf_counter() - start})
        selected = np.column_stack([predictions[chosen[i].name][:, i] for i in (0, 1)])
        for method, components in (("provided_components_baseline", predictions[recipes[0].name]),
                                   ("nested_transfer_tabpfn", selected)):
            theta = np.deg2rad(ANGLES)
            values = (components[:, 0, None] * np.cos(theta) + components[:, 1, None] * np.sin(theta)).ravel()
            result = queries(query, ANGLES)[[
                "base_id", "source_file", "source_row", "geometry_group", "radius", "re", "x", "z", "rotation_deg"
            ]].copy()
            result["target"], result["prediction"] = observed(query, ANGLES), values
            result["method"], result["fold"] = method, fold
            result["recipe_u0"] = chosen[0].name if method == "nested_transfer_tabpfn" else recipes[0].name
            result["recipe_u90"] = chosen[1].name if method == "nested_transfer_tabpfn" else recipes[0].name
            outputs.append(result)
            print(f"OUTER fold={fold} {method} pooled R2={r2_score(result.target, values):.4f}", flush=True)
        pd.concat(outputs).to_csv(out / "predictions.csv", index=False)
        write_json(out / "splits.json", manifests)
        write_json(out / "timings.json", timings)
    return save_metrics(pd.concat(outputs, ignore_index=True), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_transfer/2026-09-29"))
    parser.add_argument("--model-v25", type=Path, default=Path("models/models/tabpfn-v2.5-regressor-v2.5_default.ckpt"))
    parser.add_argument("--model-v3", type=Path, default=Path("models/tabpfn_3/tabpfn-v3-regressor-v3_default.ckpt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-only", action="store_true")
    parser.add_argument("--recipes", nargs="+", choices=[r.name for r in RECIPES])
    args = parser.parse_args()
    recipes = tuple(r for r in RECIPES if args.recipes is None or r.name in args.recipes)
    if args.chunk_size <= 0:
        parser.error("Positive chunk size required")
    for version in {r.checkpoint for r in recipes}:
        if not getattr(args, "model_" + version).is_file():
            parser.error(f"Missing checkpoint {version}")
    with experiment_run(args.output_dir, "fd-transfer", extra_sources=[
        __file__, Path(__file__).with_name("tune_rotated_fd.py"),
        Path(__file__).with_name("rotation_dataset.py"), Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("transfer_diagnostics.py"),
    ]) as out:
        base, test, snapshot = load_snapshot(args.snapshot)
        enriched, records = mesh_features(base)
        extra = [c for c in enriched if c.startswith("mesh_")]
        test = test.merge(enriched[["base_id", *extra]], on="base_id", validate="one_to_one")
        pd.DataFrame(records).to_csv(out / "mesh_audit.csv", index=False)
        write_json(out / "config.json", {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "recipes": [asdict(r) for r in recipes], "snapshot": snapshot,
            "checkpoint_hashes": {v: file_sha256(getattr(args, "model_" + v)) for v in {r.checkpoint for r in recipes}},
            "selection": "Per component, inner geometry-equal R2; all inner searches precede outer evaluation",
            "mesh_contract": "Full unlabeled input meshes determine vertices, including held-out geometry meshes",
            "target_policy": "No target repair or test exclusion; guarded recipes quarantine ambiguous training conditions only",
            "evaluation_status": "Iterative research on previously evaluated geometries, not an untouched blind test",
        })
        print(evaluate(enriched, test, args, out, recipes).to_string(index=False), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
