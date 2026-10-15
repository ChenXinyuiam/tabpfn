"""Combine compatible inner-only recipe scores and evaluate fixed ensembles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.regression.rotation_dataset import ANGLES, load_snapshot
from src.regression.transfer_fd import TransferRecipe, guard_training, mesh_features, predict_components, save_metrics
from src.regression.tune_rotated_fd import geometry_folds, observed, queries, sample_context
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run


def inner_weights(scores, groups, recipes, component, top_k=3, temperature=.1):
    ranking = []
    for recipe in recipes:
        part = scores[(scores.recipe == recipe.name) & (scores.component == component)]
        if len(part) != len(groups) or set(part.geometry_group) != groups or not np.isfinite(part.r2).all():
            raise ValueError("Incomplete inner selection evidence")
        ranking.append((recipe.name, float(part.r2.mean())))
    ranking.sort(key=lambda item: item[1], reverse=True)
    ranking = ranking[:top_k]
    values = np.asarray([score for _, score in ranking])
    weights = np.exp((values - values.max()) / temperature)
    weights /= weights.sum()
    return [{"recipe": name, "inner_r2": score, "weight": float(weight)}
            for (name, score), weight in zip(ranking, weights, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--test-snapshot", type=Path, help="Optional denser test snapshot of the identical base dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_transfer_ensemble/2026-09-29"))
    args = parser.parse_args()
    configurations = [json.loads((run / "config.json").read_text(encoding="utf-8")) for run in args.runs]
    origin = configurations[0]
    recipes, inner = {}, []
    for run, config in zip(args.runs, configurations, strict=True):
        status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError("Only completed studies can supply inner scores")
        if config["snapshot"] != origin["snapshot"] or config["seed"] != origin["seed"]:
            raise ValueError("Incompatible datasets or split seeds")
        for key in set(config["checkpoint_hashes"]) & set(origin["checkpoint_hashes"]):
            if config["checkpoint_hashes"][key] != origin["checkpoint_hashes"][key]:
                raise ValueError("Incompatible checkpoint hashes")
        frame = pd.read_csv(run / "inner_metrics.csv")
        for values in config["recipes"]:
            values["powers"] = tuple(values["powers"])
            recipe = TransferRecipe(**values)
            if recipe.name in recipes:
                if recipes[recipe.name] != recipe:
                    raise ValueError("Same name used for different recipe settings")
                continue
            recipes[recipe.name] = recipe
            inner.append(frame[frame.recipe == recipe.name])
    scores = pd.concat(inner, ignore_index=True)
    model_args = SimpleNamespace(model_v25=Path(origin["model_v25"]), model_v3=Path(origin["model_v3"]),
                                 seed=origin["seed"], device=origin["device"], chunk_size=origin["chunk_size"])
    for version in {r.checkpoint for r in recipes.values()}:
        expected = next(c["checkpoint_hashes"][version] for c in configurations if version in c["checkpoint_hashes"])
        if file_sha256(getattr(model_args, "model_" + version)) != expected:
            raise ValueError("Checkpoint changed after inner evaluation")
    with experiment_run(args.output_dir, "fd-transfer-ensemble", extra_sources=[
        __file__, Path(__file__).with_name("transfer_fd.py"), Path(__file__).with_name("tune_rotated_fd.py"),
        Path(__file__).with_name("transfer_diagnostics.py"), Path(__file__).with_name("rotation_dataset.py"),
        Path(__file__).with_name("evaluation.py"),
    ]) as out:
        base, test, snapshot = load_snapshot(args.snapshot)
        if snapshot != origin["snapshot"]:
            raise ValueError("Snapshot changed after inner evaluation")
        evaluation_snapshot = snapshot
        if args.test_snapshot is not None:
            reference_base, test, evaluation_snapshot = load_snapshot(args.test_snapshot)
            pd.testing.assert_frame_equal(base, reference_base, rtol=1e-12, atol=1e-12)
        base, audit = mesh_features(base)
        extra = [c for c in base if c.startswith("mesh_")]
        test = test.merge(base[["base_id", *extra]], on="base_id", validate="one_to_one")
        pd.DataFrame(audit).to_csv(out / "mesh_audit.csv", index=False)
        splits = geometry_folds(base.geometry_group, 5, model_args.seed)
        selections = []
        for fold, (train_groups, test_groups) in enumerate(splits, 1):
            rows = scores[scores.outer_fold == fold]
            selections.append({"fold": fold, "test_groups": sorted(test_groups),
                               "components": [inner_weights(rows, train_groups, list(recipes.values()), c)
                                              for c in (0, 1)]})
        write_json(out / "selections.json", selections)
        write_json(out / "config.json", {
            "input_runs": [str(p.resolve()) for p in args.runs], "snapshot": evaluation_snapshot,
            "inner_snapshot": origin["snapshot"],
            "test_snapshot_dir": str((args.test_snapshot or args.snapshot).resolve()),
            "recipes": [asdict(r) for r in recipes.values()], "seed": model_args.seed,
            "top_k": 3, "temperature": .1,
            "policy": "Fixed top-3 softmax weights from inner geometry-mean native R2, independently per component",
            "evaluation_status": "Exploratory repeated-geometry evaluation; no fresh blind test",
        })
        scores.to_csv(out / "inner_metrics.csv", index=False)
        outputs, manifests = [], []
        for fold, (train_groups, test_groups) in enumerate(splits, 1):
            selected = selections[fold - 1]["components"]
            required = sorted({row["recipe"] for component in selected for row in component})
            pool = base[base.geometry_group.isin(train_groups)]
            guarded, report = guard_training(pool) if any(recipes[name].guarded for name in required) else (pool, [])
            query = test[test.geometry_group.isin(test_groups)]
            predictions = {}
            for name in required:
                recipe = recipes[name]
                context = sample_context(guarded if recipe.guarded else pool, recipe.rows, model_args.seed)
                predictions[name] = predict_components(context, query, recipe, model_args)
                manifests.append({"fold": fold, "recipe": name, "fit_groups": sorted(train_groups),
                                  "test_groups": sorted(test_groups), "base_ids": context.base_id.tolist()})
                manifests[-1]["quarantine"] = report if recipe.guarded else []
                print(f"outer={fold} evaluated {name}", flush=True)
            best = np.column_stack([predictions[selected[c][0]["recipe"]][:, c] for c in (0, 1)])
            blend = np.column_stack([sum(row["weight"] * predictions[row["recipe"]][:, c] for row in selected[c])
                                     for c in (0, 1)])
            for method, components in (("nested_union_best", best), ("nested_union_blend", blend)):
                theta = np.deg2rad(ANGLES)
                values = (components[:, 0, None] * np.cos(theta) + components[:, 1, None] * np.sin(theta)).ravel()
                result = queries(query, ANGLES)[[
                    "base_id", "source_file", "source_row", "geometry_group", "radius", "re", "x", "z", "rotation_deg"
                ]].copy()
                result["target"], result["prediction"] = observed(query, ANGLES), values
                result["method"], result["fold"] = method, fold
                outputs.append(result)
            pd.concat(outputs).to_csv(out / "predictions.csv", index=False)
            write_json(out / "splits.json", manifests)
        print(save_metrics(pd.concat(outputs), out).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
