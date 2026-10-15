"""TabPFN-only nested FD-u tuning with a predeclared candidate menu."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.regression.evaluation import FEATURE_COLUMNS
from src.regression.optimize_fd import (
    Recipe,
    evaluate,
    load_native_fd,
    report,
)
from src.regression.optimize_fd import (
    features as relative_features,
)
from src.regression.optimize_fd import (
    predict as original_predict,
)
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run


@dataclass(frozen=True)
class TabRecipe:
    name: str
    estimators: int = 4
    numerical: bool = True
    augment: bool = False
    re_power: int = 0
    radius_power: int = 0


RECIPES = (
    TabRecipe("raw_tabpfn", 1, False),
    TabRecipe("raw_e4", 4, False),
    TabRecipe("numeric_raw_e4"),
    TabRecipe("numeric_augmented_e4", augment=True),
    TabRecipe("numeric_re1_e4", re_power=1),
    TabRecipe("numeric_re2_e4", re_power=2),
    TabRecipe("numeric_re2_radius_e4", re_power=2, radius_power=1),
    TabRecipe("numeric_augmented_re2_e4", augment=True, re_power=2),
)


def features(data: pd.DataFrame, recipe: TabRecipe) -> pd.DataFrame:
    raw = data[FEATURE_COLUMNS].copy()
    if recipe.augment:
        relative = relative_features(data, Recipe("relative", "tabpfn", True))
        for name in relative.columns:
            if name not in raw.columns:
                raw[name] = relative[name]
    return raw


def factor(data: pd.DataFrame, recipe: TabRecipe) -> np.ndarray:
    if (data.re <= 0).any() or (data.radius <= 0).any():
        raise ValueError("Positive Re and radius labels required")
    return (data.re.to_numpy() / 30) ** recipe.re_power * (20 / data.radius.to_numpy()) ** recipe.radius_power


def inference_config(recipe: TabRecipe) -> dict:
    # An empty categorical_features_indices list alone DOES NOT disable inference.
    return {"MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 1} if recipe.numerical else {}


def predict(train: pd.DataFrame, test: pd.DataFrame, recipe: TabRecipe, args) -> np.ndarray:
    if set(train.geometry_group) & set(test.geometry_group):
        raise ValueError("Geometry leakage")
    if recipe == RECIPES[0]:
        return original_predict(train, test, Recipe("raw_tabpfn", "tabpfn"), args)
    from tabpfn import TabPFNRegressor

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(features(train, recipe))
    x_test = scaler.transform(features(test, recipe))
    y = train.target.to_numpy() / factor(train, recipe)
    model = TabPFNRegressor(
        model_path=args.model_path,
        device=args.device,
        n_estimators=recipe.estimators,
        inference_config=inference_config(recipe),
        random_state=args.seed,
        ignore_pretraining_limits=True,
        show_progress_bar=False,
    )
    try:
        model.fit(x_train, y)
        predicted = np.concatenate(
            [
                np.asarray(model.predict(x_test[start : start + args.chunk_size])).reshape(-1)
                for start in range(0, len(test), args.chunk_size)
            ]
        ) * factor(test, recipe)
        if not np.isfinite(predicted).all():
            raise ValueError("Non-finite TabPFN predictions")
        return predicted
    finally:
        del model
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/rawdata/2026-06-20"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_tabpfn_tuning"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/models/tabpfn-v2.5-regressor-v2.5_default.ckpt")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-rows", type=int, default=3000)
    parser.add_argument("--validation-rows", type=int, default=200)
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("max_train_rows", "validation_rows", "test_rows", "chunk_size"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not args.model_path.is_file():
        parser.error("Local checkpoint required")
    sources = [__file__, Path(__file__).with_name("optimize_fd.py"), Path(__file__).with_name("evaluation.py")]
    with experiment_run(args.output_dir, "tabpfn-fd-nested-tuning", extra_sources=sources) as out:
        write_json(
            out / "config.json",
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "recipes": [asdict(r) for r in RECIPES],
                "selection": "mean_inner_geometry_r2",
                "checkpoint_sha256": file_sha256(args.model_path),
                "scope": "native FD-u research, no production release; repeated-use geometries, not prospective test",
            },
        )
        data, inventory = load_native_fd(args.data_root / "FD")
        if args.max_train_rows < data.source_file.nunique():
            raise ValueError("Training budget must cover source files")
        pd.DataFrame(inventory).to_csv(out / "data_inventory.csv", index=False)
        timings = []

        def measured_predict(train, test, recipe, settings):
            started = time.perf_counter()
            prediction = predict(train, test, recipe, settings)
            timings.append(
                {
                    "recipe": recipe.name,
                    "train_groups": sorted(train.geometry_group.unique()),
                    "test_groups": sorted(test.geometry_group.unique()),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "seconds": time.perf_counter() - started,
                }
            )
            write_json(out / "timings.json", timings)
            return prediction

        folds, summary = evaluate(data, args, out, RECIPES, measured_predict)
        report(out, folds, summary)
        # The shared benchmark report contains first-round-specific descriptions.
        (out / "report.md").write_text(
            "# TabPFN-only FD-u tuning\n\n"
            "Four outer geometry folds; each recipe selected only by three inner geometry folds.\n"
            "Same 3000-row default context budget and test positions for every candidate.\n"
            "Eight frozen candidates; details in config.json. No ExtraTrees substitution.\n"
            "Augmented recipes retain all raw features. Target factors are empirical, not physical laws.\n"
            "Repeated-use geometries: research estimate, not a prospective independent test.\n"
            "Native target semantics/data issues remain unresolved; no production defaults changed.\n\n"
            "```text\n" + summary.to_string(index=False) + "\n```\n\n"
            "```text\n" + folds.to_string(index=False) + "\n```\n",
            encoding="utf-8",
        )
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
