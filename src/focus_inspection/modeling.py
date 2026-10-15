"""TabPFN inference with fold-local preprocessing and explicit checkpoints."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for _key in ("POSTHOG_DISABLED", "DO_NOT_TRACK", "TABPFN_DISABLE_TELEMETRY"):
    os.environ.setdefault(_key, "1")


def resolve_device(requested: str) -> str:
    if requested != "prefer-cuda":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def predict_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_path: Path,
    device: str,
    random_state: int,
    n_estimators: int = 1,
) -> np.ndarray:
    from tabpfn import TabPFNClassifier

    if not model_path.is_file():
        raise FileNotFoundError(f"TabPFN classifier checkpoint not found: {model_path}")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if set(train["focus_label"].unique()) != {0, 1}:
        raise ValueError("Training samples must contain both F and UF")
    preprocessing = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    x_train = preprocessing.fit_transform(train[feature_columns].replace([np.inf, -np.inf], np.nan))
    x_test = preprocessing.transform(test[feature_columns].replace([np.inf, -np.inf], np.nan))
    model = TabPFNClassifier(
        n_estimators=n_estimators,
        model_path=model_path,
        device=device,
        ignore_pretraining_limits=True,
        random_state=random_state,
        show_progress_bar=False,
    )
    model.fit(x_train, train["focus_label"].to_numpy(dtype=int))
    probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise RuntimeError("Invalid focus probabilities returned by TabPFN")
    return probability


def grouped_tabpfn_oof(data, folds, *, feature_columns, model_path, device, random_state, n_estimators=1):
    probability = np.full(len(data), np.nan)
    coverage = np.zeros(len(data), dtype=int)
    for fold_id, (train_index, test_index) in enumerate(folds, start=1):
        probability[test_index] = predict_probability(
            data.iloc[train_index],
            data.iloc[test_index],
            feature_columns=feature_columns,
            model_path=model_path,
            device=device,
            random_state=random_state + fold_id,
            n_estimators=n_estimators,
        )
        coverage[test_index] += 1
    if not (coverage == 1).all() or not np.isfinite(probability).all():
        raise RuntimeError("Each sample must receive exactly one out-of-fold prediction")
    return (probability > 0.5).astype(int), probability
