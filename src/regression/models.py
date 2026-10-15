"""Common, fold-local estimators for system regression and focus classification."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.focus_inspection.modeling import resolve_device


@dataclass(frozen=True)
class ModelSpec:
    backend: str = "extra-trees"
    seed: int = 42
    n_estimators: int = 200
    checkpoint: str | None = None
    device: str = "cpu"

    def __post_init__(self):
        if self.backend not in {"extra-trees", "tabpfn"} or type(self.n_estimators) is not int or self.n_estimators < 1:
            raise ValueError("Expected extra-trees/tabpfn backend and positive integer n_estimators")
        if self.backend == "tabpfn" and (not self.checkpoint or not Path(self.checkpoint).is_file()):
            raise ValueError("TabPFN requires an existing explicit checkpoint")

    def to_dict(self):
        return asdict(self)


class FittedModel:
    def __init__(self, spec, features, *, classification=False):
        self.spec, self.features, self.classification = spec, list(features), classification
        self.preprocessing = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        if spec.backend == "extra-trees":
            cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
            self.estimator = cls(n_estimators=spec.n_estimators, random_state=spec.seed, n_jobs=-1)
        else:
            from tabpfn import TabPFNClassifier, TabPFNRegressor

            cls = TabPFNClassifier if classification else TabPFNRegressor
            self.estimator = cls(
                model_path=Path(spec.checkpoint),
                device=resolve_device(spec.device),
                n_estimators=spec.n_estimators,
                random_state=spec.seed,
                ignore_pretraining_limits=True,
                show_progress_bar=False,
            )

    def fit(self, frame, target):
        values = frame[self.features].to_numpy(dtype=float)
        if not len(frame) or np.isinf(values).any():
            raise ValueError("Empty training data or infinite features")
        y = np.asarray(target)
        if len(y) != len(frame) or (not self.classification and not np.isfinite(y.astype(float)).all()):
            raise ValueError("Invalid training targets")
        if self.classification and len(set(y)) < 2:
            raise ValueError("Classification training requires at least two classes")
        self.estimator.fit(self.preprocessing.fit_transform(values), y)
        return self

    def predict(self, frame, chunk_size=2048):
        return self._predict(frame, chunk_size, False)

    def probabilities(self, frame, chunk_size=2048):
        return self._predict(frame, chunk_size, True)

    def _predict(self, frame, chunk_size, probability):
        if chunk_size < 1 or frame.empty:
            raise ValueError("Nonempty query and positive chunk size required")
        results = []
        for start in range(0, len(frame), chunk_size):
            x = self.preprocessing.transform(
                frame.iloc[start : start + chunk_size][self.features].to_numpy(dtype=float)
            )
            prediction = self.estimator.predict_proba(x) if probability else self.estimator.predict(x)
            results.append(np.asarray(prediction))
        result = np.concatenate(results)
        if probability or not self.classification:
            if not np.isfinite(result).all():
                raise RuntimeError("Estimator returned nonfinite predictions")
        if probability and (not np.allclose(result.sum(axis=1), 1) or ((result < 0) | (result > 1)).any()):
            raise RuntimeError("Invalid class probabilities")
        return result
