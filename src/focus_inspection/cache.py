"""Exact-content inference cache for resumable local grouped evaluations."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from utils.provenance import file_sha256, write_json

from .modeling import predict_probability


class PredictionCache:
    def __init__(self, directory, checkpoint, predictor=predict_probability, *, read_through=()):
        self.directory = directory
        self.predictor = predictor
        self.read_through = tuple(Path(path) for path in read_through)
        self.identity = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "modeling_sha256": file_sha256(Path(__file__).with_name("modeling.py")),
            "versions": {name: version(name) for name in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")},
        }
        self.hits = 0
        self.misses = 0

    def __call__(self, train, test, **kwargs):
        columns = kwargs["feature_columns"]
        signature = {**self.identity, **{key: str(value) for key, value in kwargs.items()}}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
        for frame in (train[[*columns, "focus_label"]], test[columns]):
            digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
            digest.update(str(frame.shape).encode())
        key = digest.hexdigest()
        destination = self.directory / f"{key}.json"
        for directory in (self.directory, *self.read_through):
            path = directory / f"{key}.json"
            if not path.is_file():
                continue
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                values = np.asarray(saved["probabilities"], dtype=float)
                if (
                    saved["key"] == key
                    and values.shape == (len(test),)
                    and np.isfinite(values).all()
                    and ((values >= 0) & (values <= 1)).all()
                ):
                    self.hits += 1
                    if path != destination:
                        write_json(destination, saved)
                    return values
            except (ValueError, KeyError, TypeError):
                pass
        values = np.asarray(self.predictor(train, test, **kwargs), dtype=float)
        if values.shape != (len(test),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid predictor output; cache was not written")
        write_json(destination, {"key": key, "probabilities": values.tolist()})
        self.misses += 1
        return values
