"""Target transformations for regression experiments."""

import numpy as np


def transform_y(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "log1p":
        return np.log1p(np.clip(y, 1e-15, None))
    if mode == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    if mode == "identity":
        return y
    raise ValueError(f"Unknown target transform: {mode}")


def inverse_y(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "log1p":
        return np.expm1(y)
    if mode == "sqrt":
        return y**2
    if mode == "identity":
        return y
    raise ValueError(f"Unknown target transform: {mode}")
