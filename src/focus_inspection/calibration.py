"""Serializable, regularized sigmoid recalibration of binary probabilities."""

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

CLIP = 1e-6


def checked_probabilities(probability):
    values = np.asarray(probability, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected a finite one-dimensional probability array in [0, 1]")
    return values


def fit_sigmoid(probability, labels):
    probability = checked_probabilities(probability)
    labels = np.asarray(labels)
    if len(probability) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires matched OOF probabilities and both classes")
    x = logit(np.clip(probability, CLIP, 1 - CLIP)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(x, labels)
    return {
        "method": "logit_sigmoid",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "clip": CLIP,
        "regularization_C": 1.0,
    }


def apply_calibration(probability, state):
    values = checked_probabilities(probability)
    if state is None:
        return values.copy()
    if state.get("method") != "logit_sigmoid":
        raise ValueError("Unsupported probability calibration method")
    coefficient, intercept, clip = float(state["coefficient"]), float(state["intercept"]), float(state["clip"])
    if not np.isfinite([coefficient, intercept, clip]).all() or not 0 < clip < 0.5:
        raise ValueError("Invalid calibration coefficients or clipping bound")
    return expit(coefficient * logit(np.clip(values, clip, 1 - clip)) + intercept)
