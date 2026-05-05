"""Regression metric helpers."""

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-12, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true) + np.abs(y_pred), 1e-12, None)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_range(y_true: np.ndarray, y_pred: np.ndarray):
    pass

def rmse_range():
    pass

def nrmse_std():
    pass

def mae_std():
    pass

def sign_accuracy():
    """should set eps for value close to 0"""
    pass

def max_abs_error():
    pass

def p95_abs_error():
    pass