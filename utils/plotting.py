"""Plotting helpers for FL heatmaps and diagnostics."""

from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def load_fl_excel(path: Path, suffix: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_excel(path, engine="openpyxl")
    df = df[["X", "Z", "FL"]].dropna()
    return df, path.name.removesuffix(suffix)


def pivot_fl(df: pd.DataFrame) -> pd.DataFrame:
    pt = df.pivot_table(index="Z", columns="X", values="FL", aggfunc="mean")
    return pt.sort_index().sort_index(axis=1)


def symmetric_norm(values: np.ndarray) -> mcolors.TwoSlopeNorm:
    vmax = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def gradient_summary(matrix: np.ndarray) -> tuple[float, float]:
    grad_x = np.gradient(matrix, axis=1)
    grad_z = np.gradient(matrix, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_z**2)
    return float(np.nanmax(grad_mag)), float(np.nanmean(grad_mag))
