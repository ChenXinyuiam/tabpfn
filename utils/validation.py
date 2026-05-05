"""Validation helpers."""

from pathlib import Path

import pandas as pd


def check_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def require_path(path: Path, label: str = "Path") -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
