"""Shared helper package for the TabPFN project."""

from .features import (
    ENGINEERED_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    add_engineered_features,
    available_feature_columns,
)
from .folds import discover_leaves, load_fold_df
from .metrics import mape, rmse, smape
from .paths import (
    batch_dirs,
    ensure_dir,
    iter_excel_files,
    iter_excel_files_recursive,
    project_root,
    select_batch_dir,
)
from .tabpfn import setup_offline_tabpfn
from .target import inverse_y, transform_y
from .validation import check_columns

__all__ = [
    "ENGINEERED_FEATURE_COLUMNS",
    "RAW_FEATURE_COLUMNS",
    "add_engineered_features",
    "available_feature_columns",
    "batch_dirs",
    "check_columns",
    "discover_leaves",
    "ensure_dir",
    "inverse_y",
    "iter_excel_files",
    "iter_excel_files_recursive",
    "load_fold_df",
    "mape",
    "project_root",
    "rmse",
    "setup_offline_tabpfn",
    "select_batch_dir",
    "smape",
    "transform_y",
]
