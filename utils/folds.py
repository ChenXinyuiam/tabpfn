"""Cross-validation fold discovery and loading."""

from pathlib import Path

import pandas as pd

from .features import add_engineered_features


def discover_leaves(root: Path, n_splits: int) -> list[dict]:
    """Collect root/<shape>/<R>/<Re>/fold{k}.csv leaves."""
    leaves = []
    if not root.exists():
        return leaves

    for shape_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for radius_dir in sorted(path for path in shape_dir.iterdir() if path.is_dir()):
            for re_dir in sorted(path for path in radius_dir.iterdir() if path.is_dir()):
                fold_paths = {
                    k: re_dir / f"fold{k}.csv"
                    for k in range(1, n_splits + 1)
                    if (re_dir / f"fold{k}.csv").is_file()
                }
                if fold_paths:
                    leaves.append(
                        {
                            "shape": shape_dir.name,
                            "R": radius_dir.name,
                            "Re": re_dir.name,
                            "fold_paths": fold_paths,
                        }
                    )
    return leaves


def load_fold_df(path: Path, bucket: str, add_features: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__bucket__"] = bucket
    if add_features:
        df = add_engineered_features(df)
    return df
