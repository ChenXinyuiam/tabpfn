"""Tune FL TabPFN settings using the shared config and feature lists."""

from pathlib import Path
import itertools

import pandas as pd

from src.config import FL_CONFIG, MODEL_DIR, TRIANGLE_GROUPS_ROOT
from src.train import run_experiment
from utils import setup_offline_tabpfn
from utils.features import ENGINEERED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS


TRANSFORMS = ["identity", "log1p", "sqrt"]

FEATURE_SETS = {
    "raw": RAW_FEATURE_COLUMNS,
    "engineered": RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS,
}

MAX_TIMES = [10, 30, 60]


def main() -> None:
    setup_offline_tabpfn(MODEL_DIR)

    results = []
    for exp_id, (transform, feat_key, max_time) in enumerate(
        itertools.product(TRANSFORMS, FEATURE_SETS.keys(), MAX_TIMES),
        start=1,
    ):
        print("\n" + "=" * 50)
        print(f"EXP {exp_id}")
        print(f"transform={transform}, features={feat_key}, max_time={max_time}")
        print("=" * 50)

        config = dict(
            FL_CONFIG,
            transform=transform,
            features=FEATURE_SETS[feat_key],
            data_root=TRIANGLE_GROUPS_ROOT,
            max_time=max_time,
            add_features=feat_key == "engineered",
        )

        try:
            r2 = run_experiment(config)
        except Exception as exc:
            print(f"failed: {exc}")
            r2 = None

        results.append(
            {
                "transform": transform,
                "feature_set": feat_key,
                "max_time": max_time,
                "r2": r2,
            }
        )

    df = pd.DataFrame(results).sort_values(by="r2", ascending=False)
    output_path = Path("tune_fl_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nTuning complete. Results saved:")
    print(output_path)
    print("\nTop results:")
    print(df.head(10))


if __name__ == "__main__":
    main()
