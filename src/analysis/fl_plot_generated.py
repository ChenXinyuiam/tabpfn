"""Create FL heatmaps and continuity summaries for generated data."""

import argparse
import sys
from pathlib import Path

from src.analysis.fl_plot import main as _plot_main
from src.config import PUBLISHED_PROCESSED_ROOT
from utils.paths import select_batch_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot *_generated.xlsx files for one processed batch.")
    parser.add_argument("--batch", default=None, help="Batch under data/processed. Defaults to the latest batch.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Explicit processed data directory.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or select_batch_dir(PUBLISHED_PROCESSED_ROOT, args.batch)
    out_dir = args.out_dir or (data_dir / "figures" / "generated")

    sys.argv = [
        "fl_plot",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--pattern",
        "*_generated.xlsx",
        "--suffix",
        "_generated.xlsx",
        "--summary-name",
        "ALL_generated_FL_summary.png",
        "--output-label-suffix",
        "_generated_FL",
    ]
    _plot_main()


if __name__ == "__main__":
    main()
