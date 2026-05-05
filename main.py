"""
TabPFN pipeline entry point.

Data pipeline (run in order):
    python -m src.data.clean --batch 2026-04-26
    python -m src.data.Feature_scale --batch 2026-04-26
    python -m src.data.calculate --batch 2026-04-26
    python -m src.data.Reverse_FL --batch 2026-04-26

If --batch is omitted, the latest batch directory is used.

Training:
    python -m src.train --target FD
    python -m src.train --target FL
    python -m src.train --target both

Inference / evaluation:
    python -m src.inference --mode focus
    python -m src.inference --mode fd
    python -m src.inference --mode fl

EDA:
    python -m src.data.explore --dir data/processedv2 --verbose
"""


def main():
    print(__doc__)


if __name__ == "__main__":
    main()
