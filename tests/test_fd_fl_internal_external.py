import unittest
from pathlib import Path

import pandas as pd

from src.regression.evaluation import (
    geometry_signature,
    internal_split,
    parse_radius,
)


class InternalExternalEvaluationTests(unittest.TestCase):
    def test_parse_radius_from_fl_and_fd_names(self) -> None:
        self.assertEqual(parse_radius(Path("750-530-反-20-FL.xlsx")), 20)
        self.assertEqual(parse_radius(Path("750-320-30-FD-结果.xlsx")), 30)

    def test_geometry_signature_ignores_reflection_order(self) -> None:
        left = pd.DataFrame({"bc": [750.0], "ab": [400.0], "ac": [850.0]})
        reflected = pd.DataFrame({"bc": [750.0], "ab": [850.0], "ac": [400.0]})
        self.assertEqual(geometry_signature(left), geometry_signature(reflected))

    def test_internal_split_keeps_positions_disjoint_across_reynolds_numbers(self) -> None:
        rows = []
        for source in ("a.xlsx", "b.xlsx"):
            for position in range(10):
                for reynolds in (6, 12, 18):
                    rows.append(
                        {
                            "source_file": source,
                            "position_group": f"{source}|{position}",
                            "re": reynolds,
                        }
                    )
        train, test = internal_split(pd.DataFrame(rows), test_fraction=0.2, seed=42)
        self.assertTrue(set(train["position_group"]).isdisjoint(test["position_group"]))
        self.assertEqual(train["source_file"].nunique(), 2)
        self.assertEqual(test["source_file"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
