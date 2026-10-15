from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.focus_inspection.train_focus_tabpfn import (
    geometry_group_from_source_file,
    load_focus_labels,
    summary_id_from_source_file,
)


class FocusTabPFNClassificationTests(unittest.TestCase):
    def test_source_names_map_to_label_summary_ids(self) -> None:
        cases = {
            "750-400-10-Fzong.xlsx": "750-400-10",
            "750-400-10-Fzong反.xlsx": "750-400反-10",
            "750-530-反-30-Fzong.xlsx": "750-530反-30",
        }
        for source_file, expected in cases.items():
            with self.subTest(source_file=source_file):
                self.assertEqual(summary_id_from_source_file(source_file), expected)

    def test_geometry_group_ignores_radius_and_orientation(self) -> None:
        names = [
            "750-400-10-Fzong.xlsx",
            "750-400-20-Fzong反.xlsx",
            "750-400-反-30-Fzong.xlsx",
        ]
        self.assertEqual({geometry_group_from_source_file(name) for name in names}, {"750-400"})

    def test_label_matrix_becomes_unique_binary_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "id": ["750-400-10", "750-400反-10"],
                6: ["F", "UF"],
                12: ["UF", "F"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.xlsx"
            frame.to_excel(path, index=False)
            labels = load_focus_labels(path)

        self.assertEqual(len(labels), 4)
        self.assertEqual(labels["focus_label"].sum(), 2)
        self.assertFalse(labels.duplicated(["summary_id", "re"]).any())


if __name__ == "__main__":
    unittest.main()
