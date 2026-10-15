import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.regression.evaluation import FEATURE_COLUMNS, sample_complete_positions
from src.regression.optimize_fd import (
    RECIPES,
    evaluate,
    features,
    load_native_fd,
    select_recipe,
    split,
    target_factor,
)


def sample_data():
    rows = []
    for g in range(4):
        for point in range(6):
            for re in (6, 12):
                rows.append(
                    {
                        **dict.fromkeys(FEATURE_COLUMNS, 1.0),
                        "geometry_group": f"g{g}",
                        "source_file": f"f{g}.xlsx",
                        "source_row": point * 2 + re // 6,
                        "position_group": f"g{g}|{point}",
                        "re": re,
                        "radius": 10.0,
                        "target": float(point + re),
                        "x": point,
                        "z": point / 2,
                    }
                )
    return pd.DataFrame(rows)


class FDOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.data = sample_data()
        self.args = SimpleNamespace(max_train_rows=20, validation_rows=5, test_rows=7, seed=42)

    def test_relative_features_do_not_use_target_or_bad_area(self):
        before = features(self.data, RECIPES[1])
        changed = self.data.assign(target=999.0, area=-1.0)
        pd.testing.assert_frame_equal(before, features(changed, RECIPES[1]))

    def test_empirical_target_transform_round_trip(self):
        for recipe in RECIPES:
            factor = target_factor(self.data, recipe)
            np.testing.assert_allclose(self.data.target / factor * factor, self.data.target)

    def test_split_is_disjoint_budgeted_and_keeps_test_re_together(self):
        train, test = split(self.data, "g0", self.args)
        self.assertEqual(len(train), 20)
        self.assertEqual(set(test.geometry_group), {"g0"})
        self.assertNotIn("g0", set(train.geometry_group))
        self.assertTrue((test.groupby("position_group").size() == 2).all())

    def test_selection_requires_all_inner_groups(self):
        rows = pd.DataFrame(
            [
                {"recipe": recipe.name, "validation_geometry": group, "r2": i}
                for i, recipe in enumerate(RECIPES[:2])
                for group in ("a", "b", "c")
            ]
        )
        self.assertEqual(select_recipe(rows, RECIPES[:2], {"a", "b", "c"}), RECIPES[1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            select_recipe(rows.iloc[:-1], RECIPES[:2], {"a", "b", "c"})

    def test_unequal_position_sizes_respect_sampling_budget(self):
        data = pd.DataFrame(
            [{"source_file": "a", "position_group": str(group)} for group in range(20) for _ in range(group + 1)]
        )
        original = data.copy(deep=True)
        sampled = sample_complete_positions(data, 40, 42)
        self.assertGreaterEqual(len(sampled), 40)
        self.assertLess(len(sampled), 40 + 20)
        for position, rows in sampled.groupby("position_group"):
            self.assertEqual(len(rows), int(position) + 1)
        pd.testing.assert_frame_equal(original, data)

    def test_nested_coverage_and_outer_labels_cannot_change_own_selection(self):
        def fake_predict(train, test, recipe, args):
            self.assertFalse(set(train.geometry_group) & set(test.geometry_group))
            return test.x.to_numpy() + 9.0 + (0 if recipe == RECIPES[0] else 1.0)

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            folds, _ = evaluate(self.data, self.args, out, RECIPES[:2], fake_predict)
            first = pd.read_json(out / "selections.json")
            changed = self.data.copy()
            changed.loc[changed.geometry_group == "g0", "target"] *= 100
            evaluate(changed, self.args, out, RECIPES[:2], fake_predict)
            second = pd.read_json(out / "selections.json")
            self.assertEqual(first.loc[0, "selected_recipe"], second.loc[0, "selected_recipe"])
            self.assertEqual(len(folds[folds.method == "nested_selected"]), 4)

    def test_loader_fails_instead_of_dropping_bad_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            data = self.data.iloc[:3].copy().assign(u=[1.0, float("inf"), 3.0])
            data.to_excel(Path(folder) / "test-10-FD.xlsx", index=False)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_native_fd(Path(folder))

    def test_requires_four_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "four"):
                evaluate(self.data[self.data.geometry_group != "g0"], self.args, Path(folder))


if __name__ == "__main__":
    unittest.main()
