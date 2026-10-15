import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.regression.rotation_dataset import ANGLES, CORE, check_alignment, prepare
from src.regression.tune_rotated_fd import (
    RECIPES,
    choose_recipe,
    direct_training,
    evaluate,
    features,
    geometry_folds,
    predict,
    sample_context,
    selection_metrics,
)


def data_fixture():
    records = []
    for group in range(8):
        for row in range(12):
            record = dict.fromkeys(CORE, 1.0)
            record.update(
                geometry_group=f"g{group}",
                source_file=f"f{group}.xlsx",
                source_row=row + 2,
                base_id=f"g{group}|{row}",
                radius=10,
                re=6 + 6 * (row % 2),
                x=row,
                area=1.0,
                perimeter=3.0,
                rotation_deg=0,
                u0=float(row + 1),
                u90=float(2 * row - 4),
            )
            for angle in ANGLES:
                record[f"observed_{angle:03d}"] = record["u0"] * np.cos(np.deg2rad(angle)) + record["u90"] * np.sin(
                    np.deg2rad(angle)
                )
            records.append(record)
    return pd.DataFrame(records)


class RotationTests(unittest.TestCase):
    def test_prepare_quarantines_missing_re_for_every_angle(self):
        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"):
            root = Path(folder) / "input"
            out = Path(folder) / "output"
            out.mkdir()
            base = data_fixture().iloc[:4][CORE].copy()
            base.loc[base.index[1], "re"] = np.nan
            base["area"] = 0.5
            for name in ("prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc"):
                base[name] = np.nan
            for angle in ANGLES:
                path = root / f"angle_{angle:03d}"
                path.mkdir(parents=True)
                data = base.assign(
                    rotation_deg=angle, u=np.arange(4) * np.cos(np.deg2rad(angle)) + 2 * np.sin(np.deg2rad(angle))
                )
                data.to_excel(path / "test-10-FD.xlsx", index=False)
            accepted, sampled = prepare(root, out, engine="openpyxl", test_rows=2)
            self.assertEqual(len(accepted), 3)
            self.assertNotIn(3, accepted.source_row.to_list())
            rejected = pd.read_csv(out / "rejected_rows.csv")
            self.assertEqual(rejected.excluded_rotation_copies.iloc[0], 24)
            self.assertTrue(sampled.filter(like="observed_").notna().all().all())

    def test_group_folds_hold_out_each_geometry_once(self):
        folds = geometry_folds([f"g{i}" for i in range(8)], 5, 42)
        self.assertEqual(sum(len(test) for _, test in folds), 8)
        for train, test in folds:
            self.assertFalse(train & test)

    def test_sampling_is_balanced_and_target_independent(self):
        data = data_fixture()
        first = sample_context(data, 40, 42)
        second = sample_context(data.assign(u0=999.0, u90=-99.0), 40, 42)
        self.assertEqual(first.base_id.tolist(), second.base_id.tolist())
        self.assertTrue((first.groupby("geometry_group").size() == 5).all())
        self.assertFalse(first.base_id.duplicated().any())

    def test_training_pairs_preserve_information(self):
        data = data_fixture()
        paired = direct_training(data, 42)
        self.assertEqual(len(paired), 2 * len(data))
        np.testing.assert_allclose(
            np.square(paired.target.to_numpy().reshape(-1, 2)).sum(axis=1), data.u0**2 + data.u90**2
        )

    def test_features_and_predictions_do_not_use_query_labels(self):
        data = data_fixture()
        train = data[data.geometry_group != "g0"]
        test = data[data.geometry_group == "g0"]
        changed = test.copy()
        changed[["u0", "u90"]] = 999.0
        for recipe in RECIPES:
            pd.testing.assert_frame_equal(features(test, recipe), features(changed, recipe))

            def fake_fit(x, y, q, recipe, args):
                return np.full(len(q), float(np.mean(y)))

            a = predict(train, test, ANGLES, recipe, SimpleNamespace(seed=42), fake_fit)
            b = predict(train, changed, ANGLES, recipe, SimpleNamespace(seed=42), fake_fit)
            np.testing.assert_allclose(a, b)
        with self.assertRaisesRegex(ValueError, "leakage"):
            predict(test, test, ANGLES, RECIPES[0], SimpleNamespace(seed=42), fake_fit)

    def test_alignment_rejects_shifted_rows(self):
        data = data_fixture()
        check_alignment(data, data, 0)
        with self.assertRaises(ValueError):
            check_alignment(data, data.assign(x=data.x + 1), 0)

    def test_selection_requires_each_geometry(self):
        rows = pd.DataFrame(
            [
                {"recipe": recipe.name, "geometry_group": group, "r2": float(i)}
                for i, recipe in enumerate(RECIPES)
                for group in ("a", "b")
            ]
        )
        self.assertEqual(choose_recipe(rows, {"a", "b"}), RECIPES[-1])
        with self.assertRaises(ValueError):
            choose_recipe(rows.iloc[:-1], {"a", "b"})

    def test_selection_weights_reference_angles_equally(self):
        data = data_fixture()
        prediction = np.zeros(len(data) * 4)
        rows = selection_metrics(data, (0, 45, 90, 135), prediction)
        for row in rows:
            self.assertAlmostEqual(row["r2"], (row["r2_000"] + row["r2_090"]) / 2)

    def test_nested_evaluation_has_complete_angle_geometry_coverage(self):
        data = data_fixture()
        args = SimpleNamespace(outer_folds=5, seed=42, label_budget=40)

        def fake_predict(context, test, angles, recipe, args):
            self.assertFalse(set(context.geometry_group) & set(test.geometry_group))
            return np.zeros(len(test) * len(angles))

        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"):
            summary = evaluate(data, data, args, Path(folder), RECIPES[:2], fake_predict)
            self.assertTrue((summary.rows == len(data) * len(ANGLES)).all())
            predictions = pd.read_csv(Path(folder) / "predictions.csv")
            self.assertFalse(predictions.duplicated(["method", "base_id", "rotation_deg"]).any())

    def test_outer_labels_do_not_select_own_recipe(self):
        data = data_fixture()
        args = SimpleNamespace(outer_folds=5, seed=42, label_budget=40)

        def fake_predict(context, test, angles, recipe, args):
            return np.full(len(test) * len(angles), 0.0 if recipe == RECIPES[0] else 1.0)

        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"):
            out = Path(folder)
            evaluate(data, data, args, out, RECIPES[:2], fake_predict)
            first = json.loads((out / "selections.json").read_text())
            changed = data.copy()
            mask = changed.geometry_group.isin(first[0]["test_groups"])
            changed.loc[mask, changed.columns.str.startswith("observed_")] *= -100
            evaluate(data, changed, args, out, RECIPES[:2], fake_predict)
            second = json.loads((out / "selections.json").read_text())
            self.assertEqual(first[0], second[0])


if __name__ == "__main__":
    unittest.main()
