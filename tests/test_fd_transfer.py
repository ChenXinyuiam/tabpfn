import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.regression.transfer_diagnostics import identical_components, repeated_conditions
from src.regression.transfer_ensemble import inner_weights
from src.regression.transfer_fd import (
    RECIPES,
    choose_components,
    evaluate,
    factor,
    features,
    guard_training,
    mesh_features,
    predict_components,
)
from tests.test_fd_rotation import data_fixture


class TransferTests(unittest.TestCase):
    def mesh_fixture(self):
        return pd.DataFrame({
            "source_file": ["a"] * 4, "x": [10., 18., 22., 16.], "z": [0., 0., 6., 2.],
            "bc": [7.5] * 4, "h": [5.3] * 4,
            "bary_a": [0.] * 4, "bary_b": [0.] * 4, "bary_c": [0.] * 4,
        })

    def test_mesh_recovers_obtuse_triangle_from_coordinates(self):
        enriched, audit = mesh_features(self.mesh_fixture())
        np.testing.assert_allclose(enriched.iloc[:3][["mesh_a", "mesh_b", "mesh_c"]],
                                   [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        np.testing.assert_allclose(enriched[["mesh_a", "mesh_b", "mesh_c"]].sum(axis=1), 1)
        self.assertEqual(audit[0]["mesh_width"], 8)
        self.assertEqual(audit[0]["mesh_height"], 6)
        self.assertEqual(enriched.mesh_apex.iloc[0], 1.5)

    def test_mesh_is_independent_of_labels_and_nominal_dimensions(self):
        original = self.mesh_fixture()
        first, _ = mesh_features(original)
        second, _ = mesh_features(original.assign(u0=999, u90=-99, bc=900, h=500))
        pd.testing.assert_frame_equal(first.filter(like="mesh_"), second.filter(like="mesh_"))

    def test_mesh_tolerates_boundary_roundoff(self):
        data = self.mesh_fixture()
        data.loc[1, "z"] = 0.000015
        enriched, audit = mesh_features(data)
        self.assertEqual(audit[0]["mesh_width"], 8)
        self.assertLess(abs(enriched.mesh_a.iloc[1]), 1e-4)

    def test_mesh_rejects_incomplete_or_nontriangular_mesh(self):
        with self.assertRaises(ValueError):
            mesh_features(self.mesh_fixture().iloc[2:])

    def test_factor_preserves_native_targets_and_uses_no_labels(self):
        data = pd.DataFrame({"re": [15., 30., 60.], "radius": [10., 20., 10.]})
        np.testing.assert_allclose(factor(data, RECIPES[3], 0), [.25, .5, 4.])
        np.testing.assert_allclose(factor(data, RECIPES[3], 1), [.5, .5, 2.])
        for component in (0, 1):
            scale = factor(data, RECIPES[3], component)
            native = np.array([-3., 0., 10.])
            np.testing.assert_allclose(native / scale * scale, native)

    def test_prediction_does_not_read_query_labels(self):
        data = data_fixture()
        for name, value in {"width": 800., "height": 400., "aspect": .5, "apex": .8,
                            "ab": 1., "ac": .5, "a": .2, "b": .3, "c": .5, "x": .7, "z": .2}.items():
            data["mesh_" + name] = value
        train, query = data[data.geometry_group != "g0"], data[data.geometry_group == "g0"]
        args = SimpleNamespace(model_v25="unused", model_v3="unused")

        def fake_fit(x, y, q, recipe, args):
            return np.full(len(q), np.mean(y))

        for recipe in RECIPES:
            changed = query.copy()
            changed.loc[:, changed.columns.str.startswith(("u", "observed_"))] = 999999
            pd.testing.assert_frame_equal(features(query, recipe), features(changed, recipe))
            np.testing.assert_allclose(predict_components(train, query, recipe, args, fake_fit),
                                       predict_components(train, changed, recipe, args, fake_fit))
        with self.assertRaisesRegex(ValueError, "leakage"):
            predict_components(query, query, RECIPES[0], args, fake_fit)

    def test_symmetric_features_preserve_orientation_and_barycentric_sum(self):
        data, _ = mesh_features(self.mesh_fixture())
        data["re"], data["radius"] = 30., 10.
        recipe = next(r for r in RECIPES if r.symmetric)
        result = features(data, recipe)
        np.testing.assert_allclose(result.mesh_apex, -.5)
        np.testing.assert_allclose(result.orientation, -1)
        np.testing.assert_allclose(result.mesh_a + result.mesh_b + result.mesh_c, 1)
        np.testing.assert_allclose(result.mesh_b, data.mesh_c)

    def test_re_local_context_never_reads_other_geometry_labels(self):
        data = data_fixture()
        train, query = data[data.geometry_group != "g0"], data[data.geometry_group == "g0"]
        recipe = next(r for r in RECIPES if r.local_re)
        from dataclasses import replace
        recipe = replace(recipe, coordinates="provided")
        calls = []

        def fake_fit(x, y, q, recipe, args):
            self.assertEqual(x.re.nunique(), 1)
            self.assertEqual(q.re.nunique(), 1)
            self.assertEqual(x.re.iloc[0], q.re.iloc[0])
            calls.append(len(x))
            return np.zeros(len(q))

        result = predict_components(train, query, recipe,
                                    SimpleNamespace(model_v25="unused", model_v3="unused"), fake_fit)
        self.assertEqual(result.shape, (len(query), 2))
        self.assertEqual(sum(calls), len(train) * 2)

    def test_repeated_condition_diagnostic_ignores_zero_boundaries(self):
        a = pd.DataFrame({"source_file": ["a"] * 12, "x": np.arange(12), "z": 1.,
                          "re": 12., "u0": np.arange(12), "u90": -np.arange(12)})
        repeated = repeated_conditions(pd.concat([a, a.assign(re=60)], ignore_index=True))
        self.assertEqual(len(repeated), 1)
        self.assertEqual(repeated.matched_positions.iloc[0], 12)
        different = repeated_conditions(pd.concat([a, a.assign(re=60, u0=a.u0 * 2)], ignore_index=True))
        self.assertTrue(different.empty)
        identical = identical_components(a)
        self.assertEqual(identical.u0_equals_sign_times_u90.iloc[0], -1)
        self.assertEqual(identical.max_abs_difference.iloc[0], 0)

    def test_training_guard_keeps_ambiguous_test_conditions_untouched(self):
        a = pd.DataFrame({"source_file": ["a"] * 12, "geometry_group": "g1", "x": np.arange(12), "z": 1.,
                          "re": 12., "u0": np.arange(12), "u90": np.arange(12) ** 2})
        pool = pd.concat([a, a.assign(re=60), a.assign(re=30, u0=a.u0 * 3)], ignore_index=True)
        original = pool.copy(deep=True)
        accepted, report = guard_training(pool)
        self.assertEqual(accepted.re.unique().tolist(), [30])
        self.assertEqual({r["re"] for r in report}, {12., 60.})
        pd.testing.assert_frame_equal(pool, original)

    def test_components_can_select_different_recipes(self):
        rows = pd.DataFrame([
            {"recipe": recipe.name, "component": c, "geometry_group": g,
             "r2": float(i == c)}
            for i, recipe in enumerate(RECIPES[:2]) for c in (0, 1) for g in ("a", "b")
        ])
        self.assertEqual(choose_components(rows, {"a", "b"}, RECIPES[:2]), list(RECIPES[:2]))
        with self.assertRaises(ValueError):
            choose_components(rows.iloc[:-1], {"a", "b"}, RECIPES[:2])
        weighted = inner_weights(rows, {"a", "b"}, RECIPES[:2], 0)
        self.assertEqual(weighted[0]["recipe"], RECIPES[0].name)
        self.assertAlmostEqual(sum(row["weight"] for row in weighted), 1)
        self.assertTrue(all(row["weight"] > 0 for row in weighted))

    def test_nested_workflow_outer_labels_do_not_select_own_recipe(self):
        data = data_fixture()
        args = SimpleNamespace(seed=42, inner_only=False)

        def fake_predict(context, query, recipe, args):
            self.assertFalse(set(context.geometry_group) & set(query.geometry_group))
            return np.full((len(query), 2), float(recipe == RECIPES[1]))

        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"):
            out = Path(folder)
            evaluate(data, data, args, out, RECIPES[:2], fake_predict)
            first = json.loads((out / "selections.json").read_text())
            result = pd.read_csv(out / "predictions.csv")
            self.assertFalse(result.duplicated(["method", "base_id", "rotation_deg"]).any())
            self.assertEqual(len(result), len(data) * 24 * 2)
            changed = data.copy()
            mask = changed.geometry_group.isin(first[0]["test_groups"])
            changed.loc[mask, changed.columns.str.startswith("observed_")] *= -100
            evaluate(data, changed, args, out, RECIPES[:2], fake_predict)
            second = json.loads((out / "selections.json").read_text())
            self.assertEqual(first[0], second[0])


if __name__ == "__main__":
    unittest.main()
