import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.regression.fd_from_fl import discover_pairs, pair_key
from src.regression.fd_to_fl import nearest_fd_distance_by_re, valid_rows
from src.regression.fl_at_fd import fit_predict_tabpfn
from src.regression.predict_fields import predict_direction, sample_training_rows, spatial_diagnostics
from src.regression.prediction_report import statistics


def frame():
    return pd.DataFrame({"bc": [10.] * 4, "h": [10.] * 4, "angle_a": [60.] * 4, "angle_b": [60.] * 4,
                         "angle_c": [60.] * 4, "re": [6.] * 4, "x": [0., 2., 0., 2.], "z": [0., 0., 2., 2.],
                         "u": [0., 2., 2., 4.], "fl": [0., 3., 3., 6.]})


class RegressionPredictionTests(unittest.TestCase):
    def test_reflection_aliases_map_to_one_distinct_pair(self):
        for name in ["750-530-反-20-FL.xlsx", "750-530-20-FL反.xlsx", "750-530反-20-FL.xlsx", "750-530-20-FD反.xlsx"]:
            self.assertEqual(pair_key(Path(name), "fl" if "FL" in name else "fd"), "750-530-反-20")
        self.assertNotEqual(pair_key(Path("750-530-20-FD.xlsx"), "fd"), "750-530-反-20")

    def test_ambiguous_pairs_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "FD").mkdir()
            (root / "FL").mkdir()
            for name in ["750-530-20-FL反.xlsx", "750-530-反-20-FL.xlsx"]:
                (root / "FL" / name).touch()
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                discover_pairs(root)

    def test_missing_pairs_require_explicit_permission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "FD").mkdir()
            (root / "FL").mkdir()
            (root / "FD/750-530-20-FD.xlsx").touch()
            (root / "FL/750-530-20-FL.xlsx").touch()
            (root / "FL/750-530-10-FL.xlsx").touch()
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                discover_pairs(root)
            self.assertEqual(list(discover_pairs(root, allow_unpaired=True)), ["750-530-20"])

    def test_distances_use_row_positions_and_matching_re(self):
        source = pd.DataFrame({"re": [6, 12], "x": [0, 100], "z": [0, 0]})
        query = pd.DataFrame({"re": [6, 12, 18], "x": [3, 101, 0], "z": [0, 0, 0]}, index=[7, 25, 90])
        values = nearest_fd_distance_by_re(source, query)
        np.testing.assert_allclose(values[:2], [3, 1])
        self.assertTrue(np.isnan(values[2]))

    def test_infinite_coordinates_are_not_valid_rows(self):
        data = frame()
        data.loc[2, "x"] = np.inf
        self.assertEqual(valid_rows(data, "u").tolist(), [True, True, False, True])

    def test_training_cap_keeps_every_re_and_is_deterministic(self):
        data = pd.concat([frame(), frame().assign(re=12)], ignore_index=True)
        selected = sample_training_rows(data, 5, 42)
        self.assertEqual(len(selected), 5)
        self.assertEqual(set(selected.re), {6, 12})
        pd.testing.assert_frame_equal(selected, sample_training_rows(data, 5, 42))
        with self.assertRaisesRegex(ValueError, "per Re"):
            sample_training_rows(data, 1, 42)

    def test_prediction_preserves_invalid_rows_and_does_not_use_query_labels(self):
        train = frame()
        query = frame().iloc[[0, 1]].copy()
        query.iloc[1, query.columns.get_loc("x")] = np.inf
        seen = []
        def predictor(x_train, y_train, x_query, **kwargs):
            seen.append((x_train.copy(), y_train.copy(), x_query.copy()))
            return np.full(len(x_query), 12.)
        options = dict(target="u", model_path=Path("unused"), device="cpu", n_estimators=1,
                       seed=42, chunk_size=10, max_train_rows=4, predictor=predictor)
        first, audit = predict_direction(train, query, **options)
        second, _ = predict_direction(train, query.assign(u=999, fl=-999), **options)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first.status.tolist(), ["predicted", "invalid_input"])
        self.assertTrue(np.isnan(first.prediction.iloc[1]))
        self.assertEqual(audit["generated_training_rows"], 0)
        self.assertNotIn("u", audit["features"])
        self.assertNotIn("fl", audit["features"])
        np.testing.assert_array_equal(seen[0][1], train.u)

    def test_spatial_warnings_do_not_cross_re(self):
        query = frame().iloc[:3].assign(x=[1., 4., 1.], z=[1., 4., 1.], re=[6., 6., 12.])
        result = spatial_diagnostics(frame(), query, ["re", "x", "z"])
        self.assertEqual(result.spatial_relation.tolist(), ["inside_training_hull", "outside_training_hull", "missing_training_re"])

    def test_chunk_validation_happens_before_loading_tabpfn(self):
        kwargs = dict(model_path=Path("unused"), device="cpu", n_estimators=1, random_state=42)
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_predict_tabpfn(np.ones((2, 1)), np.ones(2), np.ones((1, 1)), chunk_size=0, **kwargs)
        result = fit_predict_tabpfn(np.ones((2, 1)), np.ones(2), np.empty((0, 1)), chunk_size=1, **kwargs)
        self.assertEqual(result.shape, (0,))

    def test_statistics_keep_fields_separate_and_missing_not_zero(self):
        data = pd.DataFrame({"field": ["FD_u", "FD_u", "FL"], "prediction": [0., np.nan, 100.],
            "nearest_training_distance_um": [0., np.nan, 1.], "needs_review": [False, True, False],
            "training_position_overlap": [True, False, False], "spatial_relation": ["inside_training_hull", "invalid_input", "inside_training_hull"]})
        result = statistics(data, ["field"]).set_index("field")
        self.assertEqual(result.loc["FD_u", "mean"], 0)
        self.assertEqual(result.loc["FD_u", "missing_predictions"], 1)
        self.assertEqual(result.loc["FL", "mean"], 100)


if __name__ == "__main__":
    unittest.main()
