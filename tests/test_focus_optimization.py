from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.focus_inspection.dataset import build_feature_dataset
from src.focus_inspection.focus_boundary_feature import smooth_horizontal_rows, spatial_distribution_features
from src.focus_inspection.optimize_focus import nested_evaluate
from src.focus_inspection.plot_fzong_sign_by_re import points_for_plot


class FocusOptimizationTests(unittest.TestCase):
    def test_raw_plot_retains_duplicate_rows_and_mean_plot_preserves_counts(self):
        sample = pd.DataFrame({"x": [0.0, 0.0, 1.0], "z": [0.0, 0.0, 0.0], "force": [2.0, -1.0, -2.0]})
        raw = points_for_plot(sample, "raw")
        mean = points_for_plot(sample, "mean")
        self.assertEqual(len(raw), 3)
        self.assertEqual(len(mean), 2)
        self.assertEqual(mean["source_rows"].sum(), len(sample))
        np.testing.assert_array_equal(raw["x"], sample["x"])
        self.assertTrue(mean.loc[mean["x"].eq(0), "mixed_duplicate_sign"].item())

    def test_vectorized_smoothing_matches_centered_reference_and_preserves_endpoints(self):
        rng = np.random.default_rng(12)
        force = rng.normal(size=21)
        for window in (3, 5, 9, 23):
            with self.subTest(window=window):
                points = pd.DataFrame(
                    {"x": np.arange(len(force)), "z": 0, "force_raw": force, "sign_raw": np.sign(force).astype(int)}
                )
                expected = force.copy()
                half = window // 2
                for i in range(half, len(force) - half):
                    expected[i] = np.median(force[i - half : i + half + 1])
                actual = smooth_horizontal_rows(points, window_size=window)
                np.testing.assert_array_equal(actual["force_smoothed"], expected)
                np.testing.assert_array_equal(points["force_raw"], force)

    def test_spatial_features_are_invariant_to_translation_and_positive_scale(self):
        points = pd.DataFrame(
            {"x": [0, 2, 4, 1, 3, 5], "z": [0, 0, 0, 1, 1, 1], "sign_smoothed": [1, 1, -1, 1, -1, -1]}
        )
        base = spatial_distribution_features(points)
        shifted = spatial_distribution_features(points.assign(x=points["x"] * 7 + 100))
        for key in base.keys() - {"x_min", "x_max"}:
            self.assertAlmostEqual(base[key], shifted[key])

    def test_nested_validation_never_sends_test_labels_or_shared_groups_to_model(self):
        rows = []
        for index in range(4):
            for label in (0, 1):
                rows.append(
                    {
                        "source_file": f"{index}.xlsx",
                        "summary_id": str(index),
                        "geometry_group": str(index),
                        "R": 10,
                        "re": label + 1,
                        "focus_label": label,
                        "focus_label_name": "F" if label else "UF",
                        "signal": 0.8 if label else 0.2,
                    }
                )
        data = pd.DataFrame(rows)
        calls = []

        def fake_predictor(train, test, **kwargs):
            self.assertNotIn("focus_label", test)
            self.assertNotIn("focus_label_name", test)
            self.assertFalse(set(train["geometry_group"]) & set(test["geometry_group"]))
            calls.append(len(test))
            return test["signal"].to_numpy()

        with contextlib.redirect_stdout(io.StringIO()):
            result, metrics, audit = nested_evaluate(
                data,
                model_path=Path("unused"),
                device="cpu",
                inner_splits=2,
                predictor=fake_predictor,
            )
        np.testing.assert_array_equal(result["pred_nested"], data["focus_label"])
        self.assertEqual(result["cv_fold"].nunique(), 4)
        self.assertEqual(len(audit["inner_folds"]), 8)
        self.assertEqual(len(calls), 40)
        self.assertTrue(metrics["accuracy"].eq(1).all())
        for fold in audit["inner_folds"]:
            self.assertFalse(set(fold["outer_test_groups"]) & set(fold["fit_groups"] + fold["validation_groups"]))

    def test_replaced_source_file_rejects_stale_features_before_geometry_loading(self):
        features = pd.DataFrame(
            {
                "source_file": ["750-320-10-Fzong.xlsx"],
                "R": [10],
                "re": [6],
                "boundary_distance_abs": [20],
                "boundary_midpoint_x": [235],
                "source_sha256": ["old"],
            }
        )
        with (
            patch("src.focus_inspection.dataset.pd.read_csv", return_value=features),
            patch("src.focus_inspection.dataset.file_sha256", return_value="new"),
            patch("src.focus_inspection.dataset.load_geometry_features") as geometry,
        ):
            with self.assertRaisesRegex(ValueError, "Stale boundary features"):
                build_feature_dataset(Path("raw"), Path("features.csv"))
            geometry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
