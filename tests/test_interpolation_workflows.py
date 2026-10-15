"""Regression coverage for interpolation workflows moved out of experiments."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.regression.fl_lines import reduce_duplicate_points
from src.regression.fly_midpoints import build_interpolated_frame
from src.regression.fly_tabpfn import process_file
from src.regression.raw_fl_interpolation import interpolate_values


def corners():
    return pd.DataFrame({"re": [6] * 4, "x": [0., 2., 0., 2.], "z": [0., 0., 2., 2.],
                         "fly": [0., 2., 4., 6.]})


class InterpolationWorkflowTests(unittest.TestCase):
    def test_fly_centers_require_four_original_corners(self):
        complete, info = build_interpolated_frame(corners())
        self.assertEqual(info["cell_center_source_quads"], 1)
        self.assertEqual(len(complete), 9)
        self.assertFalse(complete.duplicated(["re", "x", "z"]).any())
        incomplete, info = build_interpolated_frame(corners().iloc[:3])
        self.assertEqual(info["cell_center_source_quads"], 0)
        self.assertFalse((np.isclose(incomplete.x, 1) & np.isclose(incomplete.z, 1)).any())

    def test_fly_keeps_re_slices_independent(self):
        source = pd.concat([corners(), corners().assign(re=12)], ignore_index=True)
        result, _ = build_interpolated_frame(source)
        self.assertEqual(result.groupby("re").size().to_dict(), {6: 9, 12: 9})

    def test_tabpfn_fills_only_generated_targets(self):
        def predict(train, query, **kwargs):
            pd.testing.assert_frame_equal(
                train.reset_index(drop=True),
                corners().sort_values(["re", "z", "x"]).reset_index(drop=True),
                check_dtype=False,
            )
            return np.full(len(query), 123.0), ["re", "x", "z"]

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.xlsx", Path(directory) / "output.xlsx"
            corners().to_excel(source, index=False)
            with patch("src.regression.fly_tabpfn.predict_with_tabpfn", side_effect=predict):
                process_file(source, output, model_path=Path("unused.ckpt"), device="cpu",
                             n_estimators=1, random_state=42, chunk_size=100)
            result = pd.read_excel(output)
            self.assertTrue((result.loc[result.generated, "fly"] == 123.0).all())
            original = result.loc[~result.generated, ["re", "x", "z", "fly"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(original, corners().sort_values(["re", "z", "x"]).reset_index(drop=True), check_dtype=False)

    def test_raw_fl_interpolation_preserves_observed_positions(self):
        points = corners()[["x", "z"]].to_numpy()
        values = corners().fly.to_numpy()
        prediction, _ = interpolate_values(points, values, points)
        np.testing.assert_allclose(prediction, values, atol=1e-8)

    def test_line_duplicate_reduction_remains_available(self):
        points, values = reduce_duplicate_points(np.array([[0., 0.], [0., 0.], [1., 1.]]),
                                                 np.array([2., 4., 8.]))
        self.assertEqual(len(points), 2)
        np.testing.assert_allclose(values, [3., 8.])


if __name__ == "__main__":
    unittest.main()
