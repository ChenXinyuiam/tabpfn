from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.focus_inspection.focus_boundary_feature import extract_focus_boundary_feature


def sample_frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["x", "z", "force"])


class FocusBoundaryFeatureTests(unittest.TestCase):
    def test_centered_median_removes_isolated_red_blue_red(self) -> None:
        sample = sample_frame(
            [
                (0, 0, 1.0),
                (1, 0, 2.0),
                (2, 0, -0.01),
                (3, 0, 3.0),
                (4, 0, 2.0),
                (5, 0, -2.0),
                (6, 0, -3.0),
            ]
        )

        feature, points = extract_focus_boundary_feature(sample)

        self.assertEqual(points.loc[points["x"] == 2, "sign_smoothed"].item(), 1)
        self.assertEqual(feature["immediate_rbr_before"], 1)
        self.assertEqual(feature["immediate_rbr_after"], 0)
        self.assertEqual(feature["rightmost_red_x"], 4.0)
        self.assertEqual(feature["leftmost_blue_x"], 5.0)
        self.assertEqual(feature["boundary_midpoint_x_raw"], 3.0)
        self.assertEqual(feature["boundary_midpoint_x"], 4.5)
        self.assertEqual(feature["boundary_distance_abs"], 1.0)
        self.assertEqual(feature["boundary_gap_signed"], 1.0)

    def test_smoothing_is_symmetric_for_isolated_red_point(self) -> None:
        sample = sample_frame(
            [
                (0, 0, -2.0),
                (1, 0, -1.0),
                (2, 0, 0.01),
                (3, 0, -3.0),
                (4, 0, -2.0),
                (5, 0, 1.0),
                (6, 0, 2.0),
            ]
        )

        feature, points = extract_focus_boundary_feature(sample)

        self.assertEqual(points.loc[points["x"] == 2, "sign_smoothed"].item(), -1)
        self.assertEqual(feature["immediate_brb_before"], 1)
        self.assertEqual(feature["immediate_brb_after"], 0)

    def test_boundaries_are_global_across_z_rows(self) -> None:
        sample = sample_frame(
            [
                (0, 0, 1.0),
                (1, 0, 1.0),
                (2, 0, -1.0),
                (10, 1, -1.0),
                (11, 1, -1.0),
                (12, 1, 1.0),
            ]
        )

        feature, _ = extract_focus_boundary_feature(sample)

        self.assertEqual(feature["rightmost_red_x"], 12.0)
        self.assertEqual(feature["leftmost_blue_x"], 2.0)
        self.assertEqual(feature["boundary_midpoint_x"], 7.0)
        self.assertEqual(feature["boundary_distance_abs"], 10.0)
        self.assertEqual(feature["boundary_gap_signed"], -10.0)

    def test_duplicate_coordinates_are_averaged(self) -> None:
        sample = sample_frame(
            [
                (0, 0, 1.0),
                (0, 0, 3.0),
                (1, 0, -1.0),
                (2, 0, -2.0),
            ]
        )

        feature, points = extract_focus_boundary_feature(sample)

        first = points.loc[points["x"] == 0].iloc[0]
        self.assertEqual(first["force_raw"], 2.0)
        self.assertEqual(first["source_rows"], 2)
        self.assertEqual(feature["raw_rows"], 4)
        self.assertEqual(feature["unique_coordinate_points"], 3)
        self.assertEqual(feature["duplicate_coordinate_groups"], 1)

    def test_single_color_produces_missing_boundary_feature(self) -> None:
        sample = sample_frame([(0, 0, 1.0), (1, 0, 2.0), (2, 0, 3.0)])

        feature, _ = extract_focus_boundary_feature(sample)

        self.assertFalse(feature["has_both_signs"])
        self.assertTrue(np.isnan(feature["boundary_midpoint_x"]))
        self.assertTrue(np.isnan(feature["boundary_distance_abs"]))
        self.assertTrue(np.isnan(feature["boundary_gap_signed"]))

    def test_row_endpoints_keep_original_force(self) -> None:
        sample = sample_frame([(0, 0, -1.0), (1, 0, 3.0), (2, 0, 2.0)])

        _, points = extract_focus_boundary_feature(sample)

        self.assertEqual(points.loc[points["x"] == 0, "force_smoothed"].item(), -1.0)
        self.assertEqual(points.loc[points["x"] == 2, "force_smoothed"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
