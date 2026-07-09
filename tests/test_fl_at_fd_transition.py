import unittest

import numpy as np
import pandas as pd

from src.data.template_schema import standardize_template_frame
from src.experiments.predict_fl_at_fd_points_tabpfn import augment_fl_points, select_model_features
from src.experiments.predict_fd_u_from_interpolated_fl_tabpfn import (
    discover_pairs,
    family_key,
    map_augmented_fl_to_fd,
    radius_from_key,
)
from src.experiments.predict_fl_positions_from_matching_fd_tabpfn import (
    select_features as select_fd_to_fl_features,
    valid_rows,
)
from src.experiments.interpolate_predicted_fl_files import interpolate_u_pred


class TemplateTransitionTest(unittest.TestCase):
    def test_maps_legacy_si_fl_to_current_um_schema(self):
        legacy = pd.DataFrame({
            "三角形底边": [750e-6], "高": [400e-6],
            "角度1": [60], "角度2": [60], "角度3": [60],
            "雷诺数": [12], "X": [100e-6], "Z": [80e-6], "FL": [1.25],
        })
        result, report = standardize_template_frame(legacy, kind="fl", source_length_unit="auto")
        self.assertAlmostEqual(result.loc[0, "bc"], 750.0)
        self.assertAlmostEqual(result.loc[0, "x"], 100.0)
        self.assertEqual(report["unit_multipliers"]["bc"], 1e6)

    def test_augment_adds_midpoints_and_preserves_original_fl(self):
        rows = []
        for x, z, fl in [(0, 0, 0), (2, 0, 2), (0, 2, 2), (2, 2, 4)]:
            rows.append({
                "bc": 10, "h": 10, "angle_a": 60, "angle_b": 60, "angle_c": 60,
                "re": 12, "x": x, "z": z, "fl": fl,
            })
        augmented = augment_fl_points(pd.DataFrame(rows))
        self.assertEqual(len(augmented), 9)
        self.assertEqual(int(augmented["generated"].sum()), 5)
        center = augmented[(augmented["x"] == 1) & (augmented["z"] == 1)].iloc[0]
        self.assertAlmostEqual(center["fl"], 2.0, places=6)
        corner = augmented[(augmented["x"] == 2) & (augmented["z"] == 2)].iloc[0]
        self.assertFalse(bool(corner["generated"]))
        self.assertAlmostEqual(corner["fl"], 4.0)

    def test_feature_selection_uses_shared_columns_only(self):
        fl = pd.DataFrame({"bc": [1], "h": [1], "re": [1], "x": [0], "z": [0], "fl": [0]})
        fd = pd.DataFrame({"bc": [1], "h": [1], "re": [1], "x": [0], "z": [0], "y": [0]})
        self.assertEqual(select_model_features(fl, fd), ["bc", "h", "re", "x", "z"])

    def test_transitional_english_fd_in_meters_derives_bc(self):
        transitional = pd.DataFrame({
            "h": [400e-6], "area": [1.5e-7], "perimeter": [2e-3],
            "angle_a": [61.93], "angle_b": [28.07], "angle_c": [90],
            "re": [12], "x": [10e-6], "y": [0], "z": [20e-6],
            "dean_velocity_x": [1e-5], "dean_velocity_y": [2e-5],
        })
        result, report = standardize_template_frame(transitional, kind="fd", source_length_unit="auto")
        self.assertEqual(report["unit_multipliers"]["h"], 1e6)
        self.assertAlmostEqual(result.loc[0, "h"], 400.0)
        self.assertAlmostEqual(result.loc[0, "bc"], 750.0)
        self.assertTrue(np.isfinite(result.loc[0, "ab"]))

    def test_mixed_legacy_units_are_normalized_per_column(self):
        mixed = pd.DataFrame({
            "BC": [750e-6], "高": [400.0],
            "角度A": [60], "角度B": [60], "角度C": [60],
            "雷诺数": [12], "X": [100e-6], "Z": [80e-6], "FL": [1.0],
        })
        result, report = standardize_template_frame(mixed, kind="fl", source_length_unit="auto")
        self.assertAlmostEqual(result.loc[0, "bc"], 750.0)
        self.assertAlmostEqual(result.loc[0, "h"], 400.0)
        self.assertEqual(report["unit_multipliers"]["bc"], 1e6)
        self.assertEqual(report["unit_multipliers"]["h"], 1.0)

    def test_fd_u_is_supported_as_explicit_target(self):
        frame = pd.DataFrame({
            "bc": [750], "h": [320], "angle_a": [99], "angle_b": [40.5],
            "angle_c": [40.5], "re": [6], "x": [100], "z": [50], "u": [1e-5],
        })
        result, report = standardize_template_frame(frame, kind="fd", target_column="u")
        self.assertAlmostEqual(result.loc[0, "u"], 1e-5)
        self.assertEqual(report["target_column"], "u")

    def test_augmented_fl_maps_to_all_fd_points(self):
        fl = pd.DataFrame([
            {"bc": 10, "h": 10, "angle_a": 60, "angle_b": 60, "angle_c": 60,
             "re": 6, "x": x, "z": z, "fl": x + z}
            for x, z in [(0, 0), (2, 0), (0, 2), (2, 2)]
        ])
        fd = pd.DataFrame({"re": [6, 6], "x": [1, 3], "z": [1, 3], "u": [2, 6]})
        mapped = map_augmented_fl_to_fd(augment_fl_points(fl), fd)
        self.assertEqual(int(mapped["fl_interp"].notna().sum()), 2)
        self.assertAlmostEqual(mapped.loc[0, "fl_interp"], 2.0, places=6)
        self.assertEqual(mapped.loc[1, "fl_interp_method"], "nearest")

    def test_pair_family_and_radius_parsing(self):
        self.assertEqual(family_key("750-530-反-20"), "750-530-反")
        self.assertEqual(radius_from_key("750-530-反-20"), 20)

    def test_matching_fd_to_fl_keeps_fl_out_of_model_features(self):
        fd = pd.DataFrame({"re": [6], "x": [1], "z": [2], "u": [3], "bc": [10]})
        fl = pd.DataFrame({"re": [6], "x": [1], "z": [2], "fl": [4], "bc": [10]})
        features = select_fd_to_fl_features(fd, fl)
        self.assertEqual(features, ["bc", "re", "x", "z"])
        self.assertNotIn("fl", features)

    def test_invalid_fl_placeholder_is_not_predictable(self):
        frame = pd.DataFrame({
            "bc": [750, 750], "h": [320, 0], "angle_a": [99, np.nan],
            "angle_b": [40.5, np.nan], "angle_c": [40.5, np.nan],
            "re": [6, np.nan], "x": [100, 0], "z": [50, 0], "fl": [1, 0],
        })
        self.assertEqual(valid_rows(frame).tolist(), [True, False])

    def test_u_pred_is_interpolated_per_file(self):
        original = pd.DataFrame([
            {"bc": 10, "h": 10, "angle_a": 60, "angle_b": 60, "angle_c": 60,
             "re": 6, "x": x, "z": z, "fl": x + z, "u_pred": 2 * x + z}
            for x, z in [(0, 0), (2, 0), (0, 2), (2, 2)]
        ])
        augmented = augment_fl_points(original)
        result = interpolate_u_pred(original, augmented)
        center = result[(result["x"] == 1) & (result["z"] == 1)].iloc[0]
        self.assertAlmostEqual(center["u_pred"], 3.0, places=6)
        corner = result[(result["x"] == 2) & (result["z"] == 2)].iloc[0]
        self.assertEqual(corner["u_pred_fill_method"], "original")
        self.assertAlmostEqual(corner["u_pred"], 6.0)


if __name__ == "__main__":
    unittest.main()
