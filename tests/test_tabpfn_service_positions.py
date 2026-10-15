from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "web-demo" / "tabpfn_service.py"
SPEC = importlib.util.spec_from_file_location("tabpfn_service_under_test", SERVICE_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)


class TabpfnServicePositionTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["TRIFOCUS_CACHE"] = "0"
        SERVICE.MODEL_MANAGER.ready = False
        SERVICE.MODEL_MANAGER.mode = "fallback"
        SERVICE.MODEL_MANAGER.error = "test fallback"

    def test_uses_provided_positions_and_composes_fnet(self) -> None:
        result = SERVICE.predict_triangle(
            {
                "side_ab_um": 500,
                "side_ac_um": 500,
                "side_bc_um": 600,
                "curvature_radius_um": 1200,
                "positions": [
                    {"point_id": "center", "x_um": 300, "z_um": 100},
                    {"point_id": "lower", "x_um": 300, "z_um": 20},
                ],
            }
        )

        self.assertEqual(result["prediction_meta"]["data_source"], "provided_positions")
        self.assertEqual(result["prediction_meta"]["point_count_per_re"], 2)
        self.assertEqual(result["prediction_meta"]["generated_rows"], 20)
        point = result["sweep"][0]["points"][0]
        self.assertAlmostEqual(point["fnet_pred"], point["fl_pred"] + point["fd_pred"])
        self.assertAlmostEqual(point["fnet_x"], point["total_force_x"])
        self.assertAlmostEqual(point["fnet_z"], point["total_force_y"])

    def test_samples_default_points_when_positions_are_absent(self) -> None:
        result = SERVICE.predict_triangle(
            {
                "side_ab_um": 500,
                "side_ac_um": 500,
                "side_bc_um": 600,
                "curvature_radius_um": 1200,
            }
        )
        self.assertEqual(result["prediction_meta"]["data_source"], "halton_sampled_positions")
        self.assertEqual(result["prediction_meta"]["point_count_per_re"], SERVICE.POINT_COUNT)

    def test_rejects_position_outside_triangle(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the triangle"):
            SERVICE.predict_triangle(
                {
                    "side_ab_um": 500,
                    "side_ac_um": 500,
                    "side_bc_um": 600,
                    "curvature_radius_um": 1200,
                    "positions": [{"x_um": 1000, "z_um": 1000}],
                }
            )


if __name__ == "__main__":
    unittest.main()
