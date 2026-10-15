import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("regression_service_safety", Path(__file__).resolve().parents[1] / "web-demo/tabpfn_service.py")
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)


def request(radius=1200):
    return {"side_ab_um": 500, "side_ac_um": 500, "side_bc_um": 600, "curvature_radius_um": radius,
            "positions": [{"point_id": "p", "x_um": 300, "z_um": 100}]}


class ServiceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.manager = SERVICE.ModelManager()
        self.manager.ready = True
        self.manager.mode = "joblib"
        self.calls = 0
        def predict(rows):
            self.calls += 1
            return {key: [float(index)] * len(rows) for index, key in enumerate(SERVICE.TARGET_COLUMNS)}
        self.manager.predict = predict
        self.patch_manager = patch.object(SERVICE, "MODEL_MANAGER", self.manager)
        self.patch_manager.start()
        self.addCleanup(self.patch_manager.stop)
        self.patch_env = patch.dict(os.environ, {"TRIFOCUS_CACHE": "1", "TRIFOCUS_CACHE_MAX_ENTRIES": "2"})
        self.patch_env.start()
        self.addCleanup(self.patch_env.stop)
        SERVICE.PREDICTION_CACHE.clear()

    def test_cache_is_bounded_and_model_revision_aware(self):
        first = SERVICE.predict_triangle(request())
        second = SERVICE.predict_triangle(request())
        self.assertTrue(second["prediction_meta"]["cache_hit"])
        self.assertEqual(self.calls, 1)
        self.manager.revision = "new-model"
        SERVICE.predict_triangle(request())
        self.assertEqual(self.calls, 2)
        SERVICE.predict_triangle(request(1500))
        self.assertEqual(len(SERVICE.PREDICTION_CACHE), 2)
        self.assertFalse(first["prediction_meta"]["cache_hit"])

    def test_close_numeric_inputs_do_not_share_rounded_cache_keys(self):
        SERVICE.predict_triangle(request(1200))
        result = SERVICE.predict_triangle(request(1200.0000001))
        self.assertFalse(result["prediction_meta"]["cache_hit"])
        self.assertEqual(self.calls, 2)

    def test_cache_returns_copies(self):
        first = SERVICE.predict_triangle(request())
        first["sweep"][0]["points"][0]["fd_pred"] = 999
        second = SERVICE.predict_triangle(request())
        self.assertNotEqual(second["sweep"][0]["points"][0]["fd_pred"], 999)

    def test_failure_fallback_is_not_cached(self):
        good = self.manager.predict
        self.manager.predict = lambda rows: {key: [float("nan")] * len(rows) for key in SERVICE.TARGET_COLUMNS}
        first = SERVICE.predict_triangle(request())
        self.assertFalse(first["prediction_meta"]["uses_model"])
        self.assertEqual(len(SERVICE.PREDICTION_CACHE), 0)
        self.manager.predict = good
        second = SERVICE.predict_triangle(request())
        self.assertTrue(second["prediction_meta"]["uses_model"])

    def test_strict_prediction_does_not_fall_back(self):
        self.manager.ready = False
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            SERVICE.predict_triangle({**request(), "require_model": True})
        self.manager.ready = True
        self.manager.predict = lambda rows: {}
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            SERVICE.predict_triangle({**request(), "require_model": True})

    def test_model_outputs_need_all_four_aligned_finite_components(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            SERVICE.checked_model_predictions({}, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            SERVICE.checked_model_predictions({key: [1., 2.] for key in SERVICE.TARGET_COLUMNS}, 1)

    def test_focus_and_force_units_are_not_presented_as_validated(self):
        result = SERVICE.predict_triangle(request())
        self.assertEqual(result["focus"]["method"], "heuristic_focus_rule")
        self.assertFalse(result["focus"]["confidence_is_probability"])
        self.assertFalse(result["prediction_meta"]["fnet_physically_validated"])
        self.assertFalse(result["prediction_meta"]["radius_is_model_feature"])

    def test_loading_resets_stale_models_and_revision(self):
        manager = SERVICE.ModelManager()
        manager.models = {"old": object()}
        old_revision = manager.revision
        with patch.dict(os.environ, {"TRIFOCUS_MODEL_MODE": "fallback"}):
            manager.load()
        self.assertFalse(manager.ready)
        self.assertEqual(manager.models, {})
        self.assertNotEqual(manager.revision, old_revision)

    def test_zero_capacity_disables_existing_cache_entries(self):
        SERVICE.predict_triangle(request())
        with patch.dict(os.environ, {"TRIFOCUS_CACHE_MAX_ENTRIES": "0"}):
            result = SERVICE.predict_triangle(request())
        self.assertFalse(result["prediction_meta"]["cache_hit"])
        self.assertEqual(len(SERVICE.PREDICTION_CACHE), 0)

    def test_duplicate_position_ids_are_rejected(self):
        payload = request()
        payload["positions"].append({"point_id": "p", "x_um": 310, "z_um": 100})
        with self.assertRaisesRegex(ValueError, "unique"):
            SERVICE.predict_triangle(payload)


if __name__ == "__main__":
    unittest.main()
