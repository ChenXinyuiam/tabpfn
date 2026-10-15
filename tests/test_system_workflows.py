import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import audit_sources, load_snapshot
from src.evaluation.protocols import fold_manifest, regression_metrics
from src.evaluation.system import evaluate_system
from src.geometry.triangle import FEATURES, query_grid
from src.physics.force import PhysicsUnavailable, compose_force
from src.workflows.artifacts import load_bundle, save_bundle
from src.workflows.demo import example_config as config
from src.workflows.demo import example_request as request
from src.workflows.demo import synthetic_dataset
from src.workflows.engine import crossfit_fields, fit_system, predict_system
from utils.provenance import write_json


class SystemWorkflowsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, classes=None):
        manifest = synthetic_dataset(self.root / "inputs", classes=classes)
        snapshot = self.root / "snapshot"
        audit_sources(manifest, snapshot)
        return (*load_snapshot(snapshot), snapshot)

    def test_query_and_training_features_are_unit_invariant(self):
        req = request()
        first = query_grid(req)
        req["geometry"]["unit"] = "m"
        req["geometry"]["vertices"] = (np.array(req["geometry"]["vertices"]) / 1e6).tolist()
        second = query_grid(req)
        np.testing.assert_allclose(first[FEATURES], second[FEATURES], rtol=1e-12, atol=1e-12)
        self.assertEqual(set(first.geometry_family_id), set(second.geometry_family_id))
        self.assertFalse(first.duplicated(["sample_id", "point_id"]).any())

    def test_invalid_and_duplicate_queries_fail(self):
        req = request()
        req["positions"], req["position_unit"] = [[200, 300]], "um"
        with self.assertRaisesRegex(ValueError, "outside"):
            query_grid(req)
        req["positions"] = [[30, 10], [30, 10]]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            query_grid(req)
        req = request()
        req["geometry"]["vertices"] = [[0, 0], [1, 0], [2, 0]]
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            query_grid(req)

    def test_audit_preserves_origins_and_rejects_duplicates(self):
        path = synthetic_dataset(self.root / "inputs")
        manifest = json.loads(path.read_text())
        manifest["sources"].append(manifest["sources"][0])
        write_json(path, manifest)
        result = audit_sources(path, self.root / "snapshot")
        self.assertEqual(result["status"], "REJECT")
        self.assertIn("duplicate_point_identity", (self.root / "snapshot/issues.csv").read_text())
        with self.assertRaisesRegex(ValueError, "rejected"):
            load_snapshot(self.root / "snapshot")

    def test_invalid_rows_are_recorded_and_not_silently_used(self):
        path = synthetic_dataset(self.root / "inputs")
        csv = path.parent / "0-FD.csv"
        frame = pd.read_csv(csv)
        frame.loc[0, "value"] = np.inf
        frame.to_csv(csv, index=False)
        audit = audit_sources(path, self.root / "snapshot")
        self.assertEqual(audit["status"], "REJECT")
        rejected = pd.read_csv(self.root / "snapshot/rejected_rows.csv")
        self.assertEqual(rejected.source_row.iloc[0], 2)

    def test_supplied_geometry_conflict_rejects_source(self):
        path = synthetic_dataset(self.root / "inputs")
        csv = path.parent / "0-FD.csv"
        frame = pd.read_csv(csv)
        frame["h"] = 10
        frame.to_csv(csv, index=False)
        audit = audit_sources(path, self.root / "snapshot")
        self.assertEqual(audit["status"], "REJECT")
        self.assertIn("Supplied geometry conflicts", (self.root / "snapshot/issues.csv").read_text())

    def test_raw_duplicate_headers_are_rejected_before_pandas_renames_them(self):
        path = synthetic_dataset(self.root / "inputs")
        csv = path.parent / "0-FD.csv"
        csv.write_text("x,z,re,value,value\n1,1,6,1,2\n", encoding="utf-8")
        audit = audit_sources(path, self.root / "snapshot")
        self.assertEqual(audit["status"], "REJECT")
        self.assertIn("Headers must be", (self.root / "snapshot/issues.csv").read_text())

    def test_inference_has_no_dependency_on_query_targets(self):
        points, samples, meta, _ = self.dataset()
        fitted = fit_system(points, samples, meta, config(), "fields")
        query = query_grid(request())
        before, _, _ = predict_system(fitted, query.assign(target=1e20), meta, config())
        after, _, _ = predict_system(fitted, query.assign(target=-1e20), meta, config())
        np.testing.assert_allclose(before.FD_pred, after.FD_pred)
        np.testing.assert_allclose(before.FL_pred, after.FL_pred)

    def test_unverified_targets_allow_regression_but_block_physics(self):
        points, samples, meta, _ = self.dataset()
        meta["targets"]["FD"].update(quantity="unverified", unit="native")
        fitted = fit_system(points, samples, meta, config(), "fields")
        predicted, focus, status = predict_system(fitted, query_grid(request()), meta, config())
        self.assertIn("FD_pred", predicted)
        self.assertNotIn("Fzong_pred", predicted)
        self.assertTrue(focus.empty)
        self.assertEqual(status["Fzong"], "not_executed")

    def test_force_conversion_does_not_add_velocity(self):
        targets = {
            "FD": {"quantity": "velocity", "unit": "m/s", "component": "x"},
            "FL": {"quantity": "force", "unit": "nN", "component": "x"},
        }
        cfg = config()["physics"]
        with self.assertRaises(PhysicsUnavailable):
            compose_force([1], [2], targets, cfg)
        cfg["fd_conversion"] = {
            "kind": "declared_linear",
            "confirmed": True,
            "reference": "test only",
            "gain_N_per_m_s": 1e-9,
        }
        np.testing.assert_allclose(compose_force([1], [2], targets, cfg), [3])
        targets["FD"] = {"quantity": "force", "unit": "uN", "component": "x"}
        np.testing.assert_allclose(compose_force([1], [2], targets, cfg), [1002])
        targets["FL"]["component"] = "z"
        with self.assertRaises(PhysicsUnavailable):
            compose_force([1], [2], targets, cfg)

    def test_geometry_fold_manifest_is_shared_across_tasks_and_re(self):
        points, _, _, _ = self.dataset()
        assignment = fold_manifest(points, 5)
        self.assertTrue(assignment.groupby("geometry_family_id").fold.nunique().eq(1).all())
        self.assertEqual(set(assignment.fold), {1, 2, 3, 4, 5})
        self.assertEqual(len(assignment), len(points))
        with self.assertRaisesRegex(ValueError, "independent groups"):
            fold_manifest(points, 7)

    def test_mirrors_share_geometry_group(self):
        req = request()
        original = query_grid(req)
        req["geometry"]["vertices"][0][0] = 30
        left = query_grid(req)
        req["geometry"]["vertices"][0][0] = 70
        right = query_grid(req)
        self.assertEqual(left.geometry_family_id.iloc[0], right.geometry_family_id.iloc[0])
        self.assertNotEqual(original.geometry_family_id.iloc[0], left.geometry_family_id.iloc[0])

    def test_constant_target_r2_is_not_fabricated(self):
        self.assertIsNone(regression_metrics([1, 1], [1, 1])["r2"])
        self.assertLess(regression_metrics([0, 1], [10, 10])["r2"], 0)

    def test_crossfit_does_not_train_on_query_geometry(self):
        points, _, meta, _ = self.dataset()
        predictions, audit = crossfit_fields(points, meta, config())
        self.assertEqual(len(predictions), int(points.task.eq("Fzong").sum()))
        for fold in audit:
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
        changed = points.copy()
        changed.loc[changed.task.eq("Fzong"), "target"] = 1e20
        other, _ = crossfit_fields(changed, meta, config())
        np.testing.assert_allclose(predictions.Fzong_pred, other.Fzong_pred)

    def test_inner_heldout_field_targets_cannot_change_its_predictions(self):
        points, _, meta, _ = self.dataset()
        expected, _ = crossfit_fields(points, meta, config())
        group = expected.geometry_family_id.iloc[0]
        changed = points.copy()
        changed.loc[changed.geometry_family_id.eq(group) & changed.task.isin(["FD", "FL"]), "target"] += 1e8
        actual, _ = crossfit_fields(changed, meta, config())
        left = expected.loc[expected.geometry_family_id.eq(group)].sort_values(["sample_id", "point_id"])
        right = actual.loc[actual.geometry_family_id.eq(group)].sort_values(["sample_id", "point_id"])
        np.testing.assert_allclose(left.Fzong_pred, right.Fzong_pred)

    def test_cascade_five_fold_and_three_class_predictions(self):
        points, samples, meta, _ = self.dataset()
        result = evaluate_system(points, samples, meta, config(), task="cascade", n_splits=5)
        self.assertEqual(len(result["focus_oof"]), 18)
        self.assertEqual(set(result["metrics"]["summary"]), {"FD", "FL", "Fzong", "focus"})
        self.assertTrue({"probability_F", "probability_UF", "probability_WF"} <= set(result["focus_oof"]))
        for audit in result["metrics"]["inner_audits"]:
            for inner in audit["inner_splits"]:
                self.assertFalse(set(audit["outer_test_groups"]) & set(inner["train_groups"]))
                self.assertFalse(set(audit["outer_test_groups"]) & set(inner["test_groups"]))

    def test_binary_focus_still_supported(self):
        points, samples, meta, _ = self.dataset(["UF", "F"])
        result = evaluate_system(points, samples, meta, config(), task="focus", n_splits=5)
        self.assertNotIn("probability_WF", result["focus_oof"])

    def test_focus_refuses_missing_training_class_and_spatial_protocol(self):
        points, samples, meta, _ = self.dataset()
        samples.loc[samples.label.eq("WF"), "label"] = "F"
        with self.assertRaisesRegex(ValueError, "every declared"):
            fit_system(points, samples, meta, config(), "focus")
        with self.assertRaisesRegex(ValueError, "complete geometry"):
            evaluate_system(points, samples, meta, config(), task="focus", protocol="spatial")

    def test_snapshot_tampering_is_detected(self):
        _, _, _, snapshot = self.dataset()
        with (snapshot / "points.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_snapshot(snapshot)

    def test_bundle_roundtrip_preserves_crossfit_and_predictions(self):
        points, samples, meta, snapshot = self.dataset()
        cfg = config()
        fitted = fit_system(points, samples, meta, cfg, "cascade")
        query = query_grid(request(130, 75, "new-triangle"))
        expected, expected_focus, _ = predict_system(fitted, query, meta, cfg)
        directory = self.root / "model"
        save_bundle(snapshot, directory, cfg, "cascade", fitted)
        restored, _, _, _, bundle = load_bundle(directory)
        actual, actual_focus, status = predict_system(restored, query, meta, cfg)
        np.testing.assert_allclose(expected.Fzong_pred, actual.Fzong_pred)
        np.testing.assert_array_equal(expected_focus.predicted_label, actual_focus.predicted_label)
        self.assertEqual(status["focus"], "complete")
        self.assertEqual(bundle["release_status"], "unvalidated")

    def test_bundle_rejects_modified_saved_classifier_context(self):
        points, samples, meta, snapshot = self.dataset()
        cfg = config()
        fitted = fit_system(points, samples, meta, cfg, "focus")
        directory = self.root / "model"
        save_bundle(snapshot, directory, cfg, "focus", fitted)
        with (directory / "training_focus_features.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Derived training context"):
            load_bundle(directory)


if __name__ == "__main__":
    unittest.main()
