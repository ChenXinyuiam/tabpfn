"""Input, reproducibility and grouped-evaluation boundary regression tests."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.focus_inspection.ablate_focus import evaluate_ablation
from src.focus_inspection.cache import PredictionCache
from src.focus_inspection.dataset import build_feature_dataset, summary_id_from_source_file
from src.focus_inspection.evaluation import evaluate_predictions, make_grouped_folds
from src.focus_inspection.focus_boundary_feature import resolve_source_columns
from src.focus_inspection.predict_focus import _run as predict_run
from src.focus_inspection.prediction_safety import prediction_diagnostics
from src.focus_inspection.quality import FeatureConfig, checked_source_rows, constant_geometry_value, quality_reasons
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run, latest_artifact


def samples():
    records = []
    for group in range(4):
        for label in (0, 1):
            records.append(
                {
                    "source_file": f"{group}.xlsx",
                    "summary_id": str(group),
                    "geometry_group": str(group),
                    "R": 10,
                    "re": label + 1,
                    "focus_label": label,
                    "focus_label_name": "F" if label else "UF",
                    "signal": 0.8 if label else 0.2,
                    "boundary_distance_abs": 20 if label else 150,
                    "source_sha256": f"hash{group}",
                }
            )
    return pd.DataFrame(records)


class FocusSafetyTests(unittest.TestCase):
    def test_undefined_rule_coverage_is_not_reported_as_zero_accuracy(self):
        metric = evaluate_predictions(np.array([]), np.array([]), None)
        self.assertTrue(np.isnan(metric["accuracy"]))
        self.assertEqual(metric["confusion_matrix"], [[0, 0], [0, 0]])

    def test_latest_artifact_uses_completed_run_and_rejects_escaped_pointer(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            with experiment_run(root, "features") as run:
                (run / "features.csv").write_text("fixture", encoding="utf-8")
            self.assertEqual(latest_artifact(root, "features.csv"), run / "features.csv")
            write_json(root / "latest.json", {"run_dir": str(root.parent)})
            with self.assertRaisesRegex(ValueError, "escapes"):
                latest_artifact(root, "features.csv")

    def test_recipe_configuration_is_reused_and_provenance_changes_are_rejected(self):
        from src.focus_inspection.quality import FEATURE_SCHEMA_VERSION, feature_code_hashes

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = samples().assign(boundary_midpoint_x=100)
            training = root / "training.csv"
            train.to_csv(training, index=False)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"fixture")
            recipe = {
                "training_features": "training.csv",
                "training_features_sha256": file_sha256(training),
                "model_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "feature_config": FeatureConfig(window_size=5).to_dict(),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_code_hashes": feature_code_hashes(),
                "candidate": "test",
                "feature_sets": {"test": ["signal"]},
                "seed": 42,
                "n_estimators": 4,
                "threshold": 0.5,
                "model_identity": {
                    "modeling_sha256": file_sha256(Path("src/focus_inspection/modeling.py")),
                    "versions": {},
                },
            }
            recipe_path = root / "recipe.json"
            write_json(recipe_path, recipe)
            with (
                patch("src.focus_inspection.predict_focus.build_features") as builder,
                patch("src.focus_inspection.predict_focus.build_feature_dataset", return_value=train.iloc[[0]].copy()),
                patch(
                    "src.focus_inspection.predict_focus.predict_probability", return_value=np.array([0.9], dtype=np.float32)
                ) as model,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = predict_run(recipe_path, root, root, device="cpu")
                self.assertEqual(builder.call_args.kwargs["config"].window_size, 5)
                self.assertTrue(result.known_training_sample.item())
                self.assertNotIn("legacy_recipe", result.review_reasons.item())
                self.assertEqual(result.raw_focus_probability.dtype, np.dtype(float))
                saved = pd.read_csv(root / "predictions.csv")
                np.testing.assert_array_equal(saved.raw_focus_probability, saved.focus_probability)
                calibrated_recipe = {
                    **recipe,
                    "calibration": {"method": "logit_sigmoid", "coefficient": 0.0, "intercept": 0.0, "clip": 1e-6},
                    "calibration_code_sha256": file_sha256(Path("src/focus_inspection/calibration.py")),
                }
                write_json(recipe_path, calibrated_recipe)
                transformed = predict_run(recipe_path, root, root, device="cpu")
                self.assertTrue(transformed.calibration_applied.item())
                self.assertEqual(transformed.raw_focus_probability.item(), float(np.float32(0.9)))
                self.assertEqual(transformed.focus_probability.item(), 0.5)
                write_json(recipe_path, {**calibrated_recipe, "calibration_code_sha256": "changed"})
                with self.assertRaisesRegex(ValueError, "Calibration implementation"):
                    predict_run(recipe_path, root, root, device="cpu")
                model.reset_mock()
                write_json(recipe_path, {**recipe, "feature_code_hashes": {}})
                with self.assertRaisesRegex(ValueError, "implementation changed"):
                    predict_run(recipe_path, root, root, device="cpu")
                model.assert_not_called()
                write_json(recipe_path, {**recipe, "threshold": 1.0})
                with self.assertRaisesRegex(ValueError, "threshold"):
                    predict_run(recipe_path, root, root, device="cpu")
                write_json(recipe_path, {**recipe, "training_features_sha256": "changed"})
                with self.assertRaisesRegex(ValueError, "snapshot changed"):
                    predict_run(recipe_path, root, root, device="cpu")

    def test_invalid_points_fail_by_default_and_explicit_drop_is_counted(self):
        data = pd.DataFrame({"x": [0, 1, np.inf], "z": [0] * 3, "R": [10] * 3, "re": [6] * 3, "Fzong": [1, -1, 1]})
        columns = resolve_source_columns(data)
        with self.assertRaisesRegex(ValueError, "invalid point rows"):
            checked_source_rows(data, columns, FeatureConfig())
        clean, audit = checked_source_rows(data, columns, FeatureConfig(invalid_rows="drop"))
        self.assertEqual(len(clean), 2)
        self.assertEqual(audit["dropped_rows"], 1)

    def test_explicit_source_units_convert_coordinates_but_not_radius(self):
        data = pd.DataFrame({"x": [0.001], "z": [0.0001], "R": [10], "re": [6], "Fzong": [1]})
        clean, _ = checked_source_rows(data, resolve_source_columns(data), FeatureConfig(length_unit="m"))
        self.assertEqual(clean.x.item(), 1000)
        self.assertEqual(clean.z.item(), 100)
        self.assertEqual(clean.R.item(), 10)

    def test_invalid_feature_settings_fail(self):
        for settings in ({"window_size": 2}, {"window_size": 1}, {"length_unit": "guess"}, {"invalid_rows": "silent"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                FeatureConfig(**settings)

    def test_inconsistent_or_nonfinite_geometry_is_not_silently_averaged(self):
        for values in ([100, 110], [100, np.nan], [np.inf, 100], [-1, -1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                constant_geometry_value(pd.Series(values), source_file="input", column="bc")
        self.assertEqual(constant_geometry_value(pd.Series([100, 100]), source_file="input", column="bc"), 100)

    def test_multiple_radii_use_row_radius_for_label_alignment(self):
        self.assertEqual(summary_id_from_source_file("750-400-10-Fzong.xlsx", 20), "750-400-20")
        boundary = pd.DataFrame(
            {
                "source_file": ["750-400-10-Fzong.xlsx"] * 2,
                "R": [10, 20],
                "re": [6, 6],
                "boundary_distance_abs": [20, 30],
                "boundary_midpoint_x": [100, 110],
            }
        )
        geometry = boundary[["source_file", "R", "re"]].copy()
        for column, value in {
            "bc": 750,
            "ab": 850,
            "ac": 400,
            "h": 400,
            "angle_a": 61.93,
            "angle_b": 28.07,
            "angle_c": 90,
            "area": 150000,
            "perimeter": 2000,
        }.items():
            geometry[column] = value
        with (
            patch("src.focus_inspection.dataset.pd.read_csv", return_value=boundary),
            patch("src.focus_inspection.dataset.load_geometry_features", return_value=geometry),
        ):
            result = build_feature_dataset(Path("raw"), Path("features.csv"))
        self.assertEqual(result.summary_id.tolist(), ["750-400-10", "750-400-20"])

    def test_pure_class_test_geometry_is_allowed(self):
        data = samples()
        data.loc[data.geometry_group.eq("0"), ["focus_label", "focus_label_name"]] = [1, "F"]
        folds, assignments, _ = make_grouped_folds(data, n_splits=4, random_state=42, canonical_order=True)
        self.assertEqual(len(folds), 4)
        self.assertEqual(assignments.cv_fold.nunique(), 4)
        values = evaluate_predictions(np.array([1, 1]), np.array([1, 1]), np.array([0.7, 0.9]))
        self.assertEqual(values["accuracy"], 1)
        self.assertTrue(np.isnan(values["roc_auc"]))
        self.assertTrue(np.isfinite(values["log_loss"]))

    def test_nondefault_dataframe_index_preserves_fold_assignment(self):
        data = samples().set_axis(np.arange(10, 18))
        _, assignments, _ = make_grouped_folds(data, n_splits=4, random_state=42)
        self.assertEqual(len(assignments), len(data))
        self.assertTrue(assignments.cv_fold.gt(0).all())

    def test_pure_class_training_partition_is_rejected(self):
        data = samples()
        data.loc[data.geometry_group.ne("0"), "focus_label"] = 1
        with self.assertRaisesRegex(RuntimeError, "training data"):
            make_grouped_folds(data, n_splits=4, random_state=42, canonical_order=True)

    def test_same_filename_replaced_content_is_not_known_training_sample(self):
        train = samples()
        test = train.iloc[[0]].copy()
        test["source_sha256"] = "replacement"
        result = prediction_diagnostics(train, test, np.array([0.6]), 0.5, ["signal"])
        self.assertTrue(result.seen_sample_key.item())
        self.assertFalse(result.known_training_sample.item())
        self.assertIn("same_key_changed", result.review_reasons.item())
        self.assertEqual(result.decision.item(), "REVIEW")

    def test_exact_training_content_and_unseen_geometry_are_flagged(self):
        train = samples()
        result = prediction_diagnostics(train, train.iloc[[0]], np.array([0.8]), 0.5, ["signal"])
        self.assertTrue(result.known_training_sample.item())
        test = train.iloc[[0]].copy().assign(source_file="new.xlsx", geometry_group="new", signal=10)
        result = prediction_diagnostics(train, test, np.array([0.999]), 0.5, ["signal"])
        self.assertFalse(result.seen_geometry.item())
        self.assertIn("outside_training_range", result.review_reasons.item())
        self.assertTrue(result.needs_review.item())

    def test_legacy_recipe_and_missing_boundary_require_review(self):
        train = samples()
        test = train.iloc[[0]].assign(boundary_distance_abs=np.nan, has_both_signs=False)
        result = prediction_diagnostics(
            train, test, np.array([0.99]), 0.5, ["boundary_distance_abs"], legacy_recipe=True
        )
        self.assertIn("missing_model_features", result.review_reasons.item())
        self.assertIn("legacy_recipe", result.review_reasons.item())

    def test_quality_flags_retain_real_multitransition_patterns(self):
        reasons = quality_reasons(
            {"broad_rbr_rows_after": 1, "duplicate_sign_conflicts": 2, "relative_coordinate_coverage": 0.9}
        )
        self.assertIn("multiple_sign_transitions", reasons)
        self.assertIn("conflicting_duplicate_points", reasons)
        self.assertIn("coordinate_grid_differs_within_file", reasons)

    def test_ablation_uses_equal_seeds_and_estimators_without_test_labels(self):
        calls = []

        def predictor(train, test, **kwargs):
            self.assertNotIn("focus_label", test)
            self.assertFalse(set(train.geometry_group) & set(test.geometry_group))
            calls.append((kwargs["random_state"], kwargs["n_estimators"], tuple(test.source_file)))
            return test.signal.to_numpy()

        with contextlib.redirect_stdout(io.StringIO()):
            result, metrics, _ = evaluate_ablation(
                samples(), model_path=Path("unused"), device="cpu", seeds=(7, 8), predictor=predictor
            )
        self.assertEqual(len(calls), 24)
        for start in (0, 12):
            self.assertEqual(calls[start : start + 4], calls[start + 4 : start + 8])
            self.assertEqual(calls[start : start + 4], calls[start + 8 : start + 12])
        self.assertEqual(len(result), 16)
        self.assertEqual(len(metrics), 10)

    def test_output_runs_are_isolated_and_failure_does_not_publish_latest(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            with experiment_run(root, "test") as first:
                pass
            latest = (root / "latest.json").read_text()
            with self.assertRaisesRegex(ValueError, "failed"), experiment_run(root, "test") as failed:
                raise ValueError("failed")
            self.assertEqual((root / "latest.json").read_text(), latest)
            self.assertNotEqual(first, failed)
            self.assertEqual(json.loads((failed / "run_status.json").read_text())["status"], "failed")
            with experiment_run(failed, "test", resume=True) as resumed:
                self.assertEqual(resumed, failed)
            with self.assertRaisesRegex(ValueError, "incomplete"), experiment_run(first, "test", resume=True):
                pass

    def test_cache_reuses_only_identical_content_and_parameters(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.focus_inspection.cache.version", return_value="test"),
        ):
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"fixture")
            calls = []

            def predictor(train, test, **kwargs):
                calls.append(1)
                return np.full(len(test), 0.6)

            cache = PredictionCache(root / "cache", checkpoint, predictor=predictor)
            train, test = samples().iloc[:6], samples().iloc[6:]
            args = dict(
                feature_columns=["signal"], model_path=checkpoint, device="cpu", random_state=42, n_estimators=4
            )
            cache(train, test, **args)
            cache(train, test, **args)
            cache(train.assign(signal=0.1), test, **args)
            cache(train, test, **{**args, "random_state": 43})
            self.assertEqual(len(calls), 3)
            self.assertEqual(cache.hits, 1)


if __name__ == "__main__":
    unittest.main()
