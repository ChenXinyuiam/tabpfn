"""Physical grouping and nested probability-calibration regression tests."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.focus_inspection.audit_focus import run as audit_run
from src.focus_inspection.cache import PredictionCache
from src.focus_inspection.calibrate_focus import nested_calibration
from src.focus_inspection.calibration import apply_calibration, fit_sigmoid
from src.focus_inspection.evaluation import make_grouped_folds
from src.focus_inspection.geometry_integrity import attach_geometry_audit, geometry_issues, physical_geometry_id
from src.focus_inspection.prediction_safety import prediction_diagnostics


def triangle():
    return {
        "bc": 750.0,
        "ab": 850.0,
        "ac": 400.0,
        "h": 400.0,
        "angle_a": 61.93,
        "angle_b": 28.07,
        "angle_c": 90.0,
        "perimeter": 2000.0,
        "area": 150000.0,
    }


class GeometryCalibrationTests(unittest.TestCase):
    def test_strict_audit_preserves_error_document_without_publishing_success(self):
        frame = attach_geometry_audit(
            pd.DataFrame(
                [
                    {
                        **triangle(),
                        "ac": 100.0,
                        "source_file": "sample.xlsx",
                        "geometry_group": "sample",
                        "R": 10.0,
                        "re": 6.0,
                    }
                ]
            )
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.redirect_stdout(io.StringIO()),
            patch("src.focus_inspection.audit_focus.build_features"),
            patch("src.focus_inspection.audit_focus.build_feature_dataset", return_value=frame),
        ):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "review flags"):
                audit_run(root, root, strict=True)
            documents = list(root.glob("runs/*/issues.md"))
            self.assertEqual(len(documents), 1)
            self.assertIn("height_exceeds_side", documents[0].read_text(encoding="utf-8"))
            self.assertFalse((root / "latest.json").exists())

    def test_valid_triangle_and_mirror_have_same_physical_identity(self):
        row = triangle()
        reverse = {**row, "ab": row["ac"], "ac": row["ab"], "angle_b": row["angle_c"], "angle_c": row["angle_b"]}
        self.assertEqual(geometry_issues(row), [])
        self.assertEqual(geometry_issues(reverse), [])
        self.assertEqual(physical_geometry_id(row), physical_geometry_id(reverse))

    def test_geometry_contradictions_are_flagged_without_correction(self):
        row = {**triangle(), "ac": 100.0, "area": 15000.0, "perimeter": 9999.0}
        saved = row.copy()
        issues = geometry_issues(row)
        self.assertIn("height_exceeds_side", issues)
        self.assertIn("perimeter_disagrees_with_sides", issues)
        self.assertIn("area_disagrees_with_base_height", issues)
        self.assertEqual(row, saved)

    def test_different_filenames_cannot_split_identical_physical_geometry(self):
        frame = pd.DataFrame([{**triangle(), "geometry_group": "750-400"}, {**triangle(), "geometry_group": "renamed"}])
        result = attach_geometry_audit(frame)
        self.assertEqual(result.geometry_group.nunique(), 1)
        self.assertEqual(result.named_geometry_group.nunique(), 2)
        self.assertEqual(result.physical_geometry_id.nunique(), 1)

    def test_physical_overlap_is_rejected_even_when_named_groups_differ(self):
        rows = []
        for group in range(3):
            for label in (0, 1):
                rows.append(
                    {
                        "source_file": f"{group}.xlsx",
                        "summary_id": str(group),
                        "geometry_group": str(group),
                        "R": 10,
                        "re": label + 1,
                        "focus_label": label,
                        "focus_label_name": str(label),
                        "physical_geometry_id": "same",
                    }
                )
        with self.assertRaisesRegex(RuntimeError, "Physical geometry leakage"):
            make_grouped_folds(pd.DataFrame(rows), n_splits=3, random_state=42)

    def test_prediction_geometry_detection_uses_physical_id(self):
        train = pd.DataFrame(
            {
                "source_file": ["old"],
                "R": [10],
                "re": [6],
                "geometry_group": ["old"],
                "physical_geometry_id": ["same"],
                "value": [1.0],
            }
        )
        test = train.assign(source_file="new", geometry_group="new")
        flags = prediction_diagnostics(train, test, np.array([0.8]), 0.5, ["value"])
        self.assertTrue(flags.seen_geometry.item())
        self.assertFalse(flags.seen_sample_key.item())

    def test_sigmoid_is_serializable_and_handles_exact_zero_and_one(self):
        p = np.array([0, 0.1, 0.3, 0.7, 0.9, 1])
        state = fit_sigmoid(p, np.array([0, 0, 0, 1, 1, 1]))
        result = apply_calibration(p, state)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        self.assertTrue((np.diff(result) > 0).all())
        np.testing.assert_array_equal(apply_calibration(p, None), p)

    def test_invalid_probability_or_calibration_state_is_rejected(self):
        for p in ([np.nan], [-0.1], [1.1], [[0.5]]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                apply_calibration(p, None)
        with self.assertRaisesRegex(ValueError, "both classes"):
            fit_sigmoid([0.2, 0.3], [0, 0])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            apply_calibration([0.2], {"method": "unknown"})

    def test_outer_test_labels_do_not_change_their_calibrated_predictions(self):
        rows = []
        for group in range(4):
            for label in (0, 1):
                rows.append(
                    {
                        "source_file": f"{group}.xlsx",
                        "summary_id": str(group),
                        "geometry_group": str(group),
                        "R": 10,
                        "re": label + 1,
                        "focus_label": label,
                        "focus_label_name": str(label),
                        "signal": 0.2 + 0.6 * label + 0.01 * group,
                    }
                )
        data = pd.DataFrame(rows)
        calls = []

        def predictor(train, test, **kwargs):
            self.assertNotIn("focus_label", test)
            self.assertFalse(set(train.geometry_group) & set(test.geometry_group))
            calls.append(1)
            return test.signal.to_numpy()

        with contextlib.redirect_stdout(io.StringIO()):
            first, metrics, audit = nested_calibration(
                data, model_path=Path("unused"), device="cpu", seeds=(42,), inner_splits=2, predictor=predictor
            )
            changed = data.copy()
            changed.loc[changed.geometry_group.eq("0"), "focus_label"] = [1, 0]
            second, _, _ = nested_calibration(
                changed, model_path=Path("unused"), device="cpu", seeds=(42,), inner_splits=2, predictor=predictor
            )
        mask = first.geometry_group.eq("0")
        np.testing.assert_array_equal(first.loc[mask, "proba_sigmoid"], second.loc[mask, "proba_sigmoid"])
        self.assertEqual(len(calls), 24)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(audit["inner_folds"]), 8)

    def test_read_through_cache_does_not_fit_or_modify_old_cache(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.focus_inspection.cache.version", return_value="test"),
        ):
            root = Path(temporary)
            checkpoint = root / "model"
            checkpoint.write_bytes(b"fixture")
            train = pd.DataFrame({"x": [0.0, 1.0], "focus_label": [0, 1]})
            test = pd.DataFrame({"x": [0.5]})
            args = dict(feature_columns=["x"], model_path=checkpoint, device="cpu", random_state=42, n_estimators=4)
            first = PredictionCache(root / "old", checkpoint, predictor=lambda *a, **k: np.array([0.7]))
            first(train, test, **args)
            before = {p.name: p.read_bytes() for p in (root / "old").glob("*")}
            second = PredictionCache(
                root / "new",
                checkpoint,
                predictor=lambda *a, **k: self.fail("unexpected fit"),
                read_through=[root / "old"],
            )
            np.testing.assert_array_equal(second(train, test, **args), [0.7])
            self.assertEqual(second.hits, 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "old").glob("*")})


if __name__ == "__main__":
    unittest.main()
