"""Unique-sample accounting and geometry-cluster uncertainty checks."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.focus_inspection.review_focus import (
    geometry_accuracy_intervals,
    run,
    summarize_geometries,
    summarize_samples,
    validate_predictions,
)
from utils.provenance import file_sha256, write_json


def predictions():
    rows = []
    for seed in [42, 2026, 3407]:
        for sample in range(4):
            label = sample % 2
            probability = 0.95 if sample == 0 else (0.2 if sample == 1 and seed == 42 else (0.9 if label else 0.1))
            rows.append(
                {
                    "source_file": f"sample{sample}.xlsx",
                    "R": 10,
                    "re": 6,
                    "seed": seed,
                    "geometry_group": f"g{sample // 2}",
                    "cv_fold": sample // 2 + 1,
                    "focus_label": label,
                    "proba_raw": probability,
                    "pred_raw": int(probability > 0.5),
                    "geometry_consistent": sample != 3,
                    "geometry_issues": "fixture" if sample == 3 else "",
                    "boundary_distance_abs": 50 if label else 150,
                    "boundary_midpoint_x": 123.5,
                }
            )
    return pd.DataFrame(rows)


class FocusReviewTests(unittest.TestCase):
    def test_review_counts_unique_samples_not_repeated_predictions(self):
        data = predictions()
        original = data.copy(deep=True)
        result = summarize_samples(data, "raw")
        self.assertEqual(len(result), 4)
        self.assertEqual(result.error_seeds.gt(0).sum(), 2)
        self.assertEqual(result.persistent_error.sum(), 1)
        self.assertEqual(result.decision_unstable.sum(), 1)
        self.assertEqual(result.geometry_warning.sum(), 1)
        self.assertEqual(result.iloc[0].source_file, "sample0.xlsx")
        self.assertTrue(result.reviewed_label.eq("").all())
        pd.testing.assert_frame_equal(data, original)

    def test_missing_and_duplicate_seed_rows_are_rejected(self):
        data = predictions()
        with self.assertRaisesRegex(ValueError, "complete seed coverage"):
            summarize_samples(data.iloc[1:], "raw")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_samples(pd.concat([data, data.iloc[[0]]]), "raw")

    def test_inconsistent_sample_metadata_is_rejected(self):
        for column, value in [("focus_label", 1), ("boundary_midpoint_x", 999), ("geometry_consistent", False)]:
            with self.subTest(column=column):
                data = predictions()
                data.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "metadata changed"):
                    validate_predictions(data, "raw")

    def test_invalid_probabilities_labels_and_unknown_model_are_rejected(self):
        for column, value in [("proba_raw", np.inf), ("focus_label", 2), ("pred_raw", -1), ("R", 0), ("cv_fold", 0)]:
            with self.subTest(column=column):
                data = predictions()
                data.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_predictions(data, "raw")
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_predictions(predictions(), "absent")

    def test_repeating_identical_seeds_does_not_tighten_cluster_intervals(self):
        data = predictions()
        sample = summarize_samples(data, "raw")
        groups = summarize_geometries(data, sample, "raw")
        old = geometry_accuracy_intervals(groups)
        repeated = pd.concat([data, data.assign(seed=data.seed + 10000)])
        unique = summarize_samples(repeated, "raw")
        new = geometry_accuracy_intervals(summarize_geometries(repeated, unique, "raw"))
        self.assertEqual(old, new)
        self.assertAlmostEqual(old["sample_weighted_accuracy"], 8 / 12)

    def test_sample_weighted_and_geometry_macro_accuracy_are_distinct(self):
        groups = pd.DataFrame({"correct_predictions": [1, 0], "predictions": [1, 3]})
        result = geometry_accuracy_intervals(groups)
        self.assertEqual(result["sample_weighted_accuracy"], 0.25)
        self.assertEqual(result["geometry_macro_accuracy"], 0.5)
        with self.assertRaisesRegex(ValueError, "at least two"):
            geometry_accuracy_intervals(groups.iloc[[0]])

    def test_report_preserves_source_and_escapes_html(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source = root / "evaluation"
            source.mkdir()
            data = predictions()
            data.loc[data.source_file.eq("sample0.xlsx"), "source_file"] = "<script>alert(1)</script>.xlsx"
            data.to_csv(source / "oof_predictions.csv", index=False)
            write_json(source / "run_status.json", {"status": "complete", "kind": "focus-calibrate"})
            before = file_sha256(source / "oof_predictions.csv")
            out = run(source, root / "review", model="raw")
            self.assertEqual(before, file_sha256(source / "oof_predictions.csv"))
            document = (out / "gallery.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", document)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)
            result = json.loads((out / "review_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["unique_samples"], 4)
            self.assertEqual(result["model_fits"], 0)
            self.assertFalse(result["labels_modified"])
            self.assertTrue((root / "review/latest.json").is_file())

    def test_incomplete_evaluation_does_not_publish_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "run_status.json", {"status": "running", "kind": "focus-calibrate"})
            with self.assertRaisesRegex(ValueError, "completed"):
                run(root, root / "out", model="raw")
            self.assertFalse((root / "out/latest.json").exists())


if __name__ == "__main__":
    unittest.main()
