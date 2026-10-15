"""Prediction overlap, cache stability and descriptive reporting regressions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.focus_inspection.cache import PredictionCache
from src.focus_inspection.prediction_safety import prediction_diagnostics
from src.focus_inspection.prediction_statistics import summarize_predictions


def features():
    return pd.DataFrame(
        {
            "source_file": ["a.xlsx", "b.xlsx"],
            "R": [10, 10],
            "re": [6, 12],
            "source_sha256": ["hash_a", "hash_b"],
            "geometry_group": ["g", "g"],
            "signal": [0.1, 0.9],
            "focus_label": [0, 1],
        }
    )


def predictions():
    return pd.DataFrame(
        {
            "source_file": ["a.xlsx", "a.xlsx", "b.xlsx"],
            "R": [10, 10, 20],
            "re": [6, 12, 6],
            "source_sha256": ["hash_a", "hash_a", "hash_b"],
            "geometry_group": ["g", "g", "h"],
            "focus_probability": [0.2, 0.8, 0.5],
            "threshold": [0.5] * 3,
            "focus_prediction": ["UF", "F", "UF"],
            "decision": ["UF", "REVIEW", "REVIEW"],
            "needs_review": [False, True, True],
            "known_training_sample": [False, True, False],
            "data_quality_warning": [False, False, True],
            "overlap_status": ["unseen_geometry", "training_content", "unseen_geometry"],
            "review_reasons": [
                "",
                "training_sample_not_independent",
                "undefined_boundary;near_decision_threshold;undefined_boundary",
            ],
            "boundary_distance_abs": [150.0, 0.0, np.nan],
            "boundary_midpoint_x": [120.0, 0.0, np.nan],
            "rule_available": [True, True, False],
            "rule_disagrees_with_model": [False, False, False],
        }
    )


class FocusPredictionTests(unittest.TestCase):
    def test_renamed_training_content_is_not_new_data(self):
        train = features()
        test = train.iloc[[0]].assign(source_file="renamed.xlsx")
        result = prediction_diagnostics(train, test, np.array([0.2]), 0.5, ["signal"])
        self.assertTrue(result.known_training_sample.item())
        self.assertFalse(result.seen_sample_key.item())
        self.assertEqual(result.overlap_status.item(), "training_content")
        self.assertIn("training_sample_not_independent", result.review_reasons.item())

    def test_duplicate_batch_content_is_flagged_without_dropping_rows(self):
        train = features()
        test = pd.concat([train.iloc[[0]], train.iloc[[0]].assign(source_file="copy.xlsx")], ignore_index=True)
        result = prediction_diagnostics(train, test, np.array([0.2, 0.2]), 0.5, ["signal"])
        self.assertEqual(len(result), 2)
        self.assertTrue(result.duplicate_input_content.all())

    def test_missing_hashes_are_not_evidence_of_training_overlap(self):
        train = features().assign(source_sha256=np.nan)
        result = prediction_diagnostics(train, train.iloc[[0]], np.array([0.2]), 0.5, ["signal"])
        self.assertFalse(result.known_training_sample.item())

    def test_diagnostics_reject_duplicate_keys_and_misaligned_probabilities(self):
        train = features()
        with self.assertRaisesRegex(ValueError, "duplicate sample keys"):
            prediction_diagnostics(train, pd.concat([train, train]), np.array([0.2] * 4), 0.5, ["signal"])
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            prediction_diagnostics(train, train, np.array([0.2]), 0.5, ["signal"])

    def test_prediction_statistics_preserve_zero_and_missing(self):
        summary, tables = summarize_predictions(predictions())
        self.assertEqual(summary["predicted_f"], 1)
        self.assertEqual(summary["predicted_uf"], 2)
        self.assertEqual(summary["review_samples"], 2)
        self.assertEqual(summary["width_available_samples"], 2)
        self.assertEqual(summary["width_median_um"], 75)
        self.assertEqual(tables["boundary_plot_counts"].samples.sum(), 2)
        self.assertIsNone(summary["accuracy"])
        self.assertFalse(summary["evaluation"])
        for name in ["by_file", "by_radius", "by_reynolds", "by_geometry", "by_overlap"]:
            self.assertEqual(tables[name].samples.sum(), 3)
            self.assertEqual(tables[name].predicted_f.sum(), 1)
        reasons = tables["review_reasons"].set_index("reason")
        self.assertEqual(reasons.loc["undefined_boundary", "samples"], 1)
        self.assertAlmostEqual(reasons.loc["undefined_boundary", "fraction_of_all_samples"], 1 / 3)

    def test_duplicate_content_counts_do_not_silently_change_sample_denominator(self):
        data = predictions()
        duplicate = data.iloc[[0]].assign(source_file="copy.xlsx")
        summary, _ = summarize_predictions(pd.concat([data, duplicate], ignore_index=True))
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["verified_content_sample_units"], 3)
        self.assertEqual(summary["duplicate_content_extra_samples"], 1)

    def test_empty_and_inconsistent_prediction_tables_fail(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            summarize_predictions(predictions().iloc[:0])
        data = predictions()
        data.loc[2, "focus_prediction"] = "F"
        with self.assertRaisesRegex(ValueError, "saved threshold"):
            summarize_predictions(data)
        data = predictions()
        data.loc[1, "decision"] = "F"
        with self.assertRaisesRegex(ValueError, "saved flags"):
            summarize_predictions(data)

    def test_cache_cold_and_warm_probabilities_have_identical_precision(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.focus_inspection.cache.version", return_value="test"),
        ):
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"fixture")
            cache = PredictionCache(
                root / "cache", checkpoint, predictor=lambda *a, **k: np.array([0.9], dtype=np.float32)
            )
            args = dict(feature_columns=["signal"], model_path=checkpoint, device="cpu", random_state=42)
            cold = cache(features(), features().iloc[[0]], **args)
            warm = cache(features(), features().iloc[[0]], **args)
            np.testing.assert_array_equal(cold, warm)
            self.assertEqual(cold.dtype, warm.dtype)
            self.assertEqual(cache.hits, 1)

    def test_invalid_model_probabilities_are_not_cached(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.focus_inspection.cache.version", return_value="test"),
        ):
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"fixture")
            cache = PredictionCache(root / "cache", checkpoint, predictor=lambda *a, **k: np.array([np.nan]))
            with self.assertRaisesRegex(ValueError, "cache was not written"):
                cache(
                    features(),
                    features().iloc[[0]],
                    feature_columns=["signal"],
                    model_path=checkpoint,
                    device="cpu",
                    random_state=42,
                )
            self.assertFalse(list((root / "cache").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
