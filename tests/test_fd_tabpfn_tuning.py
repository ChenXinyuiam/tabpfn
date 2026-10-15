import unittest

import numpy as np
import pandas as pd

from src.regression.evaluation import FEATURE_COLUMNS
from src.regression.tune_tabpfn_fd import RECIPES, factor, features, inference_config
from tests.test_fd_optimization import sample_data


class TabPFNTuningTests(unittest.TestCase):
    def test_augmentation_keeps_original_features(self):
        data = sample_data()
        result = features(data, RECIPES[3])
        pd.testing.assert_frame_equal(result[FEATURE_COLUMNS], data[FEATURE_COLUMNS])
        self.assertGreater(result.shape[1], len(FEATURE_COLUMNS))
        pd.testing.assert_frame_equal(result, features(data.assign(target=999.0), RECIPES[3]))

    def test_target_scaling_roundtrip_and_positive_inputs(self):
        data = sample_data()
        for recipe in RECIPES:
            values = factor(data, recipe)
            np.testing.assert_allclose(data.target / values * values, data.target)
        with self.assertRaises(ValueError):
            factor(data.assign(re=0), RECIPES[4])

    def test_continuous_dimensions_are_not_auto_categories(self):
        from tabpfn.preprocessing.datamodel import FeatureModality
        from tabpfn.preprocessing.modality_detection import detect_feature_modalities

        x = np.tile([[400.0, 10.0], [750.0, 20.0], [850.0, 30.0]], (50, 1))
        common = dict(X=x, feature_names=["side", "radius"], min_samples_for_inference=100, max_unique_for_category=30)
        default = detect_feature_modalities(**common, min_unique_for_numerical=4)
        numeric = detect_feature_modalities(
            **common, min_unique_for_numerical=inference_config(RECIPES[2])["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"]
        )
        self.assertTrue(all(f.modality == FeatureModality.CATEGORICAL for f in default.features))
        self.assertTrue(all(f.modality == FeatureModality.NUMERICAL for f in numeric.features))


if __name__ == "__main__":
    unittest.main()
