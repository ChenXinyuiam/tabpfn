import unittest

import numpy as np
import pandas as pd

from src.regression.zero_fd import positive_floor, zero_statistics


class ZeroTests(unittest.TestCase):
    def test_floor_uses_smallest_nonzero_magnitude_not_signed_minimum(self):
        context = pd.DataFrame({"u0": [-100., 0., .2], "u90": [-.01, 0., 50.]})
        original = context.copy(deep=True)
        adjusted, rows = positive_floor(context)
        self.assertAlmostEqual(adjusted.u0.iloc[1], .02)
        self.assertAlmostEqual(adjusted.u90.iloc[1], .001)
        self.assertEqual(adjusted.u0.iloc[0], -100)
        self.assertEqual(adjusted.u90.iloc[2], 50)
        self.assertEqual(rows[0]["replaced"], 1)
        pd.testing.assert_frame_equal(context, original)

    def test_all_zero_component_does_not_invent_a_threshold(self):
        context = pd.DataFrame({"u0": [0., 0.], "u90": [0., 0.]})
        adjusted, rows = positive_floor(context)
        pd.testing.assert_frame_equal(context, adjusted)
        self.assertTrue(all(row["replaced"] == 0 for row in rows))

    def test_floor_has_no_query_label_input(self):
        context = pd.DataFrame({"u0": [0., -1., .01], "u90": [0., 2., -.02],
                                "observed_000": [1000., 2000., 3000.]})
        first, _ = positive_floor(context)
        second, _ = positive_floor(context.assign(observed_000=-999999.))
        np.testing.assert_allclose(first[["u0", "u90"]], second[["u0", "u90"]])

    def test_zero_audit_distinguishes_interior_from_boundary(self):
        data = pd.DataFrame({"mesh_a": [0., .3, .3], "mesh_b": [.5, .3, .3], "mesh_c": [.5, .4, .4],
                             "u0": [0., 0., -1.], "u90": [0., 1., -2.]})
        rows = zero_statistics(data)
        self.assertEqual(rows[0]["zeros_near_mesh_boundary"], 1)
        self.assertEqual(rows[0]["zeros_interior"], 1)
        self.assertEqual(rows[1]["zeros_interior"], 0)


if __name__ == "__main__":
    unittest.main()
