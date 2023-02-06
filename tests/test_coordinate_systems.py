"""Tests functions in `coordinate_systems`."""

import unittest
from typing import Callable

import numpy as np

from aind_mri_utils import coordinate_systems as cs


class CoordinateSystemsTest(unittest.TestCase):
    """Tests functions in `coordinate_systems`."""

    test_coordinates = np.array(
        [[2.0, -3.0, 4.0], [-1.0, 4.0, -5.0], [0.0, 0.0, 0.0]]
    )
    target_coordinates = np.array(
        [[-2.0, 3.0, 4.0], [1.0, -4.0, -5.0], [0.0, 0.0, 0.0]]
    )

    def coordinate_helper_func(self, func: Callable) -> None:
        """Helper method that runs tests for lps_to_ras and ras_to_lps"""
        self.assertTrue(
            np.array_equal(
                func(self.test_coordinates), self.target_coordinates
            )
        )  # The obvious test
        self.assertTrue(
            np.array_equal(
                func(self.test_coordinates[0, :]),
                self.target_coordinates[0, :],
            )
        )  # vector

        # Test with ints
        int_test_data = self.test_coordinates.astype(int)
        int_target_data = self.target_coordinates.astype(int)
        int_transformed_test_data = func(int_test_data)
        self.assertTrue(
            np.array_equal(int_transformed_test_data, int_target_data)
        )
        self.assertTrue(
            int_target_data.dtype == int_transformed_test_data.dtype
        )

    def test_lps_to_ras(self) -> None:
        """Tests that the `lps_to_ras` function works as intended."""
        self.coordinate_helper_func(cs.lps_to_ras)

    def test_ras_to_lps(self) -> None:
        """Tests that the `ras_to_lps` function works as intended."""
        self.coordinate_helper_func(cs.ras_to_lps)


if __name__ == "__main__":
    unittest.main()
