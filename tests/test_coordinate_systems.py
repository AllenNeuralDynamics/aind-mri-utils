"""Tests functions in `coordinate_systems`."""

import unittest

import numpy as np

from aind_mri_utils import coordinate_systems as cs


class CoordinateSystemsTest(unittest.TestCase):
    """Tests functions in `coordinate_systems`."""

    test_coordinates = np.array(
        [[2.0, -3.0, 4.0], [-1.0, 4.0, -5.0], [0.0, 0.0, 0.0]]
    )

    def coordinate_helper_func(self, src, dst, expected) -> None:
        """Helper method for test_convert_coordinate_systems"""
        self.assertTrue(
            np.array_equal(
                cs.convert_coordinate_system(self.test_coordinates, src, dst),
                expected,
            )
        )  # The obvious test
        self.assertTrue(
            np.array_equal(
                cs.convert_coordinate_system(
                    self.test_coordinates[[0], :], src, dst
                ),
                expected[[0], :],
            )
        )  # different shape

        # Test with ints
        int_test_data = self.test_coordinates.astype(int)
        int_target_data = expected.astype(int)
        int_transformed_test_data = cs.convert_coordinate_system(
            int_test_data, src, dst
        )
        self.assertTrue(
            np.array_equal(int_transformed_test_data, int_target_data)
        )
        self.assertTrue(
            int_target_data.dtype == int_transformed_test_data.dtype
        )

    def test_find_coordinate_perm_and_flips(self) -> None:
        """Tests for find_coordinate_perm_and_flips"""
        perm, direction = cs.find_coordinate_perm_and_flips("RAS", "LPI")
        self.assertTrue(
            np.array_equal(perm, [0, 1, 2])
            and np.array_equal(direction, [-1, -1, -1])
        )
        perm, direction = cs.find_coordinate_perm_and_flips("ras", "LPI")
        self.assertTrue(
            np.array_equal(perm, [0, 1, 2])
            and np.array_equal(direction, [-1, -1, -1])
        )
        perm, direction = cs.find_coordinate_perm_and_flips("RAS", "RAS")
        self.assertTrue(
            np.array_equal(perm, [0, 1, 2])
            and np.array_equal(direction, [1, 1, 1])
        )
        perm, direction = cs.find_coordinate_perm_and_flips("ASR", "RAS")
        self.assertTrue(
            np.array_equal(perm, [2, 0, 1])
            and np.array_equal(direction, [1, 1, 1])
        )
        perm, direction = cs.find_coordinate_perm_and_flips("PIL", "RAS")
        self.assertTrue(
            np.array_equal(perm, [2, 0, 1])
            and np.array_equal(direction, [-1, -1, -1])
        )
        perm, direction = cs.find_coordinate_perm_and_flips("PLS", "LPS")
        self.assertTrue(
            np.array_equal(perm, [1, 0, 2])
            and np.array_equal(direction, [1, 1, 1])
        )
        perm, direction = cs.find_coordinate_perm_and_flips("PRS", "LPS")
        self.assertTrue(
            np.array_equal(perm, [1, 0, 2])
            and np.array_equal(direction, [-1, 1, 1])
        )
        self.assertRaisesRegex(
            ValueError,
            "Inputs should be the same length",
            cs.find_coordinate_perm_and_flips,
            "RA",
            "RAS",
        )
        self.assertRaisesRegex(
            ValueError,
            "Source axis 'R' not unique",
            cs.find_coordinate_perm_and_flips,
            "RRS",
            "RAS",
        )
        self.assertRaisesRegex(
            ValueError,
            "Source axis 'L' not unique",
            cs.find_coordinate_perm_and_flips,
            "RLS",
            "RAS",
        )
        self.assertRaisesRegex(
            ValueError,
            "Destination axis 'L' not unique",
            cs.find_coordinate_perm_and_flips,
            "RAS",
            "RLS",
        )
        self.assertRaisesRegex(
            ValueError,
            "Source direction 'D' not in R/L, A/P, or I/S",
            cs.find_coordinate_perm_and_flips,
            "RAD",
            "RAS",
        )
        self.assertRaisesRegex(
            ValueError,
            "Destination direction 'D' not in R/L, A/P, or I/S",
            cs.find_coordinate_perm_and_flips,
            "RAS",
            "RAD",
        )
        self.assertRaisesRegex(
            ValueError,
            "Destination direction 'S' has no match in source directions 'RA'",
            cs.find_coordinate_perm_and_flips,
            "RA",
            "RS",
        )

    def test_convert_coordinate_system(self):
        """Tests for convert_coordinate_system"""
        expected = np.array([-1, -1, -1]) * self.test_coordinates[:, [0, 1, 2]]
        self.coordinate_helper_func("RAS", "LPI", expected)
        expected = np.array([-1, -1, -1]) * self.test_coordinates[:, [1, 0, 2]]
        self.coordinate_helper_func("RAS", "PLI", expected)
        expected = self.test_coordinates[:, [1, 2, 0]]
        self.coordinate_helper_func("RAS", "ASR", expected)


if __name__ == "__main__":
    unittest.main()
