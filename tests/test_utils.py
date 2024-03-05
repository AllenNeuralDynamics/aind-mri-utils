"""Tests functions in `utils`."""

import unittest

import numpy as np

from aind_mri_utils import utils as ut


class UtilsTest(unittest.TestCase):
    """Tests functions in `utls`."""

    cross_product_sets = [
        (np.array([0, 1, 1]), np.array([1, 0, 0]), np.array([0, 1, -1])),
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
        (np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, -1])),
        (np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 0])),
    ]

    test_match_arr = np.arange(0, 9).reshape(3, 3)
    test_proj_vec = np.array([0.5, 0.2, 0.7])

    signed_angle_sets = [
        (
            np.array([1, 0, 0]),
            np.array([np.pi, np.pi, 0]),
            np.array([0, 0, 1]),
            0.785398,
        ),
        (
            np.array([np.pi, np.pi, 0]),
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
            -0.785398,
        ),
        (np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), 0.0),
    ]

    def test_skew_symmetric_cross_product_matrix(self) -> None:
        for a, b, c in self.cross_product_sets:
            self.assertTrue(
                np.allclose(ut.skew_symmetric_cross_product_matrix(a) @ b, c)
            )

    def test_norm_vec(self) -> None:
        with self.assertRaises(ValueError):
            ut.norm_vec(np.array([0, 0, 0]))
        self.assertTrue(
            np.allclose(ut.norm_vec(np.array([2, 0, 0])), np.array([1, 0, 0]))
        )

    def test_find_indices_equal_to(self) -> None:
        self.assertTrue(
            np.array_equal(
                ut.find_indices_equal_to(self.test_match_arr, 1),
                np.array([[0, 1]]),
            )
        )
        self.assertEqual(
            len(ut.find_indices_equal_to(self.test_match_arr, 9)), 0
        )
        B = np.copy(self.test_match_arr)
        B[1, :] = 9
        self.assertTrue(
            np.array_equal(
                ut.find_indices_equal_to(B, 9),
                np.array([[1, 0], [1, 1], [1, 2]]),
            )
        )

    def test_vector_rejection(self) -> None:
        for i in range(0, 3):
            ans = np.copy(self.test_proj_vec)
            ans[i] = 0
            basis_vec = np.zeros((3,))
            basis_vec[i] = 1
            self.assertTrue(
                np.allclose(
                    ut.vector_rejection(self.test_proj_vec, basis_vec),
                    ans,
                )
            )

    def test_mask_arr_by_annotation(self) -> None:
        ans = ut.mask_arr_by_annotations(
            self.test_match_arr, self.test_match_arr, [9]
        )
        self.assertTrue(np.allclose(ans, np.zeros((3, 3))))
        ans = ut.mask_arr_by_annotations(
            self.test_match_arr, self.test_match_arr, [1]
        )
        self.assertTrue(np.isclose(self.test_match_arr[0, 1], ans[0, 1]))
        modified_ans = np.copy(ans)
        modified_ans[0, 1] = 0
        self.assertTrue(np.allclose(modified_ans, np.zeros((3, 3))))

    def test_signed_angle(self) -> None:
        for a, b, n, ans in self.signed_angle_sets:
            received = ut.signed_angle_rh(a, b, n)
            self.assertTrue(np.isclose(received, ans))
            self.assertTrue(np.isclose(ans, ut.signed_angle_lh(b, a, n)))

    def test_get_first_pca_axis(self) -> None:
        pts = np.array([[1, 2], [3, 4], [5, 6]])
        expected_first_pc = np.array([0.70710678, 0.70710678])
        self.assertTrue(
            np.allclose(
                ut.get_first_pca_axis(pts),
                expected_first_pc,
            )
        )

    def test_unsigned_angle(self) -> None:
        for a, b, n, ans in self.signed_angle_sets:
            received = ut.unsigned_angle(a, b)
            self.assertTrue(np.isclose(received, abs(ans)))


if __name__ == "__main__":
    unittest.main()
