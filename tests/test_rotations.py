# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:30:09 2023

@author: yoni.browning
"""

import unittest

import numpy as np

from aind_mri_utils import rotations


class RotationsTest(unittest.TestCase):
    def test_define_euler_rotation(self) -> None:
        R = rotations.define_euler_rotation(0, 0, 0, degrees=True)
        self.assertTrue(np.array_equal(R.as_matrix(), np.eye(3)))

    def test_rotate_about(self) -> None:
        # Test 1: no rotation
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(0, 0, 0, degrees=True)
        X = rotations.rotate_about(pt, R, pivot)
        self.assertTrue(np.array_equal(X, pt))
        # Test 2: 360 rotation
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(360, 360, 360, degrees=True)
        X = rotations.rotate_about(pt, R, pivot)
        self.assertTrue(np.array_equal(X, pt))
        # Test 3: Numerical Error
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(
            np.pi * 2, np.pi * 2, np.pi * 2, degrees=False
        )
        X = rotations.rotate_about(pt, R, pivot)
        self.assertFalse(np.array_equal(X, pt))
        self.assertTrue(np.all(X - pt < 0.02))
        # Test4: with translation
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(360, 360, 360, degrees=True)
        translate = np.array((1, 1, 1))
        X = rotations.rotate_about_and_translate(
            pt, R, pivot, np.array(translate)
        )
        self.assertTrue(np.array_equal(X, pt + translate))
        # Test 5: More than one point
        pt = np.array([[1, 2, 3], [1, 2, 3]])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(0, 0, 0, degrees=True)
        X = rotations.rotate_about(pt, R, pivot)
        self.assertTrue(X.shape[1] == pt.shape[1])
        self.assertTrue(np.array_equal(X[0, :], X[1, :]))
        # Test 6: non-identity
        pt = np.array([0, 1, 0], dtype=np.float64)
        R = rotations.define_euler_rotation(90, 0, 0, degrees=True)
        pivot = np.array([0, 0, 0])
        X = rotations.rotate_about(pt, R, pivot)
        self.assertTrue(np.all(X - np.array([0.0, 0.0, 1.0]) < 0.00001))

    def test_scipy_rotation_to_sitk(self) -> None:
        R = rotations.define_euler_rotation(90, 0, 0)
        center = np.array((-1, 0, 0))
        translation = np.array((1, 0, 0))
        trans = rotations.scipy_rotation_to_sitk(
            R, center=center, translation=translation
        )
        self.assertTrue(np.array_equal(trans.GetTranslation(), translation))
        self.assertTrue(np.array_equal(trans.GetFixedParameters(), center))
        self.assertTrue(
            np.array_equal(
                R.as_matrix().reshape((9,)),
                np.array(trans.GetParameters()[:9]),
            )
        )


if __name__ == "__main__":
    unittest.main()
