# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:30:09 2023

@author: yoni.browning
"""

import unittest
from aind_mri_utils import rotations
import numpy as np


class MessageHandlerTest(unittest.TestCase):
    def test_define_euler_rotation(self) -> None:
        R = rotations.define_euler_rotation(0, 0, 0, degrees=True)
        self.assertTrue(np.all(R.as_matrix() == np.eye(3)))

    def test_rotate_about(self) -> None:
        # Test 1: no rotation
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(0, 0, 0, degrees=True)
        X = rotations.rotate_about(pt, R, pivot)
        self.assertTrue(np.all(X == pt))
        # Test 2: 360 rotation
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(360, 360, 360, degrees=True)
        X = rotations.rotate_about(pt, R, pivot)
        self.assertTrue(np.all(X == pt))
        # Test 3: Numerical Error
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(
            np.pi * 2, np.pi * 2, np.pi * 2, degrees=False
        )
        X = rotations.rotate_about(pt, R, pivot)
        self.assertFalse(np.all(X == pt))
        self.assertTrue(np.all(X - pt < 0.02))
        # Test4: with translation
        pt = np.array([1, 2, 3])
        pivot = np.array([2, 3, 4])
        R = rotations.define_euler_rotation(360, 360, 360, degrees=True)
        translate = np.array((1, 1, 1))
        X = rotations.rotate_about_and_translate(
            pt, R, pivot, np.array(translate)
        )
        self.assertTrue(np.all(X == pt - translate))


if __name__ == "__main__":
    unittest.main()
