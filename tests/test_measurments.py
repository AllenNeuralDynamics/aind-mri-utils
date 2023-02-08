# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:54:02 2023

@author: yoni.browning
"""

import unittest

import numpy as np
from aind_mri_utils import measurement


class MeasurmentTest(unittest.TestCase):
    def test_find_circle_center(self) -> None:
        x = np.array([1, 0, -1, 0])
        y = np.array([0, 1, 0, -1])

        """Tests that circle finder is working correctly."""
        xc, yc, radius = measurement.find_circle_center(x, y)

        self.assertEqual(xc, 0)
        self.assertEqual(yc, 0)
        self.assertEqual(radius, 1)


if __name__ == "__main__":
    unittest.main()
