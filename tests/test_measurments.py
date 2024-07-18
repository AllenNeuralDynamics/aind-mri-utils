# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:54:02 2023
"""

import unittest

import numpy as np

from aind_mri_utils import measurement


class MeasurementTest(unittest.TestCase):
    def test_find_circle(self) -> None:
        x = np.array([1, 0, -1, 0])
        y = np.array([0, 1, 0, -1])

        """Tests that circle finder is working correctly."""
        xc, yc, radius = measurement.find_circle(x, y)

        self.assertEqual(xc, 0)
        self.assertEqual(yc, 0)
        self.assertEqual(radius, 1)

    def test_closest_point_on_two_lines(self):
        # Test with parallel lines
        P1 = np.array([0, 0, 0])
        V1 = np.array([1, 0, 0])
        P2 = np.array([0, 1, 0])
        V2 = np.array([1, 0, 0])

        r1, r2 = measurement.closet_points_on_two_lines(P1, V1, P2, V2)
        self.assertTrue(np.array_equal(P1, r1))
        self.assertTrue(np.array_equal(P2, r2))

        # Test with orthogonal lines
        P1 = np.array([0, 0, 0])
        V1 = np.array([1, 0, 0])
        P2 = np.array([0, 1, 0])
        V2 = np.array([0, 0, 1])
        r1, r2 = measurement.closet_points_on_two_lines(P1, V1, P2, V2)
        self.assertTrue(np.array_equal(P1, r1))
        self.assertTrue(np.array_equal(P2, r2))

        # Test with intersecting lines
        P1 = np.array([0, 0, 0])
        V1 = np.array([1, 0, 0])
        P2 = np.array([0, 1, 0])
        V2 = np.array([0, 1, 0])
        r1, r2 = measurement.closet_points_on_two_lines(P1, V1, P2, V2)
        self.assertTrue(np.array_equal(P1, r1))
        self.assertTrue(np.array_equal(P1, r2))

    def test_find_line_eig(self):
        # Find the first eigenvector of a line with no variance
        points = np.tile(np.array([1, 0, 0]).T, 12).reshape((12, 3))
        ln, mn = measurement.find_line_eig(points)
        self.assertTrue(np.array_equal(ln, np.array([1, 0, 0])))
        self.assertTrue(np.array_equal(mn, np.array([1, 0, 0])))

    def test_angle(self):
        # Test code for angle between two vectors
        x = measurement.angle(np.array([1, 0, 0]), np.array([0, 1, 0]))
        self.assertEqual(x, 90)
        x = measurement.angle(np.array([1, 0, 0]), np.array([1, 0, 0]))
        self.assertEqual(x, 0)


if __name__ == "__main__":
    unittest.main()
