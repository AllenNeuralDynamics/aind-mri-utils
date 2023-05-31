import unittest

import numpy as np

from aind_mri_utils.optimization import (
    append_ones_column,
    get_headframe_hole_lines,
    optimize_transform_labeled_lines,
    optimize_transform_labeled_lines_with_plane,
)


class OptimaizationTest(unittest.TestCase):
    def test_get_headframe_hole_lines(self) -> None:
        """
        Tests for get_headframe_hole_lines
        """
        ant_hrz_hole_pts = np.array([[-6.34, 0, 2.5], [-6.34, 6.5, 2.5]])
        pts1, _, names = get_headframe_hole_lines(
            version=0.1,
            insert_underscores=False,
            coordinate_system="LPS",
            return_plane=False,
        )

        self.assertTrue(np.array_equal(pts1[0, :], ant_hrz_hole_pts[0, :]))
        self.assertFalse("_" in names[0])
        self.assertTrue(len(names) == 4)

        pts1, pts2, names = get_headframe_hole_lines(
            version=0.1,
            insert_underscores=True,
            coordinate_system="RAS",
            return_plane=True,
        )
        self.assertTrue(
            np.array_equal(
                pts2[0, :],
                np.array(
                    [
                        -ant_hrz_hole_pts[1, 0],
                        -ant_hrz_hole_pts[1, 1],
                        ant_hrz_hole_pts[1, 2],
                    ]
                ),
            )
        )
        self.assertTrue("_" in names[0])
        self.assertTrue(len(names) == 5)

    def test_append_ones_column(self) -> None:
        """
        Tests append_ones_column

        Note that copilot wrote this function... so it's probably fine
        """
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        test_array = append_ones_column(test_array)
        self.assertTrue(np.array_equal(test_array[:, 3], np.array([1, 1])))
        self.assertTrue(
            np.array_equal(
                test_array[:, 0:3], np.array([[1, 2, 3], [4, 5, 6]])
            )
        )

    def test_optimize_transform_labeled_lines(self) -> None:
        """
        Tests optimize_transform_labeled_lines
        """
        # Generate some test data by just moving the 'ground truth' point
        # around
        pts1, pts2, _ = get_headframe_hole_lines()

        move_pts = np.vstack([pts1, pts2])
        move_pts[:, 2] = move_pts[:, 2] + 1
        move_pts = append_ones_column(move_pts)
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])

        init = np.zeros((6,))
        trans, Tframe = optimize_transform_labeled_lines(
            init, pts1, pts2, move_pts, labels
        )
        self.assertTrue(Tframe[-1] - 1 < 1e-6)
        self.assertTrue(trans[-1, -1] - 1 < 1e-6)

    def test_optimize_transform_labeled_lines_with_plane(self) -> None:
        """
        Test optimize_transform_labeled_lines_with_plane
        """
        pts1, pts2, _ = get_headframe_hole_lines(
            version=0.1,
            insert_underscores=False,
            coordinate_system="LPS",
            return_plane=True,
        )

        move_pts = np.vstack([pts1[:4, :], pts2[:4, :], pts2[-1, :]])
        move_pts[:, 2] = move_pts[:, 2] + 1
        move_pts = append_ones_column(move_pts)

        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4])

        pts_for_line = np.ones(len(labels), dtype=bool)
        pts_for_line[-1] = False

        init = np.zeros((6,))
        trans, Tframe = optimize_transform_labeled_lines_with_plane(
            init, pts1, pts2, pts_for_line, move_pts, labels
        )

        self.assertTrue(Tframe[-1] - 1 < 1e-6)
        self.assertTrue(trans[-1, -1] - 1 < 1e-6)


if __name__ == "__main__":
    unittest.main()
