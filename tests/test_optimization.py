import unittest

import numpy as np

from aind_mri_utils.optimization import (
    get_headframe_hole_lines,
    optimize_transform_labeled_lines,
    optimize_transform_labeled_lines_with_plane,
)


class OptimizationTest(unittest.TestCase):
    def test_get_headframe_hole_lines(self) -> None:
        """
        Tests for get_headframe_hole_lines
        """
        ant_hrz_hole_pts = np.array([[-6.34, 0, 2.5], [-6.34, 6.5, 2.5]])
        pts1, _, names = get_headframe_hole_lines(
            version="0.1",
            insert_underscores=False,
            coordinate_system="LPS",
            return_plane=False,
        )

        self.assertTrue(np.allclose(pts1[0, :], ant_hrz_hole_pts[0, :]))
        self.assertFalse("_" in names[0])
        self.assertTrue(len(names) == 4)

        pts1, pts2, names = get_headframe_hole_lines(
            version="0.1",
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

        # Test that value error is raised if bad version in passed
        self.assertRaises(ValueError, get_headframe_hole_lines, version=12)

    def test_optimize_transform_labeled_lines(self) -> None:
        """
        Tests optimize_transform_labeled_lines
        """
        # Generate some test data by just moving the 'ground truth' point
        # around
        pts1, pts2, _ = get_headframe_hole_lines()

        move_pts = np.vstack([pts1, pts2])
        move_pts[:, 2] = move_pts[:, 2] + 1
        labels = np.array(
            [
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
            ]
        )
        weights = np.array(
            [
                1,
                1,
                1,
                1,
                0.99,
                0.99,
                0.99,
                0.99,
            ]
        )  # To test that the weights are being used
        # Test with default weights input
        init = np.zeros((6,))
        trans, T_frame = optimize_transform_labeled_lines(
            init,
            pts1,
            pts2,
            move_pts,
            labels,
        )
        self.assertTrue(T_frame[-1] - 1 < 1e-6)
        self.assertTrue(trans[-1, -1] - 1 < 1e-6)

        # Test with non-default weights input
        init = np.zeros((6,))
        trans, T_frame = optimize_transform_labeled_lines(
            init,
            pts1,
            pts2,
            move_pts,
            labels,
            weights=weights,
            gamma=0.5,
            normalize=True,
        )
        self.assertTrue(T_frame[-1] - 1 < 1e-6)
        self.assertTrue(trans[-1, -1] - 1 < 1e-6)

    def test_optimize_transform_labeled_lines_with_plane(self) -> None:
        """
        Test optimize_transform_labeled_lines_with_plane
        """
        pts1, pts2, _ = get_headframe_hole_lines(
            version="0.1",
            insert_underscores=False,
            coordinate_system="LPS",
            return_plane=True,
        )

        move_pts = np.vstack([pts1[:4, :], pts2[:4, :], pts2[-1, :]])
        move_pts[:, 2] = move_pts[:, 2] + 1

        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4])
        weights = np.ones(
            len(labels),
        )

        pts_for_line = np.ones(pts1.shape[0], dtype=bool)
        pts_for_line[-1] = False
        # Test with all no weights added
        init = np.zeros((6,))
        trans, T_frame = optimize_transform_labeled_lines_with_plane(
            init, pts1, pts2, pts_for_line, move_pts, labels
        )

        self.assertTrue(T_frame[-1] - 1 < 1e-6)
        self.assertTrue(trans[-1, -1] - 1 < 1e-6)

        # Test with weights input
        init = np.zeros((6,))
        trans, T_frame = optimize_transform_labeled_lines_with_plane(
            init,
            pts1,
            pts2,
            pts_for_line,
            move_pts,
            labels,
            weights=weights,
            gamma=0.5,
            normalize=True,
        )

        self.assertTrue(T_frame[-1] - 1 < 1e-6)
        self.assertTrue(trans[-1, -1] - 1 < 1e-6)


if __name__ == "__main__":
    unittest.main()
