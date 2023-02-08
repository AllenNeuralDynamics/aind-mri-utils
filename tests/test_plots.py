"""Tests functions in `plots`."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from aind_mri_utils import plots as mrplt


class PlotsTest(unittest.TestCase):
    """Tests functions in `plots`."""

    f1 = plt.figure(1)
    f2 = plt.figure(2)
    ax = f1.add_subplot()
    ax3d = f2.add_subplot(projection="3d")

    vertices = np.array(
        [
            (-6.859, 0.7264, 2.84),
            (-6.859, 1.5264, 2.84),
            (-8.059, 1.5264, 2.84),
            (-8.049885, 1.5264, 2.944189),
            (-8.022816, 1.5264, 3.045212),
        ]
    )
    faces = np.array(
        [
            [0, 2, 3],
            [0, 3, 4],
            [2, 3, 4],
            [1, 2, 4],
            [1, 3, 4],
        ]
    )

    expected_edges = np.array(
        [
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [3, 2],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3],
        ],
        dtype="int32",
    )

    def test_make_3d_ax_look_normal(self) -> None:
        """Tests make_3d_ax_look_normal"""
        mrplt.make_3d_ax_look_normal(self.ax3d)
        box_aspect = self.ax3d.get_box_aspect()
        self.assertTrue(
            np.array_equal(
                box_aspect / box_aspect[0], np.ones(3, dtype="float64")
            )
        )

    def test_set_axes_equal(self) -> None:
        """Tests set_axes_equal"""
        mrplt.set_axes_equal(self.ax3d)
        limits = np.array(
            [
                self.ax3d.get_xlim3d(),
                self.ax3d.get_ylim3d(),
                self.ax3d.get_zlim3d(),
            ]
        )
        limits_diff = np.diff(limits)
        self.assertTrue(np.all(limits_diff == limits_diff[0]))

    def test_plot_tri_mesh(self) -> None:
        """Tests plot_tri_mesh"""
        handles, tri = mrplt.plot_tri_mesh(
            self.ax3d, self.vertices, self.faces
        )
        self.assertTrue(np.array_equal(tri.edges, self.expected_edges))


if __name__ == "__main__":
    unittest.main()
