"""Tests functions in `plots`."""

import unittest
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

from aind_mri_utils import plots as mrplt


@contextmanager
def managed_subplot(f, *args, **kwargs):
    ax = f.add_subplot(*args, **kwargs)
    try:
        yield ax
    finally:
        f.clf()


class PlotsTest(unittest.TestCase):
    """Tests functions in `plots`."""

    f1 = plt.figure(1)

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
        with managed_subplot(self.f1, projection="3d") as ax3d:
            mrplt.make_3d_ax_look_normal(ax3d)
            box_aspect = ax3d.get_box_aspect()
            self.assertTrue(
                np.array_equal(
                    box_aspect / box_aspect[0], np.ones(3, dtype="float64")
                )
            )

    def test_set_axes_equal(self) -> None:
        """Tests set_axes_equal"""
        with managed_subplot(self.f1, projection="3d") as ax3d:
            mrplt.set_axes_equal(ax3d)
            limits = np.array(
                [
                    ax3d.get_xlim3d(),
                    ax3d.get_ylim3d(),
                    ax3d.get_zlim3d(),
                ]
            )
            limits_diff = np.diff(limits)
            self.assertTrue(np.all(limits_diff == limits_diff[0]))

    def test_plot_tri_mesh(self) -> None:
        """Tests plot_tri_mesh"""
        with managed_subplot(self.f1, projection="3d") as ax3d:
            handles, tri = mrplt.plot_tri_mesh(ax3d, self.vertices, self.faces)
            self.assertTrue(np.array_equal(tri.edges, self.expected_edges))

    def test_plot_point_cloud_3d(self) -> None:
        with managed_subplot(self.f1, projection="3d") as ax3d:
            outs = mrplt.plot_point_cloud_3d(ax3d, self.faces)
            self.assertTrue(outs.get_offsets().size == 10)

    def test_plot_vector(self) -> None:
        with managed_subplot(self.f1, projection="3d") as ax3d:
            outs = mrplt.plot_vector(ax3d, self.faces[0, :])
            self.assertTrue(len(outs) == 1)

    def create_single_colormap(self) -> None:
        """Tests create_single_color_map"""
        # Test colormap creation
        A = mrplt.create_single_colormap("magenta", start_color="white")
        self.assertTrue(np.array_equal(A(0), np.array([1, 1, 1, 1])))
        self.assertTrue(np.array_equal(A(1), np.array([1, 0, 1, 1])))
        # Test colormap creation with reverse
        A = mrplt.create_single_colormap(
            "magenta", start_color="white", is_reverse=True
        )
        self.assertTrue(np.array_equal(A(0), np.array([1, 0, 1, 1])))
        self.assertTrue(np.array_equal(A(1), np.array([1, 1, 1, 1])))
        # Test output length
        A = mrplt.create_single_colormap(
            "magenta", start_color="white", N=4192
        )
        self.assertTrue(len(A) == 4192)
        # Test transparency
        A = mrplt.create_single_colormap(
            "magenta", start_color="white", N=4192, is_transparent=True
        )
        self.assertTrue(np.array_equal(A(0), np.array([1, 1, 1, 0])))
        # Tst saturation
        A = mrplt.create_single_colormap(
            "magenta", start_color="white", saturation=4
        )
        self.assertTrue(np.array_equal(A(254), np.array([1, 0, 1, 1])))


if __name__ == "__main__":
    unittest.main()
