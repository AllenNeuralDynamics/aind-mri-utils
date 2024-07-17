# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:54:02 2023
"""

import unittest

import numpy as np
import SimpleITK as sitk

from aind_mri_utils import measurement


def add_cylinder(arr, center, radius, sel_ndx, ndx_range, value):
    """Add a cylinder to an array"""
    x, y, z = np.meshgrid(*map(np.arange, arr.shape), indexing="ij")
    ndxs = [x, y, z]
    all_axes = set([0, 1, 2])
    a, b = all_axes.difference([sel_ndx])
    mask = (ndxs[a] - center[0]) ** 2 + (ndxs[b] - center[1]) ** 2 < radius**2
    mask = mask & (
        (ndx_range[0] <= ndxs[sel_ndx]) & (ndxs[sel_ndx] < ndx_range[1])
    )
    arr[mask] = value
    return arr


def add_z_cylinder(arr, center, radius, ndx_range, value):
    """Add a cylinder to an array"""
    add_cylinder(arr, center, radius, 2, ndx_range, value)


def add_y_cylinder(arr, center, radius, ndx_range, value):
    """Add a cylinder to an array"""
    add_cylinder(arr, center, radius, 1, ndx_range, value)


def add_x_cylinder(arr, center, radius, ndx_range, value):
    """Add a cylinder to an array"""
    add_cylinder(arr, center, radius, 0, ndx_range, value)


def make_cylinders(img_size, cylinder_defs, vals):
    seg_arr = np.zeros(img_size[::-1], dtype="uint8")
    for cylinder, val in zip(cylinder_defs, vals):
        center, radius, sel_ndx, ndx_range = cylinder
        # Reminder: numpy indexing is (z, y, x) while SimpleITK uses (x, y, z)
        # Need to convert these sitk definitions to numpy
        center_np = np.array(center)[::-1]
        sel_ndx_np = 2 - sel_ndx
        add_cylinder(seg_arr, center_np, radius, sel_ndx_np, ndx_range, val)
    seg_img = sitk.GetImageFromArray(seg_arr)
    return seg_img


class MeasurementTest(unittest.TestCase):
    sitk_test_img_size = (64, 64, 32)
    sitk_test_img_center = np.array([31.5, 31.5])
    # These are SITK indices!
    cylinder_defs = [
        # center, radius, sel_ndx, ndx_range
        # axis of cylinder is in `sel_ndx` direction
        # `ndx_range` determines the range of the cylinder
        #  in `sel_ndx` direction
        ((16, 16), 5, 2, (0, 16)),
        ((45, 45), 5, 2, (16, 32)),
        ((45, 10), 5, 1, (0, 16)),
        ((16, 20), 5, 1, (48, 64)),
    ]
    seg_vals = range(1, len(cylinder_defs) + 1)
    orient_names = ("vertical", "horizontal")
    ap_names = ("anterior", "posterior")
    seg_vals_dict = {
        "vertical": {a: v for a, v in zip(ap_names, range(1, 3))},
        "horizontal": {a: v for a, v in zip(ap_names, range(3, 5))},
    }
    lps_axes = dict(
        ap=np.array([0, 1, 0]), dv=np.array([0, 0, 1]), ml=np.array([1, 0, 0])
    )
    hole_orient_axis = dict(horizontal="ap", vertical="dv")
    orient_comparison_axis = dict(horizontal="dv", vertical="ap")
    design_centers = dict(
        horizontal=dict(
            anterior=np.array([-6.34, np.nan, 2.5]),
            posterior=np.array([-5.04, np.nan, 1]),
        ),
        vertical=dict(
            anterior=np.array([-5.09, 3.209, np.nan]),
            posterior=np.array([-6.84, 9.909, np.nan]),
        ),
    )  # Bregma-relative mms (LPS)
    orient_indices = dict(horizontal=[0, 2], vertical=[0, 1])
    hole_order = dict(
        horizontal=["anterior", "posterior"],
        vertical=["posterior", "anterior"],
    )

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

    def test_slices_center_of_mass(self) -> None:
        # Reminder: numpy indexing is (z, y, x) while SimpleITK uses (x, y, z)
        # (column major vs row major)
        # So a LPS image should have the THIRD axis be the L axis in numpy

        # convert between numpy and simpleITK indexing for size
        np_test_img_size = self.sitk_test_img_size[::-1]
        img_arr = np.zeros(np_test_img_size)
        seg_arr = np.zeros_like(img_arr, dtype="uint8")
        sigma = 0.5
        grids = []
        for i in range(1, 3):
            grids.append(np.linspace(-1, 1, np_test_img_size[i]))
        # backwards because of numpy indexing
        y, x = np.meshgrid(*grids, indexing="ij")
        dst = np.sqrt(x**2 + y**2)
        normal = 1 / (sigma * np.sqrt(2 * np.pi))
        exp_normal = 1 / (2 * sigma**2)
        img_arr[0, :, :] = normal * np.exp(exp_normal * -((dst) ** 2))
        seg_arr[0, :, :] = img_arr[0, :, :] > 0.5
        # Copy the first slice to the rest of the slices
        for i in range(1, np_test_img_size[0]):
            img_arr[i, :, :] = img_arr[0, :, :]
            seg_arr[i, :, :] = seg_arr[0, :, :]
        img = sitk.GetImageFromArray(img_arr)
        seg_img = sitk.GetImageFromArray(seg_arr)
        coms = measurement.slices_centers_of_mass(img, seg_img, 2, 1, 5)
        self.assertEqual(coms.shape, (np_test_img_size[0], 3))
        for i in range(coms.shape[0]):
            self.assertTrue(
                np.allclose(coms[i, :2], self.sitk_test_img_center)
            )

    def test_get_segmentation_pca(self) -> None:
        seg_img = make_cylinders(
            self.sitk_test_img_size, self.cylinder_defs, self.seg_vals
        )
        axis = measurement.get_segmentation_pca(
            seg_img, list(self.seg_vals_dict["vertical"].values())
        )
        self.assertTrue(np.allclose(axis, self.lps_axes["dv"]))

    def test_hole_finding_and_orientation(self) -> None:

        seg_img = make_cylinders(
            self.sitk_test_img_size, self.cylinder_defs, self.seg_vals
        )
        img = make_cylinders(
            self.sitk_test_img_size,
            self.cylinder_defs,
            np.ones(len(self.seg_vals)),
        )

        this_val = self.seg_vals_dict["vertical"]["anterior"]
        hole = measurement.find_hole(
            img, seg_img, this_val, self.orient_indices["vertical"]
        )
        self.assertTrue(np.isnan(hole[2]))
        self.assertTrue(np.allclose(hole[:2], list(self.cylinder_defs[0][0])))

        none_hole = measurement.find_hole(
            img, seg_img, 5, self.orient_indices["vertical"]
        )  # 5 is not a seg value
        self.assertTrue(none_hole is None)

        bad_img = make_cylinders(
            self.sitk_test_img_size[::-1],
            self.cylinder_defs,
            np.ones(len(self.seg_vals)),
        )
        self.assertRaises(
            ValueError,
            measurement.find_hole,
            bad_img,
            seg_img,
            this_val,
            self.orient_indices["vertical"],
        )

        zero_img = make_cylinders(
            self.sitk_test_img_size,
            self.cylinder_defs,
            np.zeros(len(self.seg_vals)),
        )
        none_hole = measurement.find_hole(
            zero_img, seg_img, this_val, self.orient_indices["vertical"]
        )
        self.assertTrue(none_hole is None)

        holes_dict = measurement.find_holes_by_orientation(
            img,
            seg_img,
            self.seg_vals_dict,
            self.orient_indices,
            self.orient_names,
            self.ap_names,
        )
        self.assertTrue(
            np.allclose(
                holes_dict["vertical"]["anterior"][:2],
                list(self.cylinder_defs[0][0]),
            )
        )
        self.assertTrue(
            np.allclose(
                holes_dict["vertical"]["posterior"][:2],
                list(self.cylinder_defs[1][0]),
            )
        )
        self.assertTrue(
            np.allclose(
                holes_dict["horizontal"]["anterior"][[0, 2]],
                list(self.cylinder_defs[2][0]),
            )
        )
        self.assertTrue(
            np.allclose(
                holes_dict["horizontal"]["posterior"][[0, 2]],
                list(self.cylinder_defs[3][0]),
            )
        )

        orient_lps_vector_dict = {
            orient: self.lps_axes[self.hole_orient_axis[orient]]
            for orient in self.orient_names
        }
        centers_ang = measurement.find_hole_angles(
            holes_dict,
            self.hole_order,
            self.lps_axes,
            self.orient_comparison_axis,
            orient_lps_vector_dict,
            self.orient_names,
        )

        self.assertAlmostEqual(centers_ang["vertical"], -0.7853981633)
        self.assertAlmostEqual(centers_ang["horizontal"], 1.9028557943377)

        initial_axes = (
            measurement.estimate_hole_axes_from_segmentation_by_orientation(
                seg_img,
                self.seg_vals_dict,
                orient_lps_vector_dict,
                self.orient_names,
                self.ap_names,
            )
        )
        self.assertTrue(
            np.allclose(initial_axes["vertical"], self.lps_axes["dv"])
        )
        self.assertTrue(
            np.allclose(initial_axes["horizontal"], self.lps_axes["ap"])
        )

        coms = (
            measurement.calculate_centers_of_mass_for_image_and_segmentation(
                img,
                seg_img,
                initial_axes,
                self.seg_vals_dict,
                orient_lps_vector_dict,
                self.orient_names,
                self.ap_names,
            )
        )
        com_answer_dict = {
            "vertical": {
                "anterior": np.array([[16.0, 16.0, x] for x in range(16)]),
                "posterior": np.array(
                    [[45.0, 45.0, x] for x in range(16, 32)]
                ),
            },
            "horizontal": {
                "anterior": np.array([[45.0, x, 10.0] for x in range(16)]),
                "posterior": np.array(
                    [[16.0, x, 20.0] for x in range(48, 64)]
                ),
            },
        }
        for orient, com_answer_dict_orient in com_answer_dict.items():
            for ap, com_answer in com_answer_dict_orient.items():
                self.assertTrue(np.allclose(coms[orient][ap], com_answer))

        orient_rotation_matrices, axes = (
            measurement.estimate_axis_rotations_from_centers_of_mass(
                coms, orient_lps_vector_dict, self.orient_names, self.ap_names
            )
        )
        self.assertTrue(
            np.allclose(orient_rotation_matrices["vertical"], np.eye(3))
        )
        self.assertTrue(
            np.allclose(orient_rotation_matrices["horizontal"], np.eye(3))
        )
        self.assertTrue(np.allclose(axes["vertical"], self.lps_axes["dv"]))
        self.assertTrue(np.allclose(axes["horizontal"], self.lps_axes["ap"]))

        test_centers = dict(
            vertical=dict(
                anterior=np.array([16, 16, np.nan]),
                posterior=np.array([45, 45, np.nan]),
            ),
            horizontal=dict(
                anterior=np.array([45, np.nan, 10]),
                posterior=np.array([16, np.nan, 20]),
            ),
        )
        R, offset = measurement.find_rotation_to_match_hole_angles(
            img,
            seg_img,
            orient_rotation_matrices,
            axes,
            self.seg_vals_dict,
            test_centers,
            orient_lps_vector_dict,
            self.orient_names,
            self.ap_names,
            self.orient_indices,
            self.hole_order,
            self.lps_axes,
            self.orient_comparison_axis,
        )
        self.assertTrue(np.allclose(R, np.eye(3)))
        self.assertTrue(np.allclose(offset, np.zeros(3)))

        coms = measurement.estimate_coms_from_image_and_segmentation(
            img, seg_img, self.seg_vals_dict
        )
        for orient, com_answer_dict_orient in com_answer_dict.items():
            for ap, com_answer in com_answer_dict_orient.items():
                self.assertTrue(np.allclose(coms[orient][ap], com_answer))

        coms, R, offset = (
            measurement.estimate_rotation_and_coms_from_image_and_segmentation(
                img, seg_img, self.seg_vals_dict, design_centers=test_centers
            )
        )
        for orient, com_answer_dict_orient in com_answer_dict.items():
            for ap, com_answer in com_answer_dict_orient.items():
                self.assertTrue(np.allclose(coms[orient][ap], com_answer))
        self.assertTrue(np.allclose(R, np.eye(3)))
        self.assertTrue(np.allclose(offset, np.zeros(3)))


if __name__ == "__main__":
    unittest.main()
