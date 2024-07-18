import unittest

import numpy as np
import SimpleITK as sitk

from aind_mri_utils import headframe_rotation as hr


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


class HeadframeRotationTest(unittest.TestCase):
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
        coms = hr.slices_centers_of_mass(img, seg_img, 2, 1, 5)
        self.assertEqual(coms.shape, (np_test_img_size[0], 3))
        for i in range(coms.shape[0]):
            self.assertTrue(
                np.allclose(coms[i, :2], self.sitk_test_img_center)
            )

    def test_get_segmentation_pca(self) -> None:
        seg_img = make_cylinders(
            self.sitk_test_img_size, self.cylinder_defs, self.seg_vals
        )
        axis = hr.get_segmentation_pca(
            seg_img, list(self.seg_vals_dict["vertical"].values())
        )
        self.assertTrue(np.allclose(axis, hr.lps_axes["dv"]))

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
        these_orient_indices = hr.def_orient_indices["vertical"]
        hole = hr.find_hole(img, seg_img, this_val, these_orient_indices)
        self.assertTrue(np.isnan(hole[2]))
        self.assertTrue(np.allclose(hole[:2], list(self.cylinder_defs[0][0])))

        none_hole = hr.find_hole(
            img, seg_img, 5, these_orient_indices
        )  # 5 is not a seg value
        self.assertTrue(none_hole is None)

        bad_img = make_cylinders(
            self.sitk_test_img_size[::-1],
            self.cylinder_defs,
            np.ones(len(self.seg_vals)),
        )
        self.assertRaises(
            ValueError,
            hr.find_hole,
            bad_img,
            seg_img,
            this_val,
            these_orient_indices,
        )

        zero_img = make_cylinders(
            self.sitk_test_img_size,
            self.cylinder_defs,
            np.zeros(len(self.seg_vals)),
        )
        none_hole = hr.find_hole(
            zero_img, seg_img, this_val, these_orient_indices
        )
        self.assertTrue(none_hole is None)

        holes_dict = hr.find_holes_by_orientation(
            img,
            seg_img,
            self.seg_vals_dict,
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

        centers_ang = hr.find_hole_angles(holes_dict)
        self.assertAlmostEqual(centers_ang["vertical"], -0.7853981633)
        self.assertAlmostEqual(centers_ang["horizontal"], 1.9028557943377)

        initial_axes = hr.estimate_hole_axes_from_segmentation_by_orientation(
            seg_img,
            self.seg_vals_dict,
        )
        self.assertTrue(
            np.allclose(initial_axes["vertical"], hr.lps_axes["dv"])
        )
        self.assertTrue(
            np.allclose(initial_axes["horizontal"], hr.lps_axes["ap"])
        )

        coms = hr.calculate_centers_of_mass_for_image_and_segmentation(
            img,
            seg_img,
            initial_axes,
            self.seg_vals_dict,
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
            hr.estimate_axis_rotations_from_centers_of_mass(coms)
        )
        self.assertTrue(
            np.allclose(orient_rotation_matrices["vertical"], np.eye(3))
        )
        self.assertTrue(
            np.allclose(orient_rotation_matrices["horizontal"], np.eye(3))
        )
        self.assertTrue(np.allclose(axes["vertical"], hr.lps_axes["dv"]))
        self.assertTrue(np.allclose(axes["horizontal"], hr.lps_axes["ap"]))

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
        R, offset = hr.find_rotation_to_match_hole_angles(
            img,
            seg_img,
            orient_rotation_matrices,
            axes,
            self.seg_vals_dict,
            design_centers=test_centers,
        )
        self.assertTrue(np.allclose(R, np.eye(3)))
        self.assertTrue(np.allclose(offset, np.zeros(3)))

        coms = hr.estimate_coms_from_image_and_segmentation(
            img, seg_img, self.seg_vals_dict
        )
        for orient, com_answer_dict_orient in com_answer_dict.items():
            for ap, com_answer in com_answer_dict_orient.items():
                self.assertTrue(np.allclose(coms[orient][ap], com_answer))

        coms, R, offset = (
            hr.estimate_rotation_and_coms_from_image_and_segmentation(
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
