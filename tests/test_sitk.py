import unittest

import numpy as np
import SimpleITK as sitk

from aind_mri_utils import rotations, sitk_volume


def all_closer_than(a, b, thresh):
    return np.all(np.abs(a - b) <= thresh)


def fraction_close(a, val):
    arr = sitk.GetArrayViewFromImage(a)
    nel = np.prod(arr.shape)
    return np.sum(np.isclose(arr, val)) / nel


class SITKTest(unittest.TestCase):
    test_index_translation_sets = [
        (np.array([[0, 0, 0], [2, 2, 2]]), np.array([[0, 0, 0], [2, 2, 2]])),
        (
            np.array([[0.5, 0.5, 0.5], [2, 2, 2]]),
            np.array([[0.5, 0.5, 0.5], [2, 2, 2]]),
        ),
    ]

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

    def test_resample(self) -> None:
        testImage = sitk.GetImageFromArray(np.ones((20, 30, 10)))
        testImage_RIA = sitk.Image(testImage)
        testImage_RIA.SetDirection(
            (-1.0, 0.0, 0.0, 0.0, -0.0, -1.0, 0.0, -1.0, 0.0)
        )
        R = rotations.define_euler_rotation(90, 0, 0)
        trans = rotations.scipy_rotation_to_sitk(R)
        # Test Sizing
        new_img = sitk_volume.resample(testImage, transform=trans)
        new_img_RIA = sitk_volume.resample(testImage_RIA, transform=trans)
        # Test that size is correct. Note that there are multiple correct
        # answers, depending on package versions. This should probably be
        # revisited at some point :/
        self.assertTrue(
            all_closer_than(new_img.GetSize(), np.array([10, 20, 30]), 1)
        )
        self.assertTrue(
            all_closer_than(new_img_RIA.GetSize(), np.array([10, 30, 20]), 1)
        )  # a couple values
        self.assertTrue(fraction_close(new_img, 1) > 0.9)
        # self.assertTrue(fraction_close(new_img_RIA, 1) > 0.9) TODO: fix this
        R = rotations.define_euler_rotation(45, 0, 0)
        trans = rotations.scipy_rotation_to_sitk(R)
        new_img = sitk_volume.resample(testImage, transform=trans)
        self.assertTrue(np.isclose(new_img.GetPixel([5, 10, 15]), 1))
        self.assertTrue(np.isclose(new_img.GetPixel([0, 0, 0]), 0))
        # Make RIA oriented image

    def test_transform_sitk_indices_to_physical_points(self) -> None:
        simg = sitk.Image(256, 128, 64, sitk.sitkUInt8)
        for ndxs, ans in self.test_index_translation_sets:
            received = sitk_volume.transform_sitk_indices_to_physical_points(
                simg, ndxs
            )
            self.assertTrue(np.allclose(ans, received))


if __name__ == "__main__":
    unittest.main()
