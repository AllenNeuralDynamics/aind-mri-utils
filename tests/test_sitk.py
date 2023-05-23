import unittest

import numpy as np
import SimpleITK as sitk

from aind_mri_utils import rotations, sitk_volume


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
        testImage = sitk.GetImageFromArray(np.ones((20, 10, 10)))

        R = rotations.define_euler_rotation(90, 0, 0)
        trans = rotations.scipy_rotation_to_sitk(R)
        # Test Sizing
        new_img = sitk_volume.resample(testImage, transform=trans)
        print(new_img.GetSize())
        self.assertTrue(
            np.array_equal(new_img.GetSize(), np.array([10, 20, 10]))
        )
        # a couple values
        R = rotations.define_euler_rotation(45, 0, 0)
        trans = rotations.scipy_rotation_to_sitk(R)
        new_img = sitk_volume.resample(testImage, transform=trans)
        self.assertTrue(new_img.GetPixel([5, 5, 5]) == 1)
        self.assertTrue(new_img.GetPixel([0, 0, 0]) == 0)

    def test_transform_sitk_indices_to_physical_points(self) -> None:
        simg = sitk.Image(256, 128, 64, sitk.sitkUInt8)
        for ndxs, ans in self.test_index_translation_sets:
            received = sitk_volume.transform_sitk_indices_to_physical_points(
                simg, ndxs
            )
            self.assertTrue(np.allclose(ans, received))


if __name__ == "__main__":
    unittest.main()
