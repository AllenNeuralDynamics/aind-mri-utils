import unittest

import numpy as np
import SimpleITK as sitk

from aind_mri_utils import rotations, sitk_volume


class SITKTest(unittest.TestCase):
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
            np.all(
                R.as_matrix().reshape((9,))
                == np.array(trans.GetParameters()[:9])
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


if __name__ == "__main__":
    unittest.main()
