import os
import unittest

import numpy as np

from aind_mri_utils.file_io import simpleitk as si


class SITKTest(unittest.TestCase):
    """Tests functions in `file_io.simpleitk`."""

    def test_save_sitk_transform(self) -> None:
        """
        Tests that the `save_sitk_transform` function works as intended.
        """
        # Test that a 4x3 transform is created correctly
        test_save_path = "testsave.h5"
        R = np.eye(3)
        translation = np.zeros(3)
        si.save_sitk_transform(test_save_path, R, translation)
        load_trans, load_translation, load_center = si.load_sitk_transform(
            test_save_path
        )
        self.assertTrue(np.allclose(R, load_trans))

        # Test that the above can work from 6x1
        si.save_sitk_transform(test_save_path, np.zeros(6))
        load_trans, _, _ = si.load_sitk_transform(test_save_path)
        self.assertTrue(np.allclose(R, load_trans))

        # Test more complicated transform
        R[0, -1] = 1
        si.save_sitk_transform(test_save_path, R)
        load_trans, _, _ = si.load_sitk_transform(test_save_path)
        self.assertTrue(np.allclose(R, load_trans))

        # Test inversion functionality when saving
        si.save_sitk_transform(
            test_save_path, R, transpose_matrix=True, legacy=True
        )
        load_trans, _, _ = si.load_sitk_transform(test_save_path)
        load_trans[:3, :3] = load_trans[:3, :3].T
        self.assertTrue(np.array_equal(R, load_trans))

        # Kill the file we created- it was just a test
        os.remove("testsave.h5")


if __name__ == "__main__":
    unittest.main()
