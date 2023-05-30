import unittest

import numpy as np

from aind_mri_utils.file_io import simpleitk as si
from aind_mri_utils.optimization import create_rigid_transform

from pathlib import Path
import os

class SITKTest(unittest.TestCase):
    """Tests functions in `file_io.simpleitk`."""

    def test_save_sitk_transform(self) -> None:
        """
        Tests that the `save_sitk_transform` function works as intended.
        """
        # Test that a 4x4 transform is created correctly
        trans = create_rigid_transform([0,0,0,0,0,0])
        self.assertTrue(np.array_equal(trans,np.eye(4)))
        si.save_sitk_transform(str(Path('testsave.h5')),trans)
        load_trans = si.load_sitk_transform(str(Path('testsave.h5')))
        self.assertTrue(np.array_equal(trans,load_trans))

        # Test more complicated transform
        trans[0,-1] = 1
        si.save_sitk_transform(str(Path('testsave.h5')),trans)
        load_trans = si.load_sitk_transform(str(Path('testsave.h5')))
        self.assertTrue(np.array_equal(trans,load_trans))

        # Test inversion functionality when saving
        si.save_sitk_transform(str(Path('testsave.h5')),trans,invert=True)
        load_trans = si.load_sitk_transform(str(Path('testsave.h5')))
        self.assertTrue(np.array_equal(trans,load_trans.T))
        
        # Test inversion functionality when loading
        si.save_sitk_transform(str(Path('testsave.h5')),trans,invert=True)
        load_trans = si.load_sitk_transform(str(Path('testsave.h5')),invert=True)
        self.assertTrue(np.array_equal(trans,load_trans))

        # Kill the file we created- it was just a test
        os.remove('testsave.h5')


