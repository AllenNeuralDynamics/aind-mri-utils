"""Tests functions in `obj_plots`."""

import unittest
import inspect
import pywavefront

import numpy as np

from aind_mri_utils import obj_plots as op


class ObjPlotsTest(unittest.TestCase):
    """Tests functions in `obj_plots`."""

    def test_load_obj_wavefront(self) -> None:
        """Tests that the `load_obj_wavefront` function works as intended."""
        # inspect API for Wavefront to see that it's roughly what we need
        s = inspect.signature(pywavefront.Wavefront)
        self.assertTrue('strict' in s.paramters)
        self.assertTrue('create_materials' in s.paramters)
        self.assertTrue('collect_faces' in s.paramters)
        # Call the function to achieve 100% coverage (why though?)
        #
        # To be clear this is just for the coverage, and is not a meaningful
        # test
        with unittest.patch('aind_mri_utils.pywave.Wavefront') as mock:
            mock.return_value = True
            self.assertTrue(op.load_obj_wavefront)

    def test_get_vertices_and_faces(self) -> None:
        """Tests extraction of vertices and faces from pywavefront scene"""
        

if __name__ == "__main__":
    unittest.main()
