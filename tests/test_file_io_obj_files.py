"""Tests functions in `file_io.obj_files`."""

import inspect
import unittest
from unittest.mock import patch

import pywavefront

from aind_mri_utils.file_io import obj_files as of


class ObjFilesTest(unittest.TestCase):
    """Tests functions in `file_io.obj_files`."""

    def test_load_obj_wavefront(self) -> None:
        """Tests that the `load_obj_wavefront` function works as intended."""
        # inspect API for Wavefront to see that it's roughly what we need
        s = inspect.signature(pywavefront.Wavefront)
        self.assertTrue("strict" in s.parameters)
        self.assertTrue("create_materials" in s.parameters)
        self.assertTrue("collect_faces" in s.parameters)
        # Call the function to achieve 100% coverage (why though?)
        #
        # To be clear this is just for the coverage, and is not a meaningful
        # test
        with patch(
            "aind_mri_utils.file_io.obj_files.pywavefront.Wavefront"
        ) as mock:
            mock.return_value = True
            self.assertTrue(of.load_obj_wavefront("foobar"))


if __name__ == "__main__":
    unittest.main()
