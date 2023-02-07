"""Tests functions in `file_io.slicer_files`."""

import unittest
import inspect

import numpy as np

from aind_mri_utils.file_io import slicer_files as sf

class SlicerFilesTest(unittest.TestCase):
    """Tests functions in `file_io.slicer_files`."""

    expected_names = ['foo', 'bar']
    expected_pos = np.array([[1, 2, 3], [4, 5, 6]], dtype = 'float64')
    slicer_json_control_points_mock = {
        'markups': [{'controlPoints':
                     [{'label': expected_names[0],
                       'position': expected_pos[0,:].tolist()},
                    {'label': expected_names[1],
                     'position': expected_pos[1,:].tolist()}]
                     }]}

    def test_extract_control_points(self) -> None:
        """Tests that the `extract_control_points` function works as intended."""
        received_pos, received_names = sf.extract_control_points(
            self.slicer_json_control_points_mock
        )
        self.assertTrue(np.array_equal(received_pos, self.expected_pos))
        self.assertEqual(received_names, self.expected_names)

if __name__ == "__main__":
    unittest.main()
