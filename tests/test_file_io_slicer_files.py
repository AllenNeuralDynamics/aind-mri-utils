"""Tests functions in `file_io.slicer_files`."""

import unittest

import numpy as np

from aind_mri_utils.file_io import slicer_files as sf


class SlicerFilesTest(unittest.TestCase):
    """Tests functions in `file_io.slicer_files`."""

    expected_names = ["foo", "bar"]
    expected_pos = np.array([[1, 2, 3], [4, 5, 6]], dtype="float64")
    slicer_json_control_points_mock = {
        "markups": [
            {
                "coordinateSystem": "LPS",
                "controlPoints": [
                    {
                        "label": expected_names[0],
                        "position": expected_pos[0, :].tolist(),
                    },
                    {
                        "label": expected_names[1],
                        "position": expected_pos[1, :].tolist(),
                    },
                ],
            }
        ]
    }
    nrrd_odict_mock = {
        "Segment0_LabelValue": "1",
        "Segment0_Name": "anterior horizontal",
        "Segment1_LabelValue": "2",
        "Segment1_Name": "posterior horizontal",
        "Segment2_LabelValue": "3",
        "Segment2_Name": "anterior vertical",
        "Segment3_LabelValue": "4",
        "Segment3_Name": "posterior vertical",
    }
    nrrd_odict_ground_truth = {
        "anterior vertical": 3,
        "posterior vertical": 4,
        "posterior horizontal": 2,
        "anterior horizontal": 1,
    }

    def test_extract_control_points(self) -> None:
        """
        Tests that the `extract_control_points` function works as intended.
        """
        received_pos, received_names, coord_sys = sf.extract_control_points(
            self.slicer_json_control_points_mock
        )
        self.assertTrue(np.array_equal(received_pos, self.expected_pos))
        self.assertEqual(received_names, self.expected_names)
        self.assertEqual(coord_sys, "LPS")
        received_segment_info = sf.find_seg_nrrd_header_segment_info(
            self.nrrd_odict_mock
        )
        self.assertTrue(
            len(self.nrrd_odict_ground_truth) == len(received_segment_info)
        )
        for k, v in self.nrrd_odict_ground_truth.items():
            self.assertTrue(k in received_segment_info)
            self.assertTrue(received_segment_info[k] == v)

    def test_fcsv_read_write(self) -> None:
        """
        Tests that reading and writing of fcsv files works as expected

        """
        # Test basic read/write functionality.
        pts_dict = {"0": [0, 0, 0], "1": [1, 1, 1], "2": [2, 2, 2]}
        sf.create_slicer_fcsv(
            "test.fcsv",
            pts_dict,
        )
        read_pts_dict = sf.read_slicer_fcsv("test.fcsv")
        self.assertTrue(np.all(pts_dict["0"] == read_pts_dict["0"]))
        self.assertTrue(np.all(pts_dict["1"] == read_pts_dict["1"]))
        self.assertTrue(np.all(pts_dict["2"] == read_pts_dict["2"]))
        # Check errors from bad extensions.
        self.assertRaises(ValueError, sf.read_slicer_fcsv, "test.xlxs")
        # Assert that the function raises a ValueError when direction is wrong.
        self.assertRaises(
            ValueError, sf.read_slicer_fcsv, "test.fcsv", direction="XYZ"
        )
        # Assert that the function corrects mismatched directions
        fixed_pt_dict = sf.read_slicer_fcsv("test.fcsv", direction="RAS")
        self.assertTrue(pts_dict["0"][0] == -fixed_pt_dict["0"][0])


if __name__ == "__main__":
    unittest.main()
