"""Tests functions in `file_io.obj_files`."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pywavefront

from aind_mri_utils.file_io import obj_files as of


class ObjFilesTest(unittest.TestCase):
    """Tests functions in `file_io.obj_files`."""

    scene_mock = MagicMock()
    scene_mock.vertices = [
        (-6.859, 0.7264, 2.84),
        (-6.859, 1.5264, 2.84),
        (-8.059, 1.5264, 2.84),
        (-8.049885, 1.5264, 2.944189),
        (-8.022816, 1.5264, 3.045212),
    ]
    scene_mock.mesh_list = [MagicMock(), MagicMock()]
    scene_mock.mesh_list[0].faces = [
        [0, 2, 3],
        [0, 3, 4],
        [2, 3, 4],
        [1, 2, 4],
        [1, 3, 4],
    ]
    scene_mock.mesh_list[1].faces = scene_mock.mesh_list[0].faces

    expected_vertices = np.array(scene_mock.vertices)
    expected_faces_per_mesh = np.array(scene_mock.mesh_list[0].faces)

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

    def test_get_vertices_and_faces(self) -> None:
        """Tests `get_vertices_and_faces`"""
        vertices, faces = of.get_vertices_and_faces(self.scene_mock)
        self.assertTrue(np.array_equal(vertices, self.expected_vertices))
        self.assertEqual(len(faces), 2)
        self.assertTrue(np.array_equal(faces[1], self.expected_faces_per_mesh))
        with patch(
            "aind_mri_utils.file_io.obj_files.pywavefront.Wavefront"
        ) as mock:
            mock.return_value = self.scene_mock
            vertices, faces = of.get_vertices_and_faces("foobar")
            self.assertTrue(np.array_equal(vertices, self.expected_vertices))
            self.assertEqual(len(faces), 2)
            self.assertTrue(
                np.array_equal(faces[1], self.expected_faces_per_mesh)
            )


if __name__ == "__main__":
    unittest.main()
