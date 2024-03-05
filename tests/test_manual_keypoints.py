"""Tests functions in `manual_keypoints`."""

import unittest

import numpy as np

from aind_mri_utils import manual_keypoints as mk


class ManualKeypointsTest(unittest.TestCase):
    """Tests functions in `manual_keypoints`."""

    def test_define_transform(self):
        """Tests that the `define_transform` function works as intended."""
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        test = source.copy()
        test[:, 0] = test[:, 0] + 1

        transform = mk.define_transform(source, test)
        self.assertTrue(
            self.close_enough(mk.apply_transform(transform, source), test)
        )

    def close_enough(self, a, b):
        return np.allclose(a, b, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
