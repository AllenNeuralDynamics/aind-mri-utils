# test_arc_angles.py
"""
Unit tests for arc_angles.py

These tests cover:
* Basic, edge-case, and error-handling behavior.
* Numerical correctness for known inputs.
* Round-trip accuracy (vector → angles → vector and vice-versa).
* Correct construction of the affine matrix, while patching-out the
  external `ras_to_lps_transform` dependency so that the test-suite
  remains self-contained.
"""

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from aind_mri_utils import arc_angles as aa
from aind_mri_utils.rotations import ras_to_lps_transform


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _angle_close(a, b, tol=1e-7):
    """Return True if the two angles are equal modulo 360° (in degrees)."""
    return math.isclose(((a - b + 180) % 360) - 180, 0.0, abs_tol=tol)


def _vec_close(v1, v2, tol=1e-7):
    """Compare two vectors disregarding overall scale (both are normalized)."""
    return np.allclose(
        v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2), atol=tol
    )


# ---------------------------------------------------------------------
# Test-cases
# ---------------------------------------------------------------------
class TestVectorToArcAngles(unittest.TestCase):
    def test_zero_vector_returns_none(self):
        self.assertIsNone(aa.vector_to_arc_angles([0, 0, 0]))

    def test_vertical_and_horizontal_vectors(self):
        # Straight down/up is (0, 0)
        self.assertEqual(aa.vector_to_arc_angles([0, 0, 1]), (0.0, 0.0))
        self.assertEqual(aa.vector_to_arc_angles([0, 0, -1]), (0.0, 0.0))

        # Pure ML tilt
        rx, ry = aa.vector_to_arc_angles([1, 0, 1])
        self.assertTrue(_angle_close(rx, 0))
        self.assertTrue(_angle_close(ry, 45))

        # Pure AP (posterior) tilt is 10°  about x
        rx, ry = aa.vector_to_arc_angles(
            [0, math.sin(math.radians(10)), math.cos(math.radians(10))]
        )
        self.assertTrue(_angle_close(rx, 10))
        self.assertTrue(_angle_close(ry, 0))

    def test_round_trip_angles(self):
        """vec → angles → vec reproduces the original direction."""
        test_vecs = np.array(
            [
                [1, 2, 3],
                [-2, 0.5, 5.7],
                [0.3, 0.1, 1.0],
                [-1, -1, 4],
            ]
        )
        for vec in test_vecs:
            vec = vec / np.linalg.norm(vec)
            rx, ry = aa.vector_to_arc_angles(vec)
            vec_rt = aa.arc_angles_to_vector(rx, ry)
            self.assertTrue(
                _vec_close(vec, vec_rt),
                msg=f"Round-trip failed for vec {vec} → {(rx, ry)} → {vec_rt}",
            )


class TestArcAnglesToVector(unittest.TestCase):
    def test_deg_and_rad_inputs(self):
        # 30° ML, 45° AP        (degrees=True default)
        v_deg = aa.arc_angles_to_vector(rx=45, ry=30)
        # same in radians        (degrees=False)
        v_rad = aa.arc_angles_to_vector(
            rx=math.radians(45), ry=math.radians(30), degrees=False
        )
        self.assertTrue(_vec_close(v_deg, v_rad))

    def test_invert_ap_flag(self):
        """Flipping invert_AP changes the AP component sign."""
        v_default = aa.arc_angles_to_vector(20, 0)  # invert_AP=True
        v_no_flip = aa.arc_angles_to_vector(20, 0, invert_AP=False)
        # Only the AP (y) component should differ in sign
        self.assertAlmostEqual(v_default[0], v_no_flip[0], places=7)
        self.assertAlmostEqual(v_default[2], v_no_flip[2], places=7)
        self.assertAlmostEqual(v_default[1], -v_no_flip[1], places=7)


class TestVectorToStereotaxAngles(unittest.TestCase):
    def test_zero_vector_returns_none(self):
        self.assertIsNone(aa.vector_to_stereotax_angles([0, 0, 0]))

    def test_vertical_and_horizontal_vectors(self):
        # Straight down/up is (0, 0)
        self.assertEqual(aa.vector_to_stereotax_angles([0, 0, 1]), (0.0, 0.0))
        self.assertEqual(aa.vector_to_stereotax_angles([0, 0, -1]), (0.0, 0.0))

        # Pure ML tilt
        ry, rz = aa.vector_to_stereotax_angles([1, 0, 1])
        self.assertTrue(_angle_close(ry, 45))
        self.assertTrue(_angle_close(rz, 0))
        ry, rz = aa.vector_to_stereotax_angles([-1, 0, 1])
        self.assertTrue(_angle_close(ry, 45))
        self.assertTrue(_angle_close(rz, -180))

        # Compound rotation
        z = math.cos(math.radians(45))
        ry, rz = aa.vector_to_stereotax_angles(
            [math.cos(math.radians(10)) * z, math.sin(math.radians(10)) * z, z]
        )
        self.assertTrue(_angle_close(ry, 45))
        self.assertTrue(_angle_close(rz, 10))

    def test_round_trip(self):
        """vec → angles → vec reproduces the original direction."""
        test_vecs = np.array(
            [
                [1, 2, 3],
                [-0.2, 3.7, 0.9],
                [0.1, -0.3, 1.2],
            ]
        )
        for vec in test_vecs:
            vec = vec / np.linalg.norm(vec)
            ry, rz = aa.vector_to_stereotax_angles(vec)
            vec_rt = aa.stereotax_angles_to_vector(ry, rz)
            self.assertTrue(
                _vec_close(vec, vec_rt),
                msg=f"Round-trip failed for vec {vec} → {(ry, rz)} → {vec_rt}",
            )


class TestStereotaxAnglesToVector(unittest.TestCase):
    def test_deg_and_rad_inputs(self):
        # 30° ML, 45° AP        (degrees=True default)
        v_deg = aa.stereotax_angles_to_vector(45, 30)
        # same in radians        (degrees=False)
        v_rad = aa.stereotax_angles_to_vector(
            math.radians(45), math.radians(30), degrees=False
        )
        self.assertTrue(_vec_close(v_deg, v_rad))

    def test_zero_rz_to_the_left_flag(self):
        """Flipping zero_rz_to_left changes the rotation component sign."""
        v_default = aa.stereotax_angles_to_vector(
            20, 10
        )  # zero_rz_to_left=True
        v_flip = aa.stereotax_angles_to_vector(20, -170, zero_rz_to_left=True)
        # Only the rotation (z) component should differ in sign
        self.assertAlmostEqual(v_default[0], v_flip[0], places=7)
        self.assertAlmostEqual(v_default[1], v_flip[1], places=7)
        self.assertAlmostEqual(v_default[2], v_flip[2], places=7)


class TestArcAnglesToAffine(unittest.TestCase):
    def test_affine_matrix_contents(self):
        """
        Verify the XYZ Euler rotation sequence and the default
        invert_AP / invert_rotation logic.
        """
        AP, ML, ROT = 20, 30, 10

        # Expected rotation (after sign inversions inside the function)
        expected_R = (
            Rotation.from_euler("XYZ", [-AP, ML, -ROT], degrees=True)
            .as_matrix()
            .squeeze()
        )

        affine_R = ras_to_lps_transform(aa.arc_angles_to_affine(AP, ML, ROT))[
            0
        ]

        self.assertTrue(
            np.allclose(affine_R, expected_R, atol=1e-9),
            msg="Affine rotation matrix does not match expectation",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
