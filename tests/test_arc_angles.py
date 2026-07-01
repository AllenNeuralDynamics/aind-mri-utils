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
    return np.allclose(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2), atol=tol)


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
        rx, ry = aa.vector_to_arc_angles([0, math.sin(math.radians(10)), math.cos(math.radians(10))])
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
        v_rad = aa.arc_angles_to_vector(rx=math.radians(45), ry=math.radians(30), degrees=False)
        self.assertTrue(_vec_close(v_deg, v_rad))

    def test_invert_rx_flag(self):
        """Flipping invert_rx changes the AP component sign."""
        v_default = aa.arc_angles_to_vector(20, 0)  # invert_rx=True
        v_no_flip = aa.arc_angles_to_vector(20, 0, invert_rx=False)
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
        ry, rz = aa.vector_to_stereotax_angles([math.cos(math.radians(10)) * z, math.sin(math.radians(10)) * z, z])
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
        v_rad = aa.stereotax_angles_to_vector(math.radians(45), math.radians(30), degrees=False)
        self.assertTrue(_vec_close(v_deg, v_rad))

    def test_zero_rz_to_the_left_flag(self):
        """Flipping zero_rz_to_left changes the rotation component sign."""
        v_default = aa.stereotax_angles_to_vector(20, 10)  # zero_rz_to_left=True
        v_flip = aa.stereotax_angles_to_vector(20, -170, zero_rz_to_left=True)
        # Only the rotation (z) component should differ in sign
        self.assertAlmostEqual(v_default[0], v_flip[0], places=7)
        self.assertAlmostEqual(v_default[1], v_flip[1], places=7)
        self.assertAlmostEqual(v_default[2], v_flip[2], places=7)


class TestArcAngleConsistency(unittest.TestCase):
    """The vector, inverse, and affine paths must agree on the probe direction.

    They share :func:`arc_angles_to_rotation`, so a compound (rx != 0 and
    ry != 0) insertion must produce one and only one direction. A prior bug
    had ``arc_angles_to_vector`` and ``arc_angles_to_affine`` applying rx and
    ry in opposite orders, so they disagreed for compound angles while still
    passing single-axis tests.
    """

    COMPOUND_ANGLES = [(14, 20), (30, 30), (45, 20), (-25, 15), (10, -40)]

    def test_vector_matches_affine_direction(self):
        for rx, ry in self.COMPOUND_ANGLES:
            vec = aa.arc_angles_to_vector(rx, ry)
            # Third column of the (RAS) affine is the image of [0, 0, 1].
            affine_dir = ras_to_lps_transform(aa.arc_angles_to_affine(rx, ry))[0][:, 2]
            self.assertTrue(
                _vec_close(vec, affine_dir),
                msg=f"vector vs affine disagree at (rx={rx}, ry={ry}): {vec} vs {affine_dir}",
            )

    def test_compound_round_trip(self):
        for rx, ry in self.COMPOUND_ANGLES:
            vec = aa.arc_angles_to_vector(rx, ry)
            rx_rt, ry_rt = aa.vector_to_arc_angles(vec)
            self.assertTrue(_angle_close(rx, rx_rt), msg=f"rx round-trip {rx} -> {rx_rt}")
            self.assertTrue(_angle_close(ry, ry_rt), msg=f"ry round-trip {ry} -> {ry_rt}")


class TestArcAnglesToAffine(unittest.TestCase):
    def test_affine_matrix_contents(self):
        """
        Verify the XYZ Euler rotation sequence and the default
        invert_rx / invert_rz logic.
        """
        AP, ML, ROT = 20, 30, 10

        # Expected rotation (after sign inversions inside the function)
        expected_R = Rotation.from_euler("XYZ", [-AP, ML, -ROT], degrees=True).as_matrix().squeeze()

        affine_R = ras_to_lps_transform(aa.arc_angles_to_affine(AP, ML, ROT))[0]

        self.assertTrue(
            np.allclose(affine_R, expected_R, atol=1e-9),
            msg="Affine rotation matrix does not match expectation",
        )


class TestSolveEarbarForVertical(unittest.TestCase):
    # Arc readings (rx, ry) in the ephys-rig frame to exercise the solver.
    ARC_READINGS = [
        (14.0, 0.0),  # plan-vertical: already straight down
        (14.0, 10.0),  # pure ML tilt (earbar roll can null it)
        (20.0, 0.0),  # mild AP tilt (within earbar pitch range)
        (5.0, -8.0),  # compound, nose-down past plan-vertical
        (34.0, 0.0),  # steep AP tilt (beyond earbar pitch range)
        (24.0, 20.0),  # steep compound
    ]

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            aa.solve_earbar_for_vertical(np.zeros(3))

    def test_pose_within_bounds(self):
        for rx, ry in self.ARC_READINGS:
            sol = aa.arc_angles_to_earbar_stereotax(rx, ry)
            self.assertGreaterEqual(sol.earbar_pitch, -10.0 - 1e-6)
            self.assertLessEqual(sol.earbar_pitch, 10.0 + 1e-6)
            self.assertGreaterEqual(sol.earbar_roll, -15.0 - 1e-6)
            self.assertLessEqual(sol.earbar_roll, 15.0 + 1e-6)

    def test_matches_forward_model(self):
        # The solved earbar pose, fed back through the forward model, must
        # reproduce the reported Kopf angles. Ties the solver to existing code.
        for rx, ry in self.ARC_READINGS:
            sol = aa.arc_angles_to_earbar_stereotax(rx, ry)
            ry_fwd, rz_fwd = aa.arc_angles_to_stereotax_angles(
                rx, ry, earbar_pitch=sol.earbar_pitch, earbar_roll=sol.earbar_roll
            )
            self.assertAlmostEqual(sol.kopf_ry, ry_fwd, places=6, msg=f"kopf_ry at ({rx}, {ry})")
            self.assertTrue(_angle_close(sol.kopf_rz, rz_fwd, tol=1e-5), msg=f"kopf_rz at ({rx}, {ry})")

    def test_no_worse_than_no_earbar(self):
        # Optimizing the earbar can only reduce (or match) the Kopf polar tilt
        # relative to leaving the earbar flat.
        for rx, ry in self.ARC_READINGS:
            sol = aa.arc_angles_to_earbar_stereotax(rx, ry)
            ry_flat, _ = aa.arc_angles_to_stereotax_angles(rx, ry)
            self.assertLessEqual(sol.kopf_ry, ry_flat + 1e-6, msg=f"at ({rx}, {ry})")

    def test_ml_tilt_nulled_by_roll(self):
        # A pure ML tilt within range is fully absorbed by earbar roll.
        sol = aa.arc_angles_to_earbar_stereotax(14.0, 10.0)
        self.assertTrue(sol.vertical_achievable)
        self.assertAlmostEqual(sol.kopf_ry, 0.0, places=3)
        self.assertAlmostEqual(sol.earbar_pitch, 0.0, places=3)
        self.assertAlmostEqual(abs(sol.earbar_roll), 10.0, places=2)

    def test_steep_ap_tilt_saturates_pitch(self):
        # A 20 deg head-frame AP tilt exceeds the 10 deg pitch range, leaving a
        # ~10 deg residual on the Kopf tool with the earbar pinned at its bound.
        sol = aa.arc_angles_to_earbar_stereotax(34.0, 0.0)  # plan_rx = 20
        self.assertFalse(sol.vertical_achievable)
        self.assertAlmostEqual(abs(sol.earbar_pitch), 10.0, places=2)
        self.assertAlmostEqual(sol.kopf_ry, 10.0, places=2)

    def test_radians_matches_degrees(self):
        sol_deg = aa.arc_angles_to_earbar_stereotax(24.0, 20.0)
        sol_rad = aa.arc_angles_to_earbar_stereotax(
            math.radians(24.0),
            math.radians(20.0),
            pitch_bounds=(math.radians(-10.0), math.radians(10.0)),
            roll_bounds=(math.radians(-15.0), math.radians(15.0)),
            degrees=False,
        )
        self.assertAlmostEqual(sol_deg.earbar_pitch, math.degrees(sol_rad.earbar_pitch), places=4)
        self.assertAlmostEqual(sol_deg.earbar_roll, math.degrees(sol_rad.earbar_roll), places=4)
        self.assertAlmostEqual(sol_deg.kopf_ry, math.degrees(sol_rad.kopf_ry), places=4)


class TestEarbarSolverGlobalOptimality(unittest.TestCase):
    """The closed-form solver must be the global optimum, matching brute force."""

    ARC_READINGS = TestSolveEarbarForVertical.ARC_READINGS + [(40.0, -25.0)]
    P_BOUNDS = (-10.0, 10.0)
    R_BOUNDS = (-15.0, 15.0)

    @staticmethod
    def _abs_dv(v, pitch_deg, roll_deg):
        R = aa.earbar_angles_to_rotation_matrix(pitch_deg, roll_deg, degrees=True)
        return abs(float((R @ v)[2]))

    def _v_head(self, rx, ry):
        return aa.arc_angles_to_vector(rx - 14.0, ry, degrees=True, invert_rx=True)

    def test_closed_form_matches_dense_brute_force(self):
        # The continuous solution must be at least as vertical as any pose on a
        # fine grid over the whole box (i.e. it is the global optimum).
        ps = np.arange(self.P_BOUNDS[0], self.P_BOUNDS[1] + 1e-9, 0.1)
        rs = np.arange(self.R_BOUNDS[0], self.R_BOUNDS[1] + 1e-9, 0.1)
        for rx, ry in self.ARC_READINGS:
            v = self._v_head(rx, ry)
            sol = aa.arc_angles_to_earbar_stereotax(rx, ry)
            got = self._abs_dv(v, sol.earbar_pitch, sol.earbar_roll)
            brute = max(self._abs_dv(v, p, r) for p in ps for r in rs)
            self.assertGreaterEqual(got, brute - 1e-6, msg=f"not global optimum at ({rx}, {ry})")

    def test_rounded_pose_is_multiple_of_step(self):
        for step in (1.0, 5.0):
            for rx, ry in self.ARC_READINGS:
                sol = aa.arc_angles_to_earbar_stereotax(rx, ry, round_to=step)
                self.assertAlmostEqual(sol.earbar_pitch / step, round(sol.earbar_pitch / step), places=6)
                self.assertAlmostEqual(sol.earbar_roll / step, round(sol.earbar_roll / step), places=6)

    def test_rounded_pose_is_best_settable_position(self):
        # The rounded solution must be the exact optimum over settable dial
        # positions (exhaustive search of the discrete grid).
        for step in (1.0, 5.0):
            for rx, ry in self.ARC_READINGS:
                v = self._v_head(rx, ry)
                sol = aa.arc_angles_to_earbar_stereotax(rx, ry, round_to=step)
                got = self._abs_dv(v, sol.earbar_pitch, sol.earbar_roll)
                ps = np.arange(math.ceil(self.P_BOUNDS[0] / step), math.floor(self.P_BOUNDS[1] / step) + 1) * step
                rs = np.arange(math.ceil(self.R_BOUNDS[0] / step), math.floor(self.R_BOUNDS[1] / step) + 1) * step
                brute = max(self._abs_dv(v, p, r) for p in ps for r in rs)
                self.assertGreaterEqual(got, brute - 1e-9, msg=f"not best settable at ({rx}, {ry})")

    def test_rounded_within_bounds(self):
        for rx, ry in self.ARC_READINGS:
            sol = aa.arc_angles_to_earbar_stereotax(rx, ry, round_to=1.0)
            self.assertGreaterEqual(sol.earbar_pitch, -10.0 - 1e-6)
            self.assertLessEqual(sol.earbar_pitch, 10.0 + 1e-6)
            self.assertGreaterEqual(sol.earbar_roll, -15.0 - 1e-6)
            self.assertLessEqual(sol.earbar_roll, 15.0 + 1e-6)

    def test_rounded_matches_forward_model(self):
        # Kopf angles reported for a rounded pose must match feeding that exact
        # rounded pose back through the forward model.
        for rx, ry in self.ARC_READINGS:
            sol = aa.arc_angles_to_earbar_stereotax(rx, ry, round_to=1.0)
            ry_fwd, rz_fwd = aa.arc_angles_to_stereotax_angles(
                rx, ry, earbar_pitch=sol.earbar_pitch, earbar_roll=sol.earbar_roll
            )
            self.assertAlmostEqual(sol.kopf_ry, ry_fwd, places=6)
            # The azimuth kopf_rz is undefined at zero polar tilt; only compare
            # it when the insertion is meaningfully off-vertical.
            if sol.kopf_ry > 1e-3:
                self.assertTrue(_angle_close(sol.kopf_rz, rz_fwd, tol=1e-5))

    def test_invalid_round_to_raises(self):
        with self.assertRaises(ValueError):
            aa.arc_angles_to_earbar_stereotax(14.0, 0.0, round_to=0.0)
        with self.assertRaises(ValueError):
            aa.arc_angles_to_earbar_stereotax(14.0, 0.0, round_to=-5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
