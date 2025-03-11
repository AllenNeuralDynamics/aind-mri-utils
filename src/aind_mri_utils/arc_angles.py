"""
Tools specific to computing arc angles
"""

import numpy as np
from scipy.spatial.transform import Rotation

from aind_mri_utils.rotations import ras_to_lps_transform


def calculate_arc_angles(vec, degrees=True, invert_AP=True):
    """
    Calculate the arc angles for a given vector.

    Parameters
    ----------
    vec : array_like
        A 3-element vector.

    Returns
    -------
    tuple of float
        The calculated arc angles in degrees. The first element is the angle
        around the x-axis, and the second element is the angle around the
        y-axis.  Returns None if the input vector is a zero vector.
    """
    if np.linalg.norm(vec) == 0:
        return None
    if np.dot(vec, [0, 0, 1]) < 0:
        vec = -vec
    nv = vec / np.linalg.norm(vec)
    rx = -np.arcsin(nv[1])  # Our convention is backwards from right hand rule
    ry = np.arctan2(nv[0], nv[2])
    if degrees:
        rx = np.rad2deg(rx)
        ry = np.rad2deg(ry)
    if invert_AP:
        rx = -rx
    return rx, ry


def transform_matrix_from_angles(
    AP, ML, rotation=0, invert_AP=True, invert_rotation=True
):
    """
    Create a transform from arc angles

    Note that our convention for spin about the X axis (AP) is not
    right-handed, so use `invert_AP=True` to correct for this.

    Same with rotation
    """
    if invert_AP:
        AP = -AP
    if invert_rotation:
        rotation = -rotation
    euler_angles = np.array([AP, ML, rotation])
    R = (
        Rotation.from_euler("XYZ", euler_angles, degrees=True)
        .as_matrix()
        .squeeze()
    )
    return ras_to_lps_transform(R)[0]
