"""
Tools specific to computing arc angles
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from aind_mri_utils.rotations import ras_to_lps_transform


def vector_to_arc_angles(vec, degrees=True, invert_AP=True):
    """
    Calculate the arc angles for a given vector.

    Parameters
    ----------
    vec : array_like
        A 3-element vector with ML, AP, and DV components. Directions should be
        in RAS.

    Returns
    -------
    tuple of float
        The calculated arc angles in degrees. The first element is the angle
        around the x-axis, and the second element is the angle around the
        y-axis.  Returns None if the input vector is a zero vector.
    """
    vec = np.asarray(vec)
    if np.linalg.norm(vec) == 0:
        return None
    if np.dot(vec, [0, 0, 1]) < 0:
        vec = -vec
    nv = vec / np.linalg.norm(vec)
    # using trig identity to get the angle from vertical
    rx = -np.arcsin(nv[1])
    ry = np.arctan2(nv[0], nv[2])
    if degrees:
        rx = math.degrees(rx)
        ry = math.degrees(ry)
    if invert_AP:
        rx = -rx
    return rx, ry


def arc_angles_to_vector(
    rx, ry, degrees=True, invert_AP=True, invert_rotation=True
):
    """
    Calculate a vector from arc angles.

    Parameters
    ----------
    rx : float
        The angle around the x-axis (anterior-posterior).
    ry : float
        The angle around the y-axis (medial-lateral).
    degrees : bool, optional
        If True, input angles are in degrees (default is True).
    invert_AP : bool, optional
        If True, invert the AP angle to correct for the non-right-handed
        convention (default is True).
    invert_rotation : bool, optional
        If True, invert the rotation angle (default is True).

    Returns
    -------
    numpy.ndarray
        A 3-element vector with ML, AP, and DV components.
    """
    if degrees:
        rx = math.radians(rx)
        ry = math.radians(ry)
    if invert_AP:
        rx = -rx

    vec = np.array(
        [
            np.sin(ry) * np.cos(rx),  # ML component
            -np.sin(rx),  # AP component using trig identity
            np.cos(ry) * np.cos(rx),  # DV component
        ]
    )
    return vec / np.linalg.norm(vec)


def vector_to_stereotax_angles(vec, degrees=True, zero_rz_to_left=False):
    """
    Calculate the stereotaxic angles for a given vector.

    Used for Kopf 1500 off-plane insertion tool.

    Parameters
    ----------
    vec : array_like
        A 3-element vector with ML, AP, and DV components. Directions should be
        in RAS.
    degrees : bool, optional
        If True, return angles in degrees (default is True).
    zero_rz_to_left : bool, optional
        If True, assume the zero DV angle points to the left, Otherwise, zero
        DV points to the right (default is False).

    Returns
    -------
    tuple of float
        The calculated stereotaxic angles in degrees. The first element is the
        angle around the y-axis (AP), with zero pointing up, and the second
        element is the angle around the z-axis (DV), with zero pointing to the
        right if zero_rz_to_left is False.  Returns None if the input vector is
        a zero vector.
    """
    vec = np.asarray(vec)
    if np.linalg.norm(vec) == 0:
        return None
    if np.dot(vec, [0, 0, 1]) < 0:
        vec = -vec
    nv = vec / np.linalg.norm(vec)
    ry = np.arccos(nv[2])
    rz = np.arctan2(nv[1], nv[0])
    if zero_rz_to_left:
        # Adjust the angle so that zero DV points to the left
        rz = (rz + 2 * np.pi) % (2 * np.pi) - np.pi  # Ensure rz is in [-π, π)
    if degrees:
        ry = math.degrees(ry)
        rz = math.degrees(rz)
    return ry, rz


def stereotax_angles_to_vector(ry, rz, degrees=True, zero_rz_to_left=False):
    """Calculate a vector from stereotaxic angles.

    Used for Kopf 1500 off-plane insertion tool.

    Parameters
    ----------
    ry : float
        The angle around the y-axis (AP).
    rz : float
        The angle around the z-axis (DV).
    degrees : bool, optional
        If True, input angles are in degrees (default is True).
    zero_rz_to_left : bool, optional
        If True, assume the zero DV angle points to the left, Otherwise, zero
        DV points to the right (default is False).

    Returns
    -------
    numpy.ndarray
        A 3-element vector with ML, AP, and DV components.
    """
    if degrees:
        ry = math.radians(ry)
        rz = math.radians(rz)
    if zero_rz_to_left:
        # Adjust the angle so that zero DV points to the right
        rz = (rz + 2 * np.pi) % (2 * np.pi) - np.pi

    vec = np.array(
        [
            np.cos(rz) * np.sin(ry),  # ML component
            np.sin(rz) * np.sin(ry),  # AP component using trig
            np.cos(ry),  # DV component
        ]
    )
    return vec / np.linalg.norm(vec)


def arc_angles_to_affine(
    AP, ML, rotation=0, invert_AP=True, invert_rotation=True
):
    """
    Create a transform from arc angles.

    Parameters
    ----------
    AP : float
        The angle around the x-axis (anterior-posterior).
    ML : float
        The angle around the y-axis (medial-lateral).
    rotation : float, optional
        The rotation angle around the z-axis (default is 0).
    invert_AP : bool, optional
        If True, invert the AP angle to correct for the non-right-handed
        convention (default is True).
    invert_rotation : bool, optional
        If True, invert the rotation angle (default is True).

    Returns
    -------
    numpy.ndarray
        The transformation matrix.

    Notes
    -----
    Our convention for spin about the x-axis (AP) and the z-axis (DV) is not
    right-handed, so use `invert_AP=True` and `invert_rotation=True`,
    respectively, to correct for this.
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
