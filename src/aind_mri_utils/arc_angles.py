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
        A 3-element vector with ML, AP, and DV components. Directions should be
        in RAS.

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
    # using trig identity to get the angle from vertical
    rx = -np.arcsin(nv[1])
    ry = np.arctan2(nv[0], nv[2])
    if degrees:
        rx = np.rad2deg(rx)
        ry = np.rad2deg(ry)
    if invert_AP:
        rx = -rx
    return rx, ry


def calculate_vector_from_arc_angles(
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
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
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


def calculate_stereotax_angles(vec, degrees=True):
    """
    Calculate the stereotaxic angles for a given vector.

    Parameters
    ----------
    vec : array_like
        A 3-element vector with ML, AP, and DV components. Directions should be
        in RAS.
    degrees : bool, optional
        If True, return angles in degrees (default is True).
    mount_on_left : bool, optional
        If True, assume the mount is on the left side (default is True).

    Returns
    -------
    tuple of float
        The calculated stereotaxic angles in degrees. The first element is the
        angle around the y-axis (AP), and the second element is the angle
        around the z-axis (DV).  Returns None if the input vector is a zero
        vector.
    """
    if np.linalg.norm(vec) == 0:
        return None
    if np.dot(vec, [0, 0, 1]) < 0:
        vec = -vec
    nv = vec / np.linalg.norm(vec)
    ry = np.arccos(nv[2])
    rz = np.arctan2(nv[1], nv[0]) - np.pi
    if degrees:
        ry = np.rad2deg(ry)
        rz = np.rad2deg(rz)
    return ry, rz


def transform_matrix_from_angles(
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
