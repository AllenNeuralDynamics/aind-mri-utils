"""
Tools specific to computing arc angles
"""

import numpy as np

from scipy.spatial.transform import Rotation

from .rotations import make_homogeneous_transform


def calculate_arc_angles(target_pt, extra_pt, ap_offset=14, degrees=True):
    """
    Compute the arc angles needed for a probe trajectory that intersects 2
    points.

    Note that order matters on the points;
    currently "target" is the intended deep point and "extra" is a point
    at/above the surface.

    This should probably have some coordinate system awareness, and this
    documentation should be expanded to show logic.

    # Returns AP, ML angle
    """
    this_vector = (extra_pt - target_pt) / np.linalg.norm(extra_pt - target_pt)
    phi = np.arcsin(this_vector[0])
    theta = np.arcsin(-this_vector[1] / np.cos(phi))
    return np.rad2deg(theta) + ap_offset, -np.rad2deg(phi)


def transform_matrix_from_angles_and_target(AP, ML, target, rotation=0):
    euler_angles = np.array([np.deg2rad(x) for x in [AP, ML, rotation]])
    R = Rotation.from_euler("XYZ", euler_angles).as_matrix().squeeze()
    return make_homogeneous_transform(R, target)
