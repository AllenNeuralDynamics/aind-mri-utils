"""
Tools specific to computing arc angles
"""

import numpy as np


def arc_angles_to_hit_two_points(
    target_pt, extra_pt, ap_offset=14, degrees=True
):
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
