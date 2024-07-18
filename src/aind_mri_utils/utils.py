"""Utility functions"""

import math

import numpy as np
from numpy import linalg


def skew_symmetric_cross_product_matrix(v):
    """Find the cross product matrix for a vector v"""
    return np.cross(v, np.identity(v.shape[0]) * -1)


def norm_vec(vec):
    """Normalize input vector"""
    n = np.linalg.norm(vec)
    if n == 0:
        raise ValueError("Input has norm of zero")
    return vec / n


def vector_rejection(v, n):
    """Find the component of v orthogonal to n"""
    ndim = n.size
    nn = norm_vec(n)
    vn = (v.reshape(-1, ndim) @ nn[:, np.newaxis]) * nn[np.newaxis, :]
    return v - vn


def mask_arr_by_annotations(arr, anno_arr, seg_vals, default_val=0):
    """Sets entries of arr to default_val if anno_arr not in target set

    This function will return a copy of `arr` where the output is either
    the same as `arr` if the corresponding element of `anno_arr` is one of
    `seg_vals`, or `default_val` if not.

    Parameters
    ----------
    arr : numpy.ndarray
        Array that will be masked
    anno_arr : numpy.ndarray
        Array same size as `arr` that assigns each element to a segment
    seg_vals : set like
        Set of values that anno_arr will be compared to
    default_val : number
        value of output array if anno_arr is not in seg_vals, default = 0.

    Returns
    -------
    masked_vol : numpy.ndarray
        Copy of `arr` masked by whether `anno_arr` is one of `seg_vals`
    """

    masked_arr = np.zeros_like(arr)
    masked_arr.fill(default_val)
    mask = np.isin(anno_arr, seg_vals)
    masked_arr[mask] = arr[mask]
    return masked_arr


def find_indices_equal_to(arr, v):
    """Find array indices equal to v"""
    return np.column_stack(np.nonzero(arr == v))


def get_first_pca_axis(pts):
    """Find first PC of points"""
    centered = pts - np.mean(pts, axis=0)[np.newaxis, :]
    _, _, vh = linalg.svd(centered, full_matrices=False)
    return vh[0, :]


def signed_angle_rh(a, b, n):
    """find right-handed angle between two vectors
    Find the right-handled angle between a and b in the plane normal to n,
    by rotating a to b
    """
    # Function by Adrian Leonhard: https://stackoverflow.com/a/33920320
    # Let alpha be the direct angle between the vectors (0° to 180°) and beta
    # the angle we are looking for (0° to 360°) with beta == alpha or
    # beta == 360° - alpha
    #
    # Va . Vb == |Va| * |Vb| * cos(alpha) (by definition)
    #         == |Va| * |Vb| * cos(beta)
    # As cos(alpha) == cos(-alpha) == cos(360° - alpha)
    #
    #
    # Va x Vb == |Va| * |Vb| * sin(alpha) * n1
    # (by definition; n1 is a unit vector perpendicular to Va and Vb with
    # orientation matching the right-hand rule)
    #
    # Therefore (again assuming Vn is normalized):
    # n1 . Vn == 1 when beta < 180
    # n1 . Vn == -1 when beta > 180
    #
    # ==>  (Va x Vb) . Vn == |Va| * |Vb| * sin(beta)
    # Finally,
    # tan(beta) = sin(beta) / cos(beta) == ((Va x Vb) . Vn) / (Va . Vb)

    vn = norm_vec(n)
    return math.atan2(np.dot(np.cross(a, b), vn), np.dot(a, b))


def signed_angle_lh(a, b, n):
    """find left-handed angle between two vectors

    See `signed_angle_rh`
    """
    return signed_angle_rh(b, a, n)


def unsigned_angle(a, b):
    """
    Calculate the unsigned angle between two vectors.

    Parameters
    ----------
    a : array-like
        First vector.
    b : array-like
        Second vector.

    Returns
    -------
    angle : float
        The unsigned angle between the two vectors in radians.
    """
    an = norm_vec(a)
    bn = norm_vec(b)
    return np.arccos(np.clip(np.dot(an, bn), -1.0, 1.0))
