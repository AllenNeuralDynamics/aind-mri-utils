"""Utility functions"""
import numpy as np


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


def transform_sitk_indices_to_physical_points(simage, index_arr):
    """Transforms indices indices of simage to physical points

    For a SimpleITK image `simage` and a list of indices `index_arr`, transform
    each index to a physical point.

    Parameters
    ----------
    simage : M-d SimpleITK image
    index_arr : numpy.ndarray (NxM)
        matrix of indices of `simage`, where each row is an index

    Returns
    -------
    position_arr: numpy.ndarray (NxM)
        matrix of physical points for each index in `index_arr`
    """
    position_arr = np.zeros_like(index_arr, dtype="float32")
    npt = index_arr.shape[0]
    for ptno in range(npt):
        ndx = tuple(map(lambda x: x.item(), index_arr[ptno, :]))
        position_arr[ptno, :] = simage.TransformContinuousIndexToPhysicalPoint(ndx)
    return position_arr
