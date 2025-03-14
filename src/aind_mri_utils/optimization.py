# -*- coding: utf-8 -*-
"""
Functions for optimizing volume fits.

"""

import logging

import numpy as np
from scipy import optimize as opt

from . import rotations as rot
from .measurement import dist_point_to_line, dist_point_to_plane

logger = logging.getLogger(__name__)

headframe_hole_locations = {
    "0.1": {
        "RAS": {
            "anterior_horizontal": np.array(
                [[6.34, 0, 2.5], [6.34, -6.5, 2.5]]
            ),
            "anterior_vertical": np.array([[5.1, -3.2, -1], [5.1, -3.2, 4]]),
            "posterior_horizontal": np.array(
                [[5.04, -6.5, 1], [5.04, -12, 1]]
            ),
            "posterior_vertical": np.array([[6.85, -9.9, 0], [6.85, -9.9, 5]]),
        }
    }
}

headframe_plane_location = {"0.1": {"RAS": np.array([[0, 0, 2], [0, 0, 1]])}}

RAS_LPS_conversion_factor = np.array([-1, -1, 1])
for version in headframe_hole_locations:
    lps_dict = dict()
    for key, val in headframe_hole_locations[version]["RAS"].items():
        lps_dict[key] = val * RAS_LPS_conversion_factor
    headframe_hole_locations[version]["LPS"] = lps_dict
    headframe_plane_location[version]["LPS"] = (
        RAS_LPS_conversion_factor * headframe_plane_location[version]["RAS"]
    )


def _unpack_theta_apply_transform(theta, moving):
    """Helper function to apply a transform to a set of points."""
    R = rot.combine_angles(*theta[0:3])
    return rot.apply_rotate_translate(moving, R, theta[3:])


def revised_error_rotate_compare_weighted_lines(
    theta, pts1, pts2, moving, weights, group_err_funs=None
):
    """
    Calculate the error of rotating and translating the `moving` points to
    align with `pts1` and `pts2`, taking into account the weights assigned to
    each point.

    Parameters
    ----------
    theta : array-like
        The rotation angles and translation parameters.
    pts1 : list
        List of points to align with.
    pts2 : list
        List of corresponding points to align with.
    moving : list
        List of points to be transformed.
    weights : list
        List of weights assigned to each point.
    group_err_funs : list, optional
        List of error functions for each group of points. If not provided, the
        default error function `dist_point_to_line` will be used for all
        groups.

    Returns
    -------
    error : float
        The calculated error.

    Raises
    ------
    ValueError
        If the lengths of `pts1`, `pts2`, `moving`, and `weights` are not the
        same.
    ValueError
        If the length of `group_err_funs` is not the same as the number of
        groups in `pts1`.
    """
    n_group = len(pts1)
    if not all(len(lst) == n_group for lst in [pts2, moving, weights]):
        raise ValueError("pts1, pts2, moving, and weights must be same length")
    if group_err_funs is None:
        group_err_funs = np.full(n_group, dist_point_to_line)
    else:
        if len(group_err_funs) != n_group:
            raise ValueError(
                "group_err_fun must have the same number of groups as pts1"
            )

    R = rot.combine_angles(*theta[0:3])
    translation = theta[3:]
    error = 0.0
    for f, p1, p2, m, w in zip(group_err_funs, pts1, pts2, moving, weights):
        transformed = rot.apply_rotate_translate(m, R, translation)
        for pt_no in range(m.shape[0]):
            res = f(p1, p2, transformed[pt_no, :])
            error += res * w[pt_no]
    return error


def cost_function_weighted_labeled_lines(
    T, pts1, pts2, moving, labels, weights
):
    """
    Cost function for optimizing a rigid transform on weighted points.

    Parameters
    ----------
    T : np.array(6,)
        Rigid transform parameters.
    pts1 : np.array(N,3)
        Point on line
    pts2 : np.array(N,3)
        Second point on line
    moving : np.array(M,3)
        Position of points to be transformed.
    labels : np.array(M,dtype=int)
        Labels of points, corresponding to index in
        pts1,pts2,and pts_for_line.
    weights : np.array(M,)
        Weights of points

    Returns
    -------
    Distance
        Sum of distances between points all points and
        their corresponding lines, weighted by weights.

    """
    transformed = _unpack_theta_apply_transform(T, moving)

    D = np.zeros((moving.shape[0], 1))
    for ii in range(pts1.shape[0]):
        lst = np.nonzero(labels == ii)[0]
        for jj in range(len(lst)):
            D[lst[jj]] = (
                dist_point_to_line(
                    pts1[ii, :], pts2[ii, :], transformed[lst[jj], :]
                )
                * weights[lst[jj]]
            )

    return np.sum(D)


def cost_function_weighted_labeled_lines_with_plane(
    T, pts1, pts2, pts_for_line, moving, labels, weights
):
    """
    Cost function for optimizing a rigid transform on weighted points;
    includes labeled lines and labeled planes.

    Parameters
    ----------
    T : np.array(6,)
        Rigid transform parameters.
    pts1 : np.array(N,3)
        Point on line
    pts2 : np.array(N,3)
        Second point on line
    pts_for_line : List[bool] or np.array(N,dtype=bool)
        True if the line should be used for distance calculation,
        False if the plane should be used.
    moving : np.array(M,3)
        position of points to be transformed.
    labels : np.array(M,dtype=int)
        Labels of points, corresponding to index in pts1,pts2,and pts_for_line.
    weights : np.array(M,)

    Returns
    -------
    Distance
        Sum of distances between points all points and
        their corresponding line or plane, weighted by weights.
    """
    transformed = _unpack_theta_apply_transform(T, moving)

    D = np.zeros((moving.shape[0], 1))
    for ii in range(pts1.shape[0]):
        lst = np.where(labels == ii)[0]
        if pts_for_line[ii]:
            for jj in range(len(lst)):
                D[lst[jj]] = (
                    dist_point_to_line(
                        pts1[ii, :], pts2[ii, :], transformed[lst[jj], :]
                    )
                    * weights[lst[jj]]
                )
        else:
            for jj in range(len(lst)):
                D[lst[jj]] = (
                    dist_point_to_plane(
                        pts1[ii, :], pts2[ii, :], transformed[lst[jj], :]
                    )
                    * weights[lst[jj]]
                )

    return np.sum(D)


def _preprocess_weights(weights, positions, normalize, gamma):
    """
    Preprocess weights for use in optimization functions
    """
    if weights is None:
        weights = np.ones((positions.shape[0], 1))
    else:
        # Gamma correct
        # Taken from skimage.exposure.adjust_gamma.
        # Implementing here to avoid importing skimage.
        scale = np.max(weights) - np.min(weights)
        if abs(scale) > 1e-6:
            if gamma is not None:
                if abs(scale) > 1e-6:
                    weights = ((weights / scale) ** gamma) * scale

            if normalize:
                weights = (weights - np.min(weights)) / (scale)
    return weights


def unpack_theta(T):
    """Helper function to unpack theta to a rigid transform."""
    R = rot.combine_angles(*T[0:3])
    translation = T[3:]
    return R, translation


def unpack_theta_to_homogeneous(T):
    """Helper function to unpack theta to a homogeneous transform."""
    R_homog = rot.make_homogeneous_transform(*unpack_theta(T))
    return R_homog


def optimize_transform_labeled_lines(
    init,
    pts1,
    pts2,
    positions,
    labels,
    weights=None,
    xtol=1e-12,
    maxfun=10000,
    normalize=False,
    gamma=None,
    disp=0,
):
    """
    Function for optimizing a rigid transform on
    weighted points by minimizing distance
    from each point to a specified line.
    Multiple lines can be specified by using labels.

    Parameters
    ----------
    init : np.array(6,1)
        Initial transform, as 6x1 matrix.
    pts1 : np.array(N,3)
        First point on each line
    pts2 : np.array(N,3)
        Second point on each line.
    positions : np.array(M,3)
        Positions of points to optimize on.
    labels : np.array(M,1)
        Labels for each point in positions,
        specifying which line that point too.
    weights : np.array(M,1), optional
        Weights for each point in positions.
        If None is passed, assumes all weights are 1.
        Default is None.
    xtol : float, optional
        Stopping tolerance for optimizer. The default is 1e-12.
    maxfun : int, optional
        Max number of function calls for optimizer.
        The default is 10000.
    normalize : bool, optional
        If True, normalize weights to be between 0 and 1.
        The default is False.
    gamma : float, optional
        If value is passed, weight gamma corrected for that value.
        The default is None.
    disp : int, optional
        If 0, no output. If 1, output. The default is 0.

    Returns
    -------
    trans: np.array(4,4)
        Rigid transform matrix that minimizes the cost function.
    T: np.array(6,1)
        Parameters of the rigid transform matrix
        that minimizes the cost function.
    output: tuple
        Fitting data from scipy.optimize.fmin
        (see retol in scipy.optimize.fmin documentation)

    """

    weights = _preprocess_weights(weights, positions, normalize, gamma)

    T_frame = opt.fmin(
        cost_function_weighted_labeled_lines,
        init,
        args=(pts1, pts2, positions, labels, weights),
        xtol=xtol,
        maxfun=maxfun,
        disp=disp,
    )

    logger.debug(T_frame)
    R_homog = unpack_theta_to_homogeneous(T_frame)
    return R_homog, T_frame


def optimize_transform_labeled_lines_with_plane(
    init,
    pts1,
    pts2,
    pts_for_line,
    positions,
    labels,
    weights=None,
    xtol=1e-12,
    maxfun=10000,
    normalize=False,
    gamma=None,
    disp=0,
):
    """
    Function for optimizing a rigid transform on
    weighted points by minimizing distance
    between each point and a specified line or plane.
    Multiple lines/planes can be specified by using labels.

    Parameters
    ----------
    init : np.array(6,1)
        Initial transform, as 6x1 matrix.
    pts1 : np.array(N,3)
        First point on each line OR plane normal
    pts2 : np.array(N,3)
        Second point on each line OR point on plane
    pts_for_line : np.array(N,1)
        Boolean array specifying if each line is a line or a plane.
    positions : np.array(M,3)
        Positions of points to optimize on.
    labels : np.array(M,dtype=np.int))
        Labels for each point in positions,
        specifying which line that point too.
    weights : np.array(M,1), optional
        Weights for each point in positions.
        If None is passed, assumes all weights are 1.
        Default is None.
    xtol : float, optional
        Stopping tolerance for optimizer. The default is 1e-12.
    maxfun : int, optional
        Max number of function calls for optimizer.
        The default is 10000.
    normalize : bool, optional
        If True, normalize weights to be between 0 and 1.
        The default is False.
    gamma : float, optional
        If value is passed, weight gamma corrected for that value.
        The default is None.
    disp : int, optional
        If 0, no output. If 1, output. The default is 0.

    Returns
    -------
    trans: np.array(4,4)
        Rigid transform matrix that minimizes the cost function.
    T: np.array(6,1)
        Parameters of the rigid transform matrix
        that minimizes the cost function.
    """
    weights = _preprocess_weights(weights, positions, normalize, gamma)

    output_a = opt.fmin(
        cost_function_weighted_labeled_lines,
        init,
        args=(pts1, pts2, positions, labels, weights),
        xtol=xtol,
        maxfun=maxfun,
        disp=disp,
    )

    output_b = opt.fmin(
        cost_function_weighted_labeled_lines_with_plane,
        output_a,
        args=(pts1, pts2, pts_for_line, positions, labels, weights),
        xtol=xtol,
        maxfun=maxfun,
        disp=disp,
    )

    T_frame = output_b
    R_homog = unpack_theta_to_homogeneous(T_frame)
    return R_homog, T_frame


def get_headframe_hole_lines(
    version="0.1",
    insert_underscores=False,
    coordinate_system="LPS",
    return_plane=False,
):
    """
    Return the lines for the headframe holes,
    in a format that can be used by the cost function.

    Parameters
    ----------
    version : string, optional
        headframe version to match.
        The default is "0.1".,
        which corresponds to the first-gen zirconia hole hemisphere headframe.
    insert_underscores : bool, optional
        If true, insert underscores into the names of the lines.
        The default is False.
    coordinate_system : str, optional
        Coordinate system to return the lines in. The default is 'LPS'.
    return_plane : bool, optional
        Return point for horizontal plane as last point in list
        Default is False

    Returns
    -------
    pts1 : np.array
        One point on each line, in headframe coordinates.
    pts2 : np.array
        Another point on each line.
    names : list
        Name of each line.
    """
    if version not in headframe_hole_locations:
        raise ValueError("Version not supported")
    if coordinate_system not in headframe_hole_locations[version]:
        raise ValueError("Coordinate system not supported")

    names = []
    pts1 = []
    pts2 = []
    pt_dict = headframe_hole_locations[version][coordinate_system]
    for name, pts in pt_dict.items():
        if insert_underscores:
            store_name = name
        else:
            store_name = name.replace("_", " ")
        names.append(store_name)
        pts1.append(pts[0, :])
        pts2.append(pts[1, :])
    if return_plane:
        names.append("plane")
        headframe_pts = headframe_plane_location[version][coordinate_system]
        pts1.append(headframe_pts[0, :])
        pts2.append(headframe_pts[1, :])
    pts1 = np.vstack(pts1)
    pts2 = np.vstack(pts2)
    return pts1, pts2, names
