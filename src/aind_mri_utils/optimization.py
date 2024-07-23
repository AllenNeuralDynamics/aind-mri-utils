# -*- coding: utf-8 -*-
"""
Functions for optimizing volume fits.

"""
import numpy as np
from scipy import optimize as opt

from . import rotations as rot
from .measurement import dist_point_to_line, dist_point_to_plane


def append_ones_column(data):
    """
    Add a column of ones to the end of a matrix.

    Parameters
    ----------
    data : np.array
        Data to append a column of ones to.

    Returns
    -------
    np.array
        Data with a column of ones appended to the end.
    """
    return rot.prepare_data_for_homogeneous_transform(data)


def _unpack_theta_apply_transform(theta, moving):
    """Helper function to apply a transform to a set of points."""
    R = rot.combine_angles(*theta[0:3])
    return rot.apply_rotate_translate(moving, R, theta[3:])


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
        lst = np.where(labels == ii)[0]
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
    labels : np.array(M,dtye=int)
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


def _unpack_theta_to_homogeneous(T):
    """Helper function to unpack theta to a homogeneous transform."""
    R = rot.combine_angles(*T[0:3])
    translation = T[3:]
    R_homog = rot.make_homogeneous_transform(R, translation)
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
        If None is passed, assumes all wieghts are 1.
        Default is None.
    xtol : float, optional
        Stopping tolerence for optimizer. The default is 1e-12.
    maxfun : int, optional
        Max number of function calls for optimizer.
        The default is 10000.
    normalize : bool, optional
        If True, normalize weights to be between 0 and 1.
        The default is False.
    gamma : float, optional
        If value is passed, weight gamma corrected for that value.
        The default is None.

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

    Tframe = opt.fmin(
        cost_function_weighted_labeled_lines,
        init,
        args=(pts1, pts2, positions, labels, weights),
        xtol=xtol,
        maxfun=maxfun,
    )

    print(Tframe)
    R_homog = _unpack_theta_to_homogeneous(Tframe)
    return R_homog, Tframe


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
        If None is passed, assumes all wieghts are 1.
        Default is None.
    xtol : float, optional
        Stopping tolerence for optimizer. The default is 1e-12.
    maxfun : int, optional
        Max number of function calls for optimizer.
        The default is 10000.
    normalize : bool, optional
        If True, normalize weights to be between 0 and 1.
        The default is False.
    gamma : float, optional
        If value is passed, weight gamma corrected for that value.
        The default is None.

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
    )

    output_b = opt.fmin(
        cost_function_weighted_labeled_lines_with_plane,
        output_a,
        args=(pts1, pts2, pts_for_line, positions, labels, weights),
        xtol=xtol,
        maxfun=maxfun,
    )

    Tframe = output_b
    R_homog = _unpack_theta_to_homogeneous(Tframe)
    return R_homog, Tframe


def get_headframe_hole_lines(
    version=0.1,
    insert_underscores=False,
    coordinate_system="LPS",
    return_plane=False,
):
    """
    Return the lines for the headframe holes,
    in a format that can be used by the cost function.

    Parameters
    ----------
    version : float, optional
        headframe version to match.
        The default is 0.1.,
        which corresponds to the first-gen zircona hole hemisphere headframe.
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
    if version == 0.1:
        if coordinate_system == "RAS":
            # Idealized Headframe holes
            ant_vert_hole_pts = np.array(
                [[5.1, -3.2, -1], [5.1, -3.2, 4]]
            )  # Z doesn't matter, just needs two points
            post_vert_hole_pts = np.array(
                [[6.85, -9.9, 0], [6.85, -9.9, 5]]
            )  # Z doesn't matter, just needs two points
            ant_hrz_hole_pts = np.array(
                [[6.34, 0, 2.5], [6.34, -6.5, 2.5]]
            )  # Y doesn't matter, just needs two points
            post_hrz_hole_pts = np.array(
                [[5.04, -6.5, 1], [5.04, -12, 1]]
            )  # Y doesn't matter, just needs two points
        elif coordinate_system == "LPS":
            ant_vert_hole_pts = np.array(
                [[-5.1, 3.2, -1], [-5.1, 3.2, 4]]
            )  # Z doesn't matter, just needs two points
            post_vert_hole_pts = np.array(
                [[-6.85, 9.9, 0], [-6.85, 9.9, 5]]
            )  # Z doesn't matter, just needs two points
            ant_hrz_hole_pts = np.array(
                [[-6.34, 0, 2.5], [-6.34, 6.5, 2.5]]
            )  # Y doesn't matter, just needs two points
            post_hrz_hole_pts = np.array(
                [[-5.04, 6.5, 1], [-5.04, 12, 1]]
            )  # Y doesn't matter, just needs two points

        if insert_underscores:
            names = [
                "anterior_horizontal",
                "anterior_vertical",
                "posterior_horizontal",
                "posterior_vertical",
            ]
        else:
            names = [
                "anterior horizontal",
                "anterior vertical",
                "posterior horizontal",
                "posterior vertical",
            ]

        if return_plane:
            # Last point is point on plane
            pts1 = np.array(
                [
                    ant_hrz_hole_pts[0, :],
                    ant_vert_hole_pts[0, :],
                    post_hrz_hole_pts[0, :],
                    post_vert_hole_pts[0, :],
                    [0, 0, 2],
                ],
            )
            # Last point is plane normal
            pts2 = np.array(
                [
                    ant_hrz_hole_pts[1, :],
                    ant_vert_hole_pts[1, :],
                    post_hrz_hole_pts[1, :],
                    post_vert_hole_pts[1, :],
                    [0, 0, 1],
                ]
            )
            names.append("plane")
        else:
            # Stack in a specific order
            pts1 = np.array(
                [
                    ant_hrz_hole_pts[0, :],
                    ant_vert_hole_pts[0, :],
                    post_hrz_hole_pts[0, :],
                    post_vert_hole_pts[0, :],
                ]
            )
            pts2 = np.array(
                [
                    ant_hrz_hole_pts[1, :],
                    ant_vert_hole_pts[1, :],
                    post_hrz_hole_pts[1, :],
                    post_vert_hole_pts[1, :],
                ]
            )

        return pts1, pts2, names
    else:
        raise ValueError("Version not supported")
