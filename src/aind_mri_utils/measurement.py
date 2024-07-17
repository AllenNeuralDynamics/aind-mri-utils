"""
Measurement code.


This applies to find_circle_center, which is borrowed from scipy-cookbooks:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2017 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""

import itertools as itr

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from . import rotations as rot
from . import sitk_volume as sv
from . import utils as ut


def find_circle(x, y):
    """
    Fit a circle to a set of points using a linearized least-squares algorithm

    Borrowed, with modification, from:
    https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    Parameters
    ----------
    x : (N) array
        X values of sample points.
    y : (N) array
        Y values of sample points.

    Returns
    -------
    xc_1 : Float
        X coordinate of center
    yc_1 : Float
        Y coordinate of center.
    radius : Float
        Radius of fit circle.
    residu_1 : (N) array
        per-point residual.

    """
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # We will find the center (uc, vc) by solving the following
    # linear system.
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    # Set up:
    Suv = sum(u * v)
    Suu = sum(u**2)
    Svv = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3)

    # And Solve!
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    #
    xc_1 = x_m + uc
    yc_1 = y_m + vc

    Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    radius = np.mean(Ri_1)

    return xc_1, yc_1, radius


def find_line_eig(points):
    """
    Returns first normalized eigenvetor of data, for use in line fitting.

    Parameters
    ----------
    points : NxD numpy array
        Points to fit a line through

    Returns
    -------
    (D,) numpy array
        norm of line (eigenvector)
    points_mean : (D,)
        Average value

    """
    points_mean = np.mean(points, axis=0)
    a, b = np.linalg.eig(np.cov((points - points_mean).T))
    return b[:, 0], points_mean


def closet_points_on_two_lines(P1, V1, P2, V2):
    """
    Taken, with modification, from:
    https://math.stackexchange.com/questions/846054/...
        closest-points-on-two-line-segments

    """
    P1 = np.array(P1)
    V1 = np.array(V1)
    P2 = np.array(P2)
    V2 = np.array(V2)
    V21 = P2 - P1

    v22 = np.dot(V2, V2)
    v11 = np.dot(V1, V1)
    v21 = np.dot(V2, V1)
    v21_1 = np.dot(V21, V1)
    v21_2 = np.dot(V21, V2)
    denom = v21 * v21 - v22 * v11

    if np.isclose(denom, 0.0):
        s = 0.0
        t = (v11 * s - v21_1) / v21
    else:
        s = (v21_2 * v21 - v22 * v21_1) / denom
        t = (-v21_1 * v21 + v11 * v21_2) / denom

    p_a = P1 + s * V1
    p_b = P2 + t * V2
    return p_a, p_b


def angle(v1, v2):
    """
    Angle (in degrees) between two vectors

    Parameters
    ----------
    v1 : numpy array
        First Vector.
    v2 : numpy array
        Second Vector.

    Returns
    -------
    Angle between vectors
    """
    rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.rad2deg(rad)


def dist_point_to_line(pt_1, pt_2, query_pt):
    """
    Distance between line defined by two points and a query point
    insperation from:
        https://stackoverflow.com/questions/39840030/...
        distance-between-point-and-a-line-from-two-points

    Parameters
    ----------
    pt_1 : numpy array  (N,)
        First Vector.
    pt_2 : numpy array (N,)
        Second Vector.
    query_pt: numpy array (N,)
        Point to find distance of.

    Returns
    -------
    Distance
    """
    ln_pt = pt_1
    ln_norm = pt_1 - pt_2
    ln_norm = ln_norm / np.linalg.norm(ln_norm)
    return np.abs(
        np.linalg.norm(np.cross(ln_norm, ln_pt - query_pt))
    ) / np.linalg.norm(ln_norm)


def dist_point_to_plane(pt_0, normal, query_pt):
    """
    Distance between plane defined by point and normal and a query point

    Parameters
    ----------
    pt_0 : numpy array  (N,)
        Point on plane.
    normal : numpy array (N,)
        Normal vector of plane.
    query_pt: numpy array (N,)
        Point to find distance of.

    Returns
    -------
    Distance
    """
    D = -normal[0] * pt_0[0] - normal[1] * pt_0[1] - normal[2] * pt_0[2]
    num = np.abs(
        normal[0] * query_pt[0]
        + normal[1] * query_pt[1]
        + normal[2] * query_pt[2]
        + D
    )
    denom = np.sqrt(np.sum(normal**2))
    return num / denom


def get_segmentation_pca(seg_img, seg_vals):
    """Finds first pca axis of segmentation for segments in set seg_vals

    For each value in seg_vals, this will find the indices of seg_arr equal to
    that value and de-mean it. The first PC will then be found of the
    concatenated centered groups of indices.

    Parameters
    ---------
    seg_arr : SimpleITK.Image
        Array with annotation values in each index
    seg_vals : iterable
        Set of values that each element of seg_arr will be compared to.

    Returns
    -------
    pc_axis : first pc axis of the indices separately centered for each
    """
    # Centers each segmentation value separately
    centered = []
    for seg_val in seg_vals:
        p = sv.find_points_equal_to(seg_img, seg_val)
        m = np.mean(p, axis=0)
        centered.append(p - m)
    gp = np.concatenate(centered, axis=0)
    return ut.get_first_pca_axis(gp)


def slices_centers_of_mass(
    img, seg_img, axis_dim, seg_val, slice_seg_thresh=1
):
    """Finds the center of mass of image slices along array dimension

    Iterates through `img` along dimension `axis_dim`, finds how many elements
    of `seg_img` are equal to `seg_val`, and if that number is greater than
    or equal to `slice_seg_thresh` calculates the center of mass of the masked
    `img` on that slice. Centers of mass are based on the physical points
    corresponding to each index in `seg_img` found with
    `transform_stik_indices_to_physical_points`.

    Parameters
    ----------
    img : SimpleITK.Image
        Grayscale image used to calculate center of mass
    seg_img : SimpleITK.Image
        segmentation image used to select elements of `img`. The spatial
        information of `seg_img` will be used to determine where each element
        is in space.
    axis_dim : integer
        Axis along which `img` is sliced, in SimpleITK axis order
    seg_val : number
        elements of `img` will only be included in the center of mass if the
        corresponding element of `seg_img` is equal to `seg_val`
    slice_seg_thresh : integer
        center of mass along slices of `img` will only be calculated if at
        least `slice_seg_thresh` or more elements of `seg_img` are equal to
        `seg_val`. Default = 1.

    Returns
    -------
    com : np.ndarray (N x 3)
        center of mass for each slice of `img` meeting the criteria described
        above.
    """
    seg_arr = sitk.GetArrayViewFromImage(seg_img)
    arr = sitk.GetArrayViewFromImage(img)
    ndxs = ut.find_indices_equal_to(seg_arr, seg_val)
    ndxs_sitk = ndxs[:, ::-1]
    slice_ndxs = np.unique(ndxs_sitk[:, axis_dim])
    nmask_in_slice = np.array(
        [np.count_nonzero(ndxs_sitk[:, axis_dim] == x) for x in slice_ndxs]
    )
    sel_slice_ndxs = slice_ndxs[nmask_in_slice >= slice_seg_thresh]
    ndx_points = sv.transform_sitk_indices_to_physical_points(
        seg_img, ndxs_sitk
    )
    com = np.zeros((sel_slice_ndxs.size, 3))
    for i, slice_ndx in enumerate(sel_slice_ndxs):
        mask = ndxs_sitk[:, axis_dim] == slice_ndx
        np_ndx = tuple(ndxs[mask, :].T)
        sel_v = arr[np_ndx]
        com[i, :] = np.sum(
            sel_v[:, np.newaxis] * ndx_points[mask, :], axis=0
        ) / np.sum(sel_v)
    return com


def find_hole(img, seg_img, seg_val, sel_ndxs):
    """Find the center of a hole based on its segmentation value

    sel_ndxs is in sitk axis order!
    Returns sitk axis order
    """
    if seg_img.GetSize() != img.GetSize():
        raise ValueError("Image and segmentation must have the same shape")
    seg_arr = sitk.GetArrayViewFromImage(seg_img)
    arr = sitk.GetArrayViewFromImage(img)
    ndxs = ut.find_indices_equal_to(seg_arr, seg_val)
    if np.size(ndxs) == 0:
        return None
    ndx_points = sv.transform_sitk_indices_to_physical_points(
        seg_img, ndxs[:, [2, 1, 0]]  # convert to sitk axis order
    )
    np_ndx = tuple(ndxs.T)
    sel_v = arr[np_ndx]
    sum_sel = np.sum(sel_v)
    if sum_sel > 0:
        found_center = np.nan * np.ones(3)
        found_center[sel_ndxs] = (
            np.sum(sel_v[:, np.newaxis] * ndx_points, axis=0) / sum_sel
        )[sel_ndxs]
    else:
        return None
    return found_center


def find_holes_by_orientation(
    img, seg_img, seg_vals_dict, orient_indices, orient_names, ap_names
):
    found_centers = {orient: dict() for orient in orient_names}
    for orient in orient_names:
        for ap in ap_names:
            if ap in seg_vals_dict[orient]:
                maybe_hole = find_hole(
                    img,
                    seg_img,
                    seg_vals_dict[orient][ap],
                    orient_indices[orient],
                )
                if maybe_hole is not None:
                    found_centers[orient][ap] = maybe_hole
    return found_centers


def find_hole_angles(
    centers_dict,
    hole_order,
    lps_axes,
    orient_comparison_axis,
    orient_lps_vector,
    orient_names,
):
    centers_ang = dict()
    for orient in orient_names:
        centers_diff = (
            centers_dict[orient][hole_order[orient][0]]
            - centers_dict[orient][hole_order[orient][1]]
        )
        cd_nnan = centers_diff.copy()
        cd_nnan[np.isnan(cd_nnan)] = 0
        centers_ang[orient] = ut.signed_angle_rh(
            lps_axes[orient_comparison_axis[orient]],
            cd_nnan,
            orient_lps_vector[orient],
        )
    return centers_ang


def estimate_hole_axis_from_segmentation(seg_img, seg_vals, orient_lps_vector):
    # Estimate axis rotations from segment locations
    axis = get_segmentation_pca(seg_img, seg_vals)
    if np.dot(axis, orient_lps_vector) < 0:  # pragma: no cover
        axis *= -1
    return axis


def estimate_hole_axes_from_segmentation_by_orientation(
    seg_img, seg_val_dict, orient_lps_vector_dict, orient_names, ap_names
):
    initial_axes = dict()
    for orient in orient_names:
        seg_vals = [
            seg_val_dict[orient][ap]
            for ap in ap_names
            if ap in seg_val_dict[orient]
        ]
        initial_axes[orient] = estimate_hole_axis_from_segmentation(
            seg_img, seg_vals, orient_lps_vector_dict[orient]
        )
    return initial_axes


def calculate_centers_of_mass_for_image_and_segmentation(
    img,
    seg_img,
    initial_axes,
    seg_vals_dict,
    orient_lps_vector_dict,
    orient_names,
    ap_names,
    slice_seg_thresh=1,
):
    coms = dict()
    for orient in orient_names:
        coms[orient] = dict()
        axis_dim = np.nonzero(orient_lps_vector_dict[orient])[0][0]
        r = rot.rotation_matrix_from_vectors(
            initial_axes[orient], orient_lps_vector_dict[orient]
        )
        s = rot.rotation_matrix_to_sitk(r)
        sinv = s.GetInverse()
        seg_img_rs = sv.resample3D(
            seg_img, sinv, interpolator=sitk.sitkNearestNeighbor
        )
        img_rs = sv.resample3D(
            img, sinv, interpolator=sitk.sitkNearestNeighbor
        )
        for ap in ap_names:
            com = slices_centers_of_mass(
                img_rs,
                seg_img_rs,
                axis_dim,
                seg_vals_dict[orient][ap],
                slice_seg_thresh=slice_seg_thresh,
            )
            coms[orient][ap] = (
                r.T @ com.T
            ).T  # rotate com to original location
    return coms


def estimate_axis_rotations_from_centers_of_mass(
    coms, orient_lps_vector_dict, orient_names, ap_names
):
    orient_rotation_matrices = {orient: dict() for orient in orient_names}
    axes = dict()
    for orient in orient_names:
        # center the coms for each segment separately
        # and find the axis for this orientation
        ccoms = []
        for ap in ap_names:
            com = coms[orient][ap]
            m = np.mean(com, axis=0)
            ccoms.append(com - m[np.newaxis, :])
            joined_ccoms = np.concatenate(ccoms, axis=0)
            tmp_axis = ut.get_first_pca_axis(joined_ccoms)
        if np.dot(tmp_axis, orient_lps_vector_dict[orient]) < 0:
            tmp_axis *= -1

        # remove off-axis mean for each segment separately
        coms_deproj_centered = []
        for ap in ap_names:
            com = coms[orient][ap]
            com_proj = ut.vector_rejection(com, tmp_axis)
            proj_m = np.mean(com_proj, axis=0)
            coms_deproj_centered.append(com - proj_m[np.newaxis, :])
        axis = ut.get_first_pca_axis(joined_ccoms)
        if np.dot(axis, orient_lps_vector_dict[orient]) < 0:
            axis *= -1
        R = rot.rotation_matrix_from_vectors(
            axis, orient_lps_vector_dict[orient]
        )
        axes[orient] = R.T @ orient_lps_vector_dict[orient]
        orient_rotation_matrices[orient] = R
    return orient_rotation_matrices, axes


def find_rotation_to_match_hole_angles(
    img,
    seg_img,
    initial_orient_rotation_matrices,
    axes,
    seg_vals_dict,
    design_centers,
    orient_lps_vector_dict,
    orient_names,
    ap_names,
    orient_indices,
    hole_order,
    lps_axes,
    orient_comparison_axis,
    n_iter=10,
):
    nhole = np.prod([len(x) for x in (orient_names, ap_names)])
    # Start measuring hole location and orientation using the estimated set of
    # axes
    bases = np.zeros((3, 3))
    bases[:, 1] = axes["horizontal"]  # P
    bases[:, 2] = ut.norm_vec(
        ut.vector_rejection(axes["vertical"], bases[:, 1])
    )  # S
    bases[:, 0] = np.cross(bases[:, 1], bases[:, 2])  # L

    srot = (
        initial_orient_rotation_matrices["horizontal"] @ bases[:, 2]
    )  # rotated S axis
    rad = ut.signed_angle_rh(
        srot,
        orient_lps_vector_dict["vertical"],
        orient_lps_vector_dict["horizontal"],
    )
    Rot_y = Rotation.from_rotvec(rad * orient_lps_vector_dict["horizontal"])
    R_y = Rot_y.as_matrix()
    R = R_y @ initial_orient_rotation_matrices["horizontal"]

    Sinit = rot.rotation_matrix_to_sitk(R)
    Sinit_inv = Sinit.GetInverse()

    seg_img_current = sv.resample3D(
        seg_img, Sinit_inv, interpolator=sitk.sitkNearestNeighbor
    )
    img_current = sv.resample3D(
        img, Sinit_inv, interpolator=sitk.sitkNearestNeighbor
    )

    found_centers = find_holes_by_orientation(
        img_current,
        seg_img_current,
        seg_vals_dict,
        orient_indices,
        orient_names,
        ap_names,
    )
    found_centers_ang, design_centers_ang = [
        find_hole_angles(
            centers,
            hole_order,
            lps_axes,
            orient_comparison_axis,
            orient_lps_vector_dict,
            orient_names,
        )
        for centers in (found_centers, design_centers)
    ]
    hole_diffs = np.zeros((nhole, 3))
    for i, (orient, ap) in enumerate(itr.product(orient_names, ap_names)):
        hole_diffs[i, :] = (
            design_centers[orient][ap] - found_centers[orient][ap]
        )
    offsets = np.nanmean(hole_diffs, axis=0)

    iter_angle_err = np.zeros((n_iter + 1, 2))
    iter_hole_diff_err = np.zeros((n_iter + 1, nhole, 3))
    found_centers_curr = found_centers
    found_centers_ang_curr = found_centers_ang

    for iterno in range(n_iter):
        iter_angle_err[iterno, :] = [
            found_centers_ang_curr[orient] - design_centers_ang[orient]
            for orient in orient_names
        ]
        iter_hole_diff_err[iterno, :, :] = offsets[np.newaxis, :] - hole_diffs
        for orient in orient_names:
            ang_err = (
                design_centers_ang[orient] - found_centers_ang_curr[orient]
            )
            Rot_update = Rotation.from_rotvec(
                ang_err * orient_lps_vector_dict[orient]
            )
            R_update = Rot_update.as_matrix()
            R = R_update @ R
            S = rot.rotation_matrix_to_sitk(R)
            S_inv = S.GetInverse()
            seg_img_current = sv.resample3D(
                seg_img, S_inv, interpolator=sitk.sitkNearestNeighbor
            )
            img_current = sv.resample3D(
                img, S_inv, interpolator=sitk.sitkNearestNeighbor
            )
            found_centers_curr = find_holes_by_orientation(
                img_current,
                seg_img_current,
                seg_vals_dict,
                orient_indices,
                orient_names,
                ap_names,
            )
            found_centers_ang_curr = find_hole_angles(
                found_centers_curr,
                hole_order,
                lps_axes,
                orient_comparison_axis,
                orient_lps_vector_dict,
                orient_names,
            )
            hole_diffs = np.zeros((nhole, 3))
            for i, (orient, ap) in enumerate(
                itr.product(orient_names, ap_names)
            ):
                hole_diffs[i, :] = (
                    design_centers[orient][ap] - found_centers_curr[orient][ap]
                )
            offsets = np.nanmean(hole_diffs, axis=0)
    iter_angle_err[n_iter, :] = [
        found_centers_ang_curr[orient] - design_centers_ang[orient]
        for orient in orient_names
    ]
    iter_hole_diff_err[n_iter, :, :] = offsets[np.newaxis, :] - hole_diffs
    return R, offsets


def estimate_coms_from_image_and_segmentation(
    img,
    seg_img,
    seg_vals_dict,
    orient_names=("vertical", "horizontal"),
    ap_names=("anterior", "posterior"),
    lps_axes=dict(
        ap=np.array([0, 1, 0]), dv=np.array([0, 0, 1]), ml=np.array([1, 0, 0])
    ),
    hole_orient_axis=dict(horizontal="ap", vertical="dv"),
):
    # Derived parameters
    orient_lps_vector = {
        orient: lps_axes[hole_orient_axis[orient]] for orient in orient_names
    }

    # Estimate axis rotations from segment locations
    initial_axes = estimate_hole_axes_from_segmentation_by_orientation(
        seg_img, seg_vals_dict, orient_lps_vector, orient_names, ap_names
    )

    # Find centers of mass (COM) for slices perpendicular to intial axes
    coms = calculate_centers_of_mass_for_image_and_segmentation(
        img,
        seg_img,
        initial_axes,
        seg_vals_dict,
        orient_lps_vector,
        orient_names,
        ap_names,
    )
    return coms


def estimate_rotation_and_coms_from_image_and_segmentation(
    img,
    seg_img,
    seg_vals_dict,
    orient_names=("vertical", "horizontal"),
    ap_names=("anterior", "posterior"),
    lps_axes=dict(
        ap=np.array([0, 1, 0]), dv=np.array([0, 0, 1]), ml=np.array([1, 0, 0])
    ),
    hole_orient_axis=dict(horizontal="ap", vertical="dv"),
    orient_comparison_axis=dict(horizontal="dv", vertical="ap"),
    design_centers=dict(
        horizontal=dict(
            anterior=np.array([-6.34, np.nan, 2.5]),
            posterior=np.array([-5.04, np.nan, 1]),
        ),
        vertical=dict(
            anterior=np.array([-5.09, 3.209, np.nan]),
            posterior=np.array([-6.84, 9.909, np.nan]),
        ),
    ),  # Bregma-relative mms (LPS)
    orient_indices=dict(horizontal=[0, 2], vertical=[0, 1]),
    hole_order=dict(
        horizontal=["anterior", "posterior"],
        vertical=["posterior", "anterior"],
    ),
):
    coms = estimate_coms_from_image_and_segmentation(
        img,
        seg_img,
        seg_vals_dict,
        orient_names=("vertical", "horizontal"),
        ap_names=("anterior", "posterior"),
        lps_axes=dict(
            ap=np.array([0, 1, 0]),
            dv=np.array([0, 0, 1]),
            ml=np.array([1, 0, 0]),
        ),
        hole_orient_axis=dict(horizontal="ap", vertical="dv"),
    )
    # Derived parameters
    orient_lps_vector = {
        orient: lps_axes[hole_orient_axis[orient]] for orient in orient_names
    }

    # Estimate axis rotations from centers of mass
    (
        orient_rotation_matrices,
        axes,
    ) = estimate_axis_rotations_from_centers_of_mass(
        coms, orient_lps_vector, orient_names, ap_names
    )

    R, offsets = find_rotation_to_match_hole_angles(
        img,
        seg_img,
        orient_rotation_matrices,
        axes,
        seg_vals_dict,
        design_centers,
        orient_lps_vector,
        orient_names,
        ap_names,
        orient_indices,
        hole_order,
        lps_axes,
        orient_comparison_axis,
    )

    return coms, R, offsets
