"""
Code to find the rotation matrix to align a headframe to a set of holes.
"""

import itertools as itr

import numpy as np
import SimpleITK as sitk
from scipy import optimize as opt
from scipy.spatial.transform import Rotation

from . import measurement as mrmsr
from . import optimization as mropt
from . import rotations as rot
from . import sitk_volume as sv
from . import utils as ut
from .file_io import slicer_files as sf

lps_axes = dict(
    ap=np.array([0, 1, 0]), dv=np.array([0, 0, 1]), ml=np.array([1, 0, 0])
)
def_orient_names = ("vertical", "horizontal")
def_ap_names = ("anterior", "posterior")
def_orient_comparison_axes = dict(
    horizontal=lps_axes["dv"], vertical=lps_axes["ap"]
)
def_orient_axes_dict = {
    orient: lps_axes[direction]
    for orient, direction in dict(horizontal="ap", vertical="dv").items()
}
def_design_centers = dict(
    horizontal=dict(
        anterior=np.array([-6.34, np.nan, 2.5]),
        posterior=np.array([-5.04, np.nan, 1]),
    ),
    vertical=dict(
        anterior=np.array([-5.09, 3.209, np.nan]),
        posterior=np.array([-6.84, 9.909, np.nan]),
    ),
)
def_orient_indices = dict(horizontal=[0, 2], vertical=[0, 1])
def_hole_order = dict(
    horizontal=["anterior", "posterior"],
    vertical=["posterior", "anterior"],
)


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
    `transform_sitk_indices_to_physical_points`.

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
    img,
    seg_img,
    seg_vals_dict,
    orient_indices=def_orient_indices,
    orient_names=def_orient_names,
    ap_names=def_ap_names,
):
    """
    Find holes in an image by different orientations and anterior-posterior
    names.

    Parameters
    ----------
    img : ndarray
        The original image in which holes need to be found.
    seg_img : ndarray
        The segmented image that identifies different regions.
    seg_vals_dict : dict
        A dictionary where keys are orientation names and values are
        dictionaries.  These inner dictionaries map anterior-posterior names to
        segmentation values.
    orient_indices : dict, optional
        A dictionary where keys are orientation names and values are the
        indices used to find the holes in the image.
    orient_names : list of str, optional
        A list of orientation names.
    ap_names : list of str, optional
        A list of anterior-posterior names.

    Returns
    -------
    found_centers : dict
        A dictionary where keys are orientation names and values are
        dictionaries.  These inner dictionaries map anterior-posterior names to
        the centers of found holes.  If a hole is not found for a given
        orientation and anterior-posterior name, that entry will be missing.

    Notes
    -----
    This function iterates over the given orientations and anterior-posterior
    names to find the holes in the image using the `find_hole` function. The
    centers of the found holes are returned in a nested dictionary structure.
    """
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
    hole_order=def_hole_order,
    orient_comparison_axis=def_orient_comparison_axes,
    orient_axis_dict=def_orient_axes_dict,
    orient_names=def_orient_names,
):
    """
    Calculate angles between holes for each orientation.

    Parameters
    ----------
    centers_dict : dict
        Dictionary of hole centers for each orientation and anterior-posterior
        name.
    hole_order : dict, optional
        Dictionary defining the order of holes for each orientation, by default
        `def_hole_order`.
    orient_comparison_axis : dict, optional
        Dictionary of comparison axes for each orientation, by default
        `def_orient_comparison_axes`.
    orient_axis_dict : dict, optional
        Dictionary of axes for each orientation, by default
        `def_orient_axes_dict`.
    orient_names : list of str, optional
        List of orientation names, by default `def_orient_names`.

    Returns
    -------
    centers_ang : dict
        Dictionary of calculated angles for each orientation.
    """
    centers_ang = dict()
    for orient in orient_names:
        centers_diff = (
            centers_dict[orient][hole_order[orient][0]]
            - centers_dict[orient][hole_order[orient][1]]
        )
        cd_nnan = centers_diff.copy()
        cd_nnan[np.isnan(cd_nnan)] = 0
        centers_ang[orient] = ut.signed_angle_rh(
            orient_comparison_axis[orient],
            cd_nnan,
            orient_axis_dict[orient],
        )
    return centers_ang


def estimate_hole_axis_from_segmentation(seg_img, seg_vals, reference_axis):
    """
    Estimate the axis of a hole from segmentation values.

    Parameters
    ----------
    seg_img : SimpleITK.Image
        The segmentation image.
    seg_vals : list
        List of segmentation values used to identify the hole.
    reference_axis : ndarray
        Reference axis vector, direction of returned axis will be flipped if
        the dot product between reference_axis and the found axis is negative

    Returns
    -------
    axis : ndarray
        Estimated axis of the hole.
    """
    # Estimate axis rotations from segment locations
    axis = get_segmentation_pca(seg_img, seg_vals)
    if np.dot(axis, reference_axis) < 0:  # pragma: no cover
        axis *= -1
    return axis


def estimate_hole_axes_from_segmentation_by_orientation(
    seg_img,
    seg_val_dict,
    orient_axes_dict=def_orient_axes_dict,
    orient_names=def_orient_names,
    ap_names=def_ap_names,
):
    """
    Estimate hole axes from segmentation by orientation.

    Parameters
    ----------
    seg_img : SimpleITK.Image
        Segmentation image.
    seg_val_dict : dict
        Nested dictionary of segmentation values for each orientation and AP.
    orient_axes_dict : dict
        Nested dictionary of LPS vectors for each orientation and AP.
    orient_names : list of str
        List of orientation names.
    ap_names : list of str
        List of anterior-posterior names.

    Returns
    -------
    initial_axes : dict
        Dictionary of initial axes as a 3 element vector for each orientation.
    """
    initial_axes = dict()
    for orient in orient_names:
        seg_vals = [
            seg_val_dict[orient][ap]
            for ap in ap_names
            if ap in seg_val_dict[orient]
        ]
        initial_axes[orient] = estimate_hole_axis_from_segmentation(
            seg_img, seg_vals, orient_axes_dict[orient]
        )
    return initial_axes


def calculate_centers_of_mass_for_image_and_segmentation(
    img,
    seg_img,
    initial_axes,
    seg_vals_dict,
    orient_axes_dict=def_orient_axes_dict,
    orient_names=def_orient_names,
    ap_names=def_ap_names,
    slice_seg_thresh=1,
):
    """
    Calculate centers of mass for image and segmentation.

    Parameters
    ----------
    img : SimpleITK.Image
        The original image used for calculating center of mass.
    seg_img : SimpleITK.Image
        The segmentation image used for selecting elements of `img`.
    initial_axes : dict
        Dictionary of initial axes for each orientation.
    seg_vals_dict : dict
        Nested dictionary of segmentation values for each orientation and
        anterior-posterior names.
    orient_lps_vector_dict : dict, optional
        Dictionary of LPS vectors for each orientation.
    orient_names : list of str, optional
        List of orientation names.
    ap_names : list of str, optional
        List of anterior-posterior names.
    slice_seg_thresh : int, optional
        Minimum number of segmentation elements in a slice required for center
        of mass calculation, by default 1.

    Returns
    -------
    coms : dict
        Nested dictionary of Nx3 NDArray centers of mass for each orientation
        and anterior-posterior name.
    """
    coms = dict()
    for orient in orient_names:
        coms[orient] = dict()
        axis_dim = np.nonzero(orient_axes_dict[orient])[0][0]
        r = rot.rotation_matrix_from_vectors(
            initial_axes[orient], orient_axes_dict[orient]
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
    coms,
    orient_axes_dict=def_orient_axes_dict,
    orient_names=def_orient_names,
    ap_names=def_ap_names,
):
    """
    Estimate axis rotations from centers of mass.

    Parameters
    ----------
    coms : dict
        Nested dictionary of Nx3 NDArray centers of mass for each orientation
        and anterior-posterior name.
    orient_axes_dict : dict, optional
        Dictionary of LPS vectors for each orientation.
    orient_names : list of str, optional
        List of orientation names.
    ap_names : list of str, optional
        List of anterior-posterior names.

    Returns
    -------
    orient_rotation_matrices : dict
        Dictionary of 3x3 rotation matrices for each orientation.
    axes : dict
        Dictionary of 3 element vector axes for each orientation.
    """
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
        if np.dot(tmp_axis, orient_axes_dict[orient]) < 0:
            tmp_axis *= -1

        # remove off-axis mean for each segment separately
        coms_deproj_centered = []
        for ap in ap_names:
            com = coms[orient][ap]
            com_proj = ut.vector_rejection(com, tmp_axis)
            proj_m = np.mean(com_proj, axis=0)
            coms_deproj_centered.append(com - proj_m[np.newaxis, :])
        axis = ut.get_first_pca_axis(joined_ccoms)
        if np.dot(axis, orient_axes_dict[orient]) < 0:
            axis *= -1
        R = rot.rotation_matrix_from_vectors(axis, orient_axes_dict[orient])
        axes[orient] = R.T @ orient_axes_dict[orient]
        orient_rotation_matrices[orient] = R
    return orient_rotation_matrices, axes


def find_rotation_to_match_hole_angles(
    img,
    seg_img,
    initial_orient_rotation_matrices,
    axes,
    seg_vals_dict,
    design_centers=def_design_centers,
    orient_axes_dict=def_orient_axes_dict,
    orient_names=def_orient_names,
    ap_names=def_ap_names,
    orient_indices=def_orient_indices,
    hole_order=def_hole_order,
    orient_comparison_axis=def_orient_comparison_axes,
    n_iter=10,
):
    """
    Find rotation matrix to match hole angles.

    Parameters
    ----------
    img : SimpleITK.Image
        The original image.
    seg_img : SimpleITK.Image
        The segmentation image.
    initial_orient_rotation_matrices : dict
        Dictionary of initial orientation rotation matrices indexed by
        orientation.
    axes : dict
        Dictionary of axes for each orientation.
    seg_vals_dict : dict
        Dictionary of segmentation values for each orientation and
        anterior-posterior names.
    design_centers : dict
        Dictionary of design centers for each orientation and
        anterior-posterior names.
    orient_axes_dict : dict
        Dictionary of LPS vectors for each orientation.
    orient_names : list of str
        List of orientation names.
    ap_names : list of str
        List of anterior-posterior names.
    orient_indices : dict
        Dictionary of orientation indices.
    hole_order : dict
        Dictionary of hole order (list of AP names) for each orientation.
    orient_comparison_axis : dict
        Dictionary of comparison axes for each orientation.
    n_iter : int, optional
        Number of iterations, by default 10.

    Returns
    -------
    R : ndarray
        Rotation matrix to align img with design centers.
    translation : ndarray
        Offsets to align img with design centers.
    """
    nhole = np.prod([len(x) for x in (orient_names, ap_names)])
    # Start measuring hole location and orientation using the estimated set of
    # axes
    bases = np.zeros((3, 3))
    bases[:, 1] = axes["horizontal"]  # P
    bases[:, 2] = ut.norm_vec(
        ut.vector_rejection(axes["vertical"], bases[:, 1])
    )  # S
    bases[:, 0] = np.cross(bases[:, 1], bases[:, 2])  # L

    s_rot = (
        initial_orient_rotation_matrices["horizontal"] @ bases[:, 2]
    )  # rotated S axis
    rad = ut.signed_angle_rh(
        s_rot,
        lps_axes["dv"],
        lps_axes["ap"],
    )
    Rot_y = Rotation.from_rotvec(rad * lps_axes["ap"])
    R_y = Rot_y.as_matrix()
    R = R_y @ initial_orient_rotation_matrices["horizontal"]

    S_init = rot.rotation_matrix_to_sitk(R)
    S_init_inv = S_init.GetInverse()

    seg_img_current = sv.resample3D(
        seg_img, S_init_inv, interpolator=sitk.sitkNearestNeighbor
    )
    img_current = sv.resample3D(
        img, S_init_inv, interpolator=sitk.sitkNearestNeighbor
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
            orient_comparison_axis,
            orient_axes_dict,
            orient_names,
        )
        for centers in (found_centers, design_centers)
    ]
    hole_diffs = np.zeros((nhole, 3))
    for i, (orient, ap) in enumerate(itr.product(orient_names, ap_names)):
        hole_diffs[i, :] = (
            design_centers[orient][ap] - found_centers[orient][ap]
        )
    translation = np.nanmean(hole_diffs, axis=0)

    iter_angle_err = np.zeros((n_iter + 1, 2))
    iter_hole_diff_err = np.zeros((n_iter + 1, nhole, 3))
    found_centers_curr = found_centers
    found_centers_ang_curr = found_centers_ang

    for iter_no in range(n_iter):
        iter_angle_err[iter_no, :] = [
            found_centers_ang_curr[orient] - design_centers_ang[orient]
            for orient in orient_names
        ]
        iter_hole_diff_err[iter_no, :, :] = (
            translation[np.newaxis, :] - hole_diffs
        )
        for orient in orient_names:
            ang_err = (
                design_centers_ang[orient] - found_centers_ang_curr[orient]
            )
            Rot_update = Rotation.from_rotvec(
                ang_err * orient_axes_dict[orient]
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
                orient_comparison_axis,
                orient_axes_dict,
                orient_names,
            )
            hole_diffs = np.zeros((nhole, 3))
            for i, (orient, ap) in enumerate(
                itr.product(orient_names, ap_names)
            ):
                hole_diffs[i, :] = (
                    design_centers[orient][ap] - found_centers_curr[orient][ap]
                )
            translation = np.nanmean(hole_diffs, axis=0)
    iter_angle_err[n_iter, :] = [
        found_centers_ang_curr[orient] - design_centers_ang[orient]
        for orient in orient_names
    ]
    iter_hole_diff_err[n_iter, :, :] = translation[np.newaxis, :] - hole_diffs
    return R, translation


def estimate_coms_from_image_and_segmentation(
    img,
    seg_img,
    seg_vals_dict,
    orient_names=def_orient_names,
    ap_names=def_ap_names,
    orient_axes_dict=def_orient_axes_dict,
):
    """
    Estimate centers of mass (COMs) from image and segmentation.

    Parameters
    ----------
    img : SimpleITK.Image
        The original image.
    seg_img : SimpleITK.Image
        The segmentation image.
    seg_vals_dict : dict
        Nested dictionary of segmentation values.
    orient_names : tuple of str, optional
        Tuple of orientation names, by default ("vertical", "horizontal").
    ap_names : tuple of str, optional
        Tuple of anterior-posterior names, by default
        ("anterior", "posterior").
    lps_axes : dict, optional
        Dictionary of LPS axes, by default dict(ap=np.array([0, 1, 0]),
        dv=np.array([0, 0, 1]), ml=np.array([1, 0, 0])).
    hole_orient_axis : dict, optional
        Dictionary of hole orientation axes, by default dict(horizontal="ap",
        vertical="dv").

    Returns
    -------
    coms : dict
        Dictionary of Nx3 NDArrays of centers of mass for each orientation.
    """
    # Estimate axis rotations from segment locations
    initial_axes = estimate_hole_axes_from_segmentation_by_orientation(
        seg_img, seg_vals_dict, orient_axes_dict, orient_names, ap_names
    )

    # Find centers of mass (COM) for slices perpendicular to initial axes
    coms = calculate_centers_of_mass_for_image_and_segmentation(
        img,
        seg_img,
        initial_axes,
        seg_vals_dict,
        orient_axes_dict,
        orient_names,
        ap_names,
    )
    return coms


def estimate_rotation_and_coms_from_image_and_segmentation(
    img,
    seg_img,
    seg_vals_dict,
    orient_names=def_orient_names,
    ap_names=def_ap_names,
    orient_comparison_axis=def_orient_comparison_axes,
    design_centers=def_design_centers,  # Bregma-relative mms (LPS)
    orient_indices=def_orient_indices,
    hole_order=def_hole_order,
    orient_axes_dict=def_orient_axes_dict,
    n_iter=10,
):
    """
    Estimate rotation and centers of mass (COMs) from image and segmentation.

    Parameters
    ----------
    img : SimpleITK.Image
        The original image.
    seg_img : SimpleITK.Image
        The segmentation image.
    seg_vals_dict : dict
        Dictionary of segmentation values for each orientation and
        anterior-posterior names.
    orient_names : list of str, optional
        List of orientation names, by default `def_orient_names`.
    ap_names : list of str, optional
        List of anterior-posterior names, by default `def_ap_names`.
    orient_comparison_axis : dict, optional
        Dictionary of comparison axes for each orientation, by default
        `def_orient_comparison_axes`.
    design_centers : dict, optional
        Dictionary of design centers for each orientation and
        anterior-posterior names, by default `def_design_centers`.
    orient_indices : dict, optional
        Dictionary of orientation indices, by default `def_orient_indices`.
    hole_order : dict, optional
        Dictionary of hole order for each orientation, by default
        `def_hole_order`.
    orient_axes_dict : dict, optional
        Dictionary of axes for each orientation, by default
        `def_orient_axes_dict`.
    n_iter : int, optional
        Number of iterations, by default 10.

    Returns
    -------
    coms : dict
        Dictionary of centers of mass for each orientation and
        anterior-posterior name.
    R : ndarray
        Rotation matrix.
    translation : ndarray
        Offsets for each hole.
    """

    coms = estimate_coms_from_image_and_segmentation(
        img,
        seg_img,
        seg_vals_dict,
        orient_names=orient_names,
        ap_names=ap_names,
        orient_axes_dict=orient_axes_dict,
    )

    # Estimate axis rotations from centers of mass
    (
        orient_rotation_matrices,
        axes,
    ) = estimate_axis_rotations_from_centers_of_mass(
        coms,
        orient_axes_dict=orient_axes_dict,
        orient_names=orient_names,
        ap_names=ap_names,
    )

    R, translation = find_rotation_to_match_hole_angles(
        img,
        seg_img,
        orient_rotation_matrices,
        axes,
        seg_vals_dict,
        design_centers=design_centers,
        orient_axes_dict=orient_axes_dict,
        orient_names=orient_names,
        ap_names=ap_names,
        orient_indices=orient_indices,
        hole_order=hole_order,
        orient_comparison_axis=orient_comparison_axis,
        n_iter=n_iter,
    )

    return coms, R, translation


def make_segment_dict(
    segment_info,
    segment_format=None,
    ap_names=def_ap_names,
    orient_names=def_orient_names,
    ignore_list=[],
):
    """
    Create a dictionary of segment values based on the provided segment
    information.

    Parameters
    ----------
    segment_info : dict
        A dictionary containing segment information.
    segment_format : str, optional
        The format string used to generate the segment keys. Defaults to None.
    ap_names : list, optional
        A list of names for the anterior-posterior (AP) segments. Defaults to
        def_ap_names.
    orient_names : list, optional
        A list of names for the orientation segments. Defaults to
        def_orient_names.
    ignore_list : list, optional
        A list of segment names to ignore. Defaults to an empty list.

    Returns
    -------
    dict
        A dictionary of segment values, organized by orientation and AP
        segment.

    """
    if segment_format is None:
        segment_format = "{}_{}"
    seg_vals = dict()
    ignore_set = set(ignore_list)
    for orient in orient_names:
        seg_vals[orient] = dict()
        for ap in ap_names:
            key_name = segment_format.format(ap, orient)
            if key_name in segment_info and key_name not in ignore_set:
                seg_vals[orient][ap] = segment_info[key_name]
    return seg_vals


def segment_dict_from_seg_odict(
    seg_odict,
    segment_format=None,
    ap_names=def_ap_names,
    orient_names=def_orient_names,
    ignore_list=[],
):
    """
    Create a dictionary of segments from a segmentation ordered dictionary.

    Parameters
    ----------
    seg_odict : OrderedDict
        The segmentation ordered dictionary.
    segment_format : str, optional
        The format of the segment names. Defaults to None.
    ap_names : list, optional
        The list of anatomical plane names. Defaults to def_ap_names.
    orient_names : list, optional
        The list of orientation names. Defaults to def_orient_names.

    Returns
    -------
    dict
        A dictionary of segments, where the keys are the segment names and the
        values are the segment information.

    Raises
    ------
    ValueError
        If no segments are found in the segmentation ordered dictionary.
    """
    segment_info = sf.find_seg_nrrd_header_segment_info(seg_odict)
    segment_dict = make_segment_dict(
        segment_info, segment_format, ap_names, orient_names, ignore_list
    )
    if all([len(d) == 0 for d in segment_dict.values()]):
        raise ValueError(
            "No segments found. Is the key format {key_format} correct?"
        )
    return segment_dict


def _compress_each(mask, *args):
    """
    Compresses each iterable in `*args` using `mask`.

    Parameters
    ----------
    mask : iterable
        A boolean mask indicating which elements to keep.
    *args : iterable
        Variable number of iterables to compress.

    Returns
    -------
    list
        A list of iterables, each compressed using `mask`.

    Examples
    --------
    >>> mask = [True, False, True]
    >>> arg1 = [1, 2, 3]
    >>> arg2 = ['a', 'b', 'c']
    >>> _compress_each(mask, arg1, arg2)
    [(1, 3), ('a', 'c')]
    """
    return list(zip(*itr.compress(zip(*args), mask)))


def find_hf_rotation_from_seg_and_lowerplane(
    img,
    seg_img,
    seg_odict,
    plane_pts,
    segment_format=None,
    ap_names=def_ap_names,
    orient_names=def_orient_names,
    niter_rot=10,
    niter_com=50000,
    niter_com_plane=10000,
    xtol=1e-12,
    ignore_list=[],
):
    """
    Calculate the headframe rotation and translation from segmentation data and
    a lower plane.

    Parameters
    ----------
    img : ndarray
        The input image.
    seg_img : ndarray
        The segmented image.
    seg_odict : dict
        Dictionary containing segmentation information.
    plane_pts : ndarray
        Points defining the lower plane.
    segment_format : optional
        Format of the segment dictionary. Default is None.
    ap_names : list, optional
        List of anterior-posterior names. Default is `def_ap_names`.
    orient_names : list, optional
        List of orientation names. Default is `def_orient_names`.
    niter_rot : int, optional
        Number of iterations for the initial rotation estimation. Default is
        10.
    niter_com : int, optional
        Number of iterations for the optimization considering only holes.
        Default is 50000.
    niter_com_plane : int, optional
        Number of iterations for the optimization including the lower plane.
        Default is 10000.
    xtol : float, optional
        Tolerance for termination by change in the optimization. Default is
        1e-12.
    ignore_list : list, optional
        List of segment names to ignore. Default is an empty list.

    Returns
    -------
    theta0 : ndarray
        Initial guess for the rotation and translation parameters.
    output_holes_only : ndarray
        Optimized parameters considering only the headframe holes.
    output_all : ndarray
        Final optimized parameters including the lower plane.

    Notes
    -----
    The function performs the following steps:
    1. Converts the segmentation dictionary.
    2. Estimates centers of mass and initial rotation and translation.
    3. Retrieves design centers and hole locations.
    4. Prepares data for optimization.
    5. Optimizes the transformation considering only the headframe holes.
    6. Refines the optimization by including the lower plane.
    """
    segment_dict = segment_dict_from_seg_odict(
        seg_odict, segment_format, ap_names, orient_names, ignore_list
    )

    # Get centers of mass and initial guess at rotation and translation
    coms, R, translation = (
        estimate_rotation_and_coms_from_image_and_segmentation(
            img, seg_img, segment_dict, n_iter=niter_rot
        )
    )
    euler0 = Rotation.from_matrix(R).as_euler("xyz")
    theta0 = np.concatenate((euler0, translation))

    # Get design centers and hole locations
    pts1, pts2, names = mropt.get_headframe_hole_lines(
        insert_underscores=True, return_plane=True
    )

    # slice and dice data for optimization
    moving = []
    pts1l = []
    pts2l = []
    weights = []
    plane_ndx = -1
    for i, name in enumerate(names):
        if name == "plane":
            plane_ndx = i
            moving.append(plane_pts)
            pts1l.append(pts1[i, :])
            pts2l.append(pts2[i, :])
            npt_plane = plane_pts.shape[0]
            weights.append(np.full(npt_plane, 1 / npt_plane))
        else:
            pos, orient = name.split("_")
            if orient in coms:
                d = coms[orient]
                if pos in d:
                    these_coms = d[pos]
                    npt = these_coms.shape[0]
                    moving.append(these_coms)
                    pts1l.append(pts1[i, :])
                    pts2l.append(pts2[i, :])
                    weights.append(np.full(npt, 1 / npt))

    hole_mask = np.full(len(moving), True)
    hole_mask[plane_ndx] = False
    hole_only_list_of_lists = _compress_each(
        hole_mask, pts1l, pts2l, moving, weights
    )

    # Optimize transform only considering holes in the headframe
    output_holes_only = opt.fmin(
        mropt.revised_error_rotate_compare_weighted_lines,
        theta0,
        args=tuple(hole_only_list_of_lists),
        xtol=xtol,
        maxfun=niter_com,
        retall=1,
    )
    R_holes_only, transl_holes_only = mropt.unpack_theta(output_holes_only[0])

    # Now include the lower plane
    group_err_funs = [mrmsr.dist_point_to_line] * len(moving)
    group_err_funs[plane_ndx] = mrmsr.dist_point_to_plane

    output_all = opt.fmin(
        mropt.revised_error_rotate_compare_weighted_lines,
        output_holes_only[0],
        args=(pts1l, pts2l, moving, weights, group_err_funs),
        xtol=xtol,
        maxfun=niter_com_plane,
        retall=1,
    )
    R_all, transl_all = mropt.unpack_theta(output_all[0])

    return R, translation, R_holes_only, transl_holes_only, R_all, transl_all
