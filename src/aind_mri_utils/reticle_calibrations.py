"""
Functions to read reticle calibration data, find a transformation between
coordinate frames, and apply the transformation.
"""

import numpy as np
from openpyxl import load_workbook
from scipy import optimize as opt
from scipy.spatial.transform import Rotation

from . import rotations as rot
from . import utils as ut


def extract_calibration_metadata(ws):
    """
    Extract calibration metadata from an Excel worksheet.

    Parameters
    ----------
    ws : openpyxl.worksheet.worksheet.Worksheet
        The worksheet object from which to extract the calibration metadata.

    Returns
    -------
    tuple
        A tuple containing:
        - global_factor (float): The global scale value.
        - global_rotation_degrees (float): The global rotation in degrees.
        - manipulator_factor (float): The manipulator scale value.
        - global_offset (numpy.ndarray): The global offset as a 3-element
          array.
        - reticle_name (str): The name of the reticle.
    """
    rowiter = ws.iter_rows(min_row=1, max_row=2, values_only=True)
    colname_lookup = {k: i for i, k in enumerate(next(rowiter))}
    metadata_values = next(rowiter)
    global_factor = metadata_values[colname_lookup["GlobalFactor"]]
    global_rotation_degrees = metadata_values[
        colname_lookup["GlobalRotationDegrees"]
    ]
    manipulator_factor = metadata_values[colname_lookup["ManipulatorFactor"]]
    reticle_name = metadata_values[colname_lookup["Reticule"]]
    offset_x_pos = colname_lookup["GlobalOffsetX"]
    global_offset = np.array(
        metadata_values[offset_x_pos : offset_x_pos + 3],  # noqa: E203
        dtype=float,
    )
    return (
        global_factor,
        global_rotation_degrees,
        manipulator_factor,
        global_offset,
        reticle_name,
    )


def extract_calibration_pairs(ws):
    """
    Extract calibration pairs from an Excel worksheet.

    Parameters
    ----------
    ws : openpyxl.worksheet.worksheet.Worksheet
        The worksheet object from which to extract the calibration pairs.

    Returns
    -------
    dict
        A dictionary where keys are probe names and values are lists of tuples,
        each containing a reticle point and a probe point as numpy arrays.
    """
    pairs_by_probe = dict()
    for row in ws.iter_rows(min_row=2, max_col=7, values_only=True):
        probe_name = row[0]
        if probe_name is None:
            continue
        reticle_pt = np.array(row[1:4])
        probe_pt = np.array(row[4:7])
        if probe_name not in pairs_by_probe:
            pairs_by_probe[probe_name] = []
        pairs_by_probe[probe_name].append((reticle_pt, probe_pt))
    return pairs_by_probe


def _combine_pairs(list_of_pairs):
    """
    Combine lists of pairs into separate global and manipulator points
    matrices.

    Parameters
    ----------
    list_of_pairs : list of tuple
        A list of tuples, each containing a reticle point and a probe point as
        numpy arrays.

    Returns
    -------
    tuple
        Two numpy arrays, one for global points and one for manipulator points.
    """
    global_pts, manipulator_pts = [np.vstack(x) for x in zip(*list_of_pairs)]
    return global_pts, manipulator_pts


def _apply_metadata_to_pair_mats(
    global_pts,
    manipulator_pts,
    global_factor,
    global_rotation_degrees,
    global_offset,
    manipulator_factor,
):
    """
    Apply calibration metadata to global and manipulator points matrices.

    Parameters
    ----------
    global_pts : numpy.ndarray
        The global points matrix.
    manipulator_pts : numpy.ndarray
        The manipulator points matrix.
    global_factor : float
        The global factor value.
    global_rotation_degrees : float
        The global rotation in degrees.
    global_offset : numpy.ndarray
        The global offset as a 3-element array.
    manipulator_factor : float
        The manipulator factor value.

    Returns
    -------
    tuple
        The adjusted global points and manipulator points matrices.
    """
    if global_rotation_degrees != 0:
        rotmat = (
            Rotation.from_euler("z", global_rotation_degrees, degrees=True)
            .as_matrix()
            .squeeze()
        )
        # Transposed because points are row vectors
        global_pts = global_pts @ rotmat.T
    global_pts = global_pts * global_factor + global_offset
    manipulator_pts = manipulator_pts * manipulator_factor
    return global_pts, manipulator_pts


def _apply_metadata_to_pair_lists(
    list_of_pairs,
    global_factor,
    global_rotation_degrees,
    global_offset,
    manipulator_factor,
):
    """
    Apply calibration metadata to lists of pairs.

    Parameters
    ----------
    list_of_pairs : list of tuple
        A list of tuples, each containing a reticle point and a probe point as
        numpy arrays.
    global_factor : float
        The global factor value.
    global_rotation_degrees : float
        The global rotation in degrees.
    global_offset : numpy.ndarray
        The global offset as a 3-element array.
    manipulator_factor : float
        The manipulator factor value.

    Returns
    -------
    tuple
        The adjusted global points and manipulator points matrices.
    """
    global_pts, manipulator_pts = _combine_pairs(list_of_pairs)
    return _apply_metadata_to_pair_mats(
        global_pts,
        manipulator_pts,
        global_factor,
        global_rotation_degrees,
        global_offset,
        manipulator_factor,
    )


def read_reticle_calibration(
    filename, points_sheet_name="points", metadata_sheet_name="metadata"
):
    """
    Read reticle calibration data from an Excel file.

    Parameters
    ----------
    filename : str
        The path to the Excel file containing the calibration data.
    points_sheet_name : str, optional
        The name of the sheet containing the calibration points.
        The default is "points".
    metadata_sheet_name : str, optional
        The name of the sheet containing the calibration metadata.
        The default is "metadata".

    Returns
    -------
    tuple
        A tuple containing:
        - adjusted_pairs_by_probe (dict): Adjusted calibration pairs by probe
          name.
        - global_offset (numpy.ndarray): The global offset as a 3-element
          array.
        - global_rotation_degrees (float): The global rotation in degrees.
        - reticle_name (str): The name of the reticle.

    Raises
    ------
    ValueError
        If the specified sheets are not found in the Excel file.
    """
    wb = load_workbook(filename, read_only=True, data_only=True)
    if points_sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet {points_sheet_name} not found in {filename}")
    if metadata_sheet_name not in wb.sheetnames:
        raise ValueError(
            f"Sheet {metadata_sheet_name} not found in {filename}"
        )
    (
        global_factor,
        global_rotation_degrees,
        manipulator_factor,
        global_offset,
        reticle_name,
    ) = extract_calibration_metadata(wb[metadata_sheet_name])
    pairs_by_probe = extract_calibration_pairs(wb["points"])
    adjusted_pairs_by_probe = {
        k: _apply_metadata_to_pair_lists(
            v,
            global_factor,
            global_rotation_degrees,
            global_offset,
            manipulator_factor,
        )
        for k, v in pairs_by_probe.items()
    }
    return (
        adjusted_pairs_by_probe,
        global_offset,
        global_rotation_degrees,
        reticle_name,
    )


def _unpack_theta(theta):
    """Helper function to unpack theta into rotation matrix and translation."""
    R = rot.combine_angles(*theta[0:3])
    offset = theta[3:6]
    return R, offset


def fit_rotation_params(
    reticle_pts, probe_pts, legacy_outputs=False, **kwargs
):
    """
    Fit rotation parameters to align reticle points with probe points using
    least squares optimization. The rotation matrix and translation vector
    are the solution for the equation

    probe_pts = R @ reticle_pts + translation

    where each point is a column vector.

    Because numpy is row-major, points are often stored as row vectors. In this
    case, you should use the transpose of this equation:

    probe_pts = reticle_pts @ R.T + translation

    Parameters
    ----------
    reticle_pts : numpy.ndarray
        The reticle points to be transformed.
    probe_pts : numpy.ndarray
        The probe points to align with.
    legacy_outputs : bool, optional
        If True, return the translation in the global frame and the transpose
        of the rotation matrix.  The default is False.
    **kwargs : dict
        Additional keyword arguments to pass to the least squares optimization
        function.

    Returns
    -------
    tuple
        A tuple containing:
        - R (numpy.ndarray): The 3x3 rotation matrix.
        - translation (numpy.ndarray): The 3-element translation vector.
    """
    if reticle_pts.shape != probe_pts.shape:
        raise ValueError("reticle_pts and probe_pts must have the same shape")
    if reticle_pts.shape[1] != 3:
        raise ValueError("reticle_pts and probe_pts must have 3 columns")

    R_homog = np.eye(4)
    reticle_pts_homog = ut.prepare_data_for_homogeneous_transform(reticle_pts)
    transformed_pts_homog = np.empty_like(reticle_pts_homog)

    def fun(theta):
        """cost function for least squares optimization"""
        R_homog[0:3, 0:3] = rot.combine_angles(*theta[0:3])
        R_homog[0:3, 3] = theta[3:6]  # translation
        np.matmul(reticle_pts_homog, R_homog.T, out=transformed_pts_homog)
        residuals = (transformed_pts_homog[:, 0:3] - probe_pts).flatten()
        return residuals

    # Initial guess of parameters
    theta0 = np.zeros(6)

    if probe_pts.shape[0] > 1:
        # Initial guess of rotation: align the vectors between the first two points
        probe_diff = np.diff(probe_pts[:2, :], axis=0)
        reticle_diff = np.diff(reticle_pts[:2, :], axis=0)
        Rinit = rot.rotation_matrix_from_vectors(
            reticle_diff.squeeze(), probe_diff.squeeze()
        )
        theta0[0:3] = Rotation.from_matrix(Rinit).as_euler("xyz")

    # Initial guess of translation: find the point on the reticle closest to zero
    smallest_pt = np.argmin(np.linalg.norm(reticle_pts, axis=1))
    theta0[3:6] = probe_pts[smallest_pt, :]

    res = opt.least_squares(fun, theta0, **kwargs)
    R, translation = _unpack_theta(res.x)
    if legacy_outputs:
        # last version had translation in global frame
        #
        # Also the last version for some reason calculated the tranpose of the
        # rotation matrix. Application of the rotation matrix was consistently
        # wrong in a way that accounted for this transpose, so the results were
        # correct.
        #
        # All of the transposes are confusing here: this is the inverse of the
        # rotation matrix, accounting for numpy being row-major, and
        # data points being row vectors

        translation = translation @ R  # Not R.T!
        return translation, R.T  # Not R!
    return R, translation


def apply_rotate_translate(pts, R, translation):
    """
    Apply rotation and translation to a set of points.

    Parameters
    ----------
    pts : numpy.ndarray
        The input points to be transformed.
    R : numpy.ndarray
        The 3x3 rotation matrix.
    translation : numpy.ndarray
        The 3-element translation vector.

    Returns
    -------
    numpy.ndarray
        The transformed points.
    """
    R_homog = ut.make_homogeneous_transform(R, translation)
    pts_homog = ut.prepare_data_for_homogeneous_transform(pts)
    # Transposed because points are assumed to be row vectors
    transformed_pts_homog = pts_homog @ R_homog.T
    return ut.extract_data_for_homogeneous_transform(transformed_pts_homog)


def inverse_rotate_translate(R, translation):
    """
    Compute the inverse rotation and translation.

    Parameters
    ----------
    R : numpy.ndarray
        The 3x3 rotation matrix.
    translation : numpy.ndarray
        The 3-element translation vector.

    Returns
    -------
    tuple
        A tuple containing:
        - R_inv (numpy.ndarray): The transpose of the rotation matrix.
        - tinv (numpy.ndarray): The inverse translation vector.
    """
    tinv = -translation @ R
    return R.T, tinv


def transform_reticle_to_probe(reticle_pts, R, translation):
    """
    Transform reticle points to probe points using rotation and translation.

    Parameters
    ----------
    probe_pts : np.array(N,3)
        Probe points to transform.
    R : np.array(3,3)
        Rotation matrix.
    translation : np.array(3,)
        Translation vector.

    Returns
    -------
    np.array(N,3)
        Transformed points.
    """
    return apply_rotate_translate(reticle_pts, R, translation)


def transform_probe_to_reticle(probe_pts, R, translation):
    """
    Transform probe points to reticle points using rotation and translation.

    Parameters
    ----------
    probe_pts : np.array(N,3)
        Probe points to transform.
    R : np.array(3,3)
        Rotation matrix.
    translation : np.array(3,)
        Translation vector.

    Returns
    -------
    np.array(N,3)
        Transformed points.
    """
    Rinv, tinv = inverse_rotate_translate(R, translation)
    return apply_rotate_translate(probe_pts, Rinv, tinv)
