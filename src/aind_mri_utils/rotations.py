"""
Code for rotations of points
"""

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from . import utils as ut


def define_euler_rotation(rx, ry, rz, degrees=True, order="xyz"):
    """
    Wrapper of scipy.spatial.transform.Rotation.from_euler

    Parameters
    ----------
    rx : float
        Angle to rotate about X.
    ry : float
        Angle to rotate about Y.
    rz : float
        Angle to rotate about Z.
    degrees : bool, optional
        Are the rotations in degrees? The default is True.
    order : str, optional
        Order of axes to transform as string. Default is 'xyz',
        meaning transform will happen x-->y-->z.

    Returns
    -------
    scipy.spatial.transform.Rotation
        Scipy 3D rotation object.
    """
    return Rotation.from_euler(order, [rx, ry, rz], degrees=True)


def rotate_about_and_translate(points, rotation, pivot, translation):
    """
    Rotates points about a pivot point, then apply translation.

    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rotation : scipy.spatial.transform.Rotation
        Rotation object.
    pivot : numpy.ndarray
        Point to rotate around.
    translation : numpy.ndarray
        Additional translation to apply to points.

    Returns
    -------
    numpy.ndarray
        Rotated points.
    """
    return rotate_about(points, rotation, pivot) + translation


def rotate_about(points, rotation, pivot):
    """
    Rotates points about a pivot point.

    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rotation : scipy.spatial.transform.Rotation
        Rotation object.
    pivot : numpy.ndarray
        Point to rotate around.

    Returns
    -------
    (Nx3) numpy array
            Rotated points

    """
    return rotation.apply(points - pivot) + pivot


def rotation_matrix_to_sitk(
    rotation, center=np.array((0, 0, 0)), translation=np.array((0, 0, 0))
):
    """
    Convert numpy array rotation matrix to sitk affine.

    Parameters
    ----------
    rotation : numpy.ndarray
        Matrix representing rotation matrix in three dimensions.
    center : numpy.ndarray, optional
        Vector representing center of rotation, default is origin.
    translation : numpy.ndarray, optional
        Vector representing translation of transform (after rotation), default
        is zero.

    Returns
    -------
    sitk.AffineTransform
        SITK transform with parameters matching the input object.
    """
    S = sitk.AffineTransform(3)
    S.SetMatrix(tuple(rotation.flatten()))
    S.SetTranslation(translation.tolist())
    S.SetCenter(center.tolist())
    return S


def sitk_to_rotation_matrix(S):
    """
    Convert sitk affine transform to numpy array rotation matrix.

    Parameters
    ----------
    S : sitk.AffineTransform
        Affine transform object.

    Returns
    -------
    numpy.ndarray
        Matrix representing rotation matrix in three dimensions.
    numpy.ndarray
        Translation vector.
    numpy.ndarray
        Center of rotation.
    """
    R = np.array(S.GetMatrix()).reshape((3, 3))
    translation = np.array(S.GetTranslation())
    center = np.array(S.GetCenter())
    return R, translation, center


def scipy_rotation_to_sitk(
    rotation, center=np.array((0, 0, 0)), translation=np.array((0, 0, 0))
):
    """
    Convert Scipy 'Rotation' object to equivalent sitk.

    Parameters
    ----------
    rotation : scipy.spatial.transform.Rotation
        Rotation object.
    center : numpy.ndarray, optional
        Center of rotation, default is origin.
    translation : numpy.ndarray, optional
        Translation vector, default is zero.

    Returns
    -------
    sitk.AffineTransform
        SITK transform with parameters matching the input object.
    """
    S = rotation_matrix_to_sitk(rotation.as_matrix(), center, translation)
    return S


def rotation_matrix_from_vectors(a, b):
    """
    Find rotation matrix to align a with b.

    Parameters
    ----------
    a : numpy.ndarray
        Vector to be aligned with b.
    b : numpy.ndarray
        Vector.

    Returns
    -------
    numpy.ndarray
        Rotation matrix such that `rotation_matrix @ a` is parallel to `b`.
    """
    # Follows Rodrigues` rotation formula
    # https://math.stackexchange.com/a/476311

    nd = a.shape[0]
    if nd != b.shape[0]:
        raise ValueError("a must be same size as b")
    na = ut.norm_vec(a)
    nb = ut.norm_vec(b)
    c = np.dot(na, nb)
    if c == -1:
        return -np.eye(nd)
    v = np.cross(na, nb)
    ax = ut.skew_symmetric_cross_product_matrix(v)
    rotation_matrix = np.eye(nd) + ax + ax @ ax * (1 / (1 + c))
    return rotation_matrix


def _rotate_mat_by_single_euler(mat, axis, angle):
    "Helper function that rotates a matrix by a single Euler angle"
    rotation_matrix = Rotation.from_euler(axis, angle).as_matrix().squeeze()
    return mat @ rotation_matrix


def roll(input_mat, angle):
    """
    Apply a rotation around the x-axis (roll/bank angle) to the input matrix.

    Parameters
    ----------
    input_mat : numpy.ndarray
        The input matrix to be rotated.
    angle : float
        The angle of rotation around the x-axis in radians.

    Returns
    -------
    numpy.ndarray
        The rotated matrix.
    """
    return _rotate_mat_by_single_euler(input_mat, "x", angle)


def pitch(input_mat, angle):
    """
    Apply a rotation around the y-axis (pitch/elevation angle) to the input
    matrix.

    Parameters
    ----------
    input_mat : numpy.ndarray
        The input matrix to be rotated.
    angle : float
        The angle of rotation around the y-axis in radians.

    Returns
    -------
    numpy.ndarray
        The rotated matrix.
    """
    return _rotate_mat_by_single_euler(input_mat, "y", angle)


def yaw(input_mat, angle):
    """
    Apply a rotation around the z-axis (yaw/heading angle) to the input matrix.

    Parameters
    ----------
    input_mat : numpy.ndarray
        The input matrix to be rotated.
    angle : float
        The angle of rotation around the z-axis in radians.

    Returns
    -------
    numpy.ndarray
        The rotated matrix.
    """
    return _rotate_mat_by_single_euler(input_mat, "z", angle)


def extract_angles(mat):
    """
    Extract the Euler angles (roll, pitch, yaw) from a rotation matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        The rotation matrix from which to extract the Euler angles.

    Returns
    -------
    tuple of float
        The extracted Euler angles (roll, pitch, yaw) in radians.
    """
    return tuple(Rotation.from_matrix(mat).as_euler("xyz"))


def combine_angles(x, y, z):
    """
    Combine Euler angles (roll, pitch, yaw) into a rotation matrix.

    Parameters
    ----------
    x : float
        The roll angle (rotation around the x-axis) in radians.
    y : float
        The pitch angle (rotation around the y-axis) in radians.
    z : float
        The yaw angle (rotation around the z-axis) in radians.

    Returns
    -------
    numpy.ndarray
        The resulting rotation matrix.
    """
    return Rotation.from_euler("xyz", [x, y, z]).as_matrix().squeeze()


def make_homogeneous_transform(R, translation, scaling=None):
    """
    Combines a rotation matrix and translation into a homogeneous transform.

    Parameters
    ----------
    R : numpy.array(N,N)
        Rotation matrix.
    translation : numpy.array(N,)
        Translation vector.
    scaling : numpy.ndarray, optional
        Scaling factors, by default None.

    Returns
    -------
    numpy.ndarray
        Homogeneous transformation matrix.
    """
    N, M = R.shape
    if N != M:
        raise ValueError("R must be square")
    if N != translation.shape[0]:
        raise ValueError("R and translation must have same size")
    if scaling is not None and scaling.shape[0] != N:
        raise ValueError("scaling must have same size as R")

    if scaling is None:
        R_adj = R
    else:
        R_adj = np.diag(scaling) @ R
    R_homog = np.eye(N + 1)
    R_homog[0:N, 0:N] = R_adj
    R_homog[0:N, N] = translation
    return R_homog


def affine_and_translation_from_homogeneous(R_homog):
    """
    Extract rotation and translation from a homogeneous transform.

    Parameters
    ----------
    R_homog : numpy.ndarray
        Homogeneous transformation matrix.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    numpy.ndarray
        Translation vector.
    """
    N, M = R_homog.shape
    if N != M:
        raise ValueError("R_homog must be square")
    R = R_homog[0:N, 0:N]
    translation = R_homog[0:N, N]
    return R, translation


def prepare_data_for_homogeneous_transform(pts):
    """
    Prepare points for homogeneous transformation.

    Parameters
    ----------
    pts : np.array(N,M) or np.array(M)
        array np.array(N,M) or np. poin(M)
        aeturns
    -------
    numpy.ndarray
        (M+1)-D points with 1 in the last position.
    """
    nd = pts.ndim
    if nd == 1:
        M = pts.shape[0]
        pts_homog = np.ones(M + 1)
        pts_homog[0:M] = pts
    elif nd == 2:
        N, M = pts.shape
        pts_homog = np.ones((N, M + 1))
        pts_homog[:, 0:M] = pts
    else:
        raise ValueError("pts must be 1D or 2D")
    return pts_homog


def extract_data_for_homogeneous_transform(pts_homog):
    """
    Extract points formatted for homogeneous transformation.

    Parameters
    ----------
    pts_homog : np.array(N,M+1) or np.array(M+1)
        (M+1)-D points with 1 in the last position.

    Returns
    -------
    np.array(N,M) or np.array(M)
        array of N M-D points.
    """
    nd = pts_homog.ndim
    if nd == 1:
        M = pts_homog.shape[0] - 1
        pts = pts_homog[0:M]
    elif nd == 2:
        N, M = pts_homog.shape
        pts = pts_homog[:, 0 : (M - 1)]  # noqa: E203
    else:
        raise ValueError("pts_homog must be 1D or 2D")
    return pts


def _apply_homogeneous_transform_to_transposed_pts(pts, R_homog):
    pts_homog = prepare_data_for_homogeneous_transform(pts)
    transformed_pts_homog = pts_homog @ R_homog.T
    return extract_data_for_homogeneous_transform(transformed_pts_homog)


def apply_affine(pts, affine_R, translation):
    """
    Apply an affine transformation to a set of points.

    Parameters
    ----------
    pts : numpy.ndarray
        The input points to be transformed.
    affine_R : numpy.ndarray
        The affine rotation matrix.
    translation : numpy.ndarray
        The translation vector.

    Returns
    -------
    numpy.ndarray
        The transformed points.
    """
    R_homog = make_homogeneous_transform(affine_R, translation)
    return _apply_homogeneous_transform_to_transposed_pts(pts, R_homog)


def invert_affine(affine_R, translation):
    """
    Invert an affine transformation.

    Parameters
    ----------
    affine_R : numpy.ndarray
        The affine rotation matrix.
    translation : numpy.ndarray
        The translation vector.

    Returns
    -------
    numpy.ndarray
        The inverted rotation matrix.
    numpy.ndarray
        The inverted translation vector.
    """
    R_inv = np.linalg.inv(affine_R)
    t_inv = -R_inv @ translation
    return R_inv, t_inv


def apply_inverse_affine(pts, affine_R, translation):
    """
    Apply the inverse of an affine transformation to a set of points.

    Parameters
    ----------
    pts : numpy.ndarray
        The input points to be transformed.
    affine_R : numpy.ndarray
        The affine rotation matrix.
    translation : numpy.ndarray
        The translation vector.

    Returns
    -------
    numpy.ndarray
        The transformed points.
    """
    R_inv, t_inv = invert_affine(affine_R, translation)
    return apply_affine(pts, R_inv, t_inv)


def affine_from_rotate_scale(R, scale=None):
    """
    Create an affine transformation matrix from a rotation matrix and scaling
    factors.

    Parameters
    ----------
    R : numpy.ndarray
        The rotation matrix.
    scale : numpy.ndarray, optional
        The scaling factors, by default None.

    Returns
    -------
    numpy.ndarray
        The affine transformation matrix.
    """
    if scale is None:
        scale = np.ones(3)
    R_adj = np.diag(scale) @ R
    return R_adj


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
    return apply_affine(pts, R, translation)


def invert_rotate_translate(R, translation):
    """
    Compute the inverse rotation and translation.

    Parameters
    ----------
    R : numpy.ndarray
        The 3x3 rotation matrix. Must satisfy R.T @ R = I.
    translation : numpy.ndarray
        The 3-element translation vector.

    Returns
    -------
    tuple
        A tuple containing:
        - R_inv (numpy.ndarray): The transpose of the rotation matrix.
        - t_inv (numpy.ndarray): The inverse translation vector.

    Notes
    -----
    R is assumed to be a rotation matrix, but this function does not check
    that it is orthogonal. The caller is responsible for ensuring that R is
    a valid rotation matrix.
    """
    R_inv = R.T
    t_inv = -translation @ R
    return R_inv, t_inv


def create_homogeneous_from_euler_and_translation(rx, ry, rz, tx, ty, tz):
    """
    Create a homogeneous transformation matrix from Euler angles and
    translation.

    Parameters
    ----------
    rx : float
        Rotation angle around the x-axis in radians.
    ry : float
        Rotation angle around the y-axis in radians.
    rz : float
        Rotation angle around the z-axis in radians.
    tx : float
        Translation along the x-axis.
    ty : float
        Translation along the y-axis.
    tz : float
        Translation along the z-axis.

    Returns
    -------
    numpy.ndarray
        Homogeneous transformation matrix.
    """
    R = combine_angles(rx, ry, rz)
    t = np.array([tx, ty, tz])
    return make_homogeneous_transform(R, t)


def ras_to_lps_transform(R, translation=None):
    """
    Transforms a rotation matrix and translation vector from RAS to LPS
    coordinate system, or vice-versa.

    Parameters
    ----------
    R : numpy.ndarray
        A 3x3 rotation matrix.
    translation : numpy.ndarray, optional
        A 3-element translation vector. If None, a zero vector is used. Default
        is None.

    Returns
    -------
    numpy.ndarray
        The transformed 3x3 rotation matrix in LPS coordinate system.
    numpy.ndarray
        The transformed 3-element translation vector in LPS coordinate system.

    Raises
    ------
    ValueError
        If R is not a 3x3 matrix.
    """
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    if translation is None:
        translation = np.zeros(3)
    T = make_homogeneous_transform(R, translation)
    ras2lps = np.diag([-1, -1, 1, 1])
    T_out = ras2lps @ T @ ras2lps
    R_out = T_out[:3, :3]
    translation_out = T_out[:3, 3]
    return R_out, translation_out


def compose_transforms(R_1, translation_1, *args):
    """
    Compose a series of rotation matrices and translation vectors.

    Parameters
    ----------
    R_1 : numpy.ndarray
        The initial rotation matrix.
    translation_1 : numpy.ndarray
        The initial translation vector.
    *args : tuple
        Additional rotation matrices and translation vectors. Must be provided
        in pairs (R_2, translation_2, ...). The rotation matrix R_1 is applied
        first, followed by the rotation matrix R_2, and so on. The translation
        vector translation_1 is applied first, followed by the translation
        vector translation_2, and so on.

    Returns
    -------
    numpy.ndarray
        The composed rotation matrix.
    numpy.ndarray
        The composed translation vector.

    Raises
    ------
    ValueError
        If the number of additional arguments is not even.
    """
    nargs = len(args)
    if nargs == 0:
        return R_1, translation_1
    elif nargs >= 2:
        if nargs % 2 != 0:
            raise ValueError("Invalid number of arguments")
        R_2 = args[0]
        translation_2 = args[1]
        R = R_2 @ R_1
        translation = translation_2 + R_2 @ translation_1
        return compose_transforms(R, translation, *args[2:])
    else:
        raise ValueError("Invalid number of arguments")


def itk_to_slicer_transform(itk_transform):
    """
    Converts an ITK transform to a Slicer transform.

    This function converts a given ITK transform, which uses the LPS coordinate
    system, to a Slicer transform, which uses the RAS coordinate system. The
    ITK transform is assumed to be a 4x4 homogeneous transformation matrix.

    Parameters
    ----------
    itk_transform : numpy.ndarray
        A 4x4 homogeneous transformation matrix representing the ITK transform.

    Returns
    -------
    numpy.ndarray
        A 3x3 numpy.ndarray representing the rotation matrix of the Slicer
        transform.
    numpy.ndarray
        A 1x3 numpy.ndarray representing the translation vector of the Slicer
        transform.
    """
    R, translation = ras_to_lps_transform(
        itk_transform[:3, :3], itk_transform[:3, 3]
    )
    T = make_homogeneous_transform(R, translation)
    transform_to_parent_RAS = np.linalg.inv(T)
    return transform_to_parent_RAS[:3, :3], transform_to_parent_RAS[:3, 3]
