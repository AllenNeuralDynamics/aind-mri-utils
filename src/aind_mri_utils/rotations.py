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
    rx : Float
        Angle to rotate about X
    ry : Float
        Angle to rotate about Y
    rz : Float
        Angle to rotate about Z
    degrees : Bool, optional
        Are the rotations in degrees?. The default is True.
    order: string, optional
        Order of axes to transform as string. Default is 'xyz',
        meaning transform will happen x-->y-->z

    Returns
    -------
    Scipy 3d rotation
        scipy 3.

    """
    return Rotation.from_euler(order, [rx, ry, rz], degrees=True)


def rotate_about_and_translate(points, rotation, pivot, translation):
    """
    Rotates points about a pivot point,
    then apply translation (add the translation values)


    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rototation : Scipy `Rotation` object
        use `define_euler_rotation` or
        `scipy.spatial.transform.Rotation` constructor to create
    pivot : (1x3) numpy array
        Point to rotate around
    translation: (1x3) numpy array
        Additional translation to apply to points


    Returns
    -------
    (Nx3) numpy array
        Rotated points

    """
    return rotate_about(points, rotation, pivot) + translation


def rotate_about(points, rotation, pivot):
    """
    Rotates points about a pivot point

    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rototation : Scipy `Rotation` object
        use `define_euler_rotation` or
        `scipy.spatial.transform.Rotation` constructor to create
    pivot : (1x3) numpy array
        Point to rotate around

    Returns
    -------
    (Nx3) numpy array
        Rotated points

    """
    return rotation.apply(points - pivot) + pivot


def rotation_matrix_to_sitk(
    rotation, center=np.array((0, 0, 0)), translation=np.array((0, 0, 0))
):
    """Convert numpy array rotation matrix to sitk affine

    Parameters
    ----------
    rotation : np.ndarray (3 x 3)
        matrix representing rotation matrix in three dimensions
    center : np.ndarray (3)
        vector representing center of rotation, default is origin
    translation : np.ndarray (3)
        vector representing translation of transform (after rotation), default
        is zero

    Returns
    -------
    SITK transform
        with parameters matching the input object

    """
    S = sitk.AffineTransform(3)
    S.SetMatrix(tuple(rotation.flatten()))
    S.SetTranslation(translation.tolist())
    S.SetCenter(center.tolist())
    return S


def sitk_to_rotation_matrix(S):
    """Convert sitk affine transform to numpy array rotation matrix

    Parameters
    ----------
    S : SITK transform
        affine transform object

    Returns
    -------
    np.ndarray (3 x 3)
        matrix representing rotation matrix in three dimensions

    """
    R = np.array(S.GetMatrix()).reshape((3, 3))
    translation = np.array(S.GetTranslation())
    center = np.array(S.GetCenter())
    return R, translation, center


def scipy_rotation_to_sitk(
    rotation, center=np.array((0, 0, 0)), translation=np.array((0, 0, 0))
):
    """
    Convert Scipy 'Rotation' object to equivalent sitk

    Parameters
    ----------
    rotation : Scipy `Rotation` object
        use `define_euler_rotation` or
        `scipy.spatial.transform.Rotation` constructor to create

    Returns
    -------
    SITK transform
        with parameters matching the input object

    """
    S = rotation_matrix_to_sitk(rotation.as_matrix(), center, translation)
    return S


def rotation_matrix_from_vectors(a, b):
    """Find rotation matrix to align a with b


    Parameters
    ----------
    a : np.ndarray (N)
        vector to be aligned with b
    b : np.ndarray (N)
        vector

    Returns
    -------
    rmat : np.ndarray (NxN)
        Rotation matrix such that `rmat @ a` is parallel to `b`
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
    rotmat = np.eye(nd) + ax + ax @ ax * (1 / (1 + c))
    return rotmat


def _rotate_mat_by_single_euler(mat, axis, angle):
    "Helper function that rotates a matrix by a single Euler angle"
    rotmat = Rotation.from_euler(axis, angle).as_matrix().squeeze()
    return mat @ rotmat


def roll(input_mat, angle):  # rotation around x axis (bank angle)
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


def pitch(input_mat, angle):  # rotation around y axis (elevation angle)
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


def yaw(input_mat, angle):  # rotation around z axis (heading angle)
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
    R : np.array(N,N)
        Rotation matrix.
    translation : np.array(N,)
        Translation vector.

    Returns
    -------
    np.array(N+1,N+1)
        homogeneous transformation matrix
    """
    N, M = R.shape
    if N != M:
        raise ValueError("R must be square")
    if N != translation.shape[0]:
        raise ValueError("R and translation must have same size")
    if scaling is not None and scaling.shape[0] != N:
        raise ValueError("scaling must have same size as R")

    if scaling is None:
        Radj = R
    else:
        Rscaling = np.diag(scaling)
        Radj = Rscaling @ R
    R_homog = np.eye(N + 1)
    R_homog[0:N, 0:N] = Radj
    R_homog[0:N, N] = translation
    return R_homog


def prepare_data_for_homogeneous_transform(pts):
    """
    Prepare points for homogeneous transformation.

    Parameters
    ----------
    pts : np.array(N,M) or np.array(M)
        array of N M-D points.

    Returns
    -------
    np.array(N,M+1) or np.array(M+1)
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
    R_homog = make_homogeneous_transform(R, translation)
    return _apply_homogeneous_transform_to_transposed_pts(pts, R_homog)


def apply_rotate_translate_scale(pts, R, translation, scaling):
    R_homog = make_homogeneous_transform(R, translation, scaling)
    return _apply_homogeneous_transform_to_transposed_pts(pts, R_homog)


def apply_inverse_rotate_translate_scale(pts, R, translation, scaling):
    scaling_inv = 1 / scaling
    R_inv = R.T @ np.diag(scaling_inv)
    t_inv = -R_inv @ translation
    R_homog = make_homogeneous_transform(R_inv, t_inv)
    return _apply_homogeneous_transform_to_transposed_pts(pts, R_homog)


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
    Rinv = R.T
    return Rinv, tinv


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
