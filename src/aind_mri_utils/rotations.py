"""
Code for rotations of points
"""

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation


def define_euler_rotation(rx, ry, rz, degrees=True, order="xyz"):
    """
    wrapper on scipy.spatial.transform.rotation

    Parameters
    ----------
    rx : Scalar
        DESCRIPTION.
    ry : Scalar
        DESCRIPTION.
    rz : Scalar
        DESCRIPTION.
    degrees : Bool, optional
        Are the rotations in degrees?. The default is True.
    order: string,optional
        Order of axes to transform as a sorted string. Default is 'xyz'

    Returns
    -------
    Scipy 3d rotation
        scipy 3.

    """
    return Rotation.from_euler(order, [rx, ry, rz], degrees=True)


def rotate_about_and_translate(points, rotation, pivot, translation):
    """
    Rotates points about a particular pivot point, then apply translation


    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rototation : Scipy rotation
        use "define_euler_rotation" to create
    pivot : (1x3) numpy array
        Point to rotate around
    translation: (1x3) numpy array
        Additonal translation to apply to points


    Returns
    -------
    (Nx3) numoy array
        Rotated points

    """
    return rotate_about(points, rotation, pivot) - translation


def rotate_about(points, rotation, pivot):
    """
    Rotates points about a particular pivot point

    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rototation : Scipy rotation
        use "define_euler_rotation" to create
    pivot : (1x3) numpy array
        Point to rotate around

    Returns
    -------
    (Nx3) numoy array
        Rotated points

    """
    return rotation.apply(points - pivot) + pivot


def scipy_rotation_to_sitk(
    rotation, center=np.array((0, 0, 0)), translation=np.array((0, 0, 0))
):
    """


    Parameters
    ----------
    rotation : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    rotmat = rotation.as_matrix().reshape((9,))
    params = np.concatenate((rotmat, np.zeros((3,), dtype=np.float64)))
    newTransform = sitk.AffineTransform(3)
    newTransform.SetParameters(params.tolist())
    newTransform.SetTranslation(translation.tolist())
    newTransform.SetCenter(center.tolist())
    return newTransform
