"""
Code for rotations of points
"""

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation


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

    rotmat = rotation.as_matrix().reshape((9,))
    params = np.concatenate((rotmat, np.zeros((3,), dtype=np.float64)))
    newTransform = sitk.AffineTransform(3)
    newTransform.SetParameters(params.tolist())
    newTransform.SetTranslation(translation.tolist())
    newTransform.SetCenter(center.tolist())
    return newTransform
