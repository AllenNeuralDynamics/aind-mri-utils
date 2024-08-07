"""
Functions for saving and loading transforms using SimpleITK.
"""

import numpy as np
import SimpleITK as sitk

from .. import rotations as rot


def save_sitk_transform(
    filename,
    rotation_matrix,
    translation=None,
    transpose_matrix=False,
    legacy=False,
):
    """
    Save transform to a sitk file (readable by Slicer);

    Parameters
    ----------
    filename : string
        filename to save to.
    T : np.array(4,3), np.array(4x4), or np.array(6,)
        Rigid transform t
        o save.
        if a np.array(6,) is passed, a rigid transform is created using
        aind_mri_utils.optimization.crate_rigid_transform.
    invert = bool, optional
        If true, invert the transform before saving.
        Default is False.


    """
    if len(rotation_matrix) == 6:
        R = rot.combine_angles(*rotation_matrix[:3])
        found_translation = rotation_matrix[3:]
    elif rotation_matrix.shape == (4, 4):
        if legacy:
            found_translation = rotation_matrix[3, :3]
        else:
            found_translation = rotation_matrix[:3, 3]
        R = rotation_matrix[:3, :3]
    elif rotation_matrix.shape == (3, 4) and not legacy:
        R = rotation_matrix[:, :3]
        found_translation = rotation_matrix[:, 3]
    elif rotation_matrix.shape == (4, 3) and legacy:
        R = rotation_matrix[:3, :]
        found_translation = rotation_matrix[3, :]
    elif rotation_matrix.shape == (3, 3):
        R = rotation_matrix
        found_translation = np.zeros(3)
    else:
        raise ValueError("Invalid transform shape and legacy flag")
    if translation is not None:
        found_translation = translation
    if transpose_matrix:
        if not legacy:
            raise ValueError(
                "transpose_matrix only valid for legacy transforms"
            )
        R = R.T
    A = rot.rotation_matrix_to_sitk(R, translation=found_translation)
    sitk.WriteTransform(A, filename)


def load_sitk_transform(
    filename, homogeneous=False, legacy=False, invert=False
):
    """
    Convert a sitk transform file to a 4x3 numpy array.

    Parameters
    ----------
    filename : string
        filename to load from.
    homogeneous : bool, optional
        If True, return a 4x4 homogeneous transform matrix. Default is False.
    legacy : bool, optional
        If True, return a 4x3 transform matrix with the translation as the
        last row. Default is False

    Returns
    -------
    R: np.array(N,M)
        Rotation matrix. For three dimensional transforms: np.array(3,3). If
        homogeneous: np.array(4, 4), if legacy: np.array(4, 3)
    translation: np.array(L,)
        Translation vector. Not returned if legacy is True.
    center: np.array(L,)
        Center of rotation. Not returned if legacy is True.
    """
    A = sitk.ReadTransform(filename)
    if invert:
        A = A.GetInverse()
    R, translation, center = rot.sitk_to_rotation_matrix(A)
    if legacy:
        R = np.vstack((R, translation))
        return R
    if homogeneous:
        if legacy:
            raise ValueError(
                "homogeneous only valid for non-legacy transforms"
            )
        R = rot.make_homogeneous_transform(R, translation)
    return R, translation, center
