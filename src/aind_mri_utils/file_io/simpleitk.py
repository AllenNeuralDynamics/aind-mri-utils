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
):
    """Save a rigid transform to a SimpleITK (sitk) transform file

    The current implementation assumes that rotations are applied as: y = Rx +
    t, where R is the rotation matrix, x is the input point, and t is the
    translation vector. These vectors are assumed to be column vectors.

    A SimpleITK transform is written to `filename` using `sitk.WriteTransform`.

    Parameters
    ----------
    filename : str
        The name of the file where the transform will be saved.
    rotation_matrix : np.ndarray
        The transform to save. See notes for supported shapes.
    translation : np.ndarray, optional
        An explicit translation vector to override any translation inferred
        from `rotation_matrix`. Defaults to None.

    Raises
    ------
    ValueError
        If the shape of `rotation_matrix` is not supported.

    Notes
    -----
    - If `rotation_matrix` is (6,), the function calculates the rotation matrix
    using `rot.combine_angles` for the first three values and assigns the last
    three values as the translation vector.
    - If `translation` is provided, it overrides any translation inferred from
    `rotation_matrix`.
    - Supported shapes for `rotation_matrix` include:
    - np.array of shape (6,): Interpreted as rotation angles (first 3 values)
    and translation (last 3 values). A rigid transform is constructed using
    `aind_mri_utils.optimization.create_rigid_transform`.
    - np.array of shape (4, 4): A homogeneous transformation matrix.
    - np.array of shape (3, 4): A rigid transform with a rotation
    matrix and translation vector.
    - np.array of shape (3, 3): A rotation matrix without translation.
    Translation defaults to a zero vector.

    """

    if len(rotation_matrix) == 6:
        R = rot.combine_angles(*rotation_matrix[:3])
        found_translation = rotation_matrix[3:]
    elif rotation_matrix.shape == (4, 4):
        found_translation = rotation_matrix[:3, 3]
        R = rotation_matrix[:3, :3]
    elif rotation_matrix.shape == (3, 4):
        R = rotation_matrix[:, :3]
        found_translation = rotation_matrix[:, 3]
    elif rotation_matrix.shape == (3, 3):
        R = rotation_matrix
        found_translation = np.zeros(3)
    else:
        raise ValueError("Invalid transform shape")
    if translation is not None:
        found_translation = translation
    A = rot.rotation_matrix_to_sitk(R, translation=found_translation)
    sitk.WriteTransform(A, filename)


def load_sitk_transform(filename, homogeneous=False, invert=False):
    """
    Convert a sitk transform file to a 4x3 numpy array.

    Parameters
    ----------
    filename : string
        filename to load from.
    homogeneous : bool, optional
        If True, return a 4x4 homogeneous transform matrix. Default is False.

    Returns
    -------
    R: np.array(N,M)
        Rotation matrix. For three dimensional transforms: np.array(3,3). If
        homogeneous: np.array(4, 4)
    translation: np.array(L,)
        Translation vector.
    center: np.array(L,)
        Center of rotation.
    """
    A = sitk.ReadTransform(filename)
    if invert:
        A = A.GetInverse()
    R, translation, center = rot.sitk_to_rotation_matrix(A)
    if homogeneous:
        if not np.allclose(center, 0):
            raise NotImplementedError(
                "homogeneous only valid for transforms with center at 0"
            )
        R = rot.make_homogeneous_transform(R, translation)
    return R, translation, center
