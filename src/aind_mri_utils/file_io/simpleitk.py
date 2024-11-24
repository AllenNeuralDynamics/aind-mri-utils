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
    Save a rigid transform to a SimpleITK (sitk) transform file, compatible
    with tools like Slicer.

    The current implementation assumes that rotations are applied as:
    y = Rx + t, where R is the rotation matrix, x is the input point, and t
    is the translation vector. These vectors are assumed to be column vectors.

    The legacy flag is true, this behavior is changed to:
    transpose(y) = transpose(x) * R + transpose(t), here the vectors are row
    vectors. Importantly, the rotation matrix is the transposed rotation matrix
    when operating on column vectors. Since the saved transform is intended to
    be applied to column vectors, the rotation matrix is effectively inverted
    when saved.

    To achieve the behavior of the old version, `transpose_matrix` and `legacy`
    must both be true.

    Parameters
    ----------
    filename : str
        The name of the file where the transform will be saved.
    rotation_matrix : np.ndarray
        The transform to save. Accepted formats:
        - np.array of shape (6,): Interpreted as rotation angles (first 3
          values)
          and translation (last 3 values). A rigid transform is constructed
          using `aind_mri_utils.optimization.create_rigid_transform`.
        - np.array of shape (4, 4): A homogeneous transformation matrix.
        - np.array of shape (3, 4): A non-legacy rigid transform with a
          rotation matrix and translation vector.
        - np.array of shape (4, 3): A legacy rigid transform with a rotation
          matrix and translation vector.
        - np.array of shape (3, 3): A rotation matrix without translation.
          Translation defaults to a zero vector.
    translation : np.ndarray, optional
        An explicit translation vector to override any translation inferred
        from `rotation_matrix`. Defaults to None.
    transpose_matrix : bool, optional
        If True and `legacy` is enabled, transposes the rotation matrix before
        saving.  Only applicable for legacy transforms. Defaults to False.
    legacy : bool, optional
        Determines how the transform is interpreted:
        - When `legacy` is True, the function expects rotation and translation
          to follow older conventions:
            * For (4, 4) matrices, translation is derived from row 3, columns
              0–2.
            * For (4, 3) matrices, rotation is taken from rows 0–2, and
              translation is taken from row 3.
            * Transposing the rotation matrix (`transpose_matrix=True`) is
              allowed.
        - When `legacy` is False, the function uses a more modern
          interpretation:
            * For (4, 4) matrices, translation is derived from row 0–2, column
              3.
            * For (3, 4) matrices, rotation is taken from columns 0–2, and
              translation is taken from column 3.
            * Transposing the rotation matrix is disallowed.
        Defaults to False.

    Raises
    ------
    ValueError
        - If `rotation_matrix` has an unsupported shape.
        - If `transpose_matrix` is True when `legacy` is False.
        - If a mismatch between the shape of `rotation_matrix` and the `legacy`
          flag is detected.

    Notes
    -----
    - If `rotation_matrix` is (6,), the function calculates the rotation matrix
      using `rot.combine_angles` for the first three values and assigns the
      last three values as the translation vector.
    - If `translation` is provided, it overrides any translation inferred
      from `rotation_matrix`.

    Outputs
    -------
    A SimpleITK transform is written to `filename` using `sitk.WriteTransform`.
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
        if not np.allclose(center, 0):
            raise NotImplementedError(
                "homogeneous only valid for transforms with center at 0"
            )
        if legacy:
            raise ValueError(
                "homogeneous only valid for non-legacy transforms"
            )
        R = rot.make_homogeneous_transform(R, translation)
    return R, translation, center
