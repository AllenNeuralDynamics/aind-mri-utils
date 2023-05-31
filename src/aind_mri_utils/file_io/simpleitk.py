"""
Functions for saving and loading transforms using SimpleITK.
"""
import numpy as np
import SimpleITK as sitk

from ..optimization import create_rigid_transform


def save_sitk_transform(filename, T, transpose_matrix=False):
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
    if len(T) == 6:
        trans = create_rigid_transform(T[0], T[1], T[2], T[3], T[4], T[5])

    elif T.shape == (4, 3):
        trans = T

    else:
        NotImplementedError("T must be a 4x3 or 6x1 array")
        return None

    A = sitk.AffineTransform(3)
    if transpose_matrix:
        A.SetMatrix(trans[:3, :3].T.flatten())
    else:
        A.SetMatrix(trans[:3, :3].flatten())
    A.SetTranslation(trans[3, :])
    sitk.WriteTransform(A, filename)


def load_sitk_transform(filename, transpose_matrix=False, outshape=(4, 3)):
    """
    Convert a sitk transform file to a 4x3 numpy array.

    Parameters
    ----------
    filename : string
        filename to load from.
    invert = bool, optional
        If true, invert the transform before saving.
        Default is False.

    Returns
    -------
    trans: np.array(4,3)
        Affine transform.

    """
    A = sitk.ReadTransform(filename)

    if outshape == (4, 3):
        matrix = np.array(A.GetParameters())[:9].reshape((3, 3))
        if transpose_matrix:
            matrix = matrix.T
        offset = np.array(A.GetParameters()[-3:])
        trans = np.vstack([matrix, offset])
        return trans

    else:
        NotImplementedError("outshape must be (4,3)")
