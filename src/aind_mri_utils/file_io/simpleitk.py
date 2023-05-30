from ..optimization import create_rigid_transform
import SimpleITK as sitk
import numpy as np
from unittest.mock import MagicMock, patch


def save_sitk_transform(filename,trans,invert = False):
    """
    Save transform to a sitk file (readable by Slicer);
    
    Parameters
    ----------
    filename : string
        filename to save to.
    T : np.array(4,4) or np.array(6,)
        Rigid transform to save.
        if a np.array(6,) is passed, a rigid transform is created using 
        aind_mri_utils.optimization.crate_rigid_transform.
    invert = bool, optional
        If true, invert the transform before saving.
        Default is False.
    
    """
    if len(T) == 6:
        T = create_rigid_transform(T[0],T[1],T[2],T[3],T[4],T[5])
    else:
        assert(T.shape==(4,4),"T must be a 4x4 matrix")
        trans = T
    
    A = sitk.AffineTransform(3)
    if invert:
        A.SetMatrix(trans[:3,:3].T.flatten())
    else:
        A.SetMatrix(trans[:3,:3].flatten())

    A.SetTranslation(trans[3,:])
    sitk.WriteTransform(A.GetInverse(),filename)


def load_sitk_transform(filename,invert = False):
    """
    Convert a sitk transform file to a 4x4 numpy array.

    Parameters
    ----------
    filename : string
        filename to load from.
    invert = bool, optional
        If true, invert the transform before saving.
        Default is False.
    
    Returns
    -------
    trans: np.array(4,4)
        Affine transform.

    """
    A = sitk.ReadTransform(filename)
    if invert:
        A=A.GetInverse()
    matrix = np.array(A.GetParameters())[:9].reshape((3,3)).T
    offset = np.array(A.GetParameters()[-3:])
    trans = np.vstack([matrix,offset])
    return trans

