"""
IO functions for SITK
"""

import SimpleITK as sitk # pragma: no cover
import os # pragma: no cover


def read_dicom(filename):  # pragma: no cover
    """
    Reader to import Dicom file and convert to sitk image

    Parameters
    ----------
    filename : String
        folder of .dcm image. If an individual image file name is passes, will
        read all .dcm files in that folder

    Returns
    -------
    SITK image
        SITK image from loaded dicom files.

    """

    if os.path.isdir(filename):
        dirname = filename
    else:
        dirname = os.path.dirname(filename)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dirname)
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def read_dcm(filename):  # pragma: no cover
    """
    Reader to import Dicom file and convert to sitk image.
    This fucntion is a wrapper on read_dicom to handle multiple naming
    conventions

    Parameters
    ----------
    filename : String
        folder of .dcm image. If an individual image file name is passes, will
        read all .dcm files in that folder

    Returns
    -------
    SITK image
        SITK image from loaded dicom files.

    """
    return read_dicom(filename)


def read_nii(filename):  # pragma: no cover
    """
    Reader to import nifti file and convert to sitk image
    This function is just a wrapper to match convention.

    Parameters
    ----------
    filename : String
        filename of .nii file.

    Returns
    -------
    SITK image
        SITK image from loaded dicom files..

    """
    return sitk.ReadImage(filename)


def read_nifti(filename):  # pragma: no cover
    """
    Reader to import nifti file and convert to sitk image
    This fucntion is a wrapper on read_nii to handle multiple naming
    conventions, which is in turn just an sitk wrapper.

    Parameters
    ----------
    filename : String
        filename of .nii file.

    Returns
    -------
    SITK image
        SITK image from loaded dicom files..

    """
    return read_nii(filename)


def read_tiff_stack(folder):  # pragma: no cover
    """
    Code to read a tiff stack
    THIS CODE IS INCOMPLETE: needs metatdata handling (resolution, etc.) and
    some thought about how to deal with large images.

    Parameters
    ----------
    folder : String folder with numerically ordered tiff images
        DESCRIPTION.

    Returns
    -------
    SITK image
        Tiff images stacked.

    """
    reader = sitk.ImageSeriesReader()
    lst = [
        x
        for x in os.listdir(
            folder,
        )
        if (".tif" in x)
    ]
    lst = [os.path.join(folder, x) for x in lst]
    reader.SetFileNames(lst)
    return reader.Execute()
