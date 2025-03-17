"""
IO functions for SITK
"""

import os

import SimpleITK as sitk


def read_image(filename):
    """
    Reads generic image files/folders in SITK using
    Currently explicitly supported: .dcm, .nii, .tiff
    Folders/stacks will be read for .dcm and .tiff files
    Other formats work only if supported they work with sitk.ReadImage()

    Parameters
    ----------
    filename : String
        filename or folder of files.

    Returns
    -------
    SITK image
        SITK image from loaded dicom files.

    """

    if os.path.isdir(filename):
        # Look for .tifs
        tiff_list = [
            x
            for x in os.listdir(
                filename,
            )
            if (".tif" in x)
        ]
        # Read tifs if they exist
        if len(tiff_list) > 0:
            return read_tiff_stack(filename)
        else:
            # in the absence of tifs, assume folders are dcm
            return read_dicom(filename)

    else:
        if (".nii" in filename) or (".nifti" in filename):
            return read_nii(filename)
        elif (".dcm" in filename) or (
            os.path.splitext(filename)[0] == filename
        ):
            return read_dicom(filename)
        else:
            # If none of the conditions above are reached, try to
            # use the default reader. This will throw an error if there
            # are any problems
            return sitk.ReadImage(filename)


def read_dicom(filename):
    """Reader to import Dicom file and convert to sitk image

    See
    https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReader_docs.html#lbl-dicom-series-reader

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
    dicom_names = reader.GetGDCMSeriesFileNames(
        dirname, useSeriesDetails=True, loadSequences=True
    )
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def read_dcm(filename):
    """
    Reader to import Dicom file and convert to sitk image.
    This function is a wrapper on read_dicom to handle multiple naming
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


def read_nii(filename):
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


def read_nifti(filename):
    """
    Reader to import nifti file and convert to sitk image
    This function is a wrapper on read_nii to handle multiple naming
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


def read_tiff_stack(folder):
    """
    Code to read a tiff stack
    THIS CODE IS INCOMPLETE: needs metadata handling (resolution, etc.) and
    some thought about how to deal with large images.

    Parameters
    ----------
    folder : String folder with numerically ordered tiff images
        Folder containing ONLY .tif images

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


def write_nii(image, filename):
    """

    Write an sitk image to .nii file


    Parameters
    ----------
    image : SITK image
        Image to save.

    filename : filename
        Filename to save

    """
    f_name, f_ext = os.path.splitext(filename)
    if len(f_ext) == 0:
        f_ext = ".nii"
    elif f_ext != ".nii":
        raise Exception("Filename should be a .nii file")
    sitk.WriteImage(image, f_name + f_ext)


def write_dicom(image, foldername):
    """
    Save SITK image as dicom stack.
    Heavily borrowed from
    https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html

    Parameters
    ----------
    image : SITK image
        Image to save.

    filename : filename
        Filename to save

    """
    # Need to figure out options to copy metadata
    raise NotImplementedError(
        "Dicom file writer still needs to be implemented"
    )
