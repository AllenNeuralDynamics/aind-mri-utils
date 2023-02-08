import SimpleITK as sitk
import os


def read_dicom(filename):
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


def read_dcm(filename):
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


def read_tiff_stack(folder):
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


def resample(
    image,
    transform=None,
    output_spacing=None,
    output_direction=None,
    output_origin=None,
    output_size=None,
    interpolator=sitk.sitkLinear,
):
    """
    Wrapper to generically handle sitk resampling on different imag
    matricies. Includes optional application of a transform.
    Only 3d is currently implemented; need to at at least 2D.

    Parameters
    ----------
    image : SITK image
        image to transform.
    transform : SITK Affine Transform, optional
        If no transform is passed, use a identity transform matrix
    output_spacing : (Nx1) array, optional
        If not passed, coppies from image
    output_direction : (N^2x1) array, optional
        If not passed, coppies from image
    output_origin : (Nx1) array, optional
        If not passed, coppies from image
    output_size : (Nx1) array, optional
        If not passed, computes automatically to fully encompus
        transformed image.
    interpolator: sitk Interpolator,optional
        If not passed, defaults to sitk.sitkLinear
        See sitk documentation for optios.

    Returns
    -------
    resampled_image : SITK image
        resampled image with transform applied.

    """
    if len(image.GetSize()) == 3:
        return resample3D(
            image,
            transform=transform,
            output_spacing=output_spacing,
            output_direction=output_direction,
            output_origin=output_origin,
            output_size=output_size,
            interpolator=interpolator,
        )
    else:
        raise NotImplementedError(
            "Resample currently only supports 3D transformations"
        )


def resample3D(
    image,
    transform=None,
    output_spacing=None,
    output_direction=None,
    output_origin=None,
    output_size=None,
    interpolator=sitk.sitkLinear,
):
    """
    Resampler for 3D sitk images, with the option to apply a transform

    Parameters
    ----------
    image : SITK image
        image to transform.
    transform : SITK Affine Transform, optional
        If no transform is passed, use a identity transform matrix
    output_spacing : (3x1) array, optional
        If not passed, coppies from image
    output_direction : (9x1) array, optional
        If not passed, coppies from image
    output_origin : (3x1) array, optional
        If not passed, coppies from image
    output_size : (3x1) array, optional
        If not passed, computes automatically to fully encompus
        transformed image.

    Returns
    -------
    resampled_image : SITK image
        resampled image with transform applied.

    """
    if transform is None:
        transform = sitk.AffineTransform(3)

    extrema = image.GetSize()
    extreme_points = [
        image.TransformIndexToPhysicalPoint((0, 0, 0)),
        image.TransformIndexToPhysicalPoint((extrema[0] + 1, 0, 0)),
        image.TransformIndexToPhysicalPoint((0, extrema[1] + 1, 0)),
        image.TransformIndexToPhysicalPoint((0, 0, extrema[2] + 1)),
        image.TransformIndexToPhysicalPoint(
            (extrema[0] + 1, extrema[1] + 1, 0)
        ),
        image.TransformIndexToPhysicalPoint(
            (extrema[0] + 1, 0, extrema[2] + 1)
        ),
        image.TransformIndexToPhysicalPoint(
            (0, extrema[1] + 1, extrema[2] + 1)
        ),
        image.TransformIndexToPhysicalPoint(
            (extrema[0] + 1, extrema[1] + 1, extrema[2] + 1)
        ),
    ]

    inv_transform = transform.GetInverse()

    extreme_points_transformed = [
        inv_transform.TransformPoint(pnt) for pnt in extreme_points
    ]

    min_x = min(extreme_points_transformed, key=lambda p: p[0])[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
    max_x = max(extreme_points_transformed, key=lambda p: p[0])[0]
    max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
    max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

    #
    if output_spacing is None:
        output_spacing = image.GetSpacing()

    if output_direction is None:
        output_direction = image.GetDirection()

    if output_origin is None:
        output_origin = [0, 0, 0]
        if output_direction[0] > 0:
            output_origin[0] = min_x
        else:
            output_origin[0] = max_x

        if output_direction[4] > 0:
            output_origin[1] = min_y
        else:
            output_origin[1] = max_y

        if output_direction[8] > 0:
            output_origin[2] = min_z
        else:
            output_origin[2] = max_z

    # Compute grid size based on the physical size and spacing.
    if output_size is None:
        output_size = [
            int((max_x - min_x) / output_spacing[0]),
            int((max_y - min_y) / output_spacing[1]),
            int((max_z - min_z) / output_spacing[2]),
        ]

    resampled_image = sitk.Resample(
        image,
        output_size,
        transform,
        interpolator,
        output_origin,
        output_spacing,
        output_direction,
    )
    return resampled_image
