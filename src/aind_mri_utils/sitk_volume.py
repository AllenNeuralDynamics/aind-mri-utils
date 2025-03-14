"""
Code to handle sitk volume loading and rotating

SimpleITK example code is under Apache License, see:
https://github.com/SimpleITK/TUTORIAL/blob/main/LICENSE

"""

import itertools as itr

import numpy as np
import SimpleITK as sitk


def resample(
    image,
    transform=None,
    output_spacing=None,
    output_origin=None,
    output_size=None,
    interpolator=sitk.sitkLinear,
):
    """Resample a SimpleITK image with an optional transform.

    Wrapper to generically handle SimpleITK resampling on different image
    matrices. Includes optional application of a transform.  Only 3d is
    currently implemented.

    Code is modified from the 2d example in.
    https://simpleitk.org/SPIE2018_COURSE/images_and_resampling.pdf and
    https://github.com/SimpleITK/TUTORIAL/blob/main/02_images_and_resampling.ipynb

    Parameters
    ----------
    image : SimpleITK image
        image to transform.
    transform : SimpleITK Affine Transform, optional
        If no transform is passed, use a identity transform matrix
    output_spacing : (Nx1) array, optional
        If not passed, copies from image
    output_origin : (Nx1) array, optional
        If not passed, copies from image
    output_size : (Nx1) array, optional
        If not passed, computes automatically to fully encompass
        transformed image.
    interpolator: SimpleITK Interpolator, optional
        If not passed, defaults to sitk.sitkLinear
        See sitk documentation for options.

    Returns
    -------
    resampled_image : SimpleITK image
        resampled image with transform applied.

    """
    if len(image.GetSize()) == 3:
        return resample3D(
            image,
            transform=transform,
            output_spacing=output_spacing,
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
    output_origin=None,
    output_size=None,
    interpolator=sitk.sitkLinear,
):
    """
    Resample a 3D sitk image, with the option to apply a transform

    Parameters
    ----------
    image : SimpleITK image
        image to transform.
    transform : SimpleITK Affine Transform, optional
        If no transform is passed, use a identity transform matrix
    output_spacing : (3x1) array, optional
        If not passed, copies from image
    output_origin : (3x1) array, optional
        If not passed, copies from image
    output_size : (3x1) array, optional
        If not passed, computes automatically to fully encompass
        transformed image.

    Returns
    -------
    resampled_image : SimpleITK image
        resampled image with transform applied.

    """
    if transform is None:
        transform = sitk.AffineTransform(3)

    inv_transform = transform.GetInverse()
    extrema_transformed = list(
        map(
            lambda x: inv_transform.TransformPoint(  # Apply inverse transform
                image.TransformIndexToPhysicalPoint(x)  # To the physical point
            ),
            itr.product(
                *map(lambda x: (0, x), image.GetSize())
            ),  # for all pairs of extreme indices
        )
    )

    extrema_arr = np.vstack(extrema_transformed)
    min_max = np.vstack(
        list(map(lambda x: x(extrema_arr, axis=0), [np.min, np.max]))
    )

    #
    if output_spacing is None:
        spacing = np.empty(3)
        spacing.fill(np.median(np.array(image.GetSpacing())))
        output_spacing = tuple(spacing)

    if output_origin is None:
        output_origin = min_max[0, :].tolist()

    # Compute grid size based on the physical size and spacing.
    if output_size is None:
        output_size = (
            np.round(np.diff(min_max, axis=0).squeeze() / spacing)
            .astype(int)
            .tolist()
        )

    resampled_image = sitk.Resample(
        image,
        output_size,
        transform,
        interpolator,
        output_origin,
        output_spacing,
        tuple(np.eye(3).flatten()),
    )
    return resampled_image


def transform_sitk_indices_to_physical_points(image, index_arr):
    """Transforms indices indices of image to physical points

    For a SimpleITK image `image` and a list of indices `index_arr`, transform
    each index to a physical point.

    Parameters
    ----------
    image : M-d SimpleITK image
    index_arr : numpy.ndarray (NxM)
        matrix of indices of `image`, where each row is an index

    Returns
    -------
    position_arr: numpy.ndarray (NxM)
        matrix of physical points for each index in `index_arr`
    """
    position_arr = np.zeros_like(index_arr, dtype="float32")
    npt = index_arr.shape[0]
    for pt_no in range(npt):
        ndx = tuple(map(lambda x: x.item(), index_arr[pt_no, :]))
        position_arr[pt_no, :] = image.TransformContinuousIndexToPhysicalPoint(
            ndx
        )
    return position_arr


def find_points_equal_to(image, label_value):
    """
    Get the physical positions of all voxels in the implant volume that match
    the given label value.

    Parameters
    ----------
    image: SimpleITK.Image
        The implant volume to query.
    label_value : int
        The label value to search for in the volume.

    Returns
    -------
    ndarray
        A NumPy array of physical positions corresponding to the label value.
    """
    implant_vol_arr = sitk.GetArrayViewFromImage(image)
    indices = np.nonzero(implant_vol_arr == label_value)

    if len(indices[0]) == 0:
        return np.empty((0, implant_vol_arr.ndim))

    positions = [
        image.TransformIndexToPhysicalPoint(tuple([int(x) for x in idx[::-1]]))
        for idx in zip(*indices)
    ]
    return np.vstack(positions)
