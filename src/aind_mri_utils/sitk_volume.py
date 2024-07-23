"""
Code to handle sitk volume loading and rotating

SimpleITK example code is under Apache License, see:
https://github.com/SimpleITK/TUTORIAL/blob/main/LICENSE

"""

import itertools as itr

import numpy as np
import SimpleITK as sitk

from . import utils as ut


def resample(
    image,
    transform=None,
    output_spacing=None,
    output_origin=None,
    output_size=None,
    interpolator=sitk.sitkLinear,
):
    """
    Wrapper to generically handle SimpleITK resampling on different image
    matrices. Includes optional application of a transform.
    Only 3d is currently implemented.

    Code is modified from the 2d example in.
    https://simpleitk.org/SPIE2018_COURSE/images_and_resampling.pdf
    and
    https://github.com/SimpleITK/TUTORIAL/blob/main/...
        02_images_and_resampling.ipynb

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
    minmax = np.vstack(
        list(map(lambda x: x(extrema_arr, axis=0), [np.min, np.max]))
    )

    #
    if output_spacing is None:
        spacing = np.empty(3)
        spacing.fill(np.median(np.array(image.GetSpacing())))
        output_spacing = tuple(spacing)

    if output_origin is None:
        output_origin = minmax[0, :].tolist()

    # Compute grid size based on the physical size and spacing.
    if output_size is None:
        output_size = (
            np.round(np.diff(minmax, axis=0).squeeze() / spacing)
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


def transform_sitk_indices_to_physical_points(simage, index_arr):
    """Transforms indices indices of simage to physical points

    For a SimpleITK image `simage` and a list of indices `index_arr`, transform
    each index to a physical point.

    Parameters
    ----------
    simage : M-d SimpleITK image
    index_arr : numpy.ndarray (NxM)
        matrix of indices of `simage`, where each row is an index

    Returns
    -------
    position_arr: numpy.ndarray (NxM)
        matrix of physical points for each index in `index_arr`
    """
    position_arr = np.zeros_like(index_arr, dtype="float32")
    npt = index_arr.shape[0]
    for ptno in range(npt):
        ndx = tuple(map(lambda x: x.item(), index_arr[ptno, :]))
        position_arr[ptno, :] = simage.TransformContinuousIndexToPhysicalPoint(
            ndx
        )
    return position_arr


def find_points_equal_to(simage, val):
    """Find physical points in simage equal to val
    Results are in LPS
    """
    arr = sitk.GetArrayViewFromImage(simage)
    ndxs = ut.find_indices_equal_to(arr, val)
    if ndxs.size == 0:
        return np.empty((0, 3))
    ndxs_xyz = ndxs[
        :, [2, 1, 0]
    ]  # convert between numpy and simpleITK indexing
    return transform_sitk_indices_to_physical_points(simage, ndxs_xyz)
