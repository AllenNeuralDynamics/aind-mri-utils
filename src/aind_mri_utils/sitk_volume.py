"""
Code to handle sitk volume loading and rotating

SimpleITK example code is under Apache License, see:
https://github.com/SimpleITK/TUTORIAL/blob/main/LICENSE

"""
import SimpleITK as sitk


def resample(
    image,
    transform=None,
    output_spacing=None,
    output_direction=None,
    output_origin=None,
    output_size=None,
    interpolator=sitk.sitkLinear,
):  # pragma: no cover
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
    output_direction : (N^2x1) array, optional
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
):  # pragma: no cover
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
    output_direction : (9x1) array, optional
        If not passed, copies from image
    output_origin : (3x1) array, optional
        If not passed, coppies from image
    output_size : (3x1) array, optional
        If not passed, computes automatically to fully encompus
        transformed image.

    Returns
    -------
    resampled_image : SimpleITK image
        resampled image with transform applied.

    """
    if transform is None:
        transform = sitk.AffineTransform(3)

    im_size = image.GetSize()
    extrema = [
        image.TransformIndexToPhysicalPoint((0, 0, 0)),
        image.TransformIndexToPhysicalPoint((im_size[0], 0, 0)),
        image.TransformIndexToPhysicalPoint((0, im_size[1], 0)),
        image.TransformIndexToPhysicalPoint((0, 0, im_size[2])),
        image.TransformIndexToPhysicalPoint((im_size[0], im_size[1], 0)),
        image.TransformIndexToPhysicalPoint((im_size[0], 0, im_size[2])),
        image.TransformIndexToPhysicalPoint((0, im_size[1], im_size[2])),
        image.TransformIndexToPhysicalPoint(
            (im_size[0], im_size[1], im_size[2])
        ),
    ]

    inv_transform = transform.GetInverse()

    extrema_transformed = [
        inv_transform.TransformPoint(pnt) for pnt in extrema
    ]

    min_x = min(extrema_transformed, key=lambda p: p[0])[0]
    min_y = min(extrema_transformed, key=lambda p: p[1])[1]
    min_z = min(extrema_transformed, key=lambda p: p[2])[2]
    max_x = max(extrema_transformed, key=lambda p: p[0])[0]
    max_y = max(extrema_transformed, key=lambda p: p[1])[1]
    max_z = max(extrema_transformed, key=lambda p: p[2])[2]

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
