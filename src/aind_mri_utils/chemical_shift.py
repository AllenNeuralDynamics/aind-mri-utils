"""
Functions for correcting for chemical shift in MRI images
"""

import numpy as np


def compute_chemical_shift(
    image_or_spacing,
    ppm=(3.7 + 4.1) / 2,
    mag_freq=599.0,
    pixel_bandwidth=500.0,
    frequency_encoding_direction="AP",
):
    """Calculate the chemical shift for an MRI image or spacing.

    The chemical shift is calculated based on the parts per million (ppm),
    magnetic frequency of the scanner (mag_freq), per-pixel bandwidth
    of the image, and the frequency encoding direction. The function can
    accept either a SimpleITK.Image object or a voxel spacing value for the
    frequency-encoding direction.

    Parameters
    ----------
    image_or_spacing : SimpleITK.Image or float
        The input image (SimpleITK.Image) or a spacing float representing voxel
        dimension in the frequency encoding direction.
    ppm : float, optional
        The parts per million value for the chemical shift calculation.
        Default is the average of 3.7 and 4.1.
    mag_freq : float, optional
        The magnetic frequency in MHz. Default is 599.
    pixel_bandwidth : float, optional
        The pixel bandwidth in Hz. Default is 500.
    frequency_encoding_direction : str, optional
        The direction of frequency encoding. It can be 'AP'
        (anterior-posterior), 'ML' (medial-lateral), or 'IS'
        (inferior-superior). Default is 'AP'.  Only needed if image_or_spacing
        is a SimpleITK.Image.

    Returns
    -------
    float
        The computed chemical shift value.

    Raises
    ------
    ValueError
        If the frequency encoding direction is invalid.
    """
    # Check if image_or_spacing is a simpleitk image:
    if hasattr(image_or_spacing, "GetSpacing"):
        # If it is, use the spacing from the image
        spacing_tuple = image_or_spacing.GetSpacing()
        dir_mat = np.array(image_or_spacing.GetDirection()).reshape(3, 3)
        readout_axes = {
            "ML": np.array([1, 0, 0]),
            "AP": np.array([0, 1, 0]),
            "IS": np.array([0, 0, 1]),
        }
        direction = readout_axes.get(frequency_encoding_direction, None)
        if direction is None:
            raise ValueError(
                "Invalid frequency encoding direction "
                f"{frequency_encoding_direction}"
            )
        dot_products = np.abs(dir_mat @ direction)
        index_axis = np.argmax(dot_products)
        spacing = spacing_tuple[index_axis]
    else:
        # If it is not, assume it is a spacing tuple
        spacing = image_or_spacing
    shift = spacing * (ppm * mag_freq) / pixel_bandwidth
    return shift


def chemical_shift_transform(shift, readout="AP"):
    """Create chemical shift transformation matrix.

    Creates a transformation matrix that accounts for the chemical
    shift in MRI images. The direction of the readout (either
    anterior-posterior (AP) or left-right (LR)) determines the configuration of
    the transformation matrix.

    Parameters
    ----------
    shift : float
        The chemical shift value to be applied.
    readout : str, optional
        The direction of the readout. It can be either 'AP' for
        anterior-posterior direction or 'LR' for left-right direction.  Default
        is 'AP'.

    Returns
    -------
    R : np.ndarray
        A 3x3 rotation matrix.
    translation : np.ndarray
        A 3-element translation

    Raises
    ------
    ValueError
        If the readout direction is not recognized.
    """
    # raise deprecation warning if readout direction is HF
    if readout == "HF":
        readout = "AP"
        raise (
            DeprecationWarning(
                "HF readout direction is deprecated, using AP instead."
            )
        )

    if readout == "AP":
        translation = np.array([0, shift, 0])
    elif readout == "LR":
        translation = np.array([shift, 0, 0])
    elif readout == "IS":
        translation = np.array([0, 0, shift])
    else:
        raise ValueError("Readout direction not recognized")
    R = np.eye(3)
    return R, translation
