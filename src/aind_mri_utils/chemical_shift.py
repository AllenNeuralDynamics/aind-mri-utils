"""
Functions for correcting for chemical shift in MRI images
"""

import numpy as np

from . import utils as ut
from . import rotations as rot

def compute_chemical_shift(
    image, ppm=(3.7 + 4.1) / 2, mag_freq=599, pixel_bandwidth=500
):
    """
    This function calculates the chemical shift for an MRI image using the
    provided parameters. The chemical shift is calculated based on the parts
    per million (ppm), magnetic frequency of the scanner (mag_freq), and
    per-pixel bandwidth of the image.

    Parameters
    -----------
    image (SimpleITK.Image): The input image, which is an instance of a
        SimpleITK.Image object.  ppm (float, optional): The parts per million
        value for the chemical shift calculation.  Default is the average of
        3.7 and 4.1.
    mag_freq (float, optional): The magnetic frequency in MHz. Default is 599.
        pixel_bandwidth (float, optional): The pixel bandwidth in Hz. Default
        is 500.

    Returns:
    float: The computed chemical shift value.
    """
    shift = (ppm * mag_freq) / pixel_bandwidth * image.GetSpacing()[1]
    return shift


def chemical_shift_transform(shift, readout="HF"):
    """
    Creates a transformation matrix that accounts for the chemical
    shift in MRI images. The direction of the readout (either head-foot (HF) or
    left-right (LR)) determines the configuration of the transformation matrix.

    Parameters:
    shift (float): The chemical shift value to be applied.
    readout (str, optional): The direction of the readout. It can be either
        'HF' for head-foot direction or 'LR' for left-right direction.  Default
        is 'HF'.

    Returns
    -------
    np.ndarray: A homogeneous transformation matrix that incorporates the
    chemical shift.

    Raises:
    ValueError: If the readout direction is not recognized.
    """
    if readout == "HF":
        return rot.make_homogeneous_transform(
            np.eye(3), np.array([0, shift, 0])
        )
    elif readout == "LR":
        return rot.make_homogeneous_transform(
            np.eye(3), np.array([shift, 0, 0])
        )
    else:
        raise ValueError("Readout direction not recognized")