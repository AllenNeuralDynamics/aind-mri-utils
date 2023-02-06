""" Module to deal with coordinate systems
"""

import numpy as np


def lps_to_ras(arr: np.ndarray) -> np.ndarray:
    """
    Convert from LPS to RAS coordinate systems.

    There are two common coordinate systems used in medical imaging, LPS
    (Left, Posterior, Superior) and RAS (Right, Anterior, Superior), where each
    axis increases in the named direction (i.e. Left means the axis increases
    as you travel to the subject's left). This function converts between the
    two by flipping the first two dimensions. This is equivalent to taking the
    dot product between the input and `np.array([-1, -1, 1])`.

    Parameters
    ==========
    arr - numpy.ndarray (N x 3) of N LPS coordinates

    Returns
    =======
    out - numpy.ndarray (N x 3) of N input coordinates transformed into RAS
    """
    out = arr * np.array([-1, -1, 1])
    return out


def ras_to_lps(arr: np.ndarray) -> np.ndarray:
    """
    Convert from RAS to LPS coordinate systems.

    See `lps_to_ras` for more information.

    Parameters
    ==========
    arr - numpy.ndarray (N x 3) of N RAS coordinates

    Returns
    =======
    out - numpy.ndarray (N x 3) of N input coordinates transformed into LPS
    """
    return lps_to_ras(arr)
