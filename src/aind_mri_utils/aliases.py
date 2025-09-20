"""Aliases for functions. This module is not guaranteed to be stable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from aind_mri_utils.file_io import simpleitk as sitk_io
from aind_mri_utils.rotations import (
    create_homogeneous_from_euler_and_translation,
    prepare_data_for_homogeneous_transform,
)

append_ones_columns = prepare_data_for_homogeneous_transform
create_rigid_transform = create_homogeneous_from_euler_and_translation


def save_sitk_transform(
    filename: str, T: NDArray[np.floating[Any]], transpose_matrix: bool = False
) -> None:
    """
    This is an alias for `sitk_io.save_sitk_transform` that has the same
    interface as the original function that Yoni wrote.
    """

    sitk_io.save_sitk_transform(
        filename,
        rotation_matrix=T,
    )
