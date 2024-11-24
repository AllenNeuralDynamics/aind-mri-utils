"""Aliases for functions. This module is not guaranteed to be stable."""

from aind_mri_utils.file_io import simpleitk as sitk_io
from aind_mri_utils.rotations import (
    create_homogeneous_from_euler_and_translation,
    prepare_data_for_homogeneous_transform,
)

append_ones_columns = prepare_data_for_homogeneous_transform
create_rigid_transform = create_homogeneous_from_euler_and_translation


def save_sitk_transform(filename, T, transpose_matrix=False):
    """
    This is an alias for `sitk_io.save_sitk_transform` that has the same
    interface as the original function that Yoni wrote.
    """

    sitk_io.save_sitk_transform(
        filename,
        rotation_matrix=T,
        transpose_matrix=transpose_matrix,
        legacy=True,
    )
