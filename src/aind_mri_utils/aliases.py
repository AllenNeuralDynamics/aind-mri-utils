"""Aliases for functions. This module is not guaranteed to be stable."""

from aind_mri_utils.rotations import (
    create_homogeneous_from_euler_and_translation,
    prepare_data_for_homogeneous_transform,
)

append_ones_columns = prepare_data_for_homogeneous_transform
create_rigid_transform = create_homogeneous_from_euler_and_translation