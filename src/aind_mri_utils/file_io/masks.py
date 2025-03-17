"""
Note that this code was primarily written by ChatGPT

The docstrings were created using CoPilot

Code was tested and modified by Yoni
"""

import numpy as np
import SimpleITK as sitk
import skimage.measure
import trimesh

from aind_mri_utils.sitk_volume import (
    transform_sitk_indices_to_physical_points,
)


def load_nrrd_mask(file_path):
    """
    Load a mask from a NRRD file.

    Parameters
    ----------
    file_path : str
        Path to the NRRD file.

    Returns
    -------
    numpy.ndarray
        Binary mask.
    tuple of float
        Voxel spacing.
    tuple of float
        Origin.
    numpy.ndarray
        Direction

    """
    mask = sitk.ReadImage(file_path)
    mask_array = sitk.GetArrayViewFromImage(mask)
    spacing = mask.GetSpacing()
    origin = mask.GetOrigin()
    direction = np.array(mask.GetDirection()).reshape(3, 3)
    return mask_array, spacing, origin, direction


# Ensure normals point outward
def ensure_normals_outward(mesh, verbose=True):
    """
    Ensure normals point outward.

    Parameters
    ----------
    mesh : trimesh.base.Trimesh
        Input mesh.

    Returns
    -------
    trimesh.base.Trimesh
        Mesh with outward-pointing normals.
    """
    if not mesh.is_watertight and verbose:
        print(
            "Warning: Mesh is not watertight. "
            "Normal orientation may not be reliable."
        )
    mesh.fix_normals()
    return mesh


def mask_to_trimesh(sitk_mask, level=0.5, smooth_iters=0):
    """
    Converts a SimpleITK binary mask into a 3D mesh in the same physical space.

    Parameters:
        sitk_mask (sitk.Image): A 3D SimpleITK binary mask image.
        level (float): The threshold value for the marching cubes algorithm.
        smooth_iters (int): Number of iterations for mesh smoothing. If zero,
            no smoothing is applied.

    Returns:
        trimesh.Trimesh:
            A 3D mesh in the same physical space as the input image.
    """
    # Get voxel data as a NumPy array
    mask_array = sitk.GetArrayFromImage(sitk_mask)  # Shape: (Z, Y, X)

    # Extract surface mesh using Marching Cubes
    ndxs, faces, normals, _ = skimage.measure.marching_cubes(
        mask_array, level=level
    )
    ndxs_sitk = ndxs[:, ::-1]

    # Convert voxel indices to physical space
    vertices = transform_sitk_indices_to_physical_points(sitk_mask, ndxs_sitk)

    # Create a trimesh object
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_normals=normals
    )

    if smooth_iters > 0:
        mesh = trimesh.smoothing.filter_mut_dif_laplacian(
            mesh, iterations=smooth_iters
        )

    return mesh
