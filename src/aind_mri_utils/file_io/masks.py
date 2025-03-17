"""
Note that this code was primarily written by ChatGPT

The docstrings were created using CoPilot

Code was tested and modified by Yoni
"""

import numpy as np
import SimpleITK as sitk
import trimesh
from pymeshfix import MeshFix
import skimage.measure


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
    mask_array = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()
    origin = mask.GetOrigin()
    direction = np.array(mask.GetDirection()).reshape(3, 3)
    return mask_array, spacing, origin, direction


# Generate a surface mesh from the mask
def generate_mesh_from_mask(mask_array, spacing):
    """
    Generate a surface mesh from the mask.

    Parameters
    ----------
    mask_array : numpy.ndarray
        Binary mask.
    spacing : tuple of float
        Voxel spacing.

    Returns
    -------
    numpy.ndarray
        Vertex coordinates.
    numpy.ndarray
        Face indices.
    """

    # Use marching cubes to extract the surface
    from skimage import measure

    vertices, faces, _, _ = measure.marching_cubes(
        mask_array, level=0.5, spacing=spacing
    )
    return vertices, faces


# Create a trimesh object
def create_trimesh(vertices, faces):
    """
    Create a trimesh object from vertices and faces.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex coordinates.
    faces : numpy.ndarray
        Face indices.

    Returns
    -------
    trimesh.base.Trimesh
        A trimesh object representing the surface mesh.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


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
    else:
        mesh.fix_normals()
    return mesh


# Smooth the mesh using Laplacian smoothing
def smooth_mesh(mesh, iterations=10, lambda_param=0.5):
    """
    Smooth the mesh using Laplacian smoothing.

    Parameters
    ----------
    mesh : trimesh.base.Trimesh
        Input mesh.
    iterations : int, optional
        Number of iterations. The default is 10.
    lambda_param : float, option
        Number of iterations. The default is 10.

    Returns
    -------
    trimesh.base.Trimesh
        Smoothed mesh.
    """
    from trimesh.smoothing import filter_laplacian

    smooth_mesh = mesh.copy()
    filter_laplacian(smooth_mesh, lamb=lambda_param, iterations=iterations)
    return smooth_mesh


def repair_mesh(mesh, verbose=True):
    """
    repair_mesh(mesh)

    Check if a mesh is watertight and repair it if necessary.

    Parameters
    ----------
    mesh : trimesh.base.Trimesh
        Input mesh.
    verbose : bool, optional
        If True, print messages. The default is True.

    Returns
    -------
    trimesh.base.Trimesh
        Repaired mesh.
    """

    # Check if the mesh is watertight
    if mesh.is_watertight and verbose:
        print("The mesh is already watertight.")
    else:
        if verbose:
            print("The mesh is not watertight. Proceeding with repair...")

        # Use PyMeshFix to repair the mesh
        meshfix = MeshFix(mesh.vertices, mesh.faces)
        meshfix.repair(
            verbose=True, joincomp=True, remove_smallest_components=True
        )

        # Create a new trimesh object from the repaired mesh
        mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)

        if verbose:
            if mesh.is_watertight:
                print("Mesh successfully repaired.")
            else:
                print(
                    "Mesh repair failed. The mesh may still not be watertight."
                )
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
    verts, faces, normals, _ = skimage.measure.marching_cubes(
        mask_array, level=level
    )

    # Convert voxel indices to physical coordinates
    spacing = np.array(sitk_mask.GetSpacing())  # (X, Y, Z)
    origin = np.array(sitk_mask.GetOrigin())  # (X, Y, Z)
    direction = np.array(sitk_mask.GetDirection()).reshape(3, 3)  # 3x3 matrix

    # Convert voxel indices to physical space
    verts = verts[:, [2, 1, 0]]  # Convert (Z, Y, X) -> (X, Y, Z)
    verts = verts * spacing  # Scale by spacing
    verts = (
        np.dot(direction, verts.T).T + origin
    )  # Apply direction and shift by origin

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    if smooth_iters > 0:
        mesh = trimesh.smoothing.filter_mut_dif_laplacian(
            mesh, iterations=smooth_iters
        )

    return mesh
