"""
Note that this code was primarily written by ChatGPT

The docstrings were created using CoPilot

Code was tested and modified by Yoni
"""

import numpy as np
import SimpleITK as sitk
import trimesh
from pymeshfix import MeshFix


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


def mask_to_trimesh(
    nrrd_file_path, verbose=True, smoothing_iterations=10, smoothing_lambda=1
):
    """
    mask_to_trimesh(nrrd_file_path,smoothing_iterations  = 10, smoothing_lambda
    = 1)

    Load a mask from a NRRD file, generate a surface mesh, and export it as a
    trimesh object.

    Parameters
    ----------
    nrrd_file_path : str
        Path to the NRRD file.
    smoothing_iterations : int, optional
        Number of iterations for Laplacian smoothing. The default is 10.
    smoothing_lambda : float, optional
        Lambda parameter for Laplacian smoothing. The default is 1.

    Returns
    -------
    trimesh.base.Trimesh
        A trimesh object representing the surface mesh.
    """
    # Load mask
    mask_array, spacing, origin, direction = load_nrrd_mask(nrrd_file_path)

    # Generate mesh
    vertices, faces = generate_mesh_from_mask(mask_array, spacing)

    # Transform vertices to physical space
    vertices = np.dot(direction, vertices.T).T + origin

    # Create a trimesh object
    mesh = create_trimesh(vertices, faces)

    # smooth the mesh
    mesh = smooth_mesh(
        mesh, iterations=smoothing_iterations, lambda_param=smoothing_lambda
    )

    # Repair the mesh
    mesh = repair_mesh(mesh, verbose=verbose)

    # Clean up the normals
    mesh = ensure_normals_outward(mesh, verbose=verbose)

    # Export to a file
    return mesh
