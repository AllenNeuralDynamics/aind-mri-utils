# -*- coding: utf-8 -*-
"""
Functions for Loading and manipulating meshes during insertion planning.
"""

import logging
import warnings

import numpy as np
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils.coordinate_systems import (
    convert_coordinate_system,
)
from aind_anatomical_utils.sitk_volume import (
    transform_sitk_indices_to_physical_points,
)
from skimage.measure import marching_cubes

from aind_mri_utils.rotations import make_homogeneous_transform

logger = logging.getLogger(__name__)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.

    see https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_newscale_trimesh(
    filename,
    move_down=0,
    src_coordinate_system="LSA",
    dst_coordinate_system="LPS",
):
    """Load a newscale model mesh.

    Parameters
    ----------
    filename : str
        Path to the mesh file.
    move_down : float, optional
        Amount to move the mesh down along the z-axis, by default 0.
        DEPRECATED: use apply_transform_to_trimesh instead.
    src_coordinate_system : str, optional
        Source coordinate system, by default "LSA".
    dst_coordinate_system : str, optional
        Destination coordinate system, by default "LPS".

    Returns
    -------
    trimesh.Trimesh
        The loaded and transformed mesh.
    """
    mesh = trimesh.load_mesh(filename)
    mesh = as_mesh(mesh)
    new_vertices = convert_coordinate_system(
        mesh.vertices, src_coordinate_system, dst_coordinate_system
    )
    if move_down != 0:
        warnings.warn(
            "move_down is deprecated, use apply_transform_to_trimesh instead",
            category=DeprecationWarning,
        )
        new_vertices[:, 2] = new_vertices[:, 2] - move_down
    mesh.vertices = new_vertices
    # Repair broken stuff from bad blender-ing
    trimesh.repair.broken_faces(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    return mesh


def apply_transform_to_trimesh(mesh, R, translation=None):
    """
    Apply a transform to a trimesh Mesh object
    """
    if translation is None:
        translation = np.zeros(3)
    T = make_homogeneous_transform(R, translation)
    mesh.vertices = trimesh.transform_points(mesh.vertices, T)
    return mesh


def create_uv_spheres(positions, radius=0.25, color=[255, 0, 255, 255]):
    """
    Create UV spheres at specified positions with a given radius and color.

    Parameters
    ----------
    positions : ndarray
        Array of positions where spheres should be created.
    radius : float, optional
        Radius of the spheres, by default 0.25.
    color : list, optional
        RGBA color of the spheres, by default [255, 0, 255, 255].

    Returns
    -------
    list
        List of trimesh objects representing the spheres.
    """
    meshes = [
        trimesh.creation.uv_sphere(radius=radius)
        for _ in range(len(positions))
    ]
    for i, mesh in enumerate(meshes):
        mesh.apply_translation(positions[i, :])
        mesh.visual.vertex_colors = color
    return meshes


def distances_to_triangle(points, triangle):
    """
    Calculate the distance from a set of points to a triangle.

    Parameters
    ----------
    points : array-like, shape (n, 3)
        The points from which the distance to the triangle is calculated. Each
        row represents a point with x, y, z coordinates.
    triangle : array-like, shape (3, 3)
        The vertices of the triangle. Each row represents a vertex with x, y, z
        coordinates.

    Returns
    -------
    distances : numpy.ndarray, shape (n,)
        The distances from each point to the triangle.
    nearest_points : numpy.ndarray, shape (n, 3)
        The nearest points on the triangle for each input point.
    """
    tri_mesh = trimesh.Trimesh(vertices=triangle, faces=[[0, 1, 2]])
    nearest_points, distances, _ = trimesh.proximity.closest_point(
        tri_mesh, points
    )
    return distances, nearest_points


def distance_to_all_triangles_in_mesh(mesh, points, normalize=True):
    """Calculate the distances from points to all triangles in a mesh.

    This function computes the minimum distance between a set of points and
    each triangle in a mesh, along with the nearest points on the triangles.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A triangular mesh object containing the triangles to measure distance
        from
    points : numpy.ndarray
        Array of points to calculate distances from, shape (N, 3) where N is
        number of points
    normalize : boolean, optional
        Whether to normalize distances by number of points, by default True

    Returns
    -------
    distances : numpy.ndarray
        Array of distances from points to each triangle in the mesh
        If normalize=True, distances are divided by number of points
    nearest_points : list
        List of arrays containing the nearest points on each triangle for each
        input point

    See Also
    --------
    distance_to_triangle : Function that calculates distance from points to a
    single triangle
    """
    distances = []
    nearest_points = []
    for triangle in mesh.triangles:
        this_distance, this_nearest_points = distances_to_triangle(
            points, triangle
        )
        distances.append(this_distance)
        nearest_points.append(this_nearest_points)
    distances = np.array(distances)
    if normalize:
        distances = distances / len(points)
    return distances, nearest_points


def distance_to_closest_point_for_each_triangle_in_mesh(
    mesh, points, normalize=True
):
    """
    Calculate the distance to the closest point for each triangle in a mesh.

    Parameters
    ----------
    mesh : object
        A mesh object that contains triangles.
    points : array-like
        An array of points to calculate the distance from.
    normalize : boolean, optional
        If set to True, the distances will be normalized by the number of
        triangles (default is True).

    Returns
    -------
    distances : numpy.ndarray
        An array of the minimum distances from each triangle to the closest
        point.
    nearest_points : list of array-like
        A list of the nearest points corresponding to each triangle.

    Notes
    -----
    This function assumes that the `mesh` object has an attribute `triangles`
    which is an array of triangles.  The `distance_to_triangle` function is
    used to calculate the distance from a point to a triangle.
    """
    triangles = mesh.triangles
    distances = []
    nearest_points = []
    for triangle in triangles:
        distances_to_tri, nearest_points_tri = distances_to_triangle(
            points, triangle
        )
        min_ndx = np.argmin(distances_to_tri)
        distances.append(distances_to_tri[min_ndx])
        nearest_points.append(nearest_points_tri[min_ndx, :])
    distances = np.array(distances)
    if normalize:
        distances = distances / len(triangles)
    return distances, nearest_points


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
        logger.warning(
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
    ndxs, faces, normals, _ = marching_cubes(mask_array, level=level)
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
