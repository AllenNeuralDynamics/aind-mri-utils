# -*- coding: utf-8 -*-
"""
Functions for Loading and manipulating meshes during insertion planning.
"""

import trimesh


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
):
    """
    Load a newscale model mesh

    """
    mesh = trimesh.load_mesh(filename)
    mesh = as_mesh(mesh)
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]  # Should be made into params.
    mesh.vertices[:, 2] = mesh.vertices[:, 2] - move_down
    # Repair broken stuff from bad blender-ing
    trimesh.repair.broken_faces(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    return mesh


def apply_transform_to_trimesh(mesh, T):
    """
    Apply a transform to a trimesh Mesh object
    """
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
