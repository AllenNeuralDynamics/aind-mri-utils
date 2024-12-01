"""Module to fit implant rotations to MRI data."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import fmin

from aind_mri_utils.file_io.slicer_files import get_segmented_labels
from aind_mri_utils.meshes import (
    distance_to_all_triangles_in_mesh,
    distance_to_closest_point_for_each_triangle_in_mesh,
)
from aind_mri_utils.rotations import apply_rotate_translate, combine_angles
from aind_mri_utils.sitk_volume import find_points_equal_to


def _implant_cost_fun(T, hole_mesh_dict, hole_seg_dict):
    """
    Computes the total distance cost for implant alignment based on the
    provided transformation parameters.

    Parameters
    ----------
    T : array-like
        Transformation parameters including Euler angles and translation
        vector.
    hole_mesh_dict : dict
        Dictionary where keys are hole IDs and values are mesh objects
        representing the holes.
    hole_seg_dict : dict
        Dictionary where keys are hole IDs and values are segmented points
        corresponding to the holes.

    Returns
    -------
    float
        The total distance cost calculated by summing the distances between
        transformed points and mesh triangles.
    """
    rotation_matrix = combine_angles(*T[:3])
    translation = T[3:]
    tasks = []
    for hole_id in hole_mesh_dict.keys():
        # TODO: Fix this so it actually works for the brain outline
        if hole_id not in hole_seg_dict:
            continue
        mesh = hole_mesh_dict[hole_id]
        pts = hole_seg_dict[hole_id]
        transformed_pts = apply_rotate_translate(
            pts, rotation_matrix, translation
        )
        args = (mesh, transformed_pts)
        if hole_id == -1:
            func = distance_to_closest_point_for_each_triangle_in_mesh
        else:
            func = distance_to_all_triangles_in_mesh
        tasks.append((func, args))
    total_distance = 0.0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, *args) for func, args in tasks]
        for future in as_completed(futures):
            distances, _ = future.result()
            total_distance += np.sum(distances)
    return total_distance


def fit_implant_to_mri(hole_seg_dict, hole_mesh_dict, initialization_hole=4):
    """
    Fits an implant model to MRI data by optimizing the alignment of hole
    segments.

    Parameters
    ----------
    hole_seg_dict : dict
        Dictionary containing segmented hole data from MRI. Keys are hole
        identifiers, and values are numpy arrays of coordinates.
    hole_mesh_dict : dict
        Dictionary containing mesh data for the implant model. Keys are hole
        identifiers, and values are mesh objects with vertex coordinates. Lower
        face has key -1.
    initialization_hole : int, optional
        The hole to use for initialization, by default 4.

    Returns
    -------
    output : ndarray
        The optimized transformation parameters that align the implant model to
        the MRI data.
    """
    annotation_mean = np.mean(hole_seg_dict[initialization_hole], axis=0)
    model_mean = np.mean(hole_mesh_dict[initialization_hole].vertices, axis=0)
    init_offset = model_mean - annotation_mean
    T = [0, 0, 0, init_offset[0], init_offset[1], init_offset[2]]

    output = fmin(
        _implant_cost_fun,
        T,
        args=(hole_mesh_dict, hole_seg_dict),
        xtol=1e-6,
        maxiter=2000,
    )
    return output


def make_hole_seg_dict(implant_annotations):
    """
    Creates a dictionary mapping hole names to their segmented positions.

    Parameters
    ----------
    implant_annotations : numpy.ndarray
        An array containing the implant annotations.

    Returns
    -------
    dict
        A dictionary where the keys are hole names (as integers) and the values
        are lists of positions where the segmented values are found.
    """
    # TODO: Fix this so it actually works for the brain outline
    implant_annotations_names = get_segmented_labels(implant_annotations)
    hole_seg_dict = {}
    for hole_name, seg_val in implant_annotations_names.items():
        positions = find_points_equal_to(implant_annotations, seg_val)
        hole_seg_dict[int(hole_name)] = positions
    return hole_seg_dict
