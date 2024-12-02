"""Module to fit implant rotations to MRI data."""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import fmin
from scipy.spatial.transform import Rotation

from aind_mri_utils.file_io.slicer_files import get_segmented_labels
from aind_mri_utils.meshes import (
    distance_to_all_triangles_in_mesh,
    distance_to_closest_point_for_each_triangle_in_mesh,
)
from aind_mri_utils.rotations import (
    apply_rotate_translate,
    combine_angles,
    rotation_matrix_from_vectors,
)
from aind_mri_utils.sitk_volume import find_points_equal_to


def _implant_cost_fun(T, hole_mesh_dict, hole_seg_dict, run_parallel=True):
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
    if run_parallel:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(func, *args) for func, args in tasks]
            for future in as_completed(futures):
                distances, _ = future.result()
                total_distance += np.sum(distances)
    else:
        for func, args in tasks:
            distances, _ = func(*args)
            total_distance += np.sum(distances)
    return total_distance


def fit_implant_to_mri(
    hole_seg_dict,
    hole_mesh_dict,
    initialization_hole=4,
    other_init_holes=[3, 9],
):
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
    other_init_holes : int, optional
        The hole to use for initialization of the rotation, by default [3, 9].


    Returns
    -------
    output : ndarray
        The optimized transformation parameters that align the implant model to
        the MRI data.
    """
    T = np.zeros(6)
    initialize_translation = initialization_hole in hole_seg_dict
    if initialize_translation:
        annotation_mean = np.mean(hole_seg_dict[initialization_hole], axis=0)
        model_mean = np.mean(
            hole_mesh_dict[initialization_hole].vertices, axis=0
        )
        init_offset = model_mean - annotation_mean
        T[3:] = init_offset
    else:
        warnings.warn(
            f"Could not find hole {initialization_hole} in MRI data "
            "for initialization"
        )

    # Initial guess of rotation
    initialize_rotation = initialize_translation and len(other_init_holes) >= 2
    for other_hole in other_init_holes:
        initialize_rotation = (
            initialize_rotation
            and other_hole in hole_seg_dict
            and other_hole != initialization_hole
        )
        if not initialize_rotation:
            break

    if initialize_rotation:
        # Initialize rotation based on pairs of holes
        #
        # Find the rotation that aligns the vector between the first two holes
        # in the annotation to the vector between the first two holes in the
        # model. Refine with subsequent pairs of holes.
        R_init = np.eye(3)
        for other_hole in other_init_holes:
            other_anno_mean = np.mean(hole_seg_dict[other_hole], axis=0)
            other_model_mean = np.mean(
                hole_mesh_dict[other_hole].vertices, axis=0
            )
            anno_diff_rotated = R_init @ (other_anno_mean - annotation_mean)
            R_update = rotation_matrix_from_vectors(
                anno_diff_rotated,
                other_model_mean - model_mean,
            )
            R_init = R_update @ R_init
        T[:3] = Rotation.from_matrix(R_init).as_euler("xyz")
    else:
        warnings.warn("Could not find holes for rotation initialization")

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
