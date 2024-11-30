"""Module to fit implant rotations to MRI data."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy.optimize import fmin

from aind_mri_utils import coordinate_systems as cs
from aind_mri_utils.file_io.slicer_files import get_segmented_labels
from aind_mri_utils.meshes import (
    distance_to_all_triangles_in_mesh,
    distance_to_closest_point_for_each_triangle_in_mesh,
)
from aind_mri_utils.rotations import (
    create_homogeneous_from_euler_and_translation,
    prepare_data_for_homogeneous_transform,
)
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
    trans = create_homogeneous_from_euler_and_translation(*T)

    tasks = []
    for hole_id in hole_mesh_dict.keys():
        if hole_id not in hole_seg_dict.keys():
            continue
        if hole_id != -1:
            this_hole_mesh = hole_mesh_dict[hole_id]
            these_hole_pts = hole_seg_dict[hole_id]

            transformed_hole_pts = np.dot(
                prepare_data_for_homogeneous_transform(these_hole_pts), trans
            )
            func = distance_to_all_triangles_in_mesh
            args = (this_hole_mesh, transformed_hole_pts)
            tasks.append((func, args))
        elif hole_id == -1:
            lower_mesh = hole_mesh_dict[hole_id]
            brain_outline = hole_seg_dict[hole_id]
            transformed_brain_outline = np.dot(
                prepare_data_for_homogeneous_transform(brain_outline), trans
            )
            func = distance_to_closest_point_for_each_triangle_in_mesh
            args = (lower_mesh, transformed_brain_outline)
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
        identifiers, and values are mesh objects with vertex coordinates.
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


def make_hole_mesh_dict(hole_files, lower_face_file):
    """
    Creates a dictionary of hole meshes and a lower face mesh.

    Parameters
    ----------
    hole_files : list of str
        List of file paths to the hole mesh files.
    lower_face_file : str
        File path to the lower face mesh file.

    Returns
    -------
    dict
        A dictionary where the keys are hole numbers (int) and the values are
        the corresponding trimesh objects. The lower face mesh is stored with
        the key -1.
    """
    hole_mesh_dict = {}
    for file_name in hole_files:
        file_stem = Path(file_name).stem
        hole_num = int(file_stem.split("Hole")[-1])
        mesh = trimesh.load(file_name)
        mesh.vertices = cs.convert_coordinate_system(
            mesh.vertices, "ASR", "LPS"
        )
        hole_mesh_dict[hole_num] = mesh

    # Get the lower face, store with key -1
    hole_mesh_dict[-1] = trimesh.load(lower_face_file)
    hole_mesh_dict[-1].vertices = cs.convert_coordinate_system(
        hole_mesh_dict[-1].vertices, "ASR", "LPS"
    )  # Preserves shape!

    return hole_mesh_dict


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
    implant_annotations_names = get_segmented_labels(implant_annotations)
    hole_seg_dict = {}
    for hole_name, seg_val in implant_annotations_names.items():
        positions = find_points_equal_to(implant_annotations, seg_val)
        hole_seg_dict[int(hole_name)] = positions
    return hole_seg_dict


def fit_implant_to_mri_from_files(
    implant_annotations_file, hole_files, lower_face_file
):
    """
    Fits an implant to MRI data using provided files.

    Parameters
    ----------
    implant_annotations_file : str or Path
        Path to the file containing implant annotations.
    hole_files : list of str or list of Path
        List of paths to the files containing hole data.
    lower_face_file : str or Path
        Path to the file containing lower face data.

    Returns
    -------
    dict
        A dictionary containing the fitted implant data.
    """
    implant_annotations = sitk.ReadImage(str(implant_annotations_file))
    hole_mesh_dict = make_hole_mesh_dict(hole_files, lower_face_file)
    hole_seg_dict = make_hole_seg_dict(implant_annotations)
    return fit_implant_to_mri(hole_seg_dict, hole_mesh_dict)
