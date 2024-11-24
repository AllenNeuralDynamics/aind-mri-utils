"""Module to fit implant rotations to MRI data."""

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import fmin
import SimpleITK as sitk
import trimesh
import os
from aind_mri_utils import coordinate_systems as cs

from aind_mri_utils.meshes import (
    distance_to_all_triangles_in_mesh,
    distance_to_closest_point_for_each_triangle_in_mesh,
)
from aind_mri_utils.rotations import (
    create_homogeneous_from_euler_and_translation,
    prepare_data_for_homogeneous_transform,
)
from aind_mri_utils.file_io.slicer_files import (
    get_segmented_labels,
)


def cost_function_v2(T, hole_dict, annotate_hole_pts):
    trans = create_homogeneous_from_euler_and_translation(*T)

    jobs = []
    for hole_id in hole_dict.keys():
        if hole_id not in annotate_hole_pts.keys():
            continue
        if hole_id != -1:
            this_hole_mesh = hole_dict[hole_id]
            these_hole_pts = annotate_hole_pts[hole_id]

            transformed_hole_pts = np.dot(
                prepare_data_for_homogeneous_transform(these_hole_pts), trans
            )
            jobs.append(
                delayed(distance_to_all_triangles_in_mesh)(
                    this_hole_mesh, transformed_hole_pts
                )
            )
        elif hole_id == -1:
            lower_mesh = hole_dict[hole_id]
            brain_outline = annotate_hole_pts[hole_id]
            transformed_brain_outline = np.dot(
                prepare_data_for_homogeneous_transform(brain_outline), trans
            )
            jobs.append(
                delayed(distance_to_closest_point_for_each_triangle_in_mesh)(
                    lower_mesh, transformed_brain_outline
                )
            )

    results = Parallel(n_jobs=-1)(jobs)
    distances, _ = zip(*results)
    total_distance = np.sum(
        np.concatenate([np.array(x).flatten() for x in distances])
    )
    return total_distance


def fit_implant_to_mri(annotate_hole_pts, hole_dict):
    initialization_hole = 4
    annotation_mean = np.mean(annotate_hole_pts[initialization_hole], axis=0)
    model_mean = np.mean(hole_dict[initialization_hole].vertices, axis=0)
    init_offset = model_mean - annotation_mean
    T = [0, 0, 0, init_offset[0], init_offset[1], init_offset[2]]

    output = fmin(
        cost_function_v2,
        T,
        args=(hole_dict, annotate_hole_pts),  # ,hole_plot_dict),
        xtol=1e-6,
        maxiter=2000,
    )
    return output


def read_hole_annotations(
    implant_annotations_file, hole_files, lower_face_file
):
    implant_annotations = sitk.ReadImage(str(implant_annotations_file))

    hole_dict = {}
    for filename in hole_files:
        hole_num = int(filename.split("Hole")[-1].split(".")[0])
        hole_dict[hole_num] = trimesh.load(os.path.join(hole_folder, filename))
        hole_dict[hole_num].vertices = cs.convert_coordinate_system(
            hole_dict[hole_num].vertices, "ASR", "LPS"
        )  # Preserves shape!

    # Get the lower face, store with key -1
    hole_dict[-1] = trimesh.load(lower_face_file)
    hole_dict[-1].vertices = cs.convert_coordinate_system(
        hole_dict[-1].vertices, "ASR", "LPS"
    )  # Preserves shape!
    implant_annotations_names = get_segmented_labels(implant_annotations)

    annotate_hole_pts = {}
