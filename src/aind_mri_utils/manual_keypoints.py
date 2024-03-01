"""
Code for handling manual keypoints transforms using vtk
"""

import numpy as np
import vtk


def define_transform(source_landmarks, target_landmarks):
    """
    Defines a non-linear warp between a set of source and target landmarks

    Parameters
    ==========
    source_landmarks - np.ndarray (N x 3)
    target_landmarks - np.ndarray (N x 3)

    Returns
    =======
    transform - vtkThinPlateSplineTransform

    """

    transform = vtk.vtkThinPlateSplineTransform()
    source_points = vtk.vtkPoints()
    target_points = vtk.vtkPoints()

    for i in range(source_landmarks.shape[0]):
        source_points.InsertNextPoint(source_landmarks[i, :])

    for i in range(target_landmarks.shape[0]):
        target_points.InsertNextPoint(target_landmarks[i, :])

    transform.SetBasisToR()  # for 3D transform
    transform.SetSourceLandmarks(source_points)
    transform.SetTargetLandmarks(target_points)
    transform.Update()

    return transform


def apply_transform(transform, points):
    """
    Applies a non-linear warp to a set of points

    Parameters
    ==========
    transform - vtkThinPlateSplineTransform
    points - np.ndarray (N x 3)

    Returns
    =======
    warped_points - np.ndarray (N x 3)

    """

    warped_points = np.zeros(points.shape)
    for i in range(points.shape[0]):
        warped_points[i, :] = transform.TransformPoint(points[i, :])

    return warped_points
