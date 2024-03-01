"""Functions for working with slicer files"""

import json
import re
from typing import Tuple

import numpy as np
import SimpleITK as sitk

from aind_mri_utils.coordinate_systems import convert_coordinate_system


def extract_control_points(json_data: dict) -> Tuple[np.ndarray, list]:
    """
    Extract points and names from slicer json dict

    Parameters
    ----------
    json_data : dict
        Contents of json file

    Returns
    -------
    pts : numpy.ndarray (N x 3)
        point positions
    labels : list
        labels of controlPoints
    coord_str : str
        String specifying coordinate system of pts, e.g. 'LPS'
    """
    pts = json_data["markups"][0]["controlPoints"]
    coord_str = json_data["markups"][0]["coordinateSystem"]
    labels = []
    pos = []
    for ii, pt in enumerate(pts):
        labels.append(pt["label"])
        pos.append(pt["position"])
    return np.array(pos), labels, coord_str


def find_seg_nrrd_header_segment_info(header):
    """
    parse keys of slicer created dict to find segment names and values


    Parameters
    ----------
    header : dict-like

    Returns
    -------
    segment_info: dict
        pairs of segment name : segment value
    """
    matches = filter(
        None,
        map(lambda s: re.match("^([^_]+)_LabelValue$", s), header.keys()),
    )
    segment_info = dict()
    for m in matches:
        segment_name = header["{}_Name".format(m[1])]
        segment_info[segment_name] = int(header[m[0]])
    return segment_info


def load_segmentation_points(
    label_vol, order=None, image=None
):  # pragma: no cover
    """
    Load segmentation points from a 3D Slicer generated .seg.nrrd file

    Note that, because this is effectively an image loader, it is excluded from
    coverage tests

    Parameters
    ----------
    label_vol : SimpleITK.Image or str
        SimpleITK.Image or filename to open. Must be .seg.nrrd
    order : list of strings, optional
        list of segment names to load.
        Labels will be in order specified by order.
        If None, labels will be loaded in the order they are found in the file.
        Default is None.
    image : SimpleITK.Image, optional
        Image to use for extracting the weights
        (image intensity) for each labeled point

    Returns
    -------
    positions : np.array(N,3)
        xyz positions of the labeled points
    labels : np.array(N,)
        labels of the labeled points
    weights : np.array(N,)  (optional)
        weights of the labeled points. Only returned if image is not None.

    """
    # Read the volume if a string is passed
    if isinstance(label_vol, str):
        if not label_vol.endswith(".seg.nrrd"):
            raise ValueError("label_vol must be a .seg.nrrd file")
        label_vol = sitk.ReadImage(label_vol)

    # Get the labels from the header
    odict = {k: label_vol.GetMetaData(k) for k in label_vol.GetMetaDataKeys()}
    label_dict = find_seg_nrrd_header_segment_info(odict)

    if order is None:
        order = list(label_dict.keys())

    labels = []
    positions = []
    if image is None:  # Do not extract weights
        # Loop through and Load the labels
        for jj, key in enumerate(order):
            filt = sitk.EqualImageFilter()
            is_label = filt.Execute(label_vol, label_dict[key])
            idxx = np.where(is_label)
            idx = np.vstack((idxx[2], idxx[1], idxx[0])).T  # convert to xyz
            this_position = np.zeros(idx.shape)
            for ii in range(idx.shape[0]):
                this_position[ii, :] = is_label.TransformIndexToPhysicalPoint(
                    idx[ii, :].tolist()
                )
            positions.append(this_position)
            labels.append(np.ones(this_position.shape[0], 1) * jj)
        return np.concatenate(positions), np.concatenate(labels)
    else:
        weights = []
        for jj, key in enumerate(order):
            filt = sitk.EqualImageFilter()
            is_label = filt.Execute(label_vol, label_dict[key])
            this_masked_image = sitk.Mask(image, is_label)
            idxx = np.where(sitk.GetArrayViewFromImage(this_masked_image))
            idx = np.vstack((idxx[2], idxx[1], idxx[0])).T
            this_weight = np.zeros(idx.shape[0])
            this_position = np.zeros(idx.shape)
            for ii in range(idx.shape[0]):
                this_weight[ii] = this_masked_image.GetPixel(
                    idx[ii, :].tolist()
                )
                this_position[ii, :] = (
                    this_masked_image.TransformIndexToPhysicalPoint(
                        idx[ii, :].tolist()
                    )
                )

            weights.append(this_weight)
            positions.append(this_position)
            labels.append(np.ones(this_weight.shape) * jj)
        return (
            np.concatenate(positions),
            np.concatenate(labels),
            np.concatenate(weights),
        )


def markup_json_to_numpy(filename):  # pragma: no cover
    """
    Extract control points from a 3D Slicer generated markup JSON file

    Parameters
    ----------
    filename : string
        filename to open. Must be .json
        .mrk.json is ok
    Returns
    -------
    pts, names - numpy.ndarray (N x 3) of point positions and list of
                 controlPoint names

    """
    with open(filename) as f:
        data = json.load(f)
    return extract_control_points(data)


def markup_json_to_dict(filename):  # pragma: no cover
    """
    Extract control points from a 3D Slicer generated markup JSON file

    Parameters
    ----------
    filename : string
        filename to open. Must be .json
        .mrk.json is ok

    Returns
    -------
    Dictionary
        dictionary with keys = point names and values = np.array of points.


    """
    pos, names = markup_json_to_numpy(filename)
    return dict(zip(names, pos))


def create_slicer_fcsv(filename, pts_dict, direction="LPS"):
    """
    Save fCSV file that is slicer readable.
    """
    # Create output file
    with open(filename, "w+") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# CoordinateSystem = {}\n".format(direction))
        f.write(
            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        )

        for ptno, key in enumerate(pts_dict.keys()):
            x, y, z = pts_dict[key]
            f.write(
                "{:d},{:f},{:f},{:f},0,0,0,1,1,1,0,{!s},,vtkMRMLScalarVolumeNode1\n".format(
                    ptno + 1, x, y, z, key
                )
            )


def read_slicer_fcsv(filename, direction="LPS"):
    """
    Read fscv into dictionary.
    While reading, points will be converted to the specified direction.

    Parameters
    ----------
    filename : string
        filename to open. Must be .fcsv
    direction : string (optional)
        direction of the coordinate system of the points in the file.
        Must be one of 'LPS','RAS','LAS','LAI','RAI','RPI','LPI','LAI'
        Default is 'LPS'

    Returns
    -------
    Dictionary
        dictionary with keys = point names and values = np.array of points.
    """
    if not filename.endswith(".fcsv"):
        raise ValueError("File must be a .fcsv file")
    valid_directions = {"LPS", "RAS", "LAS", "LAI", "RAI", "RPI", "LPI", "LAI"}
    if direction not in valid_directions:
        raise ValueError(f"Direction must be one of {valid_directions}")

    point_dictionary = {}
    coordinate_system = None
    columns = []

    with open(filename, "r") as f:

        for ii, line in enumerate(f):
            if line.startswith("#"):
                if "CoordinateSystem" in line:
                    coordinate_system = line.split("=")[1].strip()
                elif "columns" in line:
                    columns = [
                        col.strip() for col in line.split("=")[1].split(",")
                    ]
            else:
                point_data = line.strip().split(",")
                if "label" in columns and all(
                    axis in columns for axis in ["x", "y", "z"]
                ):
                    point_key = point_data[columns.index("label")]
                    point_values = np.array(
                        [
                            float(point_data[columns.index(axis)])
                            for axis in ["x", "y", "z"]
                        ]
                    )
                    point_dictionary[point_key] = point_values
    if coordinate_system and coordinate_system != direction:
        for key, value in point_dictionary.items():
            point_dictionary[key] = convert_coordinate_system(
                value, coordinate_system, direction
            )

    return point_dictionary
