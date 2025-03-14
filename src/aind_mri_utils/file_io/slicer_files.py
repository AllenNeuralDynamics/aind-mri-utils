"""Functions for working with slicer files"""

import json
import logging
import re
from typing import Tuple

import numpy as np
import SimpleITK as sitk

from aind_mri_utils.coordinate_systems import convert_coordinate_system

from ..sitk_volume import (
    find_points_equal_to,
    transform_sitk_indices_to_physical_points,
)
from ..utils import find_indices_equal_to

logger = logging.getLogger(__name__)


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


def get_segmented_labels(label_vol):
    """
    Extract metadata from the implant volume and return the segmentation label
    dictionary.

    Parameters
    ----------
    label_vol : SimpleITK.Image
        The label volume from which to extract segmentation information.

    Returns
    -------
    dict
        A dictionary mapping segmentation labels to their corresponding values
        in the volume.
    """
    metadata_dict = {
        k: label_vol.GetMetaData(k) for k in label_vol.GetMetaDataKeys()
    }
    label_dict = find_seg_nrrd_header_segment_info(metadata_dict)
    return label_dict


def load_segmentation_points(label_vol, order=None, image=None):
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
    label_dict = get_segmented_labels(label_vol)

    if order is None:
        order = list(label_dict.keys())

    labels = []
    positions = []
    if image is None:
        for label_ndx, label in enumerate(order):
            these_positions = find_points_equal_to(
                label_vol, label_dict[label]
            )
            label_ndxs = np.full(these_positions.shape[0], label_ndx)
            positions.append(these_positions)
            labels.append(label_ndxs)
        return np.concatenate(positions), np.concatenate(labels)
    else:
        weights = []
        label_arr = sitk.GetArrayViewFromImage(label_vol)
        img_arr = sitk.GetArrayViewFromImage(image)
        for label_ndx, label in enumerate(order):
            val = label_dict[label]
            ndxs = find_indices_equal_to(label_arr, val)
            sitk_ndxs = ndxs[:, ::-1]
            these_positions = transform_sitk_indices_to_physical_points(
                label_vol, sitk_ndxs
            )
            label_ndxs = np.full(these_positions.shape[0], label_ndx)
            np_ndxs = tuple(ndxs.T)
            these_weights = img_arr[np_ndxs]
            positions.append(these_positions)
            weights.append(these_weights)
            labels.append(label_ndxs)
        return (
            np.concatenate(positions),
            np.concatenate(labels),
            np.concatenate(weights),
        )


def markup_json_to_numpy(filename):
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


def markup_json_to_dict(filename):
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
    pos, names, coord_string = markup_json_to_numpy(filename)
    return dict(zip(names, pos)), coord_string


def create_slicer_fcsv(filename, pts_dict, direction="LPS"):
    """
    Save fCSV file that is slicer readable.
    """
    # Create output file
    with open(filename, "w+") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# CoordinateSystem = {}\n".format(direction))
        f.write(
            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"  # noqa: E501
        )

        for pt_no, key in enumerate(pts_dict.keys()):
            x, y, z = pts_dict[key]
            f.write(
                "{:d},{:f},{:f},{:f},0,0,0,1,1,1,0,{!s},,vtkMRMLScalarVolumeNode1\n".format(  # noqa: E501"
                    pt_no + 1, x, y, z, key
                )
            )


def _parse_slicer_fcsv_header(file):
    "Parse the header of a slicer fcsv file, returning the last line read"
    line = file.readline()
    source_coordinate_system = None
    column_headers = []
    # Parse the header
    while line and line.startswith("#"):
        key, _, value = line.partition("=")
        value = value.strip()
        if "CoordinateSystem" in key:
            source_coordinate_system = value
        elif "columns" in key:
            column_headers = [col.strip() for col in value.split(",")]
        line = file.readline()
    if not column_headers:
        raise ValueError("No column headers found in file")
    if source_coordinate_system is None:
        raise ValueError("No CoordinateSystem found in file")
    col_ndxs = {col: i for i, col in enumerate(column_headers)}
    if "label" not in col_ndxs:
        raise ValueError("No 'label' column found in file")

    # Get the indices of the needed columns
    label_ndx = col_ndxs["label"]
    coord_ndxs = [col_ndxs[axis] for axis in ["x", "y", "z"]]
    last_line = line

    return last_line, source_coordinate_system, label_ndx, coord_ndxs


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
    logger.debug(f"Reading {filename} with direction {direction}")

    point_dictionary = {}

    with open(filename, "r") as file:
        line, source_coordinate_system, label_ndx, coord_ndxs = (
            _parse_slicer_fcsv_header(file)
        )
        need_conversion = source_coordinate_system != direction

        while line:
            point_parts = line.strip().split(",")
            point_key = point_parts[label_ndx]
            point_values = np.array(
                [float(point_parts[i]) for i in coord_ndxs]
            )
            if need_conversion:
                point_values = convert_coordinate_system(
                    point_values, source_coordinate_system, direction
                )
            point_dictionary[point_key] = point_values
            line = file.readline()

    return point_dictionary
