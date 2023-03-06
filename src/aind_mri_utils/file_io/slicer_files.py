"""Functions for working with slicer files"""

import json
from typing import Tuple

import numpy as np


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
