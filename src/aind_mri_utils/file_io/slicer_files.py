"""Functions for working with slicer files"""

from typing import Tuple

import numpy as np
import json


def extract_control_points(json_data: dict) -> Tuple[np.ndarray, list]:
    """
    Extract points and names from slicer json dict

    Parameters
    ==========
    json_data - `dict` with contents of json file

    Returns
    =======
    pts, names - numpy.ndarray (N x 3) of point positions and list of
                 controlPoint names
    """
    pts = json_data["markups"][0]["controlPoints"]
    names = []
    pos = []
    for ii, pt in enumerate(pts):
        names.append(pt["label"])
        pos.append(pt["position"])
    return np.array(pos), names


def markup_json_to_numpy(filename):  # pragma: no cover
    """
    Extract control points from a 3D Slicer generated markup json file

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
    Extract control points from a 3D Slicer generated markup json file

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
    with open(filename) as f:
        data = json.load(f)
    pos, names = extract_control_points(data)
    return dict(zip(names, pos))
