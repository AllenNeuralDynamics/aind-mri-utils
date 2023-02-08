import json
from numpy import array
import numpy as np


def markup_json_to_numpy(filename):
    """
    Extract points from slice markup file, return as numpy array

    Parameters
    ----------
    filename : string
        filename to open. Must be .json
        .mrk.json is ok

    Returns
    -------
    (Nx3) numpy array
        points, one point per row.
    Names : list of strings
        names of points

    """
    f = open(
        filename,
    )
    data = json.load(f)
    pts = data["markups"][0]["controlPoints"]
    name = []
    pos = []
    for ii, pt in enumerate(pts):
        name.append(pt["label"])
        pos.append(pt["position"])
    return array(pos), name


def markup_json_to_dict(filename):
    """
    Extract points from slice markup file, return as labeled dict

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
    f = open(
        filename,
    )
    data = json.load(f)
    pts = data["markups"][0]["controlPoints"]

    output = {}
    for ii, pt in enumerate(pts):
        output[pt["label"]] = array(pt["position"])
    return output
