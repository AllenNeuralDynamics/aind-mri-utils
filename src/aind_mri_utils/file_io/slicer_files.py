"""Functions for working with slicer files"""

import numpy as np
from typing import Tuple

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
    pts = json_data['markups'][0]['controlPoints']
    names = []
    pos = []
    for ii,pt in enumerate(pts):
        names.append(pt['label'])
        pos.append(pt['position'])
    return np.array(pos), names
