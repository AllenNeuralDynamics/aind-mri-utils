import json
import numpy as np

def read_slicer_markup_json(filename):
    with open(filename) as file:
        data = json.load(file)
    pts = data['markups'][0]['controlPoints']
    name = []
    pos = []
    for ii,pt in enumerate(pts):
        name.append(pt['label'])
        pos.append(pt['position'])
    return np.array(pos), name
