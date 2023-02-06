# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:28:07 2023

@author: yoni.browning
"""

## THIS IS FOR READING SLICER ANNOTATION
import json
from numpy import array


def markup_json_to_numpy(filename):
    f = open(filename,)
    data = json.load(f)
    pts = data['markups'][0]['controlPoints']
    name = []
    pos = []
    for ii,pt in enumerate(pts):
        name.append(pt['label'])
        pos.append(pt['position'])
    return array(pos),name


def markup_json_to_dict(filename):
    f = open(filename,)
    data = json.load(f)
    pts = data['markups'][0]['controlPoints']
    
    output = {}
    for ii,pt in enumerate(pts):
        output[pt['label']] = array(pt['position'])
    return output