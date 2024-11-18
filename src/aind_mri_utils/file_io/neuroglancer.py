import json
import numpy as np

import warnings


def read_neuroglancer_probes_and_annotations(filename,
                                 probe_layers = None,
                                 annotation_layers = None):

    """
    Reads a Neuroglancer JSON file and extracts probe locations and annotation data.

    This function parses a Neuroglancer JSON file that contains both probe and annotation layers,
    extracts relevant information, and returns the probe coordinates, annotation coordinates,
    voxel spacing, and dimension order. The probe and annotation layers are either automatically
    determined from the file or can be specified by the user.

    Function written by Yoni, Docstring written by ChatGPT.

    Parameters:
    -----------
    filename : str
        Path to the Neuroglancer JSON file to read.

    probe_layers : list, optional
        List of specific probe layers to extract. If None, it will automatically extract
        all layers whose names are numeric strings (default is None).
        Use flag -1 to skip.

    annotation_layers : list, optional
        List of specific annotation layers to extract. If None, it will automatically extract
        all layers not used as probe layers (default is None).
        Use flag -1 to skip.

    Returns:
    --------
    probes : dict
        A dictionary where the keys are the probe layer names (strings) and the values are
        NumPy arrays of the probe coordinates, with the shape (N, 3) where N is the number of
        probe points in that layer.

    annotations : dict
        A dictionary where the keys are annotation descriptions (strings) and the values are
        NumPy arrays of the annotation coordinates, with the shape (3,).

    spacing : numpy.ndarray
        A NumPy array representing the voxel spacing in the x, y, z dimensions.

    dimension_order : numpy.ndarray
        A NumPy array containing the dimension order in x, y, z format for reordering.

    Notes:
    ------
    - The function assumes that the dimensions in the Neuroglancer file are ordered in a certain
      way and reorders the data to match the 'xyz' format for consistency.
    - Probes are expected to be defined in layers where the layer name is numeric.
    - If no probe or annotation layers are provided, the function will attempt to autodetect them.
    """
    warnings.warn(
        "This function is deprecated and will be removed soon. Please use read_neuroglancer_annotation_layers for a more stable/long term solution",
        DeprecationWarning
    )

    # Read the json file into memory
    with open(filename) as f:
        data = json.load(f)

    # Get the dimension order/spacing for file.
    dimension_order = list(data['dimensions'].keys())[:3]
    spacing = np.zeros(len(dimension_order))
    for ii,key in enumerate(dimension_order):
        spacing[ii] = data['dimensions'][key][0]
    # For reordering data into xyz
    dim_order = list(np.argsort(dimension_order))


    layers = data['layers']
    probes = {}

    if probe_layers==-1:
        probes = False
    elif probe_layers == None:
        for ii,this_layer in enumerate(layers):
            if this_layer['type']!='annotation':
                continue
            if this_layer['name'].isdigit():
                probes[this_layer['name']] = []
                for jj,this_point in enumerate(this_layer['annotations']):
                    probes[this_layer['name']].append(this_point['point'][:-1])
                probes[this_layer['name']]  = np.array(probes[this_layer['name']])*spacing
                probes[this_layer['name']] = probes[this_layer['name']][:,dim_order]
    else:
        for ii,this_layer in enumerate(probe_layers):
            probes[this_layer['name']] = []
            for jj,this_point in enumerate(this_layer['annotations']):
                probes[this_layer['name']].append(this_point['point'][:-1])
            probes[this_layer['name']]  = np.array(probes[this_layer['name']])*spacing
            probes[this_layer['name']] = probes[this_layer['name']][:,dim_order]


    annotations = {}
    if annotation_layers==-1:
        annotations = False
    elif annotation_layers == None:
        if probe_layers==None:
            probe_layers = list(probes.keys())
        for ii,this_layer in enumerate(layers):
            if (this_layer['type']!='annotation') or (this_layer['name'] in probe_layers):
                continue
            else:
                print(this_layer['name'])

                for jj,this_point in enumerate(this_layer['annotations']):
                    annotations[this_point['description'][:-1]] = np.array(this_point['point'][:-1])*spacing
                    annotations[this_point['description'][:-1]] = annotations[this_point['description'][:-1]][dim_order]
    else:
        for ii,this_layer in enumerate(annotation_layers):
            for jj,this_point in enumerate(this_layer['annotations']):
                annotations[this_point['description'][:-1]] = np.array(this_point['point'][:-1])*spacing
                annotations[this_point['description'][:-1]] = annotations[this_point['description'][:-1]][dim_order]

    return probes,annotations,spacing[dim_order],np.array(dimension_order)[dim_order]

def read_neuroglancer_annotation_layers(filename,layer_names = None,return_description = False):
    """
    Reads annotation layers from a Neuroglancer JSON file and returns the corresponding annotation points.

    Function written by Yoni, Docstring written by ChatGPT.

    Parameters:
    -----------
    filename : str
        The path to the Neuroglancer JSON file containing annotations.

    layer_names : str, list, or None, optional, default=None
        The name or list of names of the annotation layers to be read. If None, all layers of type 'annotation'
        will be automatically selected.

    return_description : bool, optional, default=False
        If True, returns the descriptions of the annotations (if available) along with the points.

    Returns:
    --------
    annotations : dict
        A dictionary where keys are layer names and values are arrays of annotation points in the
        corresponding layer. The points are scaled according to the spacing defined in the JSON file
        and ordered in XYZ format.

    descriptions : dict, optional
        A dictionary where keys are layer names and values are arrays of descriptions corresponding
        to the annotation points. This is only returned if `return_description` is set to True.

    Raises:
    -------
    ValueError
        If an invalid `layer_names` input is provided or if the specified layer is not found in the file.

    Notes:
    ------
    - The function reorders the spatial dimensions into XYZ format based on the dimension order in the file.
    - Annotations are adjusted according to the spacing of the dimensions.
    - If `return_description` is set to True, missing descriptions will be returned as None.

    Example:
    --------
    annotations, descriptions = read_neuroglancer_annotation_layers("annotations.json", layer_names="layer1", return_description=True)
    """

    # Read the json file
    with open(filename) as f:
        data = json.load(f)

    ng_layers = data['layers']
    ng_layer_names = [x['name'] for x in ng_layers]

    # Get the dimension order/spacing for file.
    dimension_order = list(data['dimensions'].keys())[:3]
    spacing = np.zeros(len(dimension_order))
    for ii,key in enumerate(dimension_order):
        spacing[ii] = data['dimensions'][key][0]

    # For reordering data into xyz
    dim_order = list(np.argsort(dimension_order))

    if isinstance(layer_names,str):
        layer_names = [layer_names]
    elif (layer_names==None):
        is_annotation = [x['type']=='annotation' for x in ng_layers]
        is_annotation_idx = np.where(is_annotation)[0]
        layer_names = np.array(ng_layer_names)[is_annotation_idx]

    elif not isinstance(layer_names,list):
        raise ValueError('Inputs "layer_names" must be annotation name string or list of annotation name strings')

    annotations = {}
    descriptions = {}

    for ii,layer_name in enumerate(layer_names):
        if layer_name in ng_layer_names:
            this_ng_layer_idx = ng_layer_names.index(layer_name)
        else:
            raise ValueError(f' {layer_name} not found. Variable "layer_names" must match neuroglancer file layers.')
        this_layer = ng_layers[this_ng_layer_idx]

        annotations[layer_name] = []
        descriptions[layer_name] = []
        for jj,this_point in enumerate(this_layer['annotations']):
            annotations[this_layer['name']].append(this_point['point'][:-1])
            if return_description:
                if 'description' in list(this_point.keys()):
                    descriptions[layer_name].append(this_point['description'])
                else:
                    descriptions[layer_name].append(None)
        annotations[layer_name]  = np.array(annotations[layer_name])*spacing
        annotations[layer_name] = annotations[layer_name][:,dim_order]
        descriptions[layer_name] = np.array(descriptions[layer_name])
    if return_description:
        return annotations,descriptions
    else:
        return annotations
