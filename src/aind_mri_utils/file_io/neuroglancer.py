import json

import numpy as np


def read_neuroglancer_annotation_layers(
    filename, layer_names=None, return_description=False
):
    """
    Reads annotation layers from a Neuroglancer JSON file.

    Parameters
    ----------
    filename : str
        Path to the Neuroglancer JSON file.
    layer_names : str, list of str, or None, optional
        Names of annotation layers to extract. If None, auto-detects all
        annotation layers.
    return_description : bool, optional, default=False
        If True, returns annotation descriptions alongside points.

    Returns
    -------
    annotations : dict
        Dictionary of annotation coordinates for each layer.
    descriptions : dict, optional
        Dictionary of annotation descriptions for each layer. Returned only if
        `return_description` is True.
    """
    data = _load_json_file(filename)
    _, spacing, dim_order = _extract_spacing_and_order(data["dimensions"])

    layers = data["layers"]
    layer_names = _resolve_layer_names(
        layers, layer_names, layer_type="annotation"
    )
    annotations, descriptions = _process_annotation_layers(
        layers, layer_names, spacing, dim_order, return_description
    )

    if return_description:
        return annotations, descriptions
    return annotations


def _load_json_file(filename):
    """
    Loads and parses a JSON file.

    Parameters
    ----------
    filename : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON data.
    """
    with open(filename, "r") as f:
        return json.load(f)


def _extract_spacing_and_order(dimension_data):
    """
    Extracts voxel spacing and dimension order from the Neuroglancer file.

    Parameters
    ----------
    data : dict
        Neuroglancer JSON data.

    Returns
    -------
    dimension_order : list of str
        Dimension keys (e.g., ['x', 'y', 'z']).
    spacing : numpy.ndarray
        Voxel spacing in each dimension.
    dim_order : numpy.ndarray
        Indices to reorder dimensions into x, y, z order.
    """
    dimension_order = list(dimension_data.keys())[:3]
    spacing = np.array([dimension_data[key][0] for key in dimension_order])
    dim_order = np.argsort(dimension_order)
    return dimension_order, spacing, dim_order


def _resolve_layer_names(layers, layer_names, layer_type):
    """
    Resolves layer names based on user input or auto-detects layers of the
    given type.

    Parameters
    ----------
    layers : list
        Neuroglancer JSON layers.
    layer_names : str, list of str, or None
        User-specified layer names or None to auto-detect.
    layer_type : str
        Type of layer to extract ('annotation' or 'probe').

    Returns
    -------
    list of str
        List of resolved layer names.

    Raises
    ------
    ValueError
        If the input `layer_names` is invalid.
    """
    if isinstance(layer_names, str):
        return [layer_names]
    if layer_names is None:
        return [
            layer["name"] for layer in layers if layer["type"] == layer_type
        ]
    if isinstance(layer_names, list):
        return layer_names
    raise ValueError(
        "Invalid input for layer_names. Expected a string, list of strings, "
        "or None."
    )


def _extract_layers(
    layers, layer_names, spacing, dim_order, layer_type, exclude_layers=None
):
    """
    Extracts data for specified layers of a given type.

    Parameters
    ----------
    layers : list
        Neuroglancer JSON data.
    layer_names : list of str or None
        Names of layers to extract. If None, auto-detects layers of the given
        type.
    spacing : numpy.ndarray
        Voxel spacing for scaling.
    dim_order : numpy.ndarray
        Indices to reorder dimensions into x, y, z order.
    layer_type : str
        Type of layer to extract ('annotation' or 'probe').
    exclude_layers : list of str, optional
        Layers to exclude from extraction.

    Returns
    -------
    dict
        Dictionary of layer data, keyed by layer name.
    """
    resolved_layer_names = _resolve_layer_names(
        layers, layer_names, layer_type
    )
    if exclude_layers:
        resolved_layer_names = [
            name for name in resolved_layer_names if name not in exclude_layers
        ]

    sel_layers = {
        name: _process_layer(
            _get_layer_by_name(layers, name), spacing, dim_order
        )
        for name in resolved_layer_names
    }
    return sel_layers


def _process_layer(layer, spacing, dim_order):
    """
    Processes a single layer and extracts scaled and reordered data.

    Parameters
    ----------
    layer : dict
        Neuroglancer layer data.
    spacing : numpy.ndarray
        Voxel spacing for scaling.
    dim_order : numpy.ndarray
        Indices to reorder dimensions into x, y, z order.

    Returns
    -------
    numpy.ndarray
        Array of processed data points (N, 3).
    """
    points = [annotation["point"][:-1] for annotation in layer["annotations"]]
    return np.array(points) * spacing[:, dim_order]


def _process_annotation_layers(
    layers, layer_names, spacing, dim_order, return_description
):
    """
    Processes annotation layers to extract points and descriptions.

    Parameters
    ----------
    data : dict
        Neuroglancer JSON data.
    layer_names : list of str
        Names of annotation layers to extract.
    spacing : numpy.ndarray
        Voxel spacing for scaling.
    dim_order : numpy.ndarray
        Indices to reorder dimensions into x, y, z order.
    return_description : bool
        Whether to extract descriptions alongside points.

    Returns
    -------
    dict
        Annotation points for each layer.
    dict or None
        Annotation descriptions for each layer, or None if not requested.
    """
    annotations = {}
    descriptions = {} if return_description else None
    for layer_name in layer_names:
        layer = _get_layer_by_name(layers, layer_name)
        points, layer_descriptions = _process_layer_and_descriptions(
            layer, spacing, dim_order, return_description
        )
        annotations[layer_name] = points
        if return_description:
            descriptions[layer_name] = layer_descriptions

    return annotations, descriptions


def _get_layer_by_name(layers, name):
    """
    Retrieves a layer by its name.

    Parameters
    ----------
    data : dict
        Neuroglancer JSON data.
    name : str
        Layer name to retrieve.

    Returns
    -------
    dict
        Layer data.

    Raises
    ------
    ValueError
        If the layer is not found.
    """
    for layer in layers:
        if layer["name"] == name:
            return layer
    raise ValueError(f'Layer "{name}" not found in the Neuroglancer file.')


def _process_layer_and_descriptions(
    layer, spacing, dim_order, return_description
):
    """
    Processes layer points and descriptions.

    Parameters
    ----------
    layer : dict
        Layer data.
    spacing : numpy.ndarray
        Voxel spacing for scaling.
    dim_order : numpy.ndarray
        Indices to reorder dimensions into x, y, z order.
    return_description : bool
        Whether to extract descriptions.

    Returns
    -------
    numpy.ndarray
        Scaled and reordered points.
    numpy.ndarray or None
        Descriptions, or None if not requested.
    """
    points = [annotation["point"][:-1] for annotation in layer["annotations"]]
    points = np.array(points) * spacing
    points = points[:, dim_order]

    if return_description:
        descriptions = [
            annotation.get("description", None)
            for annotation in layer["annotations"]
        ]
        return points, np.array(descriptions)
    return points, None


def get_image_source(filename):
    """
    Reads image source url from a Neuroglancer JSON file.

    If there are multiple image layers, returns a list of image sources.
    """
    data = _load_json_file(filename)

    image_layer = [x for x in data["layers"] if x["type"] == "image"]
    return [x["source"]["url"] for x in image_layer]
