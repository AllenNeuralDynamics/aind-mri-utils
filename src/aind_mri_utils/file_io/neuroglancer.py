import json

import numpy as np


def read_neuroglancer_annotation_layers(
    filename,
    layer_names=None,
    return_description=True,
    reorder=False,
):
    """
    Reads annotation layers from a Neuroglancer JSON file.

    Parameters
    ----------
    filename : str
        Path to the Neuroglancer JSON file.
    layer_names : str or list of str or None, optional
        Names of annotation layers to extract. If None, auto-detects all
        annotation layers. Default is None.
    return_description : bool, optional
        If True, returns annotation descriptions alongside points. Default is
        True.
    reorder : bool, optional
        If True, reorders the dimensions to x, y, z order. Default is False.

    Returns
    -------
    annotations : dict
        Dictionary of annotation coordinates for each layer.
    units : list of str
        Units of each dimension (e.g., ['m', 'm', 'm']).
    dimension_order : list of str
        Dimension keys in the order used (e.g., ['x', 'y', 'z']).
    descriptions : dict or None
        Dictionary of annotation descriptions for each layer. Returned only if
        `return_description` is True, otherwise None.
    """
    data = _load_json_file(filename)
    dimension_order, spacing, units, keep_dims_ndxs = (
        _extract_spacing_and_order(data["dimensions"])
    )
    if reorder:
        dim_sp = np.argsort(dimension_order)
        dimension_order = dimension_order[dim_sp]
    else:
        dim_sp = None

    layers = data["layers"]
    layer_names = _resolve_layer_names(
        layers, layer_names, layer_type="annotation"
    )
    annotations, descriptions = _process_annotation_layers(
        layers,
        layer_names,
        keep_dims_ndxs,
        spacing,
        dim_sp,
        return_description,
    )

    return annotations, units, dimension_order, descriptions


def get_neuroglancer_annotation_points(
    filename,
    layer_names=None,
    return_description=True,
):
    """
    Reads annotation layers from a Neuroglancer JSON file.

    Parameters
    ----------
    filename : str
        Path to the Neuroglancer JSON file.
    layer_names : str or list of str or None, optional
        Names of annotation layers to extract. If None, auto-detects all
        annotation layers. Default is None.
    return_description : bool, optional
        If True, returns annotation descriptions alongside points. Default is
        True.

    Returns
    -------
    annotations : dict
        Dictionary of annotation coordinates for each layer.
    dimension_order : list of str
        Dimension keys in the order used (e.g., ['x', 'y', 'z']).
    descriptions : dict or None
        Dictionary of annotation descriptions for each layer. Returned only if
        `return_description` is True, otherwise None.
    """
    data = _load_json_file(filename)
    dimension_order, _, _, keep_dims_ndxs = _extract_spacing_and_order(
        data["dimensions"]
    )

    layers = data["layers"]
    layer_names = _resolve_layer_names(
        layers, layer_names, layer_type="annotation"
    )
    annotations, descriptions = _process_annotation_layers(
        layers,
        layer_names,
        keep_dims_ndxs=keep_dims_ndxs,
        return_description=return_description,
    )

    return (
        annotations,
        dimension_order,
        descriptions,
    )


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


def _extract_spacing_and_order(dimension_data, keep_dims=["z", "y", "x"]):
    """
    Extracts voxel spacing and dimension order from the Neuroglancer file.

    Parameters
    ----------
    dimension_data : dict
        Neuroglancer JSON dimension data.
    keep_dims : list of str, optional
        List of dimensions to keep. Default is ['z', 'y', 'x'].

    Returns
    -------
    dimension_order : numpy.ndarray
        Dimension keys (e.g., ['x', 'y', 'z']).
    spacing : numpy.ndarray
        Voxel spacing in each dimension.
    units : list of str
        Units of each dimension (e.g., ['m', 'm', 'm']).
    keep_dim_ndxs : numpy.ndarray
        Indices of dimensions to keep.
    """
    keep_set = set(keep_dims)
    dimension_list = np.array(list(dimension_data.keys()))
    keep_dim_ndxs = np.array(
        [i for i, d in enumerate(dimension_list) if d in keep_set]
    )
    dimension_order = dimension_list[keep_dim_ndxs]
    spacing = []
    units = []
    for key in dimension_order:
        spacing.append(dimension_data[key][0])
        units.append(dimension_data[key][1])
    spacing = np.array(spacing)
    return dimension_order, spacing, units, keep_dim_ndxs


def _resolve_layer_names(layers, layer_names, layer_type):
    """
    Resolves layer names based on user input or auto-detects layers of the
    given type.

    Parameters
    ----------
    layers : list of dict
        Neuroglancer JSON layers.
    layer_names : str or list of str or None
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
        "Invalid input for layer_names. Expected a string, "
        "list of strings, or None."
    )


def _process_annotation_layers(
    layers,
    layer_names,
    keep_dims_ndxs=None,
    spacing=None,
    dim_order=None,
    return_description=True,
    convert_indices=False,
):
    """
    Processes annotation layers to extract points and descriptions.

    Parameters
    ----------
    layers : list of dict
        Neuroglancer JSON layers.
    layer_names : list of str
        Names of annotation layers to extract.
    keep_dims_ndxs : numpy.ndarray or None, optional
        Indices of dimensions to keep. If None, all dimensions are kept.
        Default is None.
    spacing : numpy.ndarray or None, optional
        Voxel spacing for scaling. If None, no scaling is done. Default is
        None.
    dim_order : numpy.ndarray or None, optional
        Indices to reorder dimensions into x, y, z order. If None, no
        reordering is done. Default is None.
    return_description : bool, optional
        Whether to extract descriptions alongside points. Default is True.
    convert_indices : bool, optional
        If True, converts indices to physical coordinates. Default is False.

    Returns
    -------
    annotations : dict
        Annotation points for each layer.
    descriptions : dict or None
        Annotation descriptions for each layer, or None if not requested.
    """
    annotations = {}
    descriptions = {} if return_description else None
    for layer_name in layer_names:
        layer = _get_layer_by_name(layers, layer_name)
        points, layer_descriptions = _process_layer_and_descriptions(
            layer,
            keep_dims_ndxs=keep_dims_ndxs,
            spacing=spacing,
            dim_order=dim_order,
            return_description=return_description,
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
    layers : list of dict
        Neuroglancer JSON layers.
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
    layer,
    keep_dims_ndxs=None,
    spacing=None,
    dim_order=None,
    return_description=True,
):
    """
    Processes layer points and descriptions.

    Parameters
    ----------
    layer : dict
        Layer data.
    keep_dims_ndxs : numpy.ndarray or None, optional
        Indices of dimensions to keep. If None, all dimensions are kept.
        Default is None.
    spacing : numpy.ndarray or None, optional
        Voxel spacing for scaling. If None, no scaling is done. Default is
        None.
    dim_order : numpy.ndarray or None, optional
        Indices to reorder dimensions into x, y, z order. Default is None.
    return_description : bool, optional
        Whether to extract descriptions. Default is True.

    Returns
    -------
    points : numpy.ndarray
        Scaled and reordered points.
    descriptions : numpy.ndarray or None
        Descriptions, or None if not requested.
    """
    points = []
    for annotation in layer["annotations"]:
        point_arr = np.array(annotation["point"])
        if keep_dims_ndxs is not None:
            point_arr = point_arr[keep_dims_ndxs]
        points.append(point_arr)
    points = np.array(points)
    if spacing is not None:
        points *= spacing
    if dim_order is not None:
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
    Reads image source URL(s) from a Neuroglancer JSON file.

    Parameters
    ----------
    filename : str
        Path to the Neuroglancer JSON file.

    Returns
    -------
    list of str
        List of image source URLs.
    """
    data = _load_json_file(filename)

    image_layer = [x for x in data["layers"] if x["type"] == "image"]
    return [x["source"]["url"] for x in image_layer]
