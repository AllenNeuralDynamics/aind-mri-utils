import json

import numpy as np


def read_neuroglancer_probes_and_annotations(
    filename, probe_layers=None, annotation_layers=None
):
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

    # Read the json file into memory
    with open(filename) as f:
        data = json.load(f)

    # Get the dimension order/spacing for file.
    dimension_order = list(data["dimensions"].keys())[:3]
    spacing = np.zeros(len(dimension_order))
    for ii, key in enumerate(dimension_order):
        spacing[ii] = data["dimensions"][key][0]
    # For reordering data into xyz
    dim_order = list(np.argsort(dimension_order))

    layers = data["layers"]
    probes = {}

    if probe_layers == -1:
        probes = False
    elif probe_layers == None:
        for ii, this_layer in enumerate(layers):
            if this_layer["type"] != "annotation":
                continue
            if this_layer["name"].isdigit():
                probes[this_layer["name"]] = []
                for jj, this_point in enumerate(this_layer["annotations"]):
                    probes[this_layer["name"]].append(this_point["point"][:-1])
                probes[this_layer["name"]] = (
                    np.array(probes[this_layer["name"]]) * spacing
                )
                probes[this_layer["name"]] = probes[this_layer["name"]][
                    :, dim_order
                ]
    else:
        for ii, this_layer in enumerate(probe_layers):
            probes[this_layer["name"]] = []
            for jj, this_point in enumerate(this_layer["annotations"]):
                probes[this_layer["name"]].append(this_point["point"][:-1])
            probes[this_layer["name"]] = (
                np.array(probes[this_layer["name"]]) * spacing
            )
            probes[this_layer["name"]] = probes[this_layer["name"]][
                :, dim_order
            ]

    annotations = {}
    if annotation_layers == -1:
        annotations = False
    elif annotation_layers == None:
        if probe_layers == None:
            probe_layers = list(probes.keys())
        for ii, this_layer in enumerate(layers):
            if (this_layer["type"] != "annotation") or (
                this_layer["name"] in probe_layers
            ):
                continue
            else:
                print(this_layer["name"])

                for jj, this_point in enumerate(this_layer["annotations"]):
                    annotations[this_point["description"][:-1]] = (
                        np.array(this_point["point"][:-1]) * spacing
                    )
                    annotations[this_point["description"][:-1]] = annotations[
                        this_point["description"][:-1]
                    ][dim_order]
    else:
        for ii, this_layer in enumerate(annotation_layers):
            for jj, this_point in enumerate(this_layer["annotations"]):
                annotations[this_point["description"][:-1]] = (
                    np.array(this_point["point"][:-1]) * spacing
                )
                annotations[this_point["description"][:-1]] = annotations[
                    this_point["description"][:-1]
                ][dim_order]

    return (
        probes,
        annotations,
        spacing[dim_order],
        np.array(dimension_order)[dim_order],
    )
