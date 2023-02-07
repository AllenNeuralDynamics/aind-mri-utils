"""Functions for working with obj files"""

import pywavefront


def load_obj_wavefront(filename):
    """
    Wrapper for loading a pywavefront scene

    Parameters
    ==========
    filename - name of file to load

    Returns
    =======
    scene - scene with default params so I don't need to remember to type them
    """

    scene = pywavefront.Wavefront(
        filename, strict=False, create_materials=True, collect_faces=True
    )
    return scene
