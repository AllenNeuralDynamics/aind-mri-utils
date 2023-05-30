"""Plotting functions"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mpt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def plot_tri_mesh(ax, vertices, faces, *plot_args, **plot_kwargs):
    """
    Adds a mesh to a 3d plot

    Parameters
    ==========
    ax - 3d axis to plot on
    vertices - N x 3 ndarray of coordinates for N vertices
    faces - N x 3 ndarray of vertex indices defining triangular faces
    *plot_args - varargs passed to plot call
    **plot_kwargs - keyword args passed to plot call

    Returns
    =======
    handles, tri - handles to plot polygons, and triangulation
    """
    tri = mpt.Triangulation(vertices[:, 0], vertices[:, 1], triangles=faces)
    handles = ax.plot_trisurf(tri, vertices[:, 2], *plot_args, **plot_kwargs)
    return handles, tri


# Function from @Mateen Ulhaq and @karlo
def _set_axes_radius(ax, origin, radius):
    """
    Set all three axes to have the same distance around the origin

    Parameters
    ==========
    ax - matplotlib axis handle
    origin - np.ndarray (3) specifying origin point
    radius - scalar radius around origin used to set axis limits
    """

    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


# Function from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def make_3d_ax_look_normal(ax: plt.Axes):
    """
    Changes the aspect ratio of a 3d plot so that dimensions are approximately
    the same size

     Parameters
    ==========
    ax - matplotlib 3d axis
    """
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)


def create_single_colormap(
    colorname,
    N=256,
    saturation=0,
    start_color="white",
    is_transparent=True,
    is_reverse=False,
):
    """
    Creats a matplotlib colormap mapping from start color( )

    Parameters
    ----------
    colorname : TYPE
        DESCRIPTION.
    N : TYPE, optional
        DESCRIPTION. The default is 256.
    saturation : TYPE, optional
        DESCRIPTION. The default is 0.
    start_color : TYPE, optional
        DESCRIPTION. The default is "white".
    is_transparent : TYPE, optional
        DESCRIPTION. The default is True.
    is_reverse : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cmap : TYPE
        DESCRIPTION.

    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    cmap = ListedColormap([start_color, colorname])
    start_color = np.array(cmap(0))
    if is_transparent:
        start_color[-1] = 0
    if not is_reverse:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.linspace(start_color, cmap(1), N),
                    np.tile(cmap(1), (int(saturation * N), 1)),
                )
            )
        )
    else:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.tile(cmap(1), (int(saturation * N), 1)),
                    np.linspace(cmap(1), start_color, N),
                )
            )
        )
    return cmap


def create_single_colormap(
    colorname,
    N=256,
    saturation=0,
    start_color="white",
    is_transparent=True,
    is_reverse=False,
):
    """
    Returns a matplotlib colormap that moves from "start_color" to "colorname"
    Default settings have a transparent start, so that images can be overlaid.

    Parameters
    ----------
    colorname : string
        Matplotlib color name or color hex code
    N : Int, optional
        number of discrete points in map. The default is 256.
    saturation : float, optional
        Fraction of map to be saturated.
        The default is 0, which means the map
        will reach the max color only at the end. Using e.g. 1, the map would
        saturate halfway. Using 2 2/3rds of the map would be saturated, etc.
    start_color : string, optional
        Matplotlib color name or color hex code.
        Gives where the map should start.
        The default is "white".
    is_transparent : Bool, optional
        If true, start_color is transparent. The default is True.
    is_reverse : Bool, optional
        If true, colors run in reverse (i.e. colormap==>startcolor).
        The default is False.

    Returns
    -------
    cmap : matplotlib colormap sequence
    """
    cmap = ListedColormap([start_color, colorname])
    start_color = np.array(cmap(0))
    if is_transparent:
        start_color[-1] = 0
    if not is_reverse:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.linspace(start_color, cmap(1), N),
                    np.tile(cmap(1), (int(saturation * N), 1)),
                )
            )
        )
    else:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.tile(cmap(1), (int(saturation * N), 1)),
                    np.linspace(cmap(1), start_color, N),
                )
            )
        )
    return cmap


def get_prop_cycle():
    """
    Returns the default matplotlib color cycle
    This can be useful for color-matching plots.
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]
