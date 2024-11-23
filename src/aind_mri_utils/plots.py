"""Plotting functions"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mpt
from matplotlib.colors import ListedColormap


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


def get_prop_cycle():
    """
    Returns the colors in the current prop cycle
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    return prop_cycle.by_key()["color"]


def plot_point_cloud_3d(a, pts, *args, **kwargs):
    """Scatter plots a Nx3 array as N 3D points"""
    return a.scatter(pts[:, 0], pts[:, 1], pts[:, 2], *args, **kwargs)


def create_single_colormap(
    color_name,
    N=256,
    saturation=0,
    start_color="white",
    is_transparent=True,
    is_reverse=False,
):
    """ "
    Creates a colormap with a single color

    Parameters
    ==========
    color_name - string name of color
    N - number of colors in colormap
    saturation - number of colors to add to the colormap
    start_color - color to start the colormap with
    is_transparent - whether to make the start color transparent
    is_reverse - whether to reverse the colormap
    Returns
    =======
    cmap - matplotlib colormap

    """
    cmap = ListedColormap([start_color, color_name])
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


def plot_vector(a, pt, *args, **kwargs):
    """Plots a 3D point as a vector from the origin"""
    plt_pts = np.vstack([np.array([0, 0, 0]), pt])
    return a.plot(plt_pts[:, 0], plt_pts[:, 1], plt_pts[:, 2], *args, **kwargs)
