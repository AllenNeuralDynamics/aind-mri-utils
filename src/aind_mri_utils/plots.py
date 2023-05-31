"""Plotting functions"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mpt


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


def get_prop_cycle():  # pragma: no cover
    """
    Returns the colors in the current prop cycle
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    return prop_cycle.by_key()["color"]
