"""Plotting functions"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mpt
from matplotlib.colors import ListedColormap

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray


def plot_tri_mesh(
    ax: Any,
    vertices: NDArray[np.floating[Any]],
    faces: NDArray[np.integer[Any]],
    *plot_args: Any,
    **plot_kwargs: Any,
) -> tuple[Any, Any]:
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
def _set_axes_radius(
    ax: Any, origin: NDArray[np.floating[Any]], radius: float
) -> None:
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
def set_axes_equal(ax: Axes3D) -> None:
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


def make_3d_ax_look_normal(ax: Axes3D) -> None:
    """
    Changes the aspect ratio of a 3d plot so that dimensions are approximately
    the same size

    Parameters
    ==========
    ax - matplotlib 3d axis
    """
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)


def get_prop_cycle() -> Any:
    """
    Returns the colors in the current prop cycle
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    return prop_cycle.by_key()["color"]


def plot_point_cloud_3d(
    a: Any, pts: NDArray[np.floating[Any]], *args: Any, **kwargs: Any
) -> Any:
    """Scatter plots a Nx3 array as N 3D points"""
    return a.scatter(pts[:, 0], pts[:, 1], pts[:, 2], *args, **kwargs)


def create_single_colormap(
    color_name: str,
    N: int = 256,
    saturation: float = 0,
    start_color: str = "white",
    is_transparent: bool = True,
    is_reverse: bool = False,
) -> ListedColormap:
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
    start_color_array = np.array(cmap(0))
    if is_transparent:
        start_color_array[-1] = 0
    if not is_reverse:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.linspace(start_color_array, cmap(1), N),
                    np.tile(cmap(1), (int(saturation * N), 1)),
                )
            )
        )
    else:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.tile(cmap(1), (int(saturation * N), 1)),
                    np.linspace(cmap(1), start_color_array, N),
                )
            )
        )
    return cmap


def plot_vector(
    a: Any, pt: NDArray[np.floating[Any]], *args: Any, **kwargs: Any
) -> Any:
    """Plots a 3D point as a vector from the origin"""
    plt_pts = np.vstack([np.array([0, 0, 0]), pt])
    return a.plot(plt_pts[:, 0], plt_pts[:, 1], plt_pts[:, 2], *args, **kwargs)


def rgb_to_int(r: int, g: int, b: int, a: int | None = None) -> int:
    """Converts an RGB color to an integer.

    Parameters
    ----------
    r : int
        Red component of the color, must be in the range [0, 255].
    g : int
        Green component of the color, must be in the range [0, 255].
    b : int
        Blue component of the color, must be in the range [0, 255].
    a : int, optional
        Alpha component of the color, must be in the range [0, 255].
        If None, only RGB is used.

    Returns
    -------
    int
        The integer representation of the RGB color.

    Raises
    ------
    ValueError
        If any of the RGB values are not in the range [0, 255].

    Examples
    --------
    >>> rgb_2_int(255, 0, 0)
    16711680
    >>> rgb_2_int(0, 255, 0)
    65280
    >>> rgb_2_int(0, 0, 255)
    255

    """
    comps = (b, g, r) if a is None else (b, g, r, a)
    out = 0

    for i, v in enumerate(comps):
        if v < 0 or v > 255:
            raise ValueError(
                f"RGB values must be in the range [0, 255], got {v}"
            )
        out |= v << (8 * i)
    return out


def int_to_rgb(
    color_int: int, has_alpha: bool | None = None
) -> tuple[int, int, int] | tuple[int, int, int, int]:
    """Converts an integer color representation to an RGB color.

    Parameters
    ----------
    color_int : int
        The integer representation of the color.
    has_alpha : bool, optional
        If True, returns RGBA. If False, returns RGB. If None,
        automatically detects based on whether the integer uses
        more than 24 bits.

    Returns
    -------
    tuple
        A tuple (r, g, b) representing the RGB color.

    Examples
    --------
    >>> int_to_rgb(16711680)
    (255, 0, 0)
    >>> int_to_rgb(2164195328)
    (255, 0, 0, 128)
    >>> int_to_rgb(65280)
    (0, 255, 0)
    >>> int_to_rgb(255)
    (0, 0, 255)
    """

    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    # Auto-detect alpha if not specified
    if has_alpha is None:
        has_alpha = color_int > 0xFFFFFF

    if has_alpha:
        a = (color_int >> 24) & 0xFF
        return (r, g, b, a)
    else:
        return (r, g, b)


def rgb_to_hex_string(r: int, g: int, b: int, a: int | None = None) -> str:
    """Converts an RGB color to a hex string.

    Parameters
    ----------
    r : int
        Red component of the color (0-255).
    g : int
        Green component of the color (0-255).
    b : int
        Blue component of the color (0-255).
    a : int, optional
        Alpha component of the color (0-255).
        If None, only RGB is used.


    Returns
    -------
    str
        Hexadecimal string representation of the color.

    Examples
    --------
    >>> rgb_to_hex_string(255, 0, 0)
    '0xFF0000'
    >>> rgb_to_hex_string(255, 0, 0, 128)
    '0x80FF0000'
    """
    color_int = rgb_to_int(r, g, b, a)
    if a is None:
        return f"0x{color_int:06X}"
    else:
        return f"0x{color_int:08X}"


def hex_string_to_int(hx: str) -> int:
    """Converts a hex color string to an integer.

    Parameters
    ----------
    hx : str
        Hexadecimal string representation of the color.

    Returns
    -------
    int
        The integer representation of the hex color.

    Examples
    --------
    >>> hex_string_2_int("#FF0000")
    16711680
    >>> hex_string_2_int("0x00FF00")
    65280
    >>> hex_string_2_int("0000FF")
    255

    """
    return int(hx.lstrip("#"), 16)


def hex_string_to_rgb(
    hx: str,
) -> tuple[int, int, int] | tuple[int, int, int, int]:
    """Converts a hex color string to an RGB or RGBA color.

    Parameters
    ----------
    hx : str
        Hexadecimal string representation of the color.
        Supports both RGB (6 chars) and RGBA (8 chars) formats.

    Returns
    -------
    tuple
        A tuple (r, g, b) for RGB or (r, g, b, a) for RGBA color.

    Examples
    --------
    >>> hex_string_to_rgb("#FF0000")
    (255, 0, 0)
    >>> hex_string_to_rgb("0x80FF0000")
    (255, 0, 0, 128)
    >>> hex_string_to_rgb("0x00FF00")
    (0, 255, 0)
    >>> hex_string_to_rgb("0000FF")
    (0, 0, 255)
    """
    color_int = hex_string_to_int(hx)

    # Determine if this is RGBA based on hex string length
    clean_hex = hx.lstrip("#").lstrip("0x")
    has_alpha = len(clean_hex) > 6

    return int_to_rgb(color_int, has_alpha)
