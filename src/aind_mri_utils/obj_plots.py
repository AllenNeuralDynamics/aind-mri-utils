import numpy as np
import pywavefront
from matplotlib import pyplot as plt
from matplotlib import tri as mpt


def plot_tri_mesh(ax, vertices, faces, *plot_args, **plot_kwargs):
    """
    Creates a figure showing the translation between
    source and target landmarks

    Parameters
    ==========
    source_landmarks - np.ndarray (N x 3)
    target_landmarks - np.ndarray (N x 3)
    volume_size - list of x, y, z max dimensions

    Returns
    =======
    fig - figure handle

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


def get_vertices_and_faces(scene):
    """
    Collect vertices and faces for a pywavefront object

    Parameters
    ==========
    scene - a Wavefront object made with `collect_materials=True`

    Returns
    =======
    vertices - np.ndarray (floating point N x 3) array of N 3d points
    faces - np.ndarray (int M x 3) array of M triangles, where each element
            corresponds to the index of a vertex in `vertices`
    """
    face_list = []
    for mesh in scene.mesh_list:
        face_list = face_list + mesh.faces
    vertices = np.array(scene.vertices)
    faces = np.array(face_list)
    return vertices, faces


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
