"""Measurement code.


This applies to find_circle_center, which is borrowed from scipy-cookbooks:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2017 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np


def find_circle(x, y):
    """Fit a circle to a set of points

    Fit a circle to a set of points using a linearized least-squares
    algorithm

    Borrowed, with modification, from:
    https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    Parameters
    ----------
    x : (N) array
        X values of sample points.
    y : (N) array
        Y values of sample points.

    Returns
    -------
    xc_1 : Float
        X coordinate of center
    yc_1 : Float
        Y coordinate of center.
    radius : Float
        Radius of fit circle.
    residu_1 : (N) array
        per-point residual.

    """
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # We will find the center (uc, vc) by solving the following
    # linear system.
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    # Set up:
    Suv = sum(u * v)
    Suu = sum(u**2)
    Svv = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3)

    # And Solve!
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    #
    xc_1 = x_m + uc
    yc_1 = y_m + vc

    Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    radius = np.mean(Ri_1)

    return xc_1, yc_1, radius


def find_line_eig(points):
    """
    Returns first normalized eigenvector of data, for use in line fitting.

    Parameters
    ----------
    points : NxD numpy array
        Points to fit a line through

    Returns
    -------
    (D,) numpy array
        norm of line (eigenvector)
    points_mean : (D,)
        Average value

    """
    points_mean = np.mean(points, axis=0)
    a, b = np.linalg.eig(np.cov((points - points_mean).T))
    return b[:, 0], points_mean


def closet_points_on_two_lines(P1, V1, P2, V2):
    """Calculate the closest points on two lines in 3D space.

    Parameters
    ----------
    P1 : array-like
        A point on the first line.
    V1 : array-like
        The direction vector of the first line.
    P2 : array-like
        A point on the second line.
    V2 : array-like
        The direction vector of the second line.

    Returns
    -------
    p_a : ndarray
        The closest point on the first line.
    p_b : ndarray
        The closest point on the second line.
    """
    P1 = np.array(P1)
    V1 = np.array(V1)
    P2 = np.array(P2)
    V2 = np.array(V2)
    V21 = P2 - P1

    v22 = np.dot(V2, V2)
    v11 = np.dot(V1, V1)
    v21 = np.dot(V2, V1)
    v21_1 = np.dot(V21, V1)
    v21_2 = np.dot(V21, V2)
    denom = v21 * v21 - v22 * v11

    if np.isclose(denom, 0.0):
        s = 0.0
        t = (v11 * s - v21_1) / v21
    else:
        s = (v21_2 * v21 - v22 * v21_1) / denom
        t = (-v21_1 * v21 + v11 * v21_2) / denom

    p_a = P1 + s * V1
    p_b = P2 + t * V2
    return p_a, p_b


def angle(v1, v2):
    """
    Angle (in degrees) between two vectors

    Parameters
    ----------
    v1 : numpy array
        First Vector.
    v2 : numpy array
        Second Vector.

    Returns
    -------
    Angle between vectors
    """
    rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.rad2deg(rad)


def dist_point_to_line(pt_1, pt_2, query_pt):
    """Distance between line defined by two points and a query point

    inspiration from:
        https://stackoverflow.com/questions/39840030/...
        distance-between-point-and-a-line-from-two-points

    Parameters
    ----------
    pt_1 : numpy array  (N,)
        First Vector.
    pt_2 : numpy array (N,)
        Second Vector.
    query_pt: numpy array (N,)
        Point to find distance of.

    Returns
    -------
    Distance
    """
    ln_pt = pt_1
    ln_norm = pt_1 - pt_2
    ln_norm = ln_norm / np.linalg.norm(ln_norm)
    return np.abs(
        np.linalg.norm(np.cross(ln_norm, ln_pt - query_pt))
    ) / np.linalg.norm(ln_norm)


def dist_point_to_plane(pt_0, normal, query_pt):
    """
    Distance between plane defined by point and normal and a query point

    Parameters
    ----------
    pt_0 : numpy array  (N,)
        Point on plane.
    normal : numpy array (N,)
        Normal vector of plane.
    query_pt: numpy array (N,)
        Point to find distance of.

    Returns
    -------
    Distance
    """
    D = -normal[0] * pt_0[0] - normal[1] * pt_0[1] - normal[2] * pt_0[2]
    num = np.abs(
        normal[0] * query_pt[0]
        + normal[1] * query_pt[1]
        + normal[2] * query_pt[2]
        + D
    )
    denom = np.sqrt(np.sum(normal**2))
    return num / denom
