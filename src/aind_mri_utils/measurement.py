"""
Measurement code.


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
    """
    Fit a circle to a set of points using a linearized least-squares algorithm

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

def mask_segmented_voxels(full_vol, seg_vol, seg_vals):
    masked_vol = np.zeros_like(full_vol)
    mask = np.isin(seg_vol, seg_vals)
    masked_vol[mask] = full_vol[mask]
    return masked_vol

def find_mask_ndxs(seg_vol, v):
    return np.column_stack(np.nonzero(seg_vol == v))

def translate_ndxs(simage, index_arr):
    position_arr = np.zeros_like(index_arr, dtype="float32")
    npt = index_arr.shape[0]
    for ptno in range(npt):
        ndx = tuple(map(lambda x: x.item(), index_arr[ptno, :]))
        position_arr[ptno, :] = simage.TransformIndexToPhysicalPoint(ndx)
    return position_arr
