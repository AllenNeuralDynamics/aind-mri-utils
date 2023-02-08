# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:41:08 2023

@author: yoni.browning
"""
import numpy as np


def find_circle_center(x, y):
    """
    Use least squares to find center of a set of points

    Parameters
    ----------
    x : (N) array
        DESCRIPTION.
    y : (N) array
        DESCRIPTION.

    Returns
    -------
    xc_1 : Scalar
        X coordinate of center
    yc_1 : Scalar
        Y coordinate of center.
    radius : Scalar
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

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = sum(u * v)
    Suu = sum(u**2)
    Svv = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    # R_1      = np.mean(Ri_1)
    # residu_1 = sum((Ri_1-R_1)**2)
    radius = np.mean(Ri_1)

    return xc_1, yc_1, radius
