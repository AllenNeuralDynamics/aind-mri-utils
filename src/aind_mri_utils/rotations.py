# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:18:01 2023

@author: yoni.browning
"""
import numpy as np
import scipy.spatial.transform.rotation as rotation

def define_euler_rotation(rx,ry,rz,degrees = True,order = 'xyz'):
    """
    wrapper on scipy.spatial.transform.rotation

    Parameters
    ----------
    rx : Scalar
        DESCRIPTION.
    ry : Scalar
        DESCRIPTION.
    rz : Scalar
        DESCRIPTION.
    degrees : Bool, optional
        Are the rotations in degrees?. The default is True.
    order: string,optional
        Order of axes to transform as a sorted string. Default is 'xyz'

    Returns
    -------
    Scipy 3d rotation
        scipy 3.

    """
    return rotation.Rotation.from_euler(order,[rx,ry,rz],degrees=True)

def rotate_about_and_translate(points,rotation,pivot,translation):
    '''
    Rotates points about a particular pivot point, then apply translation


    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rototation : Scipy rotation
        use "define_euler_rotation" to create
    pivot : (1x3) numpy array
        Point to rotate around
    translation: (1x3) numpy array
        Additonal translation to apply to points
        
    
    Returns
    -------
    (Nx3) numoy array
        Rotated points

    '''
    return rotate_about(points,rototation,pivot)-translation

def rotate_about(points,rototation,pivot):
    '''
    Rotates points about a particular pivot point

    Parameters
    ----------
    points : (Nx3) numpy array
        Points to rotate. Each point gets its own row.
    rototation : Scipy rotation
        use "define_euler_rotation" to create
    pivot : (1x3) numpy array
        Point to rotate around

    Returns
    -------
    (Nx3) numoy array
        Rotated points

    '''
    return rotation.apply(points-pivot)+pivot
