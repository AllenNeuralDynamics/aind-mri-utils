# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:18:01 2023

@author: yoni.browning
"""
import numpy as np
import scipy.spatial.transform.rotation as rotation

def define_euler_rotation(rx,ry,rz,degrees = True):
    return rotation.Rotation.from_euler('xyz',[rx,ry,rz],degrees=True)

def rotate_about_and_translate(points,rotation,pivot,translation):
    return rotation.apply(points-pivot)-translation
    
def rotate_about(points,rototation,origin):
    return rotation.apply(points-origin)+origin
