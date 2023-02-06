# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:00:42 2023

@author: yoni.browning
"""

def transform_image_index_to_resample_index(image_index,image,resampled_image,transform):
    """
    Transform from image index to resample image index

    Parameters
    ----------
    image_index : (Nx1) array 
        index coordinate in image
    image : SITK Image
        Origional Image
    resampled_image : SITK Image
        Resampled Image
    transform : SITK transform
        Mapping between image and resampled image.

    Returns
    -------
    (Nx1) array 
        Index in resampled image

    """
    raise NotImplementedError("Function needs testing; don't trust yet.")
    
    image_point = image.TransformContinuousIndexToPhysicalPoint(image_index)
    resample_point = transform.TransformPoint(image_point)
    return resampled_image.TransformPhyicalPointToIndex(resample_point)

def transform_resample_index_to_image_index(resample_index,image,resampled_image,transform):
    """
    Transfrom from resampled image index to image index

    Parameters
    ----------
    resample_index : (Nx1) array 
        index coordinate in resampled imate.
    image : SITK Image
        Origional Image
    resampled_image : SITK Image
        Resampled Image
    transform : SITK transform
        Mapping between image and resampled image.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    raise NotImplementedError("Function needs testing; don't trust yet.")

    resample_point = resampled_image.TransformContinuousIndexToPhysicalPoint(resample_index)
    image_point = transform.GetInverse().TransformPoint(resample_point)
    return image.TransformPhyicalPointToIndex(image_point)

def transform_resample_point_to_image_index(resample_point,image,transform):
    """
    transform from physical point in resampled image to index in origional image

    Parameters
    ----------
    resample_point : (Nx1) Array
        Physical point coordinate in reampled image.
    image : SITK Image
        Origional Image
    transform : SITK transform
        Mapping between image and resampled image.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    raise NotImplementedError("Function needs testing; don't trust yet.")

    # CHECK IF THIS IS BACKWARDS??
    image_point = transform.GetInverse().TransformPoint(resample_point)
    return image.TransformPhyicalPointToIndex(image_point)
    
def transform_image_point_to_resample_index(image_point,resampled_image,transform):
    """
    transform from physical point in image to index in resampled image

    Parameters
    ----------
    image_point : (Nx1) array
        Physical point coordinate in image
    resampled_image : SITK Image
        Resampled Image
    transform : SITK transform
        Mapping between image and resampled image.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    raise NotImplementedError("Function needs testing; don't trust yet.")

    resample_point = transform.TransformPoint(image_point)
    return resampled_image.TransformPhyicalPointToIndex(resample_point)
    
def transform_image_point_to_resample_point(image_point,transform):
    """
    transform from physical point in image to physical point in resampled image


    Parameters
    ----------
    image_point : (Nx1) Array
        Physical point coordinate in image.
    transform : SITK transform
        Mapping between image and resampled image.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    raise NotImplementedError("Function needs testing; don't trust yet.")

    return transform.TransformPoint(image_point)
    
def transform_resample_point_to_image_point(resample_point,transform):
    """
    transform from physical point inresampled image to physical point in image


    Parameters
    ----------
    resample_point : (Nx1) Array
        Physical point coordinate in reampled image.
    transform : SITK transform
        Mapping between image and resampled image.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    raise NotImplementedError("Function needs testing; don't trust yet.")

    return transform.GetInverse().TransformPoint(resample_point,transform)

def identity_transform(N = 3):
    return sitk.AffineTransform(N)