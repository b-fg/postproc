# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: CL CD St calculations
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np

# Internal functions
def avg(u):
    """
    Return the spanwise spatial average of a three-dimensional field.
    """
    return np.trapz(u, axis=2)/u.shape[2]


def decomp(u):
    """
    Return the average and the fluctuating parts of a three-dimensional field spatially averaged
    in the spanwise direction.
    """
    u_avg = avg(u)
    u_f = u-u_avg[:, :, None]
    return u_avg, u_f


def ddx(u):
    """
    Return the first-order derivative in the i direction of (n>=1 dimensional) field.
    """
    return np.gradient(u, axis=0, edge_order=2)


def ddy(u):
    """
    Return the first-order derivative in the j direction of (n>=2 dimensional) field.
    """
    return np.gradient(u, axis=1, edge_order=2)


def ddz(u, periodic):
    """
    Return the first-order derivative in the k direction of (n>=3 dimensional) field.
    The periodic argument is used to determine if the field is spanwise periodic o not.
    """
    N, M, L = u.shape[0], u.shape[1], u.shape[2]
    if periodic:
        u_temp = np.zeros((N, M, L+3))
        u_temp[:, :, 1:L+2], u_temp[:, :, 0], u_temp[:, :, L+2] = u, u[:, :, L-1], u[:, :, 1]
        del u
        dudz = np.gradient(u_temp, axis=2, edge_order=2)
        del u_temp
        dudz = dudz[:, :, 1:L+2]
        return dudz
    else:
        return np.gradient(u, axis=2, edge_order=2)

def vortz(u, v):
    """
    Return the z-vorticity of a two-dimensional velocity fild
    """
    if not (len(u.shape)==2 and len(v.shape)==2):
        raise ValueError("Fields must be two-dimensional")
    else:
        return ddx(v)-ddy(u)

def vort(u, v, w, periodic):
    """
    Return the vorticity of a three-dimensional velocity fild
    """
    if not (len(u.shape)==3 and len(v.shape)==3 and len(w.shape)==3):
        raise ValueError("Fields must be three-dimensional")
    else:
        return ddy(w)-ddz(v, periodic), ddz(u, periodic)-ddx(w), ddx(v)-ddy(u)