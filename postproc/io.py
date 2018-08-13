# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: CL CD St calculations
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np

# Internal functions
def read_data(file, shape, dtype, stream, periodic):
    """
    Return the velocity components of a velocity vector field stored in binary format.
    The data field is supposed to have been written as: (for k; for j; for i;) where the last dimension
    is the quickest varying index. Each record should have been written as: u, v, w.
    The return velocity components are always converted in np.double precision type.
    Args:
        file: file to read from
        shape: Shape of the data as (Nx,Ny) for 2D or (Nx,Ny,Nz) for 3D.
        dtype: numpy dtype object. Single or double precision expected.
        stream: type of access of the binary output. If true, only a pure binary output of the velocity
            vector field is assumed. If false, there is a 4-byte header and footer around each "record"
            in the binary file (can happen in some Fortran compilers if access != 'stream').
        periodic: If the user desires to make the data spanwise periodic (true) or not (false).
    """
    shape = tuple(reversed(shape))
    if len(shape) == 2:
        if stream:
            shape_rec = shape + (2,)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape_rec)
            f.close()
            u = data[:, :, 0].transpose(1, 0)
            v = data[:, :, 1].transpose(1, 0)
            del data
        else:
            shape_rec = shape + (4,)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape_rec)
            f.close()
            u = data[:, :, 1].transpose(1, 0)
            v = data[:, :, 2].transpose(1, 0)
            del data
        u = u.astype(np.float64, copy=False)
        v = v.astype(np.float64, copy=False)
        return u, v
    if len(shape) == 3:
        if stream:
            shape_rec = shape + (3,)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape_rec)
            f.close()
            u = data[:, :, :, 0].transpose(2, 1, 0)
            v = data[:, :, :, 1].transpose(2, 1, 0)
            w = data[:, :, :, 2].transpose(2, 1, 0)
            del data
        else:
            shape_rec = shape + (5,)
            f = open(file, 'rb')
            data = np.fromfile(file=f, dtype=dtype).reshape(shape_rec)
            f.close()
            u = data[:, :, :, 1].transpose(2, 1, 0)
            v = data[:, :, :, 2].transpose(2, 1, 0)
            w = data[:, :, :, 3].transpose(2, 1, 0)
            del data
        # Make the data periodic
        if periodic:
            u = np.dstack((u, u[:, :, 0]))
            v = np.dstack((v, v[:, :, 0]))
            w = np.dstack((w, w[:, :, 0]))
        # Convert to double precision
        u = u.astype(np.float64, copy=False)
        v = v.astype(np.float64, copy=False)
        w = w.astype(np.float64, copy=False)
        return u, v, w
    else:
        return -1

def unpack2Dforces(D, file):
    tD, dt, fx, fy, _, _ = np.loadtxt(file, unpack=True)  # 2D
    return tD*D, fx, fy

def unpack3Dforces(D, file):
    tD, dt, fx, fy, _, _, _, _ = np.loadtxt(file, unpack=True) #3D
    return tD*D, fx, fy