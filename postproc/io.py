# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Input/output module to read bindary Fortran data of 2D and 3D fields with different
    number of components. Also useful to unpack text data in column-written.
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np

# Functions
def read_data(file, shape, **kwargs):
    """
    Return the velocity components of a velocity vector field stored in binary format.
    The data field is supposed to have been written as: (for k; for j; for i;) where the last dimension
    is the quickest varying index. Each record should have been written as: u, v, w.
    The return velocity components are always converted in np.double precision type.
    Args:
        - file: file to read from
        - shape: Shape of the data as (Nx,Ny) for 2D or (Nx,Ny,Nz) for 3D.
        - dtype: numpy dtype object. Single or double precision expected.
        - stream (depracated, use always stream output): type of access of the binary output. If false, there is a 4-byte header
            and footer around each "record" in the binary file (means +2 components at each record) (can happen in some
            Fortran compilers if access != 'stream').
        - periodic: If the user desires to make the data spanwise periodic (true) or not (false).
        - ncomponents: Specify the number of components. Default = ndims of the field
    """
    if not 'dtype' in kwargs:
        dtype = np.single
    else:
        dtype = kwargs['dtype']
    if not 'periodic' in kwargs:
        periodic = False
    else:
        periodic = kwargs['periodic']
    if not 'ncomponents' in kwargs:
        ncomponents = len(shape)
    else:
        ncomponents = kwargs['ncomponents']

    shape = tuple(reversed(shape))
    shape_comp = shape + (ncomponents,)

    f = open(file, 'rb')
    data = np.fromfile(file=f, dtype=dtype).reshape(shape_comp)
    f.close()

    if len(shape) == 2:
        if ncomponents == 1:
            u = data[:, :, 0].transpose(1, 0)
            u = u.astype(np.float64, copy=False)
            del data
            return u
        elif ncomponents == 2:
            u = data[:, :, 0].transpose(1, 0)
            v = data[:, :, 1].transpose(1, 0)
            del data
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            return u, v
        elif ncomponents == 3:
            u = data[:, :, 0].transpose(1, 0)
            v = data[:, :, 1].transpose(1, 0)
            w = data[:, :, 2].transpose(1, 0)
            del data
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            w = w.astype(np.float64, copy=False)
            return u, v, w
        else:
            raise ValueError("Number of components is not <=3")
    elif len(shape) == 3:
        if ncomponents == 1:
            u = data[:, :, :, 0].transpose(2, 1, 0)
            del data
            if periodic:
                u = np.dstack((u, u[:, :, 0]))
            u = u.astype(np.float64, copy=False)
            return u
        elif ncomponents == 2:
            u = data[:, :, :, 0].transpose(2, 1, 0)
            v = data[:, :, :, 1].transpose(2, 1, 0)
            del data
            if periodic:
                u = np.dstack((u, u[:, :, 0]))
                v = np.dstack((v, v[:, :, 0]))
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            return u, v
        elif ncomponents == 3:
            u = data[:, :, :, 0].transpose(2, 1, 0)
            v = data[:, :, :, 1].transpose(2, 1, 0)
            w = data[:, :, :, 2].transpose(2, 1, 0)
            del data
            if periodic:
                u = np.dstack((u, u[:, :, 0]))
                v = np.dstack((v, v[:, :, 0]))
                w = np.dstack((w, w[:, :, 0]))
            u = u.astype(np.float64, copy=False)
            v = v.astype(np.float64, copy=False)
            w = w.astype(np.float64, copy=False)
            return u, v, w
        else:
            raise ValueError("Number of components is not <=3")
    else:
        raise ValueError("Shape is not two- nor three-dimensional")


def unpack2Dforces(D, file):
    tD, dt, fx, fy, _, _ = np.loadtxt(file, unpack=True)  # 2D
    return tD*D, fx, fy


def unpack3Dforces(D, file):
    tD, dt, fx, fy, _, _, _, _ = np.loadtxt(file, unpack=True) #3D
    return tD*D, fx, fy


def unpackTimeSeries(npoints, file):
    if npoints == 1:
        t, p = np.loadtxt(file, unpack=True)  # 3D
        return t, p
    elif npoints == 2:
            t, p1, p2 = np.loadtxt(file, unpack=True)  # 3D
            return t, p1, p2
    elif npoints == 3:
            t, p1, p2, p3 = np.loadtxt(file, unpack=True)  # 3D
            return t, p1, p2, p3
    else:
        raise ValueError("Number of points is not <=3")
