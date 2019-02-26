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
    :param file: file to read from.
    :param shape: Shape of the data as (Nx,Ny) for 2D or (Nx,Ny,Nz) for 3D.
    :param kwargs:
        dtype: numpy dtype object. Single or double precision expected.
        stream (depracated, use always stream output): type of access of the binary output. If false, there is a 4-byte header
            and footer around each "record" in the binary file (means +2 components at each record) (can happen in some
            Fortran compilers if access != 'stream').
        periodic: If the user desires to make the data spanwise periodic (true) or not (false).
        ncomponents: Specify the number of components. Default = ndims of the field
    :return: the components of the vector or scalar field.
    """
    dtype = kwargs.get('dtype', np.single)
    periodic = kwargs.get('periodic', False)
    ncomponents = kwargs.get('ncomponents', len(shape))

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


def read_data_raw(file, shape, ncomponents):
    shape = tuple(reversed(shape))
    shape_comp = shape + (ncomponents,)

    f = open(file, 'rb')
    data = np.fromfile(file=f, dtype=np.single).reshape(shape_comp)
    f.close()

    return data


def read_and_write_fractioned_mean_data(f_w_list, shape, **kwargs):
    """
    Computes a weighted average of files which containg partial averages of quantities.
    :param f_w_list: list of tuples containing (file, weight).
    :param shape: Shape of the data as (Nx,Ny) for 2D or (Nx,Ny,Nz) for 3D.
    :param kwargs:
    :param kwargs:
        dtype: numpy dtype object. Single or double precision expected.
        stream (depracated, use always stream output): type of access of the binary output. If false, there is a 4-byte header
            and footer around each "record" in the binary file (means +2 components at each record) (can happen in some
            Fortran compilers if access != 'stream').
        periodic: If the user desires to make the data spanwise periodic (true) or not (false).
        ncomponents: Specify the number of components. Default = ndims of the field
    :return: The weighted average of the files containing partial averages.
    """
    dtype = kwargs.get('dtype', np.single)
    ncomponents = kwargs.get('ncomponents', len(shape))
    if ncomponents == 1:
        aw_tot = 0
        w_tot = 0
        for tup in f_w_list:
            file = tup[0]
            w = tup[1]
            data = read_data_raw(file, shape, ncomponents)
            if len(shape) == 2:
                a = data[:, :, 0]
            elif len(shape) == 3:
                a = data[:, :, :, 0]
            aw_tot += a*w
            w_tot += w

        a_mean = aw_tot/w_tot
        a_mean = a_mean.astype(np.float64, copy=False)

        a_mean.tofile(file[:-4]+'_python_mean.dat')
        return

    elif ncomponents == 2:
        aw_tot = 0
        bw_tot = 0
        w_tot = 0
        for tup in f_w_list:
            file = tup[0]
            w = tup[1]
            data = read_data_raw(file, shape, ncomponents)
            if len(shape) == 2:
                a = data[:, :, 0]
                b = data[:, :, 1]
            elif len(shape) == 3:
                a = data[:, :, :, 0]
                b = data[:, :, :, 1]
            aw_tot += a*w
            bw_tot += b*w
            w_tot += w

        a_mean, b_mean = aw_tot/w_tot, bw_tot/w_tot
        a_mean = a_mean.astype(np.float64, copy=False)
        b_mean = b_mean.astype(np.float64, copy=False)
        r_mean = np.stack((a_mean, b_mean), axis=-1)
        r_mean.tofile(file[:-4]+'_python_mean.dat')
        return

    elif ncomponents == 3:
        aw_tot = 0
        bw_tot = 0
        cw_tot = 0
        w_tot = 0
        for tup in f_w_list:
            file = tup[0]
            w = tup[1]
            data = read_data_raw(file, shape, ncomponents)
            if len(shape) == 2:
                a = data[:, :, 0]
                b = data[:, :, 1]
                c = data[:, :, 2]
            elif len(shape) == 3:
                a = data[:, :, :, 0]
                b = data[:, :, :, 1]
                c = data[:, :, :, 2]
            aw_tot += a * w
            bw_tot += b * w
            cw_tot += c * w
            w_tot += w

        a_mean, b_mean, c_mean =  aw_tot/w_tot, bw_tot/w_tot, cw_tot/w_tot
        a_mean = a_mean.astype(np.float64, copy=False)
        b_mean = b_mean.astype(np.float64, copy=False)
        c_mean = c_mean.astype(np.float64, copy=False)
        r_mean = np.stack((a_mean, b_mean, c_mean), axis=-1)

        f_root = '/'.join(file.split('/')[0:-3])+'/'
        f_name = file.split('/')[-1][:-4]
        r_mean.tofile(f_root + 'output/' +f_name + '_python_mean.dat')
        return
    else:
         raise ValueError("Number of components is not <=3")


def unpack2Dforces(file, D, U=1):
    """
    Unpacks ASCII files coontaining the following columns: non-dimensional time, time step, fx, fy, vx, vy.
    fx, fy are the pressure forces and vx, vy the viscous forces (not returned with this function).
    :param file: File to read from.
    :param D: Characteristic length.
    :param U: characteristic velocity.
    :return: time, fx, fy.
    """
    tD, dt, fx, fy, _, _ = np.loadtxt(file, unpack=True)  # 2D
    return tD*D/U, fx, fy


def unpack3Dforces(file, D, U=1):
    """
    Unpacks ASCII files containing the following columns: non-dimensional time, time step, fx, fy, fz, vx, vy, vz.
    fx, fy, fz are the pressure forces and vx, vy, vz the viscous forces (not returned with this function).
    :param file: File to read from.
    :param D: Characteristic length.
    :param U: characteristic velocity.
    :return: time, fx, fy.
    """
    tD, dt, fx, fy, _, _, _, _ = np.loadtxt(file, unpack=True) #3D
    return tD*D/U, fx, fy


def unpackTimeSeries(file, npoints):
    """
    Unpacks ASCII files containing the following columns: non-dimensional time, point1, point2, ...
    :param file:
    :param npoints: number of points recorded in the file
    :return: time, point1, point2, ...
    """
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

def readTimeSeries(file):
    """
    Reads ASCII files containing the following columns: non-dimensional time, point1, point2, ...
    :param file:
    :return: 2D numpy array. Each column is a time series. Normally time is the first column. t = a[:,0]
    """
    return np.loadtxt(file)

def readPressureArcLength(file):
    p, L = np.loadtxt(file, unpack=True, skiprows=1, delimiter=',')
    return p, L

def importExpCpTheta(file):
    theta, cp = np.loadtxt(file, unpack=True, delimiter=',')
    return theta, cp
