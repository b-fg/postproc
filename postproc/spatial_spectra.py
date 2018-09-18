# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module to compute the spatial spectra of a 2D field.
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np
import scipy.signal as signal

# Functions
def spatial_spectra(a, *args, **kwargs):
    dims = len(a.shape)
    if dims != len(args):
        raise ValueError('The field dimensions must much the spatial vector components passed to the function')

    k = wavenumbers(*args) # k = (kx, ky, kz)
    kspace = np.meshgrid(*k, indexing='ij') # kspace = (kx_grid, ky_grid, kz_grid)

    #windowing
    print(a.shape)
    # a = _nd_window(a)

    #fft
    ak = np.fft.fft2(a)/a.size
    # Pair every ak(k) with its k modulus
    ak_pairs = []
    k_mod_list = []
    for index in np.ndindex(ak.shape):
        ak_i = ak[index]
        k_i2_sum = 0
        for d in range(dims):
            k_i2_sum += kspace[d][index]**2
        k_i_mod = np.sqrt(k_i2_sum)
        ak_pairs.append((k_i_mod, ak_i))
        k_mod_list.append(k_i_mod)

    # Integrate all the modes in a bin of dk size
    k_res = kwargs.get('k_res', 200)
    k_mod_min = np.min(k_mod_list)
    k_mod_max = np.max(k_mod_list)
    dk = (k_mod_max-k_mod_min)/k_res
    k_line = np.linspace(k_mod_min, k_mod_max, k_res+1)  + dk/2 # k values at half of each bandwidth
    k_line = k_line[:-1]
    ak_line = np.zeros(k_line.size)
    for pair in ak_pairs:
        if pair[0] < k_line[0]: ak_line[0] += np.abs(pair[1])
        elif pair[0] > k_line[k_line.size-1]: ak_line[k_line.size-1] += np.abs(pair[1])
        else:
            for i in range(k_line.size):
                if abs(pair[0]-k_line[i]) <= dk/2:
                    ak_line[i] += np.abs(pair[1])
                    break
            else: # Executed if loop didnt break (could not find a bin)
                print(i, pair[0], k_line[i])
                print(abs(pair[0]-k_line[i])<=dk/2)
                raise ValueError('Could not find bin for value')
    return k_line, ak_line


def wavenumbers(*args):
    """
    Return the wavenumber vector for a position vector (1D, 2D or 3D) such as: x, y, z
    of a uniform grid defined in [xmax-xmin, ymax-ymin, zmax-zmin] containing [N, M, L] grid points.

    For the one- and two-dimensional cases, an axis needs to be specified e.g. wavenumbers(dim=2, axis=3)
    will return the two-dimensional wavenumber vector k = (kx, ky) of a 2D domain [xmax-xmin, ymax-ymin]
    """
    k = () # wavenumber vector; k = (kx, ky, kz), kx = 1D array type
    for arg in args:
        N = arg.size # number of points in the spatial vector component
        alpha = 2*np.pi/(np.min(arg)-np.max(arg)) # basic wavenumber
        index = np.fft.fftfreq(N, d=1/N) # index per wavenumber direction: eg x: 0,1,2,...,N/2-1,-N/2,-N/2+1,...,-1

        k_i = np.zeros(N) # wavenumber vector component k_i
        for i in range(0, N):
            k_i[i] = alpha*index[i]
        k = k + (k_i,)
    return k


def window(a):
    # w = signal.blackman(len(u_regular))
    w = signal.hanning(len(a))
    return a * w

def _nd_window(data):
    """
    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.

    Parameters
    ----------
    data : ndarray
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    """
    nx = data.shape[0]
    ny = data.shape[1]
    ax = (0,1)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        window = signal.blackman(axis_size)
        window = np.stack([window]*data.shape[axis-1], axis=ax[axis-1])
        # np.power(window, (1.0/data.ndim))
        data *= window

    return data