# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module to compute the spatial spectra of a 2D field.
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np
import scipy.signal as signal
import time
from tqdm import tqdm

# Functions
def tke_from_ui(u_i, x_i, **kwargs):
    if len(u_i) > 3 or len(u_i) < 1 or len(x_i) > 3 or len(x_i) < 1:
        raise ValueError('Invalid field dimensions')
    elif len(u_i) != len(x_i):
        raise ValueError('Field dimensions must much the spatial vector components passed to the function')
    # Wavenumbers
    k_i = wavenumbers(*x_i) # kvec = (kx, ky, kz)
    # FFT to compute TKE
    tke = 0
    for u in u_i:
        # u = window_ndim(u) # Windowing
        uk_i = np.fft.fftn(u) # FFT
        tke += uk_i*uk_i.conjugate() #TKE
    tke = 0.5*tke
    # Calc spectra
    return pair_integrate_fast(tke, *k_i, **kwargs)

def spatial_spectra_slow(a, *args, **kwargs):
    dims = len(a.shape)
    if dims != len(args):
        raise ValueError('The field dimensions must much the spatial vector components passed to the function')
    k = wavenumbers(*args) # k = (kx, ky, kz)
    kspace = np.meshgrid(*k, indexing='ij') # kspace = (kx_grid, ky_grid, kz_grid)

    # Windowing
    # a = window_ndim(a)
    # FFT
    ak = np.fft.fft2(a)/a.size
    # Calc spectra
    print('    Pair every ak(k) with its k modulus')
    ak_pairs, k_mod_list = ak_k_list(ak, kspace)
    print('    Integrate all the modes in a bin of dk size')
    k_res = kwargs.get('k_res', 200)
    k_line, ak_line = ak_k_integrate_slow(ak_pairs, k_mod_list, k_res)
    return k_line, ak_line

def spatial_spectra_fast(a, *args, **kwargs):
    dims = len(a.shape)
    if dims != len(args):
        raise ValueError('The field dimensions must much the spatial vector components passed to the function')
    k = wavenumbers(*args) # k = (kx, ky, kz)
    kspace = np.meshgrid(*k, indexing='ij') # kspace = (kx_grid, ky_grid, kz_grid)
    # Windowing
    # a = window_ndim(a)
    # FFT
    ak = np.fft.fft2(a)/a.size
    # Calc spectra
    print('    Pair every ak(k) with its k modulus')
    ak_pairs, k_mod_list = ak_k_list(ak, kspace)
    print('    Integrate all the modes in a bin of dk size')
    k_res = kwargs.get('k_res', 200)
    k_line, ak_line = ak_k_integrate_fast(ak_pairs, k_mod_list, k_res)
    return k_line, ak_line


def spatial_spectra_superfast(a, *args, **kwargs):
    if a.ndim > 3 or a.ndim < 1:
        raise ValueError('Invalid field dimensions')
    elif a.ndim != len(args):
        raise ValueError('Field dimensions must much the spatial vector components passed to the function')

    k = wavenumbers(*args) # k = (kx, ky, kz)

    # Windowing
    # a = window_ndim(a)

    # FFT
    ak = np.fft.fftn(a)

    # Calc spectra
    return pair_integrate_fast(ak, *k, **kwargs)


def ak_k_list(ak, kspace):
    ak_pairs = []
    k_mod_list = []
    for index in np.ndindex(ak.shape):
        ak_i = ak[index]
        k_i2_sum = 0
        for d in range(ak.ndim):
            k_i2_sum += kspace[d][index] ** 2
        k_i_mod = np.sqrt(k_i2_sum)
        ak_pairs.append((k_i_mod, ak_i))
        k_mod_list.append(k_i_mod)
    return ak_pairs, k_mod_list

def pair_integrate_fast(ak, *args, **kwargs):
    k_res = kwargs.get('k_res', 200)
    k_i2_sum_max = 0
    k_i2_sum_min = 0
    for k_i in args:
        k_i2_sum_max += np.max(k_i**2)
        k_i2_sum_min += np.min(k_i**2)
    kmin, kmax = np.sqrt(k_i2_sum_min), np.sqrt(k_i2_sum_max)

    dk = (kmax - kmin) / k_res

    ak_line = np.zeros(k_res)
    k_line = np.linspace(0, k_res - 1, k_res) * dk + dk / 2  # k values at half of each bandwidth

    print('    Find ak(k_i) with its k modulus and integrate it in a(k)')
    with tqdm(total=ak.size) as pbar:
        for index in np.ndindex(ak.shape):
            ak_i = ak[index]
            k_i2_sum = 0
            for i, k_i in enumerate(args):
                k_i2_sum += k_i[index[i]] ** 2
            k_i_mod = np.sqrt(k_i2_sum)

            kint = int(k_i_mod / dk)
            if kint >= k_res:
                ak_line[-1] += np.abs(ak_i)
            else:
                ak_line[kint] += np.abs(ak_i)
            pbar.update(1)

    return k_line, ak_line


def ak_k_integrate_slow(ak_pairs, k_mod_list, k_res):
    k_mod_min = np.min(k_mod_list)
    k_mod_max = np.max(k_mod_list)
    dk = (k_mod_max - k_mod_min) / k_res
    k_line = np.linspace(k_mod_min, k_mod_max, k_res + 1) + dk / 2  # k values at half of each bandwidth
    k_line = k_line[:-1]
    ak_line = np.zeros(k_line.size)
    for pair in ak_pairs:
        if pair[0] < k_line[0]:
            ak_line[0] += np.abs(pair[1])
        elif pair[0] > k_line[k_line.size - 1]:
            ak_line[k_line.size - 1] += np.abs(pair[1])
        else:
            for i in range(k_line.size):
                if abs(pair[0] - k_line[i]) <= dk / 2:
                    ak_line[i] += np.abs(pair[1])
                    break
            else:  # Executed if loop didnt break (could not find a bin)
                print(i, pair[0], k_line[i])
                print(abs(pair[0] - k_line[i]) <= dk / 2)
                raise ValueError('Could not find bin for value')
    return k_line, ak_line


def ak_k_integrate_fast(ak_pairs, k_mod_list, k_res):
    kmin, kmax = np.min(k_mod_list), np.max(k_mod_list)
    dk = (kmax - kmin) / k_res
    ak_line = np.zeros(k_res)
    k_line = np.linspace(0, k_res - 1, k_res) * dk + dk / 2  # k values at half of each bandwidth
    for pair in ak_pairs:
        kmod = pair[0]
        kint = int(kmod / dk)
        if kint >= k_res:
            ak_line[-1] += np.abs(pair[1])
        else:
            ak_line[kint] += np.abs(pair[1])
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
        alpha = 2*np.pi/(np.max(arg)-np.min(arg)) # basic wavenumber
        index = np.fft.fftfreq(N, d=1/N) # index per wavenumber direction: eg x: 0,1,2,...,N/2-1,-N/2,-N/2+1,...,-1

        k_i = np.zeros(N) # wavenumber vector component k_i
        for i in range(0, N):
            k_i[i] = alpha*index[i]
        k = k + (k_i,)
    return k


def window_ndim(a):
    """
    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.

    Parameters
    ----------
    a : ndarray
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    """
    wfunction = signal.blackman

    if a.ndim == 1:
        return a * wfunction(len(a))
    else:
        axis_idxs = np.arange(len(a.shape))
        for axis, axis_size in enumerate(a.shape):
            window = wfunction(axis_size)
            window = np.stack([window]*a.shape[axis-1], axis=axis_idxs[axis-1])
            # np.power(window, (1.0/a.ndim))
            a *= window
        return a