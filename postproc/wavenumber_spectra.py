# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module to compute the wavenumber spectra of a n-dimensional field.
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np
import scipy.signal as signal
import time
from tqdm import tqdm

# Functions
def ke_from_ui(u_i, x_i, **kwargs):
    """
    Computes the kinetic energy (KE) of a n-dimensional velocity vector field [u_i = (u, v, w) for 3D]. If the velocity
    vector field passed is the normal Reynolds stresses r_i = (<u'u'>, <v'v'>, <w'w'>) the result is the turbulent
    kinetic energy (TKE).
    :param u_i: n-dimensional velocity vector field
    :param x_i: n-dimensional spatial vector field
    :param kwargs: k_res for the resolution of k_mod_line which defines the bandwitdh dk.
    :return: k_mod 1D array and ke_integral 1D array.
    """
    if len(u_i) > 3 or len(u_i) < 1 or len(x_i) > 3 or len(x_i) < 1:
        raise ValueError('Invalid field dimensions')
    elif len(u_i) != len(x_i):
        raise ValueError('Field dimensions must much the spatial vector components passed to the function')
    # Wavenumbers
    k_i = _wavenumbers(*x_i) # kvec = (kx, ky, kz)
    # FFT to compute TKE
    tke = 0
    for u in u_i:
        # u = window_ndim(u) # Windowing
        uk_i = np.fft.fftn(u) # FFT
        tke += uk_i*uk_i.conjugate() #TKE
    tke = 0.5*tke
    # Calc spectra
    return _pair_integrate_fast(tke, *k_i, **kwargs)


def wn_spectra(a, *args, **kwargs):
    """
    Computes the wavenumber spectra of a n-dimensional scalar field by means of nFFT. The spectra is for the wavenumber
    modulus of ak. Hence it performs and spherical integration for all the components of k_i (the wavenumber vector)
    :param a: The scalar field
    :param args: The spatial vector x_i = (x, y, z)
    :param kwargs: k_res for the resolution of k_mod_line which defines the bandwitdh dk.
    :return: k_mod 1D array and ak_integral 1D array.
    """
    if a.ndim > 3 or a.ndim < 1:
        raise ValueError('Invalid field dimensions')
    elif a.ndim != len(args):
        raise ValueError('Field dimensions must much the spatial vector components passed to the function')

    k_i = _wavenumbers(*args) # k_i = (kx, ky, kz)
    # a = window_ndim(a) # Windowing
    ak = np.fft.fftn(a) # FFT
    return _pair_integrate_fast(ak, *k_i, **kwargs) # Calc spectra

def _pair_integrate_fast(ak, *args, **kwargs):
    """
    Internal function which computes the wavenumber modulus k_mod for each fft coefficient of the ak ndarray and
    integrates the components contributing to the same k_mod with a certain bandwidth dk
    :param ak: The nFFT of (a)
    :param args: the wavenumber vector *k_i
    :param kwargs: k_res for the resolution of k_mod_line which defines the bandwitdh dk.
    :return: k_mod 1D array and ak_integral 1D array.
    """
    k_res = kwargs.get('k_res', 200)
    k2_sum_max = 0
    k2_sum_min = 0
    for k in args:
        k2_sum_max += np.max(k**2)
        k2_sum_min += np.min(k**2)
    k_min, k_max = np.sqrt(k2_sum_min), np.sqrt(k2_sum_max)
    dk = (k_max - k_min) / k_res

    ak_integral = np.zeros(k_res)
    k_mod_line = np.linspace(0, k_res - 1, k_res) * dk + dk / 2  # k values at half of each bandwidth

    print('    Find ak(k_i) with its k modulus and integrate it in ak(k)')
    with tqdm(total=ak.size) as pbar:
        for index in np.ndindex(ak.shape):
            ak_p = ak[index]
            k2_sum = 0
            for i, k in enumerate(args):
                k2_sum += k[index[i]] ** 2
            k_mod = np.sqrt(k2_sum)

            kint = int(k_mod / dk)
            if kint >= k_res:
                ak_integral[-1] += np.abs(ak_p)
            else:
                ak_integral[kint] += np.abs(ak_p)
            pbar.update(1)

    return k_mod_line, ak_integral

def _wavenumbers(*args):
    """
    :param args: the wavenumber vector *k_i
    :return: the wavenumber vector for a position vector (1D, 2D or 3D) such as: x, y, z of a uniform grid.
    """
    k_i = () # wavenumber vector; k_i = (kx, ky, kz). Each component is a 1D array type.
    for arg in args:
        N = arg.size # number of points in the spatial vector component
        alpha = 2*np.pi/(np.max(arg)-np.min(arg)) # basic wavenumber
        index = np.fft.fftfreq(N, d=1/N) # index per wavenumber direction: eg x: 0,1,2,...,N/2-1,-N/2,-N/2+1,...,-1

        k = np.zeros(N) # wavenumber vector component k_i
        for i in range(0, N):
            k[i] = alpha*index[i]
        k_i = k_i + (k,)
    return k_i


def _window_ndim(a, wfunction):
    """
    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.
    :param a: Input data to be windowed, modified in place.
    :param wfunction: 1D window generation function. Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    :return: windowed a
    """
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