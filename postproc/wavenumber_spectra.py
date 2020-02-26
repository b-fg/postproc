# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module to compute the wavenumber spectra of a n-dimensional field.
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np
import scipy.signal as signal
from tqdm import tqdm

# Functions
def spectra(u_i, x_i, **kwargs):
	"""
	Computes the kinetic energy (KE) of a n-dimensional velocity vector field such as u_i = (u, v, w) for 3D.
	:param u_i: n-dimensional velocity vector field
	:param x_i: n-dimensional spatial vector field
	:param kwargs: k_res for the resolution of k_mod_line which defines the bandwitdh dk.
	:return: k_mod 1D array and ke_integral 1D array.
	"""
	if len(u_i) > 3 or len(u_i) < 1 or len(x_i) > 3 or len(x_i) < 1 or any([u.ndim != len(x_i) for u in u_i]):
		raise ValueError('Invalid field dimensions')
	# Wavenumbers
	k_i = _wavenumbers(*x_i) # k_i = (kx, ky, kz)
	# FFT to compute KE
	ke = 0
	for u in u_i:
		u = _window_ndim(u, signal.hanning) # Windowing
		uk = np.fft.fftn(u)/u.size # FFT
		ke += uk*uk.conjugate() # KE
	ke = 0.5*ke
	# Calc spectra
	workers = kwargs.get('workers', 1)
	if workers > 1:
		return _pair_integrate_fast(ke, *k_i, **kwargs)
	else:
		return _pair_integrate(ke, *k_i, **kwargs)


def _pair_integrate(ak, *args, **kwargs):
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


def _pair_integrate_fast(ak, *args, **kwargs):
	"""
	Internal function which computes the wavenumber modulus k_mod for each fft coefficient of the ak ndarray and
	integrates the components contributing to the same k_mod with a certain bandwidth dk
	:param ak: The nFFT of (a)
	:param args: the wavenumber vector *k_i
	:param kwargs: k_res for the resolution of k_mod_line which defines the bandwitdh dk.
	:return: k_mod 1D array and ak_integral 1D array.
	"""
	from concurrent.futures import ThreadPoolExecutor
	from pathos.multiprocessing import Pool, cpu_count
	def spherical_integration(index):
		k2_sum = 0
		for i, k in enumerate(args):
			k2_sum += k[index[i]] ** 2
		k_mod = np.sqrt(k2_sum)
		kint = int(k_mod / dk)
		if kint >= k_res:
			ak_integral[-1] += np.abs(ak[index])
		else:
			ak_integral[kint] += np.abs(ak[index])

	workers = kwargs.get('workers', 1)
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

	# Threading
	# with ThreadPoolExecutor(max_workers=workers) as executor:
	# 	_ = list(tqdm(executor.map(spherical_integration, np.ndindex(ak.shape)),
	# 	                    total=ak.size))

	# Processing
	with Pool(processes=workers) as pool:
		_ = list(tqdm(pool.map(spherical_integration, np.ndindex(ak.shape)), total=ak.size))

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
	:param a: n-dimensional array input data to be windowed, modified in place.
	:param wfunction: 1D window generation function. Function should accept one argument: the window length.
		   Example: scipy.signal.hamming
	:return: windowed n-dimensional array a
	"""
	if a.ndim == 0:
		raise ValueError('Input data to be windowed cannot be scalar')
	for axis, axis_size in enumerate(a.shape):
		window = wfunction(axis_size)
		for i in range(len(a.shape)):
			if i == axis:
				continue
			else:
				window = np.stack([window] * a.shape[i], axis=i)
		a *= window
	return a