# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module to compute the temporal spectra of a certain quantity.
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np
from scipy import signal

# Functions
def time_spectra(t, u, **kwargs):
    import numpy as np
    from scipy.interpolate import interp1d

    resample = kwargs.get('resample', True)
    lowpass = kwargs.get('lowpass', True)
    windowing = kwargs.get('windowing', True)
    downsample = kwargs.get('downsample', 4)

    # Re-sample u on a evenly spaced time series (constant dt)
    if resample:
        u_function = interp1d(t, u, kind='cubic')
        t_min, t_max = np.min(t), np.max(t)
        dt = (t_max-t_min)/len(t)
        t_regular = np.arange(t_min, t_max, dt)[:-1] # Skip last one because can be problematic if > than actual t_max
        u = u_function(t_regular)
    else:
        dt = t[1]-t[0]
    if lowpass: u = low_pass_filter(u) # Signal filtering for high frequencies
    if windowing: u = window(u) # Windowing

    # Compute fft and associated frequencies
    uk = np.abs(np.fft.fft(u)) / len(u)
    freqs = np.fft.fftfreq(uk.size, d=dt)

    # Downsample averaging
    if downsample > 0:
        uk = downsample_avg(uk, downsample)
        freqs = downsample_avg(freqs, downsample)

    # Take only positive frequencies and return arrays
    freqs = freqs[freqs > 0]
    uk = uk[:len(freqs)]
    return freqs, uk

def time_spectra_splits(t, u, n=8, OL=0.5):
    import numpy as np
    from scipy.interpolate import interp1d

    # Re-sample u on a evenly spaced time series (constant dt)
    u_function = interp1d(t, u, kind='cubic')
    t_min, t_max = np.min(t), np.max(t)
    dt = (t_max - t_min) / len(t)
    t = np.arange(t_min, t_max, dt)[:-1]  # Regularize t and Skip last one because can be problematic if > than actual t_max
    u = u_function(t) # Regularize u

    # Split signal
    u_partial_OL_list = split_overlap(u, n, OL)
    t_partial_OL_list = split_overlap(t, n, OL)

    uk_partial_OL_list = []
    freqs_partial_OL_list = []
    for tup in list(zip(t_partial_OL_list, u_partial_OL_list)):
        freqs, uk = time_spectra(tup[0], tup[1], resample=False, downsample=False, lowpass=True)
        freqs_partial_OL_list.append(freqs)
        uk_partial_OL_list.append(uk)

    uk_mean = np.mean(uk_partial_OL_list, axis=0)
    freqs_mean = np.mean(freqs_partial_OL_list, axis=0)

    return freqs_mean, uk_mean


def split_overlap(a, n, OL):
    """""
    a: array to split and overlap
    n: number of splits of a
    OL: overlap 
    """""
    splits_size = int(round(a.size/n))
    nOL = int(round(splits_size * OL))
    skip = splits_size - nOL
    b = [a[i: i + splits_size] for i in range(0, len(a), skip)]
    c = []
    for i, item in enumerate(b):
        if len(item) == splits_size:
            c.append(item)
    return c


def window(a):
    w = signal.blackman(len(a))
    # w = signal.hanning(len(a))
    return a * w

def downsample_avg(arr, n): # n is the size of the averaging subarray
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def downsample_simple(arr, n): # n is the size of the averaging subarray
    return arr[::n]

def low_pass_filter(u):
    b, a = signal.butter(3, 0.4, 'low') # 2nd arg: Fraction of fs that wants to be filtered
    return signal.filtfilt(b, a, u)
