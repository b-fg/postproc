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

    resample = kwargs.get['resample', True]
    downsample = kwargs.get['downsample', True]

    # Re-sample u on a evenly spaced time series (constant dt)
    # if resample:
    u_function = interp1d(t, u, kind='cubic')
    t_min, t_max = np.min(t), np.max(t)
    dt = (t_max-t_min)/len(t)
    t_regular = np.arange(t_min, t_max, dt)[:-1] # Skip last one because can be problematic if > than actual t_max
    u_regular = u_function(t_regular)

    # Signal filtering for high frequencies
    u_regular = low_pass_filter(u_regular)
    # Windowing
    u_regular = window(u_regular)
    # Compute fft and associated frequencies
    uk = np.abs(np.fft.fft(u_regular)) / len(u_regular)
    freqs = np.fft.fftfreq(uk.size, d=dt)
    # Downsample averaging
    n = 4
    # uk = downsample_avg(uk, n)
    # freqs = downsample_avg(freqs, n)
    uk = downsample_avg(uk, n)
    freqs = downsample_avg(freqs, n)
    # Take only positive frequencies
    freqs = freqs[freqs > 0]
    uk = uk[:len(freqs)]

    return freqs, uk

def time_spectra_splits(t, u, n, OL):
    import numpy as np
    from scipy.interpolate import interp1d

    # Re-sample u on a evenly spaced time series (constant dt)
    u_function = interp1d(t, u, kind='cubic')
    t_min, t_max = np.min(t), np.max(t)
    dt = (t_max - t_min) / len(t)
    t_regular = np.arange(t_min, t_max, dt)[:-1]  # Skip last one because can be problematic if > than actual t_max
    u_regular = u_function(t_regular)

    # Split signal
    u_partial_OL_list = split_overlap(u, n, OL)
    t_partial_OL_list = split_overlap(t, n, OL)

    #


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
    b, a = signal.butter(3, 0.5, 'low') # 2nd arg: Fraction of fs that wants to be filtered
    return signal.filtfilt(b, a, u)
