# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module to compute the temporal spectra of a certain quantity.
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np

# Functions
def time_spectra(t, u):
    from scipy.interpolate import interp1d
    # Re-sample u on a evenly spaced time series (constant dt)
    u_function = interp1d(t, u)
    t_min, t_max = np.min(t), np.max(t)
    dt = (t_max-t_min)/len(t)
    t_regular = np.arange(t_min, t_max, dt)[:-1] # Skip last one because can be problematic if > than actual t_max
    u_regular = u_function(t_regular)

    # Compute fft and associated frequencies
    uk = np.abs(np.fft.fft(u_regular)) / len(u_regular)
    freqs = np.fft.fftfreq(u_regular.size, d=dt)

    # Take only positive frequencies
    freqs = freqs[freqs > 0]
    uk = uk[:len(freqs)]

    return freqs, uk