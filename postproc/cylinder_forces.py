# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: CL CD St calculations
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np

# Internal functions
def find_St(t, fy, D, U):
    from scipy.interpolate import interp1d
    # Re-sample fy on a evenly spaced time series (constant dt)
    fy_function = interp1d(t, fy)
    t_min, t_max = np.min(t), np.max(t)
    dt = (t_max-t_min)/len(t)
    df = 1/dt
    reg_t = np.arange(t_min, t_max, dt)[:-1] # Skip last one because can be problematic if > than actual t_max
    reg_fy = fy_function(reg_t)

    # Get the dominant frequency
    fy_fft = np.fft.fft(reg_fy)
    freqs = np.fft.fftfreq(len(reg_fy))
    index_fy_fft_max = np.argmax(np.abs(fy_fft))
    freq = freqs[index_fy_fft_max]
    return abs(freq*df)*D/U

def rms(y):
    return np.sqrt(np.mean(y**2))

def find_num_periods(y):
    n_periods = 0
    for i in np.arange(0, len(y)-2):
        if y[i]>0 and y[i+1] <= 0:
            n_periods += 1

    return n_periods-1
