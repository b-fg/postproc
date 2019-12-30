# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: St, number of periocs and CL, CD rms calculations
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np
from scipy.interpolate import interp1d


# Functions
def find_St(t, fy, D, U):
	"""
	:param t: Time series.
	:param fy: Lift force series.
	:param D: Characteristic length
	:param U: Characteristic velocity
	:return: Strouhal number, or non-dimensional shedding frequency.
	"""

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
	"""
	Compute the rms value of a signal y
	:param y: Signal
	:return: y_rms
	"""
	return np.sqrt(np.mean(y**2))

def find_num_periods(y):
	"""
	Find the number of periods in a signal
	:param y: Signal
	:return: Number of periods.
	"""
	n_periods = 0
	for i in np.arange(0, len(y)-2):
		if y[i]>0 and y[i+1] <= 0:
			n_periods += 1
	return n_periods-1

def calc_Cp(p, L, R=45, p_inf=0.02, rho_inf=1, U_inf=1):
	"""
	Find the Pressure coefficient Cp and the angle from a pressure-arclength series.
	:param p:
	:param L:
	:param R:
	:param p_inf:
	:param rho_inf:
	:param U_inf:
	:return:
	"""
	cp = (p-p_inf)/(0.5*rho_inf*U_inf)
	theta = L/R*180/np.pi

	cp_function = interp1d(theta, cp, kind='linear')
	theta_min, theta_max = np.min(theta), np.max(theta)
	dtheta = (theta_max - theta_min) / len(theta)
	reg_theta = np.arange(theta_min, theta_max, dtheta)[:-1]  # Skip last one because can be problematic if > than actual t_max
	reg_cp = cp_function(reg_theta)

	index_cp_max = np.argmax(reg_cp)

	args_sort = np.argsort(reg_theta)
	reg_theta = reg_theta[args_sort]
	reg_cp = reg_cp[args_sort]
	reg_theta = reg_theta[:index_cp_max+1]
	reg_cp = reg_cp[:index_cp_max+1]
	reg_cp = reg_cp[::-1]

	# index_roll = index_cp_max-reg_theta.size
	# reg_theta = np.roll(reg_theta, index_cp_max)
	# args_sort = np.argsort(reg_theta)
	# reg_theta = reg_theta[args_sort]
	# reg_cp = reg_cp[args_sort]

	reg_theta = reg_theta[reg_theta<=180]
	reg_cp = reg_cp[:reg_theta.size]

	return reg_theta, reg_cp


