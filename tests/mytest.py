# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: package tests
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np
import postproc.cylinder_forces as cf
import postproc.io as io
import postproc.averages as averages
import postproc.plotter as plotter

N = 992
M = 446
D = 64
U = 1
x= [-94, 897]
y= [-223, 222]

FFTfile = '/home/b-fg/Workspace/Lotus/post_proc_data/2D_D64/2DFFTdata_Re100.dat'
CLfile = '/home/b-fg/Workspace/Lotus/post_proc_data/2D_D64/fort_Re100.9'
shape = (N, M)

u, v = io.read_data(file=FFTfile, shape=shape, dtype=np.single, stream=True, periodic=False)
t, fx, fy = io.unpack2Dforces(D, CLfile)
vort = averages.vortz(u, v)
plotter.plot2D(vort, cmap='bwr', lvls=100, lim=[-0.15, 0.15], file='vort.pdf', x=x, y=y)
plotter.plotCL(fy, t, 'CLplot.pdf', St=cf.find_St(t,fy,D,U), CL_rms=cf.rms(fy), CD_rms=cf.rms(fx), n_periods=cf.find_num_periods(fy))

