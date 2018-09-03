# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: package tests
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import postproc.io as io
import postproc.calc as averages
import postproc.plotter as plotter

# Constants
N = 992                     # Number of grid points in i direction
M = 446                     # Number of grid points in j direction
L = 1                       # Number of grid points in k direction

xmin, xmax = -94, 897       # Domain size in i direction
ymin, ymax = -223, 222      # Domain size in j direction
zmin, zmax = 0, 0           # Domain size in k direction

D = 64
U = 1
file = 'sample_data/2D_Re100.dat' # File containig velocity field

# Main

shape = (N, M)
u, v = io.read_data(file=file, shape=shape, ncomponents=2)
vort = averages.vortz(u, v)
plotter.plot2D(vort, cmap='bwr', lvls=100, lim=[-0.15, 0.15], file='vort.pdf', x=[xmin,xmax], y=[ymin,ymax])


