
## **PostProc**  
### A post-processing package for my PhD CFD simulations

This library has the goal of gathering all the python scripts I write for post-processing Computational Fluid Dynamics simulations (CFD) data I produce for my PhD. By making it public I hope someone else can find this useful as well.

This repository contains the ``postproc`` package provided with the following tools

- ``cylinder_forces.py``: a set of functions which allow to extract useful information from the forces on a body provided with a time series of the lift force.

- ``calc.py``: module containing functions to compute flow field operations such as derivatives, averages, decompositions and vorticity vector calculation.

- ``io.py``: functions related to import binary data from Fortran simulations. Imports 2D or 3D flow fields with a specified number of components. It also imports text files written in columns with the ``unpack*`` functions.

- ``plotter.py``: module equipped with subroutines to generate many different type of plots according to the input data. From 2D contours to CL-t graphs and turbulent kinetic energy plots, Lumley's triagle plot and more.

- ``anisotropy_tensor.py``: functions to compute the Reynolds stresses normalized anisotropy tensor from the Reynolds stresses tensor and its invariants.

- ``frequency_spectra.py``: module to compute the frequency spectra of a temporal signal.

- ``wavenumber_spectra.py``: module to compute the wavenumber spectra of a spatial vector or scalar field.



### Installation

To install this package in your workstation you only need ``git`` and ``pip3``. To install the `postproc` package run:

	git clone https://github.com/b-fg/postproc.git
	cd postproc
	sudo pip3 install . -e

This will install the `postproc` package at you `python3`  libraries folder (probably `/usr/lib/python3/dist-packages`). In order to be able to perform modification on the package without the need of reinstalling the `-e` (editable) argument is used. This provides the source path of the package to the installed library so any modifications on the source code (the folder you have downloaded and installed from) is immediately available with no need to re-install.

### Usage

Add in your python script:

	import postproc.averages as averages
	import postproc.cylinder_forces as cf
	import postproc.io as io
	import postproc.plotter as plotter

To add some of the packages modules. See the ``/tests`` folder for sample scripts (sample data is included).
