
## **_postproc_**  
### A post-processing package for my PhD CFD simulations

This library has the goal of gathering all the python scripts I write for post-processing data from my Computational Fluid Dynamics simulations (CFD) I run for my PhD. By making it public I hope someone else can find this useful as well.

This repository contains the ``postproc`` package provided with the following tools

- ``cylinder_forces.py``: a set of functions which allow to extract useful information from the forces on a body provided with a time series of the lift force.

- ``calc.py``: module containing functions to compute flow field operations such as derivatives, averages, decompositions and vorticity.

- ``io.py``: functions related to import binary data from Fortran simulations. Imports 2D or 3D flow fields with a specified number of components. It imports also text files written in columns with the ``unpack*`` functions.

- ``plotter.py``: module equipped with subroutines to plot a 2D contour and CL-t graphs.

- ``anisotropy_tensor.py``: calculate the Reynolds stresses normalized anisotropy tensor from the Reynolds stresses tensor and its invariants.

- ``temporal_spectra.py``: module to compute the temporal spectra of a certain quantity.


### Installation

To install this package in your workstation just make sure tot have ``git`` and ``pip3`` installed and then run

	git clone https://github.com/b-fg/postproc.git
	cd postproc
	sudo pip3 install . -r requirements.txt

This will install the `postproc` package at you `python3`  libraries folder (probably `/usr/lib/python3/dist-packages`) with the declared required packages in ``requirements.txt``. This way the package will be available from any path.  
If you wish to make modifications and make them ready straight-away use also ``-e`` on the  ``pip3`` install command. This provides the source path of the package to the installed library so any modification is immediately available with no need to re-install.

### Usage

Add in your python script:

	import postproc.averages as averages
	import postproc.cylinder_forces as cf
	import postproc.io as io
	import postproc.plotter as plotter

To add all of the packages modules. See ``/tests`` folder for sample scripts (sample data is included).
