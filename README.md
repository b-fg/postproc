
## **PostProc**  
### A post-processing package for my PhD CFD simulations

`postproc` is a Python library for post-processing Computational Fluid Dynamics simulations (CFD) data I produce for my PhD. The following modules are provided:

- ``cylinder_forces.py``: a set of functions which allow to extract useful information from the forces on a body provided with a time series of the lift force.

- ``calc.py``: module containing functions to compute flow field operations such as derivatives, averages, decompositions and vorticity vector calculation.

- ``io.py``: functions related to import data from Fortran simulations in binary or ASCII format. Imports 2D or 3D flow fields with a specified number of components. It also imports text files written in columns with the ``unpack*`` functions.

- ``plotter.py``: module equipped with subroutines to generate many different type of plots according to the input data. From 2D contours to CL-t graphs and turbulent kinetic energy plots, Lumley's triagle plot and more.

- ``anisotropy_tensor.py``: functions to compute the normalized Reynolds stresses anisotropy tensor from the Reynolds stresses tensor and its invariants.

- ``frequency_spectra.py``: module to compute the frequency spectra of a temporal signal.

- ``wavenumber_spectra.py``: module to compute the wavenumber spectra of a spatial vector or scalar field.



### Installation and usage

The use of a virtual environment is recommended for the installation of the package. Here we will use ``conda``. Create a new `conda` environment with based on python 3.6 as follows:
```
conda create --name my_env python=3.6
```
Activate the environment and install the package using `pip`:
```
source activate my_env
pip install -e ~/postproc
```
The environment can be deactivated running `conda deactivate`. In order to be able to perform modification on the package without the need of reinstalling the `-e` (editable) argument is used. This provides the source path of the package to the `conda` environment so any modifications on the source code (the folder you have downloaded and installed from) is immediately available with no need to re-install.

To use the package just activate the `conda` environment in your terminal with `source activate my_env` or load it into your preferred IDE. In your python script, you can import the modules with:

	import postproc.calc as calc
	import postproc.cylinder_forces as cf
	import postproc.io as io
	import postproc.plotter as plotter
See the ``/tests`` folder for sample scripts (sample data is included).
