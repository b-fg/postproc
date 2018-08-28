# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module with function to plot 2D colormaps and CL-t graphs.
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np

# Internal functions
def plot2D(u, cmap, lvls, lim, file, **kwargs):
    """
    Return nothing and saves the figure in the specified file name.
    Args:
        cmap: matplotlib cmap. Eg: cmap = "seismic"
        lvls: number of levels of the contour. Eg: lvls = 100
        lim:  min and max values of the contour passed as array. Eg: lim = [-0.5, 0.5]
        file: Name of the file to save the plot (recommended .pdf so it can be converted get .svg).
              Eg: file = "dUdy.pdf"
    Kwargs:
        x=[xmin,xmax] is the x axis minimum and maximum specified
        y=[ymin,ymax] is the y axis minimum and maximum specified
        annotate: Boolean if annotations for min and max values of the field (and locations) are desired
    """
    # Internal imports
    import matplotlib as mpl
    # ! Uncomment for running plyplot without window display
    import matplotlib.pyplot as plt
    #    plt.switch_backend('AGG')
    #    plt.switch_backend('PS')
    plt.switch_backend('PDF')
    #    plt.switch_backend('SVG')

    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    N, M = u.shape[0], u.shape[1]

    # Get kwargs
    # for key, value in kwargs.items():
    #     if key=='x':
    #         xmin, xmax = value[0], value[1]
    #     if key=='y':
    #         ymin, ymax = value[0], value[1]
    if not 'x' in kwargs:
        xmin, xmax = 0, N-1
    else:
        xmin, xmax = kwargs['x'][0], kwargs['x'][1]
    if not 'y' in kwargs:
        ymin, ymax = -M/2, M/2-1
    else:
        ymin, ymax = kwargs['y'][0], kwargs['y'][1]
    if not 'annotate' in kwargs:
        annotate = False
    else:
        annotate = kwargs['annotate']

    # Uniform grid generation
    x, y = np.linspace(xmin, xmax, N), np.linspace(ymin, ymax, M)
    x, y = np.meshgrid(x, y)
    u = np.transpose(u)

    # Matplotlib definitions
    fig1 = plt.gcf()
    ax = plt.gca()
    # plt.rcParams['text.usetex'] = False  # Set TeX interpreter
    mpl.rc('font', family='DejaVu Sans')

    # Create contourf given a normalized (norm) colormap (cmap)
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    cf = plt.contourf(x, y, u, lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)

    # Scale contourf and set limits
    plt.axis('scaled')
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    # Scale colorbar to contourf
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, aspect=10)
    cbax = plt.colorbar(cf, cax=cax).ax
    mpl.colorbar.ColorbarBase(cbax, norm=norm, cmap=cmap)

    # Add annotation if desired
    if annotate:
        str_annotation = max_min_loc(u, xmin, ymin)
        print(str_annotation)
        ann_ax = fig1.add_subplot(133)
        ann_ax.axis('off')
        ann_ax.annotate(str_annotation, (0, 0),
                        xycoords="axes fraction", va="center", ha="center",
                        bbox=dict(boxstyle="round, pad=1", fc="w"))

    # Show, save and close figure
    plt.savefig(file, transparent=True, bbox_inches='tight')
    plt.draw()
    plt.show()
    # plt.clf()
    return

def max_min_loc(a, xmin, ymin):
    a_max = np.amax(a)
    a_min = np.amin(a)
    i_max_loc, j_max_loc = np.unravel_index(a.argmax(), a.shape)
    print(a.argmax(), i_max_loc, j_max_loc)
    x_max_loc, y_max_loc = i_max_loc + xmin, j_max_loc + ymin
    i_min_loc, j_min_loc = np.unravel_index(a.argmin(), a.shape)
    x_min_loc, y_min_loc = i_min_loc + xmin, j_min_loc + ymin
    my_str = ' max: {:.2e}, max_loc: ({},{}) \n min: {:.2e}, min_loc: ({},{})'.format(a_max, x_max_loc, y_max_loc,
                                                                                      a_min, x_min_loc, y_min_loc)
    return my_str


def plotCL(fy,t,file,**kwargs):
    # Internal imports
    import matplotlib.pyplot as plt
    plt.switch_backend('PDF')

    plt.rcParams['text.usetex'] = True  # Set TeX interpreter
    ax = plt.gca()
    fig = plt.gcf()

    # Show lines
    plt.plot(t, fy, color='blue', lw=1, label=r'$3\mathrm{D}\,\, \mathrm{total}$')

    # Set limits
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(1.5*min(fy), 1.5*max(fy))

    # Edit frame, labels and legend
    ax.axhline(linewidth=1)
    ax.axvline(linewidth=1)
    plt.xlabel(r'$t/D$')
    plt.ylabel(r'$C_L$')
    # leg = plt.legend(loc='upper right')
    # leg.get_frame().set_edgecolor('black')

    # Anotations
    for key, value in kwargs.items():
        if key=='St':
            St_str = '{:.2f}'.format(value)
            my_str = r'$S_t='+St_str+'$'
            plt.text(x=1.02*max(t), y=1.4*max(fy), s=my_str, color='black')
        if key=='CL_rms':
            CL_rms_str = '{:.2f}'.format(value)
            my_str = r'$\overline{C}_L='+CL_rms_str+'$'
            plt.text(x=1.02*max(t), y=1.2*max(fy), s=my_str, color='black')
        if key=='CD_rms':
            CD_rms_str = '{:.2f}'.format(value)
            my_str = r'$\overline{C}_D='+CD_rms_str+'$'
            plt.text(x=1.02*max(t), y=1.0*max(fy), s=my_str, color='black')
        if key=='n_periods':
            n_periods = str(value)
            my_str = r'$\textrm{periods}='+n_periods+'$'
            plt.text(x=1.02*max(t), y=0.8*max(fy), s=my_str, color='black')

    # plt.text(x=30, y=1e-4, s=r'$-5/3$', color='black')
    # plt.text(x=30, y=1e-7, s=r'$-3$', color='black')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return
