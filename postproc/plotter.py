# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Functions to plot 2D colormaps and CL-t graphs.
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np
import matplotlib as mpl
# ! Uncomment for running plyplot without window display
import matplotlib.pyplot as plt

# Functions
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
    # plt.switch_backend('AGG')
    # plt.switch_backend('PS')
    plt.switch_backend('PDF')
    # plt.switch_backend('SVG')
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    N, M = u.shape[0], u.shape[1]
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
    my_str = ' max: {:.2e}, max_loc: ({},{}) \n min: {:.2e}, min_loc: ({},{})'\
        .format(a_max, x_max_loc, y_max_loc, a_min, x_min_loc, y_min_loc)
    return my_str


def plotCL(fy,t,file,**kwargs):
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

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return

def plotTKEspatial(tke, file, **kwargs):
    """
    Generate a plot of the TKE in space
    """
    # Basic definitions
    plt.switch_backend('PDF')
    plt.rcParams['text.usetex'] = True  # Set TeX interpreter
    ax = plt.gca()
    fig  = plt.gcf()

    N = tke.shape[0]
    if not 'x' in kwargs:
        xmin, xmax = 0, N-1
    else:
        xmin, xmax = kwargs['x'][0], kwargs['x'][1]
    if not 'ylog' in kwargs:
        ylog = False
    else:
        ylog = kwargs['ylog']

    x = np.linspace(xmin, xmax, N)

    # Show lines
    plt.plot(x, tke, color='black', lw=1.5, label='$L_z = 1D$')

    # Set limits
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(tke), max(tke)*1.1)

    fig, ax = makeSquare(fig,ax)

    if ylog:
        ax.set_yscale('log')
        ax.set_ylim(min(tke), max(tke) * 2)

    # Edit frame, labels and legend
    plt.xlabel('$x/D$')
    plt.ylabel('$K$')
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return

def plotLogLogTimeSpectra(freqs, uk, file):
    """
    Generate a loglog plot of a time spectra series using the matplotlib library given the arguments
    """
    # Basic definitions
    plt.switch_backend('PDF')
    plt.rcParams['text.usetex'] = True  # Set TeX interpreter
    ax = plt.gca()
    fig  = plt.gcf()

    # Show lines
    plt.loglog(freqs, uk, color='black', lw=1.5, label='$L_z = 1D$')
    x, y = loglogLine(p2=(1,1e-4), p1x=1e-3, m=-5/3)
    plt.loglog(x, y, color='black', lw=1, ls='dotted')
    x, y = loglogLine(p2=(1, 1e-6), p1x=1e-3, m=-3)
    plt.loglog(x, y, color='black', lw=1, ls='dashdot')

    # Set limits
    ax.set_xlim(min(freqs[freqs>0.5*1e-3]), max(freqs[freqs<0.5]))
    ax.set_ylim(min(uk)*1e-1, max(uk)*1e1)

    # fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel(r'$f$')
    plt.ylabel(r'$F(v)$')
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')

    # Anotations
    plt.text(x=5e-3, y=2e-1, s='$-5/3$', color='black')
    plt.text(x=1e-2, y=1, s='$-3$', color='black')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return

def loglogLine(p2, p1x, m):
    b = np.log10(p2[1])-m*np.log10(p2[0])
    p1y = p1x**m*10**b
    return [p1x, p2[0]], [p1y, p2[1]]


def makeSquare(fig, ax):
    fwidth = fig.get_figwidth()
    fheight = fig.get_figheight()
    # get the axis size and position in relative coordinates
    # this gives a BBox object
    bb = ax.get_position()
    # calculate them into real world coordinates
    axwidth = fwidth*(bb.x1-bb.x0)
    axheight = fheight*(bb.y1-bb.y0)
    # if the axis is wider than tall, then it has to be narrowe
    if axwidth > axheight:
        # calculate the narrowing relative to the figure
        narrow_by = (axwidth-axheight)/fwidth
        # move bounding box edges inwards the same amount to give the correct width
        bb.x0 += narrow_by/2
        bb.x1 -= narrow_by/2
    # else if the axis is taller than wide, make it vertically smaller
    # works the same as above
    elif axheight > axwidth:
        shrink_by = (axheight-axwidth)/fheight
        bb.y0 += shrink_by/2
        bb.y1 -= shrink_by/2
    ax.set_position(bb)
    return fig, ax


def plotLumleysTriangle(eta, xi, file, **kwargs):
    """
    Generate a plot of the Reynolds stresses anisotropy tensor in space
    """
    # Basic definitions
    plt.switch_backend('PDF')
    plt.rcParams['text.usetex'] = True  # Set TeX interpreter
    ax = plt.gca()
    fig  = plt.gcf()

    x = np.linspace(-1/6, 1/3, 500)
    y = np.sqrt(1/27+2*x**3)

    # Show lines
    plt.plot(x, y, color='black', lw=1.5, label='$2C$')
    plt.plot([-1/6,0], [1/6,0], color='black', lw=1.5, label='$1C$')
    plt.plot([0,1/3], [0,1/3], color='black', lw=1.5, label='$1C$')
    plt.scatter(xi,eta, marker='o', c='green', s=1, linewidths=0.1)
    plt.scatter(xi,eta, marker='d', c='blue', s=1, linewidths=0.1)

    # Set limits
    ax.set_ylim(0, 0.35)
    ax.set_xlim(-0.2, 0.4)
    # Make figure squared
    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel('$\eta$')
    plt.ylabel('$ \eta $')
    leg = plt.legend(loc='upper left')
    leg.get_frame().set_edgecolor('white')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return