# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Functions to plot 2D colormaps and CL-t graphs.
@contact: b.fontgarcia@soton.ac.uk
"""


# Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('text', usetex=True )
plt.rc('font',family = 'sans-serif',  size=13)
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)
mpl.rcParams['axes.linewidth'] = 0.5
# plt.switch_backend('AGG')
# plt.switch_backend('PS')
plt.switch_backend('PDF')
# plt.switch_backend('PS')
# plt.switch_backend('SVG')
# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'STIXGeneral'

colors = ['black', 'orange', 'cyan', 'green', 'blue', 'red', 'magenta', 'yellow']
# colors = ['orange', 'cyan', 'green', 'blue', 'red', 'magenta', 'yellow']
markers = ['|', 's', '^', 'v', 'x', 'o', '*']
# markers = ['s', '^', 'v', 'x', 'o', '*']


# Functions
# ------------------------------------------------------ COUNTOURS
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
    annotate = kwargs.get('annotate', False)
    scaling = kwargs.get('scaling', 1)
    xshift = kwargs.get('xshift', 0)
    yshift = kwargs.get('yshift', 0)

    # Uniform grid generation
    x, y = np.linspace(xmin/scaling, xmax/scaling, N), np.linspace(ymin/scaling, ymax/scaling, M)
    x, y = x+xshift, y+yshift
    x, y = np.meshgrid(x, y)
    u = np.transpose(u)

    # Matplotlib definitions
    fig1 = plt.gcf()
    ax = plt.gca()

    # Create contourf given a normalized (norm) colormap (cmap)
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    # ax.contour(x, y, u, lvls, linewidths=0.2, colors='k')
    cf = ax.contourf(x, y, u, lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Scale contourf and set limits
    plt.axis('scaled')
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    # Scale colorbar to contourf
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.02, aspect=10)
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
    plt.clf()
    return


def plot2D_uv(u, cmap, lvls, lim, file, **kwargs):
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
    annotate = kwargs.get('annotate', False)
    scaling = kwargs.get('scaling', 1)
    xshift = kwargs.get('xshift', 0)
    yshift = kwargs.get('yshift', 0)

    # Uniform grid generation
    x, y = np.linspace(xmin/scaling, xmax/scaling, N), np.linspace(ymin/scaling, ymax/scaling, M)
    x, y = x+xshift, y+yshift
    x, y = np.meshgrid(x, y)
    u = np.transpose(u)

    # Matplotlib definitions
    fig1 = plt.gcf()
    ax = plt.gca()

    # Create contourf given a normalized (norm) colormap (cmap)
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    # cf = plt.contourf(x, y, u, '--', lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)
    ax.contour(x, y, u, lvls, linewidths=0.5, colors='k')
    # cf = ax.contourf(x, y, u, lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)

    ax.xaxis.set_ticks([0.5, 1.0, 1.5, 2])
    ax.yaxis.set_ticks([-0.5, 0.0, 0.5])
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Scale contourf and set limits
    plt.axis('scaled')
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    # ax.xaxis.set_ticks(np.arange(0.5, 2.5, 0.5))

    # Scale colorbar to contourf
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05, aspect=10)
    # cbax = plt.colorbar(cf, cax=cax).ax
    # mpl.colorbar.ColorbarBase(cbax, norm=norm, cmap=cmap)

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


def plot2Dvort(u, cmap, lvls, lim, file, **kwargs):
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
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rc('font', family='sans-serif', size=6)
    mpl.rc('xtick', labelsize=6)
    mpl.rc('ytick', labelsize=6)

    N, M = u.shape[0], u.shape[1]
    if not 'x' in kwargs:
        xmin, xmax = 0, N-1
    else:
        xmin, xmax = kwargs['x'][0], kwargs['x'][1]
    if not 'y' in kwargs:
        ymin, ymax = -M/2, M/2-1
    else:
        ymin, ymax = kwargs['y'][0], kwargs['y'][1]
    annotate = kwargs.get('annotate', False)
    scaling = kwargs.get('scaling', 1)
    xshift = kwargs.get('xshift', 0)
    yshift = kwargs.get('yshift', 0)

    # Uniform grid generation
    x, y = np.linspace(xmin/scaling, xmax/scaling, N), np.linspace(ymin/scaling, ymax/scaling, M)
    x, y = x+xshift, y+yshift
    x, y = np.meshgrid(x, y)
    u = np.transpose(u)

    # Matplotlib definitions
    fig1 = plt.gcf()
    ax = plt.gca()

    # Create contourf given a normalized (norm) colormap (cmap)
    norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
    # cf = plt.contourf(x, y, u, lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)
    ax.contour(x, y, u, lvls, linewidths=0.2, colors='k')
    cf = ax.contourf(x, y, u, lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)

    # Scale contourf and set limits
    plt.axis('scaled')
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    # ax.xaxis.set_ticks(np.arange(0.5, 2.5, 0.5))
    ax.yaxis.set_ticks([-2,0,2])
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Show, save and close figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    # plt.draw()
    # plt.clf()
    return

# ------------------------------------------------------ CL-t
def plotCL(fy, t, file, **kwargs):
    """
    Plot the lift force as a time series.
    :param fy: Lift force [numpy 1D array]
    :param t: Time [numpy 1D array]
    :param file: output file name [string]
    :param kwargs: Select which additional information you want to include in the plot: 'St', 'CL_rms', 'CD_rms', 'n_periods',
        passing the corresponding values. E.g. 'St=0.2'.
    :return: -
    """
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


# ------------------------------------------------------ TKE
def plotTKEspatial(tke, file, **kwargs):
    """
    1D plot of the TKE in space
    :param tke: Turbulent kinetic energy [numpy 1D array]
    :param file: output file name [string]
    :param kwargs: 'x' coordinates [numpy 1D array]
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    N = tke.shape[0]
    if not 'x' in kwargs:
        xmin, xmax = 0, N-1
    else:
        xmin, xmax = kwargs['x'][0], kwargs['x'][1]

    x = np.linspace(xmin, xmax, N)
    ylog = kwargs.get('ylog', False)

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


def plotTKEspatial_list(file, tke_tuple_list, **kwargs):
    """
    Generate a plot of a TKE list of tuples like (case, tke) in space
    :param file: output file name [string]
    :param tke_tuple_list: list containing the tuple as ('case', tke), where 'case' is a string and 'tke' is a 1D numpy array
    :param kwargs: 'x' coordinates [numpy 1D array]
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    if not tke_tuple_list:
        raise ValueError("No TKE series passed to the function.")
    else:
        N = tke_tuple_list[0][1].shape[0]
    if not 'x' in kwargs:
        xmin, xmax = 0, N-1
    else:
        xmin, xmax = kwargs['x'][0], kwargs['x'][1]
    ylog = kwargs.get('ylog', False)
    ylabel = '$' + kwargs.get('ylabel','K') + '$'
    x = np.linspace(xmin, xmax, N)

    # Show lines
    tke_list = []
    i = 0
    for tke_tuple in tke_tuple_list:
        label = tke_tuple[0]
        if 'piD' in label: label = '\pi'
        else: label = label[:-1]
        tke = tke_tuple[1]
        if 'xD_min' in kwargs:
            x = x[x > kwargs['xD_min']]
            tke = tke[-x.size:]
        label = '$'+label+'$'
        color = colors[i]
        plt.plot(x, tke, color=color, lw=1.5, label=label, marker=markers[i], markevery=50, markersize=4)
        tke_list.append(tke)
        i += 1

    # Set limits
    ylims = kwargs.get('ylims', [np.min(tke_list),  np.max(tke_list)])
    print(ylims)
    ax.set_xlim(min(x), 12)
    ax.set_ylim(ylims[0], ylims[1])
    ax.xaxis.set_ticks([0,2,4,6,8,10,12])

    fig, ax = makeSquare(fig,ax)

    if ylog:
        ax.set_yscale('log')
        ax.set_ylim(ylims[0], ylims[1])
        plt.minorticks_off()

    # Edit frame, labels and legend
    plt.xlabel('$x$')
    plt.ylabel(ylabel)
    # leg = plt.legend(loc=(0.75,0.16))
    leg = plt.legend(loc='lower right')

    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)
    # ax.yaxis.set_ticks([-2, 0, 2])
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    plt.clf()
    return


# ------------------------------------------------------ x-y
def plotXYSpatial(y, label, file, **kwargs):
    """
    Generate a x-y plot in space
    :param y: series to plot [1D numpy array]
    :param label: y axis label [string]
    :param file: output file name
    :param kwargs: 'x' coordinates [numpy 1D array], 'xD_min' left x limit, 'ylog' log plot [boolean]
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    N = y.shape[0]
    if not 'x' in kwargs:
        xmin, xmax = 0, N-1
    else:
        xmin, xmax = kwargs['x'][0], kwargs['x'][1]

    x = np.linspace(xmin, xmax, N)
    if 'xD_min' in kwargs:
        x = x[x > kwargs['xD_min']]
        y = y[-x.size:]

    ylog = kwargs.get('ylog', False)

    # Show lines
    plt.plot(x, y, color='black', lw=0.5, label='$L_z = 1D$')

    # Edit figure, axis, limits
    ax.set_xlim(min(x), max(x))
    if ylog:
        ax.set_yscale('log')

    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    y_label = '$'+label+'$'
    plt.xlabel('$x/D$')
    plt.ylabel(y_label)
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return

def plotScatter(x, y, cases, file):
    """
    Generate a x-y plot in space
    :param x: series to plot [1D numpy array]
    :param y: series to plot [1D numpy array]
    :param label: y axis label [string]
    :param file: output file name
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    # Show lines
    for i, case in enumerate(cases):
        ax.scatter(x[i], y[i], c=colors[i], marker=markers[i], s=10, linewidths=1, label=case)


    # Edit figure, axis, limits
    ax.set_xlim(0.06, 0.15)
    ax.set_ylim(0.1, 1.4)

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel('$\mathrm{max}(TKE|_{y})$')
    plt.ylabel('$\overline{C}_L$')
    leg = plt.legend(loc='lower right')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


def plotXYSpatial_list(file, y_tuple_list, **kwargs):
    """
    Generate a x-y plot in space of multiples y series
    :param file: output file name
    :param y_tuple_list: list of tuples as (case, y) where 'case' is the name of the case [string] and 'y' the series [1D numpy array]
    :param kwargs: 'x' coordinates [numpy 1D array], 'xD_min' left x limit, 'ylog' log plot [boolean]
    :return: -
    """
    """
    Generate a XY plot
    """
    ax = plt.gca()
    fig  = plt.gcf()

    if not y_tuple_list:
        raise ValueError("No TKE series passed to the function.")
    else:
        N = y_tuple_list[0][1].shape[0]
    if not 'x' in kwargs:
        xmin, xmax = 0, N-1
    else:
        xmin, xmax = kwargs['x'][0], kwargs['x'][1]
    ylog = kwargs.get('ylog', False)
    ylabel = '$' + kwargs.get('ylabel','K') + '$'
    x = np.linspace(xmin, xmax, N)

    # Show lines
    y_list = []
    i = 0
    for y_tuple in y_tuple_list:
        label = y_tuple[0]
        if 'piD' in label: label = '\pi'
        else: label = label[:-1]
        y = y_tuple[1]
        label = '$'+label+'$'
        color = colors[i]

        if 'xD_min' in kwargs:
            x = x[x > kwargs['xD_min']]
            y = y[-x.size:]

        plt.plot(x, y, color=color, lw=1, label=label, marker=markers[i],
                 markevery=50, markersize=4)#, markeredgecolor = 'black', markeredgewidth=0.1)
        y_list.append(y)
        i += 1

    # Edit figure, axis, limits
    ax.set_xlim(min(x), max(x))
    if ylog:
        ax.set_yscale('log')
        plt.minorticks_off()

    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel('$x$')
    plt.ylabel(ylabel, rotation=0)
    if 'R' in ylabel:
        leg = plt.legend(loc='upper left')
        # ax.yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        ax.yaxis.set_ticks([0.0, 0.4, 0.8, 1.2])
    else:
        leg = plt.legend(loc='upper right')

    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)
    # ax.set_xlim(min(x), 12)
    # ax.set_ylim(ylims[0], ylims[1])
    ax.xaxis.set_ticks([2, 4, 6, 8, 10, 12])

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    plt.clf()
    return

def velocity_profiles(file, profiles_tuple_list, y_list, **kwargs):
    """
    Similar to plotXYSpatial_list just for a specific test case
    """
    ax = plt.gca()
    fig  = plt.gcf()

    ylog = kwargs.get('ylog', False)
    ylabel = '$' + kwargs.get('ylabel','y') + '$'

    # Show lines
    profiles = []
    for i, profile_tuple in enumerate(profiles_tuple_list):
        label = profile_tuple[0]
        profile = profile_tuple[1]
        y = y_list[i]
        label = '$'+label+'$'
        color = colors[i]
        # plt.plot(profile, y, color=color, lw=1, label=label, marker=markers[i], markevery=10, markersize=4)
        plt.plot(profile, y, color=color, lw=1, label=label)
        profiles.append(profile)

    # Edit figure, axis, limits
    # ax.set_xlim(min(x), max(x))

    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel(r'$\overline{u}$')
    plt.ylabel(ylabel, rotation=0)
    ax.set_ylim(np.min([np.min(y) for y in y_list]), np.max([np.max(y) for y in y_list]))


    leg = plt.legend(loc='upper left')
    leg.get_frame().set_edgecolor('white')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_alpha(0)

    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    plt.clf()
    return

def plotCp_list(file, y_tuple_list, x_list, **kwargs):
    """
    Similar to plotXYSpatial_list just for a specific test case
    """
    ax = plt.gca()
    fig  = plt.gcf()

    if not y_tuple_list:
        raise ValueError("No series passed to the function.")
    else:
        N = y_tuple_list[0][1].shape[0]
    ylabel = '$' + kwargs.get('ylabel','') + '$'

    # Show lines
    y_list = []
    i = 0
    for y_tuple in y_tuple_list:
        label = y_tuple[0]
        if 'piD' in label: label = '\pi D'
        y = y_tuple[1]
        label = '$'+label+'$'
        color = colors[i]

        # if i==1:
            # plt.scatter(x_list[i], y, marker='^', facecolors='none', edgecolors='black', s=25, linewidths=0.5, label=label)
            # plt.plot(x_list[i], y, color='black', lw=1, label=label)
        if i == 0:
            plt.scatter(x_list[i], y, marker='o', facecolors='none', edgecolors='black', s=25, linewidths=0.5, label=label)
            # plt.plot(x_list[i], y, markerfacecolor='none', lw=1.5, label=label, marker='o', color='black')
        else:
            plt.plot(x_list[i], y, color='black', lw=1, label=label)
        y_list.append(y)
        i += 1

    # Edit figure, axis, limits
    fig, ax = makeSquare(fig,ax)
    ax.set_xlim(0, 180)
    ax.xaxis.set_ticks(np.arange(0, 181, 30))

    # Edit frame, labels and legend
    plt.xlabel(r'$\theta$')
    plt.ylabel(ylabel, rotation=0)
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')
    ax.tick_params(direction='in')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


# ------------------------------------------------------ LogLog Spatial
def plotLogLogSpatialSpectra(file, wn, uk):
    """
    Generate a loglog plot of a 1D spatial signal
    :param file: output file name
    :param wn: frequency [1D numpy array]
    :param uk: transformed u: uk = FFT(u). [1D numpy array]
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    # Show lines
    plt.loglog(wn, uk, color='black', lw=1.5, label='$L_z = piD$')
    x, y = loglogLine(p2=(max(wn), 5e-4), p1x=min(wn)*10, m=-5/3)
    plt.loglog(x, y, color='black', lw=1, ls='dotted')
    x, y = loglogLine(p2=(max(wn), 4e-4), p1x=min(wn)*10, m=-3)
    plt.loglog(x, y, color='black', lw=1, ls='dashdot')

    # Set limits
    # ax.set_xlim(min(wn)*10, max(wn))
    # ax.set_ylim(1e-5, 1e-1)

    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel('$kD$')
    plt.ylabel('$tke$')
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')

    # Anotations
    # plt.text(x=5e-3, y=2e-1, s='$-5/3$', color='black')
    # plt.text(x=1e-2, y=1, s='$-3$', color='black')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


def plotLogLogSpatialSpectra_list(file, uk_tuple_list, wn_list):
    """
    Generate a loglog plot of a list of 1D spatial signals
    :param file: output file name
    :param tke_tuple_list: list of tuples as (case, uk), where 'case' is the case name [string] and 'uk' is
        the transformed u: uk = FFT(u). [1D numpy array]
    :param wn_list: list of frequencies for the different cases
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    # Show lines
    for i, uk_tuple in enumerate(uk_tuple_list):
        label = uk_tuple[0]
        if 'piD' in label: label = '\pi D'
        uk = uk_tuple[1]
        label = '$'+label+'$'
        color = colors[i]
        plt.loglog(wn_list[i], uk, color=color, lw=0.5, label=label)

    # x, y = loglogLine(p2=(3,1e-4), p1x=1e-2, m=-5/3)
    # plt.loglog(x, y, color='black', lw=1, ls='dotted')
    # x, y = loglogLine(p2=(4, 2e-5), p1x=1e-2, m=-3)
    # plt.loglog(x, y, color='black', lw=1, ls='dashdot')
    # x, y = loglogLine(p2=(4, 2e-5), p1x=1e-2, m=-11/3)
    # plt.loglog(x, y, color='black', lw=1, ls='dashed')

    x, y = loglogLine(p2=(3,1e-7), p1x=1e-2, m=-5/3)
    plt.loglog(x, y, color='black', lw=1, ls='dotted')
    x, y = loglogLine(p2=(4, 1e-9), p1x=1e-2, m=-3)
    plt.loglog(x, y, color='black', lw=1, ls='dashdot')
    x, y = loglogLine(p2=(4, 1e-9), p1x=1e-2, m=-11/3)
    plt.loglog(x, y, color='black', lw=1, ls='dashed')

    # Set limits
    # ax.set_xlim(1e-3, 1.5)
    ax.set_ylim(1e-13, 10)

    fig, ax = makeSquare(fig,ax)
    # ax.xaxis.set_tick_params(labeltop='on')
    ax.tick_params(bottom="on", top="on", which='both')

    # Edit frame, labels and legend
    plt.xlabel('$kD$')
    plt.ylabel('$tke$')
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')

    # Anotations
    # plt.text(x=1e-2, y=1e1, s='$-5/3$', color='black')
    # plt.text(x=1e-2, y=3e5, s='$-3$', color='black')
    # plt.text(x=2e-2, y=4e2, s='$-11/3$', color='black')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


# ------------------------------------------------------ LogLog Time
def plotLogLogTimeSpectra(freqs, uk, file):
    """
    Generate a loglog plot of a time spectra series
    :param freqs: frequency [1D numpy array]
    :param uk: power signal of the time series u [1D numpy array]
    :param file: output file name
    :return: -
    """
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
    ax.xaxis.set_tick_params(labeltop='on')

    # Edit frame, labels and legend
    plt.xlabel(r'$f$')
    plt.ylabel(r'$F(v)$')
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')

    # Anotations
    plt.text(x=2e-3, y=1e-1, s='$-5/3$', color='black')
    plt.text(x=7e-3, y=2e-1, s='$-3$', color='black')
    plt.text(x=1e-2, y=4e-1, s='$-11/3$', color='black')

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


def plotLogLogTimeSpectra_list(file, uk_tuple_list, freqs_list):
    """
    Generate a loglog plot of a list of time spectra series
    :param file: output file name
    :param uk_tuple_list: list of tuples as (case, uk), where 'case' is the case name [string] and 'uk' is
        power signal of the time series u [1D numpy array]
    :param freqs_list: list containing the frequencies [1D numpy array] for each case
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    # Show lines
    for i, uk_tuple in enumerate(uk_tuple_list):
        label = uk_tuple[0]
        print(label)
        if 'pi' in label: label = '\pi'
        uk = uk_tuple[1]
        label = '$'+label+'$'
        color = colors[i]
        plt.loglog(freqs_list[i], uk, color=color, lw=0.5, label=label)

    x, y = loglogLine(p2=(1.e2, 1e-7), p1x=1e-2, m=-5/3)
    plt.loglog(x, y, color='black', lw=1, ls='dotted')
    x, y = loglogLine(p2=(1.2e2, 1e-9), p1x=1e-2, m=-3)
    plt.loglog(x, y, color='black', lw=1, ls='dashdot')
    # x, y = loglogLine(p2=(1e0, 1e-8), p1x=1e-3, m=-11/3)
    # plt.loglog(x, y, color='black', lw=1, ls='dashed')

    # Set limits
    # ax.set_xlim(np.min(freqs_list[0]), 2e-1)
    # ax.set_ylim(1e-8, 1e-1)
    ax.set_xlim(1e-2, 1e2) # Window
    ax.set_ylim(1e-11, 1e-1)

    fig, ax = makeSquare(fig,ax)
    # ax.xaxis.set_tick_params(labeltop='on')
    ax.tick_params(bottom="on", top="on", which='both')

    # Edit frame, labels and legend
    plt.xlabel(r'$f/UD$')
    plt.ylabel(r'$SPS$')
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_edgecolor('white')

    # Anotations
    # plt.text(x=3e-4, y=5e-1, s='$-5/3$', color='black', fontsize=10) # Power
    # plt.text(x=4e-3, y=1e0, s='$-3$', color='black', fontsize=10)
    # plt.text(x=1e-2, y=4e-1, s='$-11/3$', color='black', fontsize=10)
    # plt.text(x=3e-4, y=5e-1, s='$-5/3$', color='black', fontsize=10) # No Power
    # plt.text(x=4e-3, y=1e0, s='$-3$', color='black', fontsize=10)
    # plt.text(x=1e-2, y=4e-1, s='$-11/3$', color='black', fontsize=10)

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


def plotLogLogTimeSpectra_list_cascade(file, uk_tuple_list, freqs_list):
    """
    Same as 'plotLogLogTimeSpectra_list' but the spectras are separated a factor of 10 among them for visualization purposes
    """
    ax = plt.gca()
    fig  = plt.gcf()

    for i, uk_tup in enumerate(uk_tuple_list):
        uk = uk_tup[1] * 10 ** (-i)
        uk_tuple_list[i] = (uk_tuple_list[i][0], uk)

    # Show lines
    for i, uk_tuple in enumerate(uk_tuple_list):
        label = uk_tuple[0]
        print(label)
        if 'pi' in label: label = '\pi'
        if '2D' in label: label = '2\mathrm{D}'
        uk = uk_tuple[1]
        label = '$'+label+'$'
        color = colors[i]
        plt.loglog(freqs_list[i], uk, color=color, lw=0.5, label=label)


    for i in np.arange(4):
        x, y = loglogLine(p2=(1.e2, 1e-13*100**i), p1x=1e-2, m=-5/3)
        plt.loglog(x, y, color='black', lw=0.5, ls='dotted', alpha=0.3)
        x, y = loglogLine(p2=(1.2e2, 1e-16*100**i), p1x=1e-2, m=-3)
        plt.loglog(x, y, color='black', lw=0.5, ls='dashdot', alpha=0.3)

    # Set limits
    ax.set_xlim(1e-2, 7e1) # Window
    ax.set_ylim(9e-16, 1e-1)
    # ax.set_ylim(1e-11, 1e4)


    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    ax.tick_params(bottom="on", top="on", which='both', direction='in')
    plt.xlabel(r'$f/UD$')
    plt.ylabel(r'$\mathrm{PS}\left(v\right)$')
    leg = plt.legend(loc='lower left')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)
    # ax.yaxis.set_ticks([-2, 0, 2])
    ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
    # ax.get_yaxis().set_ticks([], minor=True)

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    plt.clf()
    return

def plotLogLogSpatialSpectra_list_cascade(file, uk_tuple_list, freqs_list):
    """
    Same as 'plotLogLogSpatialSpectra_list' but the spectras are separated a factor of 10 among them for visualization purposes
    """
    ax = plt.gca()
    fig  = plt.gcf()

    for i, uk_tup in enumerate(uk_tuple_list):
        uk = uk_tup[1] * 10 ** (-i)
        uk_tuple_list[i] = (uk_tuple_list[i][0], uk)

    # Show lines
    for i, uk_tuple in enumerate(uk_tuple_list):
        label = uk_tuple[0]
        print(label)
        if 'pi' in label: label = '\pi'
        if '2D' in label: label = '2\mathrm{D}'
        uk = uk_tuple[1]
        label = '$'+label+'$'
        color = colors[i]
        plt.loglog(freqs_list[i], uk, color=color, lw=0.8, label=label)


    for i in np.arange(5):
        x, y = loglogLine(p2=(1e3, 1e-10*10**i), p1x=1e-2, m=-5/3)
        plt.loglog(x, y, color='black', lw=0.5, ls='dotted', alpha=0.3)
        x, y = loglogLine(p2=(1e3, 1e-13*10**i), p1x=1e-2, m=-3)
        plt.loglog(x, y, color='black', lw=0.5, ls='dashdot', alpha=0.3)

    # Set limits
    # ax.set_xlim(2e0, 3e2) # Window
    ax.set_xlim(2e0, 5e2) # Window
    ax.set_ylim(1e-15, 1e-1)
    # ax.set_ylim(1e-11, 1e4)

    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    ax.tick_params(bottom="on", top="on", which='both', direction='in')
    plt.xlabel(r'$\kappa D$')
    leg = plt.legend(loc='lower left')
    leg.get_frame().set_edgecolor('white')
    ax.get_yaxis().set_ticks([])

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


# ------------------------------------------------------ Lumley's Triangle
def plotLumleysTriangle(eta, xi, file):
    """
    Generate a plot of the Reynolds stresses anisotropy tensor in the form of the Lumley's triangle
    :param eta: Invariant of the anisotropy tensor (displayed on the vertical axis) [1D array of points in space, i.e. y triangle coordinates]
    :param xi: Invariant of the anisotropy tensor (displayed on the horizontal axis) [1D array of points in space, i.e. x triangle coordinates]
    :param file: output file name
    :return: -
    """
    ax = plt.gca()
    fig  = plt.gcf()

    x = np.linspace(-1/6, 1/3, 500)
    y = np.sqrt(1/27+2*x**3)

    # Show lines
    plt.plot(x, y, color='black', lw=1.5)
    plt.plot([-1/6,0], [1/6,0], color='black', lw=1.5)
    plt.plot([0,1/3], [0,1/3], color='black', lw=1.5)
    plt.scatter(xi, eta, marker='o', c='green', s=1, linewidths=0.1)

    # Set limits
    ax.set_ylim(0, 0.35)
    ax.set_xlim(-0.2, 0.4)
    # Make figure squared
    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\xi$')
    leg = plt.legend(loc='upper left')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return

def plotLumleysTriangle_list(file, eta_tuple_list, xi_tuple_list):
    """
    Generate a plot of the Reynolds stresses anisotropy tensor in the form of the Lumley's triangle for different cases
    :param file: output file name
    :param eta_tuple_list: list of tuples as (case, eta) for the 'eta' invariant
    :param xi_tuple_list: list of tuples as (case, xi) for the 'xi' invariant
    :return:
    """
    ax = plt.gca()
    fig  = plt.gcf()

    x = np.linspace(-1/6, 1/3, 500)
    y = np.sqrt(1/27+2*x**3)

    # Show lines
    plt.plot(x, y, color='black', lw=0.5)
    plt.plot([-1/6,0], [1/6,0], color='black', lw=0.5)
    plt.plot([0,1/3], [0,1/3], color='black', lw=0.5)

    # Show lines
    i = 0
    for eta in eta_tuple_list:
        label = eta[0]
        if 'piD' in label: label = '\pi'
        else: label = label[:-1]
        eta = eta[1]
        xi = xi_tuple_list[i][1]
        label = '$'+label+'$'
        plt.scatter(xi, eta, marker=markers[i], c=colors[i], s=10, linewidths=0.1, label=label, edgecolor = 'black')
        i += 1

    # Set limits
    ax.set_ylim(0, 0.35)
    ax.set_xlim(-0.2, 0.4)
    # Make figure squared
    fig, ax = makeSquare(fig,ax)

    # Edit frame, labels and legend
    plt.xlabel(r'$\xi$')
    plt.ylabel('$ \eta $', rotation=0)
    leg = plt.legend(loc='lower right')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)
    ax.xaxis.set_ticks([-0.2,-0.1,0,0.1,0.2,0.3,0.4])
    ax.tick_params( direction='in', length=2)
    ax.tick_params(bottom="on", top="on", right='on',which='both', direction='in')
    # plt.minorticks_off()

    # Show plot and save figure
    plt.show()
    plt.savefig(file, transparent=True, bbox_inches='tight')
    return


# ------------------------------------------------------ GC plots
# def error_order(file, x, y):
#     """
#     Generate a loglog plot of a time spectra series using the matplotlib library given the arguments
#     """
#     # Basic definitions
#     ax = plt.gca()
#     fig  = plt.gcf()
#
#     plt.loglog(x, y, color='b', lw=0.5)
#
#     x, y = loglogLine(p2=(np.max(x), np.max(y)), p1x=np.min(x), m=2)
#     plt.loglog(x, y, color='black', lw=1, ls='dotted')
#     x, y = loglogLine(p2=(np.max(x), np.max(y)), p1x=np.min(x), m=1)
#     plt.loglog(x, y, color='black', lw=1, ls='dotted')
#     # x, y = loglogLine(p2=(1.2e2, 1e-9), p1x=1e-2, m=-3)
#     # plt.loglog(x, y, color='black', lw=1, ls='dashdot')
#     # x, y = loglogLine(p2=(1e0, 1e-8), p1x=1e-3, m=-11/3)
#     # plt.loglog(x, y, color='black', lw=1, ls='dashed')
#
#     # Set limits
#     # ax.set_xlim(np.min(freqs_list[0]), 2e-1)
#     # ax.set_ylim(1e-8, 1e-1)
#     # ax.set_xlim(1e-2, 1e2) # Window
#     # ax.set_ylim(1e-11, 1e-1)
#
#     # fig, ax = makeSquare(fig,ax)
#     # ax.xaxis.set_tick_params(labeltop='on')
#     # ax.tick_params(bottom="on", top="on", which='both')
#
#     # Edit frame, labels and legend
#
#     # Show plot and save figure
#     plt.show()
#     plt.savefig(file, transparent=True, bbox_inches='tight')
#     return


# ------------------------------------------------------ Utils
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
