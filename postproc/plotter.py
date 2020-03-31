# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Functions to plot 2D colormaps and CL-t graphs.
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.patches as patches
import matplotlib.colorbar as colorbar
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('text', usetex=True )
plt.rc('font',family = 'sans-serif',  size=16) # use 13(JFM) or 16(SNH)
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
plt.rcParams['animation.ffmpeg_path'] = r"/usr/bin/ffmpeg"
mpl.rcParams['axes.linewidth'] = 0.5

# plt.switch_backend('AGG') #png
# plt.switch_backend('PS')
plt.switch_backend('PDF') #pdf
# plt.switch_backend('PS')

# colors = ['black', 'orange', 'cyan', 'green', 'blue', 'red', 'magenta', 'yellow']
colors = ['orange', 'cyan', 'green', 'blue', 'red', 'magenta', 'yellow']
markers = ['|', 's', '^', 'v', 'x', 'o', '*']
# markers = ['s', '^', 'v', 'x', 'o', '*']

styles = ['-', 'dashed']

# Functions
# ------------------------------------------------------
def plot_2D(u, file='test.pdf', **kwargs):
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
	from matplotlib.ticker import FormatStrFormatter, MultipleLocator, FuncFormatter
	contour = kwargs.get('contour', False)
	contour_color = kwargs.get('contour_color', 'black')
	contourf = kwargs.get('contourf', True)
	levels = kwargs.get('levels', 50)
	lim = kwargs.get('lim', [np.min(u), np.max(u)])
	lim_max = kwargs.get('lim_max', lim)
	cmap = kwargs.get('cmap', 'Blues')
	scaling = kwargs.get('scaling', 1)
	shift = kwargs.get('shift', (0,0))
	xwindow = kwargs.get('xwindow', None)
	ywindow = kwargs.get('ywindow', None)
	title = kwargs.get('title', '')
	n_ticks = kwargs.get('n_ticks', 10)
	n_decimals = kwargs.get('n_decimals', 2)
	case = kwargs.get('case', None)
	annotate = kwargs.get('annotate', False)
	eps = kwargs.get('eps', 0.0001)
	cbar_flag = kwargs.get('cbar_flag', False)
	font_size = kwargs.get('font_size', 16)
	dpi = kwargs.get('dpi', 100)
	pad_inches = kwargs.get('pad_inches', 0)

	plt.rc('font', size=font_size)  # use 13(JFM) or 16(SNH)
	mpl.rc('xtick', labelsize=font_size)
	mpl.rc('ytick', labelsize=font_size)
	N, M = u.shape[0], u.shape[1]

	# Create uniform grid
	if 'grid' in kwargs:
		grid = kwargs['grid']
		x, y = grid[0] / scaling, grid[1] / scaling
	elif 'x' in kwargs and 'y' in kwargs:
		x = np.transpose(kwargs.get('x')) / scaling + shift[0]
		y = np.transpose(kwargs.get('y')) / scaling + shift[1]
		x, y = np.meshgrid(x, y)
	elif 'x_lims' in kwargs and 'y_lims' in kwargs:
		xlims = kwargs.get('x_lims')
		ylims = kwargs.get('y_lims')
		x, y = np.linspace(xlims[0] / scaling, xlims[1] / scaling, N), np.linspace(ylims[0] / scaling, ylims[1] / scaling, M)
		x, y = x + shift[0], y + shift[1]
		x, y = np.meshgrid(x, y)
	else:
		xmin, xmax = 0, N - 1
		ymin, ymax = -M / 2, M / 2 - 1
		x, y = np.linspace(xmin / scaling, xmax / scaling, N), np.linspace(ymin / scaling, ymax / scaling, M)
		x, y = x + shift[0], y + shift[1]
		x, y = np.meshgrid(x, y)

	# Matplotlib definitions
	fig, ax = plt.subplots(1, 1)

	# Create contourf given a normalized (norm) colormap (cmap)
	if lim[0] < 0 and lim[1] > 0:
		ll = np.linspace(lim[0], -eps, int(levels / 2))
		rr = np.linspace(eps, lim[1], int(levels / 2))
		lvls = np.append(ll, rr)
		if contour:
			extra_levels = 5
			dl = lvls[1] - lvls[0]
			clvls = np.append(lvls, np.linspace(lim[1] + dl, lim[1] + extra_levels * dl, extra_levels))
	else:
		clvls = levels
		lvls = np.linspace(lim[0], lim[1], levels+1)

	u = u.T
	if xwindow is not None:
		x_args = np.where(np.logical_and(np.any(x > xwindow[0], axis=0), np.any(x<xwindow[1], axis=0)))[0]
		x = x[:, x_args]
		y = y[:, x_args]
		u = u[:, x_args]
	if ywindow is not None:
		y_args = np.where(np.logical_and(np.any(y > ywindow[0], axis=1), np.any(y < ywindow[1], axis=1)))[0]
		x = x[y_args, :]
		y = y[y_args, :]
		u = u[y_args, :]

	if contour:
		ax.contour(x, y, u, clvls, linewidths=0.1, colors=contour_color)
	if contourf:
		cf = ax.contourf(x, y, u, lvls, vmin=lim_max[0], vmax=lim_max[1], cmap=cmap, extend='both')

	# Format figure
	ax.set_aspect(1)
	plt.xlim(np.min(x), np.max(x))
	plt.ylim(np.min(y), np.max(y))
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
	# ax.tick_params(axis='x', labelcolor='white')
	ax.xaxis.set_zorder(99999)
	ax.yaxis.set_zorder(99999)

	if case == 'taylor-green':
		ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi))
		ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
		ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi))
		ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
		ax.xaxis.set_ticks([xlims[0], xlims[1]+xlims[0], xlims[1]])
		ax.yaxis.set_ticks([ylims[0], ylims[1]+ylims[0], ylims[1]])

	elif case == 'cylinder':
		# ax.yaxis.set_ticks([-2, 0, 2])
		grey_color = '#dedede'
		cyl = patches.Circle((0, 0), 0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
		ax.add_patch(cyl)

	# -- Add colorbar
	if cbar_flag:
		divider = make_axes_locatable(ax)
		if case == 'cylinder':
			cax = divider.append_axes("right", size="5%", pad=0.0, aspect=15)
			# cax = divider.append_axes("right", size="5%", pad=0.15)
		else:
			cax = divider.append_axes("right", size="5%", pad=0.15)
		if lim[0] < 0:
			tick1 = np.linspace(lim[0], 0, n_ticks / 2)
			dl = tick1[1] - tick1[0]
			tick2 = np.linspace(dl, lim[1], n_ticks / 2 - 1)
			ticks = np.append(tick1, tick2)
		else:
			ticks = np.linspace(lim[0], lim[1], n_ticks + 1)
		norm = mpl_colors.Normalize(vmin=lim[0], vmax=lim[1])
		norm = mpl_colors.Normalize(vmin=np.min(u), vmax=np.max(u))
		cbar = fig.colorbar(cf, cax=cax, extend='both', ticks=ticks, norm=norm)
		fmt_str = r'${:.' + str(n_decimals) + 'f}$'
		cbar.ax.set_yticklabels([fmt_str.format(t) for t in ticks])
		cbar.ax.yaxis.set_tick_params(pad=5, direction='out', size=1)  # your number may vary
		cbar.ax.set_title(title, x=1, y=1.02, loc='left', size=12)

	# Add annotation if desired
	if annotate:
		str_annotation = max_min_loc(u, x, y)
		# print(str_annotation)
		ann_ax = fig.add_subplot(133)
		ann_ax.axis('off')
		ann_ax.annotate(str_annotation, (0, 0),
						xycoords="axes fraction", va="center", ha="center",
						bbox=dict(boxstyle="round, pad=1", fc="w"))

	# Show, save and close figure

	plt.savefig(file, transparent=True, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)
	# plt.draw()
	# plt.clf()
	return

def animate_2Dx2(a, b, file, **kwargs):
	plt.rc('font', size=10)
	mpl.rc('xtick', labelsize=9)
	mpl.rc('ytick', labelsize=9)
	mpl.rcParams['axes.linewidth'] = 0.1

	global c1, c2, cf1, cf2, cf

	def anim(i):
		global c1, c2, cf1, cf2, cf
		for c in cf1.collections:
			c.remove()  # removes only the contours, leaves the rest intact
		for c in cf2.collections:
			c.remove()  # removes only the contours, leaves the rest intact
		for c in c1.collections:
			c.remove()  # removes only the contours, leaves the rest intact
		for c in c2.collections:
			c.remove()  # removes only the contours, leaves the rest intact

		c1 = ax1.contour(x.T, y.T, a[i].T, clvls, linewidths=0.05, colors='k')
		c2 = ax2.contour(x.T, y.T, b[i].T, clvls, linewidths=0.05, colors='k')
		cf1 = ax1.contourf(x.T, y.T, a[i].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
		cf2 = ax2.contourf(x.T, y.T, b[i].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')

		title = r'$t = ' + '{:.2f}'.format(time[i]) + '$'
		# ax1.set_title(title, size=11, y=1.06, color='white')
		return [c1, c2, cf1, cf2]

	time = kwargs.get('time', np.arange(len(a)))
	levels = kwargs.get('levels', 50)
	lim = kwargs.get('lim', [np.min(a[0]), np.max(a[0])])
	cmap = kwargs.get('cmap', 'Blues')
	scaling = kwargs.get('scaling', 1)
	xshift = kwargs.get('xshift', 0)
	yshift = kwargs.get('yshift', 0)
	field_name = kwargs.get('field_name', '')
	names = kwargs.get('names', ['',''])
	n_ticks = kwargs.get('n_ticks', 20)
	n_decimals = kwargs.get('n_decimals', 2)
	fps = kwargs.get('fps', 10)
	dpi = kwargs.get('dpi', 300)
	pad_inches = kwargs.get('pad_inches', 0)
	N, M = a[0].shape[0], a[0].shape[1]

	# Create uniform grid
	if 'grid' in kwargs:
		grid = kwargs['grid']
		x, y = grid[0] / scaling, grid[1] / scaling
	elif 'x' in kwargs and 'y' in kwargs:
		x = np.transpose(kwargs.get('x')) / scaling + xshift
		y = np.transpose(kwargs.get('y')) / scaling + yshift
		x, y = np.meshgrid(x, y)
	else:
		xmin, xmax = 0, N - 1
		ymin, ymax = -M / 2, M / 2 - 1
		x, y = np.linspace(xmin / scaling, xmax / scaling, N), np.linspace(ymin / scaling, ymax / scaling, M)
		x, y = x + xshift, y + yshift
		x, y = np.meshgrid(x, y)

	fig, ax = plt.subplots(2, 1)
	ax1, ax2 = ax[0], ax[1]

	if lim[0] < 0:
		clvls = levels
	else:
		extra_levels = 5
		dl = levels[1] - levels[0]
		clvls = np.append(levels, np.linspace(lim[1] + dl, lim[1] + extra_levels * dl, extra_levels))
	norm = mpl_colors.Normalize(vmin=lim[0], vmax=lim[1])

	c1 = ax1.contour(x.T, y.T, a[0].T, clvls, linewidths=0.05, colors='k')
	c2 = ax2.contour(x.T, y.T, b[0].T, clvls, linewidths=0.05, colors='k')
	cf1 = ax1.contourf(x.T, y.T, a[0].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
	cf2 = ax2.contourf(x.T, y.T, b[0].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')

	cf = [c1, c2, cf1, cf2]

	# Format figure
	# ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', labelbottom='off', length=2)
	# ax1.set_xticklabels([])
	# ax2.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

	# ax1.tick_params(bottom="off", top="off", right="off", which='both',)
	# ax1.set_xticklabels([])
	# ax2.tick_params(bottom="off", top="off", right="off", which='both',)
	# ax2.set_xticklabels([])

	ax1.set_aspect(1)
	ax2.set_aspect(1)
	plt.xlim(np.min(x), np.max(x))
	plt.ylim(np.min(y), np.max(y))
	# ax1.yaxis.set_ticks([-2, 0, 2])
	ax1.xaxis.set_ticks([])
	ax1.yaxis.set_ticks([])
	# ax2.yaxis.set_ticks([-2, 0, 2])
	ax2.xaxis.set_ticks([])
	ax2.yaxis.set_ticks([])

	# Set title, circles and text
	title = r'$t = ' + '{:.2f}'.format(time[0]) + '$'
	# ax1.set_title(title, size=11, y=1.06, color='white')
	grey_color = '#dedede'
	cyl1 = patches.Circle((0, 0), 0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
	cyl2 = patches.Circle((0, 0), 0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
	ax1.add_patch(cyl1)
	ax2.add_patch(cyl2)
	ax1.text(-1, 2.3, names[0], fontsize=11)
	ax2.text(-1, 2.3, names[1], fontsize=11)
	fig.tight_layout()
	plt.subplots_adjust(bottom=0.03, hspace=0.05)

	# Add colorbar
	# if lim[0] < 0:
	#	 tick1 = np.linspace(lim[0], 0, n_ticks / 2)
	#	 dl = tick1[1] - tick1[0]
	#	 tick2 = np.linspace(dl, lim[1], n_ticks / 2 - 1)
	#	 ticks = np.append(tick1, tick2)
	# else:
	#	 ticks = np.linspace(lim[0], lim[1], n_ticks + 1)
	# cbar_ax = plt.colorbar(cf2, ax=[ax1, ax2], extend='both', norm=norm).ax
	# cbar_ax.set_title(field_name, y=1.02, loc='left', size=12)
	# fmt_str = r'${:.' + str(n_decimals) + 'f}$'
	# cbar_ax.set_yticklabels([fmt_str.format(t) for t in ticks])
	# cbar_ax.yaxis.set_tick_params(pad=5, direction='out', size=1)  # your number may vary
	# cbar_ax.set_title(field_name, x=1, y=1.02, loc='left', size=12)

	# Animate
	writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
	anim = animation.FuncAnimation(fig, anim, frames=len(a))
	savefig_kwargs = {'transparent':True, 'pad_inches':pad_inches}
	anim.save(file, writer=writer, dpi=dpi, savefig_kwargs=savefig_kwargs)
	return

def animate_2Dx4(a, b, c, d, file, **kwargs):
	import matplotlib.gridspec as gridspec
	plt.rc('font', size=9)
	mpl.rc('xtick', labelsize=9)
	mpl.rc('ytick', labelsize=9)
	global c1, c2, c3, c4, cf1, cf2, cf3, cf4, cf

	def anim(i):
		global c1, c2, c3, c4, cf1, cf2, cf3, cf4, cf
		for cc in cf1.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		for cc in cf2.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		for cc in cf3.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		for cc in cf4.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		for cc in c1.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		for cc in c2.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		for cc in c3.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		for cc in c4.collections:
			cc.remove()  # removes only the contours, leaves the rest intact
		print('{:4d}'.format(i))

		c1 = ax1.contour(x.T, y.T, a[i].T, clvls, linewidths=0.05, colors='k')
		c2 = ax2.contour(x.T, y.T, b[i].T, clvls, linewidths=0.05, colors='k')
		c3 = ax3.contour(x.T, y.T, c[i].T, clvls, linewidths=0.05, colors='k')
		c4 = ax4.contour(x.T, y.T, d[i].T, clvls, linewidths=0.05, colors='k')
		cf1 = ax1.contourf(x.T, y.T, a[i].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
		cf2 = ax2.contourf(x.T, y.T, b[i].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
		cf3 = ax3.contourf(x.T, y.T, c[i].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
		cf4 = ax4.contourf(x.T, y.T, d[i].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')

		title = r'$t = ' + '{:.2f}'.format(time[i]) + '$'
		plt.suptitle(title, size=10, y=0.84)
		return [c1, c2, c3, c4, cf1, cf2, cf3, cf4]

	time = kwargs.get('time', np.arange(len(a)))
	levels = kwargs.get('levels', 50)
	lim = kwargs.get('lim', [np.min(a[0]), np.max(a[0])])
	cmap = kwargs.get('cmap', 'Blues')
	scaling = kwargs.get('scaling', 1)
	xshift = kwargs.get('xshift', 0)
	yshift = kwargs.get('yshift', 0)
	field_name = kwargs.get('field_name', '')
	names = kwargs.get('names', ['',''])
	n_ticks = kwargs.get('n_ticks', 20)
	n_decimals = kwargs.get('n_decimals', 2)
	fps = kwargs.get('fps', 10)
	dpi = kwargs.get('dpi', 300)
	N, M = a[0].shape[0], a[0].shape[1]

	# Create uniform grid
	if 'grid' in kwargs:
		grid = kwargs['grid']
		x, y = grid[0] / scaling, grid[1] / scaling
	elif 'x' in kwargs and 'y' in kwargs:
		x = np.transpose(kwargs.get('x')) / scaling + xshift
		y = np.transpose(kwargs.get('y')) / scaling + yshift
		x, y = np.meshgrid(x, y)
	else:
		xmin, xmax = 0, N - 1
		ymin, ymax = -M / 2, M / 2 - 1
		x, y = np.linspace(xmin / scaling, xmax / scaling, N), np.linspace(ymin / scaling, ymax / scaling, M)
		x, y = x + xshift, y + yshift
		x, y = np.meshgrid(x, y)

	# fig, ax = plt.subplots(2, 2, constrained_layout=True)
	fig, ax = plt.subplots(2, 2)
	ax1, ax2, ax3, ax4 = ax[0,0], ax[1,0], ax[0,1], ax[1,1]
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.0, hspace=-0.6)

	if lim[0] < 0:
		clvls = levels
	else:
		extra_levels = 5
		dl = levels[1] - levels[0]
		clvls = np.append(levels, np.linspace(lim[1] + dl, lim[1] + extra_levels * dl, extra_levels))
	norm = mpl_colors.Normalize(vmin=lim[0], vmax=lim[1])

	c1 = ax1.contour(x.T, y.T, a[0].T, clvls, linewidths=0.05, colors='k')
	c2 = ax2.contour(x.T, y.T, b[0].T, clvls, linewidths=0.05, colors='k')
	c3 = ax3.contour(x.T, y.T, c[0].T, clvls, linewidths=0.05, colors='k')
	c4 = ax4.contour(x.T, y.T, d[0].T, clvls, linewidths=0.05, colors='k')
	cf1 = ax1.contourf(x.T, y.T, a[0].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
	cf2 = ax2.contourf(x.T, y.T, b[0].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
	cf3 = ax3.contourf(x.T, y.T, c[0].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
	cf4 = ax4.contourf(x.T, y.T, d[0].T, levels, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')

	cf = [c1, c2, c3, c4, cf1, cf2, cf3, cf4]

	# Format figure
	ax1.tick_params(bottom="on", top="on", right="on", which='both', direction='in', labelbottom='off', length=2)
	ax3.tick_params(bottom="on", top="on", right="on", which='both', direction='in', labelbottom='off', labelleft='off', length=2)
	ax1.set_xticklabels([])
	ax3.set_xticklabels([])
	ax3.set_yticklabels([])
	ax2.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
	ax4.tick_params(bottom="on", top="on", right="on", which='both', direction='in', labelleft='off', length=2)
	ax4.set_yticklabels([])
	ax1.set_aspect(1)
	ax2.set_aspect(1)
	ax3.set_aspect(1)
	ax4.set_aspect(1)
	plt.xlim(np.min(x), np.max(x))
	plt.ylim(np.min(y), np.max(y))
	ax1.yaxis.set_ticks([-2, 0, 2])
	ax2.yaxis.set_ticks([-2, 0, 2])
	ax2.xaxis.set_ticks([0,2,4,6,8,10])
	ax4.xaxis.set_ticks([0,2,4,6,8,10])

	# Set title, circles and text
	title = r'$t = ' + '{:.2f}'.format(time[0]) + '$'
	plt.suptitle(title, size=10, y=0.84)
	grey_color = '#dedede'
	cyl1 = patches.Circle((0, 0), 0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
	cyl2 = patches.Circle((0, 0), 0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
	cyl3 = patches.Circle((0, 0), 0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
	cyl4 = patches.Circle((0, 0), 0.5, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
	ax1.add_patch(cyl1)
	ax2.add_patch(cyl2)
	ax3.add_patch(cyl3)
	ax4.add_patch(cyl4)
	ax1.text(-1, 2.3, names[0])
	ax2.text(-1, 2.3, names[1])
	ax3.text(-1, 2.3, names[2])
	ax4.text(-1, 2.3, names[3])

	# Add colorbar
	# if lim[0] < 0:
	#	 tick1 = np.linspace(lim[0], 0, n_ticks / 2)
	#	 dl = tick1[1] - tick1[0]
	#	 tick2 = np.linspace(dl, lim[1], n_ticks / 2 - 1)
	#	 ticks = np.append(tick1, tick2)
	# else:
	#	 ticks = np.linspace(lim[0], lim[1], n_ticks + 1)
	# cbar_ax = plt.colorbar(cf2, ax=[ax1, ax2], extend='both', norm=norm).ax
	# cbar_ax.set_title(field_name, y=1.02, loc='left', size=12)
	# fmt_str = r'${:.' + str(n_decimals) + 'f}$'
	# cbar_ax.set_yticklabels([fmt_str.format(t) for t in ticks])
	# cbar_ax.yaxis.set_tick_params(pad=5, direction='out', size=1)  # your number may vary
	# cbar_ax.set_title(field_name, x=1, y=1.02, loc='left', size=12)

	# Animate
	writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
	anim = animation.FuncAnimation(fig, anim, frames=len(a))
	anim.save(file, writer=writer, dpi=dpi)
	return

def scatter(x, y, file='test.pdf', **kwargs):
	x_label = kwargs.get('x_label', None)
	y_label = kwargs.get('y_label', None)

	fig, ax = plt.subplots(1, 1)
	ax.scatter(x, y, color='k', s=0.1)

	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
	fig, ax = makeSquare(fig,ax)
	plt.xlabel(r'$' + x_label + '$')
	plt.ylabel(r'$' + y_label + '$')
	ax.xaxis.set_zorder(99999)
	ax.yaxis.set_zorder(99999)
	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	return


def density2D(x, y, file='test.pdf', nbins=20, **kwargs):
	from matplotlib.ticker import FormatStrFormatter
	import matplotlib.ticker as mticker
	from scipy.stats import kde

	x_label = kwargs.get('x_label', None)
	y_label = kwargs.get('y_label', None)
	x_lims = kwargs.get('x_lims', None)
	y_lims = kwargs.get('y_lims', None)
	cmap = kwargs.get('cmap', 'gist_heat')
	n_decimals = kwargs.get('n_decimals', (2,3))

	k = kde.gaussian_kde(np.array([x.flatten(),y.flatten()]))
	xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
	zi = k(np.vstack([xi.flatten(), yi.flatten()]))

	fig, ax = plt.subplots(1, 1)
	ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
	ax.contour(xi, yi, zi.reshape(xi.shape), linewidths=0.5, cmap=cmap+'_r')

	fig, ax = makeSquare(fig,ax)
	plt.xlabel(r'$' + x_label + '$')
	plt.ylabel(r'$' + y_label + '$',labelpad=-7) #labelpad=-5 for uv
	if x_lims is not None:
		plt.xlim(x_lims[0], x_lims[1])
	if y_lims is not None:
		plt.ylim(y_lims[0], y_lims[1])


	fmt_str = r'${:.' + str(n_decimals) + 'f}$'
	# ax.set_xtickslabels([fmt_str.format(t) for t in [-0.02, 0.00, 0.02]])
	# ax.set_ytickslabels([fmt_str.format(t) for t in [0.00, 0.05, 0.01, 0.015]])
	ax.tick_params(bottom="on", top="on", right="on", which='major', direction='in', length=2, color='black')
	ax.xaxis.set_major_formatter(FormatStrFormatter('$%.'+str(n_decimals[0])+'f$'))
	ax.yaxis.set_major_formatter(FormatStrFormatter('$%.'+str(n_decimals[1])+'f$'))
	# ax.yaxis.set_major_locator(mticker.FixedLocator([0.00, 0.005, 0.01, 0.015])) # for normal
	ax.yaxis.set_major_locator(mticker.FixedLocator([-0.01, 0.00, 0.01]))   # for shear stress
	ax.xaxis.set_major_locator(mticker.FixedLocator([-0.02, 0.00, 0.02]))
	# plt.set_yticks=[0.00, 0.005, 0.01, 0.015]
	# plt.set_yticklabels=[0.00, 0.005, 0.01, 0.015]
	# plt.set_xticks=[-0.02, 0.00, 0.02]
	# plt.set_xticklabels=[-0.02, 0.00, 0.02]
	ax.xaxis.set_zorder(99999)
	ax.yaxis.set_zorder(99999)
	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight', pad_inches=0)
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
	plt.rc('font', family='sans-serif', size=15)
	mpl.rc('xtick', labelsize=15)
	mpl.rc('ytick', labelsize=15)
	# mpl.rcParams["contour.negative_linestyle"] = 'dotted'

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
	r = ax.contour(x, y, u, lvls, colors='k')
	for line, lvl in zip(r.collections, r.levels):
		if lvl < 0:
			line.set_linestyle('--')
			line.set_dashes([(0, (4.0, 4.0))])
			line.set_linewidth(0.4)
		else:
			line.set_linewidth(0.6)


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

	cyl = patches.Circle((0, 0), radius=0.5, linewidth=0.5, edgecolor='black', facecolor='white', zorder=10, alpha=1)
	ax.add_patch(cyl)

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
	# plt.show()
	plt.clf()
	return


def plot2D_circulation(u, cmap, lvls, lim, file, **kwargs):
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
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

	grey_color = '#dedede'
	cyl = patches.Circle((0, 0), radius=0.5, linewidth=0.5, edgecolor='black', facecolor='white', zorder=10, alpha=1)
	rect = patches.Rectangle((0.55, -0.8), 1.5, 1.6, linewidth=0.5, edgecolor='purple', facecolor='none')
	ax.add_patch(cyl)
	# ax.add_patch(rect)

	# Show, save and close figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
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
	if not 'x_lim' in kwargs:
		xmin, xmax = 0, N-1
	else:
		xmin, xmax = kwargs['x'][0], kwargs['x'][1]
	if not 'y_lim' in kwargs:
		ymin, ymax = -M/2, M/2-1
	else:
		ymin, ymax = kwargs['y'][0], kwargs['y'][1]
	annotate = kwargs.get('annotate', False)
	scaling = kwargs.get('scaling', 1)
	xshift = kwargs.get('xshift', 0)
	yshift = kwargs.get('yshift', 0)

	# Uniform grid generation
	if 'x' not in kwargs and 'y' not in kwargs:
		x = np.linspace(xmin/scaling, xmax/scaling, N)
		y = np.linspace(ymin/scaling, ymax/scaling, M)
		x, y = x + xshift, y + yshift
		x, y = np.meshgrid(x, y)
	elif 'x' in kwargs and 'y' in kwargs:
		x = np.transpose(kwargs.get('x'))/scaling
		y = np.transpose(kwargs.get('y'))/scaling
	else:
		raise ValueError('Pass both x and y, or none.')

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

	# -- Set title, circles and text
	grey_color = '#dedede'
	cyl = patches.Circle((0, 0), 0.51, linewidth=0.2, edgecolor='black', facecolor=grey_color, zorder=9999)
	ax.add_patch(cyl)

	# Show, save and close figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	# plt.draw()
	# plt.clf()
	return


def plot2Dseparation(u, file, **kwargs):
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
	scaling = kwargs.get('scaling', 1)
	ptype = kwargs.get('ptype', 'contourf')
	xshift = kwargs.get('xshift', 0)
	yshift = kwargs.get('yshift', 0)
	cmap = kwargs.get('cmap', 'seismic')
	lvls = kwargs.get('lvls', 50)
	lim = kwargs.get('lim', [np.min(u), np.max(u)])
	if not 'grid' in kwargs:
		xmin, xmax = 0, N-1
		ymin, ymax = -M / 2, M / 2 - 1
		x, y = np.linspace(xmin / scaling, xmax / scaling, N), np.linspace(ymin / scaling, ymax / scaling, M)
		x, y = x + xshift, y + yshift
		x, y = np.meshgrid(x, y)
	else:
		grid = kwargs['grid']
		x, y = grid[0]/scaling, grid[1]/scaling

	# Matplotlib definitions
	fig = plt.gcf()
	ax = plt.gca()

	# Create contourf given a normalized (norm) colormap (cmap)
	norm = colors.Normalize(vmin=lim[0], vmax=lim[1])
	# lvls = np.linspace(lim[0], lim[1], lvls + 1)
	if ptype == 'contourf':
		# ax.contour(x, y, u, lvls, linewidths=0.2, colors='k')
		# cf = ax.contourf(x.T, y.T, u.T, levels=lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap, extend='both')
		cf = ax.contourf(x.T, y.T, u.T, levels=lvls, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)
	else:
		cf = ax.pcolormesh(x.T, y.T, u.T, vmin=lim[0], vmax=lim[1], norm=norm, cmap=cmap)

	# Scale contourf and set limits
	plt.axis('scaled')
	plt.xlim(np.min(x), np.max(x))
	plt.ylim(np.min(y), np.max(y))
	print(np.min(x), np.max(x))
	print(np.min(y), np.max(y))
	ax.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)

	# Add cylinder
	grey_color = '#dedede'
	cyl = patches.Circle((0, 0),scaling/2, linewidth=0.4, edgecolor='purple', facecolor='None')
	ax.add_patch(cyl)

	# Colormap
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# v = np.linspace(lim[0], lim[1], 10, endpoint=True)
	# c = mpl.cm.get_cmap(cmap)
	# c.set_under('r')
	# c.set_over('b')
	# plt.colorbar(cf, cax=cax, norm=norm, cmap=c, ticks=v, boundaries=v)

	# Show, save and close figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
	return

# ------------------------------------------------------
def two_point_correlations_single(a, fname):
	n_points = len(a)
	fig = plt.gcf()
	ax = plt.gca()

	for i, b in enumerate(a):
		point_str = b[0]
		n_cases = len(b[1])
		print(point_str)
		for j, t in enumerate(b[1]):
			case_name, c, d = t[0], t[1], t[2]
			d[0]=0.011111
			if 'pi' in case_name:
				case_name = '\pi'
				max_d = np.max(d)
			ax.plot(d, c, color=colors[j], lw=1.5, label='$'+case_name+'$', marker=markers[j], markevery=0.05, markersize=4)

	leg1 = ax.legend(loc='lower left')
	leg1.get_frame().set_edgecolor('black')
	leg1.get_frame().set_facecolor('white')
	leg1.get_frame().set_linewidth(0.5)
	leg1.get_frame().set_alpha(0.5)

	ax.set_ylabel(r'$\left\langle v_1, v_2 \right\rangle$')
	ax.set_xlabel(r'$\log d/D$')

	ax.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	ax.set_xscale('log', nonposx='clip')
	ax.set_xlim(1.1e-2,max_d)
	ax.set_ylim(0.48,1.02)
	ax.yaxis.set_ticks([0.5,0.6,0.7,0.8,0.9,1.0])

	fig, ax = makeSquare(fig,ax)
	plt.savefig(fname, transparent=True, bbox_inches='tight')
	return


def two_point_correlations(a, fname):
	from matplotlib.gridspec import GridSpec

	n_points = len(a)
	fig = plt.figure()
	gs = GridSpec(2, 2)
	ax = []
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
	ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
	ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
	ax.extend([ax1, ax2, ax3, ax4])

	for i, b in enumerate(a):
		point_str = b[0]
		n_cases = len(b[1])
		print(point_str)
		print(b)
		for j, t in enumerate(b[1]):
			case_name, c, d = t[0], t[1], t[2]
			d[0]=0.011111
			if 'pi' in case_name:
				case_name = '\pi'
				max_d = np.max(d)
			ax[i].plot(d, c, color=colors[j], lw=1, label='$'+case_name+'$')

	leg1 = ax1.legend(loc='lower left')
	leg1.get_frame().set_edgecolor('black')
	leg1.get_frame().set_facecolor('white')
	leg1.get_frame().set_linewidth(0.5)
	leg1.get_frame().set_alpha(0.85)

	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.setp(ax4.get_yticklabels(), visible=False)

	ax1.set_ylabel(r'$\left\langle v_1, v_2 \right\rangle$')
	ax3.set_ylabel(r'$\left\langle v_1, v_2 \right\rangle$')
	ax3.set_xlabel(r'$\log d/D$')
	ax4.set_xlabel(r'$\log d/D$')

	for q in ax:
		q.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
		q.set_xlim(xmax=max_d)
		q.set_ylim(0.5, 1.1)
		# q.set_xscale('log', nonposx='clip')

	fig_size = fig.get_size_inches()
	fig.set_size_inches(fig_size[1] * 1.1, fig_size[1] * 1.1)
	fig.tight_layout()
	plt.savefig(fname, transparent=True, bbox_inches='tight')
	return


def two_point_correlations_3_horizontal(a, fname):
	from matplotlib.gridspec import GridSpec
	plt.rc('font', family='sans-serif', size=14)  # use 13 for squared double columns figures
	mpl.rc('xtick', labelsize=14)
	mpl.rc('ytick', labelsize=14)

	n_points = len(a)
	fig = plt.figure()
	gs = GridSpec(1, 3)
	ax = []
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
	ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
	ax.extend([ax1, ax2, ax3])

	for i, b in enumerate(a):
		point_str = b[0]
		n_cases = len(b[1])
		print(point_str)
		for j, t in enumerate(b[1]):
			case_name, c, d = t[0], t[1], t[2]
			d[0]=0.011111
			if 'pi' in case_name:
				case_name = '\pi'
				max_d = np.max(d)
			every = [4,2,2,1,1]
			ax[i].plot(d, c, color=colors[j], lw=1.5, label='$'+case_name+'$', marker=markers[j], markevery=0.05, markersize=4)

	leg1 = ax1.legend(loc='upper right')
	leg1.get_frame().set_edgecolor('black')
	leg1.get_frame().set_facecolor('white')
	leg1.get_frame().set_linewidth(0.5)
	leg1.get_frame().set_alpha(0.5)

	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.setp(ax3.get_yticklabels(), visible=False)

	ax1.set_ylabel(r'$\left\langle v_1, v_2 \right\rangle$')
	ax1.set_xlabel(r'$\log d/D$')
	ax2.set_xlabel(r'$\log d/D$')
	ax3.set_xlabel(r'$\log d/D$')

	for q in ax:
		q.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
		# print(max_d)
		# q.set_ylim(0.5, 1.1)
		q.set_xscale('log', nonposx='clip')
		q.set_xlim(1.1e-2,max_d)

	fig_size = fig.get_size_inches()
	fig.set_size_inches(fig_size[1] * 2, fig_size[1] * 2/2.8)
	fig.tight_layout()
	plt.savefig(fname, transparent=True, bbox_inches='tight')
	return


def two_point_correlations_3_vertical(a, fname):
	from matplotlib.gridspec import GridSpec

	n_points = len(a)
	fig = plt.figure()
	gs = GridSpec(3, 1)
	ax = []
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
	ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
	ax.extend([ax1, ax2, ax3])

	for i, b in enumerate(a):
		point_str = b[0]
		n_cases = len(b[1])
		print(point_str)
		for j, t in enumerate(b[1]):
			case_name, c, d = t[0], t[1], t[2]
			d[0]=0.011111
			if 'pi' in case_name:
				case_name = '\pi'
				max_d = np.max(d)
			every = [4,2,2,1,1]
			ax[i].plot(d, c, color=colors[j], lw=1.5, label='$'+case_name+'$', marker=markers[j], markevery=0.05, markersize=4)

	leg1 = ax1.legend(loc='upper right')
	leg1.get_frame().set_edgecolor('black')
	leg1.get_frame().set_facecolor('white')
	leg1.get_frame().set_linewidth(0.5)
	leg1.get_frame().set_alpha(0.5)

	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)

	ax1.set_ylabel(r'$\left\langle v_1, v_2 \right\rangle$')
	ax2.set_ylabel(r'$\left\langle v_1, v_2 \right\rangle$')
	ax3.set_ylabel(r'$\left\langle v_1, v_2 \right\rangle$')
	ax3.set_xlabel(r'$\log d/D$')

	for q in ax:
		q.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
		# print(max_d)
		# q.set_ylim(0.5, 1.1)
		q.set_xscale('log', nonposx='clip')
		q.set_xlim(1.1e-2,max_d)

	fig_size = fig.get_size_inches()
	fig.set_size_inches(fig_size[1]/2, fig_size[1])
	fig.tight_layout()
	plt.savefig(fname, transparent=True, bbox_inches='tight')
	return


def CL_CD_theta(fy, fx, t, alphas, times, fname):
	from scipy.signal import savgol_filter, resample
	from scipy.interpolate import interp1d
	from matplotlib.gridspec import GridSpec

	# fig, [ax1, ax2, ax3] = plt.subplots(nrows=2, ncols=2, sharex=True)
	fig = plt.figure()
	gs = GridSpec(2, 2)
	ax3 = fig.add_subplot(gs[:, 1])
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

	ax1.plot(t, fy, color='black', lw=1, label=r'$C_L$')
	ax1.plot(t, fx, color='grey', ls='dashed', lw=1, label=r'$C_D$')

	upper, lower = zip(*alphas)
	u, l = np.array(upper), np.array(lower)
	u = savgol_filter(u, 7, 3)  # window size 51, polynomial order 3
	l = savgol_filter(l, 7, 3)  # window size 51, polynomial order 3

	ax2.plot(times, u, color='blue', lw=1, label=r'$\theta_u$')
	ax2.plot(times, 360 + l - u, color='purple', ls='dotted', lw=1, label=r'$\theta_l-\theta_u$')
	ax2.plot(times, -l, color='red', ls='dashed', lw=1, label=r'$360-\theta_l$')

	fy_function = interp1d(t, fy, kind='cubic')
	fy = fy_function(times)
	ax3.scatter(fy, u + l, s=10, linewidths=1, color='black')
	ax3.axhline(0, color='grey', lw=0.1)
	ax3.axvline(0, color='grey', lw=0.1)

	ax1.grid(axis='both', alpha=0.5)
	ax2.grid(axis='both', alpha=0.5)

	ax1.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	ax2.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	ax3.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	ax1.set_xlim(min(t), max(t))
	ax1.set_ylim(-2, 2)
	ax1.yaxis.set_ticks([-1, 0, 1])
	plt.setp(ax1.get_xticklabels(), visible=False)
	ax2.set_ylim(90, 165)
	ax2.yaxis.set_ticks([100, 120, 140, 160])
	ax2.set_xlabel(r'$tU/D$')
	ax3.set_xlabel(r'$C_L$')
	# ax3.set_ylabel(r'$\theta_l+\theta_u$') #labelpad=-3 for 0.5
	# ax3.set_ylim(-24,24) #0.5
	# ax3.set_xlim(-1.8,1.8)
	# ax3.set_ylim(-9,9) #pi
	# ax3.set_xlim(-0.8,0.8)

	leg1 = ax1.legend(loc='lower left')
	leg1.get_frame().set_edgecolor('black')
	leg1.get_frame().set_facecolor('white')
	leg1.get_frame().set_linewidth(0.5)
	leg1.get_frame().set_alpha(0.85)
	leg2 = ax2.legend(loc=(0.0375, 0.45), numpoints=1, ncol=2, columnspacing=1, labelspacing=0.1, fontsize=9)

	leg2.get_frame().set_edgecolor('black')
	leg2.get_frame().set_facecolor('white')
	leg2.get_frame().set_linewidth(0.5)
	leg2.get_frame().set_alpha(0.85)

	# fig_size = fig.get_size_inches()
	# fig.set_size_inches(fig_size[1] * 2, fig_size[1] * 2 / 2.8)
	# fig.tight_layout()

	fig.tight_layout()
	plt.savefig(fname, transparent=True, bbox_inches='tight')
	return


def CL_CD_theta_2(fy, fx, t, alphas, times, fname):
	from scipy.signal import savgol_filter, resample
	from scipy.interpolate import interp1d
	from matplotlib.gridspec import GridSpec

	# fig, [ax1, ax2, ax3] = plt.subplots(nrows=2, ncols=2, sharex=True)
	fig = plt.figure()
	gs = GridSpec(3, 1)
	ax2 = fig.add_subplot(gs[1, 0])
	ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
	ax3 = fig.add_subplot(gs[2, 0])

	ax1.plot(t, fy, color='black', lw=1, label=r'$C_L$')
	ax1.plot(t, fx, color='grey', ls='dashed', lw=1, label=r'$C_D$')

	upper, lower = zip(*alphas)
	u, l = np.array(upper), np.array(lower)
	u = savgol_filter(u, 7, 3)  # window size 51, polynomial order 3
	l = savgol_filter(l, 7, 3)  # window size 51, polynomial order 3

	ax2.plot(times, u, color='blue', lw=1, label=r'$\theta_u$')
	ax2.plot(times, 360 + l - u, color='purple', ls='dotted', lw=1, label=r'$\theta_l-\theta_u$')
	ax2.plot(times, -l, color='red', ls='dashed', lw=1, label=r'$360-\theta_l$')

	fy_function = interp1d(t, fy, kind='cubic')
	fx_function = interp1d(t, fx, kind='cubic')
	fy = fy_function(times)
	fx = fx_function(times)

	ax3.scatter(fy, u + l, s=10, linewidths=1, color='black')

	# d = {}
	# d['t'] = times
	# d['C_L'] = fy
	# d['C_D'] = fx
	# d[r'\theta_u'] = u
	# d[r'360-\theta_l'] = -l
	# df = pd.DataFrame.from_dict(d)
	# df.to_csv('spreadsheets/figure7b.csv', index=False)
	# d = {}
	# d['t'] = times
	# d['C_L'] = fy
	# d['C_D'] = fx
	# d[r'\theta_u'] = u
	# d[r'360-\theta_l'] = -l
	# df = pd.DataFrame.from_dict(d)
	# df.to_csv('spreadsheets/figure7b.csv', index=False)

	ax3.axhline(0, color='darkgrey', lw=0.1)
	ax3.axvline(0, color='darkgrey', lw=0.1)

	ax1.grid(axis='both', color='darkgrey', lw=0.1)
	ax2.grid(axis='both', color='darkgrey', lw=0.1)

	ax1.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	ax2.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	ax3.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	ax2.set_xlim(min(t), max(t))
	ax1.set_ylim(-2, 2)
	ax1.yaxis.set_ticks([-1, 0, 1])
	plt.setp(ax1.get_xticklabels(), visible=False)
	ax2.set_ylim(90, 165)
	ax2.yaxis.set_ticks([100, 120, 140, 160])
	ax2.set_xlabel(r'$tU/D$')
	ax3.set_xlabel(r'$C_L$')
	ax3.set_ylabel(r'$\theta_l+\theta_u$') #labelpad=-3 for 0.5
	ax3.set_ylim(-24,24) #0.5
	ax3.set_xlim(-1.8,1.8) #0.5
	# ax3.set_ylim(-9,9) #pi
	# ax3.set_xlim(-0.8,0.8) #pi

	leg1 = ax1.legend(loc='lower left')
	leg1.get_frame().set_edgecolor('black')
	leg1.get_frame().set_facecolor('white')
	leg1.get_frame().set_linewidth(0.5)
	leg1.get_frame().set_alpha(0.85)
	leg2 = ax2.legend(loc=(0.0375, 0.45), numpoints=1, ncol=2, columnspacing=1, labelspacing=0.1, fontsize=9)

	leg2.get_frame().set_edgecolor('black')
	leg2.get_frame().set_facecolor('white')
	leg2.get_frame().set_linewidth(0.5)
	leg2.get_frame().set_alpha(0.85)

	fig_size = fig.get_size_inches()
	fig.set_size_inches(3.5, 8.5)

	fig.tight_layout()
	plt.savefig(fname, transparent=True, bbox_inches='tight')
	return


def plot_forces_list(T, CD, CL, names, file='test.pdf'):
	fig, ax = plt.subplots(1, 1)
	colors = ['k', 'r', 'b', 'purple', 'orange', 'g', 'grey', 'c']
	for i in range(len(T)):
		CL_label = '$C_L, \mathrm{' + names[i] + '}$'
		ax.plot(T[i], CL[i], color=colors[i], alpha=0.8, ls='-', lw=1, label=CL_label)
		CD_label = '$C_D, \mathrm{' + names[i] + '}$'
		ax.plot(T[i], CD[i], color=colors[i], alpha=0.8, ls='dashed', lw=1, label=CD_label)

	ax.set_xlim(min(T[0]), max(T[0]))
	ax.set_xlabel('$T$')
	ax.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
	leg = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
	leg.get_frame().set_edgecolor('grey')
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
	return


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
	# ax.set_ylim(1.5*min(fy), 1.5*max(fy))
	ax.set_ylim(-2.5, 2.5)

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
	d = {}
	for tke_tuple in tke_tuple_list:
		label = tke_tuple[0][:-1]
		tke = tke_tuple[1]
		if 'xD_min' in kwargs:
			x = x[x > kwargs['xD_min']]
			tke = tke[-x.size:]
		if 'pi' in label:
			label = '\pi'
		label = '$'+label+'$'
		color = colors[i]
		plt.plot(x, tke, color=color, lw=1.5, label=label, marker=markers[i], markevery=50, markersize=4)
		tke_list.append(tke)

		d['x'] = x
		d[label[1:-1]] = tke

		i += 1

	df = pd.DataFrame.from_dict(d)
	df.to_csv('spreadsheets/figure5b.csv', index=False)
	
	
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
	leg.get_frame().set_alpha(0.85)
	# ax.yaxis.set_ticks([-2, 0, 2])
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

	# Show plot and save figure
	# plt.show()
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
	return


def plotProfiles(file, profiles_tuple_list, **kwargs):
	ax = plt.gca()
	fig  = plt.gcf()

	if not profiles_tuple_list:
		raise ValueError("No profile series passed to the function.")
	else:
		M = profiles_tuple_list[0][1].shape[0]
	if not 'y' in kwargs:
		ymin, ymax = 0, M-1
	else:
		ymin, ymax = kwargs['y'][0], kwargs['y'][1]

	scaling = kwargs.get('scaling', 1)
	yshift = kwargs.get('yshift', 0)

	ylabel = '$' + kwargs.get('ylabel','y') + '$'
	y = np.linspace(ymin, ymax, M)/scaling + yshift

	# Show lines
	profiles_list = []
	for i, profile_tuple in enumerate(profiles_tuple_list):
		label = profile_tuple[0]
		if 'piD' in label: label = '\pi'
		else: label = label[:-1]
		label = '$'+label+'$'
		color = colors[i]
		profile = profile_tuple[1]
		plot = plt.plot(profile, y, color=color, lw=1.5, label=label, marker=markers[i], markevery=50, markersize=4)
		profiles_list.append(profile)

	# Set limits
	ax.set_ylim(min(y), max(y))
	fig, ax = makeSquare(fig,ax)

	# Edit frame, labels and legend
	plt.xlabel(r'$\left\langle u \right\rangle$')
	plt.ylabel(ylabel, rotation=0)

	leg = plt.legend(loc='lower left')

	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.5)
	leg.get_frame().set_alpha(0.85)
	# ax.yaxis.set_ticks([-2, 0, 2])
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

	# Show plot and save figure
	plt.show()
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
	return


def plotProfiles_multiple(file, tuple_profiles_tuple_list, **kwargs):
	from collections import OrderedDict

	fig  = plt.gcf()
	ax = plt.gca(projection='3d')

	if not tuple_profiles_tuple_list:
		raise ValueError("No profile series passed to the function.")
	else:
		M = tuple_profiles_tuple_list[0][1][0][1].shape[0]
	if not 'y' in kwargs:
		ymin, ymax = 0, M-1
	else:
		ymin, ymax = kwargs['y'][0], kwargs['y'][1]

	scaling = kwargs.get('scaling', 1)
	yshift = kwargs.get('yshift', 0)

	ylabel = '$' + kwargs.get('ylabel','y') + '$'
	y = np.linspace(ymin, ymax, M)/scaling + yshift

	# Show lines
	for e in tuple_profiles_tuple_list:
		x_loc = e[0]
		profiles_tuple_list = e[1]
		for i, profile_tuple in enumerate(profiles_tuple_list):
			label = profile_tuple[0]
			if 'piD' in label: label = '\pi'
			elif 'D9' in label: label = '2\mathrm{D}'
			else: label = label[:-1]
			label = '$'+label+'$'
			color = colors[i]
			profile = profile_tuple[1]
			plot = plt.plot(profile, y, x_loc, color=color, lw=1.5, label=label, markevery=50, markersize=4)

	# Set limits
	ax.set_xlim(0.35, 1.1)
	ax.set_ylim(-3, 3)
	ax.set_zlim(2, 10.5)
	# fig, ax = makeSquare(fig,ax)

	# Edit frame, labels and legend
	ax.set_xlabel(r'$\overline{u}$', rotation=0)
	ax.set_ylabel(ylabel, rotation=0)
	ax.set_zlabel('$x$', rotation=0)

	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	leg = plt.legend(by_label.values(), by_label.keys(), loc=(0.8,0.15))
	ax.view_init(azim=0, elev=140)

	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.75)
	leg.get_frame().set_alpha(0.75)
	ax.xaxis.set_ticks([0.4,0.6,0.8,1])
	ax.yaxis.set_ticks([-2,0,2])
	ax.zaxis.set_ticks([4,6,8,10])
	# ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
	# plt.switch_backend('PDF') #pdf
	# ax.autoscale(enable=False, axis='both')  # you will need this line to change the Z-axis
	# Show plot and save figure
	# plt.show()
	fig.tight_layout()
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
	return

# ------------------------------------------------------ x-y
def plotXY(y, **kwargs):
	"""
	Generate a x-y plot in space
	:param y: series to plot [1D numpy array]
	:param label: y axis label [string]
	:param file: output file name
	:param kwargs: 'x' coordinates [numpy 1D array], 'xD_min' left x limit, 'ylog' log plot [boolean]
	:return: -
	"""
	fig, ax = plt.subplots(1, 1)

	N = y.shape[0]
	x = kwargs.get('x', np.arange(N))
	file = kwargs.get('file', 'test.pdf')
	x_label = kwargs.get('x_label', None)
	y_label = kwargs.get('y_label', None)
	ylog = kwargs.get('ylog', False)
	xlog = kwargs.get('xlog', False)

	# Show lines
	plt.plot(x, y, color='black', lw=1, label=y_label)

	# Edit figure, axis, limits
	ax.set_xlim(min(x), max(x))
	if xlog:
		ax.set_xscale('log')
	if ylog:
		ax.set_yscale('log')
	fig, ax = makeSquare(fig,ax)
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	return


def plotXY_list(Y, file='test.pdf', **kwargs):
	"""
	Generate a x-y plot in space
	:param y: series to plot [1D numpy array]
	:param label: y axis label [string]
	:param file: output file name
	:param kwargs: 'x' coordinates [numpy 1D array], 'xD_min' left x limit, 'ylog' log plot [boolean]
	:return: -
	"""
	fig, ax = plt.subplots(1, 1)

	N = Y[0].shape[0]
	X = kwargs.get('X', [np.arange(N) for i in range(len(Y))])
	xlabel = kwargs.get('xlabel', 'x')
	ylabel = kwargs.get('ylabel', 'y')
	ylog = kwargs.get('ylog', False)
	xlog = kwargs.get('xlog', False)
	names = kwargs.get('names', [None for i in range(len(Y))])
	names = [r'$\mathrm{'+name+'}$' for name in names]

	# Show lines
	for i,(x,y) in enumerate(zip(X,Y)):
		plt.plot(x, y, color=colors[i], lw=1, label=names[i])

	# Edit figure, axis, limits
	ax.set_xlim(np.min(X), np.max(X))
	if xlog:
		ax.set_xscale('log')
	if ylog:
		ax.set_yscale('log')

	plt.xlabel(r'$'+xlabel+'$')
	plt.ylabel(r'$'+ylabel+'$')

	leg = plt.legend(loc='upper left', prop={'size': 10})
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.5)

	# Show plot and save figure
	fig, ax = makeSquare(fig,ax)
	plt.savefig(file, transparent=True, bbox_inches='tight')
	return

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
		print(i)
		ax.scatter(x[i], y[i], c=colors[i], marker=markers[i], s=30, linewidths=1, label=case)


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
	leg.get_frame().set_alpha(0.85)

	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	return


def plotScatter2(x1, x2, y, cases, file):
	"""
	Generate a x-y plot in space
	:param x: series to plot [1D numpy array]
	:param y: series to plot [1D numpy array]
	:param label: y axis label [string]
	:param file: output file name
	:return: -
	"""
	ax1 = plt.gca()
	fig  = plt.gcf()
	ax2 = ax1.twiny()

	# Show lines
	for i, case in enumerate(cases):
		ax1.scatter(x1[i], y[i], c='red', marker=markers[i], s=10, linewidths=1, label=case)
		ax2.scatter(x2[i], y[i], c='blue', marker=markers[i], s=10, linewidths=1, label=case)

	# Edit figure, axis, limits
	ax1.set_xlim(1,9)
	ax2.set_xlim(1,9)
	ax1.set_ylim(0.2, 1.3)
	# ax.set_ylim(0.1, 1.4)

	ax1.tick_params(bottom="on", top="off", right="on", which='both', direction='in', length=2)
	ax2.tick_params(bottom="off", top="on", right="on", which='both', direction='in', length=2)

	fig, ax1 = makeSquare(fig,ax1)
	# Edit frame, labels and legend
	ax1.set_xlabel('$\sigma_l$', color='red')
	ax2.set_xlabel('$\sigma_u$', color='blue')

	ax1.set_ylabel('$\overline{C}_L$')
	leg = ax1.legend(loc='lower right')
	for q in leg.legendHandles:
		q.set_color('grey')
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.5)
	leg.get_frame().set_alpha(0.85)

	# Show plot and save figure
	# plt.show()
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
	# xmax=11.83
	# print(xmin, xmax, N)
	x = np.linspace(xmin, xmax, N)

	# Show lines
	y_list = []
	d = {}
	for y_tuple in enumerate(y_tuple_list):
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

		# d['x'] = x
		# d[label[1:-1]] = y

	# df = pd.DataFrame.from_dict(d)
	# df.to_csv('spreadsheets/figure9b.csv', index=False)

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
	leg.get_frame().set_alpha(0.85)
	# ax.set_xlim(min(x), 12)
	# ax.set_ylim(ylims[0], ylims[1])
	ax.xaxis.set_ticks([2, 4, 6, 8, 10, 12])

	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

	# Save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
	return


def velocity_profiles(file, profiles_tuple_list, **kwargs):
	"""
	Similar to plotXYSpatial_list just for a specific test case
	"""
	ax = plt.gca()
	fig  = plt.gcf()

	ylabel = '$' + kwargs.get('ylabel','r') + '$'

	# Show lines
	profiles = []
	for i, profile_tuple in enumerate(profiles_tuple_list):
		label, profile, y = profile_tuple[0], profile_tuple[1], profile_tuple[2]
		p = np.where((np.array(y) > 0.45) & ((np.array(y) <0.85)))
		profile=np.array(profile)[p]
		y=np.array(y)[p]
		label = '$'+label+'$'
		if 'pi' in label:
			label = '$\pi$'
		color = colors[i]
		# plt.plot(profile, y, color=color, lw=1, label=label, marker=markers[i], markevery=10, markersize=4)
		plt.plot(profile, y, color=color, lw=1, label=label)
		profiles.append(profile)

	# Edit figure, axis, limits
	# ax.set_xlim(min(x), max(x))

	fig, ax = makeSquare(fig,ax)

	# Edit frame, labels and legend
	plt.xlabel(r'$\omega_z|_z$')
	plt.ylabel(ylabel, rotation=0)
	ax.set_ylim(np.min([np.min(s) for s in y]), np.max([np.max(s) for s in y]))

	leg = plt.legend(loc='upper right')
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.75)
	leg.get_frame().set_alpha(0.75)
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

	# Show plot and save figure
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
def plotLogLogSpatialSpectra(file, k, ak):
	"""
	Generate a loglog plot of a 1D spatial signal
	:param file: output file name
	:param k: frequency [1D numpy array]
	:param uk: signal in wavenumber space [1D numpy array]
	:return: -
	"""
	fig, ax = plt.subplots(1, 1)

	# Show lines
	plt.loglog(k, ak, color='black', lw=1.5, label='$L_z = piD$')
	x, y = loglogLine(p2=(max(k), 5e-4), p1x=min(k)*10, m=-5/3)
	plt.loglog(x, y, color='black', lw=1, ls='dotted')
	x, y = loglogLine(p2=(max(k), 4e-4), p1x=min(k)*10, m=-3)
	plt.loglog(x, y, color='black', lw=1, ls='dashdot')

	# Set limits
	# ax.set_xlim(min(k)*10, max(k))
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


def plotLogLogSpatialSpectra_list(k_list, ak_list, file='test.pdf', **kwargs):
	"""
	Generate a loglog plot of a list of 1D spatial signals
	:param file: output file name
	:param tke_tuple_list: list of tuples as (case, uk), where 'case' is the case name [string] and 'uk' is
		the transformed u: uk = FFT(u). [1D numpy array]
	:param wn_list: list of frequencies for the different cases
	:return: -
	"""
	fig, ax = plt.subplots(1, 1)
	names = kwargs.get('names', ['case '+str(i) for i in range(len(k_list))])
	xlabel = kwargs.get('xlabel', 'kD/(2\pi)')
	ylabel = kwargs.get('ylabel', None)

	# Show lines
	for i, (k, ak) in enumerate(zip(k_list, ak_list)):
		plt.loglog(k, ak, color=colors[i], lw=1, label='$\mathrm{'+names[i]+'}$')

	for i in np.arange(1):
		x, y = loglogLine(p2=(np.max(k_list[0]), 1e-7*40**i), p1x=np.min(k_list[0]), m=-5/3)
		plt.loglog(x, y, color='darkgrey', lw=0.5, ls='dotted')
		x, y = loglogLine(p2=(1.2e2, 1e-10*40**i), p1x=1e-3, m=-3)
		plt.loglog(x, y, color='darkgrey', lw=0.5, ls='dashdot')
		x, y = loglogLine(p2=(np.max(k_list[0]), 1e-11*40**i), p1x=np.min(k_list[0]), m=-3.66)
		plt.loglog(x, y, color='darkgrey', lw=0.5, ls='dashed')

	# Set limits
	ax.set_xlim(np.min(k_list[1]), np.max(k_list[1]))
	ax.set_ylim(1e-13, 10)

	fig, ax = makeSquare(fig,ax)
	# ax.xaxis.set_tick_params(labeltop='on')
	ax.tick_params(bottom="on", top="on", which='both')

	# Edit frame, labels and legend
	ax.tick_params(bottom="on", top="on", which='both', direction='in')
	plt.xlabel(r'$'+xlabel+'$')
	plt.ylabel(r'$'+ylabel+'$')
	leg = plt.legend(loc='lower left')
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.5)
	leg.get_frame().set_alpha(0.85)
	# ax.yaxis.set_ticks([-2, 0, 2])
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
	# ax.get_yaxis().set_ticks([], minor=True)

	# ax.set_xticks([1.1, 1.2, 1.3, 1.4, 1.5, 1.6 ,1.7, 1.8, 1.9 ,2], minor=True)
	# ax.set_yticks([0.3, 0.55, 0.7], minor=True)
	# ax.xaxis.grid(True, which='major')

	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()


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
		if '2D' in label or 'D9' in label: label = '2\mathrm{D}'
		uk = uk_tuple[1]
		label = '$'+label+'$'
		color = colors[i]
		plt.loglog(freqs_list[i], uk, color=color, lw=0.5, label=label)

	for i in np.arange(5):
		x, y = loglogLine(p2=(1.e2, 1e-11*10**i), p1x=1e-2, m=-5/3)
		# plt.loglog(x, y, color='black', lw=0.5, ls='dotted', alpha=0.3)
		plt.loglog(x, y, color='darkgrey', lw=0.3, ls='dotted')
	for i in np.arange(2):
		# x, y = loglogLine(p2=(1.2e2, 1e-16 * 100 ** i), p1x=1e-3, m=-3)
		# plt.loglog(x, y, color='black', lw=0.5, ls='dashdot', alpha=0.3)
		# plt.loglog(x, y, color='darkgrey', lw=0.2, ls='dashdot')
		x, y = loglogLine(p2=(1.2e2, 1e-17*100**i), p1x=1e-2, m=-3.66)
		# plt.loglog(x, y, color='black', lw=0.5, ls='dashed', alpha=0.3)
		plt.loglog(x, y, color='darkgrey', lw=0.3, ls='dashed')


	# Set limits
	ax.set_xlim(1e-2, 7e1) # Window
	ax.set_ylim(9e-16, 1e-1)
	# ax.set_ylim(1e-11, 1e4)


	fig, ax = makeSquare(fig,ax)

	# Edit frame, labels and legend
	ax.tick_params(bottom="on", top="on", which='both', direction='in')
	plt.xlabel(r'$fD/U$')
	plt.ylabel(r'$\mathrm{PS}\left(v\right)$')
	leg = plt.legend(loc='lower left')
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.5)
	leg.get_frame().set_alpha(0.85)
	# ax.yaxis.set_ticks([-2, 0, 2])
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
	# ax.get_yaxis().set_ticks([], minor=True)

	# ax.set_xticks([1.1, 1.2, 1.3, 1.4, 1.5, 1.6 ,1.7, 1.8, 1.9 ,2], minor=True)
	# ax.set_yticks([0.3, 0.55, 0.7], minor=True)
	# ax.xaxis.grid(True, which='major')

	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()
	return


def plotLogLogSpatialSpectra_list_cascade(k_list, ak_list, file='test.pdf', **kwargs):
	"""
	Generate a loglog plot of a list of 1D spatial signals
	:param file: output file name
	:param tke_tuple_list: list of tuples as (case, uk), where 'case' is the case name [string] and 'uk' is
		the transformed u: uk = FFT(u). [1D numpy array]
	:param wn_list: list of frequencies for the different cases
	:return: -
	"""
	from matplotlib.lines import Line2D

	fig, ax = plt.subplots(1, 1)
	names = kwargs.get('names', ['case '+str(i) for i in range(len(k_list))])
	xlabel = kwargs.get('xlabel', 'kD/(2\pi)')
	ylabel = kwargs.get('ylabel', None)
	groups_size = kwargs.get('groups_size', 1)

	# Show lines
	for i, (k, ak) in enumerate(zip(k_list, ak_list)):
		f = 50 ** (-int(i/groups_size))
		plt.loglog(k, ak*f, color=colors[int(i/groups_size)], lw=1,
		           ls=styles[np.mod(i,groups_size)], label='$\mathrm{'+names[i]+'}$')

	for i in np.arange(3):
		x, y = loglogLine(p2=(np.max(k_list[0]), 1e-11*80**i), p1x=np.min(k_list[0]), m=-5/3)
		plt.loglog(x, y, color='darkgrey', lw=0.5, ls='dotted')
		x, y = loglogLine(p2=(1.2e2, 1e-15*80**i), p1x=1e-3, m=-3)
		plt.loglog(x, y, color='darkgrey', lw=0.5, ls='dashdot')
		x, y = loglogLine(p2=(np.max(k_list[0]), 1e-15*80**i), p1x=np.min(k_list[0]), m=-3.66)
		plt.loglog(x, y, color='darkgrey', lw=0.5, ls='dashed')

	# Set limits
	ax.set_xlim(np.min(k_list[0]), np.max(k_list[0]))
	ax.set_ylim(1e-16, 0.1)

	fig, ax = makeSquare(fig,ax)
	# ax.xaxis.set_tick_params(labeltop='on')
	ax.tick_params(bottom="on", top="on", which='both')

	# Edit frame, labels and legend
	ax.tick_params(bottom="on", top="on", which='both', direction='in')
	plt.xlabel(r'$'+xlabel+'$')
	plt.ylabel(r'$'+ylabel+'$')

	lines = [Line2D([0], [0], color=c, linewidth=1) for c in colors[:int(len(ak_list)/groups_size)]]
	lines.append(Line2D([0], [0], color='black', linewidth=1))
	lines.append(Line2D([0], [0], color='black', linewidth=1, ls='dashed'))
	labels = ['$T/4$', '$T/2$', '$T$', '$2T$',
	          '$\mathrm{'+names[0]+'}$',
	          '$\mathrm{'+names[1]+'}$',]
	leg = plt.legend(lines, labels, loc='lower left', prop={'size': 10})

	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_facecolor('white')
	leg.get_frame().set_linewidth(0.5)
	ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)

	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	plt.clf()


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
	leg.get_frame().set_alpha(0.85)

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
	# d0 = {}
	for i, eta in enumerate(eta_tuple_list):
		label = eta[0]
		if 'piD' in label: label = '\pi'
		else: label = label[:-1]
		eta = eta[1]
		xi = xi_tuple_list[i][1]
		label = r'$'+label+'$'
		plt.scatter(xi, eta, marker=markers[i], c=colors[i], s=10, linewidths=0.1, label=label, edgecolor = 'black')
		# d0[label[1:-1]] = (xi, eta)

	# d = {}
	# for k,v in d0.items():
	#	 d[r'\xi'] = v[0]
	#	 d[r'\eta'] = v[1]
	#	 df = pd.DataFrame.from_dict(d)
	#	 df.to_csv('spreadsheets/figure8d_'+k+'.csv', index=False)

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
	leg.get_frame().set_alpha(0.85)
	ax.xaxis.set_ticks([-0.2,-0.1,0,0.1,0.2,0.3,0.4])
	ax.tick_params( direction='in', length=2)
	ax.tick_params(bottom="on", top="on", right='on',which='both', direction='in')
	# plt.minorticks_off()

	# Show plot and save figure
	plt.savefig(file, transparent=True, bbox_inches='tight')
	return


# ------------------------------------------------------ GC plots
# def error_order(file, x, y):
#	 """
#	 Generate a loglog plot of a time spectra series using the matplotlib library given the arguments
#	 """
#	 # Basic definitions
#	 ax = plt.gca()
#	 fig  = plt.gcf()
#
#	 plt.loglog(x, y, color='b', lw=0.5)
#
#	 x, y = loglogLine(p2=(np.max(x), np.max(y)), p1x=np.min(x), m=2)
#	 plt.loglog(x, y, color='black', lw=1, ls='dotted')
#	 x, y = loglogLine(p2=(np.max(x), np.max(y)), p1x=np.min(x), m=1)
#	 plt.loglog(x, y, color='black', lw=1, ls='dotted')
#	 # x, y = loglogLine(p2=(1.2e2, 1e-9), p1x=1e-2, m=-3)
#	 # plt.loglog(x, y, color='black', lw=1, ls='dashdot')
#	 # x, y = loglogLine(p2=(1e0, 1e-8), p1x=1e-3, m=-11/3)
#	 # plt.loglog(x, y, color='black', lw=1, ls='dashed')
#
#	 # Set limits
#	 # ax.set_xlim(np.min(freqs_list[0]), 2e-1)
#	 # ax.set_ylim(1e-8, 1e-1)
#	 # ax.set_xlim(1e-2, 1e2) # Window
#	 # ax.set_ylim(1e-11, 1e-1)
#
#	 # fig, ax = makeSquare(fig,ax)
#	 # ax.xaxis.set_tick_params(labeltop='on')
#	 # ax.tick_params(bottom="on", top="on", which='both')
#
#	 # Edit frame, labels and legend
#
#	 # Show plot and save figure
#	 plt.show()
#	 plt.savefig(file, transparent=True, bbox_inches='tight')
#	 return


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


def max_min_loc(a, x, y):
	a_max = np.amax(a)
	a_min = np.amin(a)
	i, j = np.unravel_index(a.argmax(), a.shape)
	x_max_loc, y_max_loc = x[i,j], y[i,j]
	i, j = np.unravel_index(a.argmin(), a.shape)
	x_min_loc, y_min_loc = x[i,j], y[i,j]
	my_str = 'max val: {:.2e}, max loc: ({:.3f},{:.3f})\n' \
			 'min val: {:.2e}, min loc: ({:.3f},{:.3f})'\
		.format(a_max, x_max_loc, y_max_loc, a_min, x_min_loc, y_min_loc)
	return my_str


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
	def gcd(a, b):
		while b:
			a, b = b, a%b
		return a
	def _multiple_formatter(x, pos):
		den = denominator
		num = np.int(np.rint(den*x/number))
		com = gcd(num,den)
		(num,den) = (int(num/com),int(den/com))
		if den==1:
			if num==0:
				return r'$0$'
			if num==1:
				return r'$%s$'%latex
			elif num==-1:
				return r'$-%s$'%latex
			else:
				return r'$%s%s$'%(num,latex)
		else:
			if num==1:
				return r'$\frac{%s}{%s}$'%(latex,den)
			elif num==-1:
				return r'$\frac{-%s}{%s}$'%(latex,den)
			else:
				return r'$\frac{%s%s}{%s}$'%(num,latex,den)
	return _multiple_formatter

class Multiple:
	def __init__(self, denominator=2, number=np.pi, latex='\pi'):
		self.denominator = denominator
		self.number = number
		self.latex = latex

	def locator(self):
		return plt.MultipleLocator(self.number / self.denominator)

	def formatter(self):
		return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))