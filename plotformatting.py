# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:40:42 2019

@author: QuantumEngineer3
"""

from matplotlib import rcParams
import matplotlib as mpl
import colorcet
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['axes.grid'] = True
# Some notes
# alpha > 0 preferred to dashed lines
magnetization_colormap = colorcet.cm.coolwarm
phase_colormap = "hsv"
contrast_colormap = colorcet.cm.fire
correlation_colormap = colorcet.cm.coolwarm


def transparent_edge_plot(ax, x, y, yerr=None, marker='o', ms=12, **kwargs):
    if yerr is not None:
        base, _, _ = ax.errorbar(x, y, yerr, ms=ms, marker=marker,
                                 linestyle="None", alpha=0.6, markeredgewidth=2, **kwargs)
    else:
        base, = ax.plot(x, y, ms=ms, marker=marker, linestyle="None",
                        alpha=0.5, markeredgewidth=2, **kwargs)
    ax.plot(x, y, ms=ms, marker=marker, linestyle="None",
            markeredgecolor=base.get_color(), markerfacecolor="None", markeredgewidth=2)
    return ax
