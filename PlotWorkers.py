from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from FormattingStrings import *
from plotformatting import *
import numpy as np
import AnalysisFunctions as af
import traceback
import math
from colorcet import cm


class PlotFitWorker(QRunnable):
    def __init__(self, current_folder, fig, xlabel, units, fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=[]):
        """

        """
        super(PlotFitWorker, self).__init__()
        self.current_folder = current_folder
        self.fig = fig
        self.xlabel = xlabel
        self.units = units
        self.fit_mean = fit_mean
        self.fit_std = fit_std
        self.roi_labels = roi_labels
        self.keys_adjusted = keys_adjusted
        self.rois_to_exclude = rois_to_exclude


class Plot2DWorker(PlotFitWorker):
    def run(self):
        try:
            self.fig.clf()
            rois_to_plot = [i for i in range(
                len(self.roi_labels)) if self.roi_labels[i] not in self.rois_to_exclude]
            n_rows, n_columns = 2, math.ceil(len(rois_to_plot) / 2)
            if len(self.keys_adjusted) < 2:
                return
            extent = [-0.5, self.fit_mean.shape[2] + 0.5,
                      np.max(self.keys_adjusted) +
                      np.diff(self.keys_adjusted)[0],
                      np.min(self.keys_adjusted)]
            for e, i in enumerate(rois_to_plot):
                label = self.roi_labels[i]
                ax = self.fig.add_subplot(n_rows, n_columns, e + 1)
                cax = ax.imshow(self.fit_mean[i], aspect="auto",
                                cmap=cm.blues, extent=extent, vmin=0)
                af.save_array(self.fit_mean[i], label, self.current_folder)
                self.fig.colorbar(cax, ax=ax, label="Fitted counts")
                if label in fancy_titles.keys():
                    ax.set_title(fancy_titles[label])
                else:
                    ax.set_title(label)
                ax.set_xlabel("Trap Index")
                ax.set_ylabel(f"{self.xlabel} ({self.units})")
            af.save_figure(self.fig, "2d_plot", self.current_folder)
        except:
            traceback.print_exc()


class Plot1DWorker(PlotFitWorker):
    """
    Worker Thread to make 1D plot
    """

    def run(self):
        try:
            self.fig.clf()
            axis = self.fig.add_subplot(111)
            for state, state_std, label in zip(self.fit_mean, self.fit_std, self.roi_labels):
                if label not in self.rois_to_exclude:
                    transparent_edge_plot(axis,
                                          self.keys_adjusted,
                                          np.mean(state, axis=1),
                                          np.mean(state_std, axis=1),
                                          label=fancy_titles[label])
                    af.save_array(np.mean(state, axis=1),
                                  f"{label}_1d", self.current_folder)
            axis.legend()
            axis.set_ylabel("Average trap counts")
            axis.set_xlabel(f"{self.xlabel} ({self.units})")
            af.save_figure(self.fig, "1d_plot", self.current_folder)
        except:
            traceback.print_exc()


class Plot1DHistogramWorker(PlotFitWorker):
    def run(self):
        try:
            self.fig.clf()
            axis = self.fig.add_subplot(111)
            for state, state_std, label in zip(self.fit_mean, self.fit_std, self.roi_labels):
                if label not in self.rois_to_exclude:
                    axis.hist(np.mean(state, axis=1), bins='stone',
                              label=fancy_titles[label])
            axis.legend()
            axis.set_ylabel("Number of Shots")
            axis.set_xlabel(f"Imaging Counts")
            af.save_figure(self.fig, "1d_histplot", self.current_folder)
        except:
            traceback.print_exc()
