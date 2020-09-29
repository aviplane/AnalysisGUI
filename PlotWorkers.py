from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from FormattingStrings import *
from plotformatting import *
import numpy as np
import AnalysisFunctions as af
import traceback


class Plot1DWorker(QRunnable):
    """
    Worker Thread to make 1D plot
    """

    def __init__(self, current_folder, fig, xlabel, units, fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=[]):
        """

        """
        super(Plot1DWorker, self).__init__()
        self.current_folder = current_folder
        self.fig = fig
        self.xlabel = xlabel
        self.units = units
        self.fit_mean = fit_mean
        self.fit_std = fit_std
        self.roi_labels = roi_labels
        self.keys_adjusted = keys_adjusted
        self.rois_to_exclude = rois_to_exclude

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
