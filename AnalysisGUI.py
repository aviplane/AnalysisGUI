# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:07:40 2020

@author: Quantum Engineer
"""


"""
Start Daemon
    Time stamp -> folder
    File reached size?
        Move to folder
        Generate fit
        Append to array
        Regenerate Plots
"""
import sys
sys.path.append("Z://")
from runmanager.remote import Client

import os
import json
import traceback
import math
import collections
from units import unitsDef
from FormattingStrings import *
from plotformatting import *
from FileSorter import FileSorter
from AnalysisUI import AnalysisUI
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from colorcet import cm
import AnalysisFunctions as af
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import numpy as np
from PlotWorkers import Plot1DWorker, Plot2DWorker, Plot1DHistogramWorker, PlotPCAWorker

#from AveragedPlots import *


class AnalysisGUI(QMainWindow, AnalysisUI):
    def __init__(self, app):
        AnalysisUI.__init__(self)
        QMainWindow.__init__(self)
        app.aboutToQuit.connect(self.stop_sorting)
        self.create_ui(self)
        self.cb_script.currentIndexChanged.connect(self.set_script_folder)
        self.cb_data.activated.connect(self.set_data_folder)
        self.picker_date.dateChanged.connect(self.set_date)
        self.parameters_lineedit.returnPressed.connect(self.set_parameters)
        self.go_button.clicked.connect(
            self.make_sorter_thread)  # self.sort_all_files)
        self.checkbox_imaging_calibration.stateChanged.connect(
            self.set_imaging_calibration)
        self.checkbox_adjust_amplitudes.stateChanged.connect(
            self.set_amplitude_feedback)
        self.corr_threshold_min.sliderReleased.connect(self.set_corr_threshold)
        self.corr_threshold_max.sliderReleased.connect(self.set_corr_threshold)
        self.probe_threshold.sliderReleased.connect(self.set_probe_threshold)
        self.date = QtCore.QDate.currentDate().toString(date_format_string)
        self.script_folder = ""
        self.shot_threshold_size = 2e6
        self.data_folder_dict = {}
        self.set_script_cb()
        self.set_imaging_calibration()
        self.probe_threshold_value = 0
        self.amplitude_feedback = False
        self.rm_client = Client(host='171.64.58.213')
        self.threadpool = QThreadPool()
        self.parameters = ""

    def set_date(self, date):
        self.date = date.toString(date_format_string)
        self.set_script_cb()

    def set_imaging_calibration(self):
        self.imaging_calibration = self.checkbox_imaging_calibration.isChecked()
        try:
            self.worker.imaging_calibration = self.imaging_calibration
        except Exception as e:
            print(e, "trying to turn on imaging calibration")

    def set_parameters(self):
        parameter_text = self.parameters_lineedit.text()
        parameter_list = [i.strip() for i in parameter_text.split(",")]
        self.parameters = parameter_list
        try:
            self.worker.parameters = self.parameters
        except AttributeError as e:
            traceback.print_exc()
            print(e, "No file sorter worker yet...")

    def set_amplitude_feedback(self):
        self.amplitude_feedback = self.checkbox_adjust_amplitudes.isChecked()
        return

    def set_script_cb(self):
        try:
            folders = af.get_immediate_child_directories(
                af.get_date_data_path(self.date))
            folders = [af.get_folder_base(i) for i in folders]
            self.cb_script.clear()
            self.cb_script.addItems(folders)
        except FileNotFoundError:
            print("Selected bad date!")

    def set_script_folder(self, i):
        self.script_folder = self.cb_script.currentText()
        self.label_folder_name.setText(
            f"{analysis_folder_string}: {self.script_folder}")
        self.holding_folder = af.get_holding_folder(
            self.script_folder, data_date=self.date)
        self.set_data_cb()
        try:
            with open(self.holding_folder + "/folder_dict.json", 'r') as dict_file:
                self.data_folder_dict = json.loads(dict_file.read())
        except FileNotFoundError:
            print("No dictionary file yet...")

    def set_data_cb(self):
        """
        Get list of folders
        """
        try:
            folders = af.get_immediate_child_directories(self.holding_folder)
            folders = [af.get_folder_base(i) for i in folders]
            self.cb_data.clear()
            self.cb_data.addItems(folders)
        except FileNotFoundError:
            print("Selected bad date, or bad folder?")
        return

    def set_corr_threshold(self):
        min_value = self.corr_threshold_min.value() / 100
        self.corr_min_value.setText(f"{min_value:.2f}")

        max_value = self.corr_threshold_max.value() / 100
        self.corr_max_value.setText(f"{max_value:.2f}")

        try:
            # if "pc" in self.folder_to_plot and "time" not in self.folder_to_plot:
            self.make_correlation_plot()
        except AttributeError:
            print("Have not selected a folder yet.")

    def set_probe_threshold(self):
        try:
            current_folder = current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
            physics_probes = np.load(
                current_folder + "/fzx_probe.npy", allow_pickle=True)
            physics_means = np.array(
                [self.__mean_probe_value__(i) for i in physics_probes]
            )
            self.probe_threshold_value = self.probe_threshold.value() / 100 * \
                np.max(physics_means)
            self.probe_threshold_value_label.setText(
                f"{self.probe_threshold_value:.4f}")
            self.make_plots(self.folder_to_plot)
        except AttributeError:
            print("Can't set probe threshold, because no folder to plot set yet...")
        except Exception as e:
            print(e)
            # TODO: Error Handling

    def set_data_folder(self):
        folder_to_plot = self.cb_data.currentText()
        print(folder_to_plot)
        try:
            if not self.worker.running and folder_to_plot != "":
                self.make_plots(folder_to_plot)
        except AttributeError:
            self.make_plots(folder_to_plot)

    def make_sorter_thread(self):
        if self.go_button.text() == 'Go':
            print("Start")
            # Any other args, kwargs are passed to the run function
            self.worker = FileSorter(
                self.script_folder, self.date, self.imaging_calibration)
            self.worker.parameters = self.parameters
            self.worker.signal_output.folder_output_signal.connect(
                self.make_plots)
            self.worker.start()
            self.go_button.setText("Stop")
        elif self.go_button.text() == 'Stop':
            self.worker.stop()
            self.go_button.setText("Go")
            print("Stopping sorting thread")

    def stop_sorting(self):
        try:
            self.worker.stop()
            print("Stopping sorting thread...")
        except Exception as e:
            print(e)
            print("TODO: Add nicer error handling")

    def save_figure(self, fig, title, current_folder, extra_directory="", extra_title=""):
        """
        Save an figure at current_folder/extradirectory/title.png

        Parameters
        ----------
        fig : matplotlib figure
            The figure to save.  Hopefully all the axes are set up correctly.
        title : String
            file name.
        current_folder : String
            folder to save in
        extra_directory : String, optional

        Returns
        -------
        None.

        """
        save_folder = f"{current_folder}\\{extra_directory}"
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        title_string = f"{self.folder_to_plot}"
        if extra_title:
            title_string += f" | {extra_title}"
        fig.suptitle(title_string)
        fig.savefig(f"{save_folder}{self.folder_to_plot}_{title}.png", dpi=200)

    def save_array(self, data, title, current_folder, extra_directory=""):
        """
        Save an array at current_folder/extradirectory/title.txt

        Parameters
        ----------
        data : array
            Numpy to save.
        title : String
            file name.
        current_folder : String
            folder to save in
        extra_directory : String, optional

        Returns
        -------
        None.

        """
        save_folder = f"{current_folder}"
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        try:
            np.savetxt(f"{save_folder}{self.folder_to_plot}_{title}.txt", data)
        except OSError:
            print("Problem saving")
            # TODO: Error handling

    def make_plots(self, folder_to_plot):
        self.check_roi_boxes()
        self.folder_to_plot = folder_to_plot
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)
        fits = np.load(current_folder + "/all_fits.npy")
        xlabels = np.load(current_folder + "/xlabels.npy")
        physics_probes = np.load(
            current_folder + "/fzx_probe.npy", allow_pickle=True)
        fits, xlabels = self.select_probe_threshold(
            fits, xlabels, physics_probes)
        roi_labels = np.load(current_folder + "/roi_labels.npy")
        fit_mean, fit_std, xlabels = self.group_shot(fits, xlabels)
        fit_mean, fit_std = np.swapaxes(
            fit_mean, 0, 1), np.swapaxes(fit_std, 0, 1)
        keys_adjusted = np.array(xlabels) * sf
        plot1dworker = Plot1DWorker(current_folder, self.figure_1d, xlabel, units,
                                    fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude)
        plot2dworker = Plot2DWorker(current_folder, self.figure_2d, xlabel, units,
                                    fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude)
        self.threadpool.start(plot1dworker)
        self.threadpool.start(plot2dworker)
#        self.make_1d_plot()
#        self.make_2d_plot()
        if self.amplitude_feedback and 'reps' in folder_to_plot:
            self.adjust_amplitude_compensation()
        self.make_probe_plot()
        if "PairCreation" in self.folder_to_plot and 'time' not in self.folder_to_plot:
            self.canvas_corr.setFixedHeight(600)
            self.make_correlation_plot()
            plotPCAworker = PlotPCAWorker(
                current_folder, self.figure_6, xlabel, units, fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude)
            self.threadpool.start(plotPCAworker)
        elif "IntDuration" in self.folder_to_plot or "OG_Duration" in self.folder_to_plot:
            self.canvas_corr.setFixedHeight(600)
            self.make_phase_plot()
            self.make_magnetization_plot()
        elif xlabel == no_xlabel_string:
            self.canvas_corr.setFixedHeight(600)
            plot1dhistworker = Plot1DHistogramWorker(
                current_folder, self.figure_corr, xlabel, units,
                fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude)
            self.threadpool.start(plot1dhistworker)
        self.set_data_cb()
        index = self.cb_data.findText(self.folder_to_plot)
        self.cb_data.setCurrentIndex(index)

    def select_probe_threshold(self, fits, xlabels, physics_probes):
        if self.checkbox_probe_threhold.isChecked():
            mean_probe = np.array([self.__mean_probe_value__(i)
                                   for i in physics_probes])
            indices_to_keep = mean_probe > self.probe_threshold_value
            return fits[indices_to_keep], xlabels[indices_to_keep]
        return fits, xlabels

    def adjust_amplitude_compensation(self):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        try:
            current_compensation = np.load(compensation_path)
        except Exception as e:
            print(e)
            current_compensation = np.ones(18)
        fits = np.load(current_folder + "/all_fits.npy")
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[:, roi_labels.index('roi10')]
        fit_1m1 = fits[:, roi_labels.index('roi1-1')]
        fit_1p1 = fits[:, roi_labels.index('roi11')]
        fit_sum = fit_10 + fit_1m1 + fit_1p1
        trap_values = np.mean(fit_sum, axis=0)
        compensation = trap_values[::-1] ** (-1 / 2)
        if len(fit_sum) > 5 and len(fit_sum) % 6 == 0:
            trap_values = np.mean(fit_sum, axis=0)
            compensation = trap_values[::-1] ** (-1 / 2)
            new_compensation = current_compensation * compensation
            new_compensation = new_compensation / \
                np.linalg.norm(new_compensation)
            np.save(compensation_path, new_compensation)
            print("Engaging new scan")
#            self.rm_client.engage()
        return

    def make_probe_plot(self):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)
        xlabels = np.load(current_folder + "/xlabels.npy")
        keys_adjusted = sf * np.array(xlabels)
        bare_probes = np.load(
            current_folder + "/bare_probe.npy", allow_pickle=True)
        physics_probes = np.load(
            current_folder + "/fzx_probe.npy", allow_pickle=True)

        bare_probes = np.array(bare_probes)
        self.figure_probe.clf()

        # Bare probe scan
        sorting_order = np.argsort(xlabels)
        bare_probes = bare_probes[sorting_order]

        ax = self.figure_probe.add_subplot(1, 2, 1)
        try:
            extent = [-4, 4, np.max(keys_adjusted) + np.diff(
                keys_adjusted[sorting_order])[0], np.min(keys_adjusted)]
            cax = ax.imshow(bare_probes, aspect="auto", extent=extent)
            ax.set_ylabel(f"{xlabel} ({units})")
            ax.set_ylabel(f"Frequency (MHz)")
            self.figure_probe.colorbar(cax, ax=ax)
        except TypeError:
            print(bare_probes.shape)
            """
            TODO: Add in error handling
            """
        except IndexError:
            print(bare_probes.shape)
        # Physics probe
        physics_means = np.array(
            [self.__mean_probe_value__(i) for i in physics_probes]
        )
        ax = self.figure_probe.add_subplot(1, 2, 2)
        transparent_edge_plot(
            ax, keys_adjusted[sorting_order], physics_means[sorting_order])
        ax.axhline(float(self.probe_threshold_value),
                   ls='--',  c="r", label="Probe Threshold")
        ax.legend()
        ax.set_ylabel("Mean APD Voltage")
        ax.set_xlabel(f"{xlabel} ({units})")
        self.save_figure(self.figure_probe, "probe", current_folder)
        self.canvas_probe.draw()

    def __mean_probe_value__(self, probe):
        if len(probe) > 65:
            bg = probe[:65]
            trans = probe[65:-35]
            return np.mean(trans) - np.mean(bg)
        return 0

    def __get_magnetization__(self, mean, roi_labels):
        mag = (
            (mean[:, roi_labels.index("roi11"), :]
             - mean[:, roi_labels.index("roi1-1"), :])
            / (mean[:, roi_labels.index("roi11"), :]
               + mean[:, roi_labels.index("roi1-1"), :]
               + mean[:, roi_labels.index("roi10"), :]
               )
        )
        return mag

    def make_magnetization_plot(self):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)
        fits = np.load(current_folder + "/all_fits.npy")
        if len(fits) < 2:
            return
        xlabels = np.load(current_folder + "/xlabels.npy")
        physics_probes = np.load(
            current_folder + "/fzx_probe.npy", allow_pickle=True)
        fits, xlabels = self.select_probe_threshold(
            fits, xlabels, physics_probes)
        if len(xlabels) < 2:
            return
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_mean, fit_std, xlabels = self.group_shot(fits, xlabels)
        keys_adjusted = np.array(xlabels) * sf
        self.figure_corr.clf()
        ax = self.figure_corr.add_subplot(1, 1, 1)
        if len(keys_adjusted) < 2:
            return
        pol = self.__get_magnetization__(fit_mean, roi_labels)
        extent = [-0.5, fit_mean.shape[2] + 0.5,
                  np.max(keys_adjusted) + np.diff(keys_adjusted)[0],
                  np.min(keys_adjusted)]
        cax = ax.imshow(
            pol, aspect="auto", cmap=magnetization_colormap, extent=extent, vmin=-1, vmax=1)
        self.figure_corr.colorbar(cax, ax=ax)
        ax.set_ylabel(f"{xlabel} ({units})")
        ax.set_xlabel(f"Trap Index")
        self.save_figure(self.figure_corr, "magnetization", current_folder)
        self.canvas_corr.draw()

    def make_2d_plot(self):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)
        fits = np.load(current_folder + "/all_fits.npy")
        if len(fits) < 2:
            return
        xlabels = np.load(current_folder + "/xlabels.npy")
        physics_probes = np.load(
            current_folder + "/fzx_probe.npy", allow_pickle=True)
        fits, xlabels = self.select_probe_threshold(
            fits, xlabels, physics_probes)
        if len(xlabels) < 2:
            return
        roi_labels = np.load(current_folder + "/roi_labels.npy")
        fit_mean, fit_std, xlabels = self.group_shot(fits, xlabels)
        fit_mean, fit_std = np.swapaxes(
            fit_mean, 0, 1), np.swapaxes(fit_std, 0, 1)

        self.figure_2d.clf()
        rois_to_plot = [i for i in range(
            len(roi_labels)) if roi_labels[i] not in self.rois_to_exclude]
        n_rows, n_columns = 2, math.ceil(len(rois_to_plot) / 2)
        keys_adjusted = np.array(xlabels) * sf
        if len(keys_adjusted) < 2:
            return
        extent = [-0.5, fit_mean.shape[2] + 0.5,
                  np.max(keys_adjusted) + np.diff(keys_adjusted)[0],
                  np.min(keys_adjusted)]
        for e, i in enumerate(rois_to_plot):
            label = roi_labels[i]
            ax = self.figure_2d.add_subplot(n_rows, n_columns, e + 1)
            cax = ax.imshow(fit_mean[i], aspect="auto",
                            cmap=cm.blues, extent=extent, vmin=0)
            self.save_array(fit_mean[i], label, current_folder)
            self.figure_2d.colorbar(cax, ax=ax, label="Fitted counts")
            if label in fancy_titles.keys():
                ax.set_title(fancy_titles[label])
            else:
                ax.set_title(label)
            ax.set_xlabel("Trap Index")
            ax.set_ylabel(f"{xlabel} ({units})")
        self.save_figure(self.figure_2d, "2d_plot", current_folder)
        self.canvas_2d.draw()

    def make_1d_plot(self):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        print("1D Plot: ", current_folder)
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)
        fits = np.load(current_folder + "/all_fits.npy")
        xlabels = np.load(current_folder + "/xlabels.npy")
        physics_probes = np.load(
            current_folder + "/fzx_probe.npy", allow_pickle=True)
        fits, xlabels = self.select_probe_threshold(
            fits, xlabels, physics_probes)
        roi_labels = np.load(current_folder + "/roi_labels.npy")
        fit_mean, fit_std, xlabels = self.group_shot(fits, xlabels)
        fit_mean, fit_std = np.swapaxes(
            fit_mean, 0, 1), np.swapaxes(fit_std, 0, 1)
        keys_adjusted = np.array(xlabels) * sf

        self.axis_1d.clear()
        for state, state_std, label in zip(fit_mean, fit_std, roi_labels):
            if label not in self.rois_to_exclude:
                transparent_edge_plot(self.axis_1d,
                                      keys_adjusted,
                                      np.mean(state, axis=1),
                                      np.mean(state_std, axis=1),
                                      label=fancy_titles[label])
                self.save_array(np.mean(state, axis=1),
                                f"{label}_1d", current_folder)
        self.axis_1d.legend()
        self.axis_1d.set_ylabel("Average trap counts")
        self.axis_1d.set_xlabel(f"{xlabel} ({units})")
        self.save_figure(self.figure_1d, "1d_plot", current_folder)
        self.canvas_1d.draw()

    def make_correlation_plot(self):
        n_traps = 18
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        fits = np.load(current_folder + "/all_fits.npy")
        xlabels = np.load(current_folder + "/xlabels.npy")
        physics_probes = np.load(
            current_folder + "/fzx_probe.npy", allow_pickle=True)
        fits, xlabels = self.select_probe_threshold(
            fits, xlabels, physics_probes)
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[:, roi_labels.index('roi10')]
        fit_1m1 = fits[:, roi_labels.index('roi1-1')]
        fit_1p1 = fits[:, roi_labels.index('roi11')]
        fit_sum = fit_10 + fit_1m1 + fit_1p1
        fit_1m1, fit_1p1 = fit_1m1 / fit_sum, fit_1p1 / fit_sum
        threshold_low = self.corr_threshold_min.value() / 100
        threshold_high = self.corr_threshold_max.value() / 100
        sidemode = np.mean(fit_1m1 + fit_1p1, axis=1)
        threshold_index = np.where((sidemode >= threshold_low) &
                                   (sidemode < threshold_high))

        if threshold_low > threshold_high:
            return
        fit_1m1 = fit_1m1[threshold_index]
        fit_1p1 = fit_1p1[threshold_index]
        corr = np.corrcoef(fit_1m1.T, fit_1p1.T)[:n_traps, n_traps:]
        self.figure_corr.clear()
        self.axis_corr = (self.figure_corr.add_subplot(1, 2, 1),
                          self.figure_corr.add_subplot(1, 2, 2))
        cax = self.axis_corr[0].imshow(
            corr,
            aspect="auto",
            interpolation="None",
            vmin=-1,
            vmax=1,
            cmap=correlation_colormap)
        self.axis_corr[0].set_xlabel("1, -1 trap index")
        self.axis_corr[0].set_ylabel("1, 1 trap index")
        self.axis_corr[0].set_title("Total")

        if len(fit_1m1) > 8:
            adjacent_corr = np.mean(np.array(
                [np.corrcoef(fit_1m1.T[:, i:i + 2], fit_1p1.T[:, i:i + 2])[:n_traps, n_traps:]
                 for i in range(len(fit_1m1) - 2)]), axis=0)
            cax = self.axis_corr[1].imshow(
                adjacent_corr, aspect="auto", interpolation="None", vmin=-1, vmax=1, cmap=correlation_colormap)
            self.axis_corr[1].set_xlabel("1, -1 trap index")
            self.axis_corr[1].set_ylabel("1, 1 trap index")
            self.axis_corr[1].set_title("Adjacent")
            self.save_array(adjacent_corr, "corr_adjacent", current_folder)

        try:
            if not self.corr_cb:
                self.corr_cb = self.figure_corr.colorbar(
                    cax, ax=self.axis_corr[1])
        except:
            print("No colorbar")
            self.corr_cb = self.figure_corr.colorbar(cax, ax=self.axis_corr[1])

        self.save_array(corr, "corr_total", current_folder)
        self.save_figure(self.figure_corr, "2d_correlation", current_folder)

        self.figure_phase.clf()

        ax_total = self.figure_phase.add_subplot(1, 2, 1)
        positions = list(range(-n_traps + 1, n_traps))
        total_diag = [np.mean(np.diagonal(corr, d)) for d in positions]
        ax_total.plot(positions, total_diag)
        try:
            adjacent_diag = [np.mean(np.diagonal(adjacent_corr, d))
                             for d in positions]
            ax_adjacent = self.figure_phase.add_subplot(1, 2, 2)
            ax_adjacent.plot(positions, adjacent_diag)
            ax_adjacent.set_ylim(None, 1)
            ax_adjacent.set_xlabel("Distance (sites)")
            ax_adjacent.set_ylabel("Correlation")
            self.save_array(adjacent_diag, "corr_1d_adjacent", current_folder)
        except Exception as e:
            print(e)
        ax_total.set_xlabel("Distance (sites)")
        ax_total.set_ylabel("Correlation")
        ax_total.set_ylim(None, 1)
        self.save_array(total_diag, "corr_1d_total", current_folder)
        self.save_figure(self.figure_phase, "1d_correlation", current_folder)
        self.canvas_corr.draw()
        self.canvas_phase.draw()

    def make_phase_plot(self):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)
        if xlabel != "PR_IntDuration" and xlabel != "OG_Duration":
            return
        globals_list = np.load(current_folder + "/globals.npy",
                               allow_pickle=True)
        fits = np.load(current_folder + "/all_fits.npy")
        if len(fits) < 2:
            return
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        phase_dict = self.group_shot_globals(fits,
                                             globals_list,
                                             "Raman_RamseyPhase")
        if len(phase_dict.keys()) < 2:
            return
        fits_x, xlabels = self.sort_phase_list(phase_dict[0], xlabel)
        fits_y, _ = self.sort_phase_list(phase_dict[90], xlabel)
        if len(fits_x) != len(fits_y):
            print(f"Incompatible x/y lengths {len(fits_x)} and {len(fits_y)}")
            return
        keys_adjusted = np.array(xlabels) * sf
        x_pol = self.__get_magnetization__(fits_x, roi_labels)
        y_pol = self.__get_magnetization__(fits_y, roi_labels)
        n_traps = fits_x.shape[2]
        if len(keys_adjusted) < 2:
            return
        extent = [-0.5, n_traps + 0.5, np.max(keys_adjusted) + np.diff(keys_adjusted)[0],
                  np.min(keys_adjusted)]
        c_num = x_pol + 1j * y_pol
        if len(c_num) == 1:
            c_num = np.array([c_num])
        phase = np.angle(c_num)
        contrast = np.abs(c_num)
        self.save_array(phase, "phase", current_folder)
        self.save_array(contrast, "contrast", current_folder)
        nrows, ncolumns = 2, 2

        self.figure_phase.clf()

        ax_phase = self.figure_phase.add_subplot(nrows, ncolumns, 1)
        ax_phase.set_title("Phase")
        ax_phase.set_ylabel(f"{xlabel} ({units})")
        cax = ax_phase.imshow(
            phase, cmap=phase_colormap, aspect="auto", extent=extent,
            vmin=-np.pi, vmax=np.pi)
        self.figure_phase.colorbar(cax)

        ax_contrast = self.figure_phase.add_subplot(nrows, ncolumns, 2)

        ax_contrast.set_title("Contrast")
        ax_contrast.set_ylabel(f"{xlabel} ({units})")
        cax = ax_contrast.imshow(contrast, cmap=contrast_colormap, aspect="auto", extent=extent,
                                 vmin=0, vmax=1)
        self.figure_phase.colorbar(cax, ax=ax_contrast)

        ax_x = self.figure_phase.add_subplot(nrows, ncolumns, 3)
        ax_x.set_title("Phase = 0")
        ax_x.set_ylabel(f"{xlabel} ({units})")
        cax = ax_x.imshow(x_pol, cmap=magnetization_colormap, aspect="auto", extent=extent,
                          vmin=-1, vmax=1)
        self.figure_phase.colorbar(cax, ax=ax_x)

        ax_y = self.figure_phase.add_subplot(nrows, ncolumns, 4)
        ax_y.set_title("Phase = 90")
        ax_y.set_ylabel(f"{xlabel} ({units})")
        cax = ax_y.imshow(y_pol, cmap=magnetization_colormap, aspect="auto", extent=extent,
                          vmin=-1, vmax=1)
        self.figure_phase.colorbar(cax, ax=ax_y)
        self.save_figure(self.figure_phase, "phase_plot", current_folder)

        self.canvas_phase.draw()

    def sort_phase_list(self, fits, parameter):
        """
        Parameters
        ----------
        fits : list of (fit, globals)
            DESCRIPTION.

        Returns
        -------
        fits : sorted, averaged
        xlabels: sorted
        """
        fits, globals_list = zip(*fits)
        fit_dict = self.group_shot_globals(fits, globals_list, parameter)
        for key, value in fit_dict.items():
            fit_dict[key] = [i[0] for i in value]
        labels = sorted(fit_dict.keys())
        means = np.array([np.mean(fit_dict[label], axis=0)
                          for label in labels])
        #stds = np.array([np.std(fit_dict[label], axis = 0) for label in labels])
        return means, labels

    def group_shot_globals(self, fits, globals_list, parameter):
        fit_dict = collections.defaultdict(list)
        for fit, fit_global in zip(fits, globals_list):
            value = fit_global[parameter]
            if hasattr(value, '__iter__'):
                fit_dict[value[0]].append((fit, fit_global))
            else:
                fit_dict[value].append((fit, fit_global))
        return fit_dict

    def group_shot(self, fits, labels):
        fit_dict = collections.defaultdict(list)
        for fit, label in zip(fits, labels):
            if hasattr(label, '__iter__'):
                fit_dict[label[0]].append(fit)
            else:
                fit_dict[label].append(fit)

        labels = sorted(fit_dict.keys())
        means = np.array([np.mean(fit_dict[label], axis=0)
                          for label in labels])
        stds = np.array([np.std(fit_dict[label], axis=0) for label in labels])
        return means, stds, labels


if __name__ == "__main__":
    app = QApplication([])
    window = AnalysisGUI(app)
    window.show()
    sys.exit(app.exec_())
