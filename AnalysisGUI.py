# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:07:40 2020

@author: Quantum Engineer
"""
import csv
import time
import units
import sys
sys.path.append("Z://")
from runmanager.remote import Client

from scipy.optimize import curve_fit
from fit_functions import lorentzian
from datetime import datetime
import PlotWorkers
from PlotWorkers import Plot1DWorker, Plot2DWorker, Plot1DHistogramWorker, PlotPCAWorker, PlotCorrelationWorker, PlotXYWorker
from numpy import array
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import AnalysisFunctions as af
from colorcet import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from AnalysisUI import AnalysisUI
from FileSorter import FileSorter
from plotformatting import *
from FormattingStrings import *
from units import unitsDef
import importlib
import collections
import math
import traceback
import json
import os
import AnalysisFunctions

pi = np.pi


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
        self.index_lineedit.returnPressed.connect(self.set_list_index)
        self.f2_threshold_input.returnPressed.connect(self.set_f2_threshold)
        self.f2_threshold_checkbox.stateChanged.connect(
            self.set_f2_threshold)
        self.go_button.clicked.connect(
            self.make_sorter_thread)  # self.sort_all_files)
        self.checkbox_imaging_calibration.stateChanged.connect(
            self.set_imaging_calibration)
        self.checkbox_adjust_amplitudes.stateChanged.connect(
            self.set_amplitude_feedback)
        self.checkbox_ignore_first_shot.stateChanged.connect(
            self.set_ignore_first_shot
        )
        self.checkbox_adjust_probe.stateChanged.connect(
            self.set_adjust_probe
        )
        self.checkbox_shot_alert.stateChanged.connect(
            self.set_shot_alerts
        )
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
        self.adjust_probe = False
        self.rm_client = Client(host='171.64.56.36')
        self.threadpool = QThreadPool()
        self.parameters = ""
        self.f2_threshold = 0
        self.list_index = 0
        self.ignore_first_shot = False
        self.updated_folders = []  # BAD HACK BAD
        self.plot_saver = PlotWorkers.PlotSaveWorker()
        self.plot_saver.start()

    def set_shot_alerts(self):
        try:
            self.worker.alert_system.do_alerts = self.checkbox_shot_alert.isChecked()
        except Exception as e:
            traceback.print_exc()

    def set_date(self, date):
        self.date = date.toString(date_format_string)
        self.set_script_cb()
        # self.script_folder = ""

    def set_imaging_calibration(self):
        self.imaging_calibration = self.checkbox_imaging_calibration.isChecked()
        try:
            self.worker.imaging_calibration = self.imaging_calibration
        except Exception as e:
            print(e, "trying to turn on imaging calibration")

    def set_ignore_first_shot(self):
        self.ignore_first_shot = self.checkbox_ignore_first_shot.isChecked()
        try:
            self.make_plots()
        except:
            print("Error making plots after setting ignore first shots")
            traceback.print_exc()

    def set_delete_reps(self):
        delete_reps = self.checkbox_delete_reps.isChecked()
        try:
            self.worker.delete_reps = delete_reps
        except AttributeError as e:
            print(
                f"Trying to set delete reps to {delete_reps}, but no file sorter worker yet...")

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

    def set_adjust_probe(self):
        self.adjust_probe = self.checkbox_adjust_probe.isChecked()
        return

    def set_script_cb(self):
        self.cb_script.clear()
        try:
            folders = af.get_immediate_child_directories(
                af.get_date_data_path(self.date))
            folders = [af.get_folder_base(i) for i in folders]
            self.cb_script.addItems(folders)
            if len(folders) > 0:
                self.script_folder = folders[0]
            self.set_script_folder(0)
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
        self.cb_data.clear()
        try:
            folders = af.get_immediate_child_directories(self.holding_folder)
            folders = [af.get_folder_base(i) for i in folders]
            self.cb_data.addItems(folders)
        except FileNotFoundError:
            print("Selected bad date, or bad folder?")
        return

    def set_list_index(self):
        try:
            self.list_index = int(self.index_lineedit.text())
        except:
            self.list_index = 0
        if self.worker:
            self.worker.list_index = self.list_index

    def set_f2_threshold(self):
        try:
            self.f2_threshold = self.f2_threshold_checkbox.isChecked() * \
                float(self.f2_threshold_input.text())
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.f2_threshold = 0
        try:
            self.make_plots(self.folder_to_plot)
        except:
            traceback.print_exc()

    def set_corr_threshold(self):
        min_value = self.corr_threshold_min.value() / 100
        self.corr_min_value.setText(f"{min_value:.2f}")

        max_value = self.corr_threshold_max.value() / 100
        self.corr_max_value.setText(f"{max_value:.2f}")

        try:
            self.make_plots(self.folder_to_plot)
        except:
            traceback.print_exc()

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
            print("Can't set probe threshold, because no folder to plot set yet.")
        except Exception as e:
            print(e)
            traceback.print_exc()
            # TODO: Error Handling

    def set_data_folder(self):
        folder_to_plot = self.cb_data.currentText()
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
            self.worker.alert_system.do_alerts = self.checkbox_shot_alert.isChecked()
            self.worker.signal_output.folder_output_signal.connect(
                self.make_plots)
            self.worker.start()
            self.worker.list_index = self.list_index
            self.go_button.setText("Stop")
        elif self.go_button.text() == 'Stop':
            self.worker.stop()
            self.go_button.setText("Go")
            print("Stopping sorting thread")

    def stop_sorting(self):
        try:
            self.worker.folder_to_plot = False
            self.worker.stop()
            self.go_button.setText("Go")
            print("Stopping sorting thread...")
        except Exception as e:
            print(e)
            print("TODO: Add nicer error handling")

    def save_figure(self, fig, title, current_folder, extra_directory="",
                    extra_title=""):
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
        save_location = u'\\\\?\\' + \
            f"{save_folder}{self.folder_to_plot}_{title}.png"
        fig.savefig(save_location, dpi=200)

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

    def __seconds_since_midnight__(self) -> float:
        """
        How many seconds has it been since midnight on the same day?

        :returns seconds since midnight: float
        """
        now = datetime.now()
        seconds_since_midnight = (
            now - now.replace(hour=0, minute=0, second=0, microsecond=0)
        ).total_seconds()
        return seconds_since_midnight

    def update_date(self):
        """
        Update the date if it changed
        """
        current_date = QtCore.QDate.currentDate().toString(date_format_string)
        if current_date != self.date and self.go_button.text() == 'Stop':
            time.sleep(60 * 40)
            self.stop_sorting()
            self.date = current_date
            self.picker_date.setDate(QtCore.QDate.currentDate())
            self.set_date(QtCore.QDate.currentDate())
            self.make_sorter_thread()
        return

    def make_plots(self, folder_to_plot):
        """
        Make all the plots for some particular folder.
        """
        import units
        importlib.reload(units)
        from units import unitsDef

        import PlotWorkers
        importlib.reload(PlotWorkers)
        from PlotWorkers import Plot1DWorker, Plot2DWorker, Plot1DHistogramWorker, PlotPCAWorker, PlotCorrelationWorker, PlotXYWorker
        self.check_roi_boxes()
        self.folder_to_plot = folder_to_plot
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)

        fits = np.load(current_folder + "/all_rois.npy")
        fits = np.apply_along_axis(
            AnalysisFunctions.get_trap_counts_from_roi, 2, fits)
        print(fits.shape)
        fits = np.load(current_folder + "/all_anums.npy")
        print(fits.shape)
        # fits = AnalysisFunctions.get_atom_number_from_fluorescence(fits)
        xlabels = np.load(current_folder + "/xlabels.npy")
        physics_probes = np.load(
            current_folder + "/fzx_probe.npy", allow_pickle=True)
        if self.ignore_first_shot:
            fits = fits[1:]
            xlabels = xlabels[1:]
            physics_probes = physics_probes[1:]
        fits, xlabels = self.select_probe_threshold(
            fits, xlabels, physics_probes)
        roi_labels = np.load(current_folder + "/state_labels.npy")
#        fits, xlabels = self.select_f2_threshold(fits, xlabels, roi_labels)
        if len(fits) < 1:
            return
        fit_mean, fit_std, xlabels = self.group_shot(fits, xlabels)

        # roi x Shots X Site
        fit_mean, fit_std = np.swapaxes(
            fit_mean, 0, 1), np.swapaxes(fit_std, 0, 1)

        keys_adjusted = np.array(xlabels) * sf
        plot1dworker = Plot1DWorker(current_folder, self.figure_1d, xlabel, units,
                                    fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude)
        plot1dworker.f2_threshold = self.f2_threshold
        plot2dworker = Plot2DWorker(current_folder, self.figure_2d, xlabel, units,
                                    fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude)
        try:
            if b_field_check_string in self.folder_to_plot and self.folder_to_plot[:5] not in self.updated_folders:
                if xlabel == 'iteration':
                    self.adjust_raman_field_ramsey(fit_mean, xlabels)
            if b_field_check_imaging_string in self.folder_to_plot and self.folder_to_plot[:5] not in self.updated_folders:
                if xlabel == "MS_CheckFieldDetuning":
                    self.adjust_imaging_field(fit_mean, xlabels)
                elif xlabel == 'iteration':
                    self.adjust_imaging_field_ramsey(fit_mean, xlabels)
        except:
            traceback.print_exc()

        try:
            self.threadpool.start(plot1dworker)
            self.threadpool.start(plot2dworker)
            if self.amplitude_feedback and ('reps' in folder_to_plot or 'iteration' in folder_to_plot):
                n_traps = fit_mean.shape[-1]
                print(fit_mean.shape)
                self.adjust_amplitude_compensation(fit_mean, n_traps)
            self.make_probe_plot()
            plotXYWorker = PlotXYWorker(
                current_folder, self.figure_6, xlabel, units, fits, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude
            )
            if "PairCreation" in self.folder_to_plot and 'time' not in self.folder_to_plot or 'iteration' in self.folder_to_plot:
                self.canvas_corr.setFixedHeight(600)
                threshold_low = self.corr_threshold_min.value() / 100
                threshold_high = self.corr_threshold_max.value() / 100
                plotPCAworker = PlotPCAWorker(
                    current_folder, self.figure_phase, xlabel, units, fits, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude
                )
                plotCorrelationWorker = PlotCorrelationWorker(
                    current_folder, self.figure_corr, xlabel, units, fits, fit_std, roi_labels, keys_adjusted, rois_to_exclude=self.rois_to_exclude
                )
                plotCorrelationWorker.set_limits(threshold_low, threshold_high)
                plotCorrelationWorker.set_normalize(
                    self.checkbox_normalize_correlations.isChecked())
                plotXYWorker.xlabels = np.load(current_folder + "/xlabels.npy")
                self.threadpool.start(plotXYWorker)

                self.threadpool.start(plotPCAworker)
                # self.threadpool.start(plotCorrelationWorker)
            elif "IntDuration" in self.folder_to_plot or "OG_Duration" in self.folder_to_plot or "SpinExchange" in self.folder_to_plot or 'PhaseImprintPhase' in self.folder_to_plot:
                self.canvas_corr.setFixedHeight(600)
                try:
                    self.make_phase_plot(sf, units)
                except:
                    "Error Making Phase Plot"
                    traceback.print_exc()
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
        except:
            traceback.print_exc()

        if self.adjust_probe:
            try:
                self.adjust_probe_values()
            except:
                traceback.print_exc()

        current_date = QtCore.QDate.currentDate().toString(date_format_string)
        if current_date != self.date and self.go_button.text() == 'Stop':
            self.stop_sorting()
            time.sleep(60 * 40)
            self.date = current_date
            self.picker_date.setDate(QtCore.QDate.currentDate())
            self.set_date(QtCore.QDate.currentDate())
            self.make_sorter_thread()

    def select_probe_threshold(self, fits, xlabels, physics_probes):
        if self.checkbox_probe_threhold.isChecked():
            mean_probe = np.array([self.__mean_probe_value__(i)
                                   for i in physics_probes])
            indices_to_keep = mean_probe > self.probe_threshold_value
            return fits[indices_to_keep], xlabels[indices_to_keep]
        return fits, xlabels

    def select_f2_threshold(self, fits, xlabels, roi_labels):
        """
        Given a list of fits and xlabels, only return the fits + xlabels that
        have F = 2 population > threshold set in the QtGui

        Inputs:
            fits: Shots x States x Traps array of data
            xlabels: array of length shots with the appropriate vlaue of the xlabel
            roi_labels: The order that the states in fits are.

        Outputs:
            fits: Thresholded Shots x States x Traps aray of data
            xlabels: array of length (thresholded shots)
        """
        fit2 = fits[:, list(roi_labels).index("roi2orOther")]
        fit2 = np.mean(fit2, axis=1)
        mask = fit2 > self.f2_threshold
        return fits[mask], xlabels[mask]

    def adjust_imaging_field(self, fits, xlabels):
        """
        Adjust the frequencies of the magnetic fields to accound for slow drifts
        in the magnetic field during the imaging portion of the sequence.
        """
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[roi_labels.index('roi10')]
        fit_1m1 = fits[roi_labels.index('roi1-1')]
        fit_1p1 = fits[roi_labels.index('roi11')]
        fit_2 = fits[roi_labels.index('roi2orOther')]
        if fit_2.shape[0] < 19:
            return
        fit_2 = fit_2[:, [10]]
        fit_1m1 = fit_1m1[:, [10]]
        pol = np.sum((fit_2 - fit_1m1), axis=1) / \
            np.sum((fit_2 + fit_1m1), axis=1)
        # pol has shape n_shots X n_traps
        if np.mean(fit_2 + fit_1m1) < 200:
            return
        print("Adjusting B Field")
        optimal_detuning = xlabels[np.argmax(pol)]
        print(f"optimal_detuning: {optimal_detuning}")
        globals = self.rm_client.get_globals(raw=True)
        globals['CheckMagneticField'] = "False"
        detuning_global = globals['MS_KPDetuning']
        previous_detuning = eval(detuning_global)
        new_freq = previous_detuning + optimal_detuning
        new_freq_str = repr(new_freq)
        new_globals = {
            'MS_KPDetuning': new_freq_str
        }
        with open(f'{self.holding_folder}/b_field.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(
                [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), new_freq])

        globals['MS_KPDetuning'] = new_freq_str
        self.updated_folders.append(self.folder_to_plot[-6:])
        # self.rm_client.set_globals(new_globals, raw = True)
        # Get Current B Field with self.rm_client

    def adjust_raman_field_ramsey(self, fits, xlabels, n_shots=4):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[roi_labels.index('roi10')]
        fit_1m1 = fits[roi_labels.index('roi1-1')]
        fit_1p1 = fits[roi_labels.index('roi11')]
        fit_2 = fits[roi_labels.index('roi2orOther')]

        print('Starting magnetic field analysis')
        pol = np.sum((fit_1p1 - fit_1m1), axis=0) / \
            np.sum((fit_1p1 + fit_1m1 + fit_10), axis=0)
        print(pol.shape)
        pol = pol[[10, ]]  # TODO: Fix
        current_globals = np.load(
            current_folder + "globals.npy", allow_pickle=True)[0]
        ramsey_time = current_globals['Raman_CheckMagTime']

        if fit_1p1.shape[0] < n_shots or np.max(fit_1p1) < 100:
            return
        print("Adjusting B field via Ramsey: ")
        adjusted_delta = np.round(
            np.mean(np.arcsin(pol) / (2 * pi * ramsey_time) * 1e-6), 4)

        detuning_global = current_globals['Raman_BareFreq']
        new_freq = np.array([detuning_global + adjusted_delta])
        new_freq_str = repr(new_freq)
        new_globals = {
            'Raman_BareFreq': new_freq_str,
            'CheckMagneticField': 'False'
        }

        with open(f'{self.holding_folder}/b_field.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'), 'Raman', detuning_global + adjusted_delta])
        print(f"Adjusted Raman field from {detuning_global} to {new_freq_str}")
        self.rm_client.set_globals(new_globals, raw=True)
        # self.rm_client.engage()
        # self.rm_client.set_globals(
        #     {'CheckSimultaneousReadout': 'False'}, raw=True)

        self.updated_folders.append(self.folder_to_plot[-6:])
        return

    def adjust_imaging_field_ramsey(self, fits, xlabels, n_shots=4):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[roi_labels.index('roi10')]
        fit_1m1 = fits[roi_labels.index('roi1-1')]
        fit_1p1 = fits[roi_labels.index('roi11')]
        fit_2 = fits[roi_labels.index('roi2orOther')]

        print('Starting imaging magnetic field analysis')
        pol = np.sum((fit_2 * 1.1 - fit_1m1), axis=0) / \
            np.sum((fit_2 * 1.1 + fit_1m1), axis=0)
        print(pol.shape)
        pol = pol[[10, ]]  # TODO: Fix
        current_globals = np.load(
            current_folder + "globals.npy", allow_pickle=True)[0]
        ramsey_time = current_globals['MS_CheckFieldWaitTime']
        ramsey_detuning = current_globals['MS_CheckFieldDetuning']

        if fit_2.shape[0] < n_shots or np.max(fit_2) < 500:
            return
        print("Adjusting B field via Ramsey: ")
        adjusted_delta = np.round(
            np.mean(ramsey_detuning - np.arccos(pol) / (2 * pi * ramsey_time) * 1e-6), 4)

        if np.abs(adjusted_delta) > 5e-3 or np.isnan(adjusted_delta):
            return
        detuning_global = current_globals['MS_KPDetuning']
        new_freq = np.array([detuning_global + adjusted_delta])
        new_freq_str = repr(new_freq)
        new_globals = {
            'MS_KPDetuning': new_freq_str
        }

        with open(f'{self.holding_folder}/b_field.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'), 'Imaging', detuning_global + adjusted_delta])
        print(
            f"Adjusted imaging field from {detuning_global} to {new_freq_str}")
        self.rm_client.set_globals(new_globals, raw=True)

        self.updated_folders.append(self.folder_to_plot[-6:])
        return

    def run_spectroscopy(self, type="imaging"):
        new_globals = {
            'CheckMagneticField': 'True',
            'MeasurePairCreation': 'False',
            'PR_WaitTime': '0',
            'Tweezers_AOD0_LoadAmp': '21',
            'Tweezers_AOD0_ImageAmp': 'Tweezers_AOD0_LoadAmp + 3',
            'MS_CheckFieldWaitTime': '0',
            'MS_CheckFieldDetuning': '1e-3 * np.concatenate([arange(-40, 50, 10), arange(-5, 5, 1)])',
            'Descriptor': "'CheckSpectroscopy'",
            'iteration': 'arange(1)'
        }
        # self.rm_client.set_globals(new_globals, raw = True)
        # self.rm_client.engage()

    def check_field(self, fits, xlabel, type="imaging", n_shots=2):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[roi_labels.index('roi10')]
        fit_1m1 = fits[roi_labels.index('roi1-1')]
        fit_1p1 = fits[roi_labels.index('roi11')]
        fit_2 = fits[roi_labels.index('roi2orOther')]
        if fit_2.shape[0] < n_shots or np.max(fit_2) < 100:
            return
        pol = np.sum((fit_2 - fit_1m1), axis=0) / \
            np.sum((fit_2 + fit_1m1), axis=0)
        pol = pol[[10, ]]  # TODO: Fix
        print("Adjusting B field via Ramsey: ")
        print(pol)
        # If the pi pulse is not really a pi pulse, then do it with spectroscopy
        if pol < 0.7:
            self.run_spectroscopy(type)
            self.updated_folders.append(self.folder_to_plot[-6:])
        return

    def adjust_amplitude_compensation(self, fits, n_traps):
        """
        Look at the current atom uniformity, and update the relative trap powers
        to optimize uniformity.  This function engages with the runmanager remote
        client to automatically engage the next set.

        Inputs: n_traps
        """
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        print("Amplitude Compensation")
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[roi_labels.index('roi10')]
        fit_1m1 = fits[roi_labels.index('roi1-1')]
        fit_1p1 = fits[roi_labels.index('roi11')]
        fit_sum = fit_10 + fit_1m1 + fit_1p1
        print(fit_sum.shape)
        trap_values = np.mean(fit_sum, axis=0)
        assert n_traps == len(trap_values), f"{n_traps} {len(trap_values)}"
        # Reverse, since the frequency -> trap ordering is reversed
        if len(fit_sum) > 2 and len(fit_sum) % 4 == 0:
            sites = np.arange(4, 15, 2)  # [4, 8, 12, 16]
            saved_path = compensation_path(len(sites))
            print(f"Attempting to load from {saved_path}")
            try:
                current_compensation = np.load(saved_path)
            except Exception as e:
                print(e)
                current_compensation = np.ones(n_traps)
            trap_values = np.mean(fit_sum, axis=0)
            # compensation = trap_values[::-1] ** (-1 / 3)
            compensation = trap_values ** (-1 / 3)
            compensation[np.delete(np.arange(n_traps), sites)] = 0

            new_compensation = current_compensation * compensation
            new_compensation = new_compensation / \
                np.linalg.norm(new_compensation)
            print(f"compensation calculated: {compensation}")
            print(f"New compensation = {new_compensation}")
            print(f"saving new compensation to {saved_path}")
            np.save(saved_path, new_compensation)
            print("Engaging new scan")
            self.rm_client.engage()
        return

    def __bare_probe_filter__(self, bare_probe):
        if np.max(bare_probe) < 0.12:
            return False
        return True

    def __fit_filter__(self, popt, pstd):
        print(f"POPT: {popt}")
        if popt[0] > 4 or popt[0] < 0.03:
            return False
        if popt[1] < 0.15 or popt[1] > 0.3:
            return False
        return True

    def adjust_probe_values(self):
        print("Adjusting probe values")
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        bare_probes = np.load(
            current_folder + "/bare_probe.npy", allow_pickle=True)
        bare_probes = np.array(bare_probes)
        current_globals = np.load(
            current_folder + "globals.npy", allow_pickle=True)[0]
        current_offset = current_globals[agilent_offset_string]
        current_physics_freq = current_globals[agilent_physics_string]
        offset = current_offset
        physics_freq = current_physics_freq
        #A, full_width, x0, offset
        for i in bare_probes[-20:]:
            i = i - np.mean(i[-50:])
            if self.__bare_probe_filter__(i):
                freq = np.linspace(-4, 4, len(i))
                guess = [0.1, 0.25, freq[np.argmax(i)], 0]
                popt, pcov = curve_fit(lorentzian, freq, i,
                                       bounds=([0, 0, -np.inf, -np.inf],
                                               [4, np.inf, np.inf, np.inf]),
                                       p0=guess)
                pstd = np.diag(np.sqrt(np.abs(pcov)))
                if self.__fit_filter__(popt, pstd):
                    _, _, center, _ = popt
                    offset = np.round(current_offset - center, 1)
                    physics_freq = np.round(current_physics_freq - center, 1)
        self.rm_client.set_globals({agilent_offset_string: offset})  # ,
#                                    agilent_physics_string: physics_freq})
        print("Adjusted globals to", {agilent_offset_string: offset,
                                      }, f"from {current_offset}")

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
            ax.set_xlabel(f"Frequency (MHz)")
            self.figure_probe.colorbar(cax, ax=ax)
        except TypeError:
            print("Type Error in Probe plot")
            """
            TODO: Add in error handling
            """
            traceback.print_exc()
        except IndexError:
            print("IndexError in Probe Plot")
            traceback.print_exc()
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
        af.save_array(physics_means, "mean_probe_physics", current_folder)
        if "cavity_shift" in self.folder_to_plot:
            try:
                #A, full_width, x0, offset
                popt, pcov = curve_fit(lorentzian,
                                       keys_adjusted[sorting_order],
                                       physics_means[sorting_order],
                                       bounds=([0, 0.05, min(keys_adjusted), -0.01],
                                               [0.5, 0.5, max(keys_adjusted), 0.01]))

                pstd = np.sqrt(np.diag(pcov))
                if popt[0] > 0.03 and len(keys_adjusted) > 14:
                    self.rm_client.set_globals(
                        {shifted_resonance_string: f"{popt[2]:.4g}"},
                        raw=True)
                key_fine = np.linspace(np.min(keys_adjusted),
                                       np.max(keys_adjusted))
                ax.plot(key_fine, lorentzian(key_fine, *popt))
            except:
                traceback.print_exc()
        #af.save_figure(self.figure_probe, "probe", current_folder)
        print("Putting Probe figure in queue")
        PlotWorkers.file_save_queue.put(
            (self.figure_probe, "probe", current_folder))
        self.canvas_probe.draw()

    def __mean_probe_value__(self, probe):
        if len(probe) > 65:
            bg = probe[:65]
            trans = probe[65:-35]
            return np.mean(trans) - np.mean(bg)
        return 0

    def __get_magnetization__(self, mean, roi_labels):
        print(np.max(mean))
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
        extent = [-0.5, fit_mean.shape[2] - 0.5,
                  np.max(keys_adjusted) + np.diff(keys_adjusted)[0],
                  np.min(keys_adjusted)]
        cax = ax.imshow(
            pol, aspect="auto", cmap=magnetization_colormap, extent=extent, vmin=-1, vmax=1)
        self.figure_corr.colorbar(cax, ax=ax)
        ax.set_ylabel(f"{xlabel} ({units})")
        ax.set_xlabel(f"Trap Index")
        af.save_figure(self.figure_corr, "magnetization", current_folder)
        self.canvas_corr.draw()

    def make_phase_plot(self, sf, units):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        # sf, units = unitsDef(xlabel)
        # if xlabel != "PR_IntDuration" and xlabel != "OG_Duration":
        #     return
        globals_list = np.load(current_folder + "/globals.npy",
                               allow_pickle=True)
        fits = np.load(current_folder + "/all_rois.npy")
        print(np.max(fits))
        fits = np.apply_along_axis(
            AnalysisFunctions.get_trap_counts_from_roi, 2, fits)
        fits = AnalysisFunctions.get_atom_number_from_fluorescence(fits)
        print(np.max(fits))
        if len(fits) < 2:
            return
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        phase_dict = self.group_shot_globals(fits,
                                             globals_list,
                                             "SR_GlobalLarmor")
        if len(phase_dict.keys()) < 2:
            return
        fits_x, xlabels = self.sort_phase_list(phase_dict[0], xlabel)
        fits_y, _ = self.sort_phase_list(phase_dict[90], xlabel)
        print(np.max(fits_x))
        if len(fits_x) != len(fits_y):
            print(f"Incompatible x/y lengths {len(fits_x)} and {len(fits_y)}")
            return
        keys_adjusted = np.array(xlabels) * sf
        print(f"max x:{np.max(fits_x)}")
        x_pol = self.__get_magnetization__(fits_x, roi_labels)
        y_pol = self.__get_magnetization__(fits_y, roi_labels)
        n_traps = fits_x.shape[2]
        if len(keys_adjusted) < 2:
            return
        extent = [-0.5, n_traps - 0.5, np.max(keys_adjusted) + np.diff(keys_adjusted)[0],
                  np.min(keys_adjusted)]
        c_num = x_pol + 1j * y_pol
        if len(c_num) == 1:
            c_num = np.array([c_num])
        phase = np.angle(c_num)
        contrast = np.abs(c_num)
        contrast[:, np.delete(np.arange(n_traps), [6, 10])] = 0
        phase[:, np.delete(np.arange(n_traps), [6, 10])] = 0
        print("Contrast", contrast.dtype)
        print("phase", phase)
        self.save_array(phase, "phase", current_folder)
        self.save_array(contrast, "contrast", current_folder)
        self.save_array(keys_adjusted, "keys_adjusted", current_folder)
        nrows, ncolumns = 2, 2

        self.figure_phase.clf()

        ax_phase = self.figure_phase.add_subplot(nrows, ncolumns, 1)
        ax_phase.set_title("Phase")
        ax_phase.set_ylabel(f"{xlabel} ({units})")
        cax = ax_phase.imshow(
            phase, cmap=phase_colormap, aspect="auto", extent=extent,
            vmin=-np.pi, vmax=np.pi, interpolation=None)
        self.figure_phase.colorbar(cax)

        ax_contrast = self.figure_phase.add_subplot(nrows, ncolumns, 2)

        ax_contrast.set_title("Contrast")
        ax_contrast.set_ylabel(f"{xlabel} ({units})")
        cax = ax_contrast.imshow(contrast, cmap=contrast_colormap, aspect="auto", extent=extent,
                                 vmin=0, vmax=1, interpolation=None)
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
        af.save_figure(self.figure_phase, "phase_plot", current_folder)

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
        # stds = np.array([np.std(fit_dict[label], axis = 0) for label in labels])
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


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":

    import cgitb
    cgitb.enable(format='text')
    sys.excepthook = except_hook
    app = QApplication([])
    window = AnalysisGUI(app)
    window.show()
    sys.exit(app.exec_())
