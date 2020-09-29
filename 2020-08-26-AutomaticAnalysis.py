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


import numpy as np
import sys, traceback, json, os
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from os.path import basename, normpath, getsize
import AnalysisFunctions as af
from AveragedPlots import *
from glob import glob
from colorcet import cm
import matplotlib.pyplot as plt
from shutil import move
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import readFiles as rf
from TweezerAnalysis import *


analysis_folder_string = "Folder to analyze"
analysis_label_string = "Iteration variable"
date_format_string = "yyyy-MM-dd"
no_xlabel_string = "shotnum"
fancy_titles = {"roi1-1": "1, -1 Atoms",
                "roiSum": "Total Atoms",
                "roi2orOther": "F = 2 Atoms",
                "roiRemaining": "Leftover Atoms",
                "roi10": "1, 0 Atoms",
                "roi11": "1, 1 Atoms",
                "roi1": "F = 1 Atoms",
                "roi2": "F = 2 Atoms",
                "roiMagnetization": "Magnetization",
                "roiAll": "All atoms"}

compensation_path = "S:\\Schleier Lab Dropbox\\Cavity Lab Data\\Cavity Lab Scripts\\cavity_labscriptlib\\RbCavity\\amplitude_compensation.npy"

class FileSorterSignal(QObject):
    folder_output_signal = pyqtSignal(str)

class FileSorter(QThread):
    
    def __init__(self, script_folder, date, imaging_calibration):
        super(FileSorter, self).__init__()

        self.shot_threshold_size = 10e6
        self.script_folder = script_folder
        self.date = date
        self.holding_folder = af.get_holding_folder(self.script_folder, data_date=self.date)
        self.signal_output = FileSorterSignal()
        self.running = True
        self.folder_to_plot = False
        self.imaging_calibration = imaging_calibration
        
        try:
            with open(self.holding_folder + "/folder_dict.json", 'r') as dict_file:
                self.data_folder_dict = json.loads(dict_file.read())
        except FileNotFoundError:
            print("No dictionary file yet...")
            self.data_folder_dict = {}

    def run(self):
        while self.running:
            self.all_files, self.files_to_sort = self.get_unanalyzed_files()
            if len(self.files_to_sort) > 0:
                self.sort_files(self.files_to_sort)
                self.signal_output.folder_output_signal.emit(self.folder_to_plot)
            else:
                print("No Folders made yet")
            QThread.sleep(5)
        print("Thread ended")
    
    def stop(self):
        self.running = False
        if self.folder_to_plot:
            self.signal_output.folder_output_signal.emit(self.folder_to_plot)
        self.terminate()

        
    def get_unanalyzed_files(self):
        try:
            search_folder = af.get_holding_folder(self.script_folder, data_date=self.date)
            files = glob(search_folder + "/*.h5")
            files_big = list(
                filter(lambda x: getsize(x) > self.shot_threshold_size, 
                       files))
            files_small = list(
                filter(lambda x: getsize(x) < self.shot_threshold_size, 
                       files))
            return files, files_big
        except FileNotFoundError:
            print("Selected bad date when searching for h5!")
            return []

    def sort_files(self, files):
        scans = [self.get_scan_time(file) for file in files]
        for scan, file in zip(scans, files):
            if scan not in self.data_folder_dict.keys():
               self.data_folder_dict[scan] = self.generate_folder_name(scan)
               with open(self.holding_folder + "/folder_dict.json", 'w') as dict_file:
                   json.dump(self.data_folder_dict, dict_file)
            self.folder_to_plot = self.data_folder_dict[scan]
            current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
            new_location = current_folder + basename(file)
            self.process_file(file, current_folder)
            move(file, new_location)
        return self.folder_to_plot
    
    def process_file(self, file, current_folder):
        """
        TODO: Add in probe processing
        """
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        roi_labels, rois = af.extract_rois(file)
        file_globals = af.extract_globals(file)
        fits = np.apply_along_axis(trap_amplitudes, 1, rois, n_traps = 18)
        if self.imaging_calibration:
            print("Adjusting ROIs")
            fits = self.adjust_rois(fits, roi_labels)
        physics_probe, bare_probe = self.get_cavity_transmission(file)
        try:
            all_fits = np.load(current_folder + "/all_fits.npy")
            xlabels = np.load(current_folder + "/xlabels.npy")
            globals_list = np.load(current_folder + "/globals.npy", 
                                   allow_pickle = True)
            bare_probe_list = list(np.load(current_folder + "/bare_probe.npy",
                                           allow_pickle = True))
            physics_probe_list = list(np.load(current_folder + "/fzx_probe.npy", 
                                         allow_pickle =True))
            all_fits = np.vstack([all_fits, np.array([fits])])
            bare_probe_list.append(bare_probe)
            physics_probe_list.append(physics_probe)
            if xlabel == no_xlabel_string:
                xlabel_value = np.max(xlabels) + 1
            else:
                xlabel_value = self.get_global(file_globals, xlabel)
            xlabels = np.append(xlabels, xlabel_value)
            globals_list = np.append(globals_list, file_globals)
            np.save(current_folder + "/globals.npy", globals_list)
            np.save(current_folder + "/xlabels.npy", xlabels)
            np.save(current_folder + "/all_fits.npy", all_fits)
            np.save(current_folder + "/fzx_probe.npy", physics_probe_list)
            np.save(current_folder + "/bare_probe.npy", bare_probe_list)
        except IOError:
            if xlabel == no_xlabel_string:
                xlabel_value = 0
            else:
                xlabel_value = self.get_global(file_globals, xlabel)
            globals_list = [file_globals]
            np.save(current_folder + "/globals.npy", globals_list)
            np.save(current_folder + "/roi_labels.npy", roi_labels)    
            np.save(current_folder + "/all_fits.npy", np.array([fits]))
            np.save(current_folder + "/xlabels.npy", np.array([xlabel_value]))
            np.save(current_folder + "/fzx_probe.npy", [physics_probe])
            np.save(current_folder + "/bare_probe.npy", [bare_probe])
            print("Creating fit files...")
        return
    
    def get_cavity_transmission(self, file):
        bare_probe = rf.getdata(file, "GreyCavityTransmissionBare") 
        try:
            physics_probe = rf.getdata(file, "GreyCavityTransmissionProbe")
        except Exception as e:
            print(e)
            physics_probe = [(0, 0), (1, 0)]
            """
            TODO: Error Handling
            """
        return self.__process_trace__(physics_probe), self.__process_trace__(bare_probe)
    
    def __process_trace__(self, trace):
        return [i[1] for i in trace]
    
    def adjust_rois(self, fits, roi_labels):
        if 'roi11' not in roi_labels:
            return fits
        population_to_q = np.array([[1, -1, 0], [-1, -1, 1], [1, 1, 1]])
        alphas = np.load(f"X:\\labscript\\analysis_scripts\\roi_adjustment_alpha.npy")
        def transform_r(r_measured, alpha_fit):
            a = (
                 np.linalg.inv(population_to_q)
                 @ alpha_fit
                 @ population_to_q
                 @ r_measured
                 )
            return a
        def adjust_by_site(r_tot, alphas):
            return np.array([transform_r(r_tot[:, i], alphas[i]) for i in np.arange(r_tot.shape[1])])
        r_tot = np.array([fits[roi_labels.index('roi1-1')], 
                          fits[roi_labels.index('roi11')],
                          fits[roi_labels.index('roi10')]])
        new_r_tot = adjust_by_site(r_tot, alphas).T
        fits[roi_labels.index('roi1-1')] = new_r_tot[0]
        fits[roi_labels.index('roi11')] = new_r_tot[1]
        fits[roi_labels.index('roi10')] = new_r_tot[2]
        return fits
    
    def get_global(self, file_globals, xlabel):
        xlabel_value = file_globals[xlabel]
        if hasattr(xlabel_value, '__iter__'):
            return sorted(xlabel_value)[0]
        return xlabel_value
    
    def process_xlabel(self, xlabel):
        return xlabel.split("_")[-1]
    
    def get_scan_time(self, file_name):
        scan_time = basename(file_name).split("_")[0]
        return scan_time

    def generate_folder_name(self, scan):
        print(scan)
        scan_files = [i for i in self.all_files if scan in i]
        scan_globals = af.extract_globals(scan_files[0])
        xlabels = af.get_xlabel(scan_files)
        if xlabels == []:
            main_string = "reps"
            xlabel = no_xlabel_string
        elif 'OG_Duration' in xlabels:
            main_string = 'OG_Duration'
            xlabel = "OG_Duration"
        else:
            main_string = self.process_xlabel(xlabels[0])
            xlabel = xlabels[0]
        
        if scan_globals["MeasurePairCreation"]:
            if xlabel == "PR_IntDuration":
                main_string = f"pc_time_qc{scan_globals['Cavity_QuadCoilFactor']}_amp{scan_globals['PR_RedAmp']}"
            else:
                main_string = f"pc_{scan_globals['PR_SidebandFreq'] * 1000:.0f}kHz"
        if scan_globals["CheckCavityShift"]:
            main_string = "cavity_shift"
            xlabel = "PR_DLProbe_Agilent_FlipFlopFreq"
        """
        add_extra_parameters
        """
        main_string = f"{main_string}_{scan[-6:]}"
        filename = f"{self.holding_folder}/{main_string}/xlabel.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as text_file:
            print(xlabel, file=text_file)
        return main_string

class AnalysisUI:
    def create_ui(self, AnalysisGUI):
        self.scroll_area = QScrollArea(AnalysisGUI)
        self.scroll_area.setWidgetResizable(True)
        
        self.central_widget = QWidget()
        self.grid_layout = QGridLayout(self.central_widget)
        self.scroll_area.setWidget(self.central_widget)
        
        self.picker_date = QDateEdit(calendarPopup=True)
        self.picker_date.setDisplayFormat(date_format_string)
        self.picker_date.setDate(QtCore.QDate.currentDate())
        self.cb_script = QComboBox()
        self.cb_data = QComboBox()
        
        self.checkbox_imaging_calibration = QCheckBox("Imaging Calibration")
        self.checkbox_adjust_amplitudes = QCheckBox("Amplitude Adjustment Feedback")
        
        
        self.corr_min_label = QLabel("Sidemode Minimum Fraction: ")
        self.corr_max_label = QLabel("Sidemode Maximum Fraction: ")
        
        self.corr_min_value = QLabel("0")
        self.corr_max_value = QLabel("1")
        
        self.corr_threshold_min = QSlider(Qt.Horizontal)
        self.corr_threshold_min.setRange(0, 100)
        self.corr_threshold_min.setValue(0)        
        
        self.corr_threshold_max = QSlider(Qt.Horizontal)
        self.corr_threshold_max.setRange(0, 100)
        self.corr_threshold_max.setValue(100)
        
        self.checkbox_probe_threhold = QCheckBox("Probe Thresholding")
        self.probe_threshold_label = QLabel("Probe Threshold: ")
        self.probe_threshold = QSlider(Qt.Horizontal)
        self.probe_threshold.setRange(0, 100)
        self.probe_threshold.setValue(0)
        self.probe_threshold_value_label = QLabel("0.000")
        
        self.label_folder_name = QLabel(f"{analysis_folder_string}: ")
            
        self.parameters_label = QLabel(f"Parameters: ")
        self.parameters = QLineEdit()
        
        self.rois_selection = []
        
        n_columns = 4
        
        self.go_button = QPushButton("Go")
        
        row_num = 0
        self.grid_layout.addWidget(self.picker_date, row_num, 0, 1, 1)       
        self.grid_layout.addWidget(self.cb_script, row_num, 1, 1, 1)
        self.grid_layout.addWidget(self.cb_data, row_num, 2, 1, 1)
        self.grid_layout.addWidget(
                self.label_folder_name,
                row_num, 3, 1, n_columns - 2
                )
        
        row_num = row_num + 1
        self.grid_layout.addWidget(self.checkbox_imaging_calibration, row_num, 0, 1, 1)
        self.grid_layout.addWidget(self.checkbox_adjust_amplitudes, row_num, 1, 1, 1)
        self.probe_threshold_layout = QHBoxLayout()
        self.grid_layout.addLayout(self.probe_threshold_layout, row_num, 2, 1, 5)
        self.probe_threshold_layout.addWidget(self.checkbox_probe_threhold)
        self.probe_threshold_layout.addWidget(self.probe_threshold_label)
        self.probe_threshold_layout.addWidget(self.probe_threshold)
        self.probe_threshold_layout.addWidget(self.probe_threshold_value_label)
        row_num = row_num + 1
        self.set_roi_selector(row_num)
        row_num = row_num + 1
        self.grid_layout.addWidget(self.go_button, row_num, 0, 1, self.n_columns)
        
        self.figure_1d, self.axis_1d = plt.subplots()
        self.canvas_1d = FigureCanvas(self.figure_1d)
        self.toolbar_1d = NavigationToolbar(self.canvas_1d, self)
        
        self.figure_2d, self.axis_2d = plt.subplots()
        self.canvas_2d = FigureCanvas(self.figure_2d)
        self.toolbar_2d = NavigationToolbar(self.canvas_2d, self)
        
        self.figure_corr, self.axis_corr = plt.subplots(1, 2)
        self.canvas_corr = FigureCanvas(self.figure_corr)
        self.toolbar_corr = NavigationToolbar(self.canvas_corr, self)
        
        self.figure_phase, _ = plt.subplots()
        self.canvas_phase = FigureCanvas(self.figure_phase)
        self.toolbar_phase = NavigationToolbar(self.canvas_phase, self)
        
        self.figure_probe, _ = plt.subplots()
        self.canvas_probe = FigureCanvas(self.figure_probe)
        self.toolbar_probe = NavigationToolbar(self.canvas_probe, self)
        
        row_num = row_num + 1
        self.grid_layout.addWidget(self.toolbar_1d, row_num, 0, 1, self.n_columns/2)
        self.grid_layout.addWidget(self.canvas_1d, row_num + 1, 0, 1, self.n_columns/2)
        
        self.grid_layout.addWidget(self.toolbar_2d, row_num, self.n_columns/2, 1, self.n_columns/2)
        self.grid_layout.addWidget(self.canvas_2d, row_num + 1, self.n_columns/2, 1, self.n_columns/2)

        row_num = row_num + 2
        self.corr_threshold_layout = QHBoxLayout()
        self.grid_layout.addLayout(self.corr_threshold_layout, row_num, 0, 1, self.n_columns)
        self.corr_threshold_layout.addWidget(self.corr_min_label)
        self.corr_threshold_layout.addWidget(self.corr_min_value)
        self.corr_threshold_layout.addWidget(self.corr_threshold_min)
        self.corr_threshold_layout.addWidget(self.corr_max_label)
        self.corr_threshold_layout.addWidget(self.corr_max_value)
        self.corr_threshold_layout.addWidget(self.corr_threshold_max)

        row_num = row_num + 1
        self.grid_layout.addWidget(self.toolbar_corr, row_num, 0, 1, self.n_columns/2)
        self.grid_layout.addWidget(self.canvas_corr, row_num + 1, 0, 1, self.n_columns/2)
                
        self.grid_layout.addWidget(self.toolbar_phase, row_num, self.n_columns/2, 1, self.n_columns/2)
        self.grid_layout.addWidget(self.canvas_phase, row_num + 1, self.n_columns/2, 1, self.n_columns/2)
        
        row_num = row_num + 2
        self.grid_layout.addWidget(self.toolbar_probe, row_num, 0, 1, self.n_columns/2)
        self.grid_layout.addWidget(self.canvas_probe, row_num + 1, 0, 1, self.n_columns/2)
        
        self.canvas_1d.setFixedHeight(600)
        self.canvas_corr.setFixedHeight(600)
        self.canvas_probe.setFixedHeight(600)
        
        AnalysisGUI.setCentralWidget(self.scroll_area)
        QtCore.QMetaObject.connectSlotsByName(AnalysisGUI)

    def set_roi_selector(self, row_num = 2):
        self.n_columns = 12
        self.roi_selector_label = QLabel("ROIs to exclude:")
        rois = sorted(list(fancy_titles.keys()))
        self.roi_checkboxes = [QCheckBox(roi) for roi in rois]
        self.grid_layout.addWidget(self.roi_selector_label, 2, 0, 1, 1)
        for i in range(len(self.roi_checkboxes)):
            self.grid_layout.addWidget(self.roi_checkboxes[i], 2, i + 1, 1, 1)
        
    def check_roi_boxes(self):
        self.rois_to_exclude = [i.text() for i in self.roi_checkboxes if i.isChecked()]  
        return


    
class AnalysisGUI(QMainWindow, AnalysisUI):
    def __init__(self, app):
        AnalysisUI.__init__(self)
        QMainWindow.__init__(self)
        
        app.aboutToQuit.connect(self.stop_sorting)
        self.create_ui(self)
        self.cb_script.currentIndexChanged.connect(self.set_script_folder)
        self.cb_data.activated.connect(self.set_data_folder)
        self.picker_date.dateChanged.connect(self.set_date)
        self.go_button.clicked.connect(self.make_sorter_thread)#self.sort_all_files)
        self.checkbox_imaging_calibration.stateChanged.connect(self.set_imaging_calibration)
        self.checkbox_adjust_amplitudes.stateChanged.connect(self.set_amplitude_feedback)
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

    def set_date(self, date):
        self.date = date.toString(date_format_string)       
        self.set_script_cb()
    
    def set_imaging_calibration(self):
        self.imaging_calibration = self.checkbox_imaging_calibration.isChecked()
        try:
            self.worker.imaging_calibration = self.imaging_calibration
        except Exception as e:
            print(e, "trying to turn on imaging calibration")
    
    def set_amplitude_feedback(self):
        self.amplitude_feedback = self.checkbox_adjust_amplitudes.isChecked()
        return
            
    def set_script_cb(self):
        try:
            folders = af.get_immediate_child_directories(af.get_date_data_path(self.date))        
            folders = [af.get_folder_base(i) for i in folders]
            self.cb_script.clear()
            self.cb_script.addItems(folders)
        except FileNotFoundError:
            print("Selected bad date!")
        
    def set_script_folder(self, i):
        self.script_folder = self.cb_script.currentText()
        self.label_folder_name.setText(f"{analysis_folder_string}: {self.script_folder}")
        self.holding_folder = af.get_holding_folder(self.script_folder, data_date=self.date)
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
        min_value = self.corr_threshold_min.value()/100
        self.corr_min_value.setText(f"{min_value:.2f}")

        max_value = self.corr_threshold_max.value()/100
        self.corr_max_value.setText(f"{max_value:.2f}")

        try:
            #if "pc" in self.folder_to_plot and "time" not in self.folder_to_plot:
            self.make_correlation_plot()
        except AttributeError:
            print("Have not selected a folder yet.")
    
    
    def set_probe_threshold(self):
        try:
            current_folder = current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
            physics_probes = np.load(current_folder + "/fzx_probe.npy", allow_pickle = True)
            physics_means = np.array(
                    [self.__mean_probe_value__(i) for i in physics_probes]
                    )
            self.probe_threshold_value = self.probe_threshold.value()/100 * np.max(physics_means)
            self.probe_threshold_value_label.setText(f"{self.probe_threshold_value:.4f}")
            self.make_plots(self.folder_to_plot)
        except AttributeError:
            print("Can't set probe threshold, because no folder to plot set yet...")
        except Exception as e:
            print(e)
            ### TODO: Error Handling
        
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
            self.worker = FileSorter(self.script_folder, self.date, self.imaging_calibration) # Any other args, kwargs are passed to the run function
            self.worker.signal_output.folder_output_signal.connect(self.make_plots)
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
            
    def save_figure(self, fig, title, current_folder, extra_directory = "", extra_title = ""):
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
        if extra_title: title_string += f" | {extra_title}"
        fig.suptitle(title_string)
        fig.savefig(f"{save_folder}{self.folder_to_plot}_{title}.png", dpi = 200)

        
    def save_array(self, data, title, current_folder, extra_directory = ""):
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
        self.make_1d_plot()
        self.make_2d_plot()
        if self.amplitude_feedback and 'reps' in folder_to_plot:
            self.adjust_amplitude_compensation()
        self.make_probe_plot()
        if "pc" in self.folder_to_plot and 'time' not in self.folder_to_plot:
            self.canvas_corr.setFixedHeight(600)
            self.make_correlation_plot()
        elif "IntDuration" in self.folder_to_plot or "OG_Duration" in self.folder_to_plot:
            self.canvas_corr.setFixedHeight(600)
            self.make_phase_plot()
            self.make_magnetization_plot()
        else:
            self.canvas_corr.setFixedHeight(20)

        self.set_data_cb()
        index = self.cb_data.findText(self.folder_to_plot)
        self.cb_data.setCurrentIndex(index)
        
    def select_probe_threshold(self, fits, xlabels, physics_probes):
        if self.checkbox_probe_threhold.isChecked():
            mean_probe = np.array([self.__mean_probe_value__(i) for i in physics_probes])
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
        if len(fit_sum) == 6:
            trap_values = np.mean(fit_sum, axis = 0)
            compensation = trap_values[::-1] ** (-1/2)
            new_compensation = current_compensation * compensation
            new_compensation = new_compensation/np.linalg.norm(new_compensation)
            np.save(compensation_path, new_compensation)
            print("Engaging new scan")
            self.rm_client.engage()
        return
        
    def make_probe_plot(self):
        current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        sf, units = unitsDef(xlabel)
        xlabels = np.load(current_folder + "/xlabels.npy")        
        keys_adjusted = sf * np.array(xlabels)
        bare_probes = np.load(current_folder + "/bare_probe.npy", allow_pickle = True)
        physics_probes = np.load(current_folder + "/fzx_probe.npy", allow_pickle = True)
        
        bare_probes = np.array(bare_probes)
        self.figure_probe.clf()
        
        ### Bare probe scan
        sorting_order = np.argsort(xlabels)
        bare_probes = bare_probes[sorting_order]
        
        ax = self.figure_probe.add_subplot(1, 2, 1)
        try:
            extent = [-4, 4, np.max(keys_adjusted) + np.diff(keys_adjusted[sorting_order])[0], np.min(keys_adjusted)]        
            cax = ax.imshow(bare_probes, aspect="auto", extent = extent)
            ax.set_ylabel(f"{xlabel} ({units})")
            ax.set_ylabel(f"Frequency (MHz)")
            self.figure_probe.colorbar(cax, ax = ax)
        except TypeError:
            print(bare_probes.shape)
            """
            TODO: Add in error handling
            """
        except IndexError:
            print(bare_probes.shape)
        ### Physics probe
        physics_means = np.array(
                [self.__mean_probe_value__(i) for i in physics_probes]
                )
        ax = self.figure_probe.add_subplot(1, 2, 2)
        transparent_edge_plot(ax, keys_adjusted[sorting_order], physics_means[sorting_order])
        ax.axhline(float(self.probe_threshold_value), ls ='--',  c = "r", label = "Probe Threshold")
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
            /(mean[:, roi_labels.index("roi11"), :] 
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
        physics_probes = np.load(current_folder + "/fzx_probe.npy", allow_pickle = True)
        fits, xlabels = self.select_probe_threshold(fits, xlabels, physics_probes)
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
        extent = [-0.5, fit_mean.shape[2]+0.5, 
          np.max(keys_adjusted) + np.diff(keys_adjusted)[0], 
          np.min(keys_adjusted)]
        cax = ax.imshow(pol, aspect = "auto", cmap = magnetization_colormap, extent = extent, vmin = -1, vmax = 1)
        self.figure_corr.colorbar(cax, ax= ax)
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
        physics_probes = np.load(current_folder + "/fzx_probe.npy", allow_pickle = True)
        fits, xlabels = self.select_probe_threshold(fits, xlabels, physics_probes)
        if len(xlabels) < 2:
            return
        roi_labels = np.load(current_folder + "/roi_labels.npy")
        fit_mean, fit_std, xlabels = self.group_shot(fits, xlabels)
        fit_mean, fit_std = np.swapaxes(fit_mean, 0, 1), np.swapaxes(fit_std, 0, 1)

        self.figure_2d.clf()
        rois_to_plot = [i for i in range(len(roi_labels)) if roi_labels[i] not in self.rois_to_exclude]
        n_rows, n_columns = 2, math.ceil(len(rois_to_plot)/2)
        keys_adjusted = np.array(xlabels) * sf
        if len(keys_adjusted) < 2:
            return
        extent = [-0.5, fit_mean.shape[2]+0.5, 
                  np.max(keys_adjusted) + np.diff(keys_adjusted)[0], 
                  np.min(keys_adjusted)]
        for e, i in enumerate(rois_to_plot):
            label = roi_labels[i]
            ax = self.figure_2d.add_subplot(n_rows, n_columns, e + 1)
            cax = ax.imshow(fit_mean[i], aspect = "auto", cmap = cm.blues, extent = extent, vmin = 0)
            self.save_array(fit_mean[i], label, current_folder)
            self.figure_2d.colorbar(cax, ax = ax, label = "Fitted counts")
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
        physics_probes = np.load(current_folder + "/fzx_probe.npy", allow_pickle = True)
        fits, xlabels = self.select_probe_threshold(fits, xlabels, physics_probes)
        roi_labels = np.load(current_folder + "/roi_labels.npy")
        fit_mean, fit_std, xlabels = self.group_shot(fits, xlabels)
        fit_mean, fit_std = np.swapaxes(fit_mean, 0, 1), np.swapaxes(fit_std, 0, 1)
        keys_adjusted = np.array(xlabels) * sf

        self.axis_1d.clear()
        for state, state_std, label in zip(fit_mean, fit_std, roi_labels):
            if label not in self.rois_to_exclude:
                transparent_edge_plot(self.axis_1d,
                                      keys_adjusted, 
                                      np.mean(state, axis = 1), 
                                      np.mean(state_std, axis = 1),
                                      label = fancy_titles[label])
                self.save_array(np.mean(state, axis = 1), f"{label}_1d", current_folder)
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
        physics_probes = np.load(current_folder + "/fzx_probe.npy", allow_pickle = True)
        fits, xlabels = self.select_probe_threshold(fits, xlabels, physics_probes)
        roi_labels = list(np.load(current_folder + "/roi_labels.npy"))
        fit_10 = fits[:, roi_labels.index('roi10')]
        fit_1m1 = fits[:, roi_labels.index('roi1-1')]
        fit_1p1 = fits[:, roi_labels.index('roi11')]
        fit_sum = fit_10 + fit_1m1 + fit_1p1
        fit_1m1, fit_1p1 = fit_1m1/fit_sum, fit_1p1/fit_sum
        threshold_low = self.corr_threshold_min.value()/100
        threshold_high = self.corr_threshold_max.value()/100
        sidemode = np.mean(fit_1m1 + fit_1p1, axis = 1)
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
                aspect = "auto", 
                interpolation = "None", 
                vmin = -1, 
                vmax = 1, 
                cmap = correlation_colormap)
        self.axis_corr[0].set_xlabel("1, -1 trap index")
        self.axis_corr[0].set_ylabel("1, 1 trap index")
        self.axis_corr[0].set_title("Total")
        
        if len(fit_1m1) > 8:
            adjacent_corr = np.mean(np.array(
                [np.corrcoef(fit_1m1.T[:, i:i + 2], fit_1p1.T[:, i:i + 2])[:n_traps, n_traps:]
                             for i in range(len(fit_1m1) - 2)]), axis = 0)
            cax = self.axis_corr[1].imshow(adjacent_corr, aspect = "auto", interpolation = "None", vmin = -1, vmax = 1, cmap = correlation_colormap)
            self.axis_corr[1].set_xlabel("1, -1 trap index")
            self.axis_corr[1].set_ylabel("1, 1 trap index")
            self.axis_corr[1].set_title("Adjacent")
            self.save_array(adjacent_corr, "corr_adjacent", current_folder)

        try:
            if not self.corr_cb:
                self.corr_cb = self.figure_corr.colorbar(cax, ax = self.axis_corr[1])
        except:
            print("No colorbar")
            self.corr_cb = self.figure_corr.colorbar(cax, ax = self.axis_corr[1])

        self.save_array(corr, "corr_total", current_folder)
        self.save_figure(self.figure_corr, "2d_correlation", current_folder)
        
        self.figure_phase.clf()
        
        ax_total = self.figure_phase.add_subplot(1, 2, 1)
        positions = list(range(-n_traps + 1, n_traps))
        total_diag = [np.mean(np.diagonal(corr, d)) for d in positions]
        ax_total.plot(positions, total_diag)
        try:
            adjacent_diag = [np.mean(np.diagonal(adjacent_corr, d)) for d in positions]
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
        print("a")
        if xlabel != "PR_IntDuration" and xlabel != "OG_Duration":
            return
        print("b")
        globals_list = np.load(current_folder + "/globals.npy", 
                               allow_pickle = True)
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
        if len(fits_x)!= len(fits_y):
            print(f"Incompatible x/y lengths {len(fits_x)} and {len(fits_y)}")
            return
        keys_adjusted = np.array(xlabels) * sf
        x_pol = self.__get_magnetization__(fits_x, roi_labels)
        y_pol = self.__get_magnetization__(fits_y, roi_labels)
        n_traps = fits_x.shape[2]
        if len(keys_adjusted) < 2:
            return
        print('d')
        extent = [-0.5, n_traps + 0.5, np.max(keys_adjusted) + np.diff(keys_adjusted)[0],
                  np.min(keys_adjusted)]
        c_num = x_pol + 1j * y_pol
        if len(c_num) == 1:
            c_num = np.array([c_num])
        phase = np.angle(c_num)
        contrast = np.abs(c_num)
        nrows, ncolumns = 2, 2

        self.figure_phase.clf()
        
        ax_phase = self.figure_phase.add_subplot(nrows, ncolumns, 1)
        ax_phase.set_title("Phase")
        ax_phase.set_ylabel(f"{xlabel} ({units})")
        cax = ax_phase.imshow(
            phase, cmap = phase_colormap, aspect = "auto", extent = extent,
            vmin = -np.pi, vmax = np.pi)  
        self.figure_phase.colorbar(cax)
        
        ax_contrast = self.figure_phase.add_subplot(nrows, ncolumns, 2)
        
        ax_contrast.set_title("Contrast")
        ax_contrast.set_ylabel(f"{xlabel} ({units})")
        cax = ax_contrast.imshow(contrast, cmap = contrast_colormap, aspect = "auto", extent = extent,
                                 vmin = 0, vmax = 1)
        self.figure_phase.colorbar(cax, ax = ax_contrast)
    
        ax_x = self.figure_phase.add_subplot(nrows, ncolumns, 3)
        ax_x.set_title("Phase = 0")
        ax_x.set_ylabel(f"{xlabel} ({units})")
        cax = ax_x.imshow(x_pol, cmap = magnetization_colormap, aspect = "auto", extent = extent,
                          vmin = -1, vmax = 1)
        self.figure_phase.colorbar(cax, ax = ax_x)
    
        ax_y = self.figure_phase.add_subplot(nrows, ncolumns, 4)
        ax_y.set_title("Phase = 90")
        ax_y.set_ylabel(f"{xlabel} ({units})")
        cax = ax_y.imshow(y_pol, cmap = magnetization_colormap, aspect = "auto", extent = extent,
                          vmin = -1, vmax = 1)
        self.figure_phase.colorbar(cax, ax = ax_y)
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
        means = np.array([np.mean(fit_dict[label], axis = 0) for label in labels])
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
        means = np.array([np.mean(fit_dict[label], axis = 0) for label in labels])
        stds = np.array([np.std(fit_dict[label], axis = 0) for label in labels])
        return means, stds, labels
    

        
if __name__ == "__main__":
    app = QApplication([])
    window = AnalysisGUI(app)
    window.show()
    sys.exit(app.exec_())
