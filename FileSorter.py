from shutil import move
import readFiles as rf
from os.path import basename, normpath, getsize
from glob import glob
from PyQt5.QtCore import *
import AnalysisFunctions as af
import sys, traceback, json, os
import numpy as np
from TweezerAnalysis import *
from FormattingStrings import *

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
