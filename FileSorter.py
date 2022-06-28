from labscript_utils import h5_lock
from shutil import move
import readFiles as rf
from os.path import basename, normpath, getsize
from glob import glob
from PyQt5.QtCore import *
import AnalysisFunctions as af
import sys
import traceback
import json
import os
import numpy as np
from TweezerAnalysis import *
from FormattingStrings import *
from units import unitsDef
from playsound import playsound
import importlib
import h5py
import GmailAlert
import time


class FileSorterSignal(QObject):
    folder_output_signal = pyqtSignal(str)


class AlertSystem():
    def __init__(self):
        self.do_alerts = False
        self.gmail = GmailAlert.GmailAlert()
        self.refresh_time = time.time()
        self.to_send = ['3016054695@tmomail.net',
                        'escooper@stanford.edu',
                        'kunkel@stanford.edu',
                        'avikar@stanford.edu']

        self.refresh_period = 45  # minutes

        self.sent_alert = False
        self.bad_shot_counter = 0
        self.bad_shot_threshold = 4

        self.sent_probe_alert = False
        self.bad_probe_counter = 0
        self.bad_probe_threshold = 1
        self.n_mails = 0
        self.mail_limit = 5

    def check_refresh(self,):
        current_time = time.time()
        if current_time - self.refresh_time > 60 * self.refresh_period:
            self.gmail = GmailAlert.GmailAlert()
            self.refresh_time = current_time
        return

    def good_probe_shot(self):
        self.sent_probe_alert = False
        self.bad_probe_counter = 0

    def good_shot(self):
        self.sent_alert = False
        self.bad_shot_counter = 0

    def bad_shot(self):
        self.bad_shot_counter += 1

    def mot_problem(self, scan):
        self.bad_shot_counter += 1
        if not self.sent_alert and self.bad_shot_counter > self.bad_shot_threshold:
            [self.gmail.send_error_message(email, f"MOT out in scan {scan}")
             for email in self.to_send]
            self.sent_alert = True
        return

    # TODO: change to decorator
    def atom_problem(self, scan):
        self.bad_shot_counter += 1
        if not self.sent_alert and self.bad_shot_counter > self.bad_shot_threshold:
            [self.gmail.send_error_message(email, f"No atoms in scan {scan}, but MOT is good")
             for email in self.to_send]
            self.sent_alert = True
        return

    def probe_problem(self, scan):
        self.bad_probe_counter += 1
        if not self.sent_probe_alert and self.bad_probe_counter > self.bad_probe_threshold and self.n_mails < self.mail_limit:
            [self.gmail.send_error_message(email, f"The probe needs help since scan {scan}, but don't worry you don't have to get up, you can do it remotely")
             for email in self.to_send]
            self.sent_probe_alert = True
            self.n_mails += 1

    def crash_alert(self):
        [self.gmail.send_error_message(email, "I tried my best, but I crashed again \n Your runmanager")
         for email in self.to_send]


class FileSorter(QThread):

    def __init__(self, script_folder, date, imaging_calibration):
        super(FileSorter, self).__init__()

        self.shot_threshold_size = 2.5e6
        self.script_folder = script_folder
        self.date = date
        self.holding_folder = af.get_holding_folder(
            self.script_folder, data_date=self.date)
        self.signal_output = FileSorterSignal()
        self.running = True
        self.folder_to_plot = False
        self.imaging_calibration = imaging_calibration
        self.parameters = []
        self.delete_reps = False
        self.list_index = 0
        self.alert_system = AlertSystem()

        try:
            with open(self.holding_folder + "/folder_dict.json", 'r') as dict_file:
                self.data_folder_dict = json.loads(dict_file.read())
            with open(self.holding_folder + "/xlabel_dict.json", 'r') as dict_file:
                self.xlabel_dict = json.loads(dict_file.read())
        except FileNotFoundError:
            print("No dictionary file yet...")
            self.data_folder_dict = {}
            self.xlabel_dict = {}

    def run(self):
        ref_time = time.time()
        no_folder_limit = 300
        sent_crash_alert = False
        while self.running:
            self.all_files, self.files_to_sort = self.get_unanalyzed_files()
            self.files_to_sort = sorted(self.files_to_sort)
            if len(self.files_to_sort) > 0:
                self.sort_files(self.files_to_sort[:-1])
                self.signal_output.folder_output_signal.emit(
                    self.folder_to_plot)
                ref_time = time.time()
                if sent_crash_alert:
                    sent_crash_alert = False
            else:
                no_folder_duration = time.time() - ref_time
                if no_folder_duration > no_folder_limit and self.alert_system.do_alerts and not sent_crash_alert:
                    self.alert_system.crash_alert()
                    sent_crash_alert = True
                print("No Folders made yet")
            QThread.sleep(6)
        print("Thread ended")

    def stop(self):
        self.running = False
        if self.folder_to_plot:
            self.signal_output.folder_output_signal.emit(self.folder_to_plot)
        self.terminate()

    def get_unanalyzed_files(self):
        # TODO: reload units
        try:
            search_folder = af.get_holding_folder(
                self.script_folder, data_date=self.date)
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
        if self.delete_reps:
            print("Attempting to delete reps...")
        # iterator??
        scans = [self.get_scan_time(file) for file in files]
        for scan, file in zip(scans, files):
            if scan not in self.data_folder_dict.keys():
                self.data_folder_dict[scan], self.xlabel_dict[scan] = self.generate_folder_name(
                    scan)
                with open(self.holding_folder + "/folder_dict.json", 'w') as dict_file:
                    json.dump(self.data_folder_dict, dict_file)
                with open(self.holding_folder + "/xlabel_dict.json", 'w') as dict_file:
                    json.dump(self.xlabel_dict, dict_file)
            if self.xlabel_dict[scan] in self.parameters:
                self.xlabel_dict[scan] = no_xlabel_string
            extra_params = self.parameter_strings(file, self.parameters)
            self.folder_to_plot = f"{scan}_{self.data_folder_dict[scan]}_{extra_params}"
            current_folder = f"{self.holding_folder}/{self.folder_to_plot}/"
            filename = f"{current_folder}/xlabel.txt"
            if not os.path.exists(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as text_file:
                    print(self.xlabel_dict[scan], file=text_file)
            new_location = current_folder + basename(file)
            try:
                self.process_file(file, current_folder)
            except Exception as e:
                print(e)
                traceback.print_exc()
            try:
                move(file, new_location)
            except Exception as e:
                print(e)
                traceback.print_exc()
            self.alert_system.check_refresh()
        print("Done sorting")
        return self.folder_to_plot

    def parameter_strings(self, file, parameters):
        if parameters == [] or parameters == '':
            return ''
        file_globals = af.extract_globals(file)
        factors, units = zip(*[unitsDef(param) for param in parameters])
        parameter_strings = [f"{parameter.split('_')[-1]}{self.format_parameter(file_globals[parameter], factor)}{unit}"
                             for parameter, unit, factor in zip(parameters, units, factors)
                             if parameter in file_globals]

        return '_'.join(parameter_strings)

    def format_parameter(self, v, factor):
        try:
            v = v * factor
            if hasattr(v, '__iter__'):
                value = np.round(v, 4)
                return '-'.join(map(str, value))
            else:
                # not iterable
                return f"{np.round(v, 4)}"
        except:  # probably a string, and we're lazy bc friyay
            return v
        return

    def extract_counts(self, pic_, ref_pic, short_axis=40, long_axis=80,
                       rotation=-5, counts_to_atoms=COUNT_TO_ATOM):
        # define region to adjust background subtraction
        x_1 = 600
        x_2 = 800
        y_1 = 20
        y_2 = 80
        # compensation = np.sum(np.sum(pic_[y_1:y_2, x_1:x_2]))\
        #     / np.sum(np.sum(ref_pic[y_1:y_2, x_1:x_2]))
        compensation = 1 #???

        # Calculate subtracted ref_pic
        pic = pic_ - compensation * ref_pic ###This seems wrong?

        # Extract counts for each trap
        trap_counts = np.zeros(len(af.trap_centers))

        for c_center, val_center in enumerate(af.trap_centers):
            # cut out relevant part of the pic
            trap_region = range(int(val_center - af.trap_width / 2),
                                int(val_center + af.trap_width / 2))
            sel_trap = pic[:, trap_region]

            # find maximum coordinate
            y_max, x_max = np.where(sel_trap == np.max(sel_trap))

            x_max = x_max[0] + int(val_center - af.trap_width / 2)
            y_max = y_max[0]

            x_grid, y_grid = np.meshgrid(
                np.arange(pic.shape[1]), np.arange(pic.shape[0]))

            # # get filter for atoms
            phi = rotation / 180 * np.pi

            ellipse_boundary = ((np.cos(phi) * (x_grid - x_max) + np.sin(phi) * (y_grid - y_max))**2 / short_axis**2
                                + (np.cos(phi) * (y_grid - y_max) - np.sin(phi) * (x_grid - x_max))**2 / long_axis**2)

            trap_filter = ellipse_boundary > 1

            cut_pic = 1 * pic
            cut_pic[trap_filter] = 0

            trap_counts[c_center] = np.sum(np.sum(cut_pic)) / counts_to_atoms

        return trap_counts

    def get_anums_from_ixon(self, file):

        kin_height = 113
        f2_region = range(0, kin_height)
        f1up_region = range(4 * kin_height, 5 * kin_height)
        f1down_region = range(2 * kin_height, 3 * kin_height)
        f1mid_region = range(6 * kin_height, 7 * kin_height)
        remaining_region = range(7 * kin_height, 8 * kin_height)
        ref_region = range(8 * kin_height, 9 * kin_height)

        atom_images = []
        with h5py.File(file) as hfile:
            ixon_image = np.array(
                hfile['images/ixon/ixonatoms'][0], dtype=float)
            ref_image = ixon_image[ref_region]
            atom_images.append(ixon_image[f1down_region])
            atom_images.append(ixon_image[f1mid_region])
            atom_images.append(ixon_image[f1up_region])
            atom_images.append(ixon_image[f2_region])
            atom_images.append(ixon_image[remaining_region])

        n_traps = len(af.trap_centers)
        n_states = 5
        n_atoms = np.zeros((n_states + 1, n_traps))
        for c_state in range(n_states):
            n_atoms[c_state] = self.extract_counts(
                atom_images[c_state], ref_image)

        n_atoms[-1] = np.sum(n_atoms, axis=0)
        state_labels = ['roi1-1', 'roi10', 'roi11',
                        'roi2orOther', 'roiRemaining', 'roiSum']
        return state_labels, n_atoms

    def save_value(self, value, current_folder, file_name, numpy_array=False):
        file_location = current_folder + f"/{file_name}"
        try:
            values = np.load(file_location,
                             allow_pickle=not numpy_array)
            if numpy_array:
                values = np.vstack(values, np.array([value]))
            else:
                values = list(values)
                values.append(value)
            np.save(file_location, values)
        except IOError:
            to_save = np.array([value]) if numpy_array else [value]
            np.save(file_location, to_save)

    def check_MOT(self, file):
        mot = np.squeeze(rf.getdata(file, 'MOTatoms'))
        print("MOT max", np.max(mot))
        if np.max(mot) > 300:
            return True

    def process_file(self, file, current_folder):
        """
        TODO: Add in probe processing
        """
        with open(current_folder + "/xlabel.txt", 'r') as xlabel_file:
            xlabel = xlabel_file.read().strip()
        print(file)

        roi_labels, rois = af.extract_rois(file)
        state_labels, n_states = self.get_anums_from_ixon(file)
        file_globals = af.extract_globals(file)
        tweezer_freqs = file_globals["Tweezers_AOD1_Freqs"]
        if self.imaging_calibration:
            print("No imaging calibration yet.")
        physics_probe, bare_probe, bare_input, alarm = self.get_cavity_transmission(
            file)
        rigol_probe_trace, rigol_probe_time = self.get_rigol_transmission(file)
        probe_lock_monitor, probe_lock_time = self.get_rigol_transmission(
            file, "ProbeLockMonitor")
        probe_lock_signal, _ = self.get_rigol_transmission(
            file, "ProbeLockSignal")
        self.save_value(COUNT_TO_ATOM, current_folder,
                        "count_to_atom_conversion.npy")
        self.save_value(probe_lock_monitor,
                        current_folder,
                        "probe_lock_monitor.npy")
        self.save_value(probe_lock_signal,
                        current_folder,
                        "probe_lock_signal.npy")
        # get scan number out of file
        scan = self.get_scan_time(file)

        good_MOT = self.check_MOT(file)
        good_atoms = np.max(n_states) > 700
        good_probe = np.max(bare_probe) > 0.3
        print(np.max(n_states))
        if self.alert_system.do_alerts:
            if not good_MOT:
                self.alert_system.mot_problem(scan)
            elif not good_atoms:
                if not file_globals['CheckMagneticFieldCleaning']:
                    self.alert_system.atom_problem(scan)
            elif not good_probe:
                self.alert_system.probe_problem(scan)

        if good_probe:
            self.alert_system.good_probe_shot()

        if good_MOT and good_atoms:
            self.alert_system.good_shot()

        if "PairCreation_" in current_folder and alarm:
            print("Probe Out")
            # playsound("beep.mp3")
        try:
            xlabels = np.load(current_folder + "/xlabels.npy")
            globals_list = np.load(current_folder + "/globals.npy",
                                   allow_pickle=True)
            bare_probe_list = list(np.load(current_folder + "/bare_probe.npy",
                                           allow_pickle=True))
            bare_input_list = list(np.load(current_folder + "/bare_input.npy",
                                           allow_pickle=True))
            physics_probe_list = list(np.load(current_folder + "/fzx_probe.npy",
                                              allow_pickle=True))
            all_rois = np.load(current_folder + "/all_rois.npy")
            all_states = np.load(current_folder + "/all_anums.npy")
            rigol_probe_list = list(
                np.load(current_folder + "/rigol_probe.npy", allow_pickle=True))
            try:
                all_rois = np.vstack([all_rois, np.array([rois])])
                all_states = np.vstack([all_states, np.array([n_states])])
            except ValueError:
                print(f"Error with file: {file}")
                traceback.print_exc()
                return
            bare_probe_list.append(bare_probe)
            bare_input_list.append(bare_input)
            physics_probe_list.append(physics_probe)
            rigol_probe_list.append(rigol_probe_trace)
            if xlabel == no_xlabel_string:
                xlabel_value = np.max(xlabels) + 1
            else:
                xlabel_value = self.get_global(file_globals, xlabel)
            xlabels = np.append(xlabels, xlabel_value)
            globals_list = np.append(globals_list, file_globals)
            np.save(current_folder + "/globals.npy", globals_list)
            np.save(current_folder + "/xlabels.npy", xlabels)
            np.save(current_folder + "/all_rois.npy", all_rois)
            np.save(current_folder + "/all_anums.npy", all_states)
            np.save(current_folder + "/fzx_probe.npy", physics_probe_list)
            np.save(current_folder + "/bare_probe.npy", bare_probe_list)
            np.save(current_folder + "/bare_input.npy", bare_input_list)
            np.save(current_folder + "/rigol_probe.npy", rigol_probe_list)
        except IOError:
            if xlabel == no_xlabel_string:
                xlabel_value = 0
            else:
                xlabel_value = self.get_global(file_globals, xlabel)
            globals_list = [file_globals]
            np.save(current_folder + "/globals.npy", globals_list)
            np.save(current_folder + "/state_labels.npy", state_labels)
            np.save(current_folder + "/all_anums.npy", np.array([n_states]))
            np.save(current_folder + "/roi_labels.npy", roi_labels)
            np.save(current_folder + "/all_rois.npy", np.array([rois]))
            np.save(current_folder + "/xlabels.npy", np.array([xlabel_value]))
            np.save(current_folder + "/fzx_probe.npy", [physics_probe])
            np.save(current_folder + "/bare_probe.npy", [bare_probe])
            np.save(current_folder + "/bare_input.npy", [bare_input])
            np.save(current_folder + "/rigol_probe.npy",
                    [rigol_probe_trace])
            print("Creating files...")
        return

    def get_cavity_transmission(self, file):
        try:
            bare_probe = rf.getdata(file, "GreyCavityTransmissionBare")
        except Exception as e:
            bare_probe = [(0, 0), (1, 0)]
            traceback.print_exc()
        try:
            physics_probe = rf.getdata(file, "GreyCavityTransmissionProbe")
        except Exception as e:
            physics_probe = [(0, 0), (1, 0)]
            """
            TODO: Error Handling
            """
        try:
            probe_input = rf.getdata(file, "InputCavityTransmissionBare")
        except Exception as e:
            probe_input = [(0, 0), (1, 0)]

        bare_probe_processed = self.__process_trace__(bare_probe)
        alarm = False
        if np.max(bare_probe_processed) < 4 * np.min(bare_probe_processed):
            alarm = True
        return self.__process_trace__(physics_probe), bare_probe_processed, self.__process_trace__(probe_input), alarm

    def get_rigol_transmission(self, file, h5_string="ProbeRigolTrace"):
        """
        Input:
            file: path to h5 file with relevant rigol transmission
            h5_string: name of Rigol trace

        Output:
            data: numpy array from h5 file (voltage)
            times: numpy array from h5_file (same length as voltages)
        """
        data = rf.getdata(file, h5_string)
        times = rf.getdata(file, f"ScopeTraces/times{h5_string}")
        return data, times

    def __process_trace__(self, trace):
        return [i[1] for i in trace]

    def adjust_rois(self, fits, roi_labels):
        if 'roi11' not in roi_labels:
            return fits
        population_to_q = np.array([[1, -1, 0], [-1, -1, 1], [1, 1, 1]])
        alphas = np.load(
            f"X:\\labscript\\analysis_scripts\\roi_adjustment_alpha.npy")

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
        # return xlabel_value[0][0]
        if hasattr(xlabel_value, '__iter__'):
            return xlabel_value[min(self.list_index, len(xlabel_value) - 1)]
        return xlabel_value

    def process_xlabel(self, xlabel):
        return xlabel.split("_")[-1]

    def get_scan_time(self, file_name):
        scan_time = basename(file_name).split("_")[0]
        return scan_time

    def choose_xlabel(self, labels):
        print(labels)
        l = len(labels)
        if len(labels) == 1:
            return labels[0]
        elif len(labels) > 1:
            if 'PR_WaitTime' in labels:
                return 'PR_WaitTime'
            if 'LocalSpinor_Duration' in labels:
                return 'LocalSpinor_Duration'
            if 'Tweezers_AOD0_LoadAmp' in labels:
                return 'Tweezers_AOD0_LoadAmp'
            if 'Tweezer_RamseyPhase' in labels:
                return labels[(labels.index('Tweezer_RamseyPhase') + 1) % l]
            if 'SR_GlobalLarmor' in labels:
                return labels[(labels.index('SR_GlobalLarmor') + 1) % l]
            if 'Raman_RamseyPhaseOffset' in labels:
                return labels[(labels.index('Raman_RamseyPhaseOffset') + 1) % l]
            if 'Raman_RamseyPhase' in labels:
                return labels[(labels.index('Raman_RamseyPhase') + 1) % l]
            if 'SP_RamseyPulsePhase' in labels:
                return labels[(labels.index('SP_RamseyPulsePhase') + 1) % l]
            if 'SP_A_RamseyPulsePhase' in labels:
                return labels[(labels.index('SP_A_RamseyPulsePhase') + 1) % l]
            if 'Tweezers_Lattice_HoldTime' in labels:
                return labels[(labels.index('Tweezers_Lattice_HoldTime') + 1) % l]
            if 'iteration' in labels:
                return labels[(labels.index('iteration') + 1) % l]
            if 'waitMonitor' in labels:
                return labels[(labels.index('waitMonitor') + 1) % l]
            return labels[1]

    def generate_folder_name(self, scan):
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
            xlabel = self.choose_xlabel(xlabels)
            main_string = self.process_xlabel(xlabel)
        if scan_globals["MeasurePairCreation"]:
            main_string = 'PairCreation_' + main_string
        if scan_globals["Physics_DoSequence"]:
            add_string = scan_globals['Descriptor']
            main_string = 'PairCreationSequence_' + add_string
            print(main_string)
        if scan_globals["MeasureSpinExchange"]:
            main_string = "SpinExchange_" + main_string
        if scan_globals["CheckCavityShift"]:
            main_string = "cavity_shift"
            xlabel = "PR_DLProbe_Agilent_FlipFlopFreq"
        try:
            if scan_globals["CheckMagneticField"]:
                main_string = b_field_check_string
            if scan_globals["CheckMagneticFieldImaging"]:
                main_string = b_field_check_imaging_string
            if scan_globals["CheckMagneticFieldCleaning"]:
                main_string = b_field_check_cleaning_string
        except Exception as e:
            traceback.print_exc()
        main_string = f"{main_string}"

        return main_string, xlabel
