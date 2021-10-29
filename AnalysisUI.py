from PyQt5.QtWidgets import *
from FormattingStrings import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar


class AnalysisUI:
    def create_ui(self, AnalysisGUI):
        self.scroll_area = QScrollArea(AnalysisGUI)
        self.scroll_area.setWidgetResizable(True)

        self.central_widget = QWidget()
        self.grid_layout = QGridLayout(self.central_widget)
        self.scroll_area.setWidget(self.central_widget)

        self.picker_date = QDateEdit(calendarPopup=True)
        self.picker_date.setDisplayFormat(date_format_string)
        self.picker_date.setDate(QDate.currentDate())
        self.cb_script = QComboBox()
        self.cb_data = QComboBox()

        self.checkbox_imaging_calibration = QCheckBox("Imaging Calibration")
        self.checkbox_adjust_amplitudes = QCheckBox(
            "Amplitude Adjustment Feedback")
        self.checkbox_ignore_first_shot = QCheckBox("Ignore First Shot")
        self.checkbox_delete_reps = QCheckBox("Delete Reps")

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
        self.parameters_lineedit = QLineEdit()

        self.index_label = QLabel(f"Index: ")
        self.index_lineedit = QLineEdit()

        self.sites_label = QLabel(f"Sites: ")

        self.rois_selection = []

        n_columns = 12

        self.go_button = QPushButton("Go")

        row_num = 0
        self.grid_layout.addWidget(self.picker_date, row_num, 0, 1, 1)
        self.grid_layout.addWidget(self.cb_script, row_num, 1, 1, 1)
        self.grid_layout.addWidget(self.cb_data, row_num, 2, 1, 1)
        self.grid_layout.addWidget(
            self.label_folder_name,
            row_num, 3, 1, n_columns - 3
        )

        row_num = row_num + 1
        self.option_selector_layout = QHBoxLayout()
        self.option_selector_layout.addWidget(
            self.checkbox_imaging_calibration)
        self.option_selector_layout.addWidget(
            self.checkbox_adjust_amplitudes)
        self.option_selector_layout.addWidget(self.checkbox_ignore_first_shot)
        self.option_selector_layout.addWidget(self.checkbox_delete_reps)
        self.option_selector_layout.addWidget(self.checkbox_probe_threhold)
        self.option_selector_layout.addWidget(self.probe_threshold_label)
        self.option_selector_layout.addWidget(self.probe_threshold)
        self.option_selector_layout.addWidget(self.probe_threshold_value_label)

        self.grid_layout.addLayout(
            self.option_selector_layout, row_num, 0, 1, n_columns - 1)

        row_num = row_num + 1
        self.set_roi_selector(row_num)
        row_num += 1
        # Add in parameter field
        self.parameter_selector_layout = QHBoxLayout()
        self.grid_layout.addWidget(self.parameters_label, row_num, 0, 1, 1)
        self.grid_layout.addWidget(self.parameters_lineedit, row_num, 1, 1, 2)

        row_num += 1
        self.f2_threshold_checkbox = QCheckBox("F = 2 Thresholding")
        self.f2_threshold_input = QLineEdit("")
        self.checkbox_normalize_correlations = QCheckBox("Normalize Correlation")

        self.threshold_layout = QHBoxLayout()
        self.threshold_layout.addWidget(self.f2_threshold_checkbox)
        self.threshold_layout.addWidget(self.f2_threshold_input)
        self.threshold_layout.addWidget(self.index_label)
        self.threshold_layout.addWidget(self.index_lineedit)
        self.threshold_layout.addWidget(self.checkbox_normalize_correlations)
        self.grid_layout.addLayout(
            self.threshold_layout, row_num, 0, 1, self.n_columns)
        row_num = row_num + 1
        self.grid_layout.addWidget(
            self.go_button, row_num, 0, 1, self.n_columns)

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

        self.figure_6, _ = plt.subplots()
        self.canvas_6 = FigureCanvas(self.figure_6)
        self.toolbar_6 = NavigationToolbar(self.canvas_6, self)

        row_num = row_num + 1
        self.grid_layout.addWidget(
            self.toolbar_1d, row_num, 0, 1, self.n_columns / 2)
        self.grid_layout.addWidget(
            self.canvas_1d, row_num + 1, 0, 1, self.n_columns / 2)

        self.grid_layout.addWidget(
            self.toolbar_2d, row_num, self.n_columns / 2, 1, self.n_columns / 2)
        self.grid_layout.addWidget(
            self.canvas_2d, row_num + 1, self.n_columns / 2, 1, self.n_columns / 2)

        row_num = row_num + 2
        self.corr_threshold_layout = QHBoxLayout()
        self.grid_layout.addLayout(
            self.corr_threshold_layout, row_num, 0, 1, self.n_columns)
        self.corr_threshold_layout.addWidget(self.corr_min_label)
        self.corr_threshold_layout.addWidget(self.corr_min_value)
        self.corr_threshold_layout.addWidget(self.corr_threshold_min)
        self.corr_threshold_layout.addWidget(self.corr_max_label)
        self.corr_threshold_layout.addWidget(self.corr_max_value)
        self.corr_threshold_layout.addWidget(self.corr_threshold_max)

        row_num = row_num + 1
        self.grid_layout.addWidget(
            self.toolbar_corr, row_num, 0, 1, self.n_columns / 2)
        self.grid_layout.addWidget(
            self.canvas_corr, row_num + 1, 0, 1, self.n_columns / 2)

        self.grid_layout.addWidget(
            self.toolbar_phase, row_num, self.n_columns / 2, 1, self.n_columns / 2)
        self.grid_layout.addWidget(
            self.canvas_phase, row_num + 1, self.n_columns / 2, 1, self.n_columns / 2)

        row_num = row_num + 2
        self.grid_layout.addWidget(
            self.toolbar_probe, row_num, 0, 1, self.n_columns / 2)
        self.grid_layout.addWidget(
            self.canvas_probe, row_num + 1, 0, 1, self.n_columns / 2)

        self.grid_layout.addWidget(
            self.toolbar_6, row_num, self.n_columns / 2, 1, self.n_columns / 2)
        self.grid_layout.addWidget(
            self.canvas_6, row_num + 1, self.n_columns / 2, 1, self.n_columns / 2)

        self.canvas_1d.setFixedHeight(600)
        self.canvas_corr.setFixedHeight(600)
        self.canvas_probe.setFixedHeight(600)
        self.canvas_2d.setFixedHeight(600)

        row_num += 2
        self.reset_layout = QHBoxLayout()
        self.reset_amplitude_compensation_button = QPushButton(
            "Reset Amplitude Compensation - NOT IMPLEMENTED YET")
        self.reset_layout.addWidget(self.reset_amplitude_compensation_button)
        self.grid_layout.addLayout(
            self.threshold_layout, row_num, 0, 1, self.n_columns)

        AnalysisGUI.setCentralWidget(self.scroll_area)
        QMetaObject.connectSlotsByName(AnalysisGUI)

    def set_roi_selector(self, row_num=2):
        self.n_columns = 12
        self.roi_selector_label = QLabel("ROIs to exclude:")
        self.roi_selector_layout = QHBoxLayout()
        self.grid_layout.addLayout(
            self.roi_selector_layout, row_num, 0, 1, self.n_columns)
        rois = sorted(list(fancy_titles.keys()))
        self.roi_checkboxes = [QCheckBox(roi) for roi in rois]
        self.roi_selector_layout.addWidget(self.roi_selector_label)
        for i in range(len(self.roi_checkboxes)):
            self.roi_selector_layout.addWidget(
                self.roi_checkboxes[i])

    def check_roi_boxes(self):
        self.rois_to_exclude = [i.text()
                                for i in self.roi_checkboxes if i.isChecked()]
        return
