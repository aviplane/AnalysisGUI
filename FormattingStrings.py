analysis_folder_string = "Folder to analyze"
analysis_label_string = "Iteration variable"
date_format_string = "yyyy-MM-dd"
no_xlabel_string = "shotnum"
b_field_check_string = "b_field_check"
b_field_check_imaging_string = "b_field_imaging_check"
b_field_check_cleaning_string = "b_field_cleaning_check"
fancy_titles = {"roi1-1": "1, -1 Atoms",
                "roiSum": "Total Atoms",
                "roi2orOther": "F = 2 Atoms",
                "roiRemaining": "Leftover Atoms",
                "roi10": "1, 0 Atoms",
                "roi11": "1, 1 Atoms",
                "roi1": "F = 1 Atoms",
                "roi2": "F = 2 Atoms",
                "roiAll": "All atoms",
                'roiReference': "Reference Region"}


compensation_folder = "S:\\Schleier Lab Dropbox\\Cavity Lab Data\\Cavity Lab Scripts\\cavity_labscriptlib\\RbCavity"


def compensation_path(n_traps):
    return f"{compensation_folder}\\amplitude_compensation_{n_traps}.npy"


shifted_resonance_string = "PR_ShiftedResonance"
agilent_offset_string = "PR_DLProbe_AOM_FlipFlopFreq_Offset"
agilent_physics_string = "PR_ShiftedResonance"
microwave_check_shot_descriptor = 'CheckPiTime'

COUNT_TO_ATOM = 184
