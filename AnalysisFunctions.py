from datetime import date, timedelta
import h5py
import sys
import readFiles as rf
import numpy as np
import glob
import os
from scipy import ndimage as ndi
from numpy import *
from units import unitsDef
from os.path import basename, normpath
import pandas as pd
import traceback
#import imageFunctions as imf


state_sf = np.array([1, 1, 1])
data_location = "S:\\Schleier Lab Dropbox\\Cavity Lab Data\\"

bg_fp = "S:\\Schleier Lab Dropbox\\Cavity Lab Data\\2019\\2019-12\\2019-12-18\\2019-12-11-TweezerSpectrumCard\\background_1807"
rawim_bg = np.load(
    f"{data_location}\\Cavity Lab Scripts\\cavity_analysis\\background.npy")
background_components = np.load(
    f"{data_location}\\Cavity Lab Scripts\\cavity_analysis\\background_components_pca.npy")


def get_complete_folder_path(apd, datafolder, data_date=str(date.today())):
    fpd = get_holding_folder(apd, data_date)
    fp = fpd + "\\" + datafolder + "\\"
    return fp


def get_holding_folder(apd, data_date=str(date.today())):
    fpd = data_location + data_date[0:4] + "\\" + \
        data_date[0:7] + f"\\{data_date}\\{apd}"
    return fpd


def get_date_data_path(data_date):
    s = f"{data_location}{data_date[0:4]}\\{data_date[0:7]}\\{data_date}"
    return s


def get_folder_base(path):
    return basename(normpath(path))


def get_immediate_child_directories(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def get_all_folder_paths(apd, data_date=str(date.today())):
    dir_to_look = get_holding_folder(apd, data_date)
    dirs = os.listdir(dir_to_look)
    fps = [dir_to_look + "\\" + i + "\\" for i in dirs]
    return dirs


def get_all_h5_files(script_folder, data_folder, data_date=str(date.today())):
    fp = get_complete_folder_path(script_folder, data_folder, data_date)
    files = glob.glob(fp + "*.h5")
    return files


def extract_rois(filepath):
    with h5py.File(filepath, 'r') as hf:
        rois = hf.get('data/rois').keys()
        data = np.array([hf.get('data/rois/{}'.format(roi)) for roi in rois])
        return list(rois), data


def extract_globals(filepath):
    with h5py.File(filepath, 'r') as hf:
        try:
            variables = dict(hf.get('globals').attrs)
        except AttributeError:
            return {}
    return variables


def get_maximum_pixel_location(i):
    i = np.squeeze(i)
    return np.unravel_index(np.argmax(i), i.shape)


def get_maximum_n_pixel(i, n=50):
    i = np.ravel(i)
    temp = np.partition(-i, n)
    result = np.mean(-temp[:n])
    return result


def save_figure(fig, title, current_folder, extra_directory="", extra_title=""):
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
    current_folder = current_folder.replace("/", "\\")
    save_folder = f"{current_folder}"
    if extra_directory != "":
        save_folder += "\\{extra_directory}"
    folder_to_plot = current_folder.split("\\")[-2]
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    title_string = f"{folder_to_plot}"
    if extra_title:
        title_string += f" | {extra_title}"
    fig.suptitle(title_string)
    print(f"{save_folder}{folder_to_plot}_{title}.png")
    save_location = u'\\\\?\\' + f"{save_folder}{folder_to_plot}_{title}.png"
    fig.savefig(
        save_location, dpi=200)


def save_array(data, title, current_folder, extra_directory=""):
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
    current_folder = current_folder.replace("/", "\\")
    save_folder = f"{current_folder}"
    folder_to_plot = current_folder.split("\\")[-2]
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    try:
        np.savetxt(
            u'\\\\?\\' + f"{save_folder}{folder_to_plot}_{title}.txt", data)
    except OSError:
        traceback.print_exc()
        print("Problem saving")
        # TODO: Error handling


def get_states(f):
    dokp = rf.getxval2(f, 'KP_DoKillPulse')
    dokpuwaves = rf.getxval2(f, 'KP_DoMicrowaves')
    kpnum = rf.getxval2(f, 'KP_Number')
    kpdetunings = rf.getxval2(f, 'MS_SpectrumM4X_KPDetunings')
    transitions = rf.getxval2(f, "MS_KPDetuning")

    if not dokp:
        return ["All"]
    if not dokpuwaves and kpnum == 1:
        return ["F = 1", " F = 2"]
#    if dokpuwaves and kpnum == 1: # EE
#        return ["F = 1", " F = 2"]

    def detuninglabel(detuning):
        if detuning == transitions:
            return "F = 1, 1"
        elif detuning == -1 * transitions:
            return "F = 1, -1"
        elif detuning == 0:
            return "F = 1, 0"
        return "F = 2 or Other"

    imagingOrder = ["Remaining"] + \
        [detuninglabel(i) for i in kpdetunings[:kpnum][::-1]]
    return imagingOrder


def get_magnetization(rois, imagingOrder):
    try:
        index1p1 = imagingOrder.index('F = 1, 1')
    except:
        index1p1 = imagingOrder.index('Remaining')
    try:
        index1m1 = imagingOrder.index('F = 1, -1')
    except:
        index1m1 = imagingOrder.index('Remaining')
#    print(index1p1, index1m1)
    magnetization = rois[index1p1] - rois[index1m1]

    return magnetization


def useroi(item, xpts, ypts):
    print(np.rint(np.ptp(ypts)), int(np.rint(np.ptp(ypts))))
    b = np.zeros((1, int(np.rint(np.ptp(ypts))),
                  int(np.rint(np.ptp(xpts)))))
    a = np.array(
        [item[int(ypts[0]):int(ypts[1]), int(xpts[0]):int(xpts[1])]], dtype='float')
    b[:, :a.shape[1], :a.shape[2]] = a
    return b


def compute_rois_single_image(f, image):
    imagingOrder = get_states(f)
    numStates = len(imagingOrder)

    # PCA Background Subtraction
    # TODO: Mask trap signal
    mask = np.ones((1024, 1024))
    inverse_mask = np.ones((1024, 1024)) - mask
    mask = mask.flatten()
    components = background_components
    masked_components = components * mask[np.newaxis, :]
    inverted_mask_components = components * inverse_mask.flatten()
    orthogonal_masked_components, _ = np.linalg.qr(masked_components.T)
    orthogonal_masked_components = orthogonal_masked_components.T

    # Calculate component strengths
    data_img_flat = image.flatten()
    background_estimate_coeffs = np.einsum(
        'ic,c',
        orthogonal_masked_components,
        data_img_flat
    )
    background_estimate = np.sum(
        background_estimate_coeffs[:, np.newaxis] *
        (orthogonal_masked_components),
        axis=0
    ).reshape(1024, 1024)
    background_estimate_atoms = np.sum(
        background_estimate_coeffs[:, np.newaxis] * (inverted_mask_components),
        axis=0
    ).reshape(1024, 1024)

    # Subtract background
    image = image - background_estimate + background_estimate_atoms
    #image = image - rawim_bg
    # Rotate Image and extract ROI
    rotangle = -45.5
    rotated_image = ndi.rotate(image, rotangle, order=0, reshape=True)
    xshift = -140 * np.sin(rotangle * np.pi / 180)
    yshift = -140 * np.cos(rotangle * np.pi / 180)
    length = 650
    height = 40
    x_start = 180
    y_start = 870
    bottomROIx, bottomROIy = np.array(
        [x_start, x_start + length]), np.array([y_start, y_start + height])
    xpts = np.array([bottomROIx + i * xshift for i in range(numStates)])
    ypts = np.array([bottomROIy + i * yshift for i in range(numStates)])
    rois = [useroi(rotated_image, xpts[i], ypts[i])
            for i in range(numStates)]

    if len(rois) > 3:
        rois[1] *= state_sf[0]
        rois[2] *= state_sf[1]
        rois[3] *= state_sf[2]
        #rois_mag=get_magnetization(rois, imagingOrder)
        #rois=np.insert(rois, -2, rois_mag, axis=0)
        #imagingOrder.insert(-2, "Magnetization")

    rois_sum = np.sum(np.clip(rois[:4], 0.00001, None),  axis=0)
    rois = np.insert(rois, -1, rois_sum, axis=0)
    imagingOrder.insert(-1, "Sum")

    numStates = len(imagingOrder)
    rois = np.squeeze(rois)
    rois_2d = np.sum(rois, axis=1).astype('float')
#    if "Magnetization" in imagingOrder:
#        rois_2d[imagingOrder.index('Magnetization')]  = rois_2d[imagingOrder.index('Magnetization')]/rois_2d[imagingOrder.index('Sum')]

    return rois_2d, imagingOrder


def findxlabel(fnames):
    if len(fnames) == 1:
        return 'iteration'
    with h5py.File(fnames[0], 'r') as hf:
        xlabelList = np.array((hf['globals'].attrs))
    flabels = np.array([rf.getxvals(i, xlabelList) for i in fnames])
    labels = [list(i) for i in flabels.T]
    indices = [i for i, e in enumerate(labels) if len(list(set(e))) > 1]
    try:
        return sorted(list(xlabelList[indices]))
    except Exception as e:
        return []


def get_xlabel(paths):
    global_vals = [extract_globals(path) for path in paths]
    global_df = pd.DataFrame(global_vals)
    xlabels = [i for i in global_df.columns if len(
        global_df[i].astype(str).unique()) > 1]
    return xlabels


def choosexlabel(labels):
    global raman_ramsey, usex
    usex = True
    raman_ramsey, usex = False, True
    if len(labels) == 1:
        raman_ramsey = False
#        if labels[0] == 'SP_RamseyPulsePhase':
#            raman_ramsey = True
#            return 'run number'
        return labels[0]
    elif len(labels) == 2:
        if 'Tweezer_RamseyPhase' in labels:
            raman_ramsey = True
            return labels[(labels.index('Tweezer_RamseyPhase') + 1) % 2]
        if 'SP_RamseyPulsePhase' in labels:
            raman_ramsey = True
            return labels[(labels.index('SP_RamseyPulsePhase') + 1) % 2]
        if 'SP_A_RamseyPulsePhase' in labels:
            raman_ramsey = True
            return labels[(labels.index('SP_A_RamseyPulsePhase') + 1) % 2]
        if 'iteration' in labels:
            return labels[(labels.index('iteration') + 1) % 2]
        if 'waitMonitor' in labels:
            return labels[(labels.index('waitMonitor') + 1) % 2]
        return labels[1]
    elif len(labels) >= 2:
        if 'Tweezer_RamseyPhase' in labels or 'SP_RamseyPulsePhase' in labels:
            raman_ramsey = True
            a = [i for i in labels if i not in [
                'SP_RamseyPulsePhase', 'SP_RamseyPulseDuration']]
            if not a:
                return 'iteration'
            return a[0]
        if 'SP_A_RamseyPulsePhase' in labels:
            raman_ramsey = True
            a = [i for i in labels if i not in [
                'SP_A_RamseyPulsePhase', 'SP_A_RamseyPulseTime']]
            if not a:
                return 'iteration'
            return a[0]
        if 'SP_SpinEchoPhase' in labels:
            a = [i for i in labels if i not in ['SP_SpinEchoPhase']]
            if not a:
                return 'iteration'
            return a[0]
        return sorted(labels)[2]  # "Not there yet boss"

    else:
        usex = False
        return 'iteration'


def get_expansions(hf, group):
    expansion = hf['globals'][group]['expansion'].attrs
    group_globals = [key for key in expansion.keys() if expansion[key]]
    expansions = [expansion[key] for key in expansion.keys() if expansion[key]]
    return group_globals, expansions


def get_xlabel_single_shot(path):
    with h5py.File(path, 'r') as hf:
        variables = []
        expansions = []
        for group in hf['globals']:
            group_globals, group_expansions = get_expansions(hf, group)
            for g, e in zip(group_globals, group_expansions):
                print(g, e)
                try:
                    if len(eval(hf['globals'][group].attrs[g])) > 1 and (e not in expansions or e == "outer"):
                        variables.append(g)
                        expansions.append(e)
                except Exception as e:
                    pass
        units = [unitsDef(variable) for variable in variables]
        values = [rf.getxval2(path, variable) for variable in variables]
    return variables, values, units
