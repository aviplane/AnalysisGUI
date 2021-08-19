# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:36:30 2020

@author: avikar

Some functions to analyze on a trap by trap level
"""
import numpy as np
from fit_functions import lorentzian
from scipy import optimize
import matplotlib.pyplot as plt


def n_trap_func(position, *params):
    n_traps = (len(params) - 4)

    first_trap = params[0]
    trap_spacing = params[1]
    width = params[2]
    centers = np.arange(
        first_trap,
        first_trap + trap_spacing * n_traps,
        trap_spacing
    )
    offset = params[3]
    amps = params[4:]
    # lorentzian(x, A, full_width, x0, offset):
    gx = np.sum(
        [lorentzian(position, amp, width, center, offset)
         for center, amp in zip(centers, amps)],
        axis=0
    )
    return gx


def make_guess(roi, n_traps):
    first_trap_guess = np.argmax(roi[15:35]) + 15
    trap_spacing_guess = 34.7 * 2 / 2
    amp_guess = np.ones(n_traps) * max(np.max(roi), 1) * 2 / 3
    width_guess = 8.5
    offset_guess = 0
    guess = np.empty((n_traps + 4,), dtype=np.float32)

    guess[0] = first_trap_guess
    guess[1] = trap_spacing_guess
    guess[2] = width_guess
    guess[3] = offset_guess
    guess[4:] = amp_guess
    return tuple(guess)


def make_guess_freqs(roi, tweezer_freqs: np.ndarray):
    try:
        tweezer_spacing = np.abs(np.diff(tweezer_freqs)[0])
    except IndexError:
        tweezer_spacing = 30
    n_traps = len(tweezer_freqs)
    first_trap = np.min(tweezer_freqs)
    first_trap_start = int(20 + 17.35 * (first_trap - 85.5))
    first_trap_stop = int(np.ceil(first_trap_start + 30))
    first_trap_guess = np.argmax(roi[first_trap_start:first_trap_stop]) + first_trap_start
    trap_spacing_guess = 17.35 * tweezer_spacing
    amp_guess = np.ones(n_traps) * max(np.max(roi), 1) * 2 / 3
    width_guess = 8.5
    offset_guess = np.mean(roi[:5]) / n_traps
    guess = np.empty((n_traps + 4,), dtype=np.float32)

    guess[0] = first_trap_guess
    guess[1] = trap_spacing_guess
    guess[2] = width_guess
    guess[3] = offset_guess
    guess[4:] = amp_guess
    return tuple(guess)


def make_bounds(roi, n_traps):
    lower = np.zeros(n_traps + 4)
    lower[0] = 15
    lower[3] = -5000
    lower[2] = 3
    upper = np.ones(n_traps + 4)
    upper[0] = 50
    upper[1] = 80
    upper[2] = 20
    upper[3] = 5000
    upper[4:] = max(np.max(roi), 1)
    return lower, upper


def make_bounds_freqs(roi: np.ndarray, tweezer_freqs: np.ndarray):
    n_traps = len(tweezer_freqs)
    first_trap = np.min(tweezer_freqs)
    first_trap_start = int(15 + 17.35 * (first_trap - 85.5))
    first_trap_stop = first_trap_start + 30
    lower = np.zeros(n_traps + 4)
    lower[0] = first_trap_start
    lower[3] = -5000
    lower[2] = 3
    upper = np.ones(n_traps + 4)
    upper[0] = first_trap_stop + 10
    upper[1] = 80
    upper[2] = 20
    upper[3] = 15000 / n_traps
    upper[4:] = max(np.max(roi), 1)
    return lower, upper


def trap_amplitudes(roi, n_traps, plot=False):
    """
    Given a roi, and a number of traps, get the amplitude of each trap in the
    roi

    Inputs:
        roi - array of signal as a function of position
        n_traps - int number of expected tweezers

    Output:
        amplitudes - array of length n_traps with amplitude of each trap
    """

    guess = make_guess(roi, n_traps)
    bounds = make_bounds(roi, n_traps)
    position = np.arange(len(roi))
    popt, _ = optimize.curve_fit(
        n_trap_func, position, roi, p0=guess, bounds=bounds, xtol=.1, ftol=.01)
    amps = popt[4::]
    if plot:
        fig, ax = plt.subplots()
        ax.plot(roi)
        ax.plot(position, n_trap_func(position, *guess))
        ax.plot(position, n_trap_func(position, *popt))
    return amps


def trap_amplitudes_freqs(roi: np.ndarray, tweezer_freqs: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Given a roi, and the locations of the traps, get the amplitude of each trap in the
    roi

    :param roi: array - roi extracted from ixon image
    :param tweezer_freqs: array - the frequencies, specified in MHz, of the trap positions
    :param plot: bool - whether or not to plot the fit for diagnostic purposes.
    :return: amps: array - amplitudes for each trap extracted from the fit
    """
    guess = make_guess_freqs(roi, tweezer_freqs)
    bounds = make_bounds_freqs(roi, tweezer_freqs)
    position = np.arange(len(roi))
    try:
        popt, _ = optimize.curve_fit(
            n_trap_func, position, roi, p0=guess, bounds=bounds, xtol=.1, ftol=.01)
    except ValueError:
        return np.zeros_like(tweezer_freqs)
    amps = popt[4::]
    if plot:
        fig, ax = plt.subplots()
        ax.plot(roi)
        ax.plot(position, n_trap_func(position, *guess))
        ax.plot(position, n_trap_func(position, *popt))
    return amps
