# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:36:30 2020

@author: QuantumEngineer3

Some functions to analyze on a trap by trap level
"""
import numpy as np
from fit_functions import gaussian, lorentzian
from scipy import optimize
from scipy import signal
import matplotlib.pyplot as plt

# x0, A, sigma, offset


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
    #offsets = params[4::2]
    # lorentzian(x, A, full_width, x0, offset):
    gx = np.sum(
        [lorentzian(position, amp, width, center, 0)
         for center, amp in zip(centers, amps)],
        axis=0
    )
    return gx


def make_guess(roi, n_traps):
    first_trap_guess = np.argmax(roi[15:35]) + 15
    trap_spacing_guess = 34.7
    amp_guess = np.ones(n_traps) * max(np.max(roi), 1) * 2 / 3
    width_guess = 8.5
    offset_guess = 0
    guess = np.empty((n_traps + 4,), dtype=np.float32)

    guess[0] = first_trap_guess
    guess[1] = trap_spacing_guess
    guess[2] = width_guess
    guess[3] = offset_guess
    guess[4:] = amp_guess
    #guess[4::2] = offset_guess
    return tuple(guess)


def make_bounds(roi, n_traps):
    lower = np.zeros(n_traps + 4)
    lower[0] = 15
    lower[3] = -5000
    lower[2] = 3
    #lower[4::2] = -1000
    upper = np.ones(n_traps + 4)
    upper[0] = 50
    upper[1] = 80
    upper[2] = 20
    upper[3] = 5000
    upper[4:] = max(np.max(roi), 1)
    #upper[4::2] = 1000
    return (lower, upper)


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
