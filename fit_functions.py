# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 18:45:57 2017

@author: Stanford University
"""

# from __future__ import division
# from lyse import *
#from numpy import *
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import numexpr as ne
pi = np.pi
exp = np.exp
sin = np.sin
linspace = np.linspace
amin = np.amin
amax = np.amax
diag = np.diag
sqrt = np.sqrt


def quadratic(frac1m1, a, b, c):
    return a*frac1m1**2 + b*frac1m1 + c

def flip_flop(t, A, tau, f, phi, slope, offset):
    return A * exp(-t/tau) * sin(2*pi*f*t + phi) + slope*t + offset

def linear_fit(t,A,B):
    return A*t+B

def quad_fit(t,A,B,C):
    return A*t**2+B*t+C

def rabi_osc(t, A,t0, f, tau, offset, offset0):
 #   return A * (sin(2*pi*f*(t-t0)/2))**2 * exp(-(t-t0)/tau) + offset
    #return A/2 * (1-cos(2*pi*f*(t-t0))) * exp(-(t-t0)/tau) + offset
    return A * exp(-(t-t0)/tau)* ((sin(2*pi*f*(t-t0)/2))**2 +offset0) + offset

def rabi_osc_spin1(t, A, f, tau, offset, offset0, phi0):
    t0 = 0
    F = 1   # Total spin

    return A * exp(-(t-t0)/tau)* ((sin(2*pi*f*(t-t0)/2-phi0))**(4*F) +offset0) + offset

def rabi_osc_Fp1(t, A, f, offset, phi0):
    t0=0
    fp1 = .25*A*(np.cos(2*pi*f*(t-t0) + phi0) + 1)**2 + offset
    return fp1

def rabi_osc_F0(t, A, f, offset, phi0):
    t0=0
    f0 = .25*A*(np.cos(4*pi*f*(t-t0)+phi0)+1) + offset
    return f0

def rabi_osc_Fm1(t, A, f, offset, phi0):
    t0=0
    fm1 = .25*A*(np.cos(2*pi*f*(t-t0) + phi0))**2 + offset
    return fm1

def rabi_osc_no_decay(t,t0, A, f, offset):
#    t0 = 0
 #   return A * (sin(2*pi*f*(t-t0)/2))**2 * exp(-(t-t0)/tau) + offset
    #return A/2 * (1-cos(2*pi*f*(t-t0))) * exp(-(t-t0)/tau) + offset
    return A * (sin(2*pi*f*(t-t0)/2))**2 + offset
#might want to add t0 as free parameter

def sin_fit(phi, phi0, A, offset):
    return A/2 * sin(pi*(phi-phi0)/180) + offset


def rabi_spec(d, t, Frabi, C, K, f0):
    #d = f-f_0#detuning
    #t= 200 #0.035 #length of the Rabi pulse
    return K * (Frabi**2/(Frabi**2+(d-f0)**2)) * (sin(pi*sqrt(Frabi**2+(d-f0)**2)*t))**2+C

def tof_temp(t, T, x0):
    #t is in s
    #x0 in meters
    kb=1.38*10**-23
    m=87*1.66*10**-27
    return sqrt(x0**2 + (T*kb/m)*t**2)

def exp_decay(t,A, tau, offset):
    t0=0
#    y0=0
    return A*exp(-(t-t0)/tau) + offset

def exp_decay2(t, A, tau, A2, tau2):
    t0=0
    return A*exp(-(t-t0)/tau) + A2*exp(-(t-t0)/tau2)

def gaussian(x, x0, A, sigma, offset):
    return A*exp( -((x-x0)**2)/(2*sigma**2)) + offset

def lorentzian(x, A, full_width, x0, offset):
    return A/(1+(2*(x-x0)/full_width)**2)+offset

def double_lorentzian(x, A_1, full_width_1, x0_1, A_2, full_width_2, x0_2, offset):
#    offset = 0
    return A_1/(1+(2*(x-x0_1)/full_width_1)**2)+A_2/(1+(2*(x-x0_2)/full_width_2)**2)+offset

def generic_fit(model, xdata, ydata, guesses, hold=False, numpoints=500, meth='lm', bounds=[]):
    if hold:
        coefs = guesses
        covar = 0*guesses
    else:
        coefs, covar = curve_fit(model, xdata, ydata, guesses, method=meth)#, bounds=bounds)
#        print("Fit Parameters\n")
#        print(coefs)
#        print("Fit Errors\n")
#        print(sqrt(diag(covar)))
    #print([[coefs[nn],sqrt(covar[nn,nn])] for nn in range(len(coefs))])
    x_fit = linspace(amin(xdata),amax(xdata),numpoints)
    y_fit = model(x_fit,*coefs)
    return coefs, sqrt(diag(covar)), x_fit, y_fit

def rabi_osc_F1(xdata, ydata, guesses, hold=False, numpoints=100, meth='lm'):
    if hold:
        coefsfp1 = guesses
        covarfp1 = 0*guesses
        coefsf0 = guesses
        covarf0 = 0*guesses
        coefsfm1 = guesses
        covarfm1 = 0*guesses
    else:
        coefsfp1 = guesses
        covarfp1 = 0*guesses
        coefsf0 = guesses
        covarf0 = 0*guesses
#        coefsfp1, covarfp1 = curve_fit(rabi_osc_Fp1, xdata[0], ydata[0], guesses, method=meth)
#        coefsf0, covarf0 = curve_fit(rabi_osc_F0, xdata[1], ydata[1], guesses, method=meth)
        coefsfm1, covarfm1 = curve_fit(rabi_osc_Fm1, xdata[2], ydata[2], guesses, method=meth)

    x_fit_fp1 = np.linspace(np.amin(xdata[0]), np.amax(xdata[0]), numpoints)
    y_fit_fp1 = rabi_osc_Fp1(x_fit_fp1,*coefsfp1)
    x_fit_f0 = np.linspace(np.amin(xdata[1]), np.amax(xdata[1]), numpoints)
    y_fit_f0 = rabi_osc_F0(x_fit_f0,*coefsf0)
    x_fit_fm1 = np.linspace(np.amin(xdata[2]), np.amax(xdata[2]), numpoints)
    y_fit_fm1 = rabi_osc_Fm1(x_fit_fm1,*coefsfm1)
    return (coefsfp1, np.sqrt(np.diag(covarfp1)), x_fit_fp1, y_fit_fp1), (coefsf0, np.sqrt(np.diag(covarf0)), x_fit_f0, y_fit_f0), (coefsfm1, np.sqrt(np.diag(covarfm1)), x_fit_fm1, y_fit_fm1)


def gauss2D(x, amplitude, mux, muy, sigmax, sigmay, rotation, slopex, slopey, offset):
    """
    2D Gaussian

    Parameters:
        amplitude
        mux
        muy
        sigmax
        sigmay
        rotation
        slopex
        slopey
        offset
    """
    assert len(x) == 2
    X = x[0]
    Y = x[1]
    A = (np.cos(rotation)**2)/(2*sigmax**2) + (np.sin(rotation)**2)/(2*sigmay**2)
    B = (np.sin(rotation*2))/(4*sigmay**2) - (np.sin(2*rotation))/(4*sigmax**2)
    C = (np.sin(rotation)**2)/(2*sigmax**2) + (np.cos(rotation)**2)/(2*sigmay**2)
    G = amplitude*np.exp(-((A * (X - mux) ** 2) + (2 * B * (X - mux) * (Y - muy)) + (C * (Y - muy) ** 2))) + slopex * X + slopey * Y + offset
    return G.ravel()
