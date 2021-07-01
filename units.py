# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:20:25 2018

@author: quantumEngineer
"""


def unitsDef(xlabel_name):
    units_dict = {'CK_SpectrumM4X_TestDetuning': 'MHz',
                  'SP_A_uWaveDetuning': 'MHz',
                  'CK_TestuWaveSweepT': 'us',
                  'FI_Ixon_TweezerExpTime': 'us',
                  'FI_ImagCool_AM_Voltage': 'V',
                  'MS_TweezerSweepDuration': 'ms',
                  'MS_SpectrumM4X_SweepDuration': 'ms',
                  'MS_KillSweepDuration': 'us',
                  'MS_KillSweepDetuning': 'kHz',
                  'MS_PhaseImprintDuration': 'us',
                  'MS_MWRabiDuration': 'us',
                  'MS_MWRamseyTime': 'us',
                  'MWR_SPSweepDuration': 'us',
                  'Raman_BigRamanDuration': 'us',
                  'RamanCooling_Duration': 'ms',
                  'Tweezer_StatePrepRamseyDuration': 'us',
                  'OG_Duration': 'us',
                  'PR_RampDuration': 'ms',
                  "PR_RampHoldTime": "ms",
                  "PR_ModulationFreqs": "kHz",
                  "PR_GradientPerSite": "kHz",
                  "PR_PulseFreq": "kHz",
                  "PR_WaitTime": "us",
                  'Raman_SmallRamanDuration': 'us',
                  'Raman_MSDuration': 'us',
                  'Raman_BigRamanRamseyDuration': 'us',
                  'Raman_RamseyTime': 'us',
                  'Raman_RamseyPhaseOffset': 'deg',
                  'Raman_GradientPerSite': 'kHz',
                  'Raman_SmallRamanPiHalf': 'us',
                  'Raman_SmallRamanGap': 'us',
                  'Raman_Offset': 'us',
                  'SP_PiPulseDuration': 'us',
                  'SP_BigRamanPiTime': 'us',
                  'SP_SpinEchoDuration': 'us',
                  'SP_A_uWaveRampDuration': 'ms',
                  "SP_A_RamanRampOffDur": 'ms',
                  'SP_RamseyWaitTime': 'us',
                  'SP_A_RamseyPulseTime': 'us',
                  'Lattice_rampUpEarly': 'ms',
                  "Lattice_TrapRampDownDur": "ms",
                  'Tweezers_Imaging_RampUpDur': 'ms',
                  'Tweezers_ModulationFreq': 'kHz',
                  'Tweezers_LatticeLoadOffset': 'ms',
                  'Tweezers_LatticeRampDownDur': 'ms',
                  "Tweezers_LoopDuration": "us"
                  }
    units_multiplier = {'us': 1e6,
                        'ms': 1e3,
                        'MHz': 1,
                        'kHz': 1e3,
                        'V': 1,
                        'deg': 1}
    if xlabel_name == 'ProbeCenterSweep':
        scale_factor = 1
        units = 'MHz'
    elif xlabel_name == 'Raman_Duration':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'Hold_Time':
        scale_factor = 1e3
        units = 'ms'
    elif xlabel_name == 'D1Repump_ErrorLock' or xlabel_name == 'Lattice_Power_OP' or xlabel_name == 'Lattice_DCOffset':
        scale_factor = 1
        units = 'V'
    elif xlabel_name == 'Raman_Pulse_Freq':
        scale_factor = 1
        units = 'MHz'
    elif xlabel_name == 'PR_IntDuration':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'PR_SpectrumM4X_Duration':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'MS_DDS0_MicrowDetuning':
        scale_factor = 1
        units = 'MHz'
    elif xlabel_name == 'Raman_CenterSweepFreq':
        scale_factor = 1
        units = 'MHz'
    elif xlabel_name == 'MS_DDS0_MicrowPulseLength':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'MS_SpectrumM4X_PulseDuration':
        scale_factor = 1e6
        units = 'us'
#    elif 'SpectrumM4X_SweepDuration' in xlabel_name:
#        scale_factor = 1e6
#        units = 'us'
    elif xlabel_name == 'MS_SpectrumM4X_Detuning':
        scale_factor = 1
        units = 'MHz'
    elif xlabel_name == 'OP_Duration' or xlabel_name == 'OP_Duration2':
        scale_factor = 1e3
        units = 'ms'
    elif xlabel_name == 'PR_Cavity_Int_Duration' or xlabel_name == 'Tweezer_RamseyWaitTime':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'Cavity_Int_Duration' or xlabel_name == 'PR_IntDuration':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'Raman_BigDuration' or xlabel_name == 'Tweezer_Duration':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'Tweezer_StatePrepDuration' or xlabel_name == 'Tweezer_StatePrepTotalDuration':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'Beatnote':
        scale_factor = 1
        units = 'GHz'
    elif xlabel_name == 'RA_BigRaman_Duration' or xlabel_name == 'RA_SmallRaman_Duration' or xlabel_name == 'KP_KillTime':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'PR_ProbeOffset':
        scale_factor = 1
        units = 'MHz'
    elif xlabel_name == 'KP_LatticeDetune':
        scale_factor = 1
        units = 'MHz'
    elif xlabel_name == 'FI_Ixon_ExpTime':
        scale_factor = 1e6
        units = 'us'
    elif xlabel_name == 'OP_RepumpBeatnote':
        scale_factor = 1
        units = 'GHz'
    else:
        try:
            scale_factor, units = units_multiplier[units_dict[xlabel_name]
                                                   ], units_dict[xlabel_name]
        except KeyError as e:
            scale_factor = 1
            units = ''
    return scale_factor, units
