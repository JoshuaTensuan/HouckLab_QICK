# os.add_dll_directory(os.getcwd() + '\\PythonDrivers')
# os.add_dll_directory(os.getcwd() + '.\..\\')

from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Calib.initialize4Q import *
import time
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mTransmissionFF import CavitySpecFF
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mSingleTone import SingleTone

from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mSpecSliceFF import QubitSpecSliceFF
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mAmplitudeRabiFF import AmplitudeRabiFF
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mAmplitudeRabiFF_noUpdate import AmplitudeRabiFF_N
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mChiShift import ChiShift

from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT1FF import T1FF
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT2R import T2R
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT2EFF import T2EMUX
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mSingleShotProgramFFMUX import SingleShotProgramFFMUX
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT1_SS import T1_SS
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mOptimizeReadoutandPulse_FF import ReadOpt_wSingleShotFF, QubitPulseOpt_wSingleShotFF



soc, soccfg = makeProxy()


Qubit_Parameters = {
    '1': {'Readout': {'Frequency': 6552.897, 'Gain': 3200},
          'Qubit': {'Frequency': 1328.99, 'Gain': 1800, "sigma": 0.9, "flattop_length": None}, #PI/2: 900 ?
          'outerfoldername':"Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ03-CL01-KOH/Q1/"},
    '2': {'Readout': {'Frequency': 6661.02, 'Gain': 4000},
          'Qubit': {'Frequency': 1682.43, 'Gain': 3500, "sigma": 1.6, "flattop_length": None}, #PI/2: 1300
          'outerfoldername':"Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ03-CL01-KOH/Q2/",
          'outerfoldernameT2':"Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ03-CL01-KOH/Q2/"},
    '3': {'Readout': {'Frequency': 6769.47, 'Gain': 3500},
          'Qubit': {'Frequency': 1855.90, 'Gain': 2300, "sigma": 1, "flattop_length": None},#PI/2: 800
          'outerfoldername': "Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ03-CL01-KOH/Q3/",
          'outerfoldernameT2': "Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ03-CL01-KOH/Q3/"},
    '4': {'Readout': {'Frequency': 6853.203, 'Gain': 3000}, # 1280.6 for Qubit sweep
          'Qubit': {'Frequency': 1976.98, 'Gain': 2500, "sigma": 0.05, "flattop_length": None}, # Pi/2 = 1200# sigma 1 flattop 1 pi gain: 5000 qubit frequency: 1280.57
          'outerfoldername': "Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ03-CL01-KOH/Q4/",
          'outerfoldernameT2': "Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ03-CL01-KOH/Q4/"},
    }
# Qubit_Parameters = {
#     '1': {'Readout': {'Frequency': 6803.5, 'Gain': 4000},
#           'Qubit': {'Frequency': 2875.68, 'Gain': 1900, "sigma": 0.08, "flattop_length": None},  # pi: 1900, pi/2: 950,
#           'outerfoldername': "Z:/t1Team/Data/2026-04-01_BFF_cooldown/TAT3DP01-01/RFSOC/T1_T2E/",
#           'outerfoldernameT2': "Z:/t1Team/Data/2026-04-01_BFF_cooldown/TAT3DP01-01/RFSOC/T1_T2E/"},  # readout_time:
#
#     '2': {'Readout': {'Frequency': 6984.54, 'Gain': 3500},
#           'Qubit': {'Frequency': 2616.7, 'Gain': 2450, "sigma": 0.05, "flattop_length": None}, # pi: 6000, pi/2: ,
#           'outerfoldername': "Z:/t1Team/Data/2026-04-01_BFF_cooldown/TAT3DP01-01/RFSOC/T1/",
#           'outerfoldernameT2': "Z:/t1Team/Data/2026-04-01_BFF_cooldown/TAT3DP01-01/RFSOC/T2/"},  # readout_time:
#
#     '4': {'Readout': {'Frequency': 7372.65, 'Gain': 5500},
#           'Qubit': {'Frequency': 2376.82, 'Gain': 2850, "sigma": 0.1, "flattop_length": None},  # pi: 2850, pi/2: 1425,
#           'outerfoldername': "Z:/t1Team/Data/2026-04-01_BFF_cooldown/TAT3DP01-01/RFSOC/T1_T2E/",
#           'outerfoldernameT2': "Z:/t1Team/Data/2026-04-01_BFF_cooldown/TAT3DP01-01/RFSOC/T1_T2E/"}  # readout_time:
# }

T1_qubitsweep = False
T1_T2E_sweep = True

T1T2_params = {"qubit_swept": [2, 3, 4],
               "T1_step_list": [100, 100, 80],
               "T1_expts_list": [100, 100, 100],
               "T1_reps_list": [15, 15, 15],
               "T1_rounds_list": [15, 15, 15],
               "T2E_max_list": [10000, 10000, 8000],
               "T2E_expts_list": [50, 50, 50],
               "T2E_reps_list": [20, 20, 20],
               "T2E_rounds_list": [20, 20, 20],
               "pi2_gain_list": [1300, 800, 1200],
               "relax_delay": 9000}

# T1T2_params = {"qubit_swept": [2],
#                "T1_step_list": [100],
#                "T1_expts_list": [100],
#                "T1_reps_list": [15],
#                "T1_rounds_list": [15],
#                "T2E_max_list": [10000],
#                "T2E_expts_list": [70],
#                "T2E_reps_list": [22],
#                "T2E_rounds_list": [22],
#                "pi2_gain_list": [1300],
#                "relax_delay": 9000}

RunAmplitudeRabi = False
Amplitude_Rabi_params = {"reps": 200,
                         'rounds': 5,
                         'relax_delay': 5000,}
repetition_number = 1000


if T1_qubitsweep:
    for rep in range(repetition_number):
        for i in T1T2_params["qubit_swept"]:
            from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Calib.initialize4Q import BaseConfig

            Qubit_Readout = i
            Qubit_Pulse = i
            outerFolder = Qubit_Parameters[str(Qubit_Readout)]['outerfoldername']

            cavity_gain = Qubit_Parameters[str(Qubit_Readout)]['Readout']['Gain']
            resonator_frequency_center = Qubit_Parameters[str(Qubit_Readout)]['Readout']['Frequency']
            qubit_gain = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['Gain']
            qubit_frequency_center = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['Frequency']
            qubit_sigma = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['sigma']
            qubit_flattop = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['flattop_length']

            trans_config = {
                "reps": 1000,  # this will used for all experiements below unless otherwise changed in between trials
                "pulse_style": "const",  # --Fixed
                "readout_length": 15,  # [us]
                "pulse_gain": cavity_gain,  # [DAC units]
                "pulse_freq": resonator_frequency_center,  # [MHz] actual frequency is this number + "cavity_LO"
                "TransSpan": 1.5,  ### MHz, span will be center+/- this parameter
                "TransNumPoints": 101,  ### number of points in the transmission frequecny
                "cav_relax_delay": 30
            }

            config = BaseConfig | trans_config  ### note that UpdateConfig will overwrite elements in BaseConfig
            config["FF_Qubits"] = FF_Qubits

            cavity_min = True
            config["cavity_min"] = cavity_min  # look for dip, not peak

            number_of_steps = 3
            ARabi_config = {'gain_start': 0, "gain_end": qubit_gain,
                            'gainNumPoints': number_of_steps,
                            "reps": Amplitude_Rabi_params['reps'],
                            "rounds": Amplitude_Rabi_params['rounds'],
                            "Qubit_number": Qubit_Readout,
                            "sigma": qubit_sigma, "f_ge": qubit_frequency_center,
                            "relax_delay": 5000,
                            "flattop_length": qubit_flattop}
            config = config | ARabi_config  ### note that UpdateConfig will overwrite elements in BaseConfig
            iAmpRabi = AmplitudeRabiFF_N(path="AmplitudeRabi", cfg=config, soc=soc, soccfg=soccfg,
                                         outerFolder=outerFolder)
            dAmpRabi = AmplitudeRabiFF_N.acquire(iAmpRabi)
            rotation_angle, min_max = AmplitudeRabiFF_N.display(iAmpRabi, dAmpRabi, plotDisp=False, figNum=2)
            AmplitudeRabiFF_N.save_data(iAmpRabi, dAmpRabi)
            AmplitudeRabiFF_N.save_config(iAmpRabi)
            config["rotation_angle"] = rotation_angle
            config["min_max"] = min_max

            if T1T2_params["T1_step_list"] != None:
                T1step = T1T2_params["T1_step_list"][i-1]
            else:
                T1step = T1T2_params["T1_step"]

            if T1T2_params["T1_expts_list"] != None:
                T1expts = T1T2_params["T1_expts_list"][i-1]
            else:
                T1expts = T1T2_params["T1_expts"]

            if T1T2_params["T1_reps_list"] != None:
                T1reps = T1T2_params["T1_reps_list"][i-1]
            else:
                T1reps = T1T2_params["T1_reps"]

            if T1T2_params["T1_rounds_list"] != None:
                T1rounds = T1T2_params["T1_rounds_list"][i-1]
            else:
                T1rounds = T1T2_params["T1_rounds"]

            expt_cfg = {"start": 0,
                        "step": T1step,
                        "expts": T1expts,
                        "reps": T1reps,
                        "rounds": T1rounds, "pi_gain": qubit_gain,
                        "relax_delay": T1T2_params["relax_delay"],
                        "f_ge": qubit_frequency_center,
                        "Qubit_number": Qubit_Readout,
                        "sigma": qubit_sigma,
                        "flattop_length": qubit_flattop
                        }
            config = config | expt_cfg  ### note that UpdateConfig will overwrite elements in BaseConfig
            iT1 = T1FF(path="T1", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder)
            dT1 = T1FF.acquire(iT1)
            T1FF.display(iT1, dT1, plotDisp=False, figNum=2)
            T1FF.save_data(iT1, dT1)
            T1FF.save_config(iT1)

            time.sleep(10)
            soc.reset_gens()

if T1_T2E_sweep:
    for rep in range(repetition_number):
        for idx, i in enumerate(T1T2_params["qubit_swept"]):
            from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Calib.initialize4Q import BaseConfig

            Qubit_Readout = i
            Qubit_Pulse = i
            outerFolder = Qubit_Parameters[str(Qubit_Readout)]['outerfoldername']
            outerFolderT2 = Qubit_Parameters[str(Qubit_Readout)]['outerfoldernameT2']

            cavity_gain = Qubit_Parameters[str(Qubit_Readout)]['Readout']['Gain']
            resonator_frequency_center = Qubit_Parameters[str(Qubit_Readout)]['Readout']['Frequency']
            qubit_gain = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['Gain']
            qubit_frequency_center = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['Frequency']
            qubit_sigma = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['sigma']
            qubit_flattop = Qubit_Parameters[str(Qubit_Pulse)]['Qubit']['flattop_length']

            trans_config = {
                "reps": 1000,  # this will used for all experiements below unless otherwise changed in between trials
                "pulse_style": "const",  # --Fixed
                "readout_length": 15,  # [us]
                "pulse_gain": cavity_gain,  # [DAC units]
                "pulse_freq": resonator_frequency_center,  # [MHz] actual frequency is this number + "cavity_LO"
                "TransSpan": 1.5,  ### MHz, span will be center+/- this parameter
                "TransNumPoints": 101,  ### number of points in the transmission frequecny
                "cav_relax_delay": 30
            }

            config = BaseConfig | trans_config
            config["FF_Qubits"] = FF_Qubits
            config["cavity_min"] = True

            ARabi_config = {
                'gain_start': 0,
                "gain_end": qubit_gain,
                'gainNumPoints': 3,
                "reps": Amplitude_Rabi_params['reps'],
                "rounds": Amplitude_Rabi_params['rounds'],
                "Qubit_number": Qubit_Readout,
                "sigma": qubit_sigma,
                "f_ge": qubit_frequency_center,
                "relax_delay": 3000,
                "flattop_length": qubit_flattop
            }
            config = config | ARabi_config

            iAmpRabi = AmplitudeRabiFF_N(path="AmplitudeRabi", cfg=config, soc=soc, soccfg=soccfg,
                                         outerFolder=outerFolder)
            dAmpRabi = AmplitudeRabiFF_N.acquire(iAmpRabi)
            rotation_angle, min_max = AmplitudeRabiFF_N.display(iAmpRabi, dAmpRabi, plotDisp=False, figNum=2)
            AmplitudeRabiFF_N.save_data(iAmpRabi, dAmpRabi)
            AmplitudeRabiFF_N.save_config(iAmpRabi)
            config["rotation_angle"] = rotation_angle
            config["min_max"] = min_max

            T1step = T1T2_params["T1_step_list"][idx]
            T1expts = T1T2_params["T1_expts_list"][idx]
            T1reps = T1T2_params["T1_reps_list"][idx]
            T1rounds = T1T2_params["T1_rounds_list"][idx]

            expt_cfg = {
                "start": 0,
                "step": T1step,
                "expts": T1expts,
                "reps": T1reps,
                "rounds": T1rounds,
                "pi_gain": qubit_gain,
                "relax_delay": T1T2_params["relax_delay"],
                "f_ge": qubit_frequency_center,
                "Qubit_number": Qubit_Readout,
                "sigma": qubit_sigma,
                "flattop_length": qubit_flattop
            }
            config = config | expt_cfg

            iT1 = T1FF(path="T1", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder)
            dT1 = T1FF.acquire(iT1)
            T1FF.display(iT1, dT1, plotDisp=False, figNum=2)
            T1FF.save_data(iT1, dT1)
            T1FF.save_config(iT1)

            time.sleep(10)
            soc.reset_gens()

            T2Emax = T1T2_params["T2E_max_list"][idx]
            T2Eexpts = T1T2_params["T2E_expts_list"][idx]
            T2Ereps = T1T2_params["T2E_reps_list"][idx]
            T2Erounds = T1T2_params["T2E_rounds_list"][idx]
            qubit_gain_pi2 = T1T2_params["pi2_gain_list"][idx]

            num_pulses = 1
            int_steps = T2Emax // (0.00232515 * (num_pulses + 1) * T2Eexpts)

            T2E_cfg = {
                "start": 0,
                "step": 0.00232515 * (num_pulses + 1) * int_steps,
                "expts": T2Eexpts,
                "reps": T2Ereps,
                "rounds": T2Erounds,
                "pi_gain": qubit_gain,
                "pi2_gain": qubit_gain_pi2,
                "relax_delay": T1T2_params["relax_delay"],
                "f_ge": qubit_frequency_center,
                "num_pi_pulses": num_pulses,
                "sigma": qubit_sigma,
                "flattop_length": qubit_flattop,
                "Qubit_number": Qubit_Readout
            }

            if int_steps == 0:
                print('Step size is 0! need to increase total time or decrease experiments')
            else:
                config = config | T2E_cfg
                iT2E = T2EMUX(path="T2E", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolderT2)
                dT2E = T2EMUX.acquire(iT2E)
                T2EMUX.display(iT2E, dT2E, plotDisp=False, figNum=2)
                T2EMUX.save_data(iT2E, dT2E)
                T2EMUX.save_config(iT2E)

            time.sleep(10)
            soc.reset_gens()
