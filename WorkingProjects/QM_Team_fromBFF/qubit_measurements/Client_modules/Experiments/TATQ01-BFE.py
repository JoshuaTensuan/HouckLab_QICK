# os.add_dll_directory(os.getcwd() + '\\PythonDrivers')
# os.add_dll_directory(os.getcwd() + '.\..\\')
# Select the interactive Qt backend (PyQt6) BEFORE pyplot is first imported
# (which happens inside `from utils import *` below). Required for the
# non-blocking live two-tone display (ModifiedRamsey_params["live_display"]).
import matplotlib

matplotlib.use("QtAgg")
from utils import *
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Calib.initialize4Q import *
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.CoreLib.socProxy import *
import time
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mTransmissionFF import (
    CavitySpecFF,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mSingleTone import (
    SingleTone,
)

from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mSpecSliceFF import (
    QubitSpecSliceFF,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mChargeDispersionQuasiCW import (
    ChargeDispersionQuasiCW,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mAmplitudeRabiFF import (
    AmplitudeRabiFF,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mAmplitudeRabiFF_noUpdate import (
    AmplitudeRabiFF_N,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mChiShift import (
    ChiShift,
)

from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT1FF import (
    T1FF,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT2R import (
    T2R,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT2EFF import (
    T2EMUX,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mSingleShotProgramFFMUX import (
    SingleShotProgramFFMUX,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mT1_SS import (
    T1_SS,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mOptimizeReadoutandPulse_FF import (
    ReadOpt_wSingleShotFF,
    QubitPulseOpt_wSingleShotFF,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mConstantTwoTone import (
    ConstantTwoTone,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mChargeDispersion import (
    ChargeDispersion,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mModifiedRamsey import (
    ModifiedRamsey,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mActiveResetVerify import (
    ActiveResetVerify,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pyvisa
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import matplotlib.dates as mdates
import json
from sklearn.cluster import KMeans
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mAutoCoherence import (
    run_auto_coherence,
    AUTO_COHERENCE_PARAMS,
    find_sweet_spot,
)

# Zero-span charge-parity switching measurement (device-agnostic acquisition +
# offline analysis). pick_parity_drive_freq / chunked_acquire / ramp_to come in
# via `from utils import *` above.
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.mZeroSpanParity import (
    ZeroSpanParity,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.analyze_ZeroSpanParity import (
    analyze_parity_run,
)
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.validate_ZeroSpanParity import (
    run_static_contrast,
    run_contrast_vs_qubit_freq,
    run_modulation_check,
    run_control_suite,
    run_environment_sweep,
    build_evidence_report,
)


def modified_ramsey_timing(soccfg, cfg):
    """Mirror the BFE QICK-v1 schedule and return one repetition's timing.

    This handles both feedback strategies used here: a dedicated start-of-shot
    active reset and the BFE-specific reset that reuses the final Ramsey readout.
    It excludes one-time initialization, instruction latency, and data transfer.
    """
    res_ch = cfg["res_ch"]
    qubit_ch = cfg["qubit_ch"]
    ro_chs = cfg["ro_chs"]
    if not ro_chs:
        raise ValueError("cfg['ro_chs'] must contain at least one readout channel")
    if cfg["df"] <= 0:
        raise ValueError("cfg['df'] must be positive")
    if cfg["sigma"] <= 0:
        raise ValueError("cfg['sigma'] must be positive")
    if cfg["readout_length"] <= 0:
        raise ValueError("cfg['readout_length'] must be positive")

    reset_from_readout = bool(cfg.get("reset_from_ramsey_readout", False))
    use_active_reset = bool(cfg.get("use_active_reset", False)) and not reset_from_readout
    delay_keys = ["adc_trig_offset", "mr_relax_delay"]
    if use_active_reset or reset_from_readout:
        delay_keys.extend(["reset_readout_relax_delay", "post_reset_wait"])
    for delay_key in delay_keys:
        if cfg.get(delay_key, 0.0) < 0:
            raise ValueError(f"cfg['{delay_key}'] must be non-negative")

    required_tone_us = cfg["adc_trig_offset"] + cfg["readout_length"]
    if cfg["length"] < required_tone_us:
        raise ValueError(
            "cfg['length'] must cover cfg['adc_trig_offset'] + "
            f"cfg['readout_length'] ({cfg['length']} us < {required_tone_us} us)"
        )

    f_time = float(soccfg["tprocs"][0]["f_time"])
    res_f_fabric = float(soccfg["gens"][res_ch]["f_fabric"])
    qubit_f_fabric = float(soccfg["gens"][qubit_ch]["f_fabric"])

    requested_tone_cycles = soccfg.us2cycles(cfg["length"], gen_ch=res_ch)
    adc_offset_cycles = soccfg.us2cycles(cfg["adc_trig_offset"])
    window_cycles = {
        int(ch): soccfg.us2cycles(cfg["readout_length"], ro_ch=ch)
        for ch in ro_chs
    }
    adc_end_by_ch = {
        int(ch): (
            adc_offset_cycles
            + window_cycles[int(ch)]
            * f_time / float(soccfg["readouts"][ch]["f_output"])
        )
        for ch in ro_chs
    }
    adc_end = max(adc_end_by_ch.values())
    required_tone_cycles = int(np.ceil(
        adc_end * res_f_fabric / f_time - 1e-12
    ))
    tone_cycles = max(requested_tone_cycles, required_tone_cycles)
    tone_end = tone_cycles * f_time / res_f_fabric
    readout_slot_cycles = int(max(tone_end, adc_end))

    qubit_native_cycles = soccfg.us2cycles(
        cfg["sigma"] * 4, gen_ch=qubit_ch
    )
    qubit_pulse_cycles = int(
        qubit_native_cycles * f_time / qubit_f_fabric
    )
    use_pi_pulse = bool(cfg.get("use_pi_pulse", False))
    ramsey_pulse_count = 3 if use_pi_pulse else 2
    wait_request_us = (
        1.0 / (4.0 * cfg["df"])
        if use_pi_pulse else 1.0 / (2.0 * cfg["df"])
    )
    wait_count = 2 if use_pi_pulse else 1
    wait_cycles = wait_count * soccfg.us2cycles(wait_request_us)
    final_padding_cycles = soccfg.us2cycles(0.05)
    mr_relax_cycles = (
        0 if reset_from_readout
        else soccfg.us2cycles(cfg.get("mr_relax_delay", 0.0))
    )
    reset_delay_cycles = (
        soccfg.us2cycles(cfg.get("reset_readout_relax_delay", 1.0))
        if use_active_reset or reset_from_readout else 0
    )
    post_reset_cycles = (
        soccfg.us2cycles(cfg.get("post_reset_wait", 0.0))
        if use_active_reset or reset_from_readout else 0
    )
    reset_cycles = int(cfg.get("reset_cycles", 1)) if use_active_reset else 0
    if use_active_reset and reset_cycles < 1:
        raise ValueError(
            "cfg['reset_cycles'] must be >= 1 when use_active_reset=True"
        )

    start_reset_block_cycles = (
        reset_cycles
        * (readout_slot_cycles + reset_delay_cycles + qubit_pulse_cycles)
        + (post_reset_cycles if reset_cycles else 0)
    )
    ramsey_core_cycles = (
        ramsey_pulse_count * qubit_pulse_cycles
        + wait_cycles
        + final_padding_cycles
    )
    final_readout_block_cycles = readout_slot_cycles
    if reset_from_readout:
        # The final readout is followed by ringdown, a runtime-conditional pi,
        # and a shared sync. Compile-time scheduling reserves the pi on both paths.
        final_readout_block_cycles += (
            reset_delay_cycles + qubit_pulse_cycles + post_reset_cycles
        )
    else:
        final_readout_block_cycles += mr_relax_cycles

    scheduled_cycles = (
        start_reset_block_cycles + ramsey_core_cycles + final_readout_block_cycles
    )
    to_us = lambda cycles: float(cycles / f_time)
    return {
        "scheduled_rep_period_us": to_us(scheduled_cycles),
        "scheduled_rep_period_tproc_cycles": int(scheduled_cycles),
        "start_reset_block_us": to_us(start_reset_block_cycles),
        "ramsey_core_us": to_us(ramsey_core_cycles),
        "final_readout_block_us": to_us(final_readout_block_cycles),
        "readout_slot_us": to_us(readout_slot_cycles),
        "resonator_tone_us": float(tone_cycles / res_f_fabric),
        "requested_resonator_tone_us": float(
            requested_tone_cycles / res_f_fabric
        ),
        "tone_quantization_extension_cycles": int(
            tone_cycles - requested_tone_cycles
        ),
        "tone_coverage_margin_us": float((tone_end - adc_end) / f_time),
        "qubit_pulse_us": to_us(qubit_pulse_cycles),
        "ramsey_qubit_pulse_count": int(ramsey_pulse_count),
        "ramsey_wait_us": to_us(wait_cycles),
        "final_padding_us": to_us(final_padding_cycles),
        "mr_relax_delay_us": to_us(mr_relax_cycles),
        "use_active_reset": use_active_reset,
        "reset_from_ramsey_readout": reset_from_readout,
        "reset_cycles": int(reset_cycles),
        "reset_readout_relax_delay_us": to_us(reset_delay_cycles),
        "post_reset_wait_us": to_us(post_reset_cycles),
        "f_time_mhz": f_time,
        "res_f_fabric_mhz": res_f_fabric,
        "qubit_f_fabric_mhz": qubit_f_fabric,
        "ro_f_output_mhz": {
            int(ch): float(soccfg["readouts"][ch]["f_output"])
            for ch in ro_chs
        },
    }


def _extract_iq_from_singleshot_data(data_ss, state="g"):
    """
    Extracts SingleShotProgramFFMUX IQ arrays.

    state="g" uses ground-prep IQ.
    state="e" uses excited-prep IQ.
    state="thermal" uses thermal IQ if available.
    """
    d = data_ss["data"]

    key_pairs = {
        "g": [
            ("i_g0", "q_g0"),
            ("i_g", "q_g"),
            ("Ig", "Qg"),
        ],
        "e": [
            ("i_e0", "q_e0"),
            ("i_e", "q_e"),
            ("Ie", "Qe"),
        ],
        "thermal": [
            ("i_thermal0", "q_thermal0"),
            ("i_thermal", "q_thermal"),
        ],
    }

    for i_key, q_key in key_pairs.get(state, []):
        if i_key in d and q_key in d:
            return np.ravel(np.array(d[i_key])), np.ravel(np.array(d[q_key]))

    print("[SingleShot MR Calib] Available data keys:", d.keys())
    raise KeyError(f"Could not find raw I/Q arrays for state={state}.")


def get_apriori_separator_from_singleshot(config, soc, soccfg, outerFolder):
    """
    Runs one SingleShot calibration using a pi pulse.
    Uses the returned ground/excited blobs to define a fixed separator.
    Also saves the SingleShot data/config/png through the normal SingleShot methods.
    """

    ss_shots = ModifiedRamsey_params.get("ss_calib_shots", 1000)

    cfg_ss = config.copy()
    cfg_ss["number_of_pulses"] = 1
    cfg_ss["shots"] = ss_shots
    cfg_ss["reps"] = ss_shots
    cfg_ss["qubit_gain"] = qubit_gain
    cfg_ss["f_ge"] = qubit_frequency_center
    cfg_ss["sigma"] = qubit_sigma
    cfg_ss["flattop_length"] = qubit_flattop
    cfg_ss["Qubit_number"] = Qubit_Readout
    cfg_ss["Read_Indeces"] = Qubit_Readout

    inst_ss = SingleShotProgramFFMUX(
        path="SingleShot_MRCalib",
        outerFolder=outerFolder,
        cfg=cfg_ss,
        soc=soc,
        soccfg=soccfg,
    )

    data_ss = SingleShotProgramFFMUX.acquire(inst_ss)

    # Save SingleShot data, config, and display png.
    SingleShotProgramFFMUX.display(inst_ss, data_ss, plotDisp=False)
    SingleShotProgramFFMUX.save_data(inst_ss, data_ss)
    SingleShotProgramFFMUX.save_config(inst_ss)

    d = data_ss["data"]

    Ig = np.ravel(np.array(d["i_g0"]))
    Qg = np.ravel(np.array(d["q_g0"]))
    Ie = np.ravel(np.array(d["i_e0"]))
    Qe = np.ravel(np.array(d["q_e0"]))

    g_center = np.array([np.mean(Ig), np.mean(Qg)])
    e_center = np.array([np.mean(Ie), np.mean(Qe)])

    normal = e_center - g_center
    midpoint = 0.5 * (g_center + e_center)

    print("[ModifiedRamsey] A priori separator from saved SingleShot:")
    print(f"  g_center = {g_center}")
    print(f"  e_center = {e_center}")
    print(f"  midpoint = {midpoint}")
    print(f"  normal   = {normal}")
    print(f"  separation = {np.linalg.norm(normal):.6f}")

    return {
        "g_center": g_center,
        "e_center": e_center,
        "normal": normal,
        "midpoint": midpoint,
        "data_ss": data_ss,
    }


def calibrate_active_reset_readout(
    config, soc, soccfg, outerFolder, max_align_iter=2, align_tol_frac=0.1
):
    """
    Calibrate the readout phase + single-shot I-threshold for hardware active reset.

    The active-reset feedback (ModifiedRamsey / ActiveResetVerify) thresholds on the
    RAW in-phase (I) value only, so |g> and |e> must separate ALONG I. This:
      1) runs a SingleShot g/e calibration at the current res_phase,
      2) rotates config["res_phase"] so the g->e axis lands on +I,
      3) RE-MEASURES and reads the I-threshold directly off the rotated blobs
         (so the deg2reg sign convention is never trusted — the result is measured).

    Mutates config["res_phase"] in place. Returns dict:
      res_phase, readout_threshold, reset_ground_below_threshold,
      g_center, e_center  (rotated-frame, normalized collect_shots() units).
    """
    res_ch = config["res_ch"]

    sep = get_apriori_separator_from_singleshot(
        config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
    )
    n0 = np.asarray(sep["e_center"]) - np.asarray(sep["g_center"])
    phi_deg = float(np.degrees(np.arctan2(n0[1], n0[0])))
    base_phase = int(config.get("res_phase", 0))

    print(
        f"[ActiveReset calib] g->e axis at {phi_deg:.2f} deg; rotating res_phase "
        f"to put it on I."
    )

    # Try rotating by -phi (align g->e with +I); if the sign convention flips it,
    # fall back to +phi. Keep whichever leaves the smallest |Q separation|.
    best = None
    for sign in (-1.0, +1.0):
        config["res_phase"] = int(
            base_phase + soccfg.deg2reg(sign * phi_deg, gen_ch=res_ch)
        )
        sep_r = get_apriori_separator_from_singleshot(
            config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
        gr = np.asarray(sep_r["g_center"])
        er = np.asarray(sep_r["e_center"])
        nr = er - gr
        i_sep, q_sep = abs(nr[0]), abs(nr[1])
        if best is None or q_sep < best["q_sep"]:
            best = {
                "res_phase": config["res_phase"],
                "g": gr,
                "e": er,
                "i_sep": i_sep,
                "q_sep": q_sep,
            }
        if q_sep <= align_tol_frac * i_sep:
            break

    config["res_phase"] = best["res_phase"]
    gr, er = best["g"], best["e"]
    if best["q_sep"] > align_tol_frac * best["i_sep"]:
        print(
            f"[ActiveReset calib] WARNING: residual Q separation {best['q_sep']:.4f} "
            f"vs I separation {best['i_sep']:.4f}; reset thresholding on I may be "
            f"degraded. Improve the SingleShot fidelity or rotate manually."
        )

    readout_threshold = 0.5 * (gr[0] + er[0])
    reset_ground_below = bool(gr[0] < er[0])
    print(
        f"[ActiveReset calib] res_phase={best['res_phase']} (reg units), "
        f"readout_threshold={readout_threshold:.6f}, "
        f"reset_ground_below_threshold={reset_ground_below} "
        f"(g_I={gr[0]:.4f}, e_I={er[0]:.4f})"
    )

    return {
        "res_phase": best["res_phase"],
        "readout_threshold": float(readout_threshold),
        "reset_ground_below_threshold": reset_ground_below,
        "g_center": gr,
        "e_center": er,
    }


def wire_reset_into_mr_cfg(
    mr_cfg, apriori_sep, mr_params, use_active_reset, reset_from_readout
):
    """
    Wire the chosen feedback-reset strategy + I-threshold into mr_cfg, in place.

    Both strategies threshold on the raw in-phase (I) value, so this derives the
    threshold (and the |g>-below-threshold sign) from the apriori SingleShot
    separator. reset_from_readout TAKES PRECEDENCE over use_active_reset: when it
    is set, the single-readout feedback reset is selected and the start-of-shot
    active reset is left off. Idempotent, so it is safe to re-call on ss_cal
    refreshes to update only the (drifting) threshold.
    """
    # Always write both switches so reusing mr_cfg cannot retain a stale mode.
    if not (use_active_reset or reset_from_readout):
        mr_cfg["use_active_reset"] = False
        mr_cfg["reset_from_ramsey_readout"] = False
        mr_cfg["reset_cycles"] = 0
        return
    if apriori_sep is None:
        raise RuntimeError(
            "ModifiedRamsey feedback reset requires an apriori SingleShot "
            "separator; set use_apriori_separator=True."
        )
    g = np.asarray(apriori_sep["g_center"])
    e = np.asarray(apriori_sep["e_center"])
    mr_cfg["readout_threshold"] = float(0.5 * (g[0] + e[0]))
    mr_cfg["reset_ground_below_threshold"] = bool(g[0] < e[0])
    mr_cfg["reset_readout_relax_delay"] = mr_params.get(
        "reset_readout_relax_delay", 1.0
    )
    mr_cfg["post_reset_wait"] = mr_params.get("post_reset_wait", 0.0)
    if reset_from_readout:
        mr_cfg["reset_from_ramsey_readout"] = True
        mr_cfg["use_active_reset"] = False
        mr_cfg["reset_cycles"] = 0
    else:
        mr_cfg["reset_from_ramsey_readout"] = False
        mr_cfg["use_active_reset"] = True
        mr_cfg["reset_cycles"] = int(mr_params.get("reset_cycles", 1))


# from q4diamond.Client_modules.Experiment_Scripts.mT2R import T2R
# from q4diamond.Client_modules.Experiment_Scripts.mChiShift import ChiShift
# from q4diamond.Client_modules.Experiment_Scripts.mSingleShotProgramFF import SingleShotProgramFF
# from q4diamond.Client_modules.Experiment_Scripts.mOptimizeReadoutandPulse_FF import ReadOpt_wSingleShotFF, QubitPulseOpt_wSingleShotFF

soc, soccfg = makeProxy_RFSOC_147()

# ── Output folder root — EDIT THIS before running (placeholder) ─────────────
# Per-qubit data folders are built below as f"{_QubitFolderRoot}/Q{q}//".
_QubitFolderRoot = "Z:/t1Team/Data/2026-05-18_BFE_cooldown/TATQ01-charge/RFSOC"
QubitFolders = {str(q): f"{_QubitFolderRoot}/Q{q}//" for q in range(1, 7)}

############## TATQ01-CL01-KOH (ported from CSTQ03_BFC.py) ##################
# Readout/Qubit frequencies, gains, sigma, flattop taken verbatim from the
# active block of T1_T2E_sweep_TATQ01.py (device TATQ01-CL01-KOH).
# pi2_Gain added as Gain//2 per qubit — PLACEHOLDER; retune the pi/2 pulse
# (AmplitudeRabi / SingleShot_QubitOptimize) before trusting Ramsey results.
Qubit_Parameters = {
    "1": {
        "Readout": {"Frequency": 6582.381, "Gain": 2500},
        "Qubit": {
            "Frequency": 2550.30,
            "Gain": 8000,
            "pi2_Gain": 8000 // 2,
            "sigma": 1,
            "flattop_length": 1,
        },
        "outerfoldername": QubitFolders["1"],
    },
    "2": {
        "Readout": {"Frequency": 6673.27, "Gain": 2500},
        "Qubit": {
            "Frequency": 2847.2942,
            # "Gain": 5633,
            # "pi2_Gain": 5633 // 2,
            # "sigma": 2,
            # "flattop_length": None,
            "Gain": 26730,
            "pi2_Gain": 26730 // 2,
            "sigma": 0.7,
            "flattop_length": None,
        },
        "outerfoldername": QubitFolders["2"],
    },
    "3": {
        "Readout": {"Frequency": 6792.4, "Gain": 4000},
        "Qubit": {
            "Frequency": 3130.44,
            "Gain": 2060,
            "pi2_Gain": 2060 // 2,
            "sigma": 1,
            "flattop_length": None,
        },
        "outerfoldername": QubitFolders["3"],
    },
    "4": {
        "Readout": {"Frequency": 6882.394, "Gain": 4000},
        "Qubit": {
            "Frequency": 3335.20,
            "Gain": 8000,
            "pi2_Gain": 8000 // 2,
            "sigma": 2,
            "flattop_length": None,
        },
        "outerfoldername": QubitFolders["4"],
    },
    "5": {
        "Readout": {"Frequency": 6992.13, "Gain": 2500},
        "Qubit": {
            "Frequency": 3293.7,
            "Gain": 1000,
            "pi2_Gain": 1000 // 2,
            "sigma": 0.01,
            "flattop_length": None,
        },
        "outerfoldername": QubitFolders["5"],
    },
    "6": {
        "Readout": {"Frequency": 7066.9, "Gain": 2000},
        "Qubit": {
            "Frequency": 3593.99,
            "Gain": 4000,
            "pi2_Gain": 4000 // 2,
            "sigma": 0.15,
            "flattop_length": None,
        },
        "outerfoldername": QubitFolders["6"],
    },
}
############################################################################
# yoko
start_voltage = (
    0.000  # sets voltage for the entire experiment #0.0059 working for good T2Rs
)

rm = pyvisa.ResourceManager()
yoko = rm.open_resource(
    "GPIB2::11::INSTR"
)  # TODO(BFE): verify GPIB address for this setup's Yokogawa charge source
# yoko.write("*RST")
yoko.write(":SOUR:FUNC VOLT")
yoko.write(":OUTP ON")
ramp_to(yoko, start_voltage)

yoko_fixed = (
    False  # during a charge sweep; lazy way of sweeping two tone spec over time
)

# Readout

Qubit_Readout = 2
Qubit_Pulse = 2
outerFolder = Qubit_Parameters[str(Qubit_Readout)]["outerfoldername"]

Constant2Tone = False
tl = {"tone_length": 151}
ConstantTone = False  # determine cavity frequency

RunTransmissionSweep = False  # determine cavity frequency
Transmission_params = {"reps": 10, "rounds": 10, "num_points": 101, "span": 5}

RunTransmissionSweeps = False
ts = {"start_ts_gain": 500, "end_ts_gain": 8000, "ts_step": 500}

Run2ToneSpec = False
RunSpecGainLengthSweep = False  # nested gain × length sweep (see block below)
RunTrans_QubitSpec = False
RunChargeSweep = False
charge_params = {
    "voltage_start": 0.00,
    "voltage_end": 0.03,
    "voltage_step": 0.005,
}  # 0.0001 has two periods in it
Spec_relevant_params = {
    # "qubit_gain": 100,
    "qubit_gain": 100,
    # "SpecSpan": 0.2,
    "SpecSpan": 0.4,
    # "SpecSpan": 0.25,
    "SpecNumPoints": 201,  # 750 works Q5
    "qubit_length": 22,  # length of 50flattop pulse when gauss = False # 9.5 worked
    "reps": 10,
    "rounds": 10,
    "Gauss": False,
    "sigma": 1,
    "gain": 100,
    # "gain": 150,
    "relax_delay": 3000,  # us ≈ 3*T1 (T1 ≈ 5 ms) – thermalise between spec shots
    "display": True,
    "min_sep_MHz": 0.2,
    "fit_window_mhz": 0.5,
    "prominent_ratio": 0.1,  # 500 used for charge sweeps
    # ── RunSpecGainLengthSweep controls ──────────────────────────────────
    # sweep_gain_only=True  -> sweep only qubit_gain (qubit_length held fixed at
    #                          the "qubit_length" value above).
    # sweep_gain_only=False -> nested qubit_length × qubit_gain sweep.
    "sweep_gain_only": True,
    "sweep_lengths": list(range(1, 20, 10)),  # qubit_length values to sweep
    "sweep_gains": list(range(1, 100, 10)),  # qubit_gain values to sweep
}

StabilizeTwoTone = False

RunChargeDispersionQuasiCW = False
RunChargeDispersionRamsey = False  # uses pi and pi /2 pulses; these should already be tuned up. you should also choose the frequency at one of the extrema

# middle frequency: 2306.310396009901
ChargeDispersion_params = {
    "upper_freq": 2306.567327,
    "lower_freq": 2306.032673,
    "probe_freq": 2306.567327,
    "relax_delay": 15000,  # us ≈ 3*T1 (T1 ≈ 5 ms)
    "repetitions": 500,
}

Run2ToneChargeDispersionQuasiCW = False  # new automated mode

# Modified Ramsey for charge-parity switching: two-tone search -> fixed-tau Ramsey.
# tau is automatically set to 1/(2*df) where df is the measured peak separation.
# f_ge is automatically set to the higher-frequency peak.
# The Modified-Ramsey block below uses measurement-feedback reset; the long
# relax_delay here applies only to the preceding two-tone search/calibration.
RunModifiedRamsey = True

TwoToneChargeDispersion_params = {
    "df": 0.5,  # required peak separation in MHz
    "dV": 0.0005,  # voltage step in V
    "voltage_min": 0.000,  # absolute lower bound
    "voltage_max": 0.010,  # absolute upper bound
    "max_voltage_tries": 1000,  # max search steps per cycle
    "num_cycles": 1000,  # how many times to repeat: search -> quasiCW -> restart
    "use_upper_peak": True,  # True -> probe higher-frequency peak, False -> lower-frequency peak
    "center_peak_tol_mhz": 0.05,
    # two-tone spec settings used during the search
    "SpecSpan": 1.0,
    "SpecNumPoints": 101,
    "reps": 10,
    "rounds": 10,
    "relax_delay": 15000,  # us ≈ 3*T1 (T1 ≈ 5 ms)
    "Gauss": True,
    "sigma": 2,
    "gain": 500,
    "qubit_length": 2,
    # quasi-CW settings once the peaks are separated enough
    "qcw_repetitions": 1000,
    "qcw_relax_delay": 15000,  # us ≈ 3*T1 (T1 ≈ 5 ms)
}

ModifiedRamsey_params = {
    # --- two-tone search settings (same role as TwoToneChargeDispersion_params) ---
    "df": 0.08,  # required peak separation in MHz before running Ramsey
    "dV": 0.005,  # voltage step size [V]
    "voltage_min": 0.000,  # absolute lower voltage bound [V]
    "voltage_max": 0.05,  # absolute upper voltage bound [V]
    "max_voltage_tries": 1000,  # max search steps per cycle
    "num_cycles": 1000000,  # how many search -> Ramsey cycles to run
    "inter_cycle_delay": 10,  # wait [s] between consecutive cycles (0 = no wait)
    "use_pi_pulse": False,
    "center_peak_tol_mhz": 0.03,  # fix
    "center_peak_df_for_tau": 0.15,  # fix
    "use_apriori_separator": True,
    "ss_calib_shots": 1000,
    "ss_recalib_every_n_cycles": 10,
    # two-tone spec settings used during the voltage search
    "SpecSpan": 0.125,
    "SpecNumPoints": 101,
    "reps": 10,
    "rounds": 10,
    "relax_delay": 3000,  # us ≈ 3*T1 (T1 ≈ 5 ms) – two-tone search phase only
    "Gauss": False,
    "sigma": 2,
    "gain": 100,
    "qubit_length": 25,
    # --- parity-doublet peak-finder (utils.find_parity_doublet) ---
    # Detection is referenced to the trace's noise floor (median + MAD), not its
    # dynamic range, and prefers the most symmetric, balanced, prominent pair of
    # peaks about qubit_frequency_center. Tune these if peaks are missed / spurious.
    "prominence_snr": 5.0,  # peak prominence bar, in units of noise sigma.
    #                             Lower (~3) to catch weak 2nd peaks; raise (~8)
    #                             to reject noise spikes.
    "min_sep_MHz": 0.02,  # resolution floor: closest two peaks may sit [MHz].
    #                             This is NOT the parity threshold (that is "df");
    #                             keep it well below df so near-threshold doublets
    #                             are still detected and judged against df.
    "max_sep_MHz": 0.15,  # max doublet separation [MHz]; None -> full span.
    "symmetry_tol_MHz": None,  # max |doublet midpoint - center| [MHz]; None -> half
    #                             span (lenient). Tighten (e.g. 0.05) to strictly
    #                             enforce the symmetric-pair-about-center assumption.
    "min_height_balance": 0.3,  # min (weaker/stronger) peak-height ratio for a pair.
    "smooth_window": 5,  # Savitzky-Golay smoothing window [samples, odd].
    "fit_window_mhz": 0.1,  # +/- window for sub-bin Lorentzian peak refinement [MHz].
    # Live, non-blocking refresh of the two-tone amplitude plot (peaks + center +
    # fits) in a single reused window as the voltage search walks. Requires an
    # interactive matplotlib backend (TkAgg/QtAgg); a no-op (PNG still saved)
    # under a non-interactive backend. The search never pauses for input.
    "live_display": True,
    "live_pause": 0.05,  # GUI event-loop pause per refresh [s].
    # --- Modified Ramsey settings ---
    # tau is computed automatically as 1 / (2 * peak_sep_MHz)
    # f_ge is set automatically to the higher-frequency peak
    # No passive relax delay: measurement plus the conditional pi resets the qubit.
    "mr_reps": 100000,  # number of single-shot Ramsey measurements per cycle
    # "mr_reps": 500000,  # number of single-shot Ramsey measurements per cycle
    "average_n_shots": 100,
    # Optional delay after the final Ramsey readout when feedback does not reuse
    # that readout. Ignored by reset_from_ramsey_readout, which instead uses the
    # ringdown + corrective-pi + post-reset timing below.
    "mr_relax_delay": 0.0,
    # Marker transparency for the per-cycle IQ scatter plots. Lower = more
    # transparent, so dense overlapping shots are easier to resolve individually.
    "iq_plot_alpha": 0.5,
    # ── Continuous (uninterrupted) acquisition mode ───────────────────────────
    # When True, RunModifiedRamsey skips the per-cycle two-tone search and the
    # per-cycle plotting and instead streams ModifiedRamsey chunks (mr_reps shots
    # each, ~200k = one "snippet") back to back for a long uninterrupted run. The
    # parity drive freq + separation are fixed ONCE up front (from the manual
    # freqs when skip_two_tone_calibration=True, else a single initial two-tone
    # search). Every chunk is saved as its own .npz holding raw I/Q + the current
    # g/e separator only (no images); classify/average offline. The SingleShot
    # separator calibration (ss_cal) is re-run on a TIME interval rather than
    # every N cycles, and a fixed-path snippet file is overwritten on a TIME
    # interval so the trace can be health-checked live.
    "continuous_mode": False,
    # Total acquisition time [s]; None -> run forever (stop with Ctrl-C).
    "continuous_duration_s": None,
    # Re-run ss_cal (SingleShot g/e separator) every N seconds. None/0 -> never
    # recalibrate after the initial calibration.
    "continuous_ss_recalib_interval_s": 600,
    # Overwrite "latest_snippet.npz" every N seconds for live monitoring. Every
    # chunk is also saved separately, so this is purely a fixed path to watch.
    "snippet_save_interval_s": 60,
    "use_active_reset": True,
    # Single-readout feedback reset: reuse the FINAL Ramsey readout itself as the
    # conditional-pi measurement. After each Ramsey readout, conditionally flip
    # e->g so the next shot starts in |g>, WITHOUT firing a separate reset readout
    # at the start of the shot (saves one readout per rep). Mutually exclusive with
    # use_active_reset: when this is True it TAKES PRECEDENCE and the start-of-shot
    # active reset is disabled. Reuses the same calibrated readout_threshold /
    # reset_ground_below_threshold / pi_gain / reset_readout_relax_delay /
    # post_reset_wait as active reset (the readout still discriminates |g>/|e>
    # along I via the rotated res_phase).
    "reset_from_ramsey_readout": True,
    # Parity -> computational-state mapping (phase of the closing pi/2).
    #   False (default): final pi/2 @ 180 deg -> on-resonant branch -> |g>,
    #                    off-resonant branch -> |e>.
    #   True:            final pi/2 @ 0 deg (sign-flipped 2nd pulse) -> on-resonant
    #                    branch -> |e>, off-resonant branch -> |g>.
    "flip_final_pi2": True,
    # symmetric_ramsey: drive at the midpoint f_avg = (f_lower + f_upper)/2 instead
    # of on-resonant with the upper peak, so both branches are symmetrically
    # detuned by +/- df/2. Sequence: pi/2 @ 0 deg -> wait tau=1/(2*df) (each branch
    # rotates +/-90 deg) -> closing pi/2 @ 90 deg -> readout.
    #   False (default): standard scheme (drive on upper peak, closing pi/2 @ 180).
    #   True:            symmetric-drive scheme. Default mapping f_upper -> |e>,
    #                    f_lower -> |g>; flip_final_pi2 swaps it (90 <-> 270 deg).
    "symmetric_ramsey": True,
    # Explicit symmetric-mode drive frequency [MHz] = the center of the charge-
    # dispersion curve (the midpoint between the two parity branches). Only used
    # when symmetric_ramsey is True. None -> derive it as f_ge - df/2 (f_ge = upper
    # peak), the old behaviour. Set a number to PIN the drive to a measured center
    # instead; df (hence tau and the pi relative phase between branches) is
    # unchanged, so this only sets the common-mode drive offset. Pinning it also
    # keeps the drive fixed when df/tau is swept (e.g. the calibration sweep).
    "symmetric_drive_freq": 2847.295,
    # --- manual parity-frequency mode (skip the two-tone voltage search) ---
    # When skip_two_tone_calibration is True, the two-tone spec voltage search is
    # bypassed entirely and Modified Ramsey runs directly at the two parity
    # frequencies below (MHz). The mapping mirrors the auto path:
    #   f_ge = max(lower, upper)      (drive the higher-frequency parity peak)
    #   df   = |upper - lower|  =>  tau = 1 / (2 * df)
    # The single-shot separator calibration (used to classify shots) is NOT
    # affected by this flag and still runs as configured.
    "skip_two_tone_calibration": True,
    "manual_lower_parity_freq": 2847.22,  # MHz, e.g. 2847.20
    "manual_upper_parity_freq": 2847.37,  # MHz, e.g. 2847.36
    # Yoko charge-bias voltage [V] used in manual mode. If None, manual mode runs
    # at whatever voltage the Yokogawa is currently set to (it is not changed).
    # Only applied when skip_two_tone_calibration is True.
    "manual_voltage": 0.00,
    # Active-reset readout rounds per shot (used by both the real Ramsey, when
    # use_active_reset is True, and the verification experiment).
    "reset_cycles": 1,
    # Delay between the conditioning readout (tone end) and the corrective pi.
    # The Q2 readout resonator has kappa/2pi = 190 kHz (1/kappa = 0.84 us,
    # fitted from TransmissionFF 2026-06-06), so 5 us = 6/kappa leaves <0.3%
    # of the readout photons; firing the pi at 0 us plays it on a fully
    # Stark-shifted qubit. NOTE: this syncdelay does NOT delay the threshold
    # read (the accumulated I is already latched); it only delays the pi.
    "reset_readout_relax_delay": 5.0,  # us after each reset readout
    "post_reset_wait": 2.0,  # us settle after the reset block
    ## TODO: CHANGE READOUT PARAMETERS IN TRANS_PARAMS
}

# ── Modified Ramsey calibration: fixed f_ge (Q2), sweep tau ───────────────────
# Runs the SAME ModifiedRamsey sequence (same gains/sigma/reset/etc. pulled from
# ModifiedRamsey_params), but holds the qubit drive frequency fixed at the Q2 value
# (qubit_frequency_center) and sweeps the free-evolution time tau. The range is
# expressed as an effective splitting df (tau = 1/(2*df)): df = 100 kHz -> tau =
# 5 us down to df = 10 kHz -> tau = 50 us. n_tau sets how many tau points are
# swept across that range. Output: a population-vs-tau calibration curve (Ramsey
# fringe / T2*) at the fixed Q2 frequency.
RunModifiedRamseyCalib = False
ModifiedRamseyCalib_params = {
    "df_start_MHz": 0.1,  # 100 kHz -> tau_start = 5 us
    "df_stop_MHz": 0.01,  # 10 kHz  -> tau_stop  = 50 us
    # Number of tau points swept across the range (this is the knob that sets how
    # many tau are measured).
    "n_tau": 21,
    # How the points are distributed across the range:
    #   "linear_tau" (default) – evenly in tau (5..50 us),
    #   "linear_df"            – evenly in df (0.1..0.01 MHz),
    #   "log_tau"              – geometric in tau.
    "tau_spacing": "linear_tau",
    # Shots per tau point. Smaller than the streaming mr_reps by default for speed;
    # set equal to ModifiedRamsey_params["mr_reps"] to match exactly.
    "mr_reps": 100000,
    # Block size for the averaged time trace (same role as in ModifiedRamsey):
    # consecutive shots are block-averaged in groups of this many to build the
    # averaged excited-population-vs-time trace saved per tau point.
    "average_n_shots": 100,
    # Optional yoko bias [V] for the calibration. None -> leave the Yoko where it
    # is (run at the present operating point).
    "voltage": 0.015,
    # Keep the qubit DRIVE frequency fixed exactly at f_ge = Q2 across the sweep.
    # In symmetric_ramsey mode the drive is f_ge - df/2, which would shift by df/2
    # (5-50 kHz) as df is swept; standard mode (False) drives on f_ge so it stays
    # fixed. Set to None to inherit ModifiedRamsey_params["symmetric_ramsey"].
    "symmetric_ramsey": True,
    # Explicit symmetric-mode drive frequency [MHz] = the center of the charge-
    # dispersion curve. When symmetric_ramsey is True, pinning this keeps the drive
    # FIXED at the center for every tau point (df only sets tau, not the drive), so
    # the sweep stays at a fixed qubit frequency. None -> inherit
    # ModifiedRamsey_params["symmetric_drive_freq"] (which itself defaults to the
    # derived f_ge - df/2).
    "symmetric_drive_freq": 2847.2942,
    # ── 2D mode: ALSO sweep the symmetric-mode drive frequency ────────────────
    # When sweep_drive_freq is True the calibration becomes a 2D sweep over
    # (symmetric drive frequency) x (tau): for every drive frequency the full tau
    # list above is run, producing an excited-population heatmap vs (drive, tau)
    # -- effectively a Ramsey chevron in the symmetric-drive scheme. The vertical
    # zero-detuning column (P(e) flat / pinned near 0.5 regardless of tau) marks
    # the true qubit/parity center; read it off and pin symmetric_drive_freq for
    # the real ModifiedRamsey run. Requires symmetric_ramsey=True (the swept value
    # sets cfg["symmetric_drive_freq"], which only has effect in symmetric mode).
    # The drive axis is drive_freq_center +/- drive_freq_span/2 over n_drive_freq
    # points, UNLESS drive_freq_list_MHz is given (which then takes precedence).
    # drive_freq_center=None -> use symmetric_drive_freq above (or the Q2 center
    # qubit_frequency_center if that is None too).
    "sweep_drive_freq": True,
    "drive_freq_center_MHz": None,  # None -> symmetric_drive_freq / Q2 center
    "drive_freq_span_MHz": 0.3,  # total span; axis = center +/- span/2
    "n_drive_freq": 13,
    "drive_freq_list_MHz": None,  # explicit list [MHz]; overrides center/span/n
    # Per-(drive,tau) point PNGs (averaged trace + IQ). A 2D sweep can emit
    # hundreds of files; default off when sweeping the drive, on for a plain
    # tau sweep. Set explicitly to override.
    "per_point_plots": True,
}

# ── Active-reset verification ────────────────────────────────────────────────
# Validates the hardware active reset that ModifiedRamsey relies on. First runs a
# SingleShot g/e calibration (calibrate_active_reset_readout): rotates res_phase so
# |g>/|e> separate along I and derives the I-threshold. Then sweeps four conditions
# — prep |g>/|e>  ×  reset off/on — reading the qubit out n_verify_reads times per
# shot. A working, QND reset gives: prep|e>+reset ON  P(|g>) ≈ prep|g>+reset ON
# ≈ ground readout fidelity, and ≫ the prep|e>+reset OFF control, with P(|g>) flat
# across the repeated reads. Saves per-condition data + an overlay plot + a verdict.
RunActiveResetVerify = False
ActiveResetVerify_params = {
    "n_verify_reads": 5,  # back-to-back readouts after the reset block
    "verify_relax_delay": 5.0,  # us between consecutive verification readouts
    "reps": 2000,  # single shots per condition
    "reset_cycles": 1,  # measure->feedback rounds per shot
    # 5 us = 6/kappa (kappa/2pi = 190 kHz measured) so the corrective pi fires
    # on a photon-free qubit; keep equal to ModifiedRamsey_params so ARV
    # validates the same timing the Ramsey uses.
    "reset_readout_relax_delay": 5.0,  # us after each reset readout
    # Match ModifiedRamsey_params so the verifier exercises the same post-pi settle.
    "post_reset_wait": 2.0,  # us settle after the reset block
    "relax_delay": 3000,  # us between reps (>= 3*T1 to re-thermalise)
    "plotDisp": True,
}

RunModifiedRamsey_Control = (
    False  # two-tone search -> half-period step -> sweet-spot interpolation -> Ramsey
)

ModifiedRamsey_Control_params = {
    # --- two-tone search settings (identical role to ModifiedRamsey_params) ---
    "df": 0.5,  # required peak separation in MHz to accept V1
    "dV": 0.0005,  # voltage step [V]
    "voltage_min": 0.000,
    "voltage_max": 0.010,
    "max_voltage_tries": 1000,
    "num_cycles": 1000,
    # two-tone spec hardware settings
    "SpecSpan": 1.0,
    "SpecNumPoints": 101,
    "reps": 10,
    "rounds": 10,
    "relax_delay": 15000,  # us ≈ 3*T1 (T1 ≈ 5 ms) – two-tone search phase only
    "Gauss": True,
    "sigma": 2,
    "gain": 700,
    "qubit_length": 2,
    # --- charge-dispersion fit parameters (from independent calibration) ---
    # S(V) ≈ cd_max_mhz * |cos(2π*(V-V0)/T)| where T = cd_period_mv mV
    "cd_max_mhz": 0.682,  # peak charge dispersion from fit [MHz]
    "cd_period_mv": 4.37,  # period of dispersion vs yoko voltage [mV]
    # --- sweet-spot verification ---
    # After moving to V_sweet, a new two-tone is taken.  If the measured
    # separation is below this threshold (or only one peak is found) the
    # sweet spot is flagged as verified.  Either way the Ramsey still runs.
    "sweet_spot_max_df_mhz": 0.1,
    # --- Modified Ramsey settings at the sweet spot ---
    # tau is computed from S1 (the separation found at V1), NOT from S_sweet
    # (which is ~0), so it matches the tau that ModifiedRamsey would use at V1.
    # f_ge is set from the two-tone at V_sweet (average of peaks, or single peak).
    "mr_reps": 500,
    # hysteresis / moving-average: set via setdefault below (same as ModifiedRamsey)
}

RunChiShift = False
ChiShift_params = {
    "reps": 10,
    "rounds": 100,  # this will used for all experiements below unless otherwise changed in between trials
    "TransSpan": 1,  ### MHz, span will be center+/- this parameter
    "TransNumPoints": 101,
    "cavity_shift": 0.2,
    "relax_delay": 15000,  # us ≈ 3*T1 (T1 ≈ 5 ms) – pi pulse needs full thermalisation
}

RunAmplitudeRabi = False
Amplitude_Rabi_params = {
    "qubit_freq": Qubit_Parameters[str(Qubit_Pulse)]["Qubit"]["Frequency"],
    "max_gain": 30000,
    "number_of_steps": 101,
    "reps": 10,
    "rounds": 10,
    "relax_delay": 3000,  # us ≈ 3*T1 (T1 ≈ 5 ms) – thermalise before each shot
    "fit": True,
}  # Always change the max gain if you don't see it, also compare what you get with Transmission data

RunT1 = False
RunT2 = False
# TATQ01 long-coherence regime: T1 ≈ 5 ms, T2R ≈ 1 ms. Probe T1 out to ~3*T1
# (101 × 150 us ≈ 15 ms) and T2R out to ~3*T2R (301 × 10 us ≈ 3 ms); relax_delay
# ≈ 3*T1 = 15 ms so the qubit fully thermalises between shots.
T1T2_params = {
    "T1_step": 10,
    "T1_expts": 101,
    "T1_reps": 20,
    "T1_rounds": 20,  # 80 100 30 30
    "T2_step": 10,
    "T2_expts": 301,
    "T2_reps": 20,
    "T2_rounds": 20,
    "freq_shift": 0.0,
    "relax_delay": 3000,  # us ≈ 3*T1
    "repetitions": 1000,
}

RunT1T2E = False

RunT1T2RT2E = False

RunT2E = False
# T2E ≈ 3 ms: probe out to ~3*T2E ≈ 9 ms; relax_delay ≈ 3*T1 = 15 ms.
T2E_params = {
    "T2_max_us": 9000,
    "T2_expts": 181,
    "T2_reps": 25,
    "T2_rounds": 25,
    "freq_shift": 0.0,
    "relax_delay": 15000,  # us ≈ 3*T1
    "num_pi_pulses": 1,  # need odd number of pulses
    "rotation_angle": None,
    "min_max": None,
    "repetitions": 3000,
}

SingleShot = False
SS_params = {
    "Shots": 1000,
    "Readout_Time": 5,
    "ADC_Offset": 1,
    "Qubit_Pulse": [Qubit_Pulse],
    "number_of_pulses": 1,
    "relax_delay": 3000,  # us ≈ 3*T1 (T1 ≈ 5 ms)
    "pi2_SS": False,
}  # keep at 15

RunT1SS = False
# T1 ≈ 5 ms: probe out to ~3*T1 (101 × 150 us ≈ 15 ms); relax_delay ≈ 3*T1.
T1SS_params = {
    "T1_step": 150,
    "T1_expts": 101,
    "reps": 2000,
    "angle": 0,
    "threshold": 0,
    "relax_delay": 15000,  # us ≈ 3*T1
    "calibrate_SS": True,
    "repetitions": 3000,
}

SingleShot_ReadoutOptimize = False
SS_R_params = {
    "gain_start": 400,
    "gain_stop": 2000,
    "gain_pts": 41,
    "span": 0.1,
    "trans_pts": 21,
}

SingleShot_QubitOptimize = True
SS_Q_params = {
    "q_gain_span": 500,
    "q_gain_pts": 21,
    "q_freq_span": 0.1,
    "q_freq_pts": 3,
    "number_of_pulses": 1,
}  # for optimizing pi/2 pulse, set the gain to the half of its value and optimize for n=2

# ── Automated T1 / T2 / T2Echo calibration ──────────────────────────────────
# Set RunAutoCoherence = True to run the full automated calibration pipeline.
# Override any entry in AUTO_COHERENCE_PARAMS by adding it to the dict below.
RunAutoCoherence = False
# ── TATQ01 long-coherence regime ─────────────────────────────────────────────
# Measured scales: T1 ≈ 5 ms, T2R ≈ 1 ms, T2E ≈ 3 ms.
# Two requirements drive every value below:
#   (1) Cooldown between shots ≈ 3*T1 = 15 ms  →  all *_relax_delay = 15000 us.
#       This applies to the calibration stages (Rabi, SingleShot) too, so the
#       qubit fully thermalises to |g> before every shot.
#   (2) Each coherence experiment must span ~order of its own coherence time:
#       T1   out to ~3*T1  = 15 ms,  T2R out to ~3*T2R = 3 ms,
#       T2E  out to ~3*T2E = 9 ms.
# NOTE: with 15 ms relax delays these runs are slow; trim *_reps / *_rounds /
#       *_expts if wall-clock time is a concern.
_THREE_T1_US = 15000  # 3 * T1 (T1 ≈ 5 ms) – cooldown floor for every stage
AutoCoherence_override_params = {
    # ── Stage 1: sweet-spot search ────────────────────────────────────────────
    # Disabled by default: skip the two-tone spec sweep / yoko voltage stepping
    # and start from the qubit_params frequency. Set True to run the search.
    "run_sweet_spot_search": False,
    # ── Cooldown for the calibration stages (Rabi, SingleShot) ≈ 3*T1 ─────────
    "rabi_relax_delay": _THREE_T1_US,
    "ss_relax_delay": _THREE_T1_US,
    # Other available overrides (uncomment and edit):
    # "spec_span":        1.0,    # MHz, ±span for two-tone spec
    # "spec_sigma":       2.0,    # us,  Gaussian sigma for spec drive
    # "rabi_max_gain":    12000,  # max gain for AmplitudeRabi
    # "ss_target_fidelity": 0.70, # minimum SingleShot fidelity
    # "cd_period_mv":     4.37,   # mV, charge-dispersion period if known
    # "extended_pi_pi2_opt":  False, # use extended pi and pi/2 optimization
    # "ss_gain_span_frac": 0.05, # how much of single shot to search
    # "ss_gain_pts": 20, # how many points in single shot space to search
    # "auto_readout_opt":      True,
    # ── T1 ── probe out to ~3*T1 = 15 ms (101 pts × 150 us) ───────────────────
    "T1_step": 150,  # us – wait-time step (101 pts → 15.15 ms)
    "T1_expts": 101,  # number of time points
    "T1_reps": 20,
    "T1_rounds": 20,
    "T1_relax_delay": _THREE_T1_US,  # us ≈ 3*T1
    "T1_repetitions": 1,  # number of consecutive T1 runs
    # ── T2 Ramsey ── probe out to ~3*T2R = 3 ms (301 pts × 10 us) ─────────────
    "T2_step": 10,  # us – Ramsey delay step (301 pts → 3.01 ms)
    "T2_expts": 301,
    "T2_reps": 20,
    "T2_rounds": 20,
    "T2_relax_delay": _THREE_T1_US,  # us ≈ 3*T1
    "T2_repetitions": 1,
    "T2_freq_shift": 0.0,  # MHz – artificial detuning from f_ge
    # ── T2Echo ── probe out to ~3*T2E = 9 ms (181 pts ≈ 50 us step) ───────────
    "T2E_max_us": 9000,  # us – maximum echo time
    "T2E_expts": 181,
    "T2E_reps": 25,
    "T2E_rounds": 25,
    "T2E_relax_delay": _THREE_T1_US,  # us ≈ 3*T1
    "T2E_num_pi_pulses": 1,  # must be odd
    "T2E_repetitions": 1,
}

# ── Zero-span charge-parity switching measurement ───────────────────────────
# Device-agnostic acquisition (mZeroSpanParity) + offline analysis
# (analyze_ZeroSpanParity). Full configuration contract:
#   docs/superpowers/specs/2026-05-16-bfc-charge-parity-zero-span-design.md §5
#
# Operates on the currently-selected Qubit_Readout/Qubit_Pulse qubit, reusing the
# resonator/qubit frequencies and gains derived below (resonator_frequency_center,
# qubit_gain, cavity_gain, qubit_frequency_center). Set RunZeroSpanParity = True,
# edit the blocks here, then run this file.
#
# Hard constraints (validated fail-fast in ZeroSpanParity.__init__, spec §5.3):
#   sample_period_us >= adc_trig_offset + read_length + 1.0
#   us2cycles(sample_period_us | capture_length_us) <= 65535
#   reps_per_chunk <= soccfg['readouts'][ro_ch]['avg_maxlen']
#   decimated read_length samples <= soccfg['readouts'][ro_ch]['buf_maxlen']
RunZeroSpanParity = False

# Acquisition mode + trigger source
ZSP_RunMode = "strobe"  # "strobe" (Path A, v1) | "decimated" (Path B, v2)
ZSP_StartSrc = "internal"  # "internal" (spontaneous) | "external" (triggered)

# Recalibration toggles
ZSP_RecalibrateParityFreqs = True  # run a narrow QubitSpecSliceFF first
ZSP_RecalibrateSeparator = True  # run single-shot pi-pulse g/e calibration

# Calibration cache (used when the matching Recalibrate flag is False)
ZSP_ParityFreqs_Cached = {
    "lower_peak_MHz": None,
    "higher_peak_MHz": None,
    "which_to_park": "lower",  # "lower" | "higher"
}
ZSP_Separator_Cached = {
    "g_center": None,
    "e_center": None,
    "normal": None,
    "midpoint": None,
}

# Narrow two-tone spec used when ZSP_RecalibrateParityFreqs=True (mirrors the
# Run2ToneSpec block; centered on qubit_frequency_center).
ZSP_ParitySpec_params = {
    "SpecSpan": 1.0,
    "SpecNumPoints": 201,
    "reps": 10,
    "rounds": 10,
    "relax_delay": 15000,  # us ≈ 3*T1 (T1 ≈ 5 ms)
    "Gauss": True,
    "sigma": 2,
    "gain": 500,
    "qubit_length": 2,
    "min_sep_MHz": 0.2,
    "fit_window_mhz": 0.5,
    "prominent_ratio": 0.1,
}

# Strobe-mode params (Path A). sample_period_us floor = adc_trig_offset +
# read_length + 1.0 us; reps_per_chunk capped at avg_maxlen; total record (s) =
# reps_per_chunk * n_chunks * sample_period_us * 1e-6 (~12 s for defaults below).
ZSP_StrobeParams = {
    "sample_period_us": 20.0,
    "reps_per_chunk": 10000,
    "n_chunks": 60,
    "read_length": 5.0,
    "adc_trig_offset": 0.488,
}

# Decimated-mode params (Path B). capture_length_us must cover the readout window
# (adc_trig_offset + read_length) and stay under the 16-bit cycle cap. soft_avgs
# must be 1 unless allow_soft_avgs=True (>1 destroys parity trajectories).
# n_captures>1 concatenates back-to-back captures, marking boundaries in
# gap_indices.
ZSP_DecimatedParams = {
    "capture_length_us": 100.0,
    "soft_avgs": 1,
    "n_captures": 1,
    "read_length": 80.0,
    "adc_trig_offset": 0.488,
    "allow_soft_avgs": False,
}

# Drive params (mode-independent). qubit_gain/pulse_gain = None -> use the active
# qubit's tuned values (qubit_gain / cavity_gain globals).
ZSP_DriveParams = {
    "qubit_gain": None,
    "pulse_gain": None,
    "res_phase": 0,
}

# Offline-analysis params (see analyze_parity_run docstring).
ZSP_AnalysisParams = {
    "classifier_method": "apriori",  # "apriori" | "kmeans"
    "window_us": 1000.0,
    "k_sigma": 5.0,
    "step_us": None,
    "min_burst_duration_us": None,
    "analysis_bin_us": None,  # set < read_length for decimated apriori
    "save_plots": True,
}

# ============================ VALIDATION HARNESS (spec 2026-06-01) ============================
# Strobe-only. Each block reuses ZSP_Separator_Cached / ZSP_ParityFreqs_Cached and the
# zsp_cfg already built for ZeroSpanParity. Run order: stage1 -> stage2 -> stage1 refine ->
# stage3 (gate) -> stage4 -> 5/6 -> 8 -> 7 -> 9.  See spec 6.3.
Validate_StaticContrast = False
Validate_ContrastVsQubitFreq = False
Validate_ModulationCheck = False  # pipeline-sanity gate -- run first
Validate_ControlSuite = False
Validate_EnvironmentSweep = False
Build_EvidenceReport = False

StaticContrast_params = {"freq_span_mhz": 2.0, "n_points": 41, "reps_per_point": 2000}
ContrastVsQubit_params = {"qfreq_span_mhz": 10.0, "n_points": 81}
Modulation_params = {"modulation_freq_hz": 25, "n_periods": 10}
Control_params = {"variants": ["A", "B", "C", "D"], "detune_mhz": 50.0}
Environment_params = {"param_name": "power_dB", "values": [-10, -8, -6, -4]}

cavity_gain = Qubit_Parameters[str(Qubit_Readout)]["Readout"]["Gain"]
resonator_frequency_center = Qubit_Parameters[str(Qubit_Readout)]["Readout"][
    "Frequency"
]
qubit_gain = Qubit_Parameters[str(Qubit_Pulse)]["Qubit"]["Gain"]
pi2_gain = Qubit_Parameters[str(Qubit_Pulse)]["Qubit"]["pi2_Gain"]
qubit_frequency_center = Qubit_Parameters[str(Qubit_Pulse)]["Qubit"]["Frequency"]

qubit_sigma = Qubit_Parameters[str(Qubit_Pulse)]["Qubit"]["sigma"]
qubit_flattop = Qubit_Parameters[str(Qubit_Pulse)]["Qubit"]["flattop_length"]

trans_config = {
    "reps": 1000,  # this will used for all experiements below unless otherwise changed in between trials
    "pulse_style": "const",  # --Fixed
    # "readout_length" is the ADC integration window. The ADC starts one
    # adc_trig_offset after the tone, so "length" must cover their sum.
    "length": 11,  # us - 1 us ADC offset + 10 us integration window
    "readout_length": 10,  # us - ADC integration window
    # "readout_length": 1,  # 15 [us]
    "pulse_gain": cavity_gain,  # [DAC units]
    "pulse_freq": resonator_frequency_center,  # [MHz] actual frequency is this number + "cavity_LO"
    "TransSpan": Transmission_params[
        "span"
    ],  ### 0.75 MHz, span will be center+/- this parameter
    "TransNumPoints": Transmission_params[
        "num_points"
    ],  ### number of points in the transmission frequecny
    "cav_relax_delay": 30,
}
qubit_config = {
    "qubit_pulse_style": "const",
    "qubit_gain": Spec_relevant_params["qubit_gain"],
    "qubit_freq": qubit_frequency_center,
    "qubit_length": Spec_relevant_params[
        "qubit_length"
    ],  # 20, 100 # 10 was the best for Q4
    "SpecSpan": Spec_relevant_params[
        "SpecSpan"
    ],  ### MHz, span will be center+/- this parameter
    "SpecNumPoints": Spec_relevant_params[
        "SpecNumPoints"
    ],  ### number of points in the transmission frequecny
    "current_voltage": start_voltage,
}
expt_cfg = {
    "step": 2 * qubit_config["SpecSpan"] / qubit_config["SpecNumPoints"],
    "start": qubit_config["qubit_freq"] - qubit_config["SpecSpan"],
    "expts": qubit_config["SpecNumPoints"],
}

UpdateConfig = trans_config | qubit_config | expt_cfg | tl | ts | charge_params
config = (
    BaseConfig | UpdateConfig
)  ### note that UpdateConfig will overwrite elements in BaseConfig
print(config)
config["FF_Qubits"] = FF_Qubits
# The script later rebuilds `config` for the 5-us SingleShot family. Preserve the
# 10-us Modified-Ramsey readout regime so the verification harness truly tests
# the same physical window/tone/threshold scaling as the Ramsey acquisition.
modified_ramsey_base_config = dict(config)

#### update the qubit and cavity attenuation
# cavityAtten.SetAttenuation(config["cav_Atten"], printOut=True)

if Constant2Tone:
    for i in range(1000000):
        Instance_trans = ConstantTwoTone(
            path="ConstantTwoTone",
            cfg=config,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        data_trans = Instance_trans.acquire()

if ConstantTone:
    Instance_trans = SingleTone(
        path="TransmissionFF",
        cfg=config,
        soc=soc,
        soccfg=soccfg,
        outerFolder=outerFolder,
    )
    data_trans = SingleTone.acquire(Instance_trans)

cavity_min = True
config["cavity_min"] = cavity_min  # look for dip, not peak

if RunTransmissionSweep:
    # -------------------- acquire transmission --------------------
    config["reps"] = Transmission_params["reps"]
    config["rounds"] = Transmission_params["rounds"]

    Instance_trans = CavitySpecFF(
        path="TransmissionFF",
        cfg=config,
        soc=soc,
        soccfg=soccfg,
        outerFolder=outerFolder,
    )
    data_trans = CavitySpecFF.acquire(Instance_trans)
    CavitySpecFF.display(Instance_trans, data_trans, plotDisp=True, figNum=1)
    CavitySpecFF.save_data(Instance_trans, data_trans)
    CavitySpecFF.save_config(Instance_trans)

    sig = (
        data_trans["data"]["results"][0][0][0]
        + 1j * data_trans["data"]["results"][0][0][1]
    )
    x_pts = np.asarray(data_trans["data"]["fpts"])

    y_mag = np.abs(sig)
    y_db = 20 * np.log10(y_mag)

    idx_min = np.argmin(y_db)
    center_guess = x_pts[idx_min]

    startpoint = [
        center_guess,  # f0
        4.1e4,  # Qtot
        4.3e4,  # Qext
        1e3,  # asym
        0.0,  # offset
    ]

    fit_result = fit_hanger_transmission(freq=x_pts, amp_db=y_db, startpoint=startpoint)

    popt = fit_result["popt"]
    perr = fit_result["perr"]

    f0_fit, Qtot_fit, Qext_fit, asym_fit, offset_fit = popt
    df0_fit, dQtot_fit, dQext_fit, dasym_fit, doffset_fit = perr
    Qint_fit = fit_result["Qint"]

    # update config with fitted center frequency
    config["pulse_freq"] = f0_fit

    freq_deviation = f0_fit - resonator_frequency_center
    kappa_mhz = (f0_fit / Qext_fit) / 1e6

    print("Hanger resonator fit:")
    print(f"  f0       = {f0_fit:.6f} ± {df0_fit:.6f}")
    print(f"  Qtot     = {Qtot_fit:.6e} ± {dQtot_fit:.6e}")
    print(f"  Qext     = {Qext_fit:.6e} ± {dQext_fit:.6e}")
    print(f"  Qint     = {Qint_fit:.6e}")
    print(f"  kappa    = {kappa_mhz:.6f} MHz")
    print(f"  asym     = {asym_fit:.6e} ± {dasym_fit:.6e}")
    print(f"  offset   = {offset_fit:.6e} ± {doffset_fit:.6e}")
    print(
        f"  deviation from resonator_frequency_center "
        f"({resonator_frequency_center:.6f}) = {freq_deviation:+.6f}"
    )
    print(f"Cavity frequency found at: {config['pulse_freq']:.6f}")

    # -------------------- plot fit over data --------------------
    x_fit = np.linspace(np.min(x_pts), np.max(x_pts), 2000)
    y_fit = fit_result["model"](x_fit, *popt)

    fit_min_idx = np.argmin(y_fit)
    fit_min_freq = x_fit[fit_min_idx]
    fit_min_val = y_fit[fit_min_idx]

    fig_num_fit = 100
    while plt.fignum_exists(fig_num_fit):
        fig_num_fit += 1

    min_freq = x_pts[idx_min]

    plt.figure(fig_num_fit)
    plt.plot(x_pts, y_db, "o", label="|sig| data (dB)")
    plt.plot(
        x_fit,
        y_fit,
        "-",
        linewidth=2,
        label=f"Hanger fit\nf0={f0_fit:.6f} ± {df0_fit:.6f}",
    )
    plt.axvline(
        resonator_frequency_center,
        linestyle="--",
        color="gray",
        label=f"input center = {resonator_frequency_center:.6f}",
    )
    plt.axvline(
        fit_min_freq,
        linestyle=":",
        color="red",
        label=f"fit minimum = {fit_min_freq:.6f}",
    )
    plt.axvline(
        min_freq, linestyle=":", color="blue", label=f"min freq = {min_freq:.6f}"
    )
    plt.xlabel("Frequency")
    plt.ylabel("Transmission (dB)")
    plt.title("Transmission sweep with hanger fit")
    plt.legend()
    plt.tight_layout()
    plt.show()


# perform the cavity transmission experiment
if RunTransmissionSweeps:
    config["reps"] = 20  # fast axis number of points
    config["rounds"] = 20  # slow axis number of points
    Instance_trans = CavitySpecFF(
        path="TransmissionFF",
        cfg=config,
        soc=soc,
        soccfg=soccfg,
        outerFolder=outerFolder,
    )
    data_trans = CavitySpecFF.acquire(Instance_trans)
    CavitySpecFF.display(Instance_trans, data_trans, plotDisp=True, figNum=1)
    CavitySpecFF.save_data(Instance_trans, data_trans)
    CavitySpecFF.save_config(Instance_trans)

    # update the transmission frequency to be the peak
    if cavity_min:
        config["pulse_freq"] = Instance_trans.peakFreq_min
    else:
        config["pulse_freq"] = Instance_trans.peakFreq_max
    print("Cavity frequency found at: ", config["pulse_freq"])

# qubit spec experiment
if Run2ToneSpec:
    config["reps"] = Spec_relevant_params["reps"]
    config["rounds"] = Spec_relevant_params["rounds"]
    config["Gauss"] = Spec_relevant_params["Gauss"]
    if Spec_relevant_params["Gauss"]:
        config["sigma"] = Spec_relevant_params["sigma"]
        config["qubit_gain"] = Spec_relevant_params["gain"]

    config["qubit_gain"] = Spec_relevant_params["gain"]

    config["qubit_length"] = Spec_relevant_params["qubit_length"]
    config["SpecSpan"] = Spec_relevant_params["SpecSpan"]
    config["SpecNumPoints"] = Spec_relevant_params["SpecNumPoints"]
    config["step"] = 2 * config["SpecSpan"] / config["SpecNumPoints"]
    config["start"] = qubit_frequency_center - config["SpecSpan"]
    config["expts"] = config["SpecNumPoints"]
    config["relax_delay"] = Spec_relevant_params["relax_delay"]
    display = Spec_relevant_params["display"]
    min_sep = Spec_relevant_params["min_sep_MHz"]

    Instance_specSlice = QubitSpecSliceFF(
        path="QubitSpecFF", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
    )
    data_specSlice = QubitSpecSliceFF.acquire(Instance_specSlice)
    QubitSpecSliceFF.display(
        Instance_specSlice,
        data_specSlice,
        plotDisp=display,
        figNum=2,
        min_sep=min_sep,
        fit_window_mhz=0.5,
        prominent_ratio=0.1,
    )  # can change to True
    QubitSpecSliceFF.save_data(Instance_specSlice, data_specSlice)
    QubitSpecSliceFF.save_config(Instance_specSlice)

# Gain (and optionally length) sweep.
# sweep_gain_only=True : sweep only qubit_gain at the fixed configured qubit_length.
# sweep_gain_only=False: nested qubit_length × qubit_gain sweep (a full qubit spec
#                        at every gain for each length).
# qubit_gain and qubit_length appear in every saved plot title for easy identification.
if RunSpecGainLengthSweep:
    sweep_gain_only = Spec_relevant_params.get("sweep_gain_only", False)
    sweep_gains = Spec_relevant_params["sweep_gains"]
    if sweep_gain_only:
        # Hold qubit_length fixed at the configured value; sweep gain only.
        sweep_lengths = [Spec_relevant_params["qubit_length"]]
        print(
            f"=== GainLengthSweep: GAIN-ONLY mode, qubit_length fixed at "
            f"{Spec_relevant_params['qubit_length']} µs, "
            f"{len(sweep_gains)} gain points ==="
        )
    else:
        sweep_lengths = Spec_relevant_params["sweep_lengths"]
        print(
            f"=== GainLengthSweep: nested length × gain mode, "
            f"{len(sweep_lengths)} lengths × {len(sweep_gains)} gains ==="
        )

    _disp = Spec_relevant_params["display"]
    _minsep = Spec_relevant_params["min_sep_MHz"]
    _fw = Spec_relevant_params.get("fit_window_mhz", 0.5)
    _pr = Spec_relevant_params.get("prominent_ratio", 0.1)

    for q_length in sweep_lengths:
        for q_gain in sweep_gains:
            print(
                f"\n=== GainLengthSweep: qubit_length={q_length} µs  qubit_gain={q_gain} ==="
            )

            config["reps"] = Spec_relevant_params["reps"]
            config["rounds"] = Spec_relevant_params["rounds"]
            config["Gauss"] = Spec_relevant_params["Gauss"]
            config["qubit_gain"] = q_gain
            config["qubit_length"] = q_length
            config["SpecSpan"] = Spec_relevant_params["SpecSpan"]
            config["SpecNumPoints"] = Spec_relevant_params["SpecNumPoints"]
            config["step"] = 2 * config["SpecSpan"] / config["SpecNumPoints"]
            config["start"] = qubit_frequency_center - config["SpecSpan"]
            config["expts"] = config["SpecNumPoints"]
            config["relax_delay"] = Spec_relevant_params["relax_delay"]
            if Spec_relevant_params["Gauss"]:
                config["sigma"] = Spec_relevant_params["sigma"]

            _inst = QubitSpecSliceFF(
                path="QubitSpecFF",
                cfg=config,
                soc=soc,
                soccfg=soccfg,
                outerFolder=outerFolder,
            )
            _data = QubitSpecSliceFF.acquire(_inst)
            QubitSpecSliceFF.display(
                _inst,
                _data,
                plotDisp=_disp,
                figNum=2,
                min_sep=_minsep,
                fit_window_mhz=_fw,
                prominent_ratio=_pr,
            )
            QubitSpecSliceFF.save_data(_inst, _data)
            QubitSpecSliceFF.save_config(_inst)

if RunChiShift:
    updated_params = {
        "pi_gain": qubit_gain,
        "sigma": qubit_sigma,
        "f_ge": Amplitude_Rabi_params["qubit_freq"],
        "flattop_length": qubit_flattop,
    }
    config = config | ChiShift_params | updated_params
    iChi = ChiShift(
        path="ChiShift", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
    )
    dChi = ChiShift.acquire(iChi)
    ChiShift.display(iChi, dChi, plotDisp=True, figNum=1)
    ChiShift.save_data(iChi, dChi)
    ChiShift.save_config(iChi)

if Run2ToneChargeDispersionQuasiCW:
    save_dir = os.path.join(outerFolder, "TwoToneChargeDispersion")
    os.makedirs(save_dir, exist_ok=True)

    df_required = TwoToneChargeDispersion_params["df"]
    dV = TwoToneChargeDispersion_params["dV"]
    voltage_min = max(0.0, TwoToneChargeDispersion_params["voltage_min"])
    voltage_max = TwoToneChargeDispersion_params["voltage_max"]
    max_tries = TwoToneChargeDispersion_params["max_voltage_tries"]
    num_cycles = TwoToneChargeDispersion_params["num_cycles"]

    # start each overall experiment from the yoko's current value
    current_voltage = float(yoko.query(":SOUR:LEV?"))
    direction = +1

    cycle_summary = []

    for cycle_idx in range(num_cycles):
        print(f"\n================ Cycle {cycle_idx + 1}/{num_cycles} ================")

        success = False
        chosen_probe_freq = None
        chosen_peak_sep = None

        # ---------- search loop for this cycle ----------
        for attempt_idx in range(max_tries):
            print(
                f"[TwoToneChargeDispersion] Cycle {cycle_idx + 1}/{num_cycles}, "
                f"attempt {attempt_idx + 1}/{max_tries}, V={current_voltage:.6f} V"
            )

            # --- run two-tone spec at current voltage ---
            config["current_voltage"] = current_voltage
            config["reps"] = TwoToneChargeDispersion_params["reps"]
            config["rounds"] = TwoToneChargeDispersion_params["rounds"]
            config["Gauss"] = TwoToneChargeDispersion_params["Gauss"]
            config["relax_delay"] = TwoToneChargeDispersion_params["relax_delay"]

            if config["Gauss"]:
                config["sigma"] = TwoToneChargeDispersion_params["sigma"]
                config["qubit_gain"] = TwoToneChargeDispersion_params["gain"]

            config["qubit_length"] = TwoToneChargeDispersion_params["qubit_length"]
            config["SpecSpan"] = TwoToneChargeDispersion_params["SpecSpan"]
            config["SpecNumPoints"] = TwoToneChargeDispersion_params["SpecNumPoints"]
            config["step"] = 2 * config["SpecSpan"] / config["SpecNumPoints"]
            config["start"] = qubit_frequency_center - config["SpecSpan"]
            config["expts"] = config["SpecNumPoints"]

            Instance_specSlice = QubitSpecSliceFF(
                path="TwoToneChargeDispersion",
                cfg=config,
                soc=soc,
                soccfg=soccfg,
                outerFolder=outerFolder,
            )
            data_specSlice = QubitSpecSliceFF.acquire(Instance_specSlice)
            QubitSpecSliceFF.display(
                Instance_specSlice,
                data_specSlice,
                plotDisp=False,
                figNum=2,
                min_sep=Spec_relevant_params["min_sep_MHz"],
                fit_window_mhz=Spec_relevant_params["fit_window_mhz"],
                prominent_ratio=Spec_relevant_params["prominent_ratio"],
            )
            QubitSpecSliceFF.save_data(Instance_specSlice, data_specSlice)
            QubitSpecSliceFF.save_config(Instance_specSlice)

            x_pts = np.array(data_specSlice["data"]["x_pts"])
            avgi = np.array(data_specSlice["data"]["avgi"][0][0])
            avgq = np.array(data_specSlice["data"]["avgq"][0][0])
            # Rotate onto the signal-bearing IQ axis (background-subtracted) instead
            # of the raw magnitude |I+iQ|^2, which is dominated by the large, noisy
            # background quadrature and buries the qubit feature.
            avgamp0 = project_iq_signal(avgi, avgq)

            freq_choice = choose_two_tone_freqs_from_lorentz_or_peaks(
                data_specSlice,
                min_sep_mhz=Spec_relevant_params["min_sep_MHz"],
            )

            peak_info = freq_choice["peak_info_raw"]
            peak_info["peak_freqs"] = freq_choice["freqs"]
            peak_info["peak_sep"] = freq_choice["peak_sep"]
            peak_info["source"] = freq_choice["source"]

            save_base = save_two_tone_plot(
                x_pts=x_pts,
                avgi=avgi,
                avgq=avgq,
                avgamp0=avgamp0,
                peak_info=peak_info,
                attempt_idx=attempt_idx + cycle_idx * max_tries,
                save_dir=save_dir,
                current_voltage=current_voltage,
                qubit_gain=config.get("qubit_gain"),
                qubit_length=config.get("qubit_length"),
            )

            with open(save_base + "_summary.txt", "w") as f:
                f.write(f"cycle_idx: {cycle_idx}\n")
                f.write(f"attempt_idx: {attempt_idx}\n")
                f.write(f"current_voltage: {current_voltage:.9f}\n")
                f.write(f"peak_freqs: {peak_info['peak_freqs']}\n")
                f.write(f"peak_vals: {peak_info['peak_vals']}\n")
                f.write(f"peak_sep: {peak_info['peak_sep']}\n")
                f.write(f"df_required: {df_required}\n")

            center_peak_tol_mhz = TwoToneChargeDispersion_params.get(
                "center_peak_tol_mhz", 0.05
            )

            peak_freqs = np.asarray(peak_info.get("peak_freqs", []), dtype=float)

            highest_peak_freq = None
            highest_peak_is_centered = False

            if len(peak_freqs) > 0:
                # Use the largest response in avgamp0 as the "highest peak".
                peak_indices = [int(np.argmin(np.abs(x_pts - f))) for f in peak_freqs]
                peak_heights = np.asarray([avgamp0[idx] for idx in peak_indices])
                highest_peak_freq = float(peak_freqs[int(np.argmax(peak_heights))])

                highest_peak_is_centered = (
                    abs(highest_peak_freq - qubit_frequency_center)
                    <= center_peak_tol_mhz
                )

            if highest_peak_is_centered:
                # Calibration/centered mode: run quasi-CW on the centered strongest peak.
                chosen_probe_freq = highest_peak_freq
                chosen_peak_sep = (
                    float(peak_info["peak_sep"])
                    if peak_info["peak_sep"] is not None
                    else 0.0
                )

                print(
                    f"[TwoToneChargeDispersion] Cycle {cycle_idx + 1}: centered highest peak found, "
                    f"highest_peak={highest_peak_freq:.6f} MHz, "
                    f"center={qubit_frequency_center:.6f} MHz, "
                    f"|diff|={abs(highest_peak_freq - qubit_frequency_center):.6f} MHz <= "
                    f"{center_peak_tol_mhz:.6f} MHz. Running quasi-CW with "
                    f"probe_freq={chosen_probe_freq:.6f} MHz"
                )

                success = True
                break

            elif (
                peak_info["peak_sep"] is not None
                and peak_info["peak_sep"] >= df_required
            ):
                # Normal mode: require two sufficiently separated peaks.
                if TwoToneChargeDispersion_params["use_upper_peak"]:
                    chosen_probe_freq = float(np.max(peak_info["peak_freqs"]))
                else:
                    chosen_probe_freq = float(np.min(peak_info["peak_freqs"]))

                chosen_peak_sep = float(peak_info["peak_sep"])

                print(
                    f"[TwoToneChargeDispersion] Cycle {cycle_idx + 1}: separated peaks found, "
                    f"peak separation = {chosen_peak_sep:.6f} MHz, "
                    f"probe_freq = {chosen_probe_freq:.6f} MHz"
                )

                success = True
                break

            # --- not separated enough: move voltage ---
            next_voltage, direction = choose_next_voltage(
                current_v=current_voltage,
                dv=dV,
                vmin=voltage_min,
                vmax=voltage_max,
                direction=direction,
            )

            if abs(next_voltage - current_voltage) < 1e-15:
                print("[TwoToneChargeDispersion] Voltage step stalled at bounds.")
                break

            ramp_to(yoko, next_voltage)
            current_voltage = next_voltage

        # ---------- if search failed, record and continue to next cycle ----------
        if not success:
            print(
                f"[TwoToneChargeDispersion] Cycle {cycle_idx + 1}: failed to find sufficient peak separation."
            )
            cycle_summary.append(
                {
                    "cycle_idx": cycle_idx,
                    "success": False,
                    "final_voltage": current_voltage,
                    "chosen_probe_freq": None,
                    "peak_sep": None,
                }
            )
            continue

        # ---------- run quasi-CW for this cycle ----------
        ChargeDispersion_params["probe_freq"] = chosen_probe_freq

        config["current_voltage"] = current_voltage
        config["reps"] = TwoToneChargeDispersion_params["qcw_repetitions"]
        config["rounds"] = 1
        config["Gauss"] = Spec_relevant_params["Gauss"]
        config["relax_delay"] = TwoToneChargeDispersion_params["qcw_relax_delay"]

        if config["Gauss"]:
            config["sigma"] = Spec_relevant_params["sigma"]
            config["qubit_gain"] = Spec_relevant_params["gain"]

        config["SpecNumPoints"] = 1
        config["SpecSpan"] = 0
        config["step"] = 0
        config["start"] = ChargeDispersion_params["probe_freq"]
        config["expts"] = 1

        Instance_qcw = ChargeDispersionQuasiCW(
            path="TwoToneChargeDispersion",
            cfg=config,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )

        data_qcw = Instance_qcw.acquire(load_pulses=True, print_time=True)

        raw_i = np.ravel(np.array(data_qcw["data"]["raw_i"]))
        raw_q = np.ravel(np.array(data_qcw["data"]["raw_q"]))
        iq = np.column_stack([raw_i, raw_q])

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=20)
        labels = kmeans.fit_predict(iq)
        centers = kmeans.cluster_centers_

        c0 = centers[0]
        c1 = centers[1]
        normal = c1 - c0
        midpoint = 0.5 * (c0 + c1)
        scores = (iq - midpoint) @ normal
        binary_states = (scores > 0).astype(int)

        n0 = np.sum(binary_states == 0)
        n1 = np.sum(binary_states == 1)
        if n1 > n0:
            binary_states = 1 - binary_states
            c0, c1 = c1, c0
            normal = c1 - c0
            midpoint = 0.5 * (c0 + c1)
            scores = (iq - midpoint) @ normal

        rep_period_us = (
            1.0 + 0.5 + config["readout_length"] + 10 + config["relax_delay"]
        )
        elapsed_s = np.arange(len(raw_i)) * rep_period_us * 1e-6

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base = os.path.join(save_dir, f"Cycle_{cycle_idx:03d}_QuasiCW_{timestamp}")

        I_min, I_max = raw_i.min(), raw_i.max()
        Q_min, Q_max = raw_q.min(), raw_q.max()
        I_pad = 0.05 * (I_max - I_min if I_max > I_min else 1.0)
        Q_pad = 0.05 * (Q_max - Q_min if Q_max > Q_min else 1.0)
        I_line = np.linspace(I_min - I_pad, I_max + I_pad, 400)

        vertical_line = np.abs(normal[1]) < 1e-12
        if not vertical_line:
            Q_line = midpoint[1] - (normal[0] / normal[1]) * (I_line - midpoint[0])

        plt.figure(figsize=(6, 6))
        plt.plot(raw_i, raw_q, ".", alpha=0.35, label="IQ data")
        plt.plot(c0[0], c0[1], "o", markersize=10, label="Blob center 0")
        plt.plot(c1[0], c1[1], "o", markersize=10, label="Blob center 1")
        if vertical_line:
            plt.axvline(midpoint[0], linestyle="--", linewidth=2, label="Separator")
        else:
            plt.plot(I_line, Q_line, "--", linewidth=2, label="Separator")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title(
            f"Cycle {cycle_idx + 1}: QuasiCW IQ, V={current_voltage:.6f} V, "
            f"probe={chosen_probe_freq:.6f} MHz"
        )
        plt.xlim(I_min - I_pad, I_max + I_pad)
        plt.ylim(Q_min - Q_pad, Q_max + Q_pad)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base + "_iq_separator.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(
            raw_i[binary_states == 0],
            raw_q[binary_states == 0],
            ".",
            alpha=0.5,
            label="State 0",
        )
        plt.plot(
            raw_i[binary_states == 1],
            raw_q[binary_states == 1],
            ".",
            alpha=0.5,
            label="State 1",
        )
        if vertical_line:
            plt.axvline(midpoint[0], linestyle="--", linewidth=2, label="Separator")
        else:
            plt.plot(I_line, Q_line, "--", linewidth=2, label="Separator")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title(f"Cycle {cycle_idx + 1}: QuasiCW IQ labeled")
        plt.xlim(I_min - I_pad, I_max + I_pad)
        plt.ylim(Q_min - Q_pad, Q_max + Q_pad)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base + "_iq_labeled.png", dpi=300, bbox_inches="tight")
        plt.close()

        amps = np.abs(raw_i + 1j * raw_q) ** 2
        plt.figure(figsize=(10, 4))
        plt.plot(elapsed_s, amps, "-", linewidth=1)
        plt.xlabel("Time since start (s)")
        plt.ylabel("Amplitude$^2$")
        plt.title(f"Cycle {cycle_idx + 1}: Amplitude over time")
        plt.tight_layout()
        plt.savefig(base + "_amplitude_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.step(elapsed_s, binary_states, where="post", linewidth=1.5)
        plt.plot(elapsed_s, binary_states, "o", markersize=2)
        plt.xlabel("Time since start (s)")
        plt.ylabel("Assigned state")
        plt.yticks([0, 1])
        plt.ylim(-0.1, 1.1)
        plt.title(f"Cycle {cycle_idx + 1}: QuasiCW binary trace")
        plt.tight_layout()
        plt.savefig(base + "_binary.png", dpi=300, bbox_inches="tight")
        plt.close()

        np.savez(
            base + ".npz",
            elapsed_s=np.array(elapsed_s),
            raw_i=np.array(raw_i),
            raw_q=np.array(raw_q),
            scores=np.array(scores),
            binary_states=np.array(binary_states),
            centers=np.array([c0, c1]),
            midpoint=np.array(midpoint),
            normal=np.array(normal),
            chosen_probe_freq=np.array(chosen_probe_freq),
            final_voltage=np.array(current_voltage),
            peak_sep=np.array(chosen_peak_sep),
            cycle_idx=np.array(cycle_idx),
            config=np.array(config, dtype=object),
        )

        with open(base + "_config.json", "w") as f:
            json.dump(config, f, indent=2, default=float)

        cycle_summary.append(
            {
                "cycle_idx": cycle_idx,
                "success": True,
                "final_voltage": current_voltage,
                "chosen_probe_freq": chosen_probe_freq,
                "peak_sep": chosen_peak_sep,
            }
        )

        # after finishing this cycle, restart the search from the current voltage
        # direction is preserved so the voltage walk continues naturally

    # ---------- save summary for all cycles ----------
    summary_path = os.path.join(
        save_dir, f"CycleSummary_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(cycle_summary, f, indent=2, default=float)

if RunModifiedRamsey:
    date_tag_mr = datetime.now().strftime("%Y_%m_%d")
    save_dir_mr = os.path.join(outerFolder, "ModifiedRamsey", date_tag_mr)
    os.makedirs(save_dir_mr, exist_ok=True)

    df_required_mr = ModifiedRamsey_params["df"]
    dV_mr = ModifiedRamsey_params["dV"]
    voltage_min_mr = max(0.0, ModifiedRamsey_params["voltage_min"])
    voltage_max_mr = ModifiedRamsey_params["voltage_max"]
    max_tries_mr = ModifiedRamsey_params["max_voltage_tries"]
    num_cycles_mr = ModifiedRamsey_params["num_cycles"]

    # Manual parity-frequency mode: skip the two-tone voltage search and use the
    # operator-supplied lower/upper parity frequencies directly.
    skip_two_tone_mr = ModifiedRamsey_params.get("skip_two_tone_calibration", False)
    lower_parity_freq_mr = ModifiedRamsey_params.get("manual_lower_parity_freq", None)
    upper_parity_freq_mr = ModifiedRamsey_params.get("manual_upper_parity_freq", None)
    manual_voltage_mr = ModifiedRamsey_params.get("manual_voltage", None)
    if skip_two_tone_mr:
        if lower_parity_freq_mr is None or upper_parity_freq_mr is None:
            raise ValueError(
                "skip_two_tone_calibration=True requires both "
                "manual_lower_parity_freq and manual_upper_parity_freq (MHz) "
                "in ModifiedRamsey_params."
            )
        if float(upper_parity_freq_mr) == float(lower_parity_freq_mr):
            raise ValueError(
                "manual_upper_parity_freq and manual_lower_parity_freq must "
                "differ (their separation sets tau = 1 / (2 * |upper - lower|))."
            )

    ModifiedRamsey_params.setdefault("hysteresis_low", 0.2)
    ModifiedRamsey_params.setdefault("hysteresis_high", 0.8)
    ModifiedRamsey_params.setdefault("window_ms", 0.05)

    current_voltage_mr = float(yoko.query(":SOUR:LEV?"))
    direction_mr = +1

    # In manual mode, set the charge bias to the requested voltage up front (the
    # two-tone voltage search that would normally move the Yoko is skipped).
    if skip_two_tone_mr and manual_voltage_mr is not None:
        print(
            f"[ModifiedRamsey] Manual mode: ramping Yoko "
            f"{current_voltage_mr:.6f} V -> {float(manual_voltage_mr):.6f} V"
        )
        ramp_to(yoko, float(manual_voltage_mr))
        current_voltage_mr = float(manual_voltage_mr)

    cycle_summary_mr = []

    apriori_sep_mr = None

    # Active reset thresholds on raw I only, so |g>/|e> must separate along I.
    # Calibrate res_phase + I-threshold ONCE up front (rotates config["res_phase"]
    # so the subsequent apriori separator is measured in the rotated frame). The
    # per-cycle threshold/ground-below are recomputed from apriori_sep_mr below so
    # they track blob drift across recalibrations.
    mr_reset_from_readout = ModifiedRamsey_params.get(
        "reset_from_ramsey_readout", False
    )
    # reset_from_ramsey_readout takes precedence over use_active_reset (the two are
    # mutually exclusive; see ModifiedRamsey docstring / wire_reset_into_mr_cfg).
    mr_use_active_reset = (
        ModifiedRamsey_params.get("use_active_reset", False)
        and not mr_reset_from_readout
    )
    # Both strategies threshold on raw I, so res_phase must be rotated to separate
    # |g>/|e> along I. Calibrate up front for either one.
    if mr_use_active_reset or mr_reset_from_readout:
        calibrate_active_reset_readout(
            config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )

    if ModifiedRamsey_params.get("use_apriori_separator", False):
        apriori_sep_mr = get_apriori_separator_from_singleshot(
            config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
    ss_recalib_n_mr = ModifiedRamsey_params.get("ss_recalib_every_n_cycles", None)
    inter_cycle_delay_mr = ModifiedRamsey_params.get("inter_cycle_delay", 0)
    iq_plot_alpha_mr = ModifiedRamsey_params.get("iq_plot_alpha", 0.5)

    continuous_mode_mr = ModifiedRamsey_params.get("continuous_mode", False)
    if continuous_mode_mr:
        # ── Continuous (uninterrupted) acquisition ────────────────────────────
        # Park at fixed parity frequencies and stream ModifiedRamsey chunks back
        # to back. f_ge / df are fixed ONCE here; the per-cycle two-tone search
        # and per-cycle plotting are skipped. ss_cal and a fixed-path "latest"
        # snippet are written on TIME intervals so the trace can be monitored
        # live. Every chunk is saved (raw I/Q + separator only).
        cont_duration_s = ModifiedRamsey_params.get("continuous_duration_s", None)
        ss_recalib_interval_s = ModifiedRamsey_params.get(
            "continuous_ss_recalib_interval_s", None
        )
        snippet_interval_s = ModifiedRamsey_params.get("snippet_save_interval_s", 60)

        if (mr_use_active_reset or mr_reset_from_readout) and apriori_sep_mr is None:
            raise RuntimeError(
                "continuous_mode with active reset / reset_from_ramsey_readout "
                "requires the apriori SingleShot separator; set "
                "use_apriori_separator=True."
            )

        # --- determine f_ge / df ONCE (no per-chunk search) ---
        if skip_two_tone_mr:
            chosen_peak_sep_cont = abs(
                float(upper_parity_freq_mr) - float(lower_parity_freq_mr)
            )
            chosen_probe_freq_cont = max(
                float(lower_parity_freq_mr), float(upper_parity_freq_mr)
            )
            print(
                f"[ModifiedRamsey:continuous] manual parity mode: "
                f"lower={float(lower_parity_freq_mr):.6f} MHz, "
                f"upper={float(upper_parity_freq_mr):.6f} MHz, "
                f"f_ge={chosen_probe_freq_cont:.6f} MHz, "
                f"df={chosen_peak_sep_cont:.6f} MHz, "
                f"tau={1.0 / (2.0 * chosen_peak_sep_cont):.4f} us"
            )
        else:
            # One-time two-tone voltage search to fix the parity frequencies.
            chosen_probe_freq_cont = None
            chosen_peak_sep_cont = None
            for attempt_idx_mr in range(max_tries_mr):
                print(
                    f"[ModifiedRamsey:continuous] initial search attempt "
                    f"{attempt_idx_mr + 1}/{max_tries_mr}, V={current_voltage_mr:.6f} V"
                )
                config["current_voltage"] = current_voltage_mr
                config["reps"] = ModifiedRamsey_params["reps"]
                config["rounds"] = ModifiedRamsey_params["rounds"]
                config["Gauss"] = ModifiedRamsey_params["Gauss"]
                config["relax_delay"] = ModifiedRamsey_params["relax_delay"]
                if config["Gauss"]:
                    config["sigma"] = ModifiedRamsey_params["sigma"]
                    config["qubit_gain"] = ModifiedRamsey_params["gain"]
                config["qubit_length"] = ModifiedRamsey_params["qubit_length"]
                config["SpecSpan"] = ModifiedRamsey_params["SpecSpan"]
                config["SpecNumPoints"] = ModifiedRamsey_params["SpecNumPoints"]
                config["step"] = 2 * config["SpecSpan"] / config["SpecNumPoints"]
                config["start"] = qubit_frequency_center - config["SpecSpan"]
                config["expts"] = config["SpecNumPoints"]

                Instance_specSlice_mr = QubitSpecSliceFF(
                    path="ModifiedRamsey",
                    cfg=config,
                    soc=soc,
                    soccfg=soccfg,
                    outerFolder=outerFolder,
                )
                data_specSlice_mr = QubitSpecSliceFF.acquire(Instance_specSlice_mr)
                QubitSpecSliceFF.display(
                    Instance_specSlice_mr,
                    data_specSlice_mr,
                    plotDisp=False,
                    figNum=2,
                    min_sep=Spec_relevant_params["min_sep_MHz"],
                    fit_window_mhz=Spec_relevant_params["fit_window_mhz"],
                    prominent_ratio=Spec_relevant_params["prominent_ratio"],
                )
                QubitSpecSliceFF.save_data(Instance_specSlice_mr, data_specSlice_mr)
                QubitSpecSliceFF.save_config(Instance_specSlice_mr)

                x_pts_mr = np.array(data_specSlice_mr["data"]["x_pts"])
                avgi_mr = np.array(data_specSlice_mr["data"]["avgi"][0][0])
                avgq_mr = np.array(data_specSlice_mr["data"]["avgq"][0][0])
                avgamp0_mr = project_iq_signal(avgi_mr, avgq_mr)

                doublet_mr = find_parity_doublet(
                    x_pts_mr,
                    avgamp0_mr,
                    center_freq=qubit_frequency_center,
                    min_sep_mhz=ModifiedRamsey_params.get("min_sep_MHz", 0.02),
                    max_sep_mhz=ModifiedRamsey_params.get("max_sep_MHz", None),
                    prominence_snr=ModifiedRamsey_params.get("prominence_snr", 5.0),
                    smooth_window=ModifiedRamsey_params.get("smooth_window", 5),
                    symmetry_tol_mhz=ModifiedRamsey_params.get(
                        "symmetry_tol_MHz", None
                    ),
                    min_height_balance=ModifiedRamsey_params.get(
                        "min_height_balance", 0.3
                    ),
                    fit_window_mhz=ModifiedRamsey_params.get("fit_window_mhz", 0.1),
                    refine=True,
                )

                center_peak_tol_mhz = ModifiedRamsey_params.get(
                    "center_peak_tol_mhz", 0.05
                )
                center_peak_df_for_tau = ModifiedRamsey_params.get(
                    "center_peak_df_for_tau", df_required_mr
                )
                doublet_centered_mr = (
                    doublet_mr["center"] is not None
                    and abs(doublet_mr["center"] - qubit_frequency_center)
                    <= center_peak_tol_mhz
                )
                if (
                    doublet_mr["mode"] == "doublet"
                    and doublet_mr["peak_sep"] is not None
                    and doublet_mr["peak_sep"] >= df_required_mr
                ):
                    chosen_probe_freq_cont = float(doublet_mr["upper"])
                    chosen_peak_sep_cont = float(doublet_mr["peak_sep"])
                    break
                elif doublet_centered_mr:
                    chosen_probe_freq_cont = float(doublet_mr["center"])
                    chosen_peak_sep_cont = float(center_peak_df_for_tau)
                    break

                next_voltage_mr, direction_mr = choose_next_voltage(
                    current_v=current_voltage_mr,
                    dv=dV_mr,
                    vmin=voltage_min_mr,
                    vmax=voltage_max_mr,
                    direction=direction_mr,
                )
                if abs(next_voltage_mr - current_voltage_mr) < 1e-15:
                    print("[ModifiedRamsey:continuous] voltage step stalled at bounds.")
                    break
                ramp_to(yoko, next_voltage_mr)
                current_voltage_mr = next_voltage_mr

            if chosen_probe_freq_cont is None:
                raise RuntimeError(
                    "[ModifiedRamsey:continuous] initial two-tone search failed to "
                    "find sufficient peak separation; aborting continuous run."
                )

        tau_us_cont = 1.0 / (2.0 * chosen_peak_sep_cont)

        # Build the Ramsey config once (mirrors the per-cycle path below).
        mr_cfg = {
            "f_ge": chosen_probe_freq_cont,
            "df": chosen_peak_sep_cont,
            "pi2_gain": pi2_gain,
            "pi_gain": qubit_gain,
            "use_pi_pulse": ModifiedRamsey_params.get("use_pi_pulse", False),
            "flip_final_pi2": ModifiedRamsey_params.get("flip_final_pi2", False),
            "symmetric_ramsey": ModifiedRamsey_params.get("symmetric_ramsey", False),
            "symmetric_drive_freq": ModifiedRamsey_params.get(
                "symmetric_drive_freq", None
            ),
            "mr_relax_delay": ModifiedRamsey_params.get("mr_relax_delay", 0.0),
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
            "reps": ModifiedRamsey_params["mr_reps"],
            "rounds": 1,
            "current_voltage": current_voltage_mr,
            "Qubit_number": Qubit_Readout,
            "iq_plot_alpha": iq_plot_alpha_mr,
        }
        wire_reset_into_mr_cfg(
            mr_cfg,
            apriori_sep_mr,
            ModifiedRamsey_params,
            mr_use_active_reset,
            mr_reset_from_readout,
        )
        mr_cfg["modified_ramsey_timing"] = modified_ramsey_timing(
            soccfg, config | mr_cfg
        )
        config_mr = config | mr_cfg

        run_tag_cont = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        cont_dir = os.path.join(save_dir_mr, f"continuous_{run_tag_cont}")
        os.makedirs(cont_dir, exist_ok=True)
        latest_snippet_path = os.path.join(cont_dir, "latest_snippet.npz")
        print(
            f"[ModifiedRamsey:continuous] streaming "
            f"{ModifiedRamsey_params['mr_reps']} shots/chunk -> {cont_dir}\n"
            f"  duration="
            f"{'unbounded' if cont_duration_s is None else f'{cont_duration_s} s'}, "
            f"ss_cal every {ss_recalib_interval_s} s, "
            f"snippet every {snippet_interval_s} s\n"
            f"  scheduled cadence="
            f"{config_mr['modified_ramsey_timing']['scheduled_rep_period_us']:.6f} us/shot"
        )

        run_start_cont = time.time()
        last_ss_cal_cont = run_start_cont
        last_snippet_cont = run_start_cont
        chunk_idx_cont = 0
        try:
            while True:
                now_cont = time.time()
                if (
                    cont_duration_s is not None
                    and (now_cont - run_start_cont) >= cont_duration_s
                ):
                    break

                # Periodic ss_cal: recompute the g/e separator (and the active-reset
                # I-threshold) so they track blob drift over a long run.
                if (
                    ss_recalib_interval_s is not None
                    and ss_recalib_interval_s > 0
                    and (now_cont - last_ss_cal_cont) >= ss_recalib_interval_s
                ):
                    print(
                        f"[ModifiedRamsey:continuous] ss_cal at "
                        f"t={now_cont - run_start_cont:.1f} s (before chunk {chunk_idx_cont})"
                    )
                    apriori_sep_mr = get_apriori_separator_from_singleshot(
                        config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
                    )
                    if mr_use_active_reset or mr_reset_from_readout:
                        wire_reset_into_mr_cfg(
                            mr_cfg,
                            apriori_sep_mr,
                            ModifiedRamsey_params,
                            mr_use_active_reset,
                            mr_reset_from_readout,
                        )
                        config_mr = config | mr_cfg
                    last_ss_cal_cont = time.time()

                # Acquire one uninterrupted chunk (~mr_reps shots).
                Instance_mr = ModifiedRamsey(
                    path="ModifiedRamsey",
                    cfg=config_mr,
                    soc=soc,
                    soccfg=soccfg,
                    outerFolder=outerFolder,
                )
                data_mr = ModifiedRamsey.acquire(Instance_mr)

                raw_i_cont = np.ravel(np.array(data_mr["data"]["shots_i"]))
                raw_q_cont = np.ravel(np.array(data_mr["data"]["shots_q"]))

                if apriori_sep_mr is not None:
                    g_c = np.array(apriori_sep_mr["g_center"])
                    e_c = np.array(apriori_sep_mr["e_center"])
                    mid_c = np.array(apriori_sep_mr["midpoint"])
                    nrm_c = np.array(apriori_sep_mr["normal"])
                else:
                    g_c = e_c = mid_c = nrm_c = np.array([np.nan, np.nan])

                t_since_start = time.time() - run_start_cont
                snippet_fields = dict(
                    raw_i=raw_i_cont,
                    raw_q=raw_q_cont,
                    g_center=g_c,
                    e_center=e_c,
                    midpoint=mid_c,
                    normal=nrm_c,
                    chosen_probe_freq=np.array(chosen_probe_freq_cont),
                    peak_sep=np.array(chosen_peak_sep_cont),
                    tau_us=np.array(tau_us_cont),
                    scheduled_rep_period_us=np.array(
                        config_mr["modified_ramsey_timing"]["scheduled_rep_period_us"]
                    ),
                    scheduled_timing=np.array(
                        config_mr["modified_ramsey_timing"], dtype=object
                    ),
                    streamer_elapsed_s=np.array(
                        data_mr["data"].get("streamer_elapsed_s", np.nan)
                    ),
                    streamer_average_rep_period_us=np.array(
                        data_mr["data"].get(
                            "streamer_average_rep_period_us", np.nan
                        )
                    ),
                    final_voltage=np.array(current_voltage_mr),
                    chunk_idx=np.array(chunk_idx_cont),
                    t_since_start_s=np.array(t_since_start),
                    config=np.array(config_mr, dtype=object),
                )

                chunk_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                chunk_path = os.path.join(
                    cont_dir, f"chunk_{chunk_idx_cont:06d}_{chunk_stamp}.npz"
                )
                np.savez(chunk_path, **snippet_fields)

                # Overwrite the fixed-path snippet on the time interval.
                now_after = time.time()
                if (
                    snippet_interval_s is not None
                    and snippet_interval_s > 0
                    and (now_after - last_snippet_cont) >= snippet_interval_s
                ):
                    np.savez(latest_snippet_path, **snippet_fields)
                    last_snippet_cont = now_after
                    print(
                        f"[ModifiedRamsey:continuous] snippet -> latest_snippet.npz "
                        f"(chunk {chunk_idx_cont}, t={t_since_start:.1f} s)"
                    )

                chunk_idx_cont += 1
        except KeyboardInterrupt:
            print(
                f"[ModifiedRamsey:continuous] interrupted after {chunk_idx_cont} "
                f"chunks ({time.time() - run_start_cont:.1f} s)."
            )

        cont_summary = {
            "n_chunks": chunk_idx_cont,
            "mr_reps_per_chunk": ModifiedRamsey_params["mr_reps"],
            "duration_s": time.time() - run_start_cont,
            "f_ge": chosen_probe_freq_cont,
            "df": chosen_peak_sep_cont,
            "tau_us": tau_us_cont,
            "final_voltage": current_voltage_mr,
            "save_dir": cont_dir,
        }
        with open(os.path.join(cont_dir, "continuous_summary.json"), "w") as f:
            json.dump(cont_summary, f, indent=2, default=float)
        print(f"[ModifiedRamsey:continuous] done: {cont_summary}")

    for cycle_idx_mr in range(0 if continuous_mode_mr else num_cycles_mr):
        # Delay between cycles (skipped before the first cycle). Placed at the top
        # so it applies between every pair of consecutive cycles regardless of
        # which path the previous cycle exited through (including the failure
        # `continue` further down).
        if cycle_idx_mr > 0 and inter_cycle_delay_mr > 0:
            time.sleep(inter_cycle_delay_mr)
        if ModifiedRamsey_params.get("use_apriori_separator", False):
            if apriori_sep_mr is None or (
                ss_recalib_n_mr is not None
                and ss_recalib_n_mr > 0
                and cycle_idx_mr % ss_recalib_n_mr == 0
            ):
                apriori_sep_mr = get_apriori_separator_from_singleshot(
                    config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
                )
        print(
            f"\n================ ModifiedRamsey Cycle {cycle_idx_mr + 1}/{num_cycles_mr} ================"
        )

        success_mr = False
        chosen_probe_freq_mr = None
        chosen_peak_sep_mr = None

        if skip_two_tone_mr:
            # ---------- manual parity-frequency mode ----------
            # Same f_ge / df mapping as the auto "separated peaks" branch below.
            chosen_peak_sep_mr = abs(
                float(upper_parity_freq_mr) - float(lower_parity_freq_mr)
            )
            chosen_probe_freq_mr = max(
                float(lower_parity_freq_mr), float(upper_parity_freq_mr)
            )
            success_mr = True
            print(
                f"[ModifiedRamsey] Cycle {cycle_idx_mr + 1}: manual parity mode "
                f"(two-tone search skipped), "
                f"lower={float(lower_parity_freq_mr):.6f} MHz, "
                f"upper={float(upper_parity_freq_mr):.6f} MHz, "
                f"f_ge={chosen_probe_freq_mr:.6f} MHz, "
                f"df={chosen_peak_sep_mr:.6f} MHz, "
                f"tau={1.0 / (2.0 * chosen_peak_sep_mr):.4f} us"
            )

        # ---------- two-tone voltage search (skipped in manual mode) ----------
        for attempt_idx_mr in range(max_tries_mr if not skip_two_tone_mr else 0):
            print(
                f"[ModifiedRamsey] Cycle {cycle_idx_mr + 1}/{num_cycles_mr}, "
                f"attempt {attempt_idx_mr + 1}/{max_tries_mr}, V={current_voltage_mr:.6f} V"
            )

            config["current_voltage"] = current_voltage_mr
            config["reps"] = ModifiedRamsey_params["reps"]
            config["rounds"] = ModifiedRamsey_params["rounds"]
            config["Gauss"] = ModifiedRamsey_params["Gauss"]
            config["relax_delay"] = ModifiedRamsey_params["relax_delay"]

            if config["Gauss"]:
                config["sigma"] = ModifiedRamsey_params["sigma"]
                config["qubit_gain"] = ModifiedRamsey_params["gain"]

            config["qubit_length"] = ModifiedRamsey_params["qubit_length"]
            config["SpecSpan"] = ModifiedRamsey_params["SpecSpan"]
            config["SpecNumPoints"] = ModifiedRamsey_params["SpecNumPoints"]
            config["step"] = 2 * config["SpecSpan"] / config["SpecNumPoints"]
            config["start"] = qubit_frequency_center - config["SpecSpan"]
            config["expts"] = config["SpecNumPoints"]

            Instance_specSlice_mr = QubitSpecSliceFF(
                path="ModifiedRamsey",
                cfg=config,
                soc=soc,
                soccfg=soccfg,
                outerFolder=outerFolder,
            )
            data_specSlice_mr = QubitSpecSliceFF.acquire(Instance_specSlice_mr)
            QubitSpecSliceFF.display(
                Instance_specSlice_mr,
                data_specSlice_mr,
                plotDisp=False,
                figNum=2,
                min_sep=Spec_relevant_params["min_sep_MHz"],
                fit_window_mhz=Spec_relevant_params["fit_window_mhz"],
                prominent_ratio=Spec_relevant_params["prominent_ratio"],
            )
            QubitSpecSliceFF.save_data(Instance_specSlice_mr, data_specSlice_mr)
            QubitSpecSliceFF.save_config(Instance_specSlice_mr)

            x_pts_mr = np.array(data_specSlice_mr["data"]["x_pts"])
            avgi_mr = np.array(data_specSlice_mr["data"]["avgi"][0][0])
            avgq_mr = np.array(data_specSlice_mr["data"]["avgq"][0][0])
            # Rotate onto the signal-bearing IQ axis (background-subtracted) instead
            # of the raw magnitude |I+iQ|^2, which is dominated by the large, noisy
            # background quadrature and buries the qubit feature.
            avgamp0_mr = project_iq_signal(avgi_mr, avgq_mr)

            # Robust parity-doublet finder: noise-floor-referenced peak detection,
            # symmetric-pair-about-center selection, sub-bin Lorentzian refinement.
            doublet_mr = find_parity_doublet(
                x_pts_mr,
                avgamp0_mr,
                center_freq=qubit_frequency_center,
                min_sep_mhz=ModifiedRamsey_params.get("min_sep_MHz", 0.02),
                max_sep_mhz=ModifiedRamsey_params.get("max_sep_MHz", None),
                prominence_snr=ModifiedRamsey_params.get("prominence_snr", 5.0),
                smooth_window=ModifiedRamsey_params.get("smooth_window", 5),
                symmetry_tol_mhz=ModifiedRamsey_params.get("symmetry_tol_MHz", None),
                min_height_balance=ModifiedRamsey_params.get("min_height_balance", 0.3),
                fit_window_mhz=ModifiedRamsey_params.get("fit_window_mhz", 0.1),
                refine=True,
            )

            # peak_info compatible with save_two_tone_plot and the summary file.
            peak_info_mr = {
                "peak_inds": doublet_mr["peak_inds"],
                "peak_freqs": doublet_mr["peak_freqs"],
                "peak_vals": doublet_mr["peak_vals"],
                "peak_sep": doublet_mr["peak_sep"],
                "source": doublet_mr["mode"],
            }

            save_base_mr = save_two_tone_plot(
                x_pts=x_pts_mr,
                avgi=avgi_mr,
                avgq=avgq_mr,
                avgamp0=avgamp0_mr,
                peak_info=peak_info_mr,
                current_voltage=current_voltage_mr,
                attempt_idx=attempt_idx_mr + cycle_idx_mr * max_tries_mr,
                save_dir=save_dir_mr,
                qubit_gain=config.get("qubit_gain"),
                qubit_length=config.get("qubit_length"),
                center_freq=qubit_frequency_center,
                fit=doublet_mr.get("fit"),
                live_display=ModifiedRamsey_params.get("live_display", False),
                live_pause=ModifiedRamsey_params.get("live_pause", 0.05),
            )

            with open(save_base_mr + "_summary.txt", "w") as f:
                f.write(f"cycle_idx: {cycle_idx_mr}\n")
                f.write(f"attempt_idx: {attempt_idx_mr}\n")
                f.write(f"current_voltage: {current_voltage_mr:.9f}\n")
                f.write(f"mode: {doublet_mr['mode']}\n")
                f.write(f"lower: {doublet_mr['lower']}\n")
                f.write(f"upper: {doublet_mr['upper']}\n")
                f.write(f"center: {doublet_mr['center']}\n")
                f.write(f"peak_sep: {doublet_mr['peak_sep']}\n")
                f.write(f"noise_sigma: {doublet_mr['noise_sigma']}\n")
                f.write(f"candidates: {doublet_mr['candidates']}\n")
                f.write(f"df_required: {df_required_mr}\n")

            center_peak_tol_mhz = ModifiedRamsey_params.get("center_peak_tol_mhz", 0.05)
            center_peak_df_for_tau = ModifiedRamsey_params.get(
                "center_peak_df_for_tau", df_required_mr
            )

            doublet_centered_mr = (
                doublet_mr["center"] is not None
                and abs(doublet_mr["center"] - qubit_frequency_center)
                <= center_peak_tol_mhz
            )

            if (
                doublet_mr["mode"] == "doublet"
                and doublet_mr["peak_sep"] is not None
                and doublet_mr["peak_sep"] >= df_required_mr
            ):
                # Parity mode: a resolved doublet split widely enough for Ramsey.
                chosen_probe_freq_mr = float(doublet_mr["upper"])
                chosen_peak_sep_mr = float(doublet_mr["peak_sep"])

                print(
                    f"[ModifiedRamsey] Cycle {cycle_idx_mr + 1}: doublet found, "
                    f"lower={doublet_mr['lower']:.6f} MHz, "
                    f"upper={doublet_mr['upper']:.6f} MHz, "
                    f"sep={chosen_peak_sep_mr:.6f} MHz, "
                    f"f_ge={chosen_probe_freq_mr:.6f} MHz, "
                    f"tau={1.0 / (2.0 * chosen_peak_sep_mr):.4f} us"
                )

                success_mr = True
                break

            elif doublet_centered_mr:
                # Calibration mode: feature is centered but not split enough; run
                # Ramsey at the center with the configured df-for-tau.
                chosen_probe_freq_mr = float(doublet_mr["center"])
                chosen_peak_sep_mr = float(center_peak_df_for_tau)

                print(
                    f"[ModifiedRamsey] Cycle {cycle_idx_mr + 1}: centered feature "
                    f"(mode={doublet_mr['mode']}), center={doublet_mr['center']:.6f} MHz, "
                    f"|diff|={abs(doublet_mr['center'] - qubit_frequency_center):.6f} MHz <= "
                    f"{center_peak_tol_mhz:.6f} MHz. Running calibration Ramsey with "
                    f"f_ge={chosen_probe_freq_mr:.6f} MHz, "
                    f"df_for_tau={chosen_peak_sep_mr:.6f} MHz, "
                    f"tau={1.0 / (2.0 * chosen_peak_sep_mr):.4f} us"
                )

                success_mr = True
                break

            next_voltage_mr, direction_mr = choose_next_voltage(
                current_v=current_voltage_mr,
                dv=dV_mr,
                vmin=voltage_min_mr,
                vmax=voltage_max_mr,
                direction=direction_mr,
            )

            if abs(next_voltage_mr - current_voltage_mr) < 1e-15:
                print("[ModifiedRamsey] Voltage step stalled at bounds.")
                break

            ramp_to(yoko, next_voltage_mr)
            current_voltage_mr = next_voltage_mr

        if not success_mr:
            print(
                f"[ModifiedRamsey] Cycle {cycle_idx_mr + 1}: failed to find sufficient peak separation."
            )
            cycle_summary_mr.append(
                {
                    "cycle_idx": cycle_idx_mr,
                    "success": False,
                    "final_voltage": current_voltage_mr,
                    "chosen_probe_freq": None,
                    "peak_sep": None,
                }
            )
            continue

        # ---------- run Modified Ramsey with auto-computed tau and f_ge ----------
        tau_us_mr = 1.0 / (2.0 * chosen_peak_sep_mr)

        mr_cfg = {
            "f_ge": chosen_probe_freq_mr,
            "df": chosen_peak_sep_mr,
            "pi2_gain": pi2_gain,
            "pi_gain": qubit_gain,
            "use_pi_pulse": ModifiedRamsey_params.get("use_pi_pulse", False),
            "flip_final_pi2": ModifiedRamsey_params.get("flip_final_pi2", False),
            "symmetric_ramsey": ModifiedRamsey_params.get("symmetric_ramsey", False),
            "symmetric_drive_freq": ModifiedRamsey_params.get(
                "symmetric_drive_freq", None
            ),
            "mr_relax_delay": ModifiedRamsey_params.get("mr_relax_delay", 0.0),
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
            "reps": ModifiedRamsey_params["mr_reps"],
            "rounds": 1,
            "current_voltage": current_voltage_mr,
            "Qubit_number": Qubit_Readout,
            "iq_plot_alpha": iq_plot_alpha_mr,
        }

        # Wire the chosen feedback-reset strategy into the real Ramsey run. res_phase
        # was already rotated above so |g>/|e> separate along I; the I-threshold is
        # derived from the (rotated-frame) apriori separator so it tracks blob drift
        # across recalibrations. reset_from_ramsey_readout takes precedence over
        # use_active_reset (see wire_reset_into_mr_cfg).
        wire_reset_into_mr_cfg(
            mr_cfg,
            apriori_sep_mr,
            ModifiedRamsey_params,
            mr_use_active_reset,
            mr_reset_from_readout,
        )

        mr_cfg["modified_ramsey_timing"] = modified_ramsey_timing(
            soccfg, config | mr_cfg
        )

        config_mr = config | mr_cfg

        Instance_mr = ModifiedRamsey(
            path="ModifiedRamsey",
            cfg=config_mr,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        data_mr = ModifiedRamsey.acquire(Instance_mr)
        ModifiedRamsey.display(Instance_mr, data_mr, plotDisp=False, figNum=10)
        ModifiedRamsey.save_data(Instance_mr, data_mr)
        ModifiedRamsey.save_config(Instance_mr)

        # ---------- classify shots and build averaged 0-to-1 trace ----------
        raw_i_mr = np.ravel(np.array(data_mr["data"]["shots_i"]))
        raw_q_mr = np.ravel(np.array(data_mr["data"]["shots_q"]))

        if apriori_sep_mr is None:
            raise RuntimeError(
                "ModifiedRamsey now requires apriori_sep_mr from SingleShot calibration."
            )

        average_n_shots_mr = ModifiedRamsey_params.get("average_n_shots", 25)

        classification_mr = classify_and_average_iq(
            raw_i=raw_i_mr,
            raw_q=raw_q_mr,
            g_center=apriori_sep_mr["g_center"],
            e_center=apriori_sep_mr["e_center"],
            average_n_shots=average_n_shots_mr,
        )

        binary_states_mr = classification_mr["binary_states"]
        excited_avg_mr = classification_mr["excited_avg"]
        scores_mr = classification_mr["scores"]
        normal_mr = classification_mr["normal"]
        midpoint_mr = classification_mr["midpoint"]

        c0_mr = apriori_sep_mr["g_center"]
        c1_mr = apriori_sep_mr["e_center"]

        timing_mr = config_mr["modified_ramsey_timing"]
        rep_period_us = timing_mr["scheduled_rep_period_us"]

        elapsed_ms_mr = np.arange(len(raw_i_mr)) * rep_period_us * 1e-3
        elapsed_avg_ms_mr = (
            np.arange(len(excited_avg_mr)) * average_n_shots_mr * rep_period_us * 1e-3
        )

        timestamp_mr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_mr = os.path.join(
            save_dir_mr, f"Cycle_{cycle_idx_mr:03d}_MR_{timestamp_mr}"
        )

        I_min_mr, I_max_mr = raw_i_mr.min(), raw_i_mr.max()
        Q_min_mr, Q_max_mr = raw_q_mr.min(), raw_q_mr.max()
        I_pad_mr = 0.05 * (I_max_mr - I_min_mr if I_max_mr > I_min_mr else 1.0)
        Q_pad_mr = 0.05 * (Q_max_mr - Q_min_mr if Q_max_mr > Q_min_mr else 1.0)
        I_line_mr = np.linspace(I_min_mr - I_pad_mr, I_max_mr + I_pad_mr, 400)

        vertical_line_mr = np.abs(normal_mr[1]) < 1e-12
        if not vertical_line_mr:
            Q_line_mr = midpoint_mr[1] - (normal_mr[0] / normal_mr[1]) * (
                I_line_mr - midpoint_mr[0]
            )

        # IQ scatter with calibrated SingleShot separator
        plt.figure(figsize=(6, 6))
        plt.plot(
            raw_i_mr[binary_states_mr == 0],
            raw_q_mr[binary_states_mr == 0],
            ".",
            alpha=iq_plot_alpha_mr,
            label="Assigned 0",
        )
        plt.plot(
            raw_i_mr[binary_states_mr == 1],
            raw_q_mr[binary_states_mr == 1],
            ".",
            alpha=iq_plot_alpha_mr,
            label="Assigned 1",
        )
        plt.plot(c0_mr[0], c0_mr[1], "o", markersize=10, label="SingleShot g center")
        plt.plot(c1_mr[0], c1_mr[1], "o", markersize=10, label="SingleShot e center")

        if vertical_line_mr:
            plt.axvline(
                midpoint_mr[0], linestyle="--", linewidth=2, label="g/e separator"
            )
        else:
            plt.plot(I_line_mr, Q_line_mr, "--", linewidth=2, label="g/e separator")

        plt.xlabel("I")
        plt.ylabel("Q")
        plt.xlim(I_min_mr - I_pad_mr, I_max_mr + I_pad_mr)
        plt.ylim(Q_min_mr - Q_pad_mr, Q_max_mr + Q_pad_mr)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(
            f"Cycle {cycle_idx_mr + 1}: Modified Ramsey IQ\n"
            f"V={current_voltage_mr:.6f} V, f_ge={chosen_probe_freq_mr:.6f} MHz, "
            f"tau={tau_us_mr:.4f} us"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_mr + "_iq_labeled_apriori.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Raw 0/1 shot trace
        plt.figure(figsize=(10, 4))
        plt.step(elapsed_ms_mr, binary_states_mr, where="post", linewidth=1.2)
        plt.plot(elapsed_ms_mr, binary_states_mr, "o", markersize=2)
        plt.xlabel("Time since start (ms)")
        plt.ylabel("Single-shot state")
        plt.yticks([0, 1])
        plt.ylim(-0.1, 1.1)
        plt.title(
            f"Cycle {cycle_idx_mr + 1}: Modified Ramsey single-shot state\n"
            f"tau={tau_us_mr:.4f} us, df={chosen_peak_sep_mr:.4f} MHz"
        )
        plt.tight_layout()
        plt.savefig(base_mr + "_single_shot_binary.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Averaged 0-to-1 population trace
        plt.figure(figsize=(10, 4))
        plt.plot(elapsed_avg_ms_mr, excited_avg_mr, "o-", linewidth=1.5)
        plt.xlabel("Time since start (ms)")
        plt.ylabel("Averaged excited-state population")
        plt.ylim(-0.05, 1.05)
        plt.title(
            f"Cycle {cycle_idx_mr + 1}: Modified Ramsey averaged state\n"
            f"{average_n_shots_mr} shots per point, tau={tau_us_mr:.4f} us"
        )
        plt.tight_layout()
        plt.savefig(base_mr + "_averaged_population.png", dpi=300, bbox_inches="tight")
        plt.close()

        np.savez(
            base_mr + ".npz",
            elapsed_ms=np.array(elapsed_ms_mr),
            elapsed_avg_ms=np.array(elapsed_avg_ms_mr),
            raw_i=np.array(raw_i_mr),
            raw_q=np.array(raw_q_mr),
            scores=np.array(scores_mr),
            binary_states=np.array(binary_states_mr),
            excited_avg=np.array(excited_avg_mr),
            average_n_shots=np.array(average_n_shots_mr),
            g_center=np.array(c0_mr),
            e_center=np.array(c1_mr),
            midpoint=np.array(midpoint_mr),
            normal=np.array(normal_mr),
            chosen_probe_freq=np.array(chosen_probe_freq_mr),
            peak_sep=np.array(chosen_peak_sep_mr),
            tau_us=np.array(tau_us_mr),
            final_voltage=np.array(current_voltage_mr),
            cycle_idx=np.array(cycle_idx_mr),
            scheduled_rep_period_us=np.array(rep_period_us),
            scheduled_timing=np.array(timing_mr, dtype=object),
            streamer_elapsed_s=np.array(
                data_mr["data"].get("streamer_elapsed_s", np.nan)
            ),
            streamer_average_rep_period_us=np.array(
                data_mr["data"].get("streamer_average_rep_period_us", np.nan)
            ),
            config=np.array(config_mr, dtype=object),
        )

        with open(base_mr + "_config.json", "w") as f:
            json.dump(config_mr, f, indent=2, default=float)

        cycle_summary_mr.append(
            {
                "cycle_idx": cycle_idx_mr,
                "success": True,
                "final_voltage": current_voltage_mr,
                "chosen_probe_freq": chosen_probe_freq_mr,
                "peak_sep": chosen_peak_sep_mr,
                "tau_us": tau_us_mr,
                "scheduled_rep_period_us": rep_period_us,
                "streamer_average_rep_period_us": data_mr["data"].get(
                    "streamer_average_rep_period_us", np.nan
                ),
            }
        )

    if not continuous_mode_mr:
        summary_path_mr = os.path.join(
            save_dir_mr,
            f"CycleSummary_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
        )
        with open(summary_path_mr, "w") as f:
            json.dump(cycle_summary_mr, f, indent=2, default=float)

if RunModifiedRamseyCalib:
    # Fixed-frequency tau sweep using the ModifiedRamsey sequence. tau is set via
    # df (tau = 1/(2*df)); f_ge is held at the Q2 value. Sequence/reset parameters
    # are inherited from ModifiedRamsey_params; only tau (df) and f_ge change.
    date_tag_mrcal = datetime.now().strftime("%Y_%m_%d")
    save_dir_mrcal = os.path.join(outerFolder, "ModifiedRamsey_Calib", date_tag_mrcal)
    os.makedirs(save_dir_mrcal, exist_ok=True)

    df_start_cal = ModifiedRamseyCalib_params["df_start_MHz"]
    df_stop_cal = ModifiedRamseyCalib_params["df_stop_MHz"]
    n_tau_cal = int(ModifiedRamseyCalib_params["n_tau"])
    spacing_cal = ModifiedRamseyCalib_params.get("tau_spacing", "linear_tau")

    # tau = 1/(2*df). df_start (larger df) -> smaller tau; df_stop -> larger tau.
    tau_start_cal = 1.0 / (2.0 * df_start_cal)
    tau_stop_cal = 1.0 / (2.0 * df_stop_cal)

    if spacing_cal == "linear_df":
        df_list_cal = np.linspace(df_start_cal, df_stop_cal, n_tau_cal)
        tau_list_cal = 1.0 / (2.0 * df_list_cal)
    elif spacing_cal == "log_tau":
        tau_list_cal = np.geomspace(tau_start_cal, tau_stop_cal, n_tau_cal)
        df_list_cal = 1.0 / (2.0 * tau_list_cal)
    else:  # "linear_tau" (default)
        tau_list_cal = np.linspace(tau_start_cal, tau_stop_cal, n_tau_cal)
        df_list_cal = 1.0 / (2.0 * tau_list_cal)

    f_ge_cal = qubit_frequency_center  # qubit drive frequency fixed at the Q2 value

    # symmetric_ramsey: None -> inherit from ModifiedRamsey_params; else use the
    # calib override (default False, so the drive stays exactly on f_ge = Q2).
    symmetric_ramsey_cal = ModifiedRamseyCalib_params.get("symmetric_ramsey", False)
    if symmetric_ramsey_cal is None:
        symmetric_ramsey_cal = ModifiedRamsey_params.get("symmetric_ramsey", False)

    # symmetric_drive_freq: None -> inherit from ModifiedRamsey_params; else use the
    # calib override (pins the symmetric-mode drive at the charge-dispersion center
    # so it stays fixed across the tau sweep).
    symmetric_drive_freq_cal = ModifiedRamseyCalib_params.get(
        "symmetric_drive_freq", None
    )
    if symmetric_drive_freq_cal is None:
        symmetric_drive_freq_cal = ModifiedRamsey_params.get(
            "symmetric_drive_freq", None
        )

    # ── Build the symmetric-drive-frequency sweep axis (2D mode) ──────────────
    # sweep_drive_freq=True turns this into a 2D (drive freq) x (tau) sweep: for
    # each drive frequency the full tau list is run. The swept value sets
    # cfg["symmetric_drive_freq"], so it only takes effect in symmetric mode.
    sweep_drive_cal = bool(ModifiedRamseyCalib_params.get("sweep_drive_freq", False))
    if sweep_drive_cal:
        if not symmetric_ramsey_cal:
            print(
                "[MRCalib] WARNING: sweep_drive_freq=True needs symmetric_ramsey "
                "for the swept value to take effect; forcing symmetric_ramsey=True."
            )
            symmetric_ramsey_cal = True
        drive_list_explicit_cal = ModifiedRamseyCalib_params.get(
            "drive_freq_list_MHz", None
        )
        if drive_list_explicit_cal is not None:
            drive_list_cal = [float(f) for f in drive_list_explicit_cal]
        else:
            drive_center_cal = ModifiedRamseyCalib_params.get(
                "drive_freq_center_MHz", None
            )
            if drive_center_cal is None:
                drive_center_cal = (
                    float(symmetric_drive_freq_cal)
                    if symmetric_drive_freq_cal is not None
                    else float(qubit_frequency_center)
                )
            drive_span_cal = float(
                ModifiedRamseyCalib_params.get("drive_freq_span_MHz", 0.3)
            )
            n_drive_freq_cal = int(ModifiedRamseyCalib_params.get("n_drive_freq", 11))
            drive_list_cal = list(
                np.linspace(
                    float(drive_center_cal) - drive_span_cal / 2.0,
                    float(drive_center_cal) + drive_span_cal / 2.0,
                    n_drive_freq_cal,
                )
            )
    else:
        # 1D tau sweep (unchanged): one drive setting, possibly None (-> the
        # program derives f_ge - df/2 in symmetric mode).
        drive_list_cal = [symmetric_drive_freq_cal]
    n_drive_cal = len(drive_list_cal)

    if sweep_drive_cal:
        print(
            f"[MRCalib] 2D sweep: {n_drive_cal} drive freqs x {n_tau_cal} tau pts. "
            f"drive {min(drive_list_cal):.6f}-{max(drive_list_cal):.6f} MHz; "
            f"tau {tau_start_cal:.3f}-{tau_stop_cal:.3f} us "
            f"(df {df_start_cal * 1e3:.1f}-{df_stop_cal * 1e3:.1f} kHz), "
            f"symmetric_ramsey={symmetric_ramsey_cal}"
        )
    else:
        print(
            f"[MRCalib] sweeping {n_tau_cal} tau points "
            f"({tau_start_cal:.3f}-{tau_stop_cal:.3f} us; "
            f"df {df_start_cal * 1e3:.1f}-{df_stop_cal * 1e3:.1f} kHz), "
            f"f_ge fixed at {f_ge_cal:.6f} MHz, symmetric_ramsey={symmetric_ramsey_cal}"
        )

    # Optional fixed bias for the calibration (else stay where the Yoko is).
    voltage_cal = ModifiedRamseyCalib_params.get("voltage", None)
    if voltage_cal is not None:
        print(f"[MRCalib] ramping Yoko to {float(voltage_cal):.6f} V")
        ramp_to(yoko, float(voltage_cal))
    current_voltage_cal = float(yoko.query(":SOUR:LEV?"))

    # Reset strategy (same flags as ModifiedRamsey; readout-reset takes precedence).
    mrcal_reset_from_readout = ModifiedRamsey_params.get(
        "reset_from_ramsey_readout", False
    )
    mrcal_use_active_reset = (
        ModifiedRamsey_params.get("use_active_reset", False)
        and not mrcal_reset_from_readout
    )

    # res_phase rotation + I-threshold calibration (needed for either reset path).
    if mrcal_use_active_reset or mrcal_reset_from_readout:
        calibrate_active_reset_readout(
            config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )

    # Single-shot g/e separator for shot classification (and reset threshold).
    apriori_sep_cal = get_apriori_separator_from_singleshot(
        config=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
    )

    average_n_shots_cal = int(
        ModifiedRamseyCalib_params.get(
            "average_n_shots", ModifiedRamsey_params.get("average_n_shots", 25)
        )
    )
    iq_plot_alpha_cal = ModifiedRamsey_params.get("iq_plot_alpha", 0.5)
    mr_reps_cal = int(
        ModifiedRamseyCalib_params.get("mr_reps", ModifiedRamsey_params["mr_reps"])
    )

    # 2D result grids: rows = drive frequency, cols = tau.
    mean_excited_cal = np.full((n_drive_cal, n_tau_cal), np.nan)
    mean_i_cal = np.full((n_drive_cal, n_tau_cal), np.nan)
    mean_q_cal = np.full((n_drive_cal, n_tau_cal), np.nan)
    rep_period_us_all_cal = np.full((n_drive_cal, n_tau_cal), np.nan)
    # Averaged excited-population time traces + matching time axes per (drive,tau).
    excited_avg_all_cal = [[None] * n_tau_cal for _ in range(n_drive_cal)]
    elapsed_avg_ms_all_cal = [[None] * n_tau_cal for _ in range(n_drive_cal)]

    # Per-(drive,tau) point PNGs can number in the hundreds for a 2D sweep; off by
    # default when sweeping the drive, on for a plain tau sweep (see params).
    per_point_plots_cal = bool(
        ModifiedRamseyCalib_params.get("per_point_plots", not sweep_drive_cal)
    )

    timestamp_cal = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_cal = os.path.join(save_dir_mrcal, f"MRCalib_{timestamp_cal}")

    for i_drive_cal in range(n_drive_cal):
        drive_i_cal = drive_list_cal[i_drive_cal]
        drive_tag_cal = (
            "derived (f_ge - df/2)"
            if drive_i_cal is None
            else f"{float(drive_i_cal):.6f} MHz"
        )
        for i_tau_cal in range(n_tau_cal):
            tau_i_cal = float(tau_list_cal[i_tau_cal])
            df_i_cal = float(df_list_cal[i_tau_cal])
            print(
                f"[MRCalib] drive {i_drive_cal + 1}/{n_drive_cal} "
                f"({drive_tag_cal}), tau {i_tau_cal + 1}/{n_tau_cal}: "
                f"tau={tau_i_cal:.3f} us (df={df_i_cal * 1e3:.2f} kHz)"
            )

            mr_cfg = {
                "f_ge": f_ge_cal,
                "df": df_i_cal,
                "pi2_gain": pi2_gain,
                "pi_gain": qubit_gain,
                "use_pi_pulse": ModifiedRamsey_params.get("use_pi_pulse", False),
                "flip_final_pi2": ModifiedRamsey_params.get("flip_final_pi2", False),
                "symmetric_ramsey": symmetric_ramsey_cal,
                "symmetric_drive_freq": (
                    None if drive_i_cal is None else float(drive_i_cal)
                ),
                "mr_relax_delay": ModifiedRamsey_params.get("mr_relax_delay", 0.0),
                "sigma": qubit_sigma,
                "flattop_length": qubit_flattop,
                "reps": mr_reps_cal,
                "rounds": 1,
                "current_voltage": current_voltage_cal,
                "Qubit_number": Qubit_Readout,
                "iq_plot_alpha": iq_plot_alpha_cal,
            }
            wire_reset_into_mr_cfg(
                mr_cfg,
                apriori_sep_cal,
                ModifiedRamsey_params,
                mrcal_use_active_reset,
                mrcal_reset_from_readout,
            )
            mr_cfg["modified_ramsey_timing"] = modified_ramsey_timing(
                soccfg, config | mr_cfg
            )
            config_cal = config | mr_cfg

            Instance_cal = ModifiedRamsey(
                path="ModifiedRamsey_Calib",
                cfg=config_cal,
                soc=soc,
                soccfg=soccfg,
                outerFolder=outerFolder,
            )
            data_cal = ModifiedRamsey.acquire(Instance_cal)

            raw_i_cal = np.ravel(np.array(data_cal["data"]["shots_i"]))
            raw_q_cal = np.ravel(np.array(data_cal["data"]["shots_q"]))

            classification_cal = classify_and_average_iq(
                raw_i=raw_i_cal,
                raw_q=raw_q_cal,
                g_center=apriori_sep_cal["g_center"],
                e_center=apriori_sep_cal["e_center"],
                average_n_shots=average_n_shots_cal,
            )
            binary_cal = np.asarray(classification_cal["binary_states"])
            excited_avg_cal = classification_cal["excited_avg"]
            normal_cal = np.asarray(classification_cal["normal"])
            midpoint_cal = np.asarray(classification_cal["midpoint"])
            mean_excited_cal[i_drive_cal, i_tau_cal] = float(np.mean(binary_cal))
            mean_i_cal[i_drive_cal, i_tau_cal] = float(np.mean(raw_i_cal))
            mean_q_cal[i_drive_cal, i_tau_cal] = float(np.mean(raw_q_cal))

            # Averaged excited-population time trace (same construction as the
            # ModifiedRamsey per-cycle "_averaged_population" output): block-average
            # consecutive shots in groups of average_n_shots and lay them on a time
            # axis built from the per-rep period (tau-dependent).
            rep_period_us_cal = config_cal["modified_ramsey_timing"][
                "scheduled_rep_period_us"
            ]
            rep_period_us_all_cal[i_drive_cal, i_tau_cal] = rep_period_us_cal
            elapsed_avg_ms_cal = (
                np.arange(len(excited_avg_cal))
                * average_n_shots_cal
                * rep_period_us_cal
                * 1e-3
            )
            excited_avg_all_cal[i_drive_cal][i_tau_cal] = np.asarray(excited_avg_cal)
            elapsed_avg_ms_all_cal[i_drive_cal][i_tau_cal] = np.asarray(
                elapsed_avg_ms_cal
            )

            if not per_point_plots_cal:
                continue

            point_stub_cal = (
                f"_drv{i_drive_cal:03d}_tau{i_tau_cal:03d}_{tau_i_cal:.3f}us"
            )

            plt.figure(figsize=(10, 4))
            plt.plot(elapsed_avg_ms_cal, excited_avg_cal, "o-", linewidth=1.5)
            plt.xlabel("Time since start (ms)")
            plt.ylabel("Averaged excited-state population")
            plt.ylim(-0.05, 1.05)
            plt.title(
                f"MR calib averaged trace: tau={tau_i_cal:.3f} us "
                f"(df={df_i_cal * 1e3:.2f} kHz)\n"
                f"f_drive={drive_tag_cal}, {average_n_shots_cal} shots/point, "
                f"{mr_reps_cal} reps"
            )
            plt.tight_layout()
            plt.savefig(
                base_cal + point_stub_cal + "_averaged_population.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # SingleShot IQ blobs + g/e separator for this trace: raw shots colored
            # by assignment, with the calibrated g/e centers and separator overlaid.
            c0_cal = apriori_sep_cal["g_center"]
            c1_cal = apriori_sep_cal["e_center"]
            I_min_cal, I_max_cal = raw_i_cal.min(), raw_i_cal.max()
            Q_min_cal, Q_max_cal = raw_q_cal.min(), raw_q_cal.max()
            I_pad_cal = 0.05 * (I_max_cal - I_min_cal if I_max_cal > I_min_cal else 1.0)
            Q_pad_cal = 0.05 * (Q_max_cal - Q_min_cal if Q_max_cal > Q_min_cal else 1.0)
            I_line_cal = np.linspace(I_min_cal - I_pad_cal, I_max_cal + I_pad_cal, 400)
            vertical_line_cal = np.abs(normal_cal[1]) < 1e-12
            if not vertical_line_cal:
                Q_line_cal = midpoint_cal[1] - (normal_cal[0] / normal_cal[1]) * (
                    I_line_cal - midpoint_cal[0]
                )

            plt.figure(figsize=(6, 6))
            plt.plot(
                raw_i_cal[binary_cal == 0],
                raw_q_cal[binary_cal == 0],
                ".",
                alpha=iq_plot_alpha_cal,
                label="Assigned 0",
            )
            plt.plot(
                raw_i_cal[binary_cal == 1],
                raw_q_cal[binary_cal == 1],
                ".",
                alpha=iq_plot_alpha_cal,
                label="Assigned 1",
            )
            plt.plot(
                c0_cal[0], c0_cal[1], "o", markersize=10, label="SingleShot g center"
            )
            plt.plot(
                c1_cal[0], c1_cal[1], "o", markersize=10, label="SingleShot e center"
            )
            if vertical_line_cal:
                plt.axvline(
                    midpoint_cal[0], linestyle="--", linewidth=2, label="g/e separator"
                )
            else:
                plt.plot(
                    I_line_cal, Q_line_cal, "--", linewidth=2, label="g/e separator"
                )
            plt.xlabel("I")
            plt.ylabel("Q")
            plt.xlim(I_min_cal - I_pad_cal, I_max_cal + I_pad_cal)
            plt.ylim(Q_min_cal - Q_pad_cal, Q_max_cal + Q_pad_cal)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title(
                f"MR calib IQ: tau={tau_i_cal:.3f} us (df={df_i_cal * 1e3:.2f} kHz)\n"
                f"f_drive={drive_tag_cal}, V={current_voltage_cal:.6f} V"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                base_cal + point_stub_cal + "_iq_labeled.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # Per-drive population-vs-tau curves.
    for i_drive_cal in range(n_drive_cal):
        drive_i_cal = drive_list_cal[i_drive_cal]
        drive_tag_cal = (
            "derived (f_ge - df/2)"
            if drive_i_cal is None
            else f"{float(drive_i_cal):.6f} MHz"
        )
        plt.figure(figsize=(8, 5))
        plt.plot(tau_list_cal, mean_excited_cal[i_drive_cal], "o-")
        plt.xlabel("tau (us)")
        plt.ylabel("Excited-state population")
        plt.ylim(-0.05, 1.05)
        plt.title(
            f"Modified Ramsey calibration: population vs tau\n"
            f"f_drive={drive_tag_cal} (Q{Qubit_Pulse}), V={current_voltage_cal:.6f} V, "
            f"{mr_reps_cal} reps/pt"
        )
        plt.tight_layout()
        suffix_cal = (
            "_pop_vs_tau.png"
            if n_drive_cal == 1
            else f"_drv{i_drive_cal:03d}_pop_vs_tau.png"
        )
        plt.savefig(base_cal + suffix_cal, dpi=300, bbox_inches="tight")
        plt.close()

    # 2D chevron heatmap: excited population vs (symmetric drive freq, tau). The
    # near-vertical column where P(e) is flat / pinned ~0.5 across tau marks the
    # zero-detuning (true qubit/parity center) drive frequency.
    if n_drive_cal > 1:
        drive_arr_cal = np.array([float(f) for f in drive_list_cal])

        def _centers_to_edges(centers):
            centers = np.asarray(centers, dtype=float)
            if len(centers) == 1:
                return np.array([centers[0] - 0.5, centers[0] + 0.5])
            mids = 0.5 * (centers[1:] + centers[:-1])
            first = centers[0] - (mids[0] - centers[0])
            last = centers[-1] + (centers[-1] - mids[-1])
            return np.concatenate([[first], mids, [last]])

        tau_edges_cal = _centers_to_edges(tau_list_cal)
        drive_edges_cal = _centers_to_edges(drive_arr_cal)

        plt.figure(figsize=(9, 6))
        pcm_cal = plt.pcolormesh(
            tau_edges_cal,
            drive_edges_cal,
            mean_excited_cal,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(pcm_cal, label="Excited-state population")
        plt.xlabel("tau (us)")
        plt.ylabel("Symmetric drive frequency (MHz)")
        plt.title(
            f"Modified Ramsey calibration chevron: P(e) vs (drive, tau)\n"
            f"Q{Qubit_Pulse}, V={current_voltage_cal:.6f} V, {mr_reps_cal} reps/pt"
        )
        plt.tight_layout()
        plt.savefig(base_cal + "_pop_vs_drive_tau.png", dpi=300, bbox_inches="tight")
        plt.close()

    drive_save_cal = np.array(
        [np.nan if f is None else float(f) for f in drive_list_cal]
    )
    np.savez(
        base_cal + ".npz",
        tau_list=np.array(tau_list_cal),
        df_list=np.array(df_list_cal),
        drive_list=drive_save_cal,
        sweep_drive_freq=np.array(sweep_drive_cal),
        mean_excited=np.array(mean_excited_cal),
        mean_i=np.array(mean_i_cal),
        mean_q=np.array(mean_q_cal),
        excited_avg_all=np.array(excited_avg_all_cal, dtype=object),
        elapsed_avg_ms_all=np.array(elapsed_avg_ms_all_cal, dtype=object),
        scheduled_rep_period_us=np.array(rep_period_us_all_cal),
        f_ge=np.array(f_ge_cal),
        voltage=np.array(current_voltage_cal),
        mr_reps=np.array(mr_reps_cal),
        average_n_shots=np.array(average_n_shots_cal),
        symmetric_ramsey=np.array(symmetric_ramsey_cal),
        symmetric_drive_freq=np.array(
            np.nan
            if symmetric_drive_freq_cal is None
            else float(symmetric_drive_freq_cal)
        ),
        g_center=np.array(apriori_sep_cal["g_center"]),
        e_center=np.array(apriori_sep_cal["e_center"]),
    )
    with open(base_cal + "_config.json", "w") as f:
        json.dump(ModifiedRamseyCalib_params, f, indent=2, default=float)

    print(f"[MRCalib] done -> {base_cal}.npz")

if RunModifiedRamsey_Control:
    save_dir_mrc = os.path.join(outerFolder, "ModifiedRamsey_Control")
    os.makedirs(save_dir_mrc, exist_ok=True)

    sweet_max_df_mrc = ModifiedRamsey_Control_params["sweet_spot_max_df_mhz"]
    cd_max_mhz_mrc = ModifiedRamsey_Control_params["cd_max_mhz"]
    half_period_v_mrc = ModifiedRamsey_Control_params["cd_period_mv"] * 1e-3 / 2.0
    num_cycles_mrc = ModifiedRamsey_Control_params["num_cycles"]

    ModifiedRamsey_Control_params.setdefault("hysteresis_low", 0.4)
    ModifiedRamsey_Control_params.setdefault("hysteresis_high", 0.6)
    ModifiedRamsey_Control_params.setdefault("window_ms", 0.1)

    current_voltage_mrc = float(yoko.query(":SOUR:LEV?"))
    cycle_summary_mrc = []

    # Parameters forwarded to find_sweet_spot on every cycle.
    # Spec settings and search bounds come directly from ModifiedRamsey_Control_params
    # so the user only has to edit one dict.
    sweet_params_mrc = {
        "spec_span": ModifiedRamsey_Control_params["SpecSpan"],
        "spec_num_pts": ModifiedRamsey_Control_params["SpecNumPoints"],
        "spec_reps": ModifiedRamsey_Control_params["reps"],
        "spec_rounds": ModifiedRamsey_Control_params["rounds"],
        "spec_relax_delay": ModifiedRamsey_Control_params["relax_delay"],
        "spec_sigma": ModifiedRamsey_Control_params["sigma"],
        "spec_gain": ModifiedRamsey_Control_params["gain"],
        "spec_qubit_length": ModifiedRamsey_Control_params["qubit_length"],
        "ss_search_df_trigger": ModifiedRamsey_Control_params["df"],
        "ss_accept_sep": ModifiedRamsey_Control_params["sweet_spot_max_df_mhz"],
        "ss_dV": ModifiedRamsey_Control_params["dV"],
        "ss_voltage_min": ModifiedRamsey_Control_params["voltage_min"],
        "ss_voltage_max": ModifiedRamsey_Control_params["voltage_max"],
        "ss_max_tries": ModifiedRamsey_Control_params["max_voltage_tries"],
        "cd_period_mv": ModifiedRamsey_Control_params.get("cd_period_mv"),
    }

    for cycle_idx_mrc in range(num_cycles_mrc):
        print(
            f"\n================ ModifiedRamsey_Control Cycle {cycle_idx_mrc + 1}/{num_cycles_mrc} ================"
        )

        # ── Sweet-spot search ────────────────────────────────────────────────
        # Uses the two-strategy mAutoCoherence search:
        #   Strategy A (cd_period_mv known): one initial spec + one half-period
        #     jump → typically finds the sweet spot in 1-2 acquisitions.
        #   Strategy B (fallback): incremental walk tracking minimum separation.
        f_ge_mrc, V_sweet_mrc, pki_sw, log_mrc = find_sweet_spot(
            soc=soc,
            soccfg=soccfg,
            config=config,
            save_folder=save_dir_mrc + "/",
            qubit_freq_center=qubit_frequency_center,
            qubit_readout=Qubit_Readout,
            yoko=yoko,
            sweet_params=sweet_params_mrc,
        )
        if V_sweet_mrc is not None:
            current_voltage_mrc = V_sweet_mrc
        for line_mrc in log_mrc:
            print(f"  {line_mrc}")

        S_sweet_mrc = (
            float(pki_sw["peak_sep"]) if pki_sw["peak_sep"] is not None else 0.0
        )
        sweet_verified = pki_sw["peak_sep"] is None or S_sweet_mrc < sweet_max_df_mrc

        # tau for Ramsey: maximum sensitivity is at tau = 1 / (2 * max_dispersion).
        # We use cd_max_mhz (from the independent calibration) rather than a
        # measured S1, since the search no longer requires a detour to a
        # large-separation voltage.
        tau_mrc = 1.0 / (2.0 * cd_max_mhz_mrc)

        print(
            f"[MRC] Cycle {cycle_idx_mrc + 1}: V_sweet={current_voltage_mrc:.6f} V, "
            f"f_ge={f_ge_mrc:.6f} MHz, S_sweet={S_sweet_mrc:.4f} MHz, "
            f"verified={sweet_verified}, tau={tau_mrc:.4f} us"
        )

        # ── Run Modified Ramsey at the sweet spot ────────────────────────────
        if sweet_verified:
            mr_cfg_mrc = {
                "f_ge": f_ge_mrc,
                "df": cd_max_mhz_mrc,
                "pi2_gain": pi2_gain,
                "mr_relax_delay": ModifiedRamsey_Control_params.get(
                    "mr_relax_delay", 0.0
                ),
                "sigma": qubit_sigma,
                "flattop_length": qubit_flattop,
                "reps": ModifiedRamsey_Control_params["mr_reps"],
                "rounds": 1,
                "current_voltage": current_voltage_mrc,
                "Qubit_number": Qubit_Readout,
            }
            mr_cfg_mrc["modified_ramsey_timing"] = modified_ramsey_timing(
                soccfg, config | mr_cfg_mrc
            )
            config_mrc = config | mr_cfg_mrc

            Instance_mrc = ModifiedRamsey(
                path="ModifiedRamsey_Control",
                cfg=config_mrc,
                soc=soc,
                soccfg=soccfg,
                outerFolder=outerFolder,
            )
            data_mrc = ModifiedRamsey.acquire(Instance_mrc)
            ModifiedRamsey.display(Instance_mrc, data_mrc, plotDisp=False, figNum=11)
            ModifiedRamsey.save_data(Instance_mrc, data_mrc)
            ModifiedRamsey.save_config(Instance_mrc)

            # ---------- classify shots and build parity trace ----------
            raw_i_mrc = np.ravel(np.array(data_mrc["data"]["shots_i"]))
            raw_q_mrc = np.ravel(np.array(data_mrc["data"]["shots_q"]))
            iq_mrc = np.column_stack([raw_i_mrc, raw_q_mrc])

            kmeans_mrc = KMeans(n_clusters=2, random_state=0, n_init=20)
            kmeans_mrc.fit_predict(iq_mrc)
            centers_mrc = kmeans_mrc.cluster_centers_

            c0_mrc = centers_mrc[0]
            c1_mrc = centers_mrc[1]
            normal_mrc = c1_mrc - c0_mrc
            midpoint_mrc = 0.5 * (c0_mrc + c1_mrc)
            scores_mrc = (iq_mrc - midpoint_mrc) @ normal_mrc
            binary_states_mrc = (scores_mrc > 0).astype(int)

            n0_mrc = np.sum(binary_states_mrc == 0)
            n1_mrc = np.sum(binary_states_mrc == 1)
            if n1_mrc > n0_mrc:
                binary_states_mrc = 1 - binary_states_mrc
                c0_mrc, c1_mrc = c1_mrc, c0_mrc
                normal_mrc = c1_mrc - c0_mrc
                midpoint_mrc = 0.5 * (c0_mrc + c1_mrc)
                scores_mrc = (iq_mrc - midpoint_mrc) @ normal_mrc

            rep_period_us_mrc = config_mrc["modified_ramsey_timing"][
                "scheduled_rep_period_us"
            ]
            elapsed_ms_mrc = np.arange(len(raw_i_mrc)) * rep_period_us_mrc * 1e-3

            timestamp_mrc = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            base_mrc = os.path.join(
                save_dir_mrc, f"Cycle_{cycle_idx_mrc:03d}_MRC_{timestamp_mrc}"
            )

            I_min_mrc, I_max_mrc = raw_i_mrc.min(), raw_i_mrc.max()
            Q_min_mrc, Q_max_mrc = raw_q_mrc.min(), raw_q_mrc.max()
            I_pad_mrc = 0.05 * (I_max_mrc - I_min_mrc if I_max_mrc > I_min_mrc else 1.0)
            Q_pad_mrc = 0.05 * (Q_max_mrc - Q_min_mrc if Q_max_mrc > Q_min_mrc else 1.0)
            I_line_mrc = np.linspace(I_min_mrc - I_pad_mrc, I_max_mrc + I_pad_mrc, 400)

            vertical_line_mrc = np.abs(normal_mrc[1]) < 1e-12
            if not vertical_line_mrc:
                Q_line_mrc = midpoint_mrc[1] - (normal_mrc[0] / normal_mrc[1]) * (
                    I_line_mrc - midpoint_mrc[0]
                )

            # IQ scatter
            plt.figure(figsize=(6, 6))
            plt.plot(
                raw_i_mrc[binary_states_mrc == 0],
                raw_q_mrc[binary_states_mrc == 0],
                ".",
                alpha=0.5,
                label="State 0",
            )
            plt.plot(
                raw_i_mrc[binary_states_mrc == 1],
                raw_q_mrc[binary_states_mrc == 1],
                ".",
                alpha=0.5,
                label="State 1",
            )
            plt.plot(c0_mrc[0], c0_mrc[1], "o", markersize=10)
            plt.plot(c1_mrc[0], c1_mrc[1], "o", markersize=10)
            if vertical_line_mrc:
                plt.axvline(
                    midpoint_mrc[0], linestyle="--", linewidth=2, label="Separator"
                )
            else:
                plt.plot(I_line_mrc, Q_line_mrc, "--", linewidth=2, label="Separator")
            plt.xlabel("I")
            plt.ylabel("Q")
            plt.xlim(I_min_mrc - I_pad_mrc, I_max_mrc + I_pad_mrc)
            plt.ylim(Q_min_mrc - Q_pad_mrc, Q_max_mrc + Q_pad_mrc)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title(
                f"Cycle {cycle_idx_mrc + 1}: MRC IQ (sweet spot)\n"
                f"V_sweet={current_voltage_mrc:.6f} V, f_ge={f_ge_mrc:.6f} MHz, "
                f"tau={tau_mrc:.4f} us"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(base_mrc + "_iq_labeled.png", dpi=300, bbox_inches="tight")
            plt.close()

            # raw binary parity trace
            plt.figure(figsize=(10, 4))
            plt.step(elapsed_ms_mrc, binary_states_mrc, where="post", linewidth=1.5)
            plt.plot(elapsed_ms_mrc, binary_states_mrc, "o", markersize=2)
            plt.xlabel("Time since start (ms)")
            plt.ylabel("Parity state")
            plt.yticks([0, 1])
            plt.ylim(-0.1, 1.1)
            plt.title(
                f"Cycle {cycle_idx_mrc + 1}: MRC parity trace (sweet spot control)\n"
                f"tau={tau_mrc:.4f} us, S_sweet={S_sweet_mrc:.4f} MHz"
            )
            plt.tight_layout()
            plt.savefig(base_mrc + "_binary.png", dpi=300, bbox_inches="tight")
            plt.close()

            # ---------- moving-average + hysteresis ----------
            window_ms_mrc = ModifiedRamsey_Control_params["window_ms"]
            dt_ms_mrc = rep_period_us_mrc * 1e-3
            window_n_mrc = max(1, int(round(window_ms_mrc / dt_ms_mrc)))
            low_thresh_mrc = ModifiedRamsey_Control_params["hysteresis_low"]
            high_thresh_mrc = ModifiedRamsey_Control_params["hysteresis_high"]

            state_avg_mrc = None
            state_hyst_mrc = None
            switches_hyst_mrc = None
            switch_time_ms_mrc = (
                elapsed_ms_mrc[1:] if len(elapsed_ms_mrc) > 1 else np.array([])
            )

            if len(binary_states_mrc) >= 1:
                kernel_mrc = np.ones(window_n_mrc, dtype=float) / window_n_mrc
                state_avg_mrc = np.convolve(
                    binary_states_mrc.astype(float), kernel_mrc, mode="same"
                )

                state_hyst_mrc = np.empty_like(binary_states_mrc)
                current_state_mrc = int(binary_states_mrc[0])
                for idx_mrc, val_mrc in enumerate(state_avg_mrc):
                    if val_mrc >= high_thresh_mrc:
                        current_state_mrc = 1
                    elif val_mrc <= low_thresh_mrc:
                        current_state_mrc = 0
                    state_hyst_mrc[idx_mrc] = current_state_mrc

                switches_hyst_mrc = (np.diff(state_hyst_mrc) != 0).astype(int)

                plt.figure(figsize=(10, 4))
                plt.plot(
                    elapsed_ms_mrc, state_avg_mrc, linewidth=1.5, label="Moving average"
                )
                plt.axhline(
                    high_thresh_mrc,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"High threshold = {high_thresh_mrc:.2f}",
                )
                plt.axhline(
                    low_thresh_mrc,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Low threshold = {low_thresh_mrc:.2f}",
                )
                plt.xlabel("Time since start (ms)")
                plt.ylabel("Smoothed parity state")
                plt.ylim(-0.05, 1.05)
                plt.title(
                    f"Cycle {cycle_idx_mrc + 1}: MRC moving-average parity trace\n"
                    f"window={window_n_mrc * dt_ms_mrc:.3f} ms, tau={tau_mrc:.4f} us"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    base_mrc + "_state_moving_avg.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

                plt.figure(figsize=(10, 4))
                plt.step(
                    elapsed_ms_mrc,
                    state_hyst_mrc,
                    where="post",
                    linewidth=1.5,
                    label="Hysteresis state",
                )
                plt.plot(
                    elapsed_ms_mrc,
                    binary_states_mrc,
                    "o",
                    markersize=2,
                    alpha=0.35,
                    label="Raw binary state",
                )
                plt.xlabel("Time since start (ms)")
                plt.ylabel("Parity state")
                plt.yticks([0, 1])
                plt.ylim(-0.1, 1.1)
                plt.title(
                    f"Cycle {cycle_idx_mrc + 1}: MRC hysteresis parity trace\n"
                    f"low={low_thresh_mrc:.2f}, high={high_thresh_mrc:.2f}, "
                    f"window={window_n_mrc * dt_ms_mrc:.3f} ms"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    base_mrc + "_state_hysteresis.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

                if len(switches_hyst_mrc) >= 1:
                    plt.figure(figsize=(10, 4))
                    plt.step(
                        switch_time_ms_mrc,
                        switches_hyst_mrc,
                        where="post",
                        linewidth=1.5,
                    )
                    plt.plot(switch_time_ms_mrc, switches_hyst_mrc, "o", markersize=2)
                    plt.xlabel("Time since start (ms)")
                    plt.ylabel("Jump detected")
                    plt.yticks([0, 1])
                    plt.ylim(-0.1, 1.1)
                    plt.title(
                        f"Cycle {cycle_idx_mrc + 1}: MRC jumps from hysteresis state\n"
                        f"low={low_thresh_mrc:.2f}, high={high_thresh_mrc:.2f}"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        base_mrc + "_state_hysteresis_jumps.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

            np.savez(
                base_mrc + ".npz",
                elapsed_ms=np.array(elapsed_ms_mrc),
                raw_i=np.array(raw_i_mrc),
                raw_q=np.array(raw_q_mrc),
                scores=np.array(scores_mrc),
                binary_states=np.array(binary_states_mrc),
                state_avg=np.array(state_avg_mrc)
                if state_avg_mrc is not None
                else np.array([]),
                state_hysteresis=np.array(state_hyst_mrc)
                if state_hyst_mrc is not None
                else np.array([]),
                hysteresis_switches=np.array(switches_hyst_mrc)
                if switches_hyst_mrc is not None
                else np.array([]),
                switch_time_ms=np.array(switch_time_ms_mrc),
                centers=np.array([c0_mrc, c1_mrc]),
                midpoint=np.array(midpoint_mrc),
                normal=np.array(normal_mrc),
                V_sweet=np.array(current_voltage_mrc),
                S_sweet_mhz=np.array(S_sweet_mrc),
                f_ge=np.array(f_ge_mrc),
                tau_us=np.array(tau_mrc),
                half_period_v=np.array(half_period_v_mrc),
                cycle_idx=np.array(cycle_idx_mrc),
                scheduled_rep_period_us=np.array(rep_period_us_mrc),
                scheduled_timing=np.array(
                    config_mrc["modified_ramsey_timing"], dtype=object
                ),
                streamer_elapsed_s=np.array(
                    data_mrc["data"].get("streamer_elapsed_s", np.nan)
                ),
                streamer_average_rep_period_us=np.array(
                    data_mrc["data"].get(
                        "streamer_average_rep_period_us", np.nan
                    )
                ),
                hysteresis_low=np.array(low_thresh_mrc),
                hysteresis_high=np.array(high_thresh_mrc),
                moving_avg_window_n=np.array(window_n_mrc),
                moving_avg_window_ms=np.array(window_n_mrc * dt_ms_mrc),
                config=np.array(config_mrc, dtype=object),
            )

            with open(base_mrc + "_config.json", "w") as fh:
                json.dump(config_mrc, fh, indent=2, default=float)

            cycle_summary_mrc.append(
                {
                    "cycle_idx": cycle_idx_mrc,
                    "success": True,
                    "V_sweet": current_voltage_mrc,
                    "S_sweet_mhz": S_sweet_mrc,
                    "sweet_verified": True,
                    "f_ge": f_ge_mrc,
                    "tau_us": tau_mrc,
                    "scheduled_rep_period_us": rep_period_us_mrc,
                }
            )

        else:
            print(
                f"[MRC] Cycle {cycle_idx_mrc + 1}: sweet spot not verified "
                f"(S_sweet={S_sweet_mrc:.4f} MHz >= threshold {sweet_max_df_mrc} MHz). "
                f"Skipping Ramsey run."
            )
            cycle_summary_mrc.append(
                {
                    "cycle_idx": cycle_idx_mrc,
                    "success": False,
                    "stage_failed": "sweet_spot_not_verified",
                    "V_sweet": current_voltage_mrc,
                    "S_sweet_mhz": S_sweet_mrc,
                    "sweet_verified": False,
                }
            )

    summary_path_mrc = os.path.join(
        save_dir_mrc,
        f"CycleSummary_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json",
    )
    with open(summary_path_mrc, "w") as fh:
        json.dump(cycle_summary_mrc, fh, indent=2, default=float)

if RunChargeDispersionQuasiCW:
    config["reps"] = ChargeDispersion_params["repetitions"]
    config["rounds"] = 1
    config["Gauss"] = Spec_relevant_params["Gauss"]
    config["relax_delay"] = ChargeDispersion_params["relax_delay"]

    if config["Gauss"]:
        config["sigma"] = Spec_relevant_params["sigma"]
        config["qubit_gain"] = Spec_relevant_params["gain"]

    # one-frequency mode
    config["SpecNumPoints"] = 1
    config["SpecSpan"] = 0
    config["step"] = 0
    config["start"] = ChargeDispersion_params["probe_freq"]
    config["expts"] = 1

    Instance_specSlice = ChargeDispersionQuasiCW(
        path="ChargeDispersion",
        cfg=config,
        soc=soc,
        soccfg=soccfg,
        outerFolder=outerFolder,
    )

    data_specSlice = Instance_specSlice.acquire(load_pulses=True, print_time=True)

    x_pts = np.array(data_specSlice["data"]["x_pts"])

    # repetition-resolved IQ data
    raw_i = np.ravel(np.array(data_specSlice["data"]["raw_i"]))
    raw_q = np.ravel(np.array(data_specSlice["data"]["raw_q"]))

    # stack IQ points
    iq = np.column_stack([raw_i, raw_q])

    # ---------------------------
    # Straight-line blob separation using 2-cluster fit
    # ---------------------------
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=20)
    labels = kmeans.fit_predict(iq)
    centers = kmeans.cluster_centers_

    c0 = centers[0]
    c1 = centers[1]

    # normal vector to separating line = line connecting the centers
    normal = c1 - c0
    midpoint = 0.5 * (c0 + c1)

    # signed distance from perpendicular bisector
    scores = (iq - midpoint) @ normal

    # binary labels from side of line
    binary_amps = (scores > 0).astype(int)

    # optional relabel so state 1 is the less populated blob
    n0 = np.sum(binary_amps == 0)
    n1 = np.sum(binary_amps == 1)
    if n1 > n0:
        binary_amps = 1 - binary_amps
        labels = 1 - labels
        c0, c1 = c1, c0
        normal = c1 - c0
        midpoint = 0.5 * (c0 + c1)
        scores = (iq - midpoint) @ normal

    # amplitude for reference only
    amps = np.abs(raw_i + 1j * raw_q) ** 2

    # approximate time axis per repetition
    rep_period_us = 1.0 + 0.5 + config["readout_length"] + 10 + config["relax_delay"]
    elapsed_s = np.arange(len(raw_i)) * rep_period_us * 1e-6

    save_dir = os.path.join(outerFolder, "ChargeDispersion")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base = os.path.join(save_dir, f"ChargeDispersionQuasiCW_{timestamp}")

    # line for plotting: normal dot (x - midpoint) = 0
    # => n_x*(I - I_mid) + n_y*(Q - Q_mid) = 0
    # => Q = Q_mid - (n_x/n_y)*(I - I_mid), unless n_y ~ 0
    I_min, I_max = raw_i.min(), raw_i.max()
    I_pad = 0.05 * (I_max - I_min if I_max > I_min else 1.0)
    I_line = np.linspace(I_min - I_pad, I_max + I_pad, 400)

    vertical_line = np.abs(normal[1]) < 1e-12
    if not vertical_line:
        Q_line = midpoint[1] - (normal[0] / normal[1]) * (I_line - midpoint[0])

    # ---------------------------
    # Plot 1: raw IQ blobs with separating line
    # ---------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(raw_i, raw_q, ".", alpha=0.35, label="IQ data")
    plt.plot(c0[0], c0[1], "o", markersize=10, label="Blob center 0")
    plt.plot(c1[0], c1[1], "o", markersize=10, label="Blob center 1")

    if vertical_line:
        plt.axvline(midpoint[0], linestyle="--", linewidth=2, label="Separator")
    else:
        plt.plot(I_line, Q_line, "--", linewidth=2, label="Separator")

    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("ChargeDispersionQuasiCW IQ blobs with straight-line separator")

    i_min, i_max = raw_i.min(), raw_i.max()
    q_min, q_max = raw_q.min(), raw_q.max()
    i_pad = 0.05 * (i_max - i_min if i_max > i_min else 1.0)
    q_pad = 0.05 * (q_max - q_min if q_max > q_min else 1.0)

    plt.xlim(i_min - i_pad, i_max + i_pad)
    plt.ylim(q_min - q_pad, q_max + q_pad)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Plot 2: IQ blobs colored by assigned state
    # ---------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(
        raw_i[binary_amps == 0],
        raw_q[binary_amps == 0],
        ".",
        alpha=0.5,
        label="State 0",
    )
    plt.plot(
        raw_i[binary_amps == 1],
        raw_q[binary_amps == 1],
        ".",
        alpha=0.5,
        label="State 1",
    )
    plt.plot(c0[0], c0[1], "o", markersize=10)
    plt.plot(c1[0], c1[1], "o", markersize=10)

    if vertical_line:
        plt.axvline(midpoint[0], linestyle="--", linewidth=2, label="Separator")
    else:
        plt.plot(I_line, Q_line, "--", linewidth=2, label="Separator")

    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("IQ blobs colored by straight-line separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_iq_blobs_labeled.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # ---------------------------
    # Plot 3: projection score histogram
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins=100, alpha=0.7)
    plt.axvline(0.0, linestyle="--", linewidth=2, label="Decision boundary")
    plt.xlabel("Signed distance from separator")
    plt.ylabel("Counts")
    plt.title("Separator score histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + "_separator_score_hist.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # ---------------------------
    # Plot 4: binary parity trace over time
    # ---------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(elapsed_s, binary_states, "-", linewidth=1.5)
    plt.xlabel("Time since start (s)")
    plt.ylabel("Assigned state")
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.1)
    plt.title(f"Cycle {cycle_idx + 1}: QuasiCW binary trace")
    plt.tight_layout()
    plt.savefig(base + "_binary.png", dpi=300, bbox_inches="tight")
    plt.close()

    np.savez(
        base + ".npz",
        elapsed_s=np.array(elapsed_s),
        raw_i=np.array(raw_i),
        raw_q=np.array(raw_q),
        amps=np.array(amps),
        scores=np.array(scores),
        binary_amps=np.array(binary_amps),
        centers=np.array([c0, c1]),
        midpoint=np.array(midpoint),
        normal=np.array(normal),
        x_pts=np.array(x_pts),
        config=np.array(config, dtype=object),
    )

    with open(base + "_config.json", "w") as f:
        json.dump(config, f, indent=2, default=float)


if RunChargeDispersionRamsey:
    repetitions = T1T2_params["repetitions"]
    for i in range(repetitions):
        cd_cfg = {
            "df": ChargeDispersion_params["df"],
            "reps": 1,
            "rounds": 1,
            "pi2_gain": pi2_gain,
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
            "f_ge": qubit_frequency_center,
            "relax_delay": ChargeDispersion_params["relax_delay"],
            "Qubit_number": Qubit_Readout,
        }

        config_cd = config | cd_cfg
        iCD = ChargeDispersion(
            path="ChargeDispersion",
            cfg=config_cd,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        dCD = ChargeDispersion.acquire(iCD)
        ChargeDispersion.display(iCD, dCD, plotDisp=True, figNum=5)
        ChargeDispersion.save_data(iCD, dCD)
        ChargeDispersion.save_config(iCD)

# Amplitude Rabi
### note that UpdateConfig will overwrite elements in BaseConfig

if RunAmplitudeRabi:
    number_of_steps = Amplitude_Rabi_params["number_of_steps"]
    step = int(Amplitude_Rabi_params["max_gain"] / number_of_steps)
    ARabi_config = {
        "start": 0,
        "step": step,
        "expts": number_of_steps,
        "reps": Amplitude_Rabi_params["reps"],
        "rounds": Amplitude_Rabi_params["rounds"],
        "sigma": qubit_sigma,
        "f_ge": Amplitude_Rabi_params["qubit_freq"],
        "relax_delay": Amplitude_Rabi_params["relax_delay"],
        "flattop_length": qubit_flattop,
        "Qubit_number": Qubit_Pulse,
    }

    fit = Amplitude_Rabi_params["fit"]

    config = config | ARabi_config
    if qubit_flattop != None:
        ARabi_config = {
            "gain_start": 0,
            "gain_end": Amplitude_Rabi_params["max_gain"],
            "gainNumPoints": number_of_steps,
            "reps": Amplitude_Rabi_params["reps"],
            "rounds": Amplitude_Rabi_params["rounds"],
            "sigma": qubit_sigma,
            "f_ge": Amplitude_Rabi_params["qubit_freq"],
            "relax_delay": 15000,  # us ≈ 3*T1 (T1 ≈ 5 ms)
            "flattop_length": qubit_flattop,
        }
        config = (
            config | ARabi_config
        )  ### note that UpdateConfig will overwrite elements in BaseConfig
        iAmpRabi = AmplitudeRabiFF_N(
            path="AmplitudeRabi",
            cfg=config,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        dAmpRabi = AmplitudeRabiFF_N.acquire(iAmpRabi)
        AmplitudeRabiFF_N.display(iAmpRabi, dAmpRabi, plotDisp=True, figNum=2, fit=fit)
        AmplitudeRabiFF_N.save_data(iAmpRabi, dAmpRabi)
        AmplitudeRabiFF_N.save_config(iAmpRabi)
    else:
        iAmpRabi = AmplitudeRabiFF(
            path="AmplitudeRabi",
            cfg=config,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        dAmpRabi = AmplitudeRabiFF.acquire(iAmpRabi)
        AmplitudeRabiFF.display(iAmpRabi, dAmpRabi, plotDisp=True, figNum=2, fit=fit)
        AmplitudeRabiFF.save_data(iAmpRabi, dAmpRabi)
        AmplitudeRabiFF.save_config(iAmpRabi)

#
if RunT1:
    for i in range(T1T2_params["repetitions"]):
        if T1T2_params["repetitions"] > 1:
            plot_disp = False
        else:
            plot_disp = True
        expt_cfg = {
            "start": 0,
            "step": T1T2_params["T1_step"],
            "expts": T1T2_params["T1_expts"],
            "reps": T1T2_params["T1_reps"],
            "Qubit_number": Qubit_Readout,
            "rounds": T1T2_params["T1_rounds"],
            "pi_gain": qubit_gain,
            "relax_delay": T1T2_params["relax_delay"],
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
            "f_ge": qubit_frequency_center,
        }

        config = (
            config | expt_cfg
        )  ### note that UpdateConfig will overwrite elements in BaseConfig
        iT1 = T1FF(
            path="T1", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
        dT1 = T1FF.acquire(iT1)
        T1FF.display(iT1, dT1, plotDisp=plot_disp, figNum=2)
        T1FF.save_data(iT1, dT1)
        T1FF.save_config(iT1)

        time.sleep(10)
        soc.reset_gens()

if RunT1T2E:
    for i in range(T1T2_params["repetitions"]):
        # match your plotting behavior
        plot_disp = T1T2_params["repetitions"] <= 1

        # -------------------- T1 --------------------
        expt_cfg_T1 = {
            "start": 0,
            "step": T1T2_params["T1_step"],
            "expts": T1T2_params["T1_expts"],
            "reps": T1T2_params["T1_reps"],
            "Qubit_number": Qubit_Readout,
            "rounds": T1T2_params["T1_rounds"],
            "pi_gain": qubit_gain,
            "relax_delay": T1T2_params["relax_delay"],
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
            "f_ge": qubit_frequency_center,
        }

        config_T1 = config | expt_cfg_T1
        iT1 = T1FF(
            path="T1", cfg=config_T1, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
        dT1 = T1FF.acquire(iT1)
        T1FF.display(iT1, dT1, plotDisp=plot_disp, figNum=2)
        T1FF.save_data(iT1, dT1)
        T1FF.save_config(iT1)

        # -------------------- T2E immediately after --------------------
        num_pulses = T2E_params["num_pi_pulses"]

        # compute time step with your hardware quantization
        int_steps = T2E_params["T2_max_us"] // (
            0.00232515 * (num_pulses + 1) * T2E_params["T2_expts"]
        )
        if int_steps == 0:
            print(
                "[T2E] Step size is 0! need to increase total time or decrease experiments"
            )
            break

        expt_cfg_T2E = {
            "start": 0,
            "step": 0.00232515 * (num_pulses + 1) * int_steps,
            "expts": T2E_params["T2_expts"],
            "reps": T2E_params["T2_reps"],
            "rounds": T2E_params["T2_rounds"],
            "pi_gain": qubit_gain,
            "pi2_gain": pi2_gain,
            "relax_delay": T2E_params["relax_delay"],
            "f_ge": qubit_frequency_center + T2E_params["freq_shift"],
            "num_pi_pulses": num_pulses,
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
            "Qubit_number": Qubit_Readout,
        }

        # optional display normalization params
        if T2E_params.get("rotation_angle", False) != False:
            expt_cfg_T2E["rotation_angle"] = T2E_params["rotation_angle"]
            expt_cfg_T2E["min_max"] = T2E_params["min_max"]

        config_T2E = config | expt_cfg_T2E
        iT2E = T2EMUX(
            path="T2E", cfg=config_T2E, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
        dT2E = T2EMUX.acquire(iT2E)
        T2EMUX.display(iT2E, dT2E, plotDisp=plot_disp, figNum=3)
        T2EMUX.save_data(iT2E, dT2E)
        T2EMUX.save_config(iT2E)

        # -------------------- between iterations --------------------
        time.sleep(10)
        soc.reset_gens()

if RunT1T2RT2E:

    def _run_experiment(ExptClass, path, expt_cfg, figNum, plot_disp=False):
        cfg_run = config | expt_cfg
        inst = ExptClass(
            path=path, cfg=cfg_run, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
        data = ExptClass.acquire(inst)
        ExptClass.display(inst, data, plotDisp=plot_disp, figNum=figNum)
        ExptClass.save_data(inst, data)
        ExptClass.save_config(inst)
        return inst, data

    for i in range(T1T2_params["repetitions"]):
        plot_disp = T1T2_params["repetitions"] <= 1

        # -------------------- T1 --------------------
        expt_cfg_T1 = {
            "start": 0,
            "step": T1T2_params["T1_step"],
            "expts": T1T2_params["T1_expts"],
            "reps": T1T2_params["T1_reps"],
            "rounds": T1T2_params["T1_rounds"],
            "Qubit_number": Qubit_Readout,
            "pi_gain": qubit_gain,
            "relax_delay": T1T2_params["relax_delay"],
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
            "f_ge": qubit_frequency_center,
        }
        _run_experiment(T1FF, "T1", expt_cfg_T1, figNum=2, plot_disp=plot_disp)

        # -------------------- T2E --------------------
        num_pulses = T2E_params["num_pi_pulses"]
        int_steps = T2E_params["T2_max_us"] // (
            0.00232515 * (num_pulses + 1) * T2E_params["T2_expts"]
        )
        if int_steps == 0:
            print(
                "[T2E] Step size is 0! need to increase total time or decrease experiments"
            )
            break

        expt_cfg_T2E = {
            "start": 0,
            "step": 0.00232515 * (num_pulses + 1) * int_steps,
            "expts": T2E_params["T2_expts"],
            "reps": T2E_params["T2_reps"],
            "rounds": T2E_params["T2_rounds"],
            "Qubit_number": Qubit_Readout,
            "pi_gain": qubit_gain,
            "pi2_gain": pi2_gain,
            "relax_delay": T2E_params["relax_delay"],
            "f_ge": qubit_frequency_center + T2E_params["freq_shift"],
            "num_pi_pulses": num_pulses,
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
        }
        if T2E_params.get("rotation_angle") is not None:
            expt_cfg_T2E["rotation_angle"] = T2E_params["rotation_angle"]
            expt_cfg_T2E["min_max"] = T2E_params["min_max"]

        _run_experiment(T2EMUX, "T2E", expt_cfg_T2E, figNum=3, plot_disp=plot_disp)

        # -------------------- T2R --------------------
        expt_cfg_T2R = {
            "start": 0,
            "step": T1T2_params["T2_step"],
            "phase_step": soccfg.deg2reg(0 * 360 / 50, gen_ch=2),
            "expts": T1T2_params["T2_expts"],
            "reps": T1T2_params["T2_reps"],
            "rounds": T1T2_params["T2_rounds"],
            "pi_gain": qubit_gain,
            "pi2_gain": pi2_gain,
            "relax_delay": T1T2_params["relax_delay"],
            "f_ge": qubit_frequency_center + T1T2_params["freq_shift"],
            "sigma": qubit_sigma,
            "flattop_length": qubit_flattop,
        }
        _run_experiment(T2R, "T2R", expt_cfg_T2R, figNum=4, plot_disp=plot_disp)

        time.sleep(10)
        soc.reset_gens()


if RunT2:
    T2R_cfg = {
        "start": 0,
        "step": T1T2_params["T2_step"],
        "phase_step": soccfg.deg2reg(0 * 360 / 50, gen_ch=2),
        "expts": T1T2_params["T2_expts"],
        "reps": T1T2_params["T2_reps"],
        "rounds": T1T2_params["T2_rounds"],
        "pi_gain": qubit_gain,
        "pi2_gain": pi2_gain,
        "relax_delay": T1T2_params["relax_delay"],
        "f_ge": qubit_frequency_center + T1T2_params["freq_shift"],
        "sigma": qubit_sigma,
        "flattop_length": qubit_flattop,
    }
    for i in range(T1T2_params["repetitions"]):
        config = (
            config | T2R_cfg
        )  ### note that UpdateConfig will overwrite elements in BaseConfig
        iT2R = T2R(
            path="T2R", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
        dT2R = T2R.acquire(iT2R)
        T2R.display(iT2R, dT2R, plotDisp=False, figNum=2)
        T2R.save_data(iT2R, dT2R)
        T2R.save_config(iT2R)
        time.sleep(10)
        soc.reset_gens()


if RunT2E:
    for i in range(T2E_params["repetitions"]):
        # match T1 behavior: only show plots if single repetition
        if T2E_params["repetitions"] > 1:
            plot_disp = False
        else:
            plot_disp = True

        num_pulses = T2E_params["num_pi_pulses"]

        # compute time step with your hardware quantization
        int_steps = T2E_params["T2_max_us"] // (
            0.00232515 * (num_pulses + 1) * T2E_params["T2_expts"]
        )
        print(
            f"[T2E rep {i}] int_steps={int_steps}, step(us)={0.00232515 * (num_pulses + 1) * int_steps}, expts={T2E_params['T2_expts']}"
        )

        if int_steps == 0:
            print("Step size is 0! need to increase total time or decrease experiments")
            break  # or continue, but breaking is usually safer
        else:
            T2E_cfg = {
                "start": 0,
                "step": 0.00232515 * (num_pulses + 1) * int_steps,
                "expts": T2E_params["T2_expts"],
                "reps": T2E_params["T2_reps"],
                "rounds": T2E_params["T2_rounds"],
                "pi_gain": qubit_gain,
                "pi2_gain": pi2_gain,
                "relax_delay": T2E_params["relax_delay"],
                "f_ge": qubit_frequency_center + T2E_params["freq_shift"],
                "num_pi_pulses": num_pulses,
                "sigma": qubit_sigma,
                "flattop_length": qubit_flattop,
            }

            # optional display normalization params
            if T2E_params["rotation_angle"] != False:
                T2E_cfg["rotation_angle"] = T2E_params["rotation_angle"]
                T2E_cfg["min_max"] = T2E_params["min_max"]

            config_run = config | T2E_cfg

            # new instance each repetition (like T1)
            iT2E = T2EMUX(
                path="T2E",
                cfg=config_run,
                soc=soc,
                soccfg=soccfg,
                outerFolder=outerFolder,
            )

            dT2E = T2EMUX.acquire(iT2E)
            T2EMUX.display(iT2E, dT2E, plotDisp=plot_disp, figNum=2)
            T2EMUX.save_data(iT2E, dT2E)
            T2EMUX.save_config(iT2E)

            time.sleep(10)
            soc.reset_gens()


def sanity_dump(cfg, tag=""):
    keys = [
        "pulse_freq",
        "qubit_freq",
        "SpecSpan",
        "SpecNumPoints",
        "step",
        "start",
        "qubit_pulse_style",
        "qubit_length",
        "qubit_gain",
        "pulse_gain",
        "readout_length",
        "cavity_min",
    ]
    print(f"\n--- Sanity {tag} ---")
    for k in keys:
        if k in cfg:
            print(f"{k:>18}: {cfg[k]}")
    # If BaseConfig stores LOs/IFs/NCOs, print them too:
    for k in ["cavity_LO", "qubit_LO", "cavity_IF", "qubit_IF", "read_lo", "drive_lo"]:
        if k in cfg:
            print(f"{k:>18}: {cfg[k]}")
    print("-------------------\n")


if RunTrans_QubitSpec:
    sanity_dump(config)
    for i in range(T1T2_params["repetitions"]):
        config["reps"] = Spec_relevant_params[
            "reps"
        ]  # want more reps and rounds for qubit data
        config["rounds"] = Spec_relevant_params["rounds"]
        config["Gauss"] = Spec_relevant_params["Gauss"]
        if Spec_relevant_params["Gauss"]:
            config["sigma"] = Spec_relevant_params["sigma"]
            config["qubit_gain"] = Spec_relevant_params["gain"]
        Instance_specSlice = QubitSpecSliceFF(
            path="QubitSpecFF",
            cfg=config,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        data_specSlice = QubitSpecSliceFF.acquire(Instance_specSlice)
        QubitSpecSliceFF.display(
            Instance_specSlice,
            data_specSlice,
            plotDisp=False,
            figNum=2,
            min_sep=Spec_relevant_params["min_sep_MHz"],
            fit_window_mhz=Spec_relevant_params["fit_window_mhz"],
            prominent_ratio=Spec_relevant_params["prominent_ratio"],
        )
        QubitSpecSliceFF.save_data(Instance_specSlice, data_specSlice)
        QubitSpecSliceFF.save_config(Instance_specSlice)

if RunChargeSweep:
    sanity_dump(config)
    config["reps"] = Spec_relevant_params["reps"]
    config["rounds"] = Spec_relevant_params["rounds"]
    config["Gauss"] = Spec_relevant_params["Gauss"]

    if Spec_relevant_params["Gauss"]:
        config["sigma"] = Spec_relevant_params["sigma"]
        config["qubit_gain"] = Spec_relevant_params["gain"]

    voltage_pts = np.arange(
        config["voltage_start"], config["voltage_end"], config["voltage_step"]
    )
    # will initialize after first acquisition, once x_pts length is known
    avgamp_map = None
    fig = None
    ax = None
    im = None
    cbar = None
    for i, voltage in enumerate(voltage_pts):
        if voltage > 10:
            print("voltage too high")
            break
        if not yoko_fixed:
            ramp_to(yoko, voltage)
        config["current_voltage"] = float(voltage)
        Instance_specSlice = QubitSpecSliceFF(
            path="QubitSpecFF",
            cfg=config,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        data_specSlice = QubitSpecSliceFF.acquire(Instance_specSlice)
        QubitSpecSliceFF.display(
            Instance_specSlice,
            data_specSlice,
            plotDisp=False,
            figNum=2,
            min_sep=Spec_relevant_params["min_sep_MHz"],
            fit_window_mhz=Spec_relevant_params["fit_window_mhz"],
            prominent_ratio=Spec_relevant_params["prominent_ratio"],
        )
        QubitSpecSliceFF.save_data(Instance_specSlice, data_specSlice)
        QubitSpecSliceFF.save_config(Instance_specSlice)
        x_pts = np.array(data_specSlice["data"]["x_pts"])
        avgi = np.array(data_specSlice["data"]["avgi"][0][0])
        avgq = np.array(data_specSlice["data"]["avgq"][0][0])
        # Rotate onto the signal-bearing IQ axis (background-subtracted) instead of
        # the raw magnitude |I+iQ|^2: the magnitude is dominated by the large, noisy
        # background quadrature, which both buries per-slice peaks and washes out the
        # charge-sweep heatmap. The projection gives each slice a ~0 baseline with the
        # qubit feature as a positive bump.
        avgamp0 = project_iq_signal(avgi, avgq)
        # find up to two peaks with a minimum spacing in x-units
        min_spacing = 0.05  # same units as x_pts
        dx = np.mean(np.diff(x_pts))
        min_distance_pts = max(1, int(np.ceil(min_spacing / dx)))

        peak_inds, _ = find_peaks(avgamp0, distance=min_distance_pts)

        # keep the two tallest peaks
        if len(peak_inds) > 0:
            peak_inds = peak_inds[np.argsort(avgamp0[peak_inds])[::-1][:2]]
            peak_inds = peak_inds[
                np.argsort(x_pts[peak_inds])
            ]  # optional: sort by frequency
            peak_freqs = x_pts[peak_inds]
            peak_vals = avgamp0[peak_inds]
        else:
            peak_freqs = np.array([])
            peak_vals = np.array([])

        print("peak freqs:", peak_freqs)
        # initialize plotting objects once we know x axis length
        if avgamp_map is None:
            avgamp_map = np.full((len(voltage_pts), len(x_pts)), np.nan)
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            x_step = x_pts[1] - x_pts[0] if len(x_pts) > 1 else 1
            y_step = voltage_pts[1] - voltage_pts[0] if len(voltage_pts) > 1 else 1

            im = ax.imshow(
                avgamp_map,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=[
                    x_pts[0] - x_step / 2,
                    x_pts[-1] + x_step / 2,
                    voltage_pts[0] - y_step / 2,
                    voltage_pts[-1] + y_step / 2,
                ],
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("rotated IQ projection")

            ax.set_xlabel("Frequency")
            ax.set_ylabel("Voltage")
            ax.set_title("Charge sweep for Qubit Spec")

            plt.show(block=False)
        # store this voltage slice
        avgamp_map[i, :] = avgamp0
        # update heatmap
        im.set_data(avgamp_map)
        # optional: rescale color as data comes in
        im.set_clim(np.nanmin(avgamp_map), np.nanmax(avgamp_map))
        ax.set_title(f"\nVoltage = {voltage:.4f}")
        fig.canvas.draw_idle()
        plt.pause(0.1)

    fig.savefig(
        Instance_specSlice.iname[:-4] + "_ChargeSweepHeatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    np.savez(
        Instance_specSlice.fname[:-3] + "_ChargeSweepHeatmap.npz",
        avgamp_map=avgamp_map,
        x_pts=x_pts,
        voltage_pts=voltage_pts,
    )

    # Build and save the charge-dispersion curve (qubit frequency vs gate
    # voltage) from the projected heatmap: per-voltage Lorentzian peak fit with
    # argmax fallback, plus the heatmap+overlay figure. Writes
    # <stem>_ChargeDispersionCurve.png/.npz next to the heatmap.
    analyze_charge_dispersion(
        avgamp_map=avgamp_map,
        x_pts=x_pts,
        voltage_pts=voltage_pts,
        save_base=Instance_specSlice.fname[:-3] + "_",
        plotDisp=True,
    )

    plt.ioff()
    plt.close(fig)

#######################################################
qubit_gains = [
    Qubit_Parameters[str(Q_R)]["Qubit"]["Gain"] for Q_R in SS_params["Qubit_Pulse"]
]
qubit_frequency_centers = [
    Qubit_Parameters[str(Q_R)]["Qubit"]["Frequency"] for Q_R in SS_params["Qubit_Pulse"]
]


UpdateConfig = {
    ###### cavity
    # "pulse_freq": resonator_frequency_center,  # [MHz] actual frequency is this number + "cavity_LO"
    "read_pulse_style": "const",  # --Fixed
    # Match the active-reset calibration window and keep the resonator on from
    # tone start through ADC offset + the complete integration window.
    "readout_length": SS_params["Readout_Time"],  # us – ADC integration window
    "length": SS_params["ADC_Offset"] + SS_params["Readout_Time"],
    "adc_trig_offset": SS_params["ADC_Offset"],
    "pi2_SS": SS_params["pi2_SS"],
    # "pulse_gain": cavity_gain, # [DAC units]
    "pulse_gain": cavity_gain,  # [DAC units]
    "pulse_freq": resonator_frequency_center,  # [MHz] actual frequency is this number + "cavity_LO"
    ##### qubit spec parameters
    "qubit_pulse_style": "arb",
    "sigma": qubit_sigma,  ### units us, define a 20ns sigma
    "qubit_gain": qubit_gain,
    "f_ge": qubit_frequency_center,
    "qubit_gains": qubit_gains,
    "f_ges": qubit_frequency_centers,
    ##### define shots
    "shots": SS_params["Shots"],  ### this gets turned into "reps"
    "relax_delay": SS_params["relax_delay"],  # us
    "flattop_length": qubit_flattop,
}

config = BaseConfig | UpdateConfig
config["FF_Qubits"] = FF_Qubits
config["Read_Indeces"] = Qubit_Readout


if SingleShot:
    config["number_of_pulses"] = SS_params["number_of_pulses"]
    if SS_params["pi2_SS"]:
        config["qubit_gain"] = pi2_gain
    Instance_SingleShotProgram = SingleShotProgramFFMUX(
        path="SingleShot", outerFolder=outerFolder, cfg=config, soc=soc, soccfg=soccfg
    )
    data_SingleShotProgram = SingleShotProgramFFMUX.acquire(Instance_SingleShotProgram)
    # print(data_SingleShotProgram)
    SingleShotProgramFFMUX.display(
        Instance_SingleShotProgram, data_SingleShotProgram, plotDisp=True
    )

    SingleShotProgramFFMUX.save_data(Instance_SingleShotProgram, data_SingleShotProgram)
    SingleShotProgramFFMUX.save_config(Instance_SingleShotProgram)
    print("Angle: ", data_SingleShotProgram["data"]["angle"][0])
    print("threshold: ", data_SingleShotProgram["data"]["threshold"][0])

if RunT1SS:
    for i in range(T1SS_params["repetitions"]):
        if T1SS_params["calibrate_SS"]:
            config["number_of_pulses"] = SS_params["number_of_pulses"]
            Instance_SingleShotProgram = SingleShotProgramFFMUX(
                path="SingleShot",
                outerFolder=outerFolder,
                cfg=config,
                soc=soc,
                soccfg=soccfg,
            )
            data_SingleShotProgram = SingleShotProgramFFMUX.acquire(
                Instance_SingleShotProgram
            )
            SingleShotProgramFFMUX.display(
                Instance_SingleShotProgram, data_SingleShotProgram, plotDisp=False
            )
            SingleShotProgramFFMUX.save_data(
                Instance_SingleShotProgram, data_SingleShotProgram
            )
            SingleShotProgramFFMUX.save_config(Instance_SingleShotProgram)
            angle = data_SingleShotProgram["data"]["angle"][0]
            threshold = data_SingleShotProgram["data"]["threshold"][0]
        else:
            angle = T1SS_params["angle"]
            threshold = T1SS_params["threshold"]
        print(angle, threshold)

        expt_cfg = {
            "start": 0,
            "step": T1SS_params["T1_step"],
            "expts": T1SS_params["T1_expts"],
            "reps": T1SS_params["reps"],
            "pi_gain": qubit_gain,
            "relax_delay": T1SS_params["relax_delay"],
        }
        config = (
            config | expt_cfg
        )  ### note that UpdateConfig will overwrite elements in BaseConfig
        iT1 = T1_SS(
            path="T1SS", cfg=config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
        )
        dT1 = T1_SS.acquire(iT1, angle=angle, threshold=threshold)
        T1_SS.display(iT1, dT1, plotDisp=False, figNum=2)
        T1_SS.save_data(iT1, dT1)
        T1_SS.save_config(iT1)

        time.sleep(10)
        soc.reset_gens()


if SingleShot_ReadoutOptimize:
    span = SS_R_params["span"]
    cav_gain_start = SS_R_params["gain_start"]
    cav_gain_stop = SS_R_params["gain_stop"]
    cav_gain_pts = SS_R_params["gain_pts"]
    cav_trans_pts = SS_R_params["trans_pts"]
    config["number_of_pulses"] = 1
    exp_parameters = {
        ###### cavity
        "cav_gain_Start": cav_gain_start,
        "cav_gain_Stop": cav_gain_stop,
        "cav_gain_Points": cav_gain_pts,
        "trans_freq_start": config["pulse_freq"] - span / 2,
        "trans_freq_stop": config["pulse_freq"] + span / 2,
        "TransNumPoints": cav_trans_pts,
    }
    config = config | exp_parameters
    # Now lets optimize powers and readout frequencies
    Instance_SingleShotOptimize = ReadOpt_wSingleShotFF(
        path="SingleShot_OptReadout",
        outerFolder=outerFolder,
        cfg=config,
        soc=soc,
        soccfg=soccfg,
    )
    data_SingleShotProgramOptimize = ReadOpt_wSingleShotFF.acquire(
        Instance_SingleShotOptimize
    )
    # print(data_SingleShotProgram)
    ReadOpt_wSingleShotFF.display(
        Instance_SingleShotOptimize, data_SingleShotProgramOptimize, plotDisp=True
    )

    ReadOpt_wSingleShotFF.save_data(
        Instance_SingleShotOptimize, data_SingleShotProgramOptimize
    )
    ReadOpt_wSingleShotFF.save_config(Instance_SingleShotOptimize)


if SingleShot_QubitOptimize:
    q_gain_span = SS_Q_params["q_gain_span"]
    q_gain_pts = SS_Q_params["q_gain_pts"]
    q_freq_pts = SS_Q_params["q_freq_pts"]
    q_freq_span = SS_Q_params["q_freq_span"]
    config["number_of_pulses"] = SS_Q_params["number_of_pulses"]
    if config["pi2_SS"]:
        config["qubit_gain"] = pi2_gain
    Qubit_Pulse_Index = 0
    exp_parameters = {
        ###### cavity
        "qubit_gain_Start": max(
            [0, int(config["qubit_gain"] - int(q_gain_span))]
        ),  # - q_gain_span / 2,
        "qubit_gain_Stop": min(
            [32767, int(config["qubit_gain"] + int(q_gain_span))]
        ),  # *qubit_gains[Qubit_Pulse_Index] + q_gain_span / 2,
        "qubit_gain_Points": q_gain_pts,
        "qubit_freq_start": qubit_frequency_centers[Qubit_Pulse_Index] - q_freq_span,
        "qubit_freq_stop": qubit_frequency_centers[Qubit_Pulse_Index] + q_freq_span,
        "QubitNumPoints": q_freq_pts,
        "number_of_pulses": SS_Q_params["number_of_pulses"],
    }
    print(exp_parameters)
    config = config | exp_parameters
    g0 = qubit_gains[Qubit_Pulse_Index]
    f0 = qubit_frequency_centers[Qubit_Pulse_Index]
    # # Now lets optimize powers and readout frequencies
    Instance_SingleShotOptimize = QubitPulseOpt_wSingleShotFF(
        path="SingleShot_OptQubit",
        outerFolder=outerFolder,
        cfg=config,
        soc=soc,
        soccfg=soccfg,
    )
    data_SingleShotProgramOptimize = QubitPulseOpt_wSingleShotFF.acquire(
        Instance_SingleShotOptimize, Qubit_Sweep_Index=Qubit_Pulse_Index
    )
    # print(data_SingleShotProgram)
    QubitPulseOpt_wSingleShotFF.display(
        Instance_SingleShotOptimize, data_SingleShotProgramOptimize, plotDisp=True
    )
    print(exp_parameters)

    QubitPulseOpt_wSingleShotFF.save_data(
        Instance_SingleShotOptimize, data_SingleShotProgramOptimize
    )
    QubitPulseOpt_wSingleShotFF.save_config(Instance_SingleShotOptimize)

# ── Automated T1 / T2 / T2Echo calibration run ──────────────────────────────
if RunAutoCoherence:
    auto_results = run_auto_coherence(
        soc=soc,
        soccfg=soccfg,
        config=config,  # full config dict built above
        outerFolder=outerFolder,
        qubit_readout=Qubit_Readout,
        qubit_params=Qubit_Parameters,
        yoko=yoko,  # set to None if no charge line
        auto_params=AutoCoherence_override_params,
    )
    print("[AutoCoherence] Results:", auto_results)

# ── Zero-span charge-parity switching measurement ───────────────────────────
# Step 1 (optional): park the qubit drive at one parity-doublet peak.
# Step 2 (optional): calibrate a g/e single-shot separator for apriori
#                    classification.
# Step 3: build the ZeroSpanParity cfg from BaseConfig channel routing + the
#         active qubit's tuned readout/drive.
# Step 4: acquire (chunked for long strobe records).
# Step 5: offline analysis -> bits, switch rate, bursts, dwell stats, plots.
if RunZeroSpanParity:
    zsp_outerFolder = outerFolder + "ZeroSpanParity/"
    os.makedirs(zsp_outerFolder, exist_ok=True)

    # ---- Step 1: optional parity-doublet frequency pre-calibration ----------
    if ZSP_RecalibrateParityFreqs:
        spec_cfg = config.copy()
        spec_cfg["reps"] = ZSP_ParitySpec_params["reps"]
        spec_cfg["rounds"] = ZSP_ParitySpec_params["rounds"]
        spec_cfg["relax_delay"] = ZSP_ParitySpec_params["relax_delay"]
        spec_cfg["Gauss"] = ZSP_ParitySpec_params["Gauss"]
        spec_cfg["sigma"] = ZSP_ParitySpec_params["sigma"]
        spec_cfg["qubit_gain"] = ZSP_ParitySpec_params["gain"]
        spec_cfg["qubit_length"] = ZSP_ParitySpec_params["qubit_length"]
        spec_cfg["qubit_pulse_style"] = "const"
        spec_cfg["SpecSpan"] = ZSP_ParitySpec_params["SpecSpan"]
        spec_cfg["SpecNumPoints"] = ZSP_ParitySpec_params["SpecNumPoints"]
        spec_cfg["step"] = 2 * spec_cfg["SpecSpan"] / spec_cfg["SpecNumPoints"]
        spec_cfg["start"] = qubit_frequency_center - spec_cfg["SpecSpan"]
        spec_cfg["expts"] = spec_cfg["SpecNumPoints"]
        spec_cfg.setdefault("current_voltage", start_voltage)

        Instance_paritySpec = QubitSpecSliceFF(
            path="ZeroSpanParity_Spec",
            cfg=spec_cfg,
            soc=soc,
            soccfg=soccfg,
            outerFolder=zsp_outerFolder,
        )
        data_paritySpec = QubitSpecSliceFF.acquire(Instance_paritySpec)
        QubitSpecSliceFF.display(
            Instance_paritySpec,
            data_paritySpec,
            plotDisp=False,
            figNum=2,
            min_sep=ZSP_ParitySpec_params["min_sep_MHz"],
            fit_window_mhz=ZSP_ParitySpec_params["fit_window_mhz"],
            prominent_ratio=ZSP_ParitySpec_params["prominent_ratio"],
        )
        QubitSpecSliceFF.save_data(Instance_paritySpec, data_paritySpec)
        QubitSpecSliceFF.save_config(Instance_paritySpec)

        chosen = pick_parity_drive_freq(
            data_paritySpec, which=ZSP_ParityFreqs_Cached["which_to_park"]
        )
        ZSP_ParityFreqs_Cached["lower_peak_MHz"] = chosen["lower"]
        ZSP_ParityFreqs_Cached["higher_peak_MHz"] = chosen["higher"]
        parity_drive_freq_MHz = chosen["picked"]
        print(
            f"[ZeroSpanParity] parity doublet: lower={chosen['lower']:.6f} "
            f"higher={chosen['higher']:.6f} MHz; parking at "
            f"{ZSP_ParityFreqs_Cached['which_to_park']} "
            f"({parity_drive_freq_MHz:.6f} MHz)"
        )
    else:
        which = ZSP_ParityFreqs_Cached["which_to_park"]
        parity_drive_freq_MHz = (
            ZSP_ParityFreqs_Cached["lower_peak_MHz"]
            if which == "lower"
            else ZSP_ParityFreqs_Cached["higher_peak_MHz"]
        )
        if parity_drive_freq_MHz is None:
            raise RuntimeError(
                "[ZeroSpanParity] No cached parity freq and "
                "ZSP_RecalibrateParityFreqs=False. Populate ZSP_ParityFreqs_Cached "
                "or set ZSP_RecalibrateParityFreqs=True."
            )

    # ---- Step 2: optional g/e separator pre-calibration ---------------------
    if ZSP_RecalibrateSeparator:
        sep = get_apriori_separator_from_singleshot(
            config=config, soc=soc, soccfg=soccfg, outerFolder=zsp_outerFolder
        )
        ZSP_Separator_Cached["g_center"] = sep["g_center"]
        ZSP_Separator_Cached["e_center"] = sep["e_center"]
        ZSP_Separator_Cached["normal"] = sep["normal"]
        ZSP_Separator_Cached["midpoint"] = sep["midpoint"]
    elif ZSP_AnalysisParams["classifier_method"] == "apriori":
        # Spec §5.3 rule 7: with RecalibrateSeparator=False and apriori
        # classification, all four cached fields must be np.ndarray of shape
        # (2,). Validate fail-fast (a copy-pasted list is coerced; wrong shapes
        # are rejected).
        for _k in ("g_center", "e_center", "normal", "midpoint"):
            _v = ZSP_Separator_Cached[_k]
            if _v is None:
                raise RuntimeError(
                    f"[ZeroSpanParity §5.3 rule 7] ZSP_Separator_Cached['{_k}'] "
                    f"is None and classifier_method='apriori'. Populate "
                    f"ZSP_Separator_Cached, set ZSP_RecalibrateSeparator=True, or "
                    f"use classifier_method='kmeans'."
                )
            _arr = np.asarray(_v, dtype=float)
            if _arr.shape != (2,):
                raise RuntimeError(
                    f"[ZeroSpanParity §5.3 rule 7] ZSP_Separator_Cached['{_k}'] "
                    f"has shape {_arr.shape}, expected (2,) — an (I, Q) "
                    f"coordinate."
                )
            ZSP_Separator_Cached[_k] = _arr  # normalize to ndarray

    # ---- Step 3: build the ZeroSpanParity cfg -------------------------------
    zsp_mode_params = (
        ZSP_StrobeParams if ZSP_RunMode == "strobe" else ZSP_DecimatedParams
    )
    zsp_qubit_gain = (
        ZSP_DriveParams["qubit_gain"]
        if ZSP_DriveParams["qubit_gain"] is not None
        else qubit_gain
    )
    zsp_pulse_gain = (
        ZSP_DriveParams["pulse_gain"]
        if ZSP_DriveParams["pulse_gain"] is not None
        else cavity_gain
    )
    zsp_cfg = {
        # Channel routing sourced from BaseConfig (Calib/initialize4Q.py)
        "res_ch": config["res_ch"],
        "qubit_ch": config["qubit_ch"],
        "ro_chs": config["ro_chs"],
        "nqz": config["nqz"],
        "qubit_nqz": config["qubit_nqz"],
        "mixer_freq": config["mixer_freq"],
        # Frequencies (active qubit's tuned readout + picked parity peak)
        "read_pulse_freq": resonator_frequency_center,
        "parity_drive_freq": parity_drive_freq_MHz,
        # Drive
        "qubit_gain": zsp_qubit_gain,
        "pulse_gain": zsp_pulse_gain,
        "res_phase": ZSP_DriveParams["res_phase"],
        # Mode + trigger source
        "mode": ZSP_RunMode,
        "start_src": ZSP_StartSrc,
        # Mode-specific params (read_length, adc_trig_offset, + mode extras)
        **zsp_mode_params,
    }

    # ---- Step 4: run acquisition --------------------------------------------
    zsp_exp = ZeroSpanParity(
        soc=soc,
        soccfg=soccfg,
        path="ZeroSpanParity",
        outerFolder=zsp_outerFolder,
        cfg=zsp_cfg,
    )
    if ZSP_RunMode == "strobe" and ZSP_StrobeParams["n_chunks"] > 1:
        zsp_data = chunked_acquire(
            zsp_exp, n_chunks=ZSP_StrobeParams["n_chunks"], progress=True
        )
        # Stitched arrays replace exp.data so save_data writes the full record.
        zsp_exp.data = {"data": zsp_data}
    else:
        zsp_data = zsp_exp.acquire(progress=True)
    zsp_exp.save_data()
    zsp_exp.save_config()

    # ---- Step 5: offline analysis -------------------------------------------
    zsp_separator = (
        ZSP_Separator_Cached
        if ZSP_AnalysisParams["classifier_method"] == "apriori"
        else None
    )
    analyze_parity_run(
        h5_path=zsp_exp.fname,
        separator=zsp_separator,
        window_us=ZSP_AnalysisParams["window_us"],
        k_sigma=ZSP_AnalysisParams["k_sigma"],
        classifier_method=ZSP_AnalysisParams["classifier_method"],
        step_us=ZSP_AnalysisParams["step_us"],
        min_burst_duration_us=ZSP_AnalysisParams["min_burst_duration_us"],
        analysis_bin_us=ZSP_AnalysisParams["analysis_bin_us"],
        save_plots=ZSP_AnalysisParams["save_plots"],
        out_dir=os.path.dirname(zsp_exp.fname),
    )
    print(f"[ZeroSpanParity] complete. Raw data: {zsp_exp.fname}")

    # --- Validation harness execution ---
    _val_out_dir = (
        zsp_exp.outerFolder
        if hasattr(zsp_exp, "outerFolder")
        else os.path.dirname(zsp_exp.fname)
    )

    if Validate_StaticContrast:
        if ZSP_Separator_Cached.get("g_center") is None:
            raise RuntimeError(
                "Validate_StaticContrast needs a calibrated separator (set RecalibrateSeparator)"
            )
        _f0 = zsp_cfg["read_pulse_freq"]
        _span = StaticContrast_params["freq_span_mhz"]
        _flist = np.linspace(
            _f0 - _span / 2, _f0 + _span / 2, StaticContrast_params["n_points"]
        )
        zsp_exp.cfg["reps_per_chunk"] = StaticContrast_params["reps_per_point"]
        _sc = run_static_contrast(
            zsp_exp, _flist, qubit_gain_on=zsp_cfg["qubit_gain"], out_dir=_val_out_dir
        )
        print(
            f"[stage 1] best read_pulse_freq = {_sc['best_freq']:.4f} MHz  (contrast SNR {_sc['contrast_snr']:.1f})"
        )

    if Validate_ContrastVsQubitFreq:
        _q0 = zsp_cfg["parity_drive_freq"]
        _qspan = ContrastVsQubit_params["qfreq_span_mhz"]
        _qlist = np.linspace(
            _q0 - _qspan / 2, _q0 + _qspan / 2, ContrastVsQubit_params["n_points"]
        )
        _s2 = run_contrast_vs_qubit_freq(zsp_exp, _qlist, out_dir=_val_out_dir)
        print(f"[stage 2] parity peak sep = {_s2['peaks'].get('peak_sep')}")

    if Validate_ModulationCheck:
        _m = run_modulation_check(
            zsp_exp,
            separator=zsp_separator,
            modulation_freq_hz=Modulation_params["modulation_freq_hz"],
            n_periods=Modulation_params["n_periods"],
            out_dir=_val_out_dir,
        )
        print(
            f"[stage 3] modulation corr={_m['correlation']:.2f} depth={_m['modulation_depth']:.2f} "
            f"snr={_m['snr']:.2f}  (gate: proceed only if recovered)"
        )

    if Validate_ControlSuite:
        _pf = {
            "lower": ZSP_ParityFreqs_Cached.get("lower_peak_MHz"),
            "higher": ZSP_ParityFreqs_Cached.get("higher_peak_MHz"),
        }
        _c = run_control_suite(
            zsp_exp,
            separator=zsp_separator,
            variants=tuple(Control_params["variants"]),
            detune_mhz=Control_params["detune_mhz"],
            parity_freqs=_pf,
            out_dir=_val_out_dir,
        )
        print(
            f"[stage 8] controls: {[(k, v.get('separation_snr', v.get('separation_snr_lower'))) for k, v in _c['variants'].items()]}"
        )

    if Validate_EnvironmentSweep:

        def _set_power(_exp, _val):
            _exp.cfg["pulse_gain"] = (
                _val  # NOTE: replace with attenuator/YOKO call for real power sweep
            )
            _exp.prog = type(_exp.prog)(_exp.soccfg, _exp.cfg)

        _e = run_environment_sweep(
            zsp_exp,
            separator=zsp_separator,
            param_name=Environment_params["param_name"],
            param_values=Environment_params["values"],
            set_param=_set_power,
            out_dir=_val_out_dir,
        )
        print(f"[stage 7] swept {Environment_params['param_name']}: {_e['table']}")

    if Build_EvidenceReport:
        _rep = build_evidence_report(
            _val_out_dir, os.path.join(_val_out_dir, "EVIDENCE.md")
        )
        print(f"[stage 9] evidence report written: {_rep}")

# ── Active-reset verification ────────────────────────────────────────────────
if RunActiveResetVerify:
    arv_dir = os.path.join(outerFolder, "ActiveResetVerify")
    os.makedirs(arv_dir, exist_ok=True)

    arv_config = dict(modified_ramsey_base_config)

    # 1) Calibrate readout phase + I-threshold in the Modified-Ramsey readout
    #    regime (rotates arv_config["res_phase"] so
    #    |g>/|e> separate along I, and measures the threshold off the rotated blobs).
    arv_calib = calibrate_active_reset_readout(
        config=arv_config, soc=soc, soccfg=soccfg, outerFolder=outerFolder
    )

    # 2) Base config shared by all four conditions. g_center/e_center are in the
    #    rotated frame (consistent with config["res_phase"] set just above).
    arv_base_cfg = {
        "f_ge": qubit_frequency_center,
        "pi_gain": qubit_gain,
        "sigma": qubit_sigma,
        "flattop_length": qubit_flattop,
        "reps": ActiveResetVerify_params["reps"],
        "rounds": 1,
        "relax_delay": ActiveResetVerify_params["relax_delay"],
        "n_verify_reads": ActiveResetVerify_params["n_verify_reads"],
        "verify_relax_delay": ActiveResetVerify_params["verify_relax_delay"],
        "reset_cycles": ActiveResetVerify_params["reset_cycles"],
        "reset_readout_relax_delay": ActiveResetVerify_params[
            "reset_readout_relax_delay"
        ],
        "post_reset_wait": ActiveResetVerify_params["post_reset_wait"],
        "readout_threshold": arv_calib["readout_threshold"],
        "reset_ground_below_threshold": arv_calib["reset_ground_below_threshold"],
        "g_center": list(arv_calib["g_center"]),
        "e_center": list(arv_calib["e_center"]),
        "Qubit_number": Qubit_Readout,
    }

    # 3) Four conditions: prep |g>/|e>  ×  reset off/on, plus a force-pi
    #    diagnostic: the corrective pi fires UNCONDITIONALLY after the
    #    conditioning readout, measuring the bare post-readout pi fidelity
    #    decoupled from the threshold decision (prep|e>_forcePI should match
    #    prep|g>_resetOFF if the pi is good; prep|g>_forcePI should match
    #    prep|e>_resetOFF).
    arv_conditions = [
        ("prep|g>_resetOFF", False, False, False),
        ("prep|e>_resetOFF", True, False, False),
        ("prep|g>_resetON", False, True, False),
        ("prep|e>_resetON", True, True, False),
        ("prep|e>_forcePI", True, True, True),
        ("prep|g>_forcePI", False, True, True),
    ]
    arv_results = {}
    for arv_label, arv_prep, arv_reset, arv_force in arv_conditions:
        print(f"\n[ActiveResetVerify] Condition: {arv_label}")
        cfg_arv = (
            arv_config
            | arv_base_cfg
            | {
                "prep_excited": arv_prep,
                "use_active_reset": arv_reset,
                "reset_force_pi": arv_force,
            }
        )
        inst_arv = ActiveResetVerify(
            path="ActiveResetVerify",
            cfg=cfg_arv,
            soc=soc,
            soccfg=soccfg,
            outerFolder=outerFolder,
        )
        data_arv = ActiveResetVerify.acquire(inst_arv)
        ActiveResetVerify.display(inst_arv, data_arv, plotDisp=False, figNum=20)
        ActiveResetVerify.save_data(inst_arv, data_arv)
        ActiveResetVerify.save_config(inst_arv)
        arv_results[arv_label] = np.asarray(data_arv["data"]["p_ground"])
        print(
            f"[ActiveResetVerify] {arv_label}: P(|g>) per read = "
            f"{np.array2string(arv_results[arv_label], precision=3)}"
        )

    # 4) Overlay P(|g>) vs read index for all conditions.
    read_idx_arv = np.arange(ActiveResetVerify_params["n_verify_reads"])
    timestamp_arv = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.figure(figsize=(8, 5))
    for arv_label, *_ in arv_conditions:
        plt.plot(
            read_idx_arv, arv_results[arv_label], "o-", linewidth=1.5, label=arv_label
        )
    plt.xlabel("Verification readout index")
    plt.ylabel("P(|g>)")
    plt.ylim(-0.05, 1.05)
    plt.title(
        "Active-reset verification\n"
        f"f_ge={qubit_frequency_center:.4f} MHz, "
        f"reset_cycles={ActiveResetVerify_params['reset_cycles']}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    arv_overlay = os.path.join(
        arv_dir, f"ActiveResetVerify_overlay_{timestamp_arv}.png"
    )
    plt.savefig(arv_overlay, dpi=300, bbox_inches="tight")
    if ActiveResetVerify_params.get("plotDisp", False):
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close()

    np.savez(
        os.path.join(arv_dir, f"ActiveResetVerify_{timestamp_arv}.npz"),
        read_index=read_idx_arv,
        readout_threshold=arv_calib["readout_threshold"],
        res_phase=arv_calib["res_phase"],
        g_center=arv_calib["g_center"],
        e_center=arv_calib["e_center"],
        **{f"p_ground_{lbl}": arv_results[lbl] for lbl, *_ in arv_conditions},
    )

    # 5) Verdict.
    pg_g_off = float(np.mean(arv_results["prep|g>_resetOFF"]))
    pg_e_off = float(np.mean(arv_results["prep|e>_resetOFF"]))
    pg_g_on = float(np.mean(arv_results["prep|g>_resetON"]))
    pg_e_on = float(np.mean(arv_results["prep|e>_resetON"]))
    print("\n[ActiveResetVerify] ===== VERDICT =====")
    print(f"  prep|g> reset OFF : P(|g>)={pg_g_off:.3f}  (thermal baseline)")
    print(f"  prep|e> reset OFF : P(|g>)={pg_e_off:.3f}  (control, should be low)")
    print(f"  prep|g> reset ON  : P(|g>)={pg_g_on:.3f}")
    print(f"  prep|e> reset ON  : P(|g>)={pg_e_on:.3f}  (key proof)")
    pg_e_force = float(np.mean(arv_results["prep|e>_forcePI"][:1]))
    pg_g_force = float(np.mean(arv_results["prep|g>_forcePI"][:1]))
    print(
        f"  prep|e> force-pi  : P(|g>) read0 = {pg_e_force:.3f}  "
        f"(bare post-readout pi fidelity; expect ~ prep|g> baseline)"
    )
    print(
        f"  prep|g> force-pi  : P(|g>) read0 = {pg_g_force:.3f}  "
        f"(expect ~ prep|e> reset-OFF baseline)"
    )
    recovery_arv = pg_e_on - pg_e_off
    print(f"  reset recovery from |e> : dP(|g>) = {recovery_arv:+.3f}")
    if pg_e_on >= 0.9 * pg_g_on and recovery_arv >= 0.3:
        print("  => Active reset is WORKING (recovers |g> from |e>).")
    else:
        print(
            "  => Active reset NOT clearly working; inspect readout_threshold / "
            "res_phase / pi_gain calibration."
        )
    print(f"[ActiveResetVerify] complete. Overlay: {arv_overlay}")

# ramp_to(yoko, 0.0)
# yoko.write(":OUTP OFF")
yoko.close()
###############################################`
