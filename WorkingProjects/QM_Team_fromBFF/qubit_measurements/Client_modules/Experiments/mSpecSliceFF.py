from qick import *
import matplotlib.pyplot as plt
import numpy as np
from qick.helpers import gauss
from WorkingProjects.QM_Team_OLD.qubit_measurements.Client_modules.CoreLib.Experiment import ExperimentClass
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.utils import (
    project_iq_signal,
    find_two_tone_peaks,
)
import datetime
from tqdm.notebook import tqdm
import time


class QubitSpecSliceFFProg(RAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.declare_gen(ch=cfg["res_ch"], nqz=cfg["nqz"])  # Readout
        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nqz"])  # Qubit
        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.us2cycles(cfg["readout_length"]),
                                 freq=cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        self.q_rp = self.ch_page(self.cfg["qubit_ch"])  # get register page for qubit_ch
        self.r_freq = self.sreg(cfg["qubit_ch"], "freq")  # get frequency register for qubit_ch

        ### Start fast flux
        f_res = self.freq2reg(cfg["pulse_freq"], gen_ch=cfg["res_ch"], ro_ch=0)  # conver f_res to dac register value

        self.f_start = self.freq2reg(cfg["start"], gen_ch=cfg["qubit_ch"])  # get start/step frequencies
        self.f_step = self.freq2reg(cfg["step"], gen_ch=cfg["qubit_ch"])


        # add qubit and readout pulses to respective channels
        if cfg['Gauss']:
            self.pulse_sigma = self.us2cycles(cfg["sigma"], gen_ch = self.cfg["qubit_ch"])
            self.pulse_qubit_lenth = self.us2cycles(cfg["sigma"] * 4, gen_ch = self.cfg["qubit_ch"])
            self.add_gauss(ch=cfg["qubit_ch"], name="qubit", sigma= self.pulse_sigma, length= self.pulse_qubit_lenth)
            self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=self.f_start,
                                     phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]), gain=cfg["qubit_gain"],
                                     waveform="qubit")
            self.qubit_length_us = cfg["sigma"] * 4
        else:
            self.set_pulse_registers(ch=cfg["qubit_ch"], style="const", freq=self.f_start, phase=0, gain=cfg["qubit_gain"],
                                     length=self.us2cycles(cfg["qubit_length"]))
            self.qubit_length_us = cfg["qubit_length"]
        self.set_pulse_registers(ch=cfg["res_ch"], style="const", freq=f_res, phase=cfg["res_phase"],
                                 gain=cfg["pulse_gain"],
                                 length=self.us2cycles(cfg["length"]))


    def body(self):
        self.sync_all()
        self.pulse(ch=self.cfg["qubit_ch"], t = self.us2cycles(1))  # play probe pulse
        # trigger measurement, play measurement pulse, wait for qubit to relax
        self.sync_all(self.us2cycles(0.5))
        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=[0, 1],
                     adc_trig_offset=self.us2cycles(self.cfg["adc_trig_offset"]),
                     wait=True,
                     syncdelay=self.us2cycles(10))

        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

    def update(self):
        self.mathi(self.q_rp, self.r_freq, self.r_freq, '+', self.f_step)  # update frequency list index
# ====================================================== #

class QubitSpecSliceFF(ExperimentClass):
    """
    Basic spec experiement that takes a single slice of data
    """

    def __init__(self, soc=None, soccfg=None, path='', outerFolder='', prefix='data', cfg=None, config_file=None, progress=None):
        super().__init__(soc=soc, soccfg=soccfg, path=path, outerFolder=outerFolder, prefix=prefix, cfg=cfg, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):

        prog = QubitSpecSliceFFProg(self.soccfg, self.cfg)
        x_pts, avgi, avgq = prog.acquire(self.soc, threshold=None, angle=None, load_pulses=True,
                                         readouts_per_experiment=1, save_experiments=None,
                                         start_src="internal", progress=False, debug=False)
        signal = (avgi + 1j * avgq) * np.exp(1j * (2 * np.pi * self.cfg['cavity_winding_freq'] *
                                                   self.cfg["pulse_freq"] + self.cfg['cavity_winding_offset']))
        avgi = signal.real
        avgq = signal.imag

        data = {'config': self.cfg, 'data': {'x_pts': x_pts, 'avgi': avgi, 'avgq': avgq}}
        self.data = data

        x_pts = np.asarray(data['data']['x_pts'], dtype=float)

        #### find the frequency corresponding to the qubit feature
        # Rotate onto the signal-bearing axis (background-subtracted) before
        # locating the qubit feature. The raw |I+iQ| magnitude is dominated by
        # the large, noisy background quadrature and locks onto noise spikes
        # rather than the qubit (see utils.project_iq_signal).
        avgi_ch = np.asarray(data['data']['avgi'][0][0], dtype=float)
        avgq_ch = np.asarray(data['data']['avgq'][0][0], dtype=float)
        avgamp0 = project_iq_signal(avgi_ch, avgq_ch)
        peak_loc = int(np.argmax(avgamp0))
        self.qubitFreq = x_pts[peak_loc]

        return data

    def display(self, data=None, plotDisp = False, figNum = 1, **kwargs):
        if data is None:
            data = self.data
        x_pts = np.asarray(data['data']['x_pts'], dtype=float)
        avgi = np.asarray(data['data']['avgi'][0][0], dtype=float)
        avgq = np.asarray(data['data']['avgq'][0][0], dtype=float)

        # Rotate onto the signal-bearing axis (background-subtracted) so the
        # qubit feature stands out as a positive peak instead of being buried in
        # the raw |I+iQ| magnitude (see utils.project_iq_signal).
        avgamp0 = project_iq_signal(avgi, avgq)
        min_sep = kwargs.get('min_sep', 0.1)
        peak_info = find_two_tone_peaks(x_pts, avgamp0, min_sep_mhz=min_sep)

        # Build a title that identifies the drive gain and pulse length of this
        # spec slice (useful when sweeping gain/length). Length is sigma*4 in
        # Gaussian mode, else the explicit qubit_length, both in us.
        _gain = self.cfg.get("qubit_gain", "?")
        if self.cfg.get("Gauss", False):
            _len_txt = f"{self.cfg.get('sigma', 0) * 4:g} us (gauss σ={self.cfg.get('sigma')})"
        else:
            _len_txt = f"{self.cfg.get('qubit_length', '?')} us"
        _v_txt = (f", V={self.cfg['current_voltage']:.6f} V"
                  if "current_voltage" in self.cfg else "")
        spec_title = f"{self.titlename}\ngain={_gain}, qubit_len={_len_txt}{_v_txt}"

        # --- raw I/Q figure (unchanged behaviour) ---
        plt.figure(figNum)
        plt.plot(x_pts, avgi, '.-', color = 'Orange', label="I")
        plt.plot(x_pts, avgq, '.-', color = 'Blue', label="Q")
        plt.ylabel("a.u.")
        plt.xlabel("Qubit Frequency (MHz)")
        plt.title(spec_title)
        plt.legend()

        plt.savefig(self.iname[:-4] + '_IQ.png')
        if plotDisp:
            plt.show(block=True)
            plt.pause(0.1)
        plt.close(figNum)

        # --- rotated-projection figure with detected peaks ---
        plt.figure(figNum)
        plt.plot(x_pts, avgamp0, '.-', color = 'Green', label="rotated IQ projection")
        for k, idx in enumerate(peak_info["peak_inds"]):
            plt.axvline(x_pts[idx], linestyle='--',
                        label=f"Peak {k + 1}: {x_pts[idx]:.6f} MHz")
            plt.plot(x_pts[idx], avgamp0[idx], 'o')
        plt.ylabel("a.u.")
        plt.xlabel("Qubit Frequency (MHz)")
        plt.title(spec_title)
        plt.legend()

        plt.savefig(self.iname[:-4] + '_proj.png')
        if plotDisp:
            plt.show(block=True)
            plt.pause(0.1)
        plt.close(figNum)


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])


