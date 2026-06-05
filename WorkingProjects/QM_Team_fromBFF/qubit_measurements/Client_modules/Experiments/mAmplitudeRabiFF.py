from qick import *
import matplotlib.pyplot as plt
import numpy as np
from qick.helpers import gauss
from WorkingProjects.QM_Team_OLD.qubit_measurements.Client_modules.CoreLib.Experiment import ExperimentClass
import datetime
from tqdm.notebook import tqdm
import time


def iq_rotation_angle(signal_trace):
    """
    Angle (rad) of the principal axis of an IQ trajectory.

    Rotating the trace by -angle aligns the direction of maximum variance (the
    qubit g->e displacement axis swept out during the Rabi) with the I
    quadrature, so the oscillation lands in a single quadrature and the other
    stays flat. This is the data-driven alternative to the cavity_winding_*
    phase, which is a frequency-dependent cable-delay correction, not a
    quadrature alignment.
    """
    i = np.real(signal_trace).astype(float)
    q = np.imag(signal_trace).astype(float)
    i = i - np.mean(i)
    q = q - np.mean(q)
    # Closed-form principal-axis angle of the 2x2 covariance matrix.
    return 0.5 * np.arctan2(2.0 * np.mean(i * q), np.mean(i * i) - np.mean(q * q))


class AmplitudeRabiFFProg(RAveragerProgram):
    def initialize(self):
        cfg = self.cfg

        self.q_rp = self.ch_page(self.cfg["qubit_ch"])  # get register page for qubit_ch
        self.r_gain = self.sreg(cfg["qubit_ch"], "gain")  # get gain register for qubit_ch

        self.declare_gen(ch=cfg["res_ch"], nqz=cfg["nqz"])  # Readout
        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nqz"])  # Qubit
        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.us2cycles(cfg["readout_length"]),
                                 freq=cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        f_res = self.freq2reg(cfg["pulse_freq"], gen_ch=cfg["res_ch"], ro_ch=cfg["ro_chs"][0])  # conver f_res to dac register value
        f_ge = self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])

        self.pulse_sigma = self.us2cycles(cfg["sigma"], gen_ch=self.cfg["qubit_ch"])
        self.pulse_qubit_lenth = self.us2cycles(cfg["sigma"] * 4, gen_ch=self.cfg["qubit_ch"])
        print(self.pulse_sigma, self.pulse_qubit_lenth)

        if cfg["flattop_length"] != None:
            print('yes!')
            flattop_length = self.us2cycles(self.cfg["flattop_length"], gen_ch=self.cfg["qubit_ch"])
            print(flattop_length)
            self.add_gauss(ch=cfg["qubit_ch"], name="qubit",
                           sigma=self.pulse_sigma,
                           length=self.pulse_qubit_lenth)
            self.set_pulse_registers(ch=cfg["qubit_ch"], style='flat_top', freq=f_ge,
                                     phase=self.deg2reg(0, gen_ch=cfg["qubit_ch"]), gain=cfg["start"],
                                     waveform="qubit",
                                     length=flattop_length) #Flat part of flattop does NOT update with gain
            # self.set_pulse_registers(ch=cfg["qubit_ch"], style='const', freq=f_ge,
            #                          phase=self.deg2reg(0, gen_ch=cfg["qubit_ch"]), gain=cfg["start"],
            #                          length=flattop_length)
            self.pulse_qubit_lenth += flattop_length
        else:
            self.add_gauss(ch=cfg["qubit_ch"], name="qubit",
                           sigma= self.pulse_sigma, length= self.pulse_qubit_lenth)

            self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=f_ge,
                                     phase=self.deg2reg(90, gen_ch=cfg["qubit_ch"]), gain=cfg["start"],
                                     waveform="qubit")

        self.set_pulse_registers(ch=cfg["res_ch"], style="const", freq=f_res, phase=cfg["res_phase"],
                                 gain=cfg["pulse_gain"],
                                 length=self.us2cycles(cfg["length"]))

        trig_length = cfg["trig_buffer_start"] + cfg["trig_buffer_end"] + cfg["sigma"] * 4

        if cfg["flattop_length"] != None:
            trig_length += self.cfg["flattop_length"]
        self.trig_length = self.us2cycles(trig_length)

        self.sync_all(self.us2cycles(0.05))

    def body(self):
        self.sync_all()
        # self.pulse(ch=self.cfg['ff_ch'])
        self.trigger(pins = [0], t = self.us2cycles(1 + self.cfg["trig_delay"] -
                                                    self.cfg["trig_buffer_start"]), width = self.trig_length)

        self.pulse(ch=self.cfg["qubit_ch"], t=self.us2cycles(1))  # play probe pulse
        self.sync_all(self.us2cycles(0.05))

        self.measure(pulse_ch=self.cfg["res_ch"],
                     adcs=[0, 1],
                     adc_trig_offset=self.us2cycles(self.cfg["adc_trig_offset"]),
                     wait=True,
                     syncdelay=self.us2cycles(10))

        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

    def update(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])  # update gain of the Gaussian

class AmplitudeRabiFF(ExperimentClass):
    """
    Basic AmplitudeRabi
    """

    def __init__(self, soc=None, soccfg=None, path='', outerFolder='', prefix='data', cfg=None, config_file=None, progress=None):
        super().__init__(soc=soc, soccfg=soccfg, path=path, outerFolder=outerFolder, prefix=prefix, cfg=cfg, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):

        #### pull the data from the amp rabi sweep
        # prog = PulseProbeSpectroscopyProgram(self.soccfg, self.cfg)
        prog = AmplitudeRabiFFProg(self.soccfg, self.cfg)

        x_pts, avgi, avgq = prog.acquire(self.soc, threshold=None, angle=None, load_pulses=True,
                                         readouts_per_experiment=1, save_experiments=None,
                                         start_src="internal", progress=False, debug=False)

        signal = (avgi + 1j * avgq) * np.exp(1j * (2 * np.pi * self.cfg['cavity_winding_freq'] *
                                                   self.cfg["pulse_freq"] + self.cfg['cavity_winding_offset']))

        # Data-driven IQ rotation: align the qubit g->e response with the I
        # quadrature so the Rabi oscillation appears in a single quadrature.
        # The angle is computed from the principal axis of the IQ trajectory
        # (the [0][0] readout trace) and applied uniformly to the whole array.
        rot_angle = iq_rotation_angle(np.ravel(signal[0][0]))
        signal = signal * np.exp(-1j * rot_angle)
        print(f'[AmplitudeRabiFF] IQ rotation angle = {np.rad2deg(rot_angle):.2f} deg')

        avgi = signal.real
        avgq = signal.imag
        data = {'config': self.cfg,
                'data': {'x_pts': x_pts, 'avgi': avgi, 'avgq': avgq,
                         'iq_rotation_rad': rot_angle}}
        self.data = data

        return data


    def display(self, data=None, plotDisp = False, figNum = 1, **kwargs):
        if data is None:
            data = self.data

        x_pts = data['data']['x_pts']
        avgi = data['data']['avgi'][0][0]
        avgq = data['data']['avgq'][0][0]
        rot_angle = data['data'].get('iq_rotation_rad', None)

        while plt.fignum_exists(num=figNum): ###account for if figure with number already exists
            figNum += 1
        fig = plt.figure(figNum)
        plt.plot(x_pts, avgi, 'o-', label="i", color = 'orange')
        plt.plot(x_pts, avgq, label="q", color = 'blue')
        plt.ylabel("a.u.")
        plt.xlabel("qubit gain")
        plt.legend()
        if rot_angle is not None:
            plt.title(f'{self.titlename}\nIQ rot = {np.rad2deg(rot_angle):.1f} deg')
        else:
            plt.title(self.titlename)
        plt.savefig(self.iname)

        if plotDisp:
            plt.show(block=True)
            plt.pause(0.1)
        else:
            fig.clf(True)
            plt.close(fig)

        # fig = plt.figure(figNum+1)
        # sig = avgi + 1j * avgq
        # avgamp0 = np.abs(sig)
        # plt.plot(x_pts, avgamp0, label="Amplitude; ADC 0")
        # plt.ylabel("a.u.")
        # plt.xlabel("qubit gain")
        # plt.legend()
        # plt.title(self.titlename)
        #
        # plt.savefig(self.iname)
        #
        # if plotDisp:
        #     plt.show(block=True)
        #     plt.pause(0.1)
        # else:
        #     fig.clf(True)
        #     plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])


