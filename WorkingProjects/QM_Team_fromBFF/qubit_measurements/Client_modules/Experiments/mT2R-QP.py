from qick import *
import matplotlib.pyplot as plt
import numpy as np
from qick.helpers import gauss
from WorkingProjects.QM_Team_OLD.qubit_measurements.Client_modules.CoreLib.Experiment import ExperimentClass
from tqdm.notebook import tqdm


class ParitySwitchDetectorProgram(AveragerProgram):
    """
    Fixed-time Ramsey-like parity detector.

    Sequence per shot:
      1) active reset to |e>
      2) pi/2 pulse at f_01,min
      3) wait fixed tau chosen so f_01,max accrues pi phase relative to f_01,min
      4) second pi/2 pulse at f_01,min
      5) measure in Z

    One program acquisition averages over cfg["reps"] shots and returns one point.
    """

    def initialize(self):
        cfg = self.cfg

        self.q_rp = self.ch_page(cfg["qubit_ch"])
        self.r_phase = self.sreg(cfg["qubit_ch"], "phase")

        self.declare_gen(ch=cfg["res_ch"], nqz=cfg["nqz"])
        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nqz"])

        for ch in cfg["ro_chs"]:
            self.declare_readout(
                ch=ch,
                length=self.us2cycles(cfg["readout_length"]),
                freq=cfg["pulse_freq"],
                gen_ch=cfg["res_ch"]
            )

        f_res = self.freq2reg(
            cfg["pulse_freq"],
            gen_ch=cfg["res_ch"],
            ro_ch=cfg["ro_chs"][0]
        )

        # Here cfg["f_ge"] is used as f_01,min
        f_min = self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])

        # same Gaussian pulse definition style as original T2R
        self.pulse_sigma = self.us2cycles(cfg["sigma"], gen_ch=cfg["qubit_ch"])
        self.pulse_qubit_length = self.us2cycles(4 * cfg["sigma"], gen_ch=cfg["qubit_ch"])
        self.add_gauss(
            ch=cfg["qubit_ch"],
            name="qubit",
            sigma=self.pulse_sigma,
            length=self.pulse_qubit_length
        )

        # phase conventions
        self.phase_x = self.deg2reg(0, gen_ch=cfg["qubit_ch"])
        self.phase_minus_x = self.deg2reg(180, gen_ch=cfg["qubit_ch"])

        # default qubit pulse: pi/2 at f_01,min
        self.set_pulse_registers(
            ch=cfg["qubit_ch"],
            style="arb",
            freq=f_min,
            phase=self.phase_x,
            gain=cfg["pi2_gain"],
            waveform="qubit"
        )

        # readout pulse
        self.set_pulse_registers(
            ch=cfg["res_ch"],
            style="const",
            freq=f_res,
            phase=cfg["res_phase"],
            gain=cfg["pulse_gain"],
            length=self.us2cycles(cfg["length"])
        )

        # If tau_fixed is not given, compute it from parity splitting.
        # If parity_split_MHz = Δf in MHz, then τ(us) = 1/(2Δf)
        if "tau_fixed" in cfg and cfg["tau_fixed"] is not None:
            tau_us = cfg["tau_fixed"]
        else:
            delta_f_MHz = cfg["parity_split_MHz"]
            tau_us = 1.0 / (2.0 * delta_f_MHz)

        self.tau_cycles = self.us2cycles(tau_us)
        self.tau_us = tau_us

        self.readout_threshold = cfg["readout_threshold"]

        self.sync_all(self.us2cycles(0.2))

    def set_qubit_pulse(self, gain, phase_reg):
        f_min = self.freq2reg(self.cfg["f_ge"], gen_ch=self.cfg["qubit_ch"])
        self.set_pulse_registers(
            ch=self.cfg["qubit_ch"],
            style="arb",
            freq=f_min,
            phase=phase_reg,
            gain=gain,
            waveform="qubit"
        )

    def play_pi2_x(self):
        self.set_qubit_pulse(self.cfg["pi2_gain"], self.phase_x)
        self.pulse(ch=self.cfg["qubit_ch"])

    def play_minus_pi2_x(self):
        self.set_qubit_pulse(self.cfg["pi2_gain"], self.phase_minus_x)
        self.pulse(ch=self.cfg["qubit_ch"])

    def play_x(self):
        self.set_qubit_pulse(self.cfg["pi_gain"], self.phase_x)
        self.pulse(ch=self.cfg["qubit_ch"])

    def restore_pi2(self):
        self.set_qubit_pulse(self.cfg["pi2_gain"], self.phase_x)

    def active_reset_to_e(self):
        """
        Measure in Z. If outcome is |g>, apply X. If |e>, do nothing.

        Replace the feedback placeholder with your lab's exact QICK conditional instruction.
        """

        self.measure(
            pulse_ch=self.cfg["res_ch"],
            adcs=self.ro_chs,
            adc_trig_offset=self.us2cycles(self.cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(self.cfg["reset_readout_relax_delay"])
        )

        # ---------------------------------------------------------
        # HARDWARE-DEPENDENT FEEDBACK PLACEHOLDER
        #
        # Pseudocode:
        #   meas = single-shot result from readout
        #   if meas corresponds to |g>:
        #       play_x()
        #
        # Replace with your QICK conditional branch primitive.
        # ---------------------------------------------------------

        self.sync_all(self.us2cycles(self.cfg["post_reset_wait"]))
        self.restore_pi2()

    def body(self):
        # 1) active reset so every shot starts from |e>
        self.active_reset_to_e()

        # 2) first pi/2 pulse at f_01,min
        self.play_pi2_x()

        # 3) fixed wait tau
        self.sync_all()
        self.sync(self.q_rp, self.tau_cycles)

        # 4) second analyzer pi/2 pulse
        #
        # If the mapping comes out reversed on hardware, swap this to play_minus_pi2_x().
        self.play_pi2_x()

        # 5) final readout
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.cfg["res_ch"],
            adcs=self.ro_chs,
            adc_trig_offset=self.us2cycles(self.cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(self.cfg["relax_delay"])
        )


class ParitySwitchDetector(ExperimentClass):
    """
    Each point:
      - run the same detector sequence for cfg["reps"] shots
      - average them into one point

    Then repeat that point cfg["n_points"] times to build a time trace.
    """

    def __init__(self, soc=None, soccfg=None, path='', outerFolder='',
                 prefix='data', cfg=None, config_file=None, progress=None):
        super().__init__(
            soc=soc, soccfg=soccfg, path=path, outerFolder=outerFolder,
            prefix=prefix, cfg=cfg, config_file=config_file, progress=progress
        )

    def acquire(self, progress=False, debug=False):
        n_points = self.cfg["n_points"]

        avgi_pts = []
        avgq_pts = []
        x_pts = np.arange(n_points)

        for _ in tqdm(range(n_points), disable=not progress):
            prog = ParitySwitchDetectorProgram(self.soccfg, self.cfg)

            avgi, avgq = prog.acquire(
                self.soc,
                threshold=None,
                angle=None,
                load_pulses=True,
                readouts_per_experiment=2,   # reset measurement + final measurement
                save_experiments=None,
                start_src="internal",
                progress=False,
                debug=debug
            )

            # final readout is index 1
            avgi_pts.append(avgi[0][1])
            avgq_pts.append(avgq[0][1])

        avgi_pts = np.array(avgi_pts)
        avgq_pts = np.array(avgq_pts)

        data = {
            'config': self.cfg,
            'data': {
                'x_pts': x_pts,
                'avgi': avgi_pts,
                'avgq': avgq_pts
            }
        }
        self.data = data
        return data

    def display(self, data=None, plotDisp=False, figNum=1, **kwargs):
        if data is None:
            data = self.data

        x_pts = data['data']['x_pts']
        avgi = data['data']['avgi']
        avgq = data['data']['avgq']

        while plt.fignum_exists(num=figNum):
            figNum += 1

        fig = plt.figure(figNum)
        plt.plot(x_pts, avgi, 'o-', label="i", color='orange')
        plt.ylabel("a.u.")
        plt.xlabel("Point index")
        plt.legend()
        plt.title(self.titlename)
        plt.savefig(self.iname[:-4] + 'I_Data.png')

        if plotDisp:
            plt.show(block=True)
            plt.pause(0.1)
        else:
            fig.clf(True)
            plt.close(fig)

        fig = plt.figure(figNum + 1)
        plt.plot(x_pts, avgq, 'o-', label="q")
        plt.ylabel("a.u.")
        plt.xlabel("Point index")
        plt.legend()
        plt.title(self.titlename)
        plt.savefig(self.iname[:-4] + 'Q_Data.png')

        if plotDisp:
            plt.show(block=True)
            plt.pause(0.1)
        else:
            fig.clf(True)
            plt.close(fig)

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data['data'])