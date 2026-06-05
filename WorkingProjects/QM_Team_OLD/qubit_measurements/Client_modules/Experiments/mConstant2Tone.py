from qick import *
import matplotlib.pyplot as plt
import numpy as np
from qick.helpers import gauss
from WorkingProjects.QM_Team_OLD.qubit_measurements.Client_modules.CoreLib.Experiment import (
    ExperimentClass,
)
import datetime
from tqdm.notebook import tqdm
import time


class ConstantTwoToneProg(AveragerProgram):
    def initialize(self):
        cfg = self.cfg

        cfg["reps"] = 100000
        cfg["rounds"] = 1000

        if "length" not in cfg:
            cfg["length"] = cfg["tone_length"]

        self.declare_gen(
            ch=cfg["res_ch"],
            nqz=cfg["nqz"],
            mixer_freq=cfg.get("mixer_freq", 0),
            ro_ch=cfg["ro_chs"][0],
        )

        print(cfg)

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["qubit_nqz"])

        f_res = self.freq2reg(
            cfg["pulse_freq"], gen_ch=cfg["res_ch"], ro_ch=cfg["ro_chs"][0]
        )

        f_qub = self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])

        res_len = min(self.us2cycles(cfg["length"]), 65535)
        qub_len = min(self.us2cycles(cfg["length"]), 65535)

        print("Qubit Gain for constant two tone: ", cfg["qubit_gain"])
        print("Resonator Gain for constant two tone: ", cfg["pulse_gain"])
        print("Resonator frequency: ", cfg["pulse_freq"], "MHz")
        print("Qubit frequency: ", cfg["f_ge"], "MHz")
        print("Pulse length: ", cfg["length"], "us")

        self.set_pulse_registers(
            ch=cfg["res_ch"],
            style="const",
            freq=f_res,
            phase=0,
            gain=cfg["pulse_gain"],
            length=res_len,
        )

        self.set_pulse_registers(
            ch=cfg["qubit_ch"],
            style="const",
            freq=f_qub,
            phase=0,
            gain=cfg["qubit_gain"],
            length=qub_len,
        )

        self.synci(200)

    def body(self):
        self.pulse(ch=self.cfg["res_ch"])
        self.pulse(ch=self.cfg["qubit_ch"])
        self.sync_all()


class ConstantTwoTone(ExperimentClass):
    """
    Constant two-tone output experiment.
    Drives resonator and qubit channels simultaneously.
    """

    def __init__(
        self,
        soc=None,
        soccfg=None,
        path="",
        outerFolder="",
        prefix="data",
        cfg=None,
        config_file=None,
        progress=None,
    ):
        super().__init__(
            soc=soc,
            soccfg=soccfg,
            path=path,
            outerFolder=outerFolder,
            prefix=prefix,
            cfg=cfg,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False):
        for i in range(100):
            prog = ConstantTwoToneProg(self.soccfg, self.cfg)
            a, b = prog.acquire(self.soc, load_pulses=True)

        data = {
            "config": self.cfg,
            "data": {
                "res_freq_MHz": self.cfg["pulse_freq"],
                "res_gain": self.cfg["pulse_gain"],
                "qubit_freq_MHz": self.cfg["qubit_freq"],
                "qubit_gain": self.cfg["qubit_gain"],
                "length_us": self.cfg["length"],
            },
        }

        print("Started constant two-tone output:")
        print(
            f"  resonator: {self.cfg['pulse_freq']} MHz, gain {self.cfg['pulse_gain']}"
        )
        print(
            f"  qubit:     {self.cfg['qubit_freq']} MHz, gain {self.cfg['qubit_gain']}"
        )
        print(f"  length:    {self.cfg['length']} us")

        return data

    def display(self, data=None, plotDisp=False, figNum=1, **kwargs):
        pass

    def save_data(self, data=None):
        pass
