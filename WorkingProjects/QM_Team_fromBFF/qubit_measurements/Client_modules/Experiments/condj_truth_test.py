"""
condj_truth_test.py  --  Diagnose the jump semantics of the tProc `condj` instruction.

WHY THIS EXISTS
---------------
VerifyActiveReset (TATQ01-BFE.py) showed that hardware active reset HURTS:
P(|g>) prep|g> 0.80 -> 0.40 (reset excites ground qubits) and
P(|g>) prep|e> 0.18 -> 0.30 (reset fails to recover |e>).

That is a CLEAN INVERSION of the feedback decision: the corrective pi fires when
the qubit reads GROUND and is skipped when it reads EXCITED. The active-reset code
(mActiveResetVerify.py / mModifiedRamsey.py active_reset_to_g) is logically correct
ONLY IF `condj(page, ra, op, rb, label)` jumps to `label` when `ra op rb` is TRUE
(the standard assumption -- it is used to "skip the pi when r_read < r_thresh").

If this firmware/qick build instead jumps when the condition is FALSE, every
active-reset decision inverts and reset pumps the qubit toward |e>. This script
determines which it is, with ZERO dependence on a qubit or readout calibration.

HOW IT WORKS  (Pyro-safe: result comes back through the normal di_buf path)
-------------------------------------------------------------------------
A minimal AveragerProgram always triggers the ADC, but only plays the resonator
readout pulse when `condj` does NOT jump:

    self.trigger(adcs)              # ADC integrates every shot
    condj(p, ra, op, rb, "NOPULSE")
    self.pulse(res_ch)             # <- executes ONLY if condj did NOT jump
    label("NOPULSE")
    measure window

So: large |I| => pulse played => condj did NOT jump.
    near-zero |I| => pulse skipped => condj DID jump.

We run it twice with a comparison that is known TRUE and known FALSE, giving the
full truth table and an unambiguous verdict.

USAGE
-----
Edit the CONFIG block below to match the readout you actually use on this fridge
(copy res_ch / ro_chs / pulse_freq / pulse_gain / etc. straight from the `config`
dict in TATQ01-BFE.py), then run this file. It does NOT import TATQ01-BFE.py
(importing that module would launch the whole experiment), so it builds its own
proxy.
"""

import numpy as np
from qick import AveragerProgram
from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.CoreLib.socProxy import (
    makeProxy_RFSOC_147,
)


class CondjProbeProgram(AveragerProgram):
    """
    Conditionally plays the readout pulse based on a `condj` whose truth value is
    known at compile time. The measured |I| reveals whether condj jumped.

    Extra cfg keys:
      cfg["_condj_a"], cfg["_condj_b"] : the two integer operands loaded into
                                         registers compared by condj.
      cfg["_condj_op"]                 : comparison operator string ("<", ">", ...).
    """

    def initialize(self):
        cfg = self.cfg
        cfg["reps"] = int(cfg.get("shots", cfg.get("reps", 200)))

        # Free scratch registers on page 0 for the comparison operands.
        # (page-0 regs 13/14/15 = loop counters, 31 = trigger time; 6/7 are free.)
        self.r_a = 6
        self.r_b = 7

        self.declare_gen(ch=cfg["res_ch"], nqz=cfg["nqz"])
        for ch in cfg["ro_chs"]:
            self.declare_readout(
                ch=ch,
                length=self.us2cycles(cfg["readout_length"]),
                freq=cfg["pulse_freq"],
                gen_ch=cfg["res_ch"],
            )

        f_res = self.freq2reg(
            cfg["pulse_freq"], gen_ch=cfg["res_ch"], ro_ch=cfg["ro_chs"][0]
        )
        self.set_pulse_registers(
            ch=cfg["res_ch"],
            style="const",
            freq=f_res,
            phase=cfg["res_phase"],
            gain=cfg["pulse_gain"],
            length=self.us2cycles(cfg["length"]),
        )
        self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = self.cfg

        # Load the (compile-time-known) operands into scratch registers.
        self.regwi(0, self.r_a, int(cfg["_condj_a"]))
        self.regwi(0, self.r_b, int(cfg["_condj_b"]))

        # Trigger the ADC unconditionally so a readout window always exists.
        self.trigger(
            adcs=self.ro_chs,
            adc_trig_offset=self.us2cycles(cfg["adc_trig_offset"]),
        )

        # condj jumps to NOPULSE_DONE if the firmware treats "a op b" as a jump.
        self.condj(0, self.r_a, cfg["_condj_op"], self.r_b, "NOPULSE_DONE")

        # This pulse executes ONLY when condj did NOT jump.
        self.pulse(ch=cfg["res_ch"])

        self.label("NOPULSE_DONE")
        self.wait_all()
        self.sync_all(self.us2cycles(cfg["relax_delay"]))

    def acquire(self, soc, **kwargs):
        super().acquire(soc, readouts_per_experiment=1, load_pulses=True, **kwargs)
        norm = self.us2cycles(self.cfg["readout_length"], ro_ch=0)
        i = self.di_buf[0][: self.cfg["reps"]] / norm
        q = self.dq_buf[0][: self.cfg["reps"]] / norm
        return float(np.mean(np.abs(i + 1j * q)))


def run_condj_truth_test(soc, soccfg, cfg):
    """
    Returns dict with the measured |IQ| for a TRUE and a FALSE condition and a
    plain-language verdict on condj's jump semantics + what it means for active reset.

    Pass your LIVE `config` from TATQ01-BFE.py as `cfg` (it has the on-resonance
    pulse_freq/pulse_gain and the calibrated res_phase, so the played-vs-skipped
    readout contrast is large). Fast overrides below keep the test short -- it needs
    no thermalisation because it tests pure tProc logic, not the qubit.
    """
    a, b = 5, 10  # a < b is TRUE; a > b is FALSE

    # Don't inherit a 15 ms relax_delay (that would make 2x200 reps take minutes).
    fast = {"shots": 200, "soft_avgs": 1, "relax_delay": 5.0}
    cfg_true = cfg | fast | {"_condj_a": a, "_condj_b": b, "_condj_op": "<"}   # TRUE
    cfg_false = cfg | fast | {"_condj_a": a, "_condj_b": b, "_condj_op": ">"}  # FALSE

    amp_true = CondjProbeProgram(soccfg, cfg_true).acquire(soc)
    amp_false = CondjProbeProgram(soccfg, cfg_false).acquire(soc)

    # "pulse played" == condj did NOT jump. The played branch reads LARGE |IQ|
    # (on-resonance tone), the skipped branch reads SMALL (~background). Decide
    # purely on the RELATIVE difference so we don't need an absolute floor.
    hi, lo = max(amp_true, amp_false), min(amp_true, amp_false)
    rel_diff = (hi - lo) / hi if hi > 0 else 0.0

    if rel_diff < 0.30:
        # Same branch for both. "condj never jumps" is ruled out by the ARV data
        # (if the pi always fired, prep|e> would RISE to ~0.7, not stay ~0.3), so
        # this is a contrast problem, not a real condj result.
        verdict = "INCONCLUSIVE"
        implication = (
            f"amp_true and amp_false differ by only {rel_diff*100:.0f}% -- can't tell\n"
            "    played from skipped. This means the readout is OFF-RESONANCE (or gain~0):\n"
            "    pass your LIVE `config` (on-resonance pulse_freq/pulse_gain), not the\n"
            "    placeholder cfg. The played branch must read clearly larger than skipped."
        )
        jumped_on_true = jumped_on_false = None
    else:
        # Larger amp = pulse PLAYED = condj did NOT jump; smaller = SKIPPED = jumped.
        jumped_on_true = amp_true < amp_false   # TRUE condition skipped the pulse
        jumped_on_false = amp_false < amp_true  # FALSE condition skipped the pulse
        if jumped_on_true and not jumped_on_false:
            verdict = "JUMP-ON-TRUE (standard -- matches the code's assumption)"
            implication = (
                "condj matches what active_reset_to_g() assumes, so the reset inversion is\n"
                "    NOT a condj-sense bug. Look next at reset_ground_below_threshold / the\n"
                "    live read sign, or readout-induced heating. Ask me to add a pi-fired\n"
                "    counter (surfaced via di_buf) to localize it."
            )
        else:
            verdict = "JUMP-ON-FALSE (INVERTED vs the code's assumption)"
            implication = (
                "THIS IS THE BUG. active_reset_to_g() skips the pi when it should fire and\n"
                "    fires when it should skip -> reset pumps toward |e>. Fix: invert the skip\n"
                "    condition (flip reset_skip_op) in mActiveResetVerify.py AND\n"
                "    mModifiedRamsey.py, then re-run VerifyActiveReset."
            )

    def _branch(jumped):
        if jumped is None:
            return "pulse ? (contrast too small)"
        return ("pulse SKIPPED -> condj JUMPED" if jumped
                else "pulse PLAYED -> condj did NOT jump")

    print("\n[condj truth test] ============================================")
    print(f"  |IQ| with condition TRUE  (5<10): {amp_true:.4f}  -> {_branch(jumped_on_true)}")
    print(f"  |IQ| with condition FALSE (5>10): {amp_false:.4f}  -> {_branch(jumped_on_false)}")
    print(f"  relative difference: {rel_diff*100:.0f}%")
    print(f"  VERDICT: condj is {verdict}")
    print(f"  => {implication}")
    print("[condj truth test] ============================================\n")

    return {
        "amp_true": amp_true,
        "amp_false": amp_false,
        "jumped_on_true": jumped_on_true,
        "jumped_on_false": jumped_on_false,
        "verdict": verdict,
    }


# ── HOW TO RUN ───────────────────────────────────────────────────────────────
# The test needs an ON-RESONANCE readout so "pulse played" reads clearly larger
# than "pulse skipped". The easiest correct way is to reuse the LIVE `config` you
# already have after running TATQ01-BFE.py (it has the real pulse_freq/pulse_gain
# and the calibrated res_phase). In your interactive session run:
#
#     from WorkingProjects.QM_Team_fromBFF.qubit_measurements.Client_modules.Experiments.condj_truth_test import run_condj_truth_test
#     run_condj_truth_test(soc, soccfg, config)
#
# (run_condj_truth_test internally overrides relax_delay/shots so it stays fast.)
#
# If you instead run this file standalone, you MUST fill the cfg below with your
# real readout values -- the previous placeholders were off-resonance, which is
# why the test came back INCONCLUSIVE (amp_true ~= amp_false).
if __name__ == "__main__":
    soc, soccfg = makeProxy_RFSOC_147()

    cfg = {
        "res_ch": 0,  # <-- your resonator DAC channel
        "nqz": 2,  # <-- resonator Nyquist zone
        "ro_chs": [0],  # <-- readout ADC channel(s)
        "readout_length": 5.0,  # us  <-- your readout window
        "length": 5.0,  # us  <-- resonator const-pulse length
        "pulse_freq": 6673.27,  # MHz <-- MUST be on-resonance (resonator_frequency_center)
        "pulse_gain": 2500,  # DAC gain (cavity_gain) -- must drive the resonator
        "res_phase": 0,  # reg units (any value is fine for this logic test)
        "adc_trig_offset": 1,  # us  <-- your adc_trig_offset
    }
    if cfg["pulse_freq"] is None or cfg["pulse_gain"] is None:
        raise SystemExit(
            "Fill cfg['pulse_freq']/cfg['pulse_gain'] with on-resonance values, or "
            "better: call run_condj_truth_test(soc, soccfg, config) with your live config."
        )

    run_condj_truth_test(soc, soccfg, cfg)
