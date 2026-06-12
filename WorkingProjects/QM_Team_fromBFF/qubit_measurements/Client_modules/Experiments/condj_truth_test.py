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
    Probes BOTH the jump polarity AND the signedness of the condj comparison,
    and returns a dict of measured |IQ| per case + plain-language verdicts.

    Four cases (a op b -> expected truth value):
        A:  5 < 10   TRUE  (signed)   TRUE  (unsigned)   -> polarity reference
        B:  5 > 10   FALSE (signed)   FALSE (unsigned)   -> polarity reference
        C: -5 < 10   TRUE  (signed)   FALSE (unsigned: -5 wraps to 2^32-5)
        D: -5 > 10   FALSE (signed)   TRUE  (unsigned)

    Cases A/B establish which branch (jump vs fall-through) each truth value
    takes; cases C/D then reveal whether negative register values compare as
    signed two's complement or as huge unsigned integers. The 2026-06-05/06
    ActiveResetVerify data (corrective pi firing on every shot with negative
    accumulated I; ground-state damage tracking the below-zero fraction of the
    g blob) predicts UNSIGNED. The sign-safe offset fix in mModifiedRamsey /
    mActiveResetVerify is correct under either outcome; this test pins down
    the firmware behaviour.

    Pass your LIVE `config` from TATQ01-BFE.py as `cfg` (it has the on-resonance
    pulse_freq/pulse_gain and the calibrated res_phase, so the played-vs-skipped
    readout contrast is large). Fast overrides below keep the test short -- it
    needs no thermalisation because it tests pure tProc logic, not the qubit.
    """
    # Don't inherit a 15 ms relax_delay (that would make 4x200 reps take minutes).
    fast = {"shots": 200, "soft_avgs": 1, "relax_delay": 5.0}
    cases = {
        "A (5<10)":  (5, 10, "<", True, True),
        "B (5>10)":  (5, 10, ">", False, False),
        "C (-5<10)": (-5, 10, "<", True, False),
        "D (-5>10)": (-5, 10, ">", False, True),
    }
    amps = {}
    for name, (a, b, op, _, _) in cases.items():
        c = cfg | fast | {"_condj_a": a, "_condj_b": b, "_condj_op": op}
        amps[name] = CondjProbeProgram(soccfg, c).acquire(soc)

    amp_true, amp_false = amps["A (5<10)"], amps["B (5>10)"]
    hi, lo = max(amp_true, amp_false), min(amp_true, amp_false)
    rel_diff = (hi - lo) / hi if hi > 0 else 0.0

    out = {"amps": amps, "polarity": None, "signedness": None}

    print("\n[condj truth test] ============================================")
    for name, amp in amps.items():
        print(f"  |IQ| case {name}: {amp:.4f}")

    if rel_diff < 0.30:
        out["polarity"] = "INCONCLUSIVE"
        print(
            f"  A and B differ by only {rel_diff*100:.0f}% -- can't tell played\n"
            "    from skipped. The readout is OFF-RESONANCE (or gain~0): pass your\n"
            "    LIVE `config` (on-resonance pulse_freq/pulse_gain), not a placeholder."
        )
        print("[condj truth test] ============================================\n")
        return out

    # Larger amp = pulse PLAYED = condj did NOT jump; smaller = SKIPPED = jumped.
    mid = 0.5 * (hi + lo)
    jumped = {name: amp < mid for name, amp in amps.items()}

    # Polarity from A/B (both interpretations agree on their truth values).
    if jumped["A (5<10)"] and not jumped["B (5>10)"]:
        out["polarity"] = "JUMP-ON-TRUE (standard -- matches the code's assumption)"
        true_jumps = True
    elif jumped["B (5>10)"] and not jumped["A (5<10)"]:
        out["polarity"] = "JUMP-ON-FALSE (INVERTED vs the code's assumption!)"
        true_jumps = False
    else:
        out["polarity"] = f"INCONSISTENT (A jumped={jumped['A (5<10)']}, B jumped={jumped['B (5>10)']})"
        print(f"  VERDICT: {out['polarity']} -- rerun / check contrast.")
        print("[condj truth test] ============================================\n")
        return out

    # Signedness from C/D: under signed semantics C is TRUE / D is FALSE;
    # under unsigned semantics C is FALSE / D is TRUE.
    c_truth = jumped["C (-5<10)"] == true_jumps   # truth value the fw assigned to C
    d_truth = jumped["D (-5>10)"] == true_jumps
    if c_truth and not d_truth:
        out["signedness"] = "SIGNED (two's complement -- negative I compares correctly)"
    elif d_truth and not c_truth:
        out["signedness"] = (
            "UNSIGNED -- negative accumulated I compares as a huge positive.\n"
            "    This is the active-reset bug seen in the 06-05/06 ARV data. The\n"
            "    sign-safe offset (cmp_offset) now in mModifiedRamsey.py /\n"
            "    mActiveResetVerify.py neutralizes it."
        )
    else:
        out["signedness"] = f"INCONSISTENT (C truth={c_truth}, D truth={d_truth})"

    print(f"  VERDICT polarity  : {out['polarity']}")
    print(f"  VERDICT signedness: {out['signedness']}")
    print("[condj truth test] ============================================\n")
    return out


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
