import marimo

__generated_with = "0.13.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Import necessary packages
    import re

    import numpy as np
    import control as ct

    from typing import Callable, Tuple

    import matplotlib.pyplot as plt

    return Tuple, ct, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot
    Given a transfer function **H**, frequency range **\[start, end\]**, and propagation delay **delay**, we numerically approximate the transfer function given a square wave amplifier, which theoretically produces an infinite number of harmonics. If there is no propagation delay or it is already approximated in the transfer function, **delay** defaults to the floating point ε to approximate the limit form mentioned in Putzey's paper.
    """
    )
    return


@app.cell
def _(Tuple, calc_resp, ct, get_consts, np):
    from . import batch_resp
    import inspect

    cached_consts = dict()

    def single_resp(
        H: ct.TransferFunction,
        start: float,
        end: float,
        delay=np.finfo(np.longdouble).eps,
        num_freqs=1_000,
        num_duty_cycles=100,
        num_harmonics=100,
    ) -> Tuple[np.ndarray]:

        args = (
            locals[p.name]
            for p in inspect.signature(single_resp).parameters
        )

        if args in cached_consts.keys():
            consts = cached_consts[args]
        else:
            consts = cached_consts[args] = get_consts(num_freqs, num_duty_cycles, num_harmonics, (start, end))

        mags, phs, osc_fs, dcins, dcgains, margins = calc_resp(
            np.array(H.num[0])[:, ::-1],
            np.array(H.den[0])[:, ::-1],
            np.array([delay]),
            *consts,
        )

        margins = np.degrees(margins)
        print(np.min(margins, axis=1))
        consts[0] = np.imag(consts[0]) / (2 * np.pi)
        osc_fs /= 2 * np.pi
        phs -= np.pi

        return (mags[0], phs[0], osc_fs[0], dcins[0], dcgains[0]), consts[:3]
    return


@app.cell
def _(calc, ct, np, plt):
    def plot(
        H: ct.TransferFunction,
        start: float,
        end: float,
        delay=np.finfo(np.longdouble).eps,
    ) -> None:
        """Takes a SISO transfer function `H` and plots phase response and loop gain vs.
        frequency, oscillation frequency vs. duty cycle, DC transfer curve, and loop gain
        vs. duty cycle."""

        (mags, phs, osc_fs, dcins, dcgains), (omega, hs, ns) = calc(
            H, start, end, delay
        )

        # Plotting
        fig, ((ax_ph, ax_dcgain), (ax_osc, ax_dcin)) = plt.subplots(2, 2)

        # Set figure labels
        ax_osc.set_ylabel("Osc. Freq. (Hz)")
        ax_osc.set_xlabel("Duty Cycle")

        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (°)")
        ax_mag = ax_ph.twinx()
        ax_mag.set_ylabel("Magnitude (dB)")

        # Plot phase
        cs = ["b", "g", "c", "y"]
        for i, p in enumerate(phs[-1 : 0 : -(hs.size // 5), :]):
            ax_ph.plot(omega, np.degrees(np.unwrap(p)), f"{cs[i % len(cs)]}--")

        ax_ph.plot([start, end], [-180, -180], "k:")
        ax_ph.set_xscale("log")

        max_ph = max(np.ceil(np.degrees(np.max(phs))), 0) + 10
        ax_ph.set_yticks([0, -180, -360])
        ax_ph.set_ylim(-360 - max_ph, max_ph)

        # Plot oscillation frequencies
        ax_osc.plot(
            np.concatenate((hs, 1 - hs[-1::-1])),
            np.concatenate((osc_fs, osc_fs[-1::-1])),
        )

        # Plot dcin
        ax_dcin.plot(
            np.concatenate((-dcins, dcins[-1::-1])),
            np.concatenate((2 * hs - 1, -2 * hs[-1::-1] + 1)),
        )

        # Plot dcgain
        ax_dcgain.plot(
            np.concatenate((2 * hs - 1, -2 * hs[-1::-1] + 1)),
            np.concatenate((dcgains, dcgains[-1::-1])),
        )

        # Plot magnitudes
        mag_scaled = 20 * np.log10(dcgains[-1] * np.absolute(H(2j * np.pi * omega)))
        ax_mag.plot(omega, mag_scaled, "r-")

        max_mag = (
            round(max(*map(np.abs, (np.min(mag_scaled), np.max(mag_scaled))), 0)) + 3
        )
        ax_mag.set_ylim(-max_mag, max_mag)

        fig.tight_layout()

    return (plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Transfer Functions

    As per Bruno Putzey's 2011 paper titled ["Global Modulated Self-Oscillating Amplifier with Improved Linearity"](https://www.hypex.nl/media/3f/62/4a/1682342035/Globally%20modulated%20self-oscillating%20amplifier.pdf), we treat our Class-D amplifier as a square wave oscillator wrapped with a linear function. For simplicity's sake, we further split said linear function into three serial sections–the **propagation delay**, **low pass filter**, and **feedback network**. The **propagation delay** is calculated during numerical evaluation in order to avoid using a Padé approximation. Please define functions constructing transfer functions based on component values for the **low pass filter** and **feedback network** below.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Workspace""")
    return


@app.cell
def _():
    from modules.sapwin import Sapwin

    problem = Sapwin(
        filename="../sym_analysis/SapWin/class_d_bp_2.out",
        n_fs=2_000,
        n_hs=20,
        n_ns=150,
    )
    return (problem,)


@app.cell
def _():
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.operators.sampling.lhs import LHS

    algorithm = DE(
        pop_size=100,
        sampling=LHS(),
        variant="DE/best/1/bin",
        CR=0.9,
        F=0.9,
        dither="vector",
        jitter=True,
    )
    return (algorithm,)


@app.cell
def _(algorithm, problem):
    from pymoo.optimize import minimize

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 1_000),
        seed=1,
        save_history=True,
        verbose=True,
    )
    return (res,)


@app.cell
def _(res):
    X = res.X
    F = res.F
    return (X,)


@app.cell
def _(X, problem):
    print(" ".join(map(lambda v, x: f"{v}:{x:.2E}", problem._vars, X)))
    return


@app.cell
def _(X, ct, plot, problem):
    vals = X[:-1][None, :]
    delay = X[-1]

    num = problem._calc_num(vals)[0, ::-1]
    den = problem._calc_den(vals)[0, ::-1]

    plot(ct.tf(num, den), 10, 2e7, delay)
    return


if __name__ == "__main__":
    app.run()
