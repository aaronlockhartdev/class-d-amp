import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import socd
    return mo, socd


@app.cell
def _(socd):
    simulator = socd.CUDASimulator(devices=(0, 1), n_fs=10_000, n_ns=100, n_hs=10, batch_size=200, threads_per_device=1, iters=10)
    return (simulator,)


@app.cell
def _(socd):
    import sympy
    x, y = sympy.symbols('x y')

    loop = socd.Loop((x,) * 4, (y,) * 3, x + y, (x, y))
    return loop, x, y


@app.cell
def _():
    import numpy as np
    rng = np.random.default_rng(seed=0)
    x_arr = rng.random((10_000,))
    y_arr = rng.random((10_000,))
    return x_arr, y_arr


@app.cell
def _(loop, simulator, x, x_arr, y, y_arr):
    simulator.simulate(loop, {x: x_arr, y: y_arr})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot
    Given a transfer function **H**, frequency range **\[start, end\]**, and propagation delay **delay**, we numerically approximate the transfer function given a square wave amplifier, which theoretically produces an infinite number of harmonics. If there is no propagation delay or it is already approximated in the transfer function, **delay** defaults to the floating point ε to approximate the limit form mentioned in Putzey's paper.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Transfer Functions

    As per Bruno Putzey's 2011 paper titled ["Global Modulated Self-Oscillating Amplifier with Improved Linearity"](https://www.hypex.nl/media/3f/62/4a/1682342035/Globally%20modulated%20self-oscillating%20amplifier.pdf), we treat our Class-D amplifier as a square wave oscillator wrapped with a linear function. For simplicity's sake, we further split said linear function into three serial sections–the **propagation delay**, **low pass filter**, and **feedback network**. The **propagation delay** is calculated during numerical evaluation in order to avoid using a Padé approximation. Please define functions constructing transfer functions based on component values for the **low pass filter** and **feedback network** below.
    """
    )
    return


if __name__ == "__main__":
    app.run()
