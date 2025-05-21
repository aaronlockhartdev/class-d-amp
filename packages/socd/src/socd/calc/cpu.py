from numba import njit, prange

import functools

import numpy as np
import numpy.polynomial.polynomial as npp

def batch_resp(
    n_fs: int,
    n_hs: int,
    n_ns: int,
    fr_range: tuple[float],
    coefs: tuple[np.ndarray],
    delays: np.ndarray,
):
    return _batch_resp(coefs, delays, _get_consts(n_fs, n_hs, n_ns, fr_range))

@functools.cache
def _get_consts(n_fs, n_hs, n_ns, fr_range):
    fs = (
        2j * np.pi * np.logspace(np.log10(fr_range[0]), np.log10(fr_range[1]), num=n_fs)
    )
    (hs, h_step) = np.linspace(1e-2, 0.5, num=n_hs, retstep=True)
    ns = np.arange(1, n_ns + 1, dtype=np.complex128)

    fxn = fs[:, None] * ns[None, :] # Precompute square wave harmonic frequencies

    tmp1 = np.exp(2j * np.pi * hs[:, None] * ns[None, :])
    tmp2 = 1 - (1 / tmp1)
    fr_coef = tmp2 * (1 - tmp1) / (2 * ns)
    dc_coef = -tmp2 / (2j * ns) * (4 / np.pi)

    return (fs, hs, ns, h_step, fxn, fr_coef, dc_coef)

@njit(nogil=True, fastmath=True, parallel=True, cache=True)
def _batch_resp(coefs: tuple[np.ndarray], delays: np.ndarray, consts: tuple[np.ndarray]):
    fs, hs, ns, h_step, fxn, fr_coef, dc_coef = consts
    n_samples = delays.size
    num_coefs, den_coefs = coefs

    # Initialize arrays
    mag = np.empty((n_samples, fs.size))
    ph = np.empty((n_samples, hs.size, fs.size))
    osc_frs = np.full((n_samples, hs.size), np.inf) # Default to infinite oscillation frequency if no criterion is found
    dcins = np.empty((n_samples, hs.size))
    dcgains = np.empty((n_samples, hs.size))
    margins = np.zeros((n_samples, hs.size))

    for i in prange(n_samples):
        fr_tf = (
            npp.polyval(fxn, -num_coefs[i])
            / npp.polyval(fxn, den_coefs[i])
            * np.exp(-delays[i] * fxn)
        )

        fresp = np.zeros((hs.size, fs.size), dtype=np.complex128)
        for j in range(hs.size):
            for k in range(fs.size):
                for l in range(ns.size):
                    fresp[j, k] += fr_tf[k, l] * fr_coef[j, l]

        mag[i] = np.absolute(fr_tf[:, 0])
        ph[i] = np.unwrap(np.angle(fresp), -1)

        # Find (frequency maximal) zero-crossings in phase
        gs = np.signbit(ph[i, ...])
        zcs = ~gs[:, :-1] & gs[:, 1:]
        inds = fs.size - np.argmax(zcs[:, ::-1], axis=-1) - 1

        # Binary search for exact oscillation frequency
        tmp = np.empty((hs.size, ns.size), dtype=np.complex128)
        for j in range(hs.size):
            if inds[j] == fs.size - 1 or inds[j] == 0:
                continue

            # Binary search for exact oscillation frequency
            start = fs[inds[j] - 1]
            end = fs[inds[j]]
            iters = 0
            err = np.nan
            thresh = 1e-5
            while end - start > thresh:
                mid = (start + end) / 2
                xn = ns * mid
                tmp[j] = (
                    npp.polyval(xn, -num_coefs[i])
                    / npp.polyval(xn, den_coefs[i])
                    * np.exp(-delays[i] * xn)
                )
                err = np.angle(np.sum(tmp[j] * fr_coef[j, :]))

                if np.signbit(err):
                    end = mid
                else:
                    start = mid

                iters += 1
            else:
                osc_frs[i, j] = np.imag(mid)

            # Find phase margin before oscillation
            k = inds[j] - 1
            while not np.signbit(ph[i, j, k]):
                if k == 0:
                    break
                if ph[i, j, k] > margins[i, j]:
                    margins[i, j] = ph[i, j, k]
                k -= 1

        dcins[i] = -np.sum(np.real(tmp * dc_coef), axis=-1)

        h_step = 2 * (hs[1] - hs[0])

        for j in range(1, hs.size - 1):
            dcgains[i, j] = (2 * h_step) / (dcins[i, j - 1] - dcins[i, j + 1])

        dcgains[i, 0] = h_step / (dcins[i, 0] - dcins[i, 1])
        dcgains[i, -1] = h_step / (dcins[i, -2] - dcins[i, -1])

    return mag, ph, osc_frs, dcins, dcgains, margins
