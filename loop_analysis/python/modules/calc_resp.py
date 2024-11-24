from numba import njit, prange

import numpy as np
import numpy.polynomial.polynomial as npp


def precompute_consts(n_fs, n_hs, n_ns, fr_range):
    fs = 2j * np.pi * np.logspace(
        np.log10(fr_range[0]), 
        np.log10(fr_range[1]), 
        num=n_fs
    )
    hs = np.linspace(1e-1, 0.5, num=n_hs)
    ns = np.arange(1, n_ns+1, dtype=np.complex128)

    fxn = fs[:,None] * ns[None,:]

    tmp1 = np.exp(2j * np.pi * hs[:,None] * ns[None,:])
    tmp2 = 1 - (1 / tmp1)
    fr_tmp = tmp2 * (1-tmp1) / (2 * ns)
    dcgain_tmp = -tmp2 / (2j * ns) * (4 / np.pi)

    return fs, hs, ns, fxn, fr_tmp, dcgain_tmp

@njit(nogil=True, fastmath=True, parallel=True, cache=True)
def calc_resp(
        num_coefs, den_coefs, delays, 
        fs, hs, ns, 
        fxn, fr_tmp, dcin_tmp
    ):

    n_samples = delays.size

    mag = np.empty((n_samples, hs.size, fs.size))
    ph = np.empty((n_samples, hs.size, fs.size))
    osc_frs = np.full((n_samples, hs.size), np.inf)
    dcins = np.empty((n_samples, hs.size))
    dcgains = np.empty((n_samples, hs.size))

    for i in prange(n_samples):
        fr_tf = npp.polyval(fxn, -num_coefs[i]) / npp.polyval(fxn, den_coefs[i]) * np.exp(-delays[i] * fxn)

        fresp = np.zeros((hs.size, fs.size), dtype=np.complex128)
        for j in range(hs.size):
            for k in range(fs.size):
                for l in range(ns.size):
                    fresp[j,k] += fr_tf[k,l] * fr_tmp[j,l]

        mag[i] = np.absolute(fresp)
        ph[i] = np.unwrap(np.angle(fresp), -1)

        gs = np.signbit(ph[i,...])
        zcs = ~gs[:,:-1] & gs[:,1:]
        inds = fs.size - np.argmax(zcs[:,::-1], axis=-1) - 1

        tmp = np.empty((hs.size, ns.size), dtype=np.complex128)
        for j in range(hs.size):
            if inds[j] == fs.size - 1:
                continue
            start = fs[inds[j]-1]
            end = fs[inds[j]]
            iters = 0
            err = np.nan
            while not np.isclose(err, 0., atol=1e-8):
                if iters > 1_000: 
                    break
                mid = (start + end) / 2
                xn = ns * mid
                tmp[j] = npp.polyval(xn, -num_coefs[i]) / npp.polyval(xn, den_coefs[i]) * np.exp(-delays[i] * xn)
                err = np.angle(np.sum(tmp[j] * fr_tmp[j,:]))

                if np.signbit(err): end = mid
                else: start = mid

                iters += 1
            else:
                osc_frs[i,j] = np.imag(mid)

        dcins[i] = -np.sum(np.real(tmp * dcin_tmp), axis=-1)# * num_coefs[0,i] / den_coefs[0,i]

        diff_h = 2 * (hs[1] - hs[0])
        
        for j in range(1, hs.size - 1):
            dcgains[i,j] = (2 * diff_h) / (dcins[i,j-1] - dcins[i,j+1])

        dcgains[i,0] = diff_h / (dcins[i,0] - dcins[i,1])
        dcgains[i,-1] = diff_h / (dcins[i,-2] - dcins[i,-1])

    return mag, ph, osc_frs, dcins, dcgains

