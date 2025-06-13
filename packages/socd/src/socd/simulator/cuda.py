import queue
import functools

import marimo as mo

import numpy as np
import numpy.typing as npt

import cupy as cp
import cupy.polynomial.polynomial as cpp

from numba import cuda

import sympy

from .cache import Cache
from ..loop import Loop

class CUDACache:
    def __init__(self, cache: Cache, device: int):
        cp.cuda.Device(device).use()

        for prop in ("harmonics", "fr_coefs", "dc_coefs", "frequencies", "ns"):
            val = cache.__getattribute__(prop)
            dtype = cp.complex64 if np.iscomplexobj(val) else cp.float32
            self.__setattr__(prop, cp.asarray(val, dtype=dtype))

class CUDASimulator:
    def __init__(
        self,
        devices: tuple[int] | int = 0,
        threads_per_device: int = 8,
        batch_size: int = 16,
        max_iter: int = 10,
        osc_margin: float = 1e-1,
        n_fs: int = 1_000,
        n_hs: int = 20,
        n_ns: int = 200,
        min_duty_cycle: float = 1e-3,
        fr_range: tuple[float, float] = (1, 1e9),
    ) -> None:
        self.devices = devices if isinstance(devices, tuple) else (devices,)
        self.batch_size = batch_size
        self.threads_per_device = threads_per_device
        self.max_iter = max_iter
        self.osc_margin = osc_margin

        self._cache = Cache(n_fs, n_hs, n_ns, min_duty_cycle, fr_range)
        self._cuda_caches = dict()

        for device in self.devices:
            self._cuda_caches[device] = CUDACache(self._cache, device)

    def _simulate_batch(self, device: int, work_queue: queue.SimpleQueue):
        cp.cuda.Device(device).use()
        cp.cuda.Stream.ptds.use()
        cache = self._cuda_caches[device]

        thread = mo.current_thread()

        while not thread.should_exit:
            try:
                data = work_queue.get_nowait()
            except queue.Empty:
                break

            loop, params = data
            nums, dens, delays = map(
                functools.partial(cp.asarray, dtype=cp.float32), loop.ufunc(*params)
            )
            batch_size = delays.size

            steps = cp.full((batch_size * self._cache.n_hs), cp.inf)
            active_idxs = cp.mgrid[:batch_size, : self._cache.n_hs].reshape(2, -1)
            arange = cp.mgrid[:active_idxs.shape[1]]
            
            frequencies = cp.broadcast_to(cache.frequencies, (active_idxs.shape[1], *cache.frequencies.shape))
            harmonics = cp.broadcast_to(
                cache.harmonics, (active_idxs.shape[1], *cache.harmonics.shape)
            )

            tf_cache = cp.empty(
                (batch_size, self._cache.n_hs, self._cache.n_ns), dtype=cp.complex64
            )
            osc_freqs = cp.empty((batch_size, self._cache.n_hs), dtype=cp.float32)

            i = 0
            while active_idxs.size > 0 and i < self.max_iter:
                i += 1

                delays_masked = cp.expand_dims(delays[active_idxs[0]], (-2, -1))
                nums_masked = cp.expand_dims(nums[:, active_idxs[0]], (-2, -1))
                dens_masked = cp.expand_dims(dens[:, active_idxs[0]], (-2, -1))

                tfs = cp.exp(-delays_masked * harmonics) * (
                    cpp.polyval(harmonics, nums_masked, tensor=False) /
                    cpp.polyval(harmonics, dens_masked, tensor=False)
                )

                if iter == 0:
                    mags = cp.abs(tfs)

                frs = cp.einsum("afn,an->af", tfs, cache.fr_coefs[active_idxs[1,...]])
                phs = cp.unwrap(cp.angle(frs))

                signs = cp.signbit(phs)
                zcs = ~signs[:, :-1] & signs[:, 1:]
                inds = self._cache.n_fs - cp.argmax(cp.flip(zcs, axis=-1), axis=-1) - 1


                active_mask = steps >= self.osc_margin

                set_mask = ~active_mask
                set_idxs = active_idxs[:, set_mask]
                tf_cache[*set_idxs] = tfs[arange[set_mask], inds[set_mask], :]
                osc_freqs[*set_idxs] = cp.imag(frequencies[arange[set_mask], inds[set_mask]])

                active_idxs = active_idxs[:, active_mask]

                inds = inds[active_mask]

                arange = cp.mgrid[:active_idxs.shape[1]]
                frequencies, steps = cp.linspace(
                    frequencies[arange, inds - 1],
                    frequencies[arange, inds],
                    num=self._cache.n_fs,
                    axis=-1,
                    retstep=True
                )
                harmonics = cp.expand_dims(frequencies, -1) * cp.expand_dims(cache.ns, (0, 1))

            dcins = cp.real(cp.einsum("bhn,hn->bh", tf_cache, cache.dc_coefs))
            dcgains = cp.gradient(dcins, axis=-1)

    def simulate(
        self,
        loop: Loop,
        params: dict[sympy.Symbol | str, npt.ArrayLike] | npt.ArrayLike,
    ):

        work_threads = list()
        work_queue = queue.SimpleQueue()

        if isinstance(params, dict):
            keys = [
                v if v in params else v.name if v.name in params else None
                for v in loop.vars
            ]            
            missing = [y for (x, y) in zip(keys, loop.vars) if x is None]
            if missing:
                raise ValueError(f"`params` missing loop variables {', '.join(missing)}")
            params = np.stack([np.array(params[k]) for k in keys])              
        else:
            params = np.array(params)
            if params.shape[0] != len(loop.vars):
                raise ValueError(
                    f"`params` missing {len(loop.vars) - params.shape[0]} loop variables"
                )

        for i in range((params.shape[1] - 1) // self.batch_size + 1):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, params.shape[1])
            work_queue.put((loop, params[:, start:end]))

            
        for device in self.devices:
            work_threads.extend(
                mo.Thread(target=self._simulate_batch, args=(device, work_queue), daemon=True)
                for _ in range(self.threads_per_device)
            )

        for thread in work_threads:
            thread.start()

        for t in work_threads:
            t.join()
            
