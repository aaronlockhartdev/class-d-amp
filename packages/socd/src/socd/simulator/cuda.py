import importlib.util

if importlib.util.find_spec("marimo"):
    _use_marimo = True
    import marimo as mo
else:
    _use_marimo = False
    import threading


import queue
import functools
import dataclasses

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

        for name in dir(cache):
            if "_" == name[0]:
                continue
            value = cache.__getattribute__(name)
            if isinstance(value, np.ndarray):
                value = cp.asarray(
                    value, dtype=cp.complex64 if np.iscomplexobj(value) else cp.float32
                )
            self.__setattr__(name, value)

@dataclasses.dataclass
class CUDAWorkThread:
    device: int
    cache: CUDACache
    work_queue: queue.SimpleQueue
    batch_size: int
    max_iter: int
    osc_margin: float

    def __post_init__(self):
        self.device = cp.cuda.Device(self.device)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)

    def __call__(self, evaluate=False):       
        if _use_marimo:
            thread  = mo.current_thread()

        self.device.use()
        self.stream.use()

        dcins = cp.empty((self.batch_size, self.cache.n_hs), dtype=cp.float32)
        osc_freqs = cp.empty((self.batch_size, self.cache.n_hs), dtype=cp.float32)

        while _use_marimo and not thread.should_exit:
            try:
                loop, params = self.work_queue.get_nowait()
            except queue.Empty:
                break

            batch_size = params.shape[1]

            nums, dens, delays = map(
                functools.partial(cp.asarray, dtype=cp.float32), loop.ufunc(*params)
            )

            active_idxs = cp.mgrid[:batch_size, :self.cache.n_hs].reshape(2, -1)
            arange = cp.mgrid[:active_idxs.shape[1]]

            frequencies = cp.expand_dims(self.cache.frequencies, 0)
            harmonics = cp.expand_dims(self.cache.harmonics, 0)

            i = 0
            while (
                active_idxs.size > 0
                and i < self.max_iter
                and (_use_marimo and not thread.should_exit)
            ):
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

                frs = cp.einsum("afn,an->af", tfs, self.cache.fr_coefs[active_idxs[1]])
                phs = cp.unwrap(cp.angle(frs))

                signs = cp.signbit(phs)
                zcs = ~signs[:, :-1] & signs[:, 1:]
                inds = self.cache.n_fs - cp.argmax(cp.flip(zcs, axis=-1), axis=-1) - 1

                active_mask = cp.abs(phs[arange, inds]) > self.osc_margin

                set_mask = ~active_mask
                set_idxs = active_idxs[:, set_mask]
                dcins[*set_idxs] = cp.real(
                    cp.einsum(
                        "an,an->a",
                        tfs[arange[set_mask], inds[set_mask], :],
                        self.cache.dc_coefs[set_idxs[1]],
                    )
                )
                osc_freqs[*set_idxs] = cp.imag(frequencies[arange[set_mask], inds[set_mask]])

                active_idxs = active_idxs[:, active_mask]

                inds = inds[active_mask]

                arange = cp.mgrid[:active_idxs.shape[1]]
                frequencies = cp.linspace(
                    frequencies[arange, inds - 1],
                    frequencies[arange, inds],
                    num=self.cache.n_fs // max(1, active_idxs.shape[1]),
                    axis=-1,
                )
                harmonics = cp.expand_dims(frequencies, -1) * cp.expand_dims(self.cache.ns, (0, 1))

            #dcgains = cp.gradient(dcins, axis=-1)

class CUDASimulator:
    def __init__(
        self,
        devices: tuple[int] | int = 0,
        threads_per_device: int = 3,
        batch_size: int = 32,
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

        self.cache = Cache(n_fs, n_hs, n_ns, min_duty_cycle, fr_range)
        self._cuda_caches = dict()

        for device in self.devices:
            self._cuda_caches[device] = CUDACache(self.cache, device)

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
                raise ValueError(f"Argument `params` missing loop variables {', '.join(missing)}")
            params = np.stack([np.array(params[k]) for k in keys])              
        else:
            params = np.array(params)
            if params.shape[0] != len(loop.vars):
                raise ValueError(
                    f"Argument `params` missing {len(loop.vars) - params.shape[0]} loop variables"
                )

        for i in range((params.shape[1] - 1) // self.batch_size + 1):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, params.shape[1])
            work_queue.put((loop, params[:, start:end]))
            
        Thread = mo.Thread if _use_marimo else threading.Thread
        for device in self.devices:
            work_threads.extend(
                Thread(
                    target=CUDAWorkThread(
                        device,
                        self._cuda_caches[device],
                        work_queue,
                        self.batch_size,
                        self.max_iter,
                        self.osc_margin,
                    )
                )
                for _ in range(self.threads_per_device)
            )

        for thread in work_threads:
            thread.start()

        while alive_threads := [thread for thread in work_threads if thread.is_alive()]:
            for thread in alive_threads:
                thread.join(timeout=0.5)
           
