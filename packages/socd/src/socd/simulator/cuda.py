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

@cp.fuse
def lin_approx(highs: cp.ndarray, lows: cp.ndarray):
    return ((lows[1] * highs[0]) - (highs[1] * lows[0])) / (lows[1] - highs[1])

def calc_tfs(nums: cp.ndarray, dens: cp.ndarray, delays: cp.ndarray, harmonics: cp.ndarray):
    return cp.exp(-delays * harmonics) * (
        cpp.polyval(harmonics, nums, tensor=False)
        / cpp.polyval(harmonics, dens, tensor=False)
    )


@dataclasses.dataclass
class CUDAWorkThread:
    device: int
    cache: CUDACache
    work_queue: queue.SimpleQueue
    batch_size: int
    max_iter: int
    osc_margin: float

    def __post_init__(self):
        self._cuda_graphs = dict()

    def __call__(self, evaluate=False):       
        if _use_marimo:
            thread  = mo.current_thread()

        cp.cuda.Device(self.device).use()

        while _use_marimo and not thread.should_exit:
            try:
                loop, params = self.work_queue.get_nowait()
            except queue.Empty:
                break

            nums, dens, delays = map(
                functools.partial(cp.asarray, dtype=cp.float32), loop.ufunc(*params)
            )

            if (batch_size := params.shape[1]) in self._cuda_graphs:
                with cp.cuda.Stream.ptds as stream:
                    self._cuda_graphs[batch_size].launch()
                    stream.synchronize()
                continue

            with cp.cuda.Stream(non_blocking=True) as stream:

                stream.begin_capture()

                nums, dens, delays = map(
                    functools.partial(cp.expand_dims, axis=(-2, -1)),
                    (nums, dens, delays)
                )

                frequencies = cp.expand_dims(self.cache.frequencies, 0)
                harmonics = 1j * cp.expand_dims(self.cache.harmonics, 0)

                tfs = calc_tfs(nums, dens, delays, harmonics)
                frs = cp.sum(
                    cp.multiply(
                        cp.expand_dims(tfs, 1),
                        cp.expand_dims(self.cache.fr_coefs, (0, 2)),
                    ),
                    axis=-1,
                )

                phs = cp.unwrap(cp.angle(frs), axis=-1)

                signs = cp.signbit(phs)
                zcs = ~signs[..., :-1] & signs[..., 1:]
                inds = cp.expand_dims((self.cache.n_fs - 1) - cp.argmax(cp.flip(zcs, axis=-1), axis=-1), -1)

                frequencies = cp.expand_dims(frequencies, axis=1)

                lows, highs = (
                    cp.stack(
                        (
                            cp.squeeze(
                                cp.take_along_axis(arr, inds_, axis=-1),
                                axis=-1,
                            )
                            for arr in (frequencies, phs)
                        )
                    )
                    for inds_ in (inds - 1, inds)
                )

                for iter_ in range(self.max_iter):
                    mids = lin_approx(highs, lows)
                    harmonics = 1j * cp.expand_dims(mids, -1) * cp.expand_dims(self.cache.ns, (0, 1))

                    tfs = calc_tfs(nums, dens, delays, harmonics)

                    if iter_ == self.max_iter - 1:
                        break
                
                    frs = cp.sum(
                        cp.multiply(
                            tfs,
                            cp.expand_dims(self.cache.fr_coefs, 0),
                        ),
                        axis=-1,
                    )

                    mids = cp.stack((mids, cp.angle(frs)))

                    signbits = cp.expand_dims(cp.signbit(mids[1]), 0)
                    lows = cp.where(signbits, lows, mids)
                    highs = cp.where(signbits, mids, highs)


                osc_freqs = mids

                dcins = cp.real(
                    cp.sum(
                        cp.multiply(tfs, cp.expand_dims(self.cache.dc_coefs, 0)), axis=-1
                    )
                )

                dcgains = cp.gradient(dcins, axis=-1)

                self._cuda_graphs[batch_size] = stream.end_capture()

            self._cuda_graphs[batch_size].launch()

class CUDASimulator:
    def __init__(
        self,
        devices: tuple[int] | int = 0,
        threads_per_device: int = 3,
        batch_size: int = 32,
        iters: int = 10,
        n_fs: int = 1_000,
        n_hs: int = 20,
        n_ns: int = 200,
        min_duty_cycle: float = 1e-3,
        fr_range: tuple[float, float] = (1, 1e9),
    ) -> None:
        self.devices = devices if isinstance(devices, tuple) else (devices,)
        self.batch_size = batch_size
        self.threads_per_device = threads_per_device
        self.iters= iters

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
                        self.iters,
                        self.osc_margin,
                    ),
                    daemon=True,
                )
                for _ in range(self.threads_per_device)
            )

        for thread in work_threads:
            thread.start()

        while alive_threads := [thread for thread in work_threads if thread.is_alive()]:
            for thread in alive_threads:
                thread.join(timeout=0.5)
           
