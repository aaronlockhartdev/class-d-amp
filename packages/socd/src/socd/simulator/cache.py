import dataclasses
import functools

import numpy as np

@dataclasses.dataclass(frozen=True)
class Cache:
    n_fs: int
    n_hs: int
    n_ns: int
    min_duty_cycle: float
    fr_range: tuple[float, float]

    @functools.cached_property
    def ns(self) -> np.ndarray:
        return np.arange(1, self.n_ns + 1)

    @functools.cached_property
    def _tmp1(self) -> np.ndarray:
        return np.exp(2j * np.pi * self.duty_cycles[:, None] * self.ns[None, :])

    @functools.cached_property
    def _tmp2(self) -> np.ndarray:
        return 1 - (1 / self._tmp1)

    @functools.cached_property
    def frequencies(self) -> np.ndarray:
        return 2j * np.pi * np.geomspace(*self.fr_range, num=self.n_fs)

    @functools.cached_property
    def duty_cycles(self) -> np.ndarray:
        return np.linspace(self.min_duty_cycle, 0.5, num=self.n_hs)

    @functools.cached_property
    def harmonics(self) -> np.ndarray:
        return self.frequencies[:, None] * self.ns[None, :]

    @functools.cached_property
    def fr_coefs(self) -> tuple[np.ndarray, np.ndarray]:
        return self._tmp2 * (1 - self._tmp1) / (2 * self.ns)

    @functools.cached_property
    def dc_coefs(self) -> np.ndarray:
        return -self._tmp2 / (2j * self.ns) * (4 / np.pi)
      
