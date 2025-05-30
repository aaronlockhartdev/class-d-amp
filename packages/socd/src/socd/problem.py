from modules.calc_resp import precompute_consts, calc_resp

from pymoo.core.problem import Problem

import numpy as np


class LoopOptimization(Problem):
    def __init__(
        self,
        n_fs=1_000,
        n_hs=10,
        n_ns=100,
        fr_range=(1.0, 1e7),
        p_range=(1e-8, 1e-5),
        **kwargs,
    ):
        self._vars, self._calc_num, self._calc_den = self.__class__._load(**kwargs)
        self._eval_consts = precompute_consts(n_fs, n_hs, n_ns, fr_range)

        def create_bound(var):
            match (var[0]):
                case "R":
                    return (1e2, 1e5)
                case "C":
                    return (1e-12, 1e-4)
                case "L":
                    return (1e-7, 1e-3)
                case _:
                    raise ValueError(
                        f"Please specify `bounds` for component type {var[0]}"
                    )

        xl, xu = zip(*(list(map(create_bound, self._vars)) + [p_range]))

        super().__init__(n_var=len(self._vars) + 1, n_obj=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out):
        vals = x[:, :-1]
        delays = x[:, -1]

        mag, ph, osc_frs, dcins, dcgains, margins = calc_resp(
            self._calc_num(vals), self._calc_den(vals), delays, *self._eval_consts
        )

        fs, hs, ns = self._eval_consts[:3]

        p_band = hs > 0.25
        f_band = np.imag(fs) < 1.57e5

        osc_f_err = abs(osc_frs[:, -1] - 3.14e6)
        min_gain = 1 / np.min(
            (mag[:, None, f_band] * dcgains[:, p_band, None]), axis=(1, 2)
        )
        margin = 1 - 1 / (1 + 1e4 ** (0.2 - np.min((margins[:, p_band]), axis=1)))

        mask = np.all(np.isfinite(osc_frs[:, p_band]), axis=1)
        mask &= min_gain > 0

        err = np.full((x.shape[0],), np.finfo(float).max)
        err[mask] = (min_gain + 1e-8 * osc_f_err + 5e-4 * margin)[mask]

        out["F"] = err

    @classmethod
    def _load(cls, **kwargs):
        raise NotImplementedError("Subclass must implement _load")
