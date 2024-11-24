from modules.calc_resp import precompute_consts, calc_resp

from pymoo.core.problem import Problem

import numpy as np

class LoopOptimization(Problem):
    def __init__(
        self, 
        n_fs=1_000, 
        n_hs=10, 
        n_ns=100, 
        fr_range=(1., 1e7),
        p_range=(1e-8, 1e-5),
        **kwargs
        ):
        self._vars, self._calc_num, self._calc_den = self.__class__._load(**kwargs)
        self._eval_consts = precompute_consts(n_fs, n_hs, n_ns, fr_range)

        def create_bound(var):
            match var[0]:
                case 'R':
                    return (1e-3, 1e6)
                case 'C':
                    return (1e-12, 1e-4)
                case 'L':
                    return (1e-7, 1e-3)
                case _:
                    raise ValueError(f'Please specify `bounds` for component type {var[0]}')

        xl, xu = zip(*(list(map(create_bound, self._vars)) + [p_range]))

        super().__init__(
            n_var=len(self._vars) + 1,
            n_obj=2,
            xl=xl,
            xu=xu,
            vtype=float
        )

    def _evaluate(self, x, out):
        vals = x[:,:-1]
        delays = x[:,-1]

        mag, ph, osc_frs, dcins, dcgains = calc_resp(
            self._calc_num(vals),
            self._calc_den(vals),
            delays,
            *self._eval_consts
        )

        fs, hs, ns = self._eval_consts[:3]

        obj_lst = list()
        obj_lst.append(abs(osc_frs[:,-1] / (2 * np.pi) - 500e3))
        obj_lst.append(np.nan_to_num(1.01 ** -np.min((mag[:,-1] * dcgains[:,-1,None])[:,np.imag(fs) < 2 * np.pi * 20e3], axis=-1)))

        print(osc_frs.shape)
        print(osc_frs[:,-1] / (2 * np.pi))

        out['F'] = obj_lst

    @classmethod
    def _load(cls, **kwargs):
        raise NotImplemented("Subclass must implement _load") 
