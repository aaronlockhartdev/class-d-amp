from modules.calc_resp import precompute_consts, calc_resp

from pymoo.core.problem import Problem

class LoopOptimization(Problem):
    def __init__(
        self, 
        n_fs=1_000, 
        n_hs=20, 
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
            n_obj=4,
            xl=xl,
            xu=xu,
            vtype=float
        )

    def _evaluate(self, x, out):
        vals = x[:,:-1]
        delays = x[:,-1]

        return calc_resp(
            self._calc_num(vals),
            self._calc_den(vals),
            delays,
            *self._eval_consts
        )

    @classmethod
    def _load(cls, **kwargs):
        raise NotImplemented("Subclass must implement _load") 
