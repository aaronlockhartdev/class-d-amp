import dataclasses
import functools

import numpy as np

import sympy.core
import sympy.utilities.autowrap

@dataclasses.dataclass(frozen=True)
class Loop:
    num_array: tuple[sympy.core.Expr]
    den_array: tuple[sympy.core.Expr]
    delay: sympy.core.Expr
    vars: tuple[sympy.core.Symbol]

    @classmethod
    def from_netlist(cls, netlist: str, delay: float | None = None) -> None:
        return cls()

    @functools.cached_property
    def ufunc(self):
        ufuncify = functools.partial(sympy.utilities.autowrap.ufuncify, self.vars)
        ufuncs = (
            *[
                (*[ufuncify(expr) for expr in arr],)
                for arr in (self.num_array, self.den_array)
            ],
            ufuncify(self.delay),
        )

        def ufunc(*args: np.ndarray, **kwargs: np.ndarray) -> tuple:
            if not args:
                if not kwargs:
                    raise ValueError("Please specify each parameter's values as either a positional argument or a keyword argument")
                elif (kw_set := set(kwargs.keys())) != (
                    v_set := {v.name for v in self.vars}
                ):
                    raise ValueError(
                        f"Missing keyword arguments for {', '.join(v_set.difference(kw_set))}"
                    )

                args = [kwargs[s] for s in self.vars]
            elif len(args) != len(self.vars):
                raise ValueError("Incorrect number of arguments specified")

            return (
                *[np.stack([f(*args) for f in fs]) for fs in ufuncs[:2]],
                ufuncs[2](*args),
            )

        return ufunc
