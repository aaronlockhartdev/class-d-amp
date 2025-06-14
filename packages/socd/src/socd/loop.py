import dataclasses
import functools

import numpy as np

import sympy.core
import sympy.utilities.autowrap

class Loop:
    def __init__(
        self,
        num_array: tuple[sympy.Expr],
        den_array: tuple[sympy.Expr],
        delay: sympy.Expr,
        vars: tuple[sympy.Symbol],
    ):
        self.num_array = num_array
        self.den_array = den_array
        self.delay = delay
        self.vars = vars

        ufuncify = functools.partial(sympy.utilities.autowrap.ufuncify, self.vars)
        self.ufuncs = (
            *[
                (*[ufuncify(expr) for expr in arr],)
                for arr in (self.num_array, self.den_array)
            ],
            ufuncify(self.delay),
        )

    @classmethod
    def from_netlist(cls, netlist: str, delay: float | None = None) -> None:
        return cls()

    def ufunc(self, *args: np.ndarray, **kwargs: np.ndarray) -> tuple:
        

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
            *[np.stack([f(*args) for f in fs]) for fs in self.ufuncs[:2]],
            self.ufuncs[2](*args),
        )
