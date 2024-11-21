from problem import LoopOptimization
from sympy import lambdify
from numba import njit, prange

import numpy as np

import lcapy
import os

class Lcapy(LoopOptimization):
    @classmethod
    def _load(cls, filename='', inn='', out=''):
        ckt = lcapy.Circuit(filename)
        ckt.simplify()
        print(ckt)
        tf = ckt.transfer(inn, 0, out, 0)

        syms = list(tf.sympy.free_symbols - {lcapy.s})
        vars = [s.name for s in syms]

        num = (tf.sympy.args[1] * tf.sympy.args[0].args[0]).as_poly(lcapy.s).all_coeffs()[::-1]
        den = (1 / tf.sympy.args[0].args[1]).as_poly(lcapy.s).all_coeffs()[::-1]

        def to_func(exprs):
            lmbda = njit(lambdify(
                [syms],
                exprs,
                modules=np,
                cse=True,
                docstring_limit=0
            ))

            num_coefs = len(exprs)

            @njit(nogil=True, fastmath=True, parallel=True)
            def f(vals):
                res = np.empty((vals.shape[0], num_coefs))
                for i in prange(vals.shape[0]):
                    res[i] = np.array(lmbda(vals[i])).T
                return res

            return f

        return vars, to_func(num), to_func(den)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('inn')
    parser.add_argument('out')

    args = parser.parse_args()

    problem = Lcapy(filename=args.filename, inn=args.inn, out=args.out)

    import time

    np.random.seed(0)

    coefs = np.random.rand(100, problem.n_var)

    for i in range(10):
        print("Evaluating...")
        start = time.time_ns()
        problem._evaluate(coefs, None)

        print(f"{(time.time_ns() - start) / 1e6:.2f} ms")
