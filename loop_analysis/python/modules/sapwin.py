import sys
sys.setrecursionlimit(1_000_000)
sys.set_int_max_str_digits(10_000)

from typing import List

from modules.calc_resp import precompute_consts, calc_resp
from modules.problem import LoopOptimization

from sympy import symbols, lambdify, Matrix
from sympy.parsing.sympy_parser import parse_expr, T
from functools import partial

from numba import njit

import numpy as np

import re
import os

class Sapwin(LoopOptimization):
    @classmethod
    def _load(cls, filename='', voltage=None):
        num_exprs = list()
        den_exprs = list()

        vars = set()

        with open(filename, 'r') as f:
            exprs = num_exprs
            ord_count = 0
            
            skip = False
            v_found = False
           
            for i, line in enumerate(f):
                l = line.rstrip()

                if not l:
                    if skip:
                        skip = False
                        continue
                    break

                if not v_found and (not voltage or l == voltage + ':'):
                    skip = True
                    v_found = True
                    continue

                # If division line
                if set(l) == {'-'}:
                    ord_count = 0
                    exprs = den_exprs
                    continue
                
                parens = re.search(r'\(.*\)', l)
                coef = parens.group()[1:-1]
                terms = re.findall(r'\+|\-|\S+', coef)

                new_vars = set(re.findall(r'(?<!\w)[a-zA-Z]\w+', coef))
                vars.update(new_vars)
                
                tail = l[parens.span()[-1]:]
                if lap := re.search(r's(\^\d+)?', tail):
                    tmp = re.search(r'\d+', lap.group())
                    ordr = int(tmp.group()) if tmp else 1
                else: ordr = 0

                expr = ''

                mult = False
                for t in terms:
                    
                    if t not in {'-', '+'}:
                        if t.isdigit():
                            # Convert integers to float
                            t += '.'
                        if mult:
                            expr += "*"
                        mult = True
                    else:
                        mult = False

                    expr += t
                
                while ordr < ord_count:
                    exprs.append('0')
                    ord_count += 1

                exprs.append(expr)
                ord_count += 1
            
            vars = list(vars)

            syms = {x: symbols(x) for x in vars}
            def convert_to_func(exprs: List[str]):
                sym_exprs = [
                    parse_expr(
                        e, 
                        local_dict=syms, 
                        transformations=T[:5], 
                        evaluate=False
                    ) 
                    for e in exprs
                ]

                lmbda = njit(nogil=True, fastmath=True, parallel=True)(lambdify(
                    syms.values(), 
                    sym_exprs, 
                    modules=np, 
                    cse=True, 
                    docstring_limit=0
                ))

                return lambda x: np.stack(lmbda(*x.T))

            return vars, convert_to_func(num_exprs), convert_to_func(den_exprs)

            

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('-v', '--voltage')

    args = parser.parse_args()

    problem = Sapwin(filename=args.filename, voltage=args.voltage)

    import time

    np.random.seed(0)

    coefs = np.random.rand(100, problem.n_var)

    for i in range(10):
        print("Evaluating...")
        start = time.time_ns()
        problem._evaluate(coefs, None)

        print(f"{(time.time_ns() - start) / 1e6:.2f} ms")
