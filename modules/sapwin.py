from typing import List

from modules.calc_resp import precompute_consts, calc_resp
from modules.problem import LoopOptimization

from functools import partial
from numba import njit, prange

import numpy as np
import symengine as se

import re

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
            vars.sort()
            syms = se.symbols(' '.join(vars))

            def to_func(exprs: List[str]):
                sym_exprs = list(map(se.S, exprs))

                lmbda = se.Lambdify(
                    [syms], 
                    sym_exprs, 
                    cse=True, 
                    backend='llvm'
                )

                n_coefs = len(exprs)

                #@njit(nogil=True, fastmath=True, parallel=True)
                def f(vals):
                    n_samples = vals.shape[0]
                    res = np.empty((n_samples, n_coefs))
                    for i in prange(n_samples):
                        res[i] = lmbda(vals[i])
                    return res

                return f

            return vars, to_func(num_exprs), to_func(den_exprs)

            

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
