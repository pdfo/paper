import sys

import numpy as np
import pycutest
from pdfo import pdfo


def fun(x):
    global n_eval
    f = problem.obj(x)
    n_eval += 1
    with np.printoptions(precision=3, threshold=10, linewidth=sys.maxsize):
        print(f"{problem.name}({x}) = {f} ({n_eval})")
    return f


def grad(x):
    f = problem.obj(x)
    g = np.zeros(x.size)
    for i in range(x.size):
        sign_x = 1 if x[i] >= 0 else -1
        step = sign_x * np.sqrt(np.finfo(float).eps) * max(abs(x[i]), 1.0)
        coord_vec = np.squeeze(np.eye(1, x.size, i))
        f_forward = fun(x + step * coord_vec)
        g[i] = (f_forward - f) / step
    return g


if __name__ == '__main__':
    # print(pycutest.problem_properties("INDEF"))
    problem = pycutest.import_problem("INDEF", sifParams={"N": 50})
    n_eval = 0

    options = {"maxfev": 25_000}
    res = pdfo(fun, problem.x0, options=options)
    print(res)
