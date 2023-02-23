import warnings

import numpy as np
import pycutest
from scipy.linalg import lstsq
from scipy.optimize import Bounds, LinearConstraint, minimize


class Problem:

    def __init__(self, *args, **kwargs):
        problem = pycutest.import_problem(*args, **kwargs)

        self.name = problem.name
        self.n = problem.n
        self.m = problem.m
        self.x0 = np.array(problem.x0, copy=True)
        self.sif_params = problem.sifParams
        self.var_type = np.array(problem.vartype, copy=True)
        self.xl = np.array(problem.bl, copy=True)
        self.xu = np.array(problem.bu, copy=True)
        self.xl[self.xl <= -1e20] = -np.inf
        self.xu[self.xu >= 1e20] = np.inf
        self.cl = problem.cl
        self.cu = problem.cu
        if self.m > 0:
            self.cl[self.cl <= -1e20] = -np.inf
            self.cu[self.cu >= 1e20] = np.inf
        self.is_eq_cons = problem.is_eq_cons
        self.is_linear_cons = problem.is_linear_cons

        self.obj = problem.obj
        self.cons = problem.cons

        # The following attributes can be built from other attributes. However,
        # they may be time-consuming to build. Therefore, we construct them only
        # when they are accessed for the first time.
        self._aub = None
        self._bub = None
        self._aeq = None
        self._beq = None
        self._m_linear_ub = None
        self._m_linear_eq = None
        self._m_nonlinear_ub = None
        self._m_nonlinear_eq = None

        # Project the initial guess only the feasible polyhedron (including the
        # bound and the linear constraints).
        self.project_x0()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def aub(self):
        if self._aub is None:
            self._aub, self._bub = self.build_linear_ub_cons()
        return self._aub

    @property
    def bub(self):
        if self._bub is None:
            self._aub, self._bub = self.build_linear_ub_cons()
        return self._bub

    @property
    def aeq(self):
        if self._aeq is None:
            self._aeq, self._beq = self.build_linear_eq_cons()
        return self._aeq

    @property
    def beq(self):
        if self._beq is None:
            self._aeq, self._beq = self.build_linear_eq_cons()
        return self._beq

    @property
    def m_linear_ub(self):
        if self._m_linear_ub is None:
            if self.m == 0:
                self._m_linear_ub = 0
            else:
                iub = self.is_linear_cons & ~self.is_eq_cons
                iub_cl = self.cl[iub] >= -np.inf
                iub_cu = self.cu[iub] < np.inf
                self._m_linear_ub = np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)
        return self._m_linear_ub

    @property
    def m_linear_eq(self):
        if self._m_linear_eq is None:
            if self.m == 0:
                self._m_linear_eq = 0
            else:
                ieq = self.is_linear_cons & self.is_eq_cons
                self._m_linear_eq = np.count_nonzero(ieq)
        return self._m_linear_eq

    @property
    def m_nonlinear_ub(self):
        if self._m_nonlinear_ub is None:
            if self.m == 0:
                self._m_nonlinear_ub = 0
            else:
                iub = ~(self.is_linear_cons | self.is_eq_cons)
                iub_cl = self.cl[iub] > -np.inf
                iub_cu = self.cu[iub] < np.inf
                self._m_nonlinear_ub = np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)
        return self._m_nonlinear_ub

    @property
    def m_nonlinear_eq(self):
        if self._m_nonlinear_eq is None:
            if self.m == 0:
                self._m_nonlinear_eq = 0
            else:
                ieq = ~self.is_linear_cons & self.is_eq_cons
                self._m_nonlinear_eq = np.count_nonzero(ieq)
        return self._m_nonlinear_eq

    @property
    def type(self):
        properties = pycutest.problem_properties(self.name)
        return properties.get("constraints")

    def fun(self, x, callback=None, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        f = self.obj(x)

        # If a noise function is supplied, return both the plain and the noisy function evaluations.
        if callback is not None:
            return f, callback(x, f, *args, **kwargs)
        return f

    def cub(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        iub = ~(self.is_linear_cons | self.is_eq_cons)
        iub_cl = self.cl[iub] > -np.inf
        iub_cu = self.cu[iub] < np.inf
        c = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_index = self.cons(x, index)
            if iub_cl[i]:
                c.append(self.cl[index] - c_index)
            if iub_cu[i]:
                c.append(c_index - self.cu[index])
        return np.array(c, dtype=float)

    def ceq(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        c = []
        for index in np.flatnonzero(ieq):
            c_index = self.cons(x, index)
            c.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(c, dtype=float)

    def maxcv(self, x):
        maxcv = np.max(self.xl - x, initial=0.0)
        maxcv = np.max(x - self.xu, initial=maxcv)
        maxcv = np.max(np.dot(self.aub, x) - self.bub, initial=maxcv)
        maxcv = np.max(np.abs(np.dot(self.aeq, x) - self.beq), initial=maxcv)
        maxcv = np.max(self.cub(x), initial=maxcv)
        maxcv = np.max(np.abs(self.ceq(x)), initial=maxcv)
        return maxcv

    def build_linear_ub_cons(self):
        if self.m == 0:
            return np.empty((0, self.n)), np.empty(0)
        iub = self.is_linear_cons & np.logical_not(self.is_eq_cons)
        iub_cl = self.cl[iub] > -np.inf
        iub_cu = self.cu[iub] < np.inf
        aub = []
        bub = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_index, g_index = self.cons(np.zeros(self.n), index, True)
            if iub_cl[i]:
                aub.append(-g_index)
                bub.append(c_index - self.cl[index])
            if iub_cu[i]:
                aub.append(g_index)
                bub.append(self.cu[index] - c_index)
        return np.reshape(aub, (-1, self.n)), np.array(bub)

    def build_linear_eq_cons(self):
        if self.m == 0:
            return np.empty((0, self.n)), np.empty(0)
        ieq = self.is_linear_cons & self.is_eq_cons
        aeq = []
        beq = []
        for index in np.flatnonzero(ieq):
            c_index, g_index = self.cons(np.zeros(self.n), index, True)
            aeq.append(g_index)
            beq.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        return np.reshape(aeq, (-1, self.n)), np.array(beq)

    def project_x0(self):
        if self.m == 0:
            self.x0 = np.minimum(self.xu, np.maximum(self.xl, self.x0))
        elif self.m_linear_ub == 0 and self.m_linear_eq > 0 and np.all(self.xl == -np.inf) and np.all(self.xu == np.inf):
            self.x0 += lstsq(self.aeq, self.beq - np.dot(self.aeq, self.x0))[0]
        else:
            bounds = Bounds(self.xl, self.xu, True)
            constraints = []
            if self.m_linear_ub > 0:
                constraints.append(LinearConstraint(self.aub, -np.inf, self.bub))
            if self.m_linear_eq > 0:
                constraints.append(LinearConstraint(self.aeq, self.beq, self.beq))

            def distance_square(x):
                g = x - self.x0
                f = 0.5 * np.inner(x - self.x0, x - self.x0)
                return f, g

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(distance_square, self.x0, jac=True, bounds=bounds, constraints=constraints)
            self.x0 = np.array(res.x)
