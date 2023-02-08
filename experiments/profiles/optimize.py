import numpy as np
import pdfo
import pdfonobarriers
from scipy import optimize


class Minimizer:
    def __init__(self, problem, solver, max_eval, options, callback, *args, **kwargs):
        self.problem = problem
        self.solver = solver
        self.max_eval = max_eval
        self.options = dict(options)
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        if not self.validate():
            raise NotImplementedError

        # The following attributes store the objective function and maximum
        # constraint violation values obtained during a run.
        self.fun_history = None
        self.maxcv_history = None

    def __call__(self):
        self.fun_history = []
        self.maxcv_history = []

        options = dict(self.options)
        if self.solver.lower() in pdfo.__all__:
            method = self.solver if self.solver.lower() != "pdfo" else None
            bounds = pdfo.Bounds(self.problem.xl, self.problem.xu)
            constraints = []
            if self.problem.m_lin_ineq > 0:
                constraints.append(pdfo.LinearConstraint(self.problem.a_ineq, -np.inf, self.problem.b_ineq))
            if self.problem.m_lin_eq > 0:
                constraints.append(pdfo.LinearConstraint(self.problem.a_eq, self.problem.b_eq, self.problem.b_eq))
            if self.problem.m_nonlin_ineq > 0:
                constraints.append(pdfo.NonlinearConstraint(self.problem.c_ineq, -np.inf, np.zeros(self.problem.m_nonlin_ineq)))
            if self.problem.m_nonlin_eq > 0:
                constraints.append(pdfo.NonlinearConstraint(self.problem.c_eq, np.zeros(self.problem.m_nonlin_eq), np.zeros(self.problem.m_nonlin_eq)))
            options["maxfev"] = self.max_eval
            options["eliminate_lin_eq"] = False
            res = pdfo.pdfo(self.eval, self.problem.x0, method=method, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        elif self.solver.lower() == "pdfo-(no-barrier)":
            bounds = pdfonobarriers.Bounds(self.problem.xl, self.problem.xu)
            constraints = []
            if self.problem.m_lin_ineq > 0:
                constraints.append(pdfonobarriers.LinearConstraint(self.problem.a_ineq, -np.inf, self.problem.b_ineq))
            if self.problem.m_lin_eq > 0:
                constraints.append(pdfonobarriers.LinearConstraint(self.problem.a_eq, self.problem.b_eq, self.problem.b_eq))
            if self.problem.m_nonlin_ineq > 0:
                constraints.append(pdfonobarriers.NonlinearConstraint(self.problem.c_ineq, -np.inf, np.zeros(self.problem.m_nonlin_ineq)))
            if self.problem.m_nonlin_eq > 0:
                constraints.append(pdfonobarriers.NonlinearConstraint(self.problem.c_eq, np.zeros(self.problem.m_nonlin_eq), np.zeros(self.problem.m_nonlin_eq)))
            options["maxfev"] = self.max_eval
            options["eliminate_lin_eq"] = False
            res = pdfonobarriers.pdfonobarriers(self.eval, self.problem.x0, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        else:
            bounds = optimize.Bounds(self.problem.xl, self.problem.xu)
            constraints = []
            if self.problem.m_lin_ineq > 0:
                constraints.append(optimize.LinearConstraint(self.problem.a_ineq, -np.inf, self.problem.b_ineq))
            if self.problem.m_lin_eq > 0:
                constraints.append(optimize.LinearConstraint(self.problem.a_eq, self.problem.b_eq, self.problem.b_eq))
            if self.problem.m_nonlin_ineq > 0:
                constraints.append(optimize.NonlinearConstraint(self.problem.c_ineq, -np.inf, np.zeros(self.problem.m_nonlin_ineq)))
            if self.problem.m_nonlin_eq > 0:
                constraints.append(optimize.NonlinearConstraint(self.problem.c_eq, np.zeros(self.problem.m_nonlin_eq), np.zeros(self.problem.m_nonlin_eq)))
            options["maxiter"] = self.max_eval
            res = optimize.minimize(self.eval, self.problem.x0, method=self.solver, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        return success, np.array(self.fun_history, copy=True), np.array(self.maxcv_history, copy=True)

    def validate(self):
        valid_solvers = {"cobyla", "pdfo", "pdfo-(no-barrier)"}
        if self.problem.type not in "quadratic other":
            valid_solvers.update({"lincoa"})
            if self.problem.type not in "adjacency linear":
                valid_solvers.update({"bobyqa"})
                if self.problem.type not in "equality bound":
                    valid_solvers.update({"bfgs", "cg", "newuoa", "uobyqa"})
        return self.solver.lower() in valid_solvers

    def eval(self, x):
        f = self.problem.fun(x, self.callback, *self.args, **self.kwargs)
        if self.callback is not None:
            # If a noise function is supplied, the objective function returns
            # both the plain and the noisy function evaluations. We return the
            # noisy function evaluation, but we store the plain function
            # evaluation (used to build the performance and data profiles).
            self.fun_history.append(f[0])
            f = f[1]
        else:
            self.fun_history.append(f)
        self.maxcv_history.append(self.problem.maxcv(x))
        return f
