import numpy as np
import pdfo
import pdfonobarriers
from scipy import optimize


class Minimizer:
    def __init__(self, problem, solver, max_eval, options, callback, fd_step, *args, **kwargs):
        self.problem = problem
        self.solver = solver
        self.max_eval = max_eval
        self.options = dict(options)
        self.callback = callback
        self.fd_step = fd_step
        self.args = args
        self.kwargs = kwargs
        if not self.validate():
            raise NotImplementedError

        # The following attributes store the objective function and maximum constraint violation values.
        self.fun_history = None
        self.maxcv_history = None

    def __call__(self):
        self.fun_history = []
        self.maxcv_history = []

        options = dict(self.options)
        if self.solver.lower() in pdfo.__all__:
            method = self.solver if self.solver.lower() != "pdfo" else None
            bounds = optimize.Bounds(self.problem.xl, self.problem.xu)
            constraints = []
            if self.problem.m_linear_ub > 0:
                constraints.append(optimize.LinearConstraint(self.problem.aub, -np.inf, self.problem.bub))
            if self.problem.m_linear_eq > 0:
                constraints.append(optimize.LinearConstraint(self.problem.aeq, self.problem.beq, self.problem.beq))
            if self.problem.m_nonlinear_ub > 0:
                constraints.append(optimize.NonlinearConstraint(self.problem.cub, -np.inf, np.zeros(self.problem.m_nonlinear_ub)))
            if self.problem.m_nonlinear_eq > 0:
                constraints.append(optimize.NonlinearConstraint(self.problem.ceq, np.zeros(self.problem.m_nonlinear_eq), np.zeros(self.problem.m_nonlinear_eq)))
            options["maxfev"] = self.max_eval
            options["eliminate_lin_eq"] = False
            pdfo.pdfo(self.eval, self.problem.x0, method=method, bounds=bounds, constraints=constraints, options=options)
        elif self.solver.lower() == "pdfo-(no-barrier)":
            bounds = pdfonobarriers.Bounds(self.problem.xl, self.problem.xu)
            constraints = []
            if self.problem.m_linear_ub > 0:
                constraints.append(pdfonobarriers.LinearConstraint(self.problem.aub, -np.inf, self.problem.bub))
            if self.problem.m_linear_eq > 0:
                constraints.append(pdfonobarriers.LinearConstraint(self.problem.aeq, self.problem.beq, self.problem.beq))
            if self.problem.m_nonlinear_ub > 0:
                constraints.append(pdfonobarriers.NonlinearConstraint(self.problem.cub, -np.inf, np.zeros(self.problem.m_nonlinear_ub)))
            if self.problem.m_nonlinear_eq > 0:
                constraints.append(pdfonobarriers.NonlinearConstraint(self.problem.ceq, np.zeros(self.problem.m_nonlinear_eq), np.zeros(self.problem.m_nonlinear_eq)))
            options["maxfev"] = self.max_eval
            options["eliminate_lin_eq"] = False
            pdfonobarriers.pdfonobarriers(self.eval, self.problem.x0, bounds=bounds, constraints=constraints, options=options)
        else:
            bounds = optimize.Bounds(self.problem.xl, self.problem.xu)
            constraints = []
            if self.problem.m_linear_ub > 0:
                constraints.append(optimize.LinearConstraint(self.problem.aub, -np.inf, self.problem.bub))
            if self.problem.m_linear_eq > 0:
                constraints.append(optimize.LinearConstraint(self.problem.aeq, self.problem.beq, self.problem.beq))
            if self.problem.m_nonlinear_ub > 0:
                constraints.append(optimize.NonlinearConstraint(self.problem.cub, -np.inf, np.zeros(self.problem.m_nonlinear_ub)))
            if self.problem.m_nonlinear_eq > 0:
                constraints.append(optimize.NonlinearConstraint(self.problem.ceq, np.zeros(self.problem.m_nonlinear_eq), np.zeros(self.problem.m_nonlinear_eq)))
            options["maxiter"] = self.max_eval
            optimize.minimize(self.eval, self.problem.x0, method=self.solver, jac=self.grad, bounds=bounds, constraints=constraints, options=options)
        return np.array(self.fun_history, copy=True), np.array(self.maxcv_history, copy=True)

    def validate(self):
        valid_solvers = {"cobyla", "pdfo", "pdfo-(no-barrier)"}
        if self.problem.type not in "quadratic other":
            valid_solvers.update({"lincoa"})
            if self.problem.type not in "adjacency linear":
                valid_solvers.update({"bobyqa"})
                if self.problem.type not in "equality bound":
                    valid_solvers.update({"bfgs", "bfgs-adaptive", "cg", "cg-adaptive", "newuoa", "uobyqa"})
        return self.solver.lower() in valid_solvers

    def eval(self, x):
        f = self.problem.fun(x, self.callback, *self.args, **self.kwargs)
        if self.callback is not None:
            # If the objective function is noisy, it returns both the plain and the noisy function evaluations.
            self.fun_history.append(f[0])
            f = f[1]
        else:
            self.fun_history.append(f)
        self.maxcv_history.append(self.problem.maxcv(x))
        return f

    def grad(self, x):
        f = self.problem.fun(x, self.callback, *self.args, **self.kwargs)
        if self.callback is not None:
            f = f[1]
        g = np.empty(x.size)
        for i in range(x.size):
            coord_vec = np.squeeze(np.eye(1, x.size, i))
            f_forward = self.eval(x + self.fd_step * coord_vec)
            g[i] = (f_forward - f) / self.fd_step
        return g
