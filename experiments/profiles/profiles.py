import csv
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pycutest
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
from matplotlib.ticker import MaxNLocator
from numpy import ma

from .optimize import Minimizer
from .problem import Problem

_log = logging.getLogger(__name__)

# Set up matplotlib for plotting the profiles.
std_cycle = cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
std_cycle += cycler(linestyle=[(0, ()), (0, (3, 2)), (0, (1, 1)), (0, (3, 2, 1, 2)), (0, (5, 2))])
plt.rc("axes", prop_cycle=std_cycle)
plt.rc("lines", linewidth=1)
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=17)


class Profiles:
    BASE_DIR = Path(__file__).resolve().parent.parent
    ARCH_DIR = Path(BASE_DIR, "archives")
    EXCLUDED = {
        # The compilation of the sources is prohibitively time-consuming.
        "BA-L73", "BA-L73LS", "BDRY2", "CHANDHEU", "CHARDIS0", "CHARDIS1", "DMN15102", "DMN15102LS", "DMN15103", "DMN15103LS", "DMN15332", "DMN15332LS", "DMN15333", "DMN15333LS", "DMN37142", "DMN37142LS", "DMN37143", "DMN37143LS", "EIGENALS", "EIGENBLS", "GPP", "LEUVEN3", "LEUVEN4", "LEUVEN5", "LEUVEN6", "LIPPERT2", "LOBSTERZ", "PDE1", "PDE2", "PENALTY3", "RDW2D51F", "RDW2D51U", "RDW2D52B", "RDW2D52F", "RDW2D52U", "ROSEPETAL", "WALL100", "YATP1SQ", "YATP2SQ", "BA-L16LS", "BA-L21", "BA-L21LS", "BA-L49", "BA-L49LS", "BA-L52LS", "BA-L52", "YATP1LS", "YATP2LS",

        # The starting points contain NaN values.
        "LHAIFAM",

        # The problems contain a lot of NaN.
        "HS62", "HS112", "LIN",

        # The problems seem not lower-bounded.
        "INDEF",

        # The problems are known infeasible.
        "ARGLALE", "ARGLBLE", "ARGLCLE", "MODEL", "NASH",

        # The problems seem infeasible.
        "ANTWERP", "CRESC4", "CRESC50", "DEGENLPA", "DEGENLPB", "DIXCHLNG", "DUALC1", "DUALC2", "DUALC5", "DUALC8", "ELATTAR", "GOULDQP1", "HIMMELBJ", "HONG", "HS8", "HS13", "HS19", "HS55", "HS63", "HS64", "HS72", "HS73", "HS84", "HS86", "HS88", "HS89", "HS92", "HS101", "HS102", "HS103", "HS106", "HS107", "HS109", "HS119", "LOADBAL", "LOTSCHD", "LSNNODOC", "PORTFL1", "PORTFL2", "PORTFL3", "PORTFL4", "PORTFL6", "SNAKE", "SUPERSIM", "TAME", "WACHBIEG",

        # The projection of the initial guess fails.
        "LINCONT",

        # Classical UOBYQA and COBYLA suffer from infinite cycling.
        "GAUSS1LS", "GAUSS2LS", "GAUSS3LS", "MGH17LS", "MISRA1ALS", "MISRA1CLS", "NELSONLS", "OSBORNEA", "RAT43LS",

        # Classical COBYLA suffers from infinite cycling.
        "DANWOODLS", "KOEBHELB",
    }

    def __init__(self, n_min, n_max, constraints, m_min=0, m_max=sys.maxsize, feature="plain", callback=None, **kwargs):
        # All features:
        # 1. plain: the problems are left unmodified.
        # 2. Lq, Lh, L1: an p-regularization term is added to the objective
        #   functions of all problems, with p = 0.25, 0.5, and 1, respectively.
        #   The following keyword arguments may be supplied:
        #   2.1. regularization: corresponding parameter (default is 1.0).
        # 3. noisy: a Gaussian noise is included in the objective functions of
        #   all problems. The following keyword arguments may be supplied:
        #   3.1. noise_type: noise type (default is relative).
        #   3.2. noise_level: standard deviation of the noise (default is 1e-3).
        #   3.3. rerun: number of experiment runs (default is 10).
        # 4. digits[0-9]+: only the first digits of the objective function
        #   values are significant (the other are randomized).
        # 5. nan: the objective function values are sometimes replaced by NaN.
        #   The following keyword arguments may be supplied:
        #   5.1. nan_rate: Rate of NaNs (default is 0.1).
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.max_eval = 500 * self.n_max
        self.constraints = constraints
        self.feature = feature
        self.callback = callback

        # Extract from the keyword arguments the feature options.
        self.feature_options = self.get_feature_options(**kwargs)

        # Determinate the paths of storage.
        n_range = f"{self.n_min}-{self.n_max}"
        self.perf_dir = Path(self.ARCH_DIR, "perf", self.feature, n_range)
        self.storage_dir = Path(self.ARCH_DIR, "storage", self.feature)
        if self.feature != "plain":
            # Suffix the feature's directory name with the corresponding feature's options.
            options_suffix = dict(self.feature_options)
            if self.feature not in ["noisy", "digits", "nan"]:
                del options_suffix["rerun"]
            if self.feature in ["Lq", "Lh", "L1"]:
                del options_suffix["p"]
            options_details = "_".join(f"{k}-{v}" for k, v in options_suffix.items())
            self.perf_dir = Path(self.perf_dir, options_details)
            self.storage_dir = Path(self.storage_dir, options_details)

        # Get the CUTEst problems.
        self.problem_names = sorted(pycutest.find_problems("constant linear quadratic sum of squares other", constraints, True, origin="academic modelling real-world", n=[self.n_min, self.n_max], m=[self.m_min, self.m_max], userM=False))

    def __call__(self, solvers, solver_names=None, options=None, load=True, **kwargs):
        if solver_names is None:
            solver_names = list(solvers)
        if options is None:
            options = [{} for _ in range(len(solvers))]

        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.generate_profiles(solvers, solver_names, options, load, **kwargs)

    def get_feature_options(self, **kwargs):
        signif = re.match(r"digits(\d+)", self.feature)
        options = {"rerun": 1}
        if self.feature in ["Lq", "Lh", "L1"]:
            options["p"] = {"Lq": 0.25, "Lh": 0.5, "L1": 1.0}.get(self.feature)
            options["level"] = kwargs.get("regularization", 1.0)
        elif self.feature == "noisy":
            options["type"] = kwargs.get("noise_type", "relative")
            options["level"] = kwargs.get("noise_level", 1e-3)
            options["rerun"] = int(kwargs.get("rerun", 10))
        elif signif:
            self.feature = "digits"
            options["digits"] = int(signif.group(1))
            options["rerun"] = int(kwargs.get("rerun", 10))
        elif self.feature == "nan":
            options["rate"] = kwargs.get("nan_rate", 0.1)
            options["rerun"] = int(kwargs.get("rerun", 10))
        elif self.feature != "plain":
            raise NotImplementedError
        return options

    def get_profiles_path(self, solvers):
        if not isinstance(solvers, str):
            solvers = "_".join(sorted(solvers)).lower()
        pdf_perf_path = Path(self.perf_dir, f"perf-{solvers}-{self.constraints.replace(' ', '_')}.pdf")
        csv_perf_path = Path(self.perf_dir, f"perf-{solvers}-{self.constraints.replace(' ', '_')}.csv")
        txt_perf_path = Path(self.perf_dir, f"perf-{solvers}-{self.constraints.replace(' ', '_')}.txt")
        return pdf_perf_path, csv_perf_path, txt_perf_path

    def get_storage_path(self, problem_name, solver, k):
        cache = Path(self.storage_dir, problem_name)
        cache.mkdir(exist_ok=True)
        if self.feature_options["rerun"] == 1:
            fun_path = Path(cache, f"fun-hist-{solver.lower()}.npy")
            maxcv_path = Path(cache, f"maxcv-hist-{solver.lower()}.npy")
        else:
            fun_path = Path(cache, f"fun-hist-{solver.lower()}-{k}.npy")
            maxcv_path = Path(cache, f"maxcv-hist-{solver.lower()}-{k}.npy")
        return fun_path, maxcv_path

    def generate_profiles(self, solvers, solver_names, options, load, **kwargs):
        _log.info(f'Starting the computations with feature="{self.feature}"')
        merits, problem_names = self.run_all(solvers, options, load, **kwargs)
        n_problems = merits.shape[0]

        # Constants for the performance profiles.
        log_tau_min = 1
        log_tau_max = 9
        penalty = 2
        ratio_max = 1e-6

        # Evaluate the merit function value at x0.
        f0 = merits[:, 0, :, 0]

        # Determine the least merit function values obtained on each problem.
        f_min = np.min(merits, (1, 3))
        if self.feature in ["noisy", "digits", "nan"]:
            _log.info(f'Starting the computations with feature="plain"')
            rerun_sav = self.feature_options["rerun"]
            feature_sav = self.feature
            self.feature_options["rerun"] = 1
            self.feature = "plain"
            merits_plain, _ = self.run_all(solvers, options, load, **kwargs)
            f_min_plain = np.min(merits_plain, (1, 2, 3))
            f_min = np.minimum(f_min, f_min_plain[:, np.newaxis])
            self.feature_options["rerun"] = rerun_sav
            self.feature = feature_sav

        # Start the performance profile computations.
        pdf_perf_path, csv_perf_path, txt_perf_path = self.get_profiles_path(solvers)
        raw_col = 2 * (log_tau_max - log_tau_min + 1) * len(solvers)
        raw_perf = np.empty((2 * n_problems * self.feature_options["rerun"] + 2, raw_col))
        pdf_perf = backend_pdf.PdfPages(pdf_perf_path)
        for log_tau in range(log_tau_min, log_tau_max + 1):
            _log.info(f"Creating performance profiles with tau = 1e-{log_tau}")
            tau = 10 ** (-log_tau)

            # Determine the number of function evaluations that each solver
            # necessitates on each problem to converge.
            work = np.full((n_problems, len(solvers), self.feature_options["rerun"]), np.nan)
            for i in range(n_problems):
                for j in range(len(solvers)):
                    for k in range(self.feature_options["rerun"]):
                        if np.isfinite(f_min[i, k]):
                            threshold = max(tau * f0[i, k] + (1.0 - tau) * f_min[i, k], f_min[i, k])
                        else:
                            threshold = -np.inf
                        if np.min(merits[i, j, k, :]) <= threshold:
                            index = np.argmax(merits[i, j, k, :] <= threshold)
                            work[i, j, k] = index + 1

            # Calculate the abscissas of the performance profiles.
            perf = np.full((self.feature_options["rerun"], n_problems, len(solvers)), np.nan)
            for k in range(self.feature_options["rerun"]):
                for i in range(n_problems):
                    if not np.all(np.isnan(work[i, :, k])):
                        perf[k, i, :] = work[i, :, k] / np.nanmin(work[i, :, k])
            perf = np.maximum(np.log2(perf), 0.0)
            perf_ratio_max = np.nanmax(perf, initial=ratio_max)
            perf[np.isnan(perf)] = penalty * perf_ratio_max
            perf = np.sort(perf, 1)
            perf = np.reshape(perf, (n_problems * self.feature_options["rerun"], len(solvers)))
            i_sort_perf = np.argsort(perf, 0, "stable")
            perf = np.take_along_axis(perf, i_sort_perf, 0)

            # Calculate the ordinates of the performance profiles.
            y_perf = np.zeros((n_problems * self.feature_options["rerun"], len(solvers)))
            for k in range(self.feature_options["rerun"]):
                for j in range(len(solvers)):
                    y_loc = np.full(n_problems * self.feature_options["rerun"], np.nan)
                    y_loc[k * n_problems:(k + 1) * n_problems] = np.linspace(1 / n_problems, 1.0, n_problems)
                    y_loc = y_loc[i_sort_perf[:, j]]
                    for i in range(n_problems * self.feature_options["rerun"]):
                        if np.isnan(y_loc[i]):
                            y_loc[i] = y_loc[i - 1] if i > 0 else 0.0
                    y_perf[:, j] += y_loc
            y_perf /= self.feature_options["rerun"]

            # Plot the performance profiles.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position("both")
            ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
            ax.yaxis.set_minor_locator(MaxNLocator(10))
            ax.tick_params(direction="in", which="both")
            i_col = 2 * (log_tau - log_tau_min) * len(solvers)
            for j in range(len(solvers)):
                x = np.repeat(perf[:, j], 2)[1:]
                x = np.r_[0, x[0], x, penalty * perf_ratio_max]
                y = np.repeat(y_perf[:, j], 2)[:-1]
                y = np.r_[0, 0, y, y[-1]]
                raw_perf[:, i_col + 2 * j] = x
                raw_perf[:, i_col + 2 * j + 1] = y
                plt.plot(x, y, label=solver_names[j].replace("-", " "))
            plt.xlim(0, 1.1 * perf_ratio_max)
            plt.ylim(0, 1)
            plt.xlabel(r"$\log_2(\mathrm{NF}/\mathrm{NF}_{\min})$")
            plt.ylabel(fr"Performance profiles ($\tau=10^{{-{log_tau}}}$)")
            plt.legend(loc="lower right")
            pdf_perf.savefig(fig, bbox_inches="tight")
            plt.close()

        _log.info("Saving performance profiles.")
        pdf_perf.close()
        with open(csv_perf_path, "w") as fd:
            csv_perf = csv.writer(fd)
            header_perf = np.array(
                [[[f"x{i}_{s}", f"y{i}_{s}"] for s in solvers] for i in range(log_tau_min, log_tau_max + 1)])
            csv_perf.writerow(header_perf.flatten())
            csv_perf.writerows(raw_perf)
        with open(txt_perf_path, "w") as fd:
            fd.write("\n".join(problem_names))

    def run_all(self, solvers, options, load, **kwargs):
        merits = np.full((len(self.problem_names), len(solvers), self.feature_options["rerun"], self.max_eval), np.nan)
        n_success = 0
        problem_names = []
        for problem_name in self.problem_names:
            for j, solver in enumerate(solvers):
                for k in range(self.feature_options["rerun"]):
                    result = self.run_one(problem_name, solver, k, options[j], load, **kwargs)
                    if result is None:
                        break
                    if j == 0 and k == 0:
                        n_success += 1
                        problem_names.append(problem_name)
                    merits[n_success - 1, j, k, :] = np.r_[result, np.full(self.max_eval - result.size, result[-1])]
                else:
                    continue
                break
        return merits[:n_success, ...], problem_names

    def run_one(self, problem_name, solver, k, options, load, **kwargs):
        storage_name = problem_name
        if pycutest.problem_properties(problem_name)["n"] == "variable":
            storage_name = f"{problem_name}_N{self.get_sif_n_max(problem_name)}"
        fun_path, maxcv_path = self.get_storage_path(storage_name, solver, k)
        if load and fun_path.is_file() and maxcv_path.is_file():
            fun_history = np.load(fun_path)
            maxcv_history = np.load(maxcv_path)
            merits = self.merit(fun_history, maxcv_history, **kwargs)
            if self.max_eval < merits.size:
                merits = merits[:self.max_eval]
        else:
            problem = self.load(problem_name)
            if problem is None:
                _log.warning(f"Problem {problem_name} is not available")
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimizer = Minimizer(problem, solver, self.max_eval, options, self.noise, k)
                fun_history, maxcv_history = optimizer()
                n_eval = min(fun_history.size, self.max_eval)
                merits = self.merit(fun_history[:n_eval], maxcv_history[:n_eval], **kwargs)
                np.save(fun_path, fun_history[:n_eval])
                np.save(maxcv_path, maxcv_history[:n_eval])
        if not np.all(np.isnan(merits)):
            if self.feature_options["rerun"] > 1:
                run_description = f"{solver}({problem_name},{k})"
            else:
                run_description = f"{solver}({problem_name})"
            i = np.argmin(merits)
            _log.info(f"{run_description}: fun = {fun_history[i]}, maxcv = {maxcv_history[i]}, n_eval = {merits.size}")
        else:
            _log.warning(f"{solver}({problem_name}): no value received")
        return merits

    def load(self, problem_name):
        try:
            if problem_name not in self.EXCLUDED:
                _log.info(f"Loading {problem_name}")

                # If the problem's dimension is not fixed, we select the largest possible dimension.
                if pycutest.problem_properties(problem_name)["n"] == "variable":
                    sif_n_max = self.get_sif_n_max(problem_name)
                    if sif_n_max is not None:
                        return Problem(problem_name, sifParams={"N": sif_n_max})
                else:
                    return Problem(problem_name)
        except (AttributeError, FileNotFoundError, ModuleNotFoundError, RuntimeError) as err:
            _log.warning(f"{problem_name}: {err}")

    def get_sif_n_max(self, name):
        # Get all the available SIF parameters for all variables.
        cmd = [pycutest.get_sifdecoder_path(), "-show", name]
        sp = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        sif_stdout = sp.stdout.read()
        sp.wait()

        # Extract all the available SIF parameters for the problem's dimension.
        regex = re.compile(r"^N=(?P<param>\d+)")
        sif_n = []
        for stdout in sif_stdout.split("\n"):
            sif_match = regex.match(stdout)
            if sif_match:
                sif_n.append(int(sif_match.group("param")))
        sif_n = np.sort(sif_n)

        # Check the requirements.
        sif_n_masked = ma.masked_array(sif_n, mask=(sif_n < self.n_min) | (sif_n > self.n_max))
        if sif_n_masked.size > 0:
            sif_n_max = sif_n_masked.max()
            if sif_n_max is not ma.masked:
                return sif_n_max
        return None

    @staticmethod
    def merit(fun_history, maxcv_history, **kwargs):
        fun_history = np.atleast_1d(fun_history)
        maxcv_history = np.atleast_1d(maxcv_history)
        merits = np.empty_like(fun_history)
        for i in range(merits.size):
            if maxcv_history[i] <= kwargs.get("low_cv", 1e-12):
                merits[i] = fun_history[i]
            elif kwargs.get("barrier", False) and maxcv_history[i] >= kwargs.get("high_cv", 1e-6):
                merits[i] = np.inf
            else:
                merits[i] = fun_history[i] + kwargs.get("penalty", 1e8) * maxcv_history[i]
        return merits

    def noise(self, x, f, k=0):
        if self.feature in ["Lq", "Lh", "L1"]:
            f += self.feature_options["level"] * np.linalg.norm(x, self.feature_options["p"])
        elif self.feature == "noisy":
            rng = np.random.default_rng(int(1e8 * abs(np.sin(k) + np.sin(self.feature_options["level"]) + np.sum(np.sin(np.abs(np.sin(1e8 * x)))))))
            noise = self.feature_options["level"] * rng.standard_normal()
            if self.feature_options["type"] == "absolute":
                f += noise
            else:
                f *= 1.0 + noise
        elif self.feature == "digits" and np.isfinite(f):
            if f == 0.0:
                fx_rounded = 0.0
            else:
                fx_rounded = round(f, self.feature_options["digits"] - int(np.floor(np.log10(np.abs(f)))) - 1)
            f = fx_rounded + (f - fx_rounded) * np.abs(np.sin(
                np.sin(np.sin(self.feature_options["digits"])) + np.sin(1e8 * f) + np.sum(np.sin(np.abs(1e8 * x))) + np.sin(x.size)))
        elif self.feature == "nan":
            rng = np.random.default_rng(int(1e8 * abs(np.sin(k) + np.sin(self.feature_options["rate"]) + np.sum(np.sin(np.abs(np.sin(1e8 * x)))))))
            if rng.uniform() <= self.feature_options["rate"]:
                f = np.nan
        return f
