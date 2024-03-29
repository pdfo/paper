"""
To run this script, you just need to run:
>>> python run_profiles.py
"""
import os

from copy_profiles import copy_all_figures
from profiles import Profiles, set_loglevel

COPY_PAPER_FIGURES = True


if __name__ == "__main__":
    cwd = os.getcwd()
    set_loglevel("INFO")

    # Generate the performance and data profiles on the plain problems with n <= 50.
    profiles = Profiles(1, 50, "unconstrained")
    profiles(["NEWUOA", "CG", "BFGS"], ["PDFO", "CG", "BFGS"], load=False)

    # Generate the performance and data profiles on the noisy problems with n <= 50 and different noise levels.
    for noise_level in [1e-10, 1e-8]:
        profiles = Profiles(1, 50, "unconstrained", feature="noisy", noise_level=noise_level)
        profiles(["NEWUOA", "CG", "BFGS", "CG-adaptive", "BFGS-adaptive"], ["PDFO", "CG", "BFGS", "CG-adaptive", "BFGS-adaptive"], load=False)

    # Generate the performance and data profiles on the problems containing NaNs.
    for nan_rate in [0.01, 0.05]:
        profiles = Profiles(1, 50, "unconstrained", feature="nan", nan_rate=nan_rate)
        profiles(["PDFO", "CG", "BFGS", "PDFO-(no-barrier)"], load=False)

    # Generate the profiles for the paper.
    os.chdir(cwd)
    if COPY_PAPER_FIGURES:
        copy_all_figures()
