import os

from create_profiles import create_all_figures
from profiles import Profiles

CREATE_PAPER_FIGURES = False


if __name__ == "__main__":
    cwd = os.getcwd()

    # Generate the performance and data profiles on the plain problems with n <= 10.
    profiles = Profiles(1, 10, "unconstrained")
    profiles(["NEWUOA", "BOBYQA", "LINCOA", "COBYLA", "UOBYQA"])
    del profiles

    # Generate the performance and data profiles on the plain problems with n <= 50.
    profiles = Profiles(1, 50, "unconstrained")
    profiles(["NEWUOA", "BFGS", "CG"])
    profiles(["NEWUOA", "BOBYQA", "LINCOA", "COBYLA"])
    del profiles

    # Generate the performance and data profiles on the problems containing NaNs.
    for rerun in [1, 10]:
        profiles = Profiles(1, 50, "unconstrained", feature="nan", nan_rate=0.01, rerun=rerun)
        profiles(["PDFO", "PDFO-(no-barrier)", "BFGS", "CG"], load=False)
        del profiles

    # Generate the performance and data profiles on the noisy problems with n <= 50 and different noise levels.
    for noise_level in [1e-10, 1e-8, 1e-6]:
        profiles = Profiles(1, 50, "unconstrained", feature="noisy", noise_level=noise_level)
        profiles(["NEWUOA", "BFGS", "CG"])
        del profiles
    for noise_level in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        profiles = Profiles(1, 50, "unconstrained", feature="noisy", noise_level=noise_level)
        profiles(["NEWUOA", "BOBYQA", "LINCOA", "COBYLA"])
        del profiles

    # Generate the profiles for the paper.
    os.chdir(cwd)
    if CREATE_PAPER_FIGURES:
        create_all_figures()
