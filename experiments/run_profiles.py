import os

from create_profiles import create_all_figures
from profiles import Profiles

CREATE_PAPER_FIGURES = False


if __name__ == "__main__":
    cwd = os.getcwd()

    # Generate the performance and data profiles on the plain problems with n <= 10.
    profiles = Profiles(1, 10, "unconstrained")
    profiles(["newuoa", "bobyqa", "lincoa", "cobyla", "uobyqa"])
    del profiles

    # Generate the performance and data profiles on the plain problems with n <= 50.
    profiles = Profiles(1, 50, "unconstrained")
    profiles(["newuoa", "bfgs", "cg"])
    profiles(["newuoa", "bobyqa", "lincoa", "cobyla"])
    del profiles

    # Generate the performance and data profiles on the noisy problems with n <= 50 and different noise levels.
    for noise_level in [1e-10, 1e-8, 1e-6]:
        profiles = Profiles(1, 50, "unconstrained", feature="noisy", noise_level=noise_level)
        profiles(["newuoa", "bfgs", "cg"])
        del profiles
    profiles = Profiles(1, 50, "unconstrained", feature="noisy", noise_level=1e-2)
    profiles(["newuoa", "bobyqa", "lincoa", "cobyla"])
    del profiles

    # Generate the profiles for the paper.
    os.chdir(cwd)
    if CREATE_PAPER_FIGURES:
        create_all_figures()
