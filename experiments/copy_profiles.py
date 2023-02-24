import os
from shutil import copyfile

def copy_all_figures():
    os.makedirs("../figures", exist_ok=True)
    copyfile("archives/perf/plain/1-50/perf-bfgs_cg_newuoa-unconstrained.pdf", "../figures/perf-plain-bfgs_cg_pdfo-50.pdf")
    copyfile("archives/perf/nan/1-50/rerun-1_rate-0.01/perf-bfgs_cg_pdfo_pdfo-(no-barrier)-unconstrained.pdf", "../figures/perf-nan-bfgs_cg_pdfo-50-1-0.01.pdf")
    copyfile("archives/perf/nan/1-50/rerun-10_rate-0.01/perf-bfgs_cg_pdfo_pdfo-(no-barrier)-unconstrained.pdf", "../figures/perf-nan-bfgs_cg_pdfo-50-10-0.01.pdf")
    copyfile("archives/perf/nan/1-50/rerun-1_rate-0.05/perf-bfgs_cg_pdfo_pdfo-(no-barrier)-unconstrained.pdf", "../figures/perf-nan-bfgs_cg_pdfo-50-1-0.05.pdf")
    copyfile("archives/perf/nan/1-50/rerun-10_rate-0.05/perf-bfgs_cg_pdfo_pdfo-(no-barrier)-unconstrained.pdf", "../figures/perf-nan-bfgs_cg_pdfo-50-10-0.05.pdf")
    copyfile("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-06/perf-bfgs_cg_newuoa-unconstrained.pdf", "../figures/perf-noisy-bfgs_cg_pdfo-50-6.pdf")
    copyfile("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-08/perf-bfgs_cg_newuoa-unconstrained.pdf", "../figures/perf-noisy-bfgs_cg_pdfo-50-8.pdf")
    copyfile("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-10/perf-bfgs_cg_newuoa-unconstrained.pdf", "../figures/perf-noisy-bfgs_cg_pdfo-50-10.pdf")


if __name__ == "__main__":
    copy_all_figures()
