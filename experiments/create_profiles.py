import csv
import os
import re
import subprocess
from pathlib import Path


def get_solvers(csvfile, tau):
    with open(csvfile, "r") as fd:
        reader = csv.reader(fd, delimiter=",")
        header = next(reader)
    pattern = re.compile(f"x{tau}_(?P<solver>[\S]+)")
    solvers = []
    for column in header:
        match = pattern.match(column)
        if match:
            solver = '"' + match.group("solver") + '"'
            if solver not in solvers:
                solvers.append(solver)
    return "{" + ",".join(solvers) + "}"


def create_figure(csvfile, pdffile, tau, archive=Path(os.path.dirname(os.getcwd()), "figures")):
    csvfile = Path(csvfile)
    pdffile = Path(pdffile)
    solvers = get_solvers(csvfile, tau)
    tex_standalone = rf"""\documentclass[crop]{{standalone}}
\usepackage{{mathptmx}}
\usepackage{{amsmath}}
\usepackage[dvipsnames]{{xcolor}}

% Enhanced support for graphics
\usepackage{{pgfplots}}
\usepackage{{pgfplotstable}}
\pgfplotsset{{compat=1.18}}
\pgfplotscreateplotcyclelist{{profiles}}{{%
    thick,mark=none,NavyBlue,solid\\%
    thick,mark=none,BurntOrange,dashed\\%
    thick,mark=none,OliveGreen,dotted\\%
    thick,mark=none,BrickRed,dashdotted\\%
    thick,mark=none,Purple,densely dashed\\%
    thick,mark=none,Mahogan,densely dashed\\%
    thick,mark=none,Rhodamine,densely dashdotted\\%
    thick,mark=none,Gray,loosely dashed\\%
    thick,mark=none,LimeGreen,loosely dotted\\%
    thick,mark=none,JungleGreen,loosely dashdotted\\%
}}
\pgfplotsset{{%
    every axis/.append style={{%
        label style={{font=\small}},%
        legend cell align=left,%
        legend style={{rounded corners,thick,draw=black!15,font=\small}},%
        tick label style={{font=\small}},%
    }},%
}}

\newcommand{{\insertprofiles}}[3]{{%
    \def\selectsolvers{{#1}}%
    \def\selectcsv{{#2}}%
    \def\selectprofile{{#3}}%
    \input{{profiles.tex}}%
}}

\begin{{document}}
\insertprofiles{{{solvers}}}{{{csvfile}}}{{{tau}}}
\end{{document}}
    """
    with open(pdffile.stem + ".tex", "w") as fd:
        fd.write(tex_standalone)
    command = ["pdflatex", pdffile.stem + ".tex"]
    process = subprocess.Popen(command)  # , stdout=subprocess.DEVNULL
    process.wait()
    os.remove(pdffile.stem + ".aux")
    os.remove(pdffile.stem + ".log")
    os.remove(pdffile.stem + ".tex")
    Path(archive).mkdir(parents=True, exist_ok=True)
    figure = Path(archive, pdffile)
    os.rename(pdffile.stem + ".pdf", figure)
    print(figure)


def create_all_figures():
    for prec in range(1, 10):
        create_figure("archives/perf/plain/1-10/perf-bobyqa_cobyla_lincoa_newuoa_uobyqa-unconstrained.csv", f"perf-plain-pdfo-10-{prec}.pdf", prec)
        create_figure("archives/perf/plain/1-50/perf-bfgs_cg_newuoa-unconstrained.csv", f"perf-plain-bfgs_cg_pdfo-50-{prec}.pdf", prec)
        create_figure("archives/perf/plain/1-50/perf-bobyqa_cobyla_lincoa_newuoa-unconstrained.csv", f"perf-plain-pdfo-50-{prec}.pdf", prec)
        create_figure("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-06/perf-bfgs_cg_newuoa-unconstrained.csv", f"perf-noisy-bfgs_cg_pdfo-50-6-{prec}.pdf", prec)
        create_figure("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-08/perf-bfgs_cg_newuoa-unconstrained.csv", f"perf-noisy-bfgs_cg_pdfo-50-8-{prec}.pdf", prec)
        create_figure("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-10/perf-bfgs_cg_newuoa-unconstrained.csv", f"perf-noisy-bfgs_cg_pdfo-50-10-{prec}.pdf", prec)
        create_figure("archives/perf/noisy/1-50/rerun-10_type-relative_level-0.01/perf-bobyqa_cobyla_lincoa_newuoa-unconstrained.csv", f"perf-noisy-pdfo-50-2-{prec}.pdf", prec)


if __name__ == "__main__":
    create_all_figures()
