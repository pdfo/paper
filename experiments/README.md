# PDFO experiments

This repository contains the code for the experiments in the paper "PDFO — A Cross-Platform Package for Powell's Derivative-Free Optimization Solvers" by [Tom M. Ragonneau](https://tomragonneau.com/) and [Zaikun Zhang](https://www.zhangzk.net/).

## Getting started

To run the experiments, you will need to install [Python 3.8 or above](https://www.python.org/).
In what follows, we assume that your Python installation is called `python`.
Change it to `python3` (or others) if necessary.

### External dependencies

The profiling experiment uses the [PyCUTEst](https://jfowkes.github.io/pycutest/) package, which requires external dependencies. 
See the [PyCUTEst documentation](https://jfowkes.github.io/pycutest/_build/html/install.html) for more information.

### Python dependencies

To install the Python dependencies, run the following command:

```bash
python -m pip install -r requirements.txt
```



## Hyperparameter tuning experiment

To run the hyperparameter tuning experiment, run the following command:

```bash
python run_hyperparameter_tuning.py
```

The results will be printed to the standard output.
You may redirect the output to a file using the `>` operator if necessary.

> **Warning**
> This experiment takes a long time to run.

## Profiling experiment

To run the profiling experiment, run the following command:

```bash
python run_profiles.py
```

This will create a directory called `archives` containing the profiling results.
