# Numerical Experiments for Globally Convergent Derivative-Free Methods in Nonconvex Optimization

This package provides the numerical experiments presented in the paper:

**PD Khanh, BS Mordukhovich, DB Tran**  
"Globally Convergent Derivative-Free Methods in Nonconvex Optimization with and without Noise"  
[Optimization Online Link](https://optimization-online.org/?p=26889)

---

## I. Installation

To run the experiments, please install PyCUTEst in advance. Installation instructions can be found [here](https://jfowkes.github.io/pycutest/_build/html/index.html).

**Note:** Since CUTEst is currently available only on macOS and Linux, Windows users are recommended to install the Windows Subsystem for Linux (WSL). Detailed instructions for WSL installation are available [here](https://learn.microsoft.com/en-us/windows/wsl/install).

---

## II. Numerical Experiments

This repository includes Jupyter notebooks for the following experiments:

- **`exp_fig2.ipynb`** - Experiment presented in Figure 2
- **`exp_C1L.ipynb`** - Experiment for C1L functions presented in Subsection 6.1.1
- **`exp_C11small.ipynb`** - Experiment for C1L functions with small noise, presented in Subsection 6.1.2
- **`exp_C11large.ipynb`** - Experiment for C11 functions with large noise, presented in Subsection 6.2

---

## III. Source Code

The source code for each algorithm used in the experiments is organized as follows:

- **`adapGD.py`** - Source code for Algorithm GD-BD (Ada), used in the experiment in Figure 2.
- **`dfree.py`** - Source code for DFBD, used in the experiment for C11 functions presented in Subsection 6.2.
- **`dfreegraph.py`** - Source code for DFBD with additional output of the iterative sequence, used in the experiment in Figure 2.
- **`handcode.py`** - Source code for all other algorithms used in Subsections 6.1.2 and 6.2.
- **`handcodegraph.py`** - Similar to `handcode.py`, but with additional output of the iterative sequence for use in the experiment in Figure 2.
- **`ntqngraph.py`** - L-BFGS (Ada) used in the experiment in Figure 2.

---

If you have any questions or encounter any issues, please feel free to reach out to me at [tranbadat@wayne.edu](mailto:tranbadat@wayne.edu).
