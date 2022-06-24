# Quantum Volume Test Simulation Tools

`qvtsim` contains methods needed to recreate some of the simulations and analysis in <a href="https://arxiv.org/pdf/2110.14808.pdf">arXiv:2110.14808</a> of the quantum volume test. Most methods are based around and compatible with `qiskit`'s internal quantum volume methods.

## Project Organization
------------

    ├── README.md          
    ├── /notebooks          <- Jupyter notebook examples
    └── /qvtsim             <- Python source code

--------

## Getting Started

The repository contains two example notebooks for running the simulations:

- `QVT large sample.ipynb`: runs the scalable and numerical simulations over a set of error magnitudes for a given error model and list of qubits. It also performs the confidence interval analysis and estimates the passing error magnitudes.
- `QV fitter example.ipynb`: runs qiskit's standard QV experiment and analyzes with the QVFitter class but supplements with new functions to analyze the data and calculate the bootstrap confidence interval.
