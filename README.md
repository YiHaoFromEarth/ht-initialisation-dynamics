# Exploring Extended Criticality in Heavy-Tailed Neural Networks

This repository contains the numerical frameworks, simulation code, and manuscript artifacts for an Honours research project in the School of Physics at the University of Sydney. The study investigates the non-equilibrium dynamics and transport physics of deep multilayer perceptrons (MLPs) initialized with non-Gaussian, heavy-tailed weight distributions.

## Research Overview

Standard deep learning initialization assumes Gaussian weight statistics. This project explores the **extended critical regime** enabled by symmetric $\alpha$-stable distributions ($\alpha < 2.0$). Key areas of investigation include:

* **Dynamical Phase Separation:** Identifying functional decoupling between shallow stochastic reservoirs and terminal ballistic classifiers.
* **Transport Kinetics:** Quantifying weight manifold evolution using Time-Averaged Mean Squared Displacement (TAMSD) and diffusion exponents ($\gamma$).
* **Structural Evolution:** Tracking spectral compression, stable rank, and angular realignment through training.
* **Staircase Learning:** Validating discrete performance jumps and Lévy flights in traditionally ordered learning regimes.

## Repository Structure

The project is organized into modular components for simulation, analysis, and documentation:

```text
.
├── src/                # Core library for ht-initialization, RMT analysis, and training
├── docs/               # LaTeX source and PDFs for thesis, manuscript, and research plans
├── results/            # Compiled figures (spectral norms, correlations, diffusion curves)
├── training_runs/      # Serialized hyperparameter sweeps (MNIST/CIFAR sweeps)
├── test/               # Pytest suite for analysis and ht-library validation
├── eda.ipynb           # Exploratory data analysis for dynamical observables
└── environment.yml     # Conda environment specification
```

## Methodology

### Initialization and Scaling
Weight matrices $W^l$ are initialized using symmetric $\alpha$-stable distributions. To maintain stable signal propagation across deep architectures, we implement a normalized scaling law:
$$\sigma = \frac{g}{(2N_{eff})^{1/\alpha}}$$
where $N_{eff}$ accounts for specific layer geometries (linear or convolutional).

### Dynamical Observables
The framework tracks layer-wise displacement trajectories to distinguish between stochastic noise and directed structural realignment:
* **Net Displacement:** Captures collective centre-of-mass drift.
* **$L_2$ Energy:** Detects high-magnitude Lévy flights.
* **Spectral and Geometric Metrics:** Monitors the emergence of low-rank manifolds and dominant features.

## Dependencies and Usage

All simulations are executed using the **PyTorch** framework on NVIDIA hardware.

1.  **Environment Setup:**
    ```bash
    conda env create -f environment.yml
    conda activate ht_dynamics
    ```
2.  **Analysis:**
    Computational notebooks for spectral and dynamical analysis are provided in the root directory.

---
**Author:** Yi Hao
**Institution:** School of Physics, The University of Sydney
**Date:** April 2026
