# Magnetic Critical Behavior & MCE Analyzer

A comprehensive Python-based tool suite (featuring both GUI and CLI) designed for analyzing magnetic phase transitions, extracting critical exponents (β, γ, δ, Tc), and evaluating the Magnetocaloric Effect (MCE).

## Key Features
* **Dual Interface**: Includes an interactive PyQt5 dashboard and headless scripts for automated batch processing.
* **Critical Scaling Analysis**: Implements Modified Arrott plots, Kouvel-Fisher method, and quantitative data-collapse optimization.
* **Advanced Fitting**: Supports Universal Equation of State (EOS) parametric fitting and Gaussian Process (Bayesian) scaling.
* **MCE Evaluation**: Calculates magnetic entropy change (ΔSm) with advanced smoothing and uncertainty propagation.
* **Robust Uncertainties**: Built-in Bootstrap resampling and Metropolis MCMC routines.
* **Universality Auto-Scoring**: Automatically compares your dataset against standard universality classes (Mean-field, 3D Heisenberg, 3D Ising, etc.).

## Project Structure
* `GUI_Final_v2_kferr_FIXED_v26.py`: The main Graphical User Interface. Run this file to launch the interactive analyzer.
* `FinalArrot+KFplot_v2.py`: CLI runner for headless analysis and classic Kouvel-Fisher literature validation.
* `linear_fitting_with_gammas_and_betas_v2_2D.py`: Diagnostic CLI tool for scanning and suggesting initial β and γ parameters.
* `no_intercept_at_the_plot_several_data_sets_v2_2D.py`: Helper script to optimize background correction (Bc) across datasets.

## Quick Start

**1. Launch the GUI:**
```bash
python GUI_Final_v2_kferr_FIXED_v26.py

** 2. Run Headless CLI: **
python FinalArrot+KFplot_v2.py --data ./your_data/*.txt --bc 0.2 --out ./output_dir --use-eos --joint-fit

Dependencies
numpy, scipy, matplotlib
PyQt5 (required for the GUI)
pandas (recommended for robust file I/O)
scikit-learn (required for Gaussian Process features)
