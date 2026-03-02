
# Magnetic Critical Behavior & MCE Analyzer

A comprehensive Python-based tool suite (featuring both a GUI and CLI) designed for analyzing magnetic phase transitions, extracting critical exponents ($\beta$, $\gamma$, $\delta$, $T_c$), and evaluating the Magnetocaloric Effect (MCE).

## Key Features
* **Dual Interface**: Includes an interactive PyQt5 dashboard for visual analysis and headless CLI scripts for automated batch processing.
* **Critical Scaling Analysis**: Implements Modified Arrott plots, the Kouvel-Fisher method, and quantitative data-collapse optimization.
* **Advanced Fitting**: Supports Universal Equation of State (EOS) parametric fitting (Mean-field, 3D Heisenberg, 3D Ising, etc.) and Gaussian Process (Bayesian) scaling.
* **MCE Evaluation**: Calculates magnetic entropy change ($\Delta S_M$) with advanced smoothing, cumulative trapezoid integration, and uncertainty propagation.
* **Robust Uncertainties**: Built-in Bootstrap resampling and Metropolis MCMC routines for reliable error estimation.
* **Automatic Universality Scoring**: Automatically compares your dataset against standard universality classes using a combined scoring metric.

## Project Structure

* **`GUI_Final_v2_kferr_FIXED_v26.py`** The main Graphical User Interface. Run this file to launch the interactive analyzer dashboard.
* **`FinalArrot+KFplot_v2.py`** The CLI runner for headless analysis. It can run the full analysis pipeline without a GUI and also includes a feature to validate classic Kouvel-Fisher nickel data from literature.
* **`linear_fitting_with_gammas_and_betas_v2_2D.py`** A diagnostic CLI tool for scanning different high-field thresholds and suggesting optimal initial $\beta$ and $\gamma$ parameters based on linearity and intercept metrics.
* **`no_intercept_at_the_plot_several_data_sets_v2_2D.py`** A helper script to scan and optimize the background correction ($B_c$) across multiple datasets.

## Quick Start

**1. Launch the Interactive GUI:**
Download all scripts and run python "GUI_Final_v2_kferr_FIXED_v26.py"

**2. Run Headless CLI Analysis(Optional):**
python FinalArrot+KFplot_v2.py --data ./data/*.txt --bc 0.2 --out ./output_dir --use-eos --joint-fit

**3. Run Parameter Diagnostics(Optional):**
python linear_fitting_with_gammas_and_betas_v2_2D.py --files ./data/*.txt --min_points 10

## Requirements

* Python 3.7+
* `numpy`
* `scipy`
* `matplotlib`
* `PyQt5` (Required for the GUI)
* `pandas` (Recommended for robust CSV/Excel file ingestion)
* `scikit-learn` (Required for Gaussian Process scaling features)
* `PyMuPDF` / `fitz` (Optional, required only for extracting literature data from PDFs)
