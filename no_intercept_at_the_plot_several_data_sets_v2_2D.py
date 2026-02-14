#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""no intercept at the plot several data sets_v2.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


UNIVERSALITY_MODELS: Dict[str, Tuple[float, float]] = {
    'mean-field': (0.5, 1.0),
    '3D Ising': (0.325, 1.24),
    '3D Heisenberg': (0.365, 1.386),
    '3D XY': (0.345, 1.316),
    '2D Ising': (0.125, 1.75),
    '2D XY': (0.231, 2.2),
}


@dataclass
class FitSummary:
    T: float
    n: int
    r2: float
    intercept: float
    slope: float
    intercept_norm: float


def infer_temperature_from_name(path: str) -> float:
    base = os.path.basename(path)
    # Prefer explicit “...630K...”
    m = re.search(r'([0-9]+\.?[0-9]*)\s*[kK]', base)
    if m:
        return float(m.group(1))
    # Fallback: last number token
    nums = re.findall(r'([-+]?\d+\.?\d*)', base)
    if not nums:
        raise ValueError(f'Cannot infer temperature from filename: {base}')
    return float(nums[-1])


def load_file(path: str) -> Tuple[float, np.ndarray, np.ndarray]:
    T = infer_temperature_from_name(path)
    arr = np.loadtxt(path, delimiter=None)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f'Need at least 2 columns (H M): {path}')
    H = np.asarray(arr[:, 0], float)
    M = np.asarray(arr[:, 1], float)
    m = np.isfinite(H) & np.isfinite(M)
    H, M = H[m], M[m]
    # sort by |H| then by H
    idx = np.argsort(np.abs(H))
    return T, H[idx], M[idx]


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def map_transform(H_eff: np.ndarray, M: np.ndarray, beta: float, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """MAP coordinates (often used for line-fitting diagnostics).

    X = M^{1/beta}
    Y = (H_eff / M)^{1/gamma}

    Notes:
    - For robust diagnostics, we use absolute values inside the fractional powers.
    - We keep X positive (using |M|) because the high-field region is typically on the positive branch.
    """
    eps = 1e-30
    Mabs = np.maximum(np.abs(M), eps)
    X = Mabs ** (1.0 / beta)
    Y = np.maximum(np.abs(H_eff) / Mabs, eps) ** (1.0 / gamma)
    return X, Y


def global_score(fits: List[FitSummary]) -> float:
    """Higher is better."""
    if not fits:
        return -1e18
    r2_med = float(np.nanmedian([f.r2 for f in fits]))
    in_med = float(np.nanmedian([f.intercept_norm for f in fits]))
    # Encourage using more temperatures
    nT = len(fits)
    return r2_med - 0.35 * np.log10(1.0 + in_med) + 0.01 * nT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='M(H) isotherm files (T inferred from filename)')
    ap.add_argument('--model', default='3D Heisenberg', choices=list(UNIVERSALITY_MODELS.keys()) + ['ALL'])
    ap.add_argument('--bc_min', type=float, default=-0.02)
    ap.add_argument('--bc_max', type=float, default=0.02)
    ap.add_argument('--bc_steps', type=int, default=81)
    ap.add_argument('--high_field_frac', type=float, default=0.6)
    ap.add_argument('--min_points', type=int, default=10)
    ap.add_argument('--csv', type=str, default='bc_scan.csv')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    datasets = []
    for p in args.data:
        T, H, M = load_file(p)
        datasets.append((T, H, M, os.path.basename(p)))
    datasets.sort(key=lambda x: x[0])

    models = list(UNIVERSALITY_MODELS.keys()) if args.model == 'ALL' else [args.model]

    bcs = np.linspace(args.bc_min, args.bc_max, int(args.bc_steps))

    # Prepare CSV
    with open(args.csv, 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['model', 'beta', 'gamma', 'bc', 'score', 'nT_used', 'r2_median', 'intercept_norm_median'])

        best_overall = {}

        for model in models:
            beta, gamma = UNIVERSALITY_MODELS[model]
            best = (-1e18, None, None)  # score, bc, fits

            for bc in bcs:
                fits: List[FitSummary] = []
                for T, H, M, _name in datasets:
                    # background correction
                    H_eff = H - bc * M
                    Hmax = float(np.nanmax(np.abs(H_eff))) if H_eff.size else 0.0
                    if not np.isfinite(Hmax) or Hmax <= 0:
                        continue
                    mask = np.abs(H_eff) >= args.high_field_frac * Hmax
                    if int(np.count_nonzero(mask)) < int(args.min_points):
                        continue
                    X, Y = map_transform(H_eff[mask], M[mask], beta=beta, gamma=gamma)
                    slope, intercept, r2 = linear_fit(X, Y)
                    intercept_norm = abs(intercept) / (abs(slope) + 1e-12)
                    fits.append(FitSummary(T=float(T), n=int(np.count_nonzero(mask)), r2=float(r2),
                                           intercept=float(intercept), slope=float(slope),
                                           intercept_norm=float(intercept_norm)))

                sc = global_score(fits)
                if fits:
                    r2_med = float(np.nanmedian([ff.r2 for ff in fits]))
                    in_med = float(np.nanmedian([ff.intercept_norm for ff in fits]))
                else:
                    r2_med, in_med = 0.0, float('nan')
                w.writerow([model, beta, gamma, float(bc), float(sc), len(fits), r2_med, in_med])

                if sc > best[0]:
                    best = (sc, float(bc), fits)

            best_overall[model] = best

            print('=' * 78)
            print(f'Model: {model}  (beta={beta}, gamma={gamma})')
            print(f'High-field: |H_eff| >= {args.high_field_frac}*Hmax ; min_points={args.min_points}')
            print(f'Bc scan: [{args.bc_min}, {args.bc_max}] steps={args.bc_steps}')
            print(f'Best Bc = {best[1]:+.6g}   (score={best[0]:.6f})')
            print('T\tN\tR2\tintercept\tslope\tintercept_norm')
            for ff in (best[2] or []):
                print(f"{ff.T:.3f}\t{ff.n}\t{ff.r2:.5f}\t{ff.intercept:.5g}\t{ff.slope:.5g}\t{ff.intercept_norm:.5g}")

        print('=' * 78)
        if len(best_overall) > 1:
            print('Summary (best Bc per model):')
            for model, (sc, bc, _fits) in best_overall.items():
                beta, gamma = UNIVERSALITY_MODELS[model]
                print(f"  {model:14s}  bc={bc:+.6g}  score={sc:.5f}  (beta={beta}, gamma={gamma})")

    if args.plot:
        if plt is None:
            raise RuntimeError('matplotlib not available')
        # Plot score vs bc for the selected model (or the best model if ALL)
        import pandas as pd
        df = pd.read_csv(args.csv)
        model_to_plot = models[0] if args.model != 'ALL' else max(best_overall.items(), key=lambda kv: kv[1][0])[0]
        sub = df[df['model'] == model_to_plot].sort_values('bc')
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(sub['bc'], sub['score'], '-')
        plt.xlabel('Bc')
        plt.ylabel('global score')
        plt.title(f'Bc scan ({model_to_plot})')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
