#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FinalArrot+KFplot_v2.py

CLI runner + literature validation for the same analysis core used by GUI_Final_v2.py.

Why this file exists
- Some users want a headless / batch mode (no GUI).
- It can also validate key steps against classic Kouvel–Fisher nickel data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Import backend engine without forcing a Qt GUI
from critical_engine import PhysicsEngine, SKLEARN_AVAILABLE


# ---------------------------
# Literature parsing helpers
# ---------------------------

def _clean_token(tok: str) -> str:
    """Fix common APS/PDF text extraction quirks (g/p/q as 9/0, commas, etc)."""
    tok = tok.strip()
    if not tok:
        return tok
    tok = tok.replace('p', '0').replace('g', '9').replace('q', '9')
    tok = tok.replace('O', '0').replace('l', '1')
    tok = tok.replace(',', '.')
    tok = re.sub(r"[^0-9eE+\-\.]+", "", tok)
    return tok


def parse_kouvel_fisher_nickel_tables(pdf_path: str | Path) -> dict:
    """Parse Table II and Table III from Phys. Rev. 136, A1626 (1964).

    Returns
    - table2: T, chi_inv, dchi_inv_dT, Tstar (chi/dchi)
    - table3: H, sigma_raw, dsigma_dH_raw, eps (1/delta)

    Notes
    - We only need the numerical columns; the PDF is formatted in columns, so
      we combine a simple token scan (Table II) + a block-based parse (Table III).
    """
    import fitz  # PyMuPDF

    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)

    # ---------- Table II (page that contains 'TABLE II.') ----------
    page2 = None
    for i in range(doc.page_count):
        if 'TABLE II.' in doc[i].get_text('text'):
            page2 = i
            break
    if page2 is None:
        raise RuntimeError('TABLE II not found in PDF')

    txt2 = doc[page2].get_text('text')
    # Keep tokens like '68.g' so we can clean them.
    raw = re.findall(r"[0-9]+(?:\.[0-9a-zA-Z]+)?", txt2)
    toks = [_clean_token(t) for t in raw]

    # Find the T list: 630, 632, ... 700 (14 values)
    T = []
    start_idx = None
    for i in range(len(toks)):
        try:
            v = float(toks[i])
        except Exception:
            continue
        if 620 <= v <= 710:
            # attempt to read a monotonic run
            run = []
            j = i
            while j < len(toks) and len(run) < 30:
                try:
                    vv = float(toks[j])
                except Exception:
                    break
                if 620 <= vv <= 710:
                    run.append(vv)
                    j += 1
                else:
                    break
            if len(run) >= 10:
                T = run[:14]
                start_idx = j
                break

    if len(T) < 10 or start_idx is None:
        raise RuntimeError('Failed to parse T column in Table II')

    # Next column: chi_inv (order 1e2..1e4)
    chi_inv = []
    i = start_idx
    while i < len(toks) and len(chi_inv) < len(T):
        try:
            v = float(toks[i])
        except Exception:
            i += 1
            continue
        if 10 <= v <= 50000:
            chi_inv.append(v)
        i += 1

    # Next column: dchi_inv/dT (order ~50..200)
    dchi = []
    while i < len(toks) and len(dchi) < len(T):
        try:
            v = float(toks[i])
        except Exception:
            i += 1
            continue
        if 1 <= v <= 500:
            dchi.append(v)
        i += 1

    if not (len(chi_inv) == len(dchi) == len(T)):
        raise RuntimeError(
            f'TABLE II parse mismatch: T={len(T)}, chi_inv={len(chi_inv)}, dchi={len(dchi)}'
        )

    T = np.array(T, dtype=float)
    chi_inv = np.array(chi_inv, dtype=float)
    dchi = np.array(dchi, dtype=float)
    Tstar = chi_inv / dchi

    # ---------- Table III (block-based parse) ----------
    page3 = None
    for i in range(doc.page_count):
        if 'TABLE III.' in doc[i].get_text('text'):
            page3 = i
            break
    if page3 is None:
        raise RuntimeError('TABLE III not found in PDF')

    # Try extracting Tc from the Table III caption (paper states: Tc (= 627.2 K))
    tc_paper = None
    txt3 = doc[page3].get_text('text')
    m_tc = re.search(r"T[^0-9]{0,12}\(=\s*([0-9]+(?:\.[0-9]+)?)", txt3)
    if m_tc:
        try:
            tc_paper = float(m_tc.group(1))
        except Exception:
            tc_paper = None

    blocks = doc[page3].get_text('blocks')

    # Heuristic: locate blocks that are mostly numbers and belong to the table.
    numeric_blocks = []
    for b in blocks:
        x0, y0, x1, y1, text, _, _ = b
        if 'TABLE III' in text:
            continue
        digits = sum(ch.isdigit() for ch in text)
        # The 4 data columns each contain 8 numbers; those blocks have enough digits.
        if digits >= 20 and y0 < 230:  # table is near the top on this page
            numeric_blocks.append((x0, y0, text))

    if len(numeric_blocks) < 3:
        raise RuntimeError('TABLE III numeric blocks not detected')

    # Sort left-to-right by x0
    numeric_blocks.sort(key=lambda t: t[0])

    def parse_col(text: str) -> list[float]:
        out = []
        for ln in text.splitlines():
            ln = ln.strip().replace(' ', '')
            if not ln:
                continue
            ln = _clean_token(ln)
            if ln:
                try:
                    out.append(float(ln))
                except Exception:
                    pass
        return out

    # Expect 4 cols: H, sigma, dSigma/dH, eps
    cols = [parse_col(t[2]) for t in numeric_blocks[:4]]

    # Identify H column: integers around 2000..20000
    col_H = None
    for c in cols:
        if len(c) >= 6 and all(500 <= v <= 50000 for v in c):
            # typical H list starts 2000, 4000...
            if any(abs(c[0] - 2000) < 1e-6 for _ in [0]):
                col_H = c[:8]
                break
    if col_H is None:
        # fallback: pick the column with the largest typical magnitude
        col_H = max(cols, key=lambda c: np.nanmedian(c) if c else -1)[:8]

    # eps column: around 0.23..0.25
    col_eps = None
    for c in cols:
        if len(c) >= 6 and np.nanmedian(c) < 0.5:
            if 0.1 <= np.nanmedian(c) <= 0.4:
                # choose the column whose median is closest to 0.237
                if col_eps is None or abs(np.nanmedian(c) - 0.237) < abs(np.nanmedian(col_eps) - 0.237):
                    col_eps = c[:8]
    if col_eps is None:
        raise RuntimeError('Failed to identify epsilon column in Table III')

    # Remaining columns: sigma and derivative (not strictly needed for delta)
    rem = [c[:8] for c in cols if c[:8] != col_H and c[:8] != col_eps]
    sigma = rem[0] if rem else []
    dsigma = rem[1] if len(rem) > 1 else []

    return {
        'table2': {'T': T, 'chi_inv': chi_inv, 'dchi_inv_dT': dchi, 'Tstar': Tstar},
        'table3': {
            'H': np.array(col_H[:8], dtype=float),
            'sigma_raw': np.array(sigma[:8], dtype=float) if sigma else np.array([], dtype=float),
            'dsigma_dH_raw': np.array(dsigma[:8], dtype=float) if dsigma else np.array([], dtype=float),
            'eps': np.array(col_eps[:8], dtype=float),
        },
        'meta': {'page_table2': int(page2) + 1, 'page_table3': int(page3) + 1, 'Tc_caption': tc_paper},
    }


def validate_kouvel_fisher_nickel(pdf_path: str | Path) -> dict:
    """Compute (Tc, gamma, delta, beta) from the classic KF nickel tables.

    Notes on "matching the paper":
    - The paper quotes Tc explicitly (Table III caption shows Tc = 627.2 K).
      We report both:
        * Tc_from_Tstar_linear_fit  (pure KF extrapolation)
        * Tc_paper_caption (if detected)
    - gamma is reported as the *near-Tc* limit. We therefore compute a near-Tc
      effective exponent using the closest few points.
    """
    data = parse_kouvel_fisher_nickel_tables(pdf_path)

    T = data['table2']['T']
    Tstar = data['table2']['Tstar']

    # 1) Tc by linear extrapolation of T* vs T using the first few points
    k = min(6, len(T))
    A = np.vstack([T[:k], np.ones(k)]).T
    slope, intercept = np.linalg.lstsq(A, Tstar[:k], rcond=None)[0]
    Tc_lin = -intercept / slope

    # 1b) Tc from the paper caption (Table III) if we can detect it
    Tc_caption = None
    cap = data.get('meta', {}).get('Tc_caption', None)
    if isinstance(cap, (float, int)):
        Tc_caption = float(cap)

    Tc_used = Tc_caption if Tc_caption is not None else float(Tc_lin)

    # 2) gamma from effective exponent near Tc: gamma_eff = (T - Tc)/T*
    #    Use the closest few points above Tc.
    k_eff = min(3, len(T))
    gamma_eff = (T[:k_eff] - Tc_used) / Tstar[:k_eff]
    gamma = float(np.nanmean(gamma_eff))

    # 3) delta from Table III epsilon column (eps = 1/delta)
    eps = data['table3']['eps']
    eps_mean = float(np.nanmean(eps))
    delta = 1.0 / eps_mean

    # 4) beta from Widom relation
    beta = gamma / (delta - 1.0)

    return {
        'Tc_from_Tstar_linear_fit': float(Tc_lin),
        'Tc_paper_caption': (float(Tc_caption) if Tc_caption is not None else None),
        'Tc_used_for_gamma': float(Tc_used),
        'gamma_eff_points': [float(x) for x in gamma_eff],
        'gamma_from_near_Tc_effexp': float(gamma),
        'eps_mean_1_over_delta': float(eps_mean),
        'delta': float(delta),
        'beta_widom': float(beta),
        'meta': data['meta'],
        'table2_n': int(len(T)),
        'table3_n': int(len(eps)),
    }


# ---------------------------
# CLI pipeline
# ---------------------------



def load_mce_file(path: str) -> dict:
    """Load a precomputed MCE ΔS(T,H) table.

    Expected formats (auto-detected):
      (A) 3 columns: T, H, dS  (long format)
      (B) matrix format: first column T, first row H grid, remaining cells dS

    Returns: {'T': list[float], 'H': np.ndarray, 'dS': np.ndarray, 'sigma': np.ndarray|None}
    """
    import numpy as np

    arr = np.loadtxt(path, delimiter=None)
    if arr.ndim != 2:
        raise ValueError('MCE file must be a 2D numeric array')
    if arr.shape[1] == 3:
        T = arr[:, 0]
        H = arr[:, 1]
        dS = arr[:, 2]
        Tu = np.unique(T)
        Hu = np.unique(H)
        Tu.sort(); Hu.sort()
        mat = np.full((Tu.size, Hu.size), np.nan, dtype=float)
        for tt, hh, ss in zip(T, H, dS):
            i = int(np.where(Tu == tt)[0][0])
            j = int(np.where(Hu == hh)[0][0])
            mat[i, j] = ss
        return {'T': Tu.tolist(), 'H': Hu, 'dS': mat, 'sigma': None}

    # Matrix format: first cell could be 0, then H grid
    if arr.shape[1] >= 4:
        T = arr[1:, 0]
        H = arr[0, 1:]
        dS = arr[1:, 1:]
        return {'T': T.tolist(), 'H': H, 'dS': dS, 'sigma': None}

    raise ValueError('Unrecognized MCE file format')

# The publication pipeline is shared with the GUI. Literature-table parsing
# above is retained as a separate diagnostic utility.
from analysis_cli import run_full_pipeline, main

if __name__ == '__main__':
    raise SystemExit(main())
