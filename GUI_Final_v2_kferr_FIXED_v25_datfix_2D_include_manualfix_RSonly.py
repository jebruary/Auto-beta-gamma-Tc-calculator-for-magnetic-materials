#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI_Final.py

Magnetic critical behavior + MCE analysis GUI.

Implements (per 代码修改建议.docx):
1) Quantitative data-collapse optimization for (beta, gamma, Tc)
2) Automatic universality-class comparison
3) Uncertainty estimates via bootstrap + simple Metropolis MCMC
4) Confluent-correction fits (with omega)
5) GP/Bayesian scaling (GP likelihood + MCMC posterior)
6) RG-inspired universal EOS (parametric representation) fit (Ising/Heisenberg + mean-field)
7) Advanced MCE differentiation with smoothing + uncertainty propagation
8) Joint/global objective combining magnetization scaling + MCE scaling (self-consistency)

Author: (integrated and patched)
"""

from __future__ import annotations

import os
import re
import sys
import math
import traceback
import csv
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Optional, but provides much more robust CSV/TSV/Excel ingestion.
try:
    import pandas as pd  # type: ignore
    _PANDAS_OK = True
except Exception:
    pd = None  # type: ignore
    _PANDAS_OK = False

from scipy import interpolate
from scipy.optimize import minimize
from scipy import stats

def _safe_linregress(x, y, *, min_pts: int = 4, eps: float = 0.0):
    """Robust wrapper for scipy.stats.linregress.

    Returns None if x is degenerate (all/near-identical) or if SciPy raises ValueError.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < int(min_pts):
        return None
    ptp = float(np.ptp(x))
    scale = float(np.nanmax(np.abs(x))) if x.size else 0.0
    tol = max(float(eps), 1e-15 * max(1.0, scale))
    if (not np.isfinite(ptp)) or (ptp <= tol):
        return None
    try:
        return stats.linregress(x, y)
    except ValueError:
        return None

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

import matplotlib

# --- GUI / Qt availability ---
# We want the analysis engine to work even on headless systems without PyQt5.
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
    from PyQt5.QtWidgets import (
        QAction, QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
        QGroupBox, QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton,
        QProgressBar, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
    )
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    GUI_AVAILABLE = True
    matplotlib.use("Qt5Agg")
except Exception:
    GUI_AVAILABLE = False
    matplotlib.use("Agg")
    class _QtStub: pass
    Qt = _QtStub
    QThread = object
    def pyqtSignal(*args, **kwargs):
        return None
    QAction = object
    QAbstractItemView = object
    QEvent = object
    QApplication = QCheckBox = QDoubleSpinBox = QFileDialog = QFormLayout = object
    QGroupBox = QHBoxLayout = QLabel = QListWidget = QMainWindow = QPushButton = object
    QProgressBar = QSplitter = QTabWidget = QTextEdit = QVBoxLayout = QWidget = object
    FigureCanvas = object

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
# Optional sklearn
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ------------------------------
# Utilities
# ------------------------------

def _safe_float(x: str) -> float:
    return float(x.strip())


def _robust_sort_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    return x[idx], y[idx]


def _mad(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    med = np.nanmedian(arr)
    return np.nanmedian(np.abs(arr - med))


def _clip_finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    m = np.isfinite(a)
    return a[m]


@dataclass
class EOSModel:
    name: str
    beta: float
    gamma: float
    # Parametric EOS functions: m(u), h(u) and coexistence root u0 (>1)
    m_func: Callable[[np.ndarray], np.ndarray]
    h_func: Callable[[np.ndarray], np.ndarray]
    u0: float

    @property
    def delta(self) -> float:
        return 1.0 + self.gamma / self.beta


# ------------------------------
# Core engine
# ------------------------------

class PhysicsEngine:
    """Backend engine used by both GUI and CLI tools."""

    def __init__(self):
        self.data_files: List[Dict] = []
        self.results: Dict = {
            'beta': 0.365,
            'gamma': 1.386,
            'Tc': 0.0,
            'Tc_KF': None,
            'delta': None,
            'errors': {},
            'scores': {},
            'best_model': None,
        }
        self.hc_value: float = 0.0  # Bc, in the same unit as H
        self.mce_data: Optional[Dict] = None
        self._gp_cache: Dict = {}

        self._init_universality_models()

    # ---------- Data I/O ----------

    def load_files(self, filepaths: List[str]) -> int:
        self.data_files.clear()
        for p in filepaths:
            try:
                d = self._load_single_file(p)
                if d is None:
                    continue
                # Support "one file contains multiple temperatures" (T,H,M) tables.
                if isinstance(d, list):
                    self.data_files.extend(d)
                else:
                    self.data_files.append(d)
            except Exception:
                traceback.print_exc()
        self.data_files.sort(key=lambda x: x['T'])
        return len(self.data_files)

    def _load_single_file(self, path: str) -> Optional[Dict | List[Dict]]:
        """Load a single dataset.

        Supports common formats:
        - CSV/TSV with header or without header
        - whitespace-delimited txt/dat
        - Excel .xlsx/.xls (if pandas is available)

        Column selection heuristics:
        - Prefer columns named like H/B/Field (field) and M/Magnetization (magnetization)
        - Otherwise use the first two numeric columns
        - If a temperature column exists (T/Temp/Temperature) and contains multiple unique values,
          the file is split into multiple isotherms.
        """

        base = os.path.basename(path)

        def infer_T_from_filename(fn: str) -> float:
            m = re.search(r'([0-9]+\.?[0-9]*)\s*K', fn, re.IGNORECASE)
            if m:
                return float(m.group(1))
            nums = re.findall(r'([0-9]+\.?[0-9]*)', fn)
            if nums:
                return float(nums[-1])
            raise ValueError(f"Cannot infer temperature from filename: {fn}")

        ext = os.path.splitext(base)[1].lower()

        # --- Read table robustly ---
        # NOTE: Many lab .dat/.txt exports are whitespace-delimited and may contain:
        #   - comment lines starting with '#' or '//'
        #   - a header line (sometimes prefixed by '#')
        #   - decimal comma (e.g. 13,92) and/or Fortran 'D' exponents (1.23D+04)
        # We therefore (1) pre-clean the text, (2) try several pandas strategies,
        # and (3) fall back to a regex-based numeric extractor if needed.

        from io import StringIO

        def _slurp_clean_text(p: str) -> str:
            out_lines: List[str] = []
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.rstrip('\n')
                    st = s.strip()
                    if not st:
                        continue
                    # drop C++-style comment lines
                    if st.startswith('//'):
                        continue
                    # keep '#' header lines but remove the leading '#'
                    if st.startswith('#'):
                        s = st[1:].lstrip()
                        if not s:
                            continue
                    out_lines.append(s)
            return '\n'.join(out_lines) + '\n'

        def _first_data_line(text: str) -> str:
            for line in text.splitlines():
                st = line.strip()
                if not st:
                    continue
                return st
            return ''

        def _detect_has_header(line: str) -> bool:
            # Header if it contains any alphabetic character
            return any(ch.isalpha() for ch in line)

        def _detect_delim(line: str) -> str:
            # Prefer explicit separators; otherwise treat as whitespace.
            if '\t' in line:
                return '\t'
            if ';' in line:
                return ';'
            if ',' in line:
                return ','
            return 'WHITESPACE'

        def _coerce_numeric_series(x: 'pd.Series') -> 'pd.Series':
            # Clean typical numeric quirks then coerce.
            if x.dtype == object:
                s = x.astype(str).str.strip()
                # Fortran exponent D -> E
                s = s.str.replace(r'([0-9])([dD])([+-]?[0-9]+)', r'\1E\3', regex=True)
                # Decimal comma -> decimal point for standalone numbers: 13,92 -> 13.92
                # (We do NOT blindly replace all commas to avoid breaking comma-delimited files.)
                s = s.str.replace(r'(?<![0-9])([+-]?[0-9]+),([0-9]+)(?![0-9])', r'\1.\2', regex=True)
                return pd.to_numeric(s, errors='coerce')
            return pd.to_numeric(x, errors='coerce')

        def _regex_numeric_fallback(p: str) -> np.ndarray:
            # Extract numeric tokens from each line; returns an (N,k) array.
            num_pat = re.compile(r'[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eEdD][+-]?\d+)?')
            rows: List[List[float]] = []
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    st = line.strip()
                    if not st or st.startswith('#') or st.startswith('//'):
                        continue
                    toks = num_pat.findall(st)
                    if len(toks) < 2:
                        continue
                    vals: List[float] = []
                    for t in toks:
                        t2 = t.replace('D', 'E').replace('d', 'E').replace(',', '.')
                        try:
                            vals.append(float(t2))
                        except Exception:
                            pass
                    if len(vals) >= 2:
                        rows.append(vals)
            if not rows:
                return np.empty((0, 0))
            # pad ragged rows
            k = max(len(r) for r in rows)
            arr = np.full((len(rows), k), np.nan, dtype=float)
            for i, r in enumerate(rows):
                arr[i, :len(r)] = r
            return arr

        ext = os.path.splitext(base)[1].lower()

        # --- Read table into DataFrame if possible ---
        df = None
        clean_text = None

        if ext in {'.csv', '.tsv', '.txt', '.dat', '.prn'}:
            try:
                clean_text = _slurp_clean_text(path)
            except Exception:
                clean_text = None

        if _PANDAS_OK and ext in {'.csv', '.tsv', '.txt', '.dat', '.prn'}:
            try:
                if clean_text is None:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        clean_text = f.read()

                first = _first_data_line(clean_text)
                has_header = _detect_has_header(first)
                delim = _detect_delim(first)

                # Try a few robust strategies
                if delim == 'WHITESPACE':
                    # whitespace-delimited: use regex sep
                    df = pd.read_csv(StringIO(clean_text), sep=r'\s+', engine='python',
                                     header=0 if has_header else None)
                else:
                    df = pd.read_csv(StringIO(clean_text), sep=delim, engine='python',
                                     header=0 if has_header else None)

                # If we ended up with a single wide column, try fixed-width
                if df is not None and df.shape[1] < 2:
                    df2 = pd.read_fwf(StringIO(clean_text), header=0 if has_header else None)
                    if df2 is not None and df2.shape[1] >= 2:
                        df = df2
            except Exception:
                df = None

        if _PANDAS_OK and df is None and ext in {'.xlsx', '.xls'}:
            df = pd.read_excel(path)

        # numpy fallback (no pandas or pandas failed)
        if df is None:
            arr = _regex_numeric_fallback(path)
            if arr.size == 0 or arr.shape[1] < 2:
                # last-resort: np.loadtxt
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(4096)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', ' '])
                    delim = dialect.delimiter
                except Exception:
                    delim = None
                first_line = ''
                for line in sample.splitlines():
                    s = line.strip()
                    if not s or s.startswith('#') or s.startswith('//'):
                        continue
                    first_line = s
                    break
                has_header = any(ch.isalpha() for ch in first_line)
                arr = np.loadtxt(path, delimiter=delim, skiprows=1 if has_header else 0)
            if arr.ndim == 1:
                raise ValueError(f"File has insufficient columns: {path}")
            if arr.shape[1] < 2:
                raise ValueError(f"Need at least 2 columns (H M): {path}")
            H = np.asarray(arr[:, 0], dtype=float)
            M = np.asarray(arr[:, 1], dtype=float)
            t = infer_T_from_filename(base)
            return {'file': base, 'path': path, 'T': t, 'H': H, 'M': M}
        # If we have a DataFrame, select numeric columns and map to H/M.
        assert df is not None
        df = df.copy()
        # If header was absent, pandas assigns integer columns 0..n-1
        colnames = [str(c) for c in df.columns]
        df.columns = colnames

        # Helper: find first matching column by regex list
        def pick_col(patterns: List[str]) -> Optional[str]:
            for pat in patterns:
                for c in df.columns:
                    if re.search(pat, str(c), re.IGNORECASE):
                        return str(c)
            return None

        tcol = pick_col([r'^t$', r'temp', r'temperature'])
        hcol = pick_col([r'^h\b', r'^b\b', r'field', r'mu0\s*h', r'\boe\b', r'koe'])
        mcol = pick_col([r'^m\b', r'mag', r'magnetization', r'emu'])

        # Coerce to numeric where possible (handles decimal comma + Fortran 'D' exponents)
        for c in df.columns:
            df[c] = _coerce_numeric_series(df[c])
        df = df.dropna(how='all')

        # If H/M columns not found by name, fall back to first two numeric columns
        if hcol is None or mcol is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            numeric_cols = [c for c in numeric_cols if df[c].notna().sum() > 0]
            if len(numeric_cols) < 2:
                # Fallback: regex-based numeric extraction (handles weird .dat exports)
                arr = _regex_numeric_fallback(path)
                if arr.size > 0 and arr.shape[1] >= 2:
                    H = np.asarray(arr[:, 0], dtype=float)
                    M = np.asarray(arr[:, 1], dtype=float)
                    t = infer_T_from_filename(base)
                    return {'file': base, 'path': path, 'T': float(t), 'H': H, 'M': M}
                raise ValueError(f"Could not find >=2 numeric columns in: {path}")
            hcol = numeric_cols[0]
            mcol = numeric_cols[1]

        # Field unit heuristics based on column name
        H_raw = df[hcol].to_numpy(dtype=float)
        M_raw = df[mcol].to_numpy(dtype=float)

        scale = 1.0
        hname = str(hcol).lower()
        if 'koe' in hname:
            scale = 0.1  # 1 kOe ~ 0.1 T (cgs convention)
        elif 'oe' in hname or 'gauss' in hname or 'g' == hname:
            scale = 1e-4  # 1 Oe ~ 1e-4 T (cgs convention)
        elif 'mt' in hname:
            scale = 1e-3
        elif 't' in hname and ('tesla' in hname or hname.endswith('_t') or hname.endswith('(t)')):
            scale = 1.0
        # If unit unknown but magnitude looks like Oe, apply a gentle heuristic (won't affect exponents)
        if scale == 1.0 and np.nanmax(np.abs(H_raw)) > 200 and np.nanmax(np.abs(H_raw)) < 2e6:
            # Most "mainstream" magnetization CSVs in labs are in Oe.
            scale = 1e-4

        H = np.asarray(H_raw * scale, dtype=float)
        M = np.asarray(M_raw, dtype=float)

        # If the file contains multiple temperatures, split by T
        if tcol is not None:
            T_raw = df[tcol].to_numpy(dtype=float)
            # drop NaN temps
            ok = np.isfinite(T_raw) & np.isfinite(H) & np.isfinite(M)
            T_raw, H2, M2 = T_raw[ok], H[ok], M[ok]
            uniq = np.unique(T_raw)
            if uniq.size > 1:
                out: List[Dict] = []
                for Tval in uniq:
                    sel = (T_raw == Tval)
                    if np.sum(sel) < 3:
                        continue
                    out.append({'file': base, 'path': path, 'T': float(Tval), 'H': H2[sel], 'M': M2[sel]})
                return out if out else None
            elif uniq.size == 1:
                t = float(uniq[0])
            else:
                t = infer_T_from_filename(base)
        else:
            t = infer_T_from_filename(base)

        return {'file': base, 'path': path, 'T': float(t), 'H': H, 'M': M}

    # ---------- Preprocessing ----------

    def _get_filtered_data(self, d: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Filter low-field region using hc_value (Bc).

        Bugfix:
          - Older versions took abs(H) but kept M as-is. If the input file contains
            both +H and -H branches, abs(H) folds the curve and can pair +H with -M,
            producing non-physical (H>0, M<0) points. This can propagate NaNs into
            MAP / KF / collapse / EOS (fractional powers of M).
          - We now preferentially keep the H>=0 branch (if present) and we use |M|
            for the critical scaling analysis (standard practice for symmetric isotherms).
        """
        H = np.asarray(d['H'], dtype=float)
        M = np.asarray(d['M'], dtype=float)

        m = np.isfinite(H) & np.isfinite(M)
        H, M = H[m], M[m]

        # Prefer the H>=0 branch if it exists (avoid folding when abs(H) is used).
        if np.any(H >= 0):
            keep = H >= 0
            H, M = H[keep], M[keep]

        if self.hc_value and self.hc_value > 0:
            keep = np.abs(H) >= self.hc_value
            H, M = H[keep], M[keep]

        # remove (near) zero M (avoid division / fractional powers)
        keep = np.abs(M) > 1e-14
        H, M = H[keep], M[keep]

        # standardize to positive field and magnetization magnitude for scaling
        H = np.abs(H)
        M = np.abs(M)

        # sort
        H, M = _robust_sort_xy(H, M)
        return H, M


    # ---------- High-field linear fit helpers for (Modified) Arrott analysis ----------

    def _fit_modified_arrott_isotherm(self, H: np.ndarray, M: np.ndarray, *, beta: float, gamma: float,
                                      min_pts: int = 4) -> Optional[Dict[str, float]]:
        """Robust high-field linear fit for the Modified Arrott (Arrott–Noakes) plot.

        Fit y = M^{1/beta} vs x = (H/M)^{1/gamma} using *only* a high-field window.
        Many real datasets have ~8–12 field points per isotherm; older versions used a fixed
        min_pts=6 and high quantiles (0.5–0.8), which could leave <min_pts points and return None
        for every temperature -> KF/Arrott Tc became N/A.

        Strategy:
          - Try several 'top-fraction' windows (keep the highest fields)
          - If still not enough points, fall back to the top-K points (K>=4)
          - Choose the best fit by R^2 (with positive slope)
        """
        H = np.asarray(H, float)
        M = np.asarray(M, float)
        m = np.isfinite(H) & np.isfinite(M) & (H > 0) & (M > 0)
        H = H[m]; M = M[m]
        if H.size < 4:
            return None
        H, M = _robust_sort_xy(H, M)

        n = int(H.size)
        min_pts_eff = int(max(4, min(min_pts, n)))
        keep_fracs = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
        best = None

        def _try_fit(Hh, Mh):
            x = (Hh / Mh) ** (1.0 / float(gamma))
            y = (Mh) ** (1.0 / float(beta))
            mm = np.isfinite(x) & np.isfinite(y)
            x = x[mm]; y = y[mm]
            if x.size < min_pts_eff:
                return None
            res = _safe_linregress(x, y, min_pts=min_pts_eff)
            if res is None:
                return None
            if not (np.isfinite(res.slope) and np.isfinite(res.intercept) and np.isfinite(res.rvalue)):
                return None
            if res.slope <= 0:
                return None
            r2 = float(res.rvalue ** 2)
            return {'slope': float(res.slope), 'intercept': float(res.intercept), 'r2': r2, 'n': float(x.size)}

        for frac in keep_fracs:
            k = int(max(min_pts_eff, round(frac * n)))
            if k < min_pts_eff:
                continue
            Hh = H[-k:]; Mh = M[-k:]
            cand = _try_fit(Hh, Mh)
            if cand is None:
                continue
            cand['hcut'] = float(Hh[0])
            if (best is None) or (cand['r2'] > best['r2']):
                best = cand

        if best is None:
            for k in range(min_pts_eff, min(n, 10) + 1):
                Hh = H[-k:]; Mh = M[-k:]
                cand = _try_fit(Hh, Mh)
                if cand is None:
                    continue
                cand['hcut'] = float(Hh[0])
                if (best is None) or (cand['r2'] > best['r2']):
                    best = cand

        return best
    def _fit_arrott_mf_isotherm(self, H: np.ndarray, M: np.ndarray,
                                min_pts: int = 4) -> Optional[Dict[str, float]]:
        """High-field linear fit for the *classic* Arrott plot (mean-field): y=M^2 vs x=H/M."""
        H = np.asarray(H, float)
        M = np.asarray(M, float)
        m = np.isfinite(H) & np.isfinite(M) & (H > 0) & (M > 0)
        H = H[m]; M = M[m]
        if H.size < 4:
            return None
        H, M = _robust_sort_xy(H, M)

        n = int(H.size)
        min_pts_eff = int(max(4, min(min_pts, n)))
        keep_fracs = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
        best = None

        def _try_fit(Hh, Mh):
            x = (Hh / Mh)
            y = (Mh ** 2)
            mm = np.isfinite(x) & np.isfinite(y)
            x = x[mm]; y = y[mm]
            if x.size < min_pts_eff:
                return None
            res = _safe_linregress(x, y, min_pts=min_pts_eff)
            if res is None:
                return None
            if not (np.isfinite(res.slope) and np.isfinite(res.intercept) and np.isfinite(res.rvalue)):
                return None
            if res.slope <= 0:
                return None
            r2 = float(res.rvalue ** 2)
            return {'slope': float(res.slope), 'intercept': float(res.intercept), 'r2': r2, 'n': float(x.size)}

        for frac in keep_fracs:
            k = int(max(min_pts_eff, round(frac * n)))
            if k < min_pts_eff:
                continue
            Hh = H[-k:]; Mh = M[-k:]
            cand = _try_fit(Hh, Mh)
            if cand is None:
                continue
            cand['hcut'] = float(Hh[0])
            if (best is None) or (cand['r2'] > best['r2']):
                best = cand

        if best is None:
            for k in range(min_pts_eff, min(n, 10) + 1):
                Hh = H[-k:]; Mh = M[-k:]
                cand = _try_fit(Hh, Mh)
                if cand is None:
                    continue
                cand['hcut'] = float(Hh[0])
                if (best is None) or (cand['r2'] > best['r2']):
                    best = cand

        return best
    def _relative_slope_score(self, beta: float, gamma: float, *, window_K: float = 15.0) -> float:
        """Relative-slope (RS) score for a universality class using Modified Arrott slopes.

        For the correct (beta,gamma), MAP isotherms are nearly parallel and the *critical* isotherm
        passes close to the origin. A common quantitative diagnostic is the relative slope:
            RS(T) = S(T)/S(Tc)
        which should be close to 1 near Tc.

        Returns a non-negative score (lower is better).
        """
        # Build MAP fits (high-field) for each isotherm
        fits = []
        for d in self.data_files:
            H, M = self._get_filtered_data(d)
            fit = self._fit_modified_arrott_isotherm(H, M, beta=beta, gamma=gamma)
            if fit is None:
                continue
            fits.append((float(d['T']), float(fit['slope']), float(fit['intercept']), float(fit['r2'])))
        if len(fits) < 4:
            return 1e99

        fits.sort(key=lambda x: x[0])
        tt = np.array([f[0] for f in fits], float)
        slopes = np.array([f[1] for f in fits], float)
        b0 = np.array([f[2] for f in fits], float)
        r2 = np.array([f[3] for f in fits], float)

        # prefer linear isotherms
        m = np.isfinite(tt) & np.isfinite(slopes) & np.isfinite(b0)
        if np.sum(m & (r2 > 0.95)) >= 4:
            m = m & (r2 > 0.95)
        tt = tt[m]; slopes = slopes[m]; b0 = b0[m]
        if tt.size < 4:
            return 1e99

        # estimate Tc by b0 crossing
        tt, b0 = _robust_sort_xy(tt, b0)
        sgn = np.sign(b0)
        Tc = None
        for i in range(tt.size - 1):
            if sgn[i] == 0:
                Tc = float(tt[i]); break
            if sgn[i] * sgn[i+1] < 0:
                t1, t2 = float(tt[i]), float(tt[i+1])
                b1, b2 = float(b0[i]), float(b0[i+1])
                if (b2 - b1) != 0:
                    Tc = t1 + (0.0 - b1) * (t2 - t1) / (b2 - b1)
                    break
        if Tc is None:
            # fallback: closest-to-zero intercept
            j = int(np.nanargmin(np.abs(b0)))
            Tc = float(tt[j])

        # slope at Tc (interpolate)
        try:
            S_tc = float(np.interp(Tc, tt, slopes))
        except Exception:
            S_tc = float(np.nanmedian(slopes))
        if not (np.isfinite(S_tc) and S_tc > 0):
            return 1e99

        # compute RS near Tc
        w = np.exp(-0.5 * ((tt - Tc) / max(1e-9, window_K)) ** 2)
        rs = slopes / S_tc
        # robust deviation from 1
        dev = np.abs(rs - 1.0)
        score = float(np.nansum(w * dev) / max(1e-12, np.nansum(w)))
        return score

    def select_initial_universality_by_rs(self) -> Optional[Dict[str, float]]:
        """Pick an initial universality class based on the relative-slope diagnostic."""
        models = {
            'Mean-field': (0.5, 1.0),
            '3D Ising': (0.325, 1.24),
            '3D Heisenberg': (0.365, 1.386),
            '3D XY': (0.345, 1.316),
            '2D Ising': (0.125, 1.75),
            '2D XY': (0.231, 2.2),
        }
        best = None
        for name, (b, g) in models.items():
            s = self._relative_slope_score(b, g)
            if best is None or s < best['score']:
                best = {'name': name, 'beta': float(b), 'gamma': float(g), 'score': float(s)}
        return best

    def auto_suggest_bc(self) -> float:
        """Suggest Bc based on maximizing linearity in MAP high-field region."""
        if not self.data_files:
            return 0.0
        # Candidate Bc fractions of max field
        H_all = []
        for d in self.data_files:
            H_all.append(np.max(np.abs(d['H'])))
        Hmax = float(np.nanmax(H_all))
        if not np.isfinite(Hmax) or Hmax <= 0:
            return 0.0
        fracs = np.linspace(0.05, 0.45, 9)
        best = (1e99, 0.0)
        for f in fracs:
            bc = f * Hmax
            # score: average |intercept|/|slope| + (1-R^2)
            scores = []
            for d in self.data_files:
                H = np.abs(d['H']); M = d['M']
                m = np.isfinite(H) & np.isfinite(M) & (np.abs(M) > 1e-14)
                H = H[m]; M = M[m]
                keep = H >= bc
                if np.sum(keep) < 10:
                    continue
                x = H[keep] / M[keep]
                y = M[keep]**2
                x, y = _robust_sort_xy(x, y)
                res = _safe_linregress(x, y, min_pts=10)
                if res is None:
                    continue
                intercept_pen = abs(res.intercept) / (abs(res.slope) + 1e-12)
                scores.append(intercept_pen + (1 - (res.rvalue**2)))
            if scores:
                s = float(np.nanmedian(scores))
                if s < best[0]:
                    best = (s, bc)
        return float(best[1])

    # ---------- 1) Quantitative data-collapse optimization ----------

    def _collapse_cost(self, beta: float, gamma: float, tc: float, *,
                       sample_per_curve: int = 60,
                       xbins: int = 120,
                       use_logx: bool = True) -> float:
        """Compute a quantitative collapse cost.

        Robust version (v22):
        - Works with small/medium datasets (avoid hard failure at X.size<200).
        - Uses adaptive bin count based on available points.
        - Requires only a modest fraction of populated bins.

        Scaling:
            x = H / |eps|^(beta+gamma)
            y = M / |eps|^beta
        For T<Tc we flip x so that the two branches overlap more naturally.
        If use_logx, we bin in signed log10(|x|).

        Lower is better.
        """
        if not self.data_files:
            return 1e99
        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        if not (np.nanmin(temps) < tc < np.nanmax(temps)):
            return 1e99
        if beta <= 0 or gamma <= 0:
            return 1e99

        X_all = []
        Y_all = []
        for d in self.data_files:
            T = float(d['T'])
            eps = (T - tc) / tc
            if abs(eps) < 1e-6:
                continue

            H, M = self._get_filtered_data(d)
            if H.size < 6:
                continue

            if sample_per_curve and H.size > sample_per_curve:
                idx = np.linspace(0, H.size - 1, sample_per_curve).astype(int)
                H = H[idx]; M = M[idx]

            scale_h = abs(eps) ** (beta + gamma)
            scale_m = abs(eps) ** beta
            if not np.isfinite(scale_h) or not np.isfinite(scale_m) or scale_h == 0 or scale_m == 0:
                continue

            x = H / scale_h
            y = M / scale_m

            # flip x for T<Tc so that two branches overlap more naturally
            if eps < 0:
                x = -x

            if use_logx:
                keep = (x != 0) & np.isfinite(x) & np.isfinite(y)
                x = x[keep]; y = y[keep]
                if x.size == 0:
                    continue
                lx = np.log10(np.abs(x))
                sx = np.sign(x)
                x = sx * lx
            else:
                keep = np.isfinite(x) & np.isfinite(y)
                x = x[keep]; y = y[keep]

            if x.size:
                X_all.append(x)
                Y_all.append(y)

        if len(X_all) < 3:
            return 1e99

        X = np.concatenate(X_all)
        Y = np.concatenate(Y_all)
        m = np.isfinite(X) & np.isfinite(Y)
        X = X[m]; Y = Y[m]

        # Dataset-adaptive minimum size
        if X.size < 80:
            return 1e99

        # adaptive bins: keep enough points/bin
        xbins_eff = int(np.clip(xbins, 25, 140))
        xbins_eff = int(min(xbins_eff, max(25, X.size // 8)))
        if xbins_eff < 25:
            xbins_eff = 25

        xmin, xmax = np.nanpercentile(X, 2), np.nanpercentile(X, 98)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return 1e99

        bins = np.linspace(xmin, xmax, xbins_eff + 1)
        bin_id = np.digitize(X, bins) - 1

        # median curve + MAD scatter
        costs = []
        for b in range(xbins_eff):
            sel = bin_id == b
            if np.sum(sel) < 6:
                continue
            yb = Y[sel]
            med = np.nanmedian(yb)
            scatter = _mad(yb) + 1e-12
            costs.append(np.nanmean(((yb - med) / scatter) ** 2))

        # require only a modest number of populated bins
        if len(costs) < max(10, int(xbins_eff * 0.08)):
            return 1e99

        return float(np.nanmean(costs))

    def optimize_data_collapse(self, init_beta: float, init_gamma: float, fixed_params: bool = False,
                              init_tc: Optional[float] = None, *, tc_fixed: Optional[float] = None,
                              tc_prior: Optional[Tuple[float, float]] = None,
                              update_results: bool = True) -> Dict:
        """Optimize (beta, gamma, Tc) by minimizing collapse cost.

        Parameters
        ----------
        init_beta, init_gamma:
            Initial guesses for critical exponents.
        fixed_params:
            If True, keep beta/gamma fixed to (init_beta, init_gamma).
        init_tc:
            Initial Tc guess (only used when Tc is free).
        tc_fixed:
            If provided, Tc is fixed to this value and ONLY (beta, gamma) are optimized.
            This is the key switch to enforce self-consistency when a more reliable Tc
            (e.g. from Kouvel–Fisher) is available.
        tc_prior:
            Optional weak quadratic regularization on Tc when Tc is free: (tc0, sigma).
            penalty = ((Tc - tc0)/sigma)^2
        update_results:
            If True (default), writes optimized parameters into self.results.
            If False, returns optimized values without mutating global results.

        Notes (bugfix):
          - SciPy's Nelder-Mead does NOT enforce bounds, so Tc can wander outside
            the temperature window and the objective becomes a flat 1e99 plateau.
            That can propagate NaN/identical scores and causes "default-to-first-model"
            behavior in model selection. We therefore *clip* parameters inside the
            objective and clip the final result as well.
        """
        if not self.data_files:
            raise RuntimeError("No data loaded")

        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        tmin = float(np.nanmin(temps) + 1e-6)
        tmax = float(np.nanmax(temps) - 1e-6)

        bmin, bmax = 0.05, 1.2
        gmin, gmax = 0.2, 3.5

        def _clip_bg(b: float, g: float) -> Tuple[float, float]:
            b = float(np.clip(b, bmin, bmax))
            g = float(np.clip(g, gmin, gmax))
            if fixed_params:
                b, g = float(init_beta), float(init_gamma)
            return b, g

        def _clip_tc(tc: float) -> float:
            return float(np.clip(float(tc), tmin, tmax))

        # ------------------------------
        # Case A: Tc fixed (KF-locked / self-consistent scaling)
        # ------------------------------
        if tc_fixed is not None and isinstance(tc_fixed, (int, float)) and np.isfinite(tc_fixed):
            tc = _clip_tc(float(tc_fixed))
            x0 = np.array([init_beta, init_gamma], dtype=float)

            def obj_bg(p: np.ndarray) -> float:
                b, g = _clip_bg(float(p[0]), float(p[1]))
                return float(self._collapse_cost(b, g, tc))

            if fixed_params:
                b, g = _clip_bg(float(init_beta), float(init_gamma))
                success = True
            else:
                res = minimize(obj_bg, x0, method='Nelder-Mead',
                               options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-4})
                b, g = _clip_bg(float(res.x[0]), float(res.x[1]))
                success = bool(res.success)

            cost = float(self._collapse_cost(b, g, tc))
            out = {'beta': b, 'gamma': g, 'Tc': tc, 'cost': cost, 'success': success, 'tc_fixed': True}
            if update_results:
                self.results.update({'beta': b, 'gamma': g, 'Tc': tc, 'collapse_cost': cost})
                self.results['delta'] = 1.0 + g / b
            return out

        # ------------------------------
        # Case B: Tc free
        # ------------------------------
        tc0 = float(init_tc) if init_tc is not None else float(np.nanmean(temps))
        x0 = np.array([init_beta, init_gamma, tc0], dtype=float)

        def _clip_params(b: float, g: float, tc: float) -> Tuple[float, float, float]:
            b, g = _clip_bg(b, g)
            tc = _clip_tc(tc)
            return b, g, tc

        def obj(p: np.ndarray) -> float:
            b, g, tc = _clip_params(float(p[0]), float(p[1]), float(p[2]))
            base = float(self._collapse_cost(b, g, tc))
            if tc_prior is not None and len(tc_prior) == 2 and tc_prior[1] is not None and tc_prior[1] > 0:
                tc_mu, tc_sig = float(tc_prior[0]), float(tc_prior[1])
                base = base + float(((tc - tc_mu) / tc_sig) ** 2)
            return base

        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-4})

        b, g, tc = _clip_params(float(res.x[0]), float(res.x[1]), float(res.x[2]))
        cost = float(self._collapse_cost(b, g, tc))

        out = {'beta': b, 'gamma': g, 'Tc': tc, 'cost': cost, 'success': bool(res.success), 'tc_fixed': False}
        if update_results:
            self.results.update({'beta': b, 'gamma': g, 'Tc': tc, 'collapse_cost': cost})
            self.results['delta'] = 1.0 + g / b
        return out


    def optimize_tc_discrete_by_collapse(self,
                                        init_beta: float,
                                        init_gamma: float,
                                        *,
                                        fixed_params: bool = False,
                                        tc_candidates: Optional[List[float]] = None,
                                        update_results: bool = True) -> Dict:
        """Choose Tc from a *discrete* candidate set (typically measured isotherms) by minimizing
        the collapse cost after optimizing (beta,gamma) at each candidate Tc.

        Motivation:
          - For small datasets, the free-Tc Nelder–Mead objective can become discontinuous
            (many Tc values land on a 1e99 plateau), which biases Tc.
          - Experimental Tc is often best constrained to the measured temperature grid.

        Returns a dict with the best (beta,gamma,Tc,cost). If no candidate produces a finite cost,
        falls back to a standard free-Tc collapse.
        """
        if not self.data_files:
            raise RuntimeError("No data loaded")

        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        temps = temps[np.isfinite(temps)]
        if temps.size == 0:
            raise RuntimeError("No valid temperatures in data")

        if tc_candidates is None:
            # default: use measured isotherms as Tc candidates
            cand = np.unique(np.round(temps.astype(float), 6))
        else:
            cand = np.unique(np.round(np.array(list(tc_candidates), dtype=float), 6))
            cand = cand[np.isfinite(cand)]

        # Need Tc strictly inside (tmin,tmax) for collapse (avoid eps=0 for all curves)
        tmin = float(np.nanmin(temps)); tmax = float(np.nanmax(temps))
        cand = cand[(cand > tmin) & (cand < tmax)]
        if cand.size == 0:
            # if the dataset is too small, just do free Tc
            return self.optimize_data_collapse(init_beta, init_gamma, fixed_params=fixed_params,
                                              init_tc=float(np.nanmean(temps)), update_results=update_results)

        best_out = None
        for tc in cand:
            try:
                # Optimize (beta,gamma) with Tc fixed
                # Note: we temporarily reduce maxiter for speed; cost is evaluated consistently.
                out = self.optimize_data_collapse(init_beta, init_gamma,
                                                 fixed_params=fixed_params,
                                                 tc_fixed=float(tc),
                                                 update_results=False)
                cost = float(out.get('cost', np.nan))
                if not np.isfinite(cost) or cost >= 1e98:
                    continue
                if (best_out is None) or (cost < float(best_out['cost'])):
                    best_out = dict(out)
            except Exception:
                continue

        if best_out is None:
            # fallback to free Tc
            return self.optimize_data_collapse(init_beta, init_gamma, fixed_params=fixed_params,
                                              init_tc=float(np.nanmean(temps)), update_results=update_results)

        if update_results:
            self.results.update({
                'beta': float(best_out['beta']),
                'gamma': float(best_out['gamma']),
                'Tc': float(best_out['Tc']),
                'collapse_cost': float(best_out.get('cost', np.nan)),
            })
            self.results['delta'] = 1.0 + float(best_out['gamma']) / float(best_out['beta'])

        best_out['method'] = 'discrete_tc_collapse'
        return best_out

    # ---------- 2) Universality-class comparison ----------

    def _init_universality_models(self):
        # Mean-field (Landau) parametric EOS
        def mf_m(u):
            return u
        def mf_h(u):
            return u * (3.0 - u**2)
        # u0 = sqrt(3)
        mf = EOSModel('Mean-field', 0.5, 1.0, mf_m, mf_h, u0=math.sqrt(3.0))

        # 3D Ising: Guida & Zinn-Justin (1997) polynomial, Eq (7.1): h(theta)/theta = 1 -0.76147 theta^2 + 7.74e-3 theta^4
        # We'll use m(theta)=theta and h(theta)=theta*(1 + h3 theta^2 + h5 theta^4)
        def ising_m(u):
            return u
        h3 = -0.76147
        h5 = 7.74e-3
        def ising_h(u):
            return u * (1.0 + h3 * u**2 + h5 * u**4)
        # u0^2=1.331 from Eq (7.2)
        ising = EOSModel('3D Ising', 0.326, 1.237, ising_m, ising_h, u0=math.sqrt(1.331))

        # 3D Heisenberg: Campostrini et al. PRB 65, 144520 (2002), scheme A n=1 parameters from Table VII
        # m(u)=u(1+c1 u^2), h(u)=u(1-u^2/u0^2)^2
        c1 = -0.016  # central of -0.016(9)
        u0_sq = 3.3
        def heis_m(u):
            return u * (1.0 + c1 * u**2)
        def heis_h(u):
            return u * (1.0 - (u**2) / u0_sq) ** 2
        heis = EOSModel('3D Heisenberg', 0.3689, 1.3960, heis_m, heis_h, u0=math.sqrt(u0_sq))

        self.eos_models: Dict[str, EOSModel] = {m.name: m for m in [mf, ising, heis]}

        # exponent-only models for quick comparison (including 3D XY as optional)
        self.universality_exponents = {
            'Mean-field': {'beta': 0.5, 'gamma': 1.0},
            '3D Ising': {'beta': 0.32, 'gamma': 1.23},
            '3D Heisenberg': {'beta': 0.36, 'gamma': 1.39},
            '3D XY': {'beta': 0.34, 'gamma': 1.31},
            '2D Ising': {'beta': 0.12, 'gamma': 1.75},
            '2D XY': {'beta': 0.23, 'gamma': 2.2},
        }

    def compare_universality_classes(self, *, include_eos: bool = True) -> Dict:
        """Compute a score for each universality class (WITHOUT overwriting final fitted parameters).

        v22 improvements:
        1) Robust collapse scoring (works even for small datasets).
        2) Add an exponent-distance term so that if your fitted (beta,gamma) clearly matches a
           universality class (e.g., Ni ~ 3D Heisenberg), the classifier does not get stuck on a
           1e99 penalty plateau.
        3) EOS is treated as a *diagnostic* and is down-weighted by default in the universality score.

        The final score S is a weighted combination of:
            - exponent distance: ((beta-beta_ref)/sigma_beta)^2 + ((gamma-gamma_ref)/sigma_gamma)^2
            - collapse cost at Tc_ref (if finite)
            - optional EOS term (very small weight)
            - optional MCE n-consistency penalty (small)
        """
        if not self.data_files:
            return {}

        # ---- Preserve final fitted parameters ----
        b_keep = float(self.results.get('beta', np.nan))
        g_keep = float(self.results.get('gamma', np.nan))
        tc_keep = float(self.results.get('Tc', np.nan))
        d_keep = self.results.get('delta', None)

        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        tc_ref = tc_keep if np.isfinite(tc_keep) else float(np.nanmean(temps))

        # uncertainties for exponent-distance normalization
        sig_b = float(self.results.get('errors', {}).get('beta', np.nan))
        sig_g = float(self.results.get('errors', {}).get('gamma', np.nan))
        # reasonable fallback scales (typical experimental spread)
        if (not np.isfinite(sig_b)) or sig_b <= 0:
            sig_b = 0.03
        if (not np.isfinite(sig_g)) or sig_g <= 0:
            sig_g = 0.06

        # ---- EOS chi2 cache (prefer already computed fits) ----
        eos_chi2_by_model: Dict[str, float] = {}
        if include_eos:
            eos_fits = self.results.get('eos_fits', {})
            if isinstance(eos_fits, dict) and eos_fits:
                for name in self.eos_models:
                    fit = eos_fits.get(name, {})
                    eos_chi2_by_model[name] = float(fit.get('chi2_red', np.nan))
            else:
                # compute on the fly (may be slow); does NOT touch beta/gamma/Tc
                for name in self.eos_models:
                    try:
                        fit = self.fit_eos_parametric(model_name=name, init_tc=tc_ref, fit_scales=True, max_points=2500)
                        eos_chi2_by_model[name] = float(fit.get('chi2_red', np.nan))
                    except Exception:
                        eos_chi2_by_model[name] = np.nan

        finite_chi2 = _clip_finite(np.array(list(eos_chi2_by_model.values()), dtype=float))
        chi2_ref = float(np.nanmedian(finite_chi2)) if finite_chi2.size else np.nan

        # ---- Collapse costs for all classes (to detect penalty plateau) ----
        collapse_costs = {}
        for name, exps in self.universality_exponents.items():
            b, g = float(exps['beta']), float(exps['gamma'])
            collapse_costs[name] = float(self._collapse_cost(b, g, tc_ref))
        cc_vals = np.array(list(collapse_costs.values()), dtype=float)
        plateau = (np.nanmin(cc_vals) > 1e98) or (np.nanstd(cc_vals) < 1e-9)


        # ---- Relative-slope (RS) diagnostic for MAP (helps when collapse is ill-conditioned) ----
        rs_scores: Dict[str, float] = {}
        for _name, _exps in self.universality_exponents.items():
            bb = float(_exps['beta']); gg = float(_exps['gamma'])
            try:
                rs_scores[_name] = float(self._relative_slope_score(bb, gg))
            except Exception:
                rs_scores[_name] = float('nan')

        # weights (tuned so exponent match dominates when clear)
        w_exp = 1.0
        w_collapse = 0.15 if (not plateau) else 0.0
        w_rs = 4.0  # relative-slope diagnostic (MAP)
        w_eos = 0.05  # diagnostic only
        w_n = 0.10

        scores: Dict[str, Dict] = {}
        for name, exps in self.universality_exponents.items():
            b, g = float(exps['beta']), float(exps['gamma'])

            # exponent distance around your fitted exponents
            exp_dist = ((b - b_keep) / sig_b) ** 2 + ((g - g_keep) / sig_g) ** 2 if (np.isfinite(b_keep) and np.isfinite(g_keep)) else np.nan

            cost = collapse_costs.get(name, np.nan)
            cost_term = 0.0
            if w_collapse > 0 and np.isfinite(cost) and cost < 1e98:
                # mild normalization to keep scale ~O(1)
                cost_term = float(np.log10(1.0 + max(0.0, cost)))

            eos = eos_chi2_by_model.get(name, np.nan) if include_eos else np.nan
            eos_term = 0.0
            if include_eos and np.isfinite(eos) and np.isfinite(chi2_ref) and chi2_ref > 0:
                eos_term = float(np.log10(1.0 + eos / chi2_ref))

            rs = rs_scores.get(name, np.nan)
            rs_term = 0.0
            if np.isfinite(rs) and rs < 1e98:
                # scale to O(1)
                rs_term = float(np.log10(1.0 + 10.0 * max(0.0, rs)))

            # MCE consistency penalty (use final fit's n values)
            n_pen = 0.0
            if self.results.get('n_exp') is not None and self.results.get('n_theo') is not None:
                diff = abs(float(self.results['n_exp']) - float(self.results['n_theo']))
                n_pen = min(3.0, diff / 0.08)

            # If exp_dist is NaN (no fitted exponents), fall back to collapse only
            if not np.isfinite(exp_dist):
                exp_dist = 0.0
                w_exp_eff = 0.0
                w_collapse_eff = 1.0 if w_collapse > 0 else 0.0
            else:
                w_exp_eff = w_exp
                w_collapse_eff = w_collapse

            total = (w_exp_eff * exp_dist) + (w_collapse_eff * cost_term) + (w_rs * rs_term) + (w_eos * eos_term) + (w_n * n_pen)
            scores[name] = {
                'collapse_cost': float(cost),
                'exp_dist': float(exp_dist),
                'eos_chi2_red': float(eos) if np.isfinite(eos) else np.nan,
                'n_penalty': float(n_pen),
                'rs_score': float(rs) if np.isfinite(rs) else np.nan,
                'S': float(total),
                'Tc_ref': float(tc_ref),
                'plateau': bool(plateau),
            }

        best = min(scores, key=lambda k: scores[k]['S']) if scores else None
        self.results['scores'] = scores
        self.results['best_model'] = best

        # ---- Restore final fitted parameters (critical!) ----
        self.results['beta'] = b_keep
        self.results['gamma'] = g_keep
        self.results['Tc'] = tc_keep
        if d_keep is not None:
            self.results['delta'] = d_keep
        else:
            if np.isfinite(b_keep) and np.isfinite(g_keep) and b_keep != 0:
                self.results['delta'] = 1.0 + g_keep / b_keep

        return scores

    # ---------- 4) KF + correction-to-scaling ----------

    def perform_kf_and_correction(self, use_correction: bool = False, *, omega: float = 0.5,
                                  fit_omega: bool = True,
                                  override_exponents: bool = True,
                                  update_results: bool = True) -> Tuple[List[Dict], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract Ms(T) and Chi0^{-1}(T) from MAP and do KF (+ optional correction-to-scaling fits).

        KF background (what the code fits):
        - For T < Tc:   Ms(T) ~ (Tc - T)^beta  ->  Ms / (dMs/dT) = (T - Tc)/beta
        - For T > Tc:   Chi0^{-1}(T) ~ (T - Tc)^gamma -> ChiInv/(dChiInv/dT) = (T - Tc)/gamma

        So in a KF plot (y vs T), the *intercept* gives Tc and the *slope* gives 1/beta or 1/gamma.
        """
        if not self.data_files:
            return [], (np.array([]), np.array([]), np.array([]))

        b = float(self.results.get('beta', 0.365))
        g = float(self.results.get('gamma', 1.386))

        # ---- 1) MAP extraction: Ms(T) and ChiInv(T) ----
        map_res: List[Dict] = []
        for d in self.data_files:
            H, M = self._get_filtered_data(d)
            if H.size < 6:
                continue
            # Modified Arrott plot linearization (Arrott–Noakes), robust high-field fit:
            fit = self._fit_modified_arrott_isotherm(H, M, beta=b, gamma=g)
            if fit is None:
                continue
            b0_fit = float(fit['intercept'])
            a_fit = float(fit['slope'])
            r2_fit = float(fit.get('r2', np.nan))
            ms = (b0_fit ** b) if (np.isfinite(b0_fit) and b0_fit > 0) else np.nan
            x0 = (-b0_fit / a_fit) if (np.isfinite(b0_fit) and np.isfinite(a_fit) and a_fit != 0) else np.nan
            chi_inv = (x0 ** g) if (np.isfinite(x0) and x0 > 0) else np.nan

            # Classic Arrott (mean-field) intercept for an independent Tc diagnostic:
            mf_fit = self._fit_arrott_mf_isotherm(H, M)
            arrott_mf_b0 = float(mf_fit['intercept']) if mf_fit is not None and np.isfinite(mf_fit.get('intercept', np.nan)) else np.nan
            arrott_mf_r2 = float(mf_fit.get('r2', np.nan)) if mf_fit is not None else np.nan

            map_res.append({'T': float(d['T']),
                'Ms': float(ms) if np.isfinite(ms) else np.nan,
                'ChiInv': float(chi_inv) if np.isfinite(chi_inv) else np.nan,
                'MAP_intercept': float(b0_fit) if np.isfinite(b0_fit) else np.nan,
                'MAP_slope': float(a_fit) if np.isfinite(a_fit) else np.nan,
                'MAP_r2': float(r2_fit) if np.isfinite(r2_fit) else np.nan,
                'ArrottMF_intercept': float(arrott_mf_b0) if np.isfinite(arrott_mf_b0) else np.nan,
                'ArrottMF_r2': float(arrott_mf_r2) if np.isfinite(arrott_mf_r2) else np.nan})

        if not map_res:
            if update_results:
                self.results['kf_map'] = []
            return [], (np.array([]), np.array([]), np.array([]))

        temps = np.array([r['T'] for r in map_res], dtype=float)
        Ms = np.array([r['Ms'] for r in map_res], dtype=float)
        ChiInv = np.array([r['ChiInv'] for r in map_res], dtype=float)

        # ---- 1b) Estimate Tc from Modified Arrott Plot (intercept crossing) ----
        b0 = np.array([r.get('MAP_intercept', np.nan) for r in map_res], dtype=float)
        r2 = np.array([r.get('MAP_r2', np.nan) for r in map_res], dtype=float)
        Tc_Arrott = None
        Tc_Arrott_err = None
        try:
            mm = np.isfinite(temps) & np.isfinite(b0)
            # prefer high-linearity points if available
            if np.sum(mm & (r2 > 0.7)) >= 3:
                mm = mm & (r2 > 0.9)
            tt = np.asarray(temps[mm], float)
            bb = np.asarray(b0[mm], float)
            tt, bb = _robust_sort_xy(tt, bb)
            if tt.size >= 2:
                # find all sign changes
                sgn = np.sign(bb)
                crossings = []
                for i in range(tt.size - 1):
                    if sgn[i] == 0:
                        crossings.append((tt[i], abs(bb[i]), abs(tt[i+1]-tt[i])))
                    elif sgn[i] * sgn[i+1] < 0:
                        # linear interpolation for b0=0
                        t1, t2 = float(tt[i]), float(tt[i+1])
                        b1, b2 = float(bb[i]), float(bb[i+1])
                        if (b2 - b1) != 0:
                            tc = t1 + (0.0 - b1) * (t2 - t1) / (b2 - b1)
                            crossings.append((tc, 0.0, abs(t2 - t1)))
                if crossings:
                    # choose crossing closest to the point where |b0| is smallest (usually near Tc)
                    crossings = sorted(crossings, key=lambda x: (x[1], x[2]))
                    Tc_Arrott = float(crossings[0][0])
                    Tc_Arrott_err = float(0.5 * crossings[0][2]) if np.isfinite(crossings[0][2]) else None
                else:
                    # fallback: closest-to-zero intercept temperature
                    j = int(np.nanargmin(np.abs(bb)))
                    Tc_Arrott = float(tt[j])
                    # uncertainty floor from temperature resolution
                    dtt = np.diff(tt)
                    dt_med = float(np.nanmedian(dtt)) if dtt.size else np.nan
                    Tc_Arrott_err = float(0.5 * dt_med) if np.isfinite(dt_med) and dt_med > 0 else 0.5
        except Exception:
            Tc_Arrott = None
            Tc_Arrott_err = None

        if update_results and Tc_Arrott is not None and np.isfinite(float(Tc_Arrott)):
            self.results['Tc_Arrott'] = float(Tc_Arrott)
            if Tc_Arrott_err is not None and np.isfinite(float(Tc_Arrott_err)):
                self.results.setdefault('errors', {})['Tc_Arrott'] = float(Tc_Arrott_err)

        
        # ---- 1c) Estimate Tc from classic Arrott Plot (mean-field, M^2 vs H/M) ----
        Tc_Arrott_MF = None
        Tc_Arrott_MF_err = None
        try:
            b0mf = np.array([r.get('ArrottMF_intercept', np.nan) for r in map_res], dtype=float)
            r2mf = np.array([r.get('ArrottMF_r2', np.nan) for r in map_res], dtype=float)
            mm = np.isfinite(temps) & np.isfinite(b0mf)
            if np.sum(mm & (r2mf > 0.70)) >= 3:
                mm = mm & (r2mf > 0.90)
            tt = np.asarray(temps[mm], float)
            bb = np.asarray(b0mf[mm], float)
            tt, bb = _robust_sort_xy(tt, bb)
            if tt.size >= 2:
                sgn = np.sign(bb)
                crossings = []
                for i in range(tt.size - 1):
                    if sgn[i] == 0:
                        crossings.append((float(tt[i]), abs(float(bb[i])), abs(float(tt[i+1]-tt[i]))))
                    elif sgn[i] * sgn[i+1] < 0:
                        t1, t2 = float(tt[i]), float(tt[i+1])
                        b1, b2 = float(bb[i]), float(bb[i+1])
                        if (b2 - b1) != 0:
                            tc = t1 + (0.0 - b1) * (t2 - t1) / (b2 - b1)
                            crossings.append((tc, 0.0, abs(t2 - t1)))
                if crossings:
                    crossings = sorted(crossings, key=lambda x: (x[1], x[2]))
                    Tc_Arrott_MF = float(crossings[0][0])
                    Tc_Arrott_MF_err = float(0.5 * crossings[0][2]) if np.isfinite(crossings[0][2]) else None
                else:
                    j = int(np.nanargmin(np.abs(bb)))
                    Tc_Arrott_MF = float(tt[j])
                    dtt = np.diff(tt)
                    dt_med = float(np.nanmedian(dtt)) if dtt.size else np.nan
                    Tc_Arrott_MF_err = float(0.5 * dt_med) if np.isfinite(dt_med) and dt_med > 0 else 0.5
        except Exception:
            Tc_Arrott_MF = None
            Tc_Arrott_MF_err = None

        if update_results and Tc_Arrott_MF is not None and np.isfinite(float(Tc_Arrott_MF)):
            self.results['Tc_Arrott_MF'] = float(Tc_Arrott_MF)
            if Tc_Arrott_MF_err is not None and np.isfinite(float(Tc_Arrott_MF_err)):
                self.results.setdefault('errors', {})['Tc_Arrott_MF'] = float(Tc_Arrott_MF_err)

# ---- 2) KF fit helper ----
        def _kf_fit(t_arr: np.ndarray, y_arr: np.ndarray) -> Optional[Dict[str, float]]:
            m = np.isfinite(t_arr) & np.isfinite(y_arr)
            t = np.asarray(t_arr[m], float)
            y = np.asarray(y_arr[m], float)
            if t.size < 3:
                return None
            t, y = _robust_sort_xy(t, y)
            try:
                if t.size >= 4:
                    (a, b0), cov = np.polyfit(t, y, 1, cov=True)  # y=a*T+b0
                else:
                    a, b0 = np.polyfit(t, y, 1)
                    cov = np.full((2, 2), np.nan)

            except Exception:
                return None
            if (not np.isfinite(a)) or a == 0 or (not np.isfinite(b0)):
                return None
            Tc = -b0 / a
            # errors
            try:
                var_a = float(cov[0, 0])
                var_b = float(cov[1, 1])
                cov_ab = float(cov[0, 1])
                var_Tc = (b0 * b0 / (a ** 4)) * var_a + (1.0 / (a ** 2)) * var_b - 2.0 * (b0 / (a ** 3)) * cov_ab
                Tc_se = float(np.sqrt(max(var_Tc, 0.0)))
                a_se = float(np.sqrt(max(var_a, 0.0)))
            except Exception:
                Tc_se, a_se = np.nan, np.nan
            out = {'slope': float(a), 'intercept': float(b0), 'Tc': float(Tc),
                   'Tc_se': float(Tc_se) if np.isfinite(Tc_se) else np.nan,
                   'slope_se': float(a_se) if np.isfinite(a_se) else np.nan}
            return out

        def _kf_jackknife_tc(t_arr: np.ndarray, y_arr: np.ndarray) -> Optional[float]:
            """Jackknife standard error of Tc from KF linear fit (robust, non-zero)."""
            m = np.isfinite(t_arr) & np.isfinite(y_arr)
            t = np.asarray(t_arr[m], float)
            y = np.asarray(y_arr[m], float)
            if t.size < 6:
                return None
            # leave-one-out fits
            est = []
            for i in range(t.size):
                tt = np.delete(t, i)
                yy = np.delete(y, i)
                fit = _kf_fit(tt, yy)
                if fit is not None and np.isfinite(fit.get('Tc', np.nan)):
                    est.append(float(fit['Tc']))
            if len(est) < 4:
                return None
            est = np.asarray(est, float)
            theta_bar = float(np.nanmean(est))
            se = float(np.sqrt((len(est) - 1) / len(est) * np.nansum((est - theta_bar) ** 2)))
            if not np.isfinite(se):
                return None
            return se

        def _kf_prepare_and_fit(y_raw: np.ndarray, *, side: str, tc_seed: Optional[float]) -> Tuple[Optional[Dict[str, float]], np.ndarray]:
            """Prepare KF ordinate y/(dy/dT) and fit it with a line.

            IMPORTANT FIX (v22):
            - Ms-KF must use T < Tc (ferromagnetic side) and ChiInv-KF must use T > Tc (paramagnetic side).
              Previous versions could include 'wrong-side' points if MAP gave finite Ms above Tc (or vice versa),
              which biases both Tc and the slope (thus beta/gamma).
            """
            m = np.isfinite(temps) & np.isfinite(y_raw)
            if tc_seed is not None and isinstance(tc_seed, (int, float)) and np.isfinite(tc_seed):
                if side.lower().startswith('below'):
                    m = m & (temps < float(tc_seed))
                elif side.lower().startswith('above'):
                    m = m & (temps > float(tc_seed))

            t = np.asarray(temps[m], float)
            y = np.asarray(y_raw[m], float)
            if t.size < 3:
                return None, np.full_like(temps, np.nan, dtype=float)
            t, y = _robust_sort_xy(t, y)

            # smooth derivative with a spline if possible (more stable than raw gradient)
            try:
                s = float(max(1e-12, 0.05 * t.size * np.nanvar(y)))
                spl = UnivariateSpline(t, y, s=s, k=3)
                dy = spl.derivative()(t)
            except Exception:
                dy = np.gradient(y, t)

            kf = np.full_like(temps, np.nan, dtype=float)
            # avoid divide by tiny derivative
            good = np.isfinite(dy) & (np.abs(dy) > 1e-12) & np.isfinite(y)
            kf_vals = np.full_like(t, np.nan, dtype=float)
            kf_vals[good] = y[good] / dy[good]

            # initial fit
            fit1 = _kf_fit(t, kf_vals)

            # refine around Tc if available (narrow window around intercept)
            fit = fit1
            if fit1 is not None and np.isfinite(fit1.get('Tc', np.nan)):
                Tc0 = float(fit1['Tc'])
                # tighter window improves Tc for dense datasets like Ni
                win = float(min(0.12 * (np.nanmax(t) - np.nanmin(t)), 12.0))
                mm = np.isfinite(kf_vals) & (np.abs(t - Tc0) <= win)
                if np.sum(mm) >= 4:
                    fit2 = _kf_fit(t[mm], kf_vals[mm])
                    if fit2 is not None:
                        fit = fit2

            # map back to full temps order (for plotting)
            for ti, ki in zip(t, kf_vals):
                idx = np.where(np.isclose(temps, ti, rtol=0, atol=1e-9))[0]
                if idx.size:
                    kf[idx[0]] = ki

            return fit, kf

        # ---- 3) Build KF arrays + fits ----
        # Seed for KF side selection: use the current best Tc, with a robust fallback.
        # Older versions used Tc_Arrott first; if Tc_Arrott is biased (e.g., wrong beta/gamma),
        # it can accidentally filter out the whole FM/PM side and make KF 'N/A'.
        tc_seed = self.results.get('Tc', None)
        tc_candidates = []
        for key in ['Tc', 'Tc_Arrott', 'Tc_Arrott_MF']:
            v = self.results.get(key, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                tc_candidates.append(float(v))
        if tc_candidates:
            tc_seed = float(np.nanmedian(np.asarray(tc_candidates, float)))
        fit_ms, kf_ms = _kf_prepare_and_fit(Ms, side='below', tc_seed=tc_seed)
        fit_chi, kf_chi = _kf_prepare_and_fit(ChiInv, side='above', tc_seed=tc_seed)

        # beta_KF / gamma_KF from slopes
        beta_kf = gamma_kf = np.nan
        beta_kf_se = gamma_kf_se = np.nan
        tc_ms_val = tc_ms_err = None
        if fit_ms is not None:
            tc_ms_val = float(fit_ms.get('Tc', np.nan))
            tc_ms_err = float(fit_ms.get('Tc_se', np.nan))
            a = float(fit_ms.get('slope', np.nan))
            a_se = float(fit_ms.get('slope_se', np.nan))
            if np.isfinite(a) and a != 0:
                beta_kf = 1.0 / a
                if np.isfinite(a_se):
                    beta_kf_se = abs(a_se / (a * a))

        tc_chi_val = tc_chi_err = None
        if fit_chi is not None:
            tc_chi_val = float(fit_chi.get('Tc', np.nan))
            tc_chi_err = float(fit_chi.get('Tc_se', np.nan))
            a = float(fit_chi.get('slope', np.nan))
            a_se = float(fit_chi.get('slope_se', np.nan))
            if np.isfinite(a) and a != 0:
                gamma_kf = 1.0 / a
                if np.isfinite(a_se):
                    gamma_kf_se = abs(a_se / (a * a))

        # ---- 4) Combine Tc candidates (keep within measured window) ----
        Tc_KF = None
        Tc_KF_err = None
        cands = []
        if tc_ms_val is not None and np.isfinite(tc_ms_val):
            cands.append(('ms', float(tc_ms_val), float(tc_ms_err) if tc_ms_err is not None and np.isfinite(tc_ms_err) else None))
        if tc_chi_val is not None and np.isfinite(tc_chi_val):
            cands.append(('chi', float(tc_chi_val), float(tc_chi_err) if tc_chi_err is not None and np.isfinite(tc_chi_err) else None))

        tmin, tmax = float(np.nanmin(temps)), float(np.nanmax(temps))
        cands = [c for c in cands if (tmin < c[1] < tmax)]

        if len(cands) == 2 and cands[0][2] is not None and cands[1][2] is not None and cands[0][2] > 0 and cands[1][2] > 0:
            (_, t1, s1), (_, t2, s2) = cands
            w1, w2 = 1.0 / (s1 * s1), 1.0 / (s2 * s2)
            Tc_KF = (w1 * t1 + w2 * t2) / (w1 + w2)
            Tc_KF_err = float(np.sqrt(1.0 / (w1 + w2)))
        elif len(cands) >= 1:
            Tc_KF = cands[0][1]
            Tc_KF_err = cands[0][2]

        # ---- 4b) Robust Tc uncertainty (jackknife + minimum floor) ----
        if Tc_KF is not None and np.isfinite(float(Tc_KF)):
            # jackknife on KF arrays (more realistic than a near-zero polyfit covariance)
            jk_ms = _kf_jackknife_tc(temps, kf_ms)
            jk_chi = _kf_jackknife_tc(temps, kf_chi)
            jk_list = [v for v in [jk_ms, jk_chi] if v is not None and np.isfinite(v)]
            jk = float(np.nanmedian(jk_list)) if jk_list else np.nan

            # minimum uncertainty floor tied to temperature resolution
            tt = np.sort(np.asarray(temps, float))
            dtt = np.diff(tt[np.isfinite(tt)])
            dt_med = float(np.nanmedian(dtt)) if dtt.size else np.nan
            floor = float(0.25 * dt_med) if np.isfinite(dt_med) and dt_med > 0 else 0.5

            cand_errs = []
            if Tc_KF_err is not None and np.isfinite(float(Tc_KF_err)):
                cand_errs.append(float(Tc_KF_err))
            if np.isfinite(jk):
                cand_errs.append(float(jk))
            cand_errs.append(float(floor))
            Tc_KF_err = float(np.nanmax(cand_errs))

        # ---- 5) Optional correction-to-scaling fits (uses Tc_ref) ----
        fit_results: Dict[str, Dict] = {}
        b_fit = b
        g_fit = g

        if use_correction:
            tc_ref = float(self.results.get('Tc', np.nan))
            if (not np.isfinite(tc_ref)) and (Tc_KF is not None and np.isfinite(Tc_KF)):
                tc_ref = float(Tc_KF)

            # Below Tc: Ms
            m = np.isfinite(Ms) & (temps < tc_ref)
            if np.sum(m) > 6:
                y = Ms[m]

                def f_ms(T, M0, beta, a, omega_):
                    t = np.maximum((tc_ref - T) / tc_ref, 1e-9)
                    return M0 * t ** beta * (1.0 + a * t ** omega_)

                p0 = [float(np.nanmax(y)), b, 0.05, omega]
                bounds = ([0, 0.05, -5, 0.1], [np.inf, 2.0, 5, 2.0])
                if not fit_omega:
                    bounds = ([0, 0.05, -5, omega], [np.inf, 2.0, 5, omega])
                try:
                    popt, pcov = curve_fit(f_ms, temps[m], y, p0=p0, bounds=bounds, maxfev=20000)
                    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full_like(popt, np.nan)
                    fit_results['Ms'] = {'M0': popt[0], 'beta': popt[1], 'a': popt[2], 'omega': popt[3],
                                         'err': {'M0': perr[0], 'beta': perr[1], 'a': perr[2], 'omega': perr[3]}}
                    b_fit = float(popt[1])
                except Exception:
                    pass

            # Above Tc: ChiInv
            m = np.isfinite(ChiInv) & (temps > tc_ref)
            if np.sum(m) > 6:
                y = ChiInv[m]

                def f_chi(T, C0, gamma, b_, omega_):
                    t = np.maximum((T - tc_ref) / tc_ref, 1e-9)
                    return C0 * t ** gamma * (1.0 + b_ * t ** omega_)

                p0 = [float(np.nanmin(y)), g, 0.05, omega]
                bounds = ([0, 0.2, -5, 0.1], [np.inf, 5.0, 5, 2.0])
                if not fit_omega:
                    bounds = ([0, 0.2, -5, omega], [np.inf, 5.0, 5, omega])
                try:
                    popt, pcov = curve_fit(f_chi, temps[m], y, p0=p0, bounds=bounds, maxfev=20000)
                    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full_like(popt, np.nan)
                    fit_results['ChiInv'] = {'C0': popt[0], 'gamma': popt[1], 'b': popt[2], 'omega': popt[3],
                                             'err': {'C0': perr[0], 'gamma': perr[1], 'b': perr[2], 'omega': perr[3]}}
                    g_fit = float(popt[1])
                except Exception:
                    pass

        if update_results:
            if Tc_KF is not None and np.isfinite(Tc_KF):
                self.results['Tc_KF'] = float(Tc_KF)
                if Tc_KF_err is not None and np.isfinite(Tc_KF_err):
                    errs = self.results.setdefault('errors', {})
                    errs['Tc_KF'] = float(Tc_KF_err)
                    # If Tc is already set to Tc_KF, keep Tc uncertainty consistent
                    tc_cur = float(self.results.get('Tc', np.nan))
                    if np.isfinite(tc_cur) and abs(tc_cur - float(Tc_KF)) < 1e-6:
                        errs['Tc'] = float(Tc_KF_err)

            # store KF exponents (diagnostic)
            self.results['beta_KF'] = float(beta_kf) if np.isfinite(beta_kf) else None
            self.results['gamma_KF'] = float(gamma_kf) if np.isfinite(gamma_kf) else None
            self.results['kf_fits'] = {'Ms': fit_ms, 'ChiInv': fit_chi}
            self.results['kf_lines'] = {'Ms': kf_ms, 'ChiInv': kf_chi}

            self.results['correction_fit'] = fit_results
            if use_correction:
                # Store correction-to-scaling exponents separately; optionally override the main exponents.
                self.results['beta_corr'] = float(b_fit)
                self.results['gamma_corr'] = float(g_fit)
                self.results['delta_corr'] = 1.0 + float(g_fit) / float(b_fit) if float(b_fit) != 0 else np.nan
                if override_exponents:
                    self.results['beta'] = float(b_fit)
                    self.results['gamma'] = float(g_fit)
                    self.results['delta'] = 1.0 + float(g_fit) / float(b_fit)

            # Cache for plotting
            self.results['kf_map'] = map_res
            self.results['kf_arrays'] = (temps, Ms, ChiInv)

        return map_res, (temps, Ms, ChiInv)

    # ---------- 5) GP scaling (Bayesian) ----------

    def _build_scaled_dataset(self, beta: float, gamma: float, tc: float, *,
                              max_points: int = 2500,
                              per_curve: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for GP scaling.

        X = signed log10(x) where x = H/|eps|^{beta+gamma}; sign flipped for T<Tc
        y = log10( M/|eps|^{beta} )
        """
        X_list = []
        y_list = []
        for d in self.data_files:
            T = float(d['T'])
            eps = (T - tc) / tc
            if abs(eps) < 1e-6:
                continue
            H, M = self._get_filtered_data(d)
            if H.size < 6:
                continue
            if H.size > per_curve:
                idx = np.linspace(0, H.size - 1, per_curve).astype(int)
                H = H[idx]; M = M[idx]
            x = H / (abs(eps) ** (beta + gamma))
            y = M / (abs(eps) ** beta)
            # Avoid zeros
            m = (x > 0) & (y > 0)
            x = x[m]; y = y[m]
            if x.size < 4:
                continue
            lx = np.log10(x)
            if eps < 0:
                lx = -lx
            ly = np.log10(y)
            X_list.append(lx)
            y_list.append(ly)

        if not X_list:
            return np.empty((0, 1)), np.empty((0,))
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        m = np.isfinite(X) & np.isfinite(y)
        X = X[m]; y = y[m]
        if X.size > max_points:
            idx = np.random.choice(X.size, size=max_points, replace=False)
            X = X[idx]; y = y[idx]
        return X.reshape(-1, 1), y

    def gp_neg_log_marginal_likelihood(self, beta: float, gamma: float, tc: float) -> float:
        if not SKLEARN_AVAILABLE:
            return 1e99
        X, y = self._build_scaled_dataset(beta, gamma, tc)
        if X.shape[0] < 80:
            return 1e99
        # Kernel: C * RBF + white
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1.0))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=2, random_state=0)
        gp.fit(X, y)
        ll = gp.log_marginal_likelihood_value_
        # cache
        self._gp_cache['last_gp'] = gp
        self._gp_cache['last_Xy'] = (X, y)
        return float(-ll)

    def perform_gp_scaling(self, *, initial: Optional[Tuple[float, float, float]] = None, mcmc: bool = True,
                           n_mcmc: int = 1200, burn: int = 300, use_mcmc: Optional[bool] = None, **kwargs) -> Dict:
        """Fit GP scaling by minimizing negative log marginal likelihood.

        Optionally do Metropolis sampling to get posterior over (beta,gamma,Tc).
        """

        # Backward-compatible alias (older GUI passed use_mcmc=...)
        if use_mcmc is not None:
            mcmc = bool(use_mcmc)

        if not SKLEARN_AVAILABLE:
            self.results['gp_results'] = False
            return {'success': False, 'reason': 'sklearn unavailable'}

        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        tc0 = float(self.results['Tc'] if self.results['Tc'] else np.nanmean(temps))
        b0 = float(self.results['beta'])
        g0 = float(self.results['gamma'])
        if initial is not None:
            b0, g0, tc0 = map(float, initial)

        def obj(p):
            b, g, tc = float(p[0]), float(p[1]), float(p[2])
            if not (0.05 < b < 1.2 and 0.2 < g < 3.5 and np.nanmin(temps) < tc < np.nanmax(temps)):
                return 1e99
            return self.gp_neg_log_marginal_likelihood(b, g, tc)

        x0 = np.array([b0, g0, tc0], dtype=float)
        res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 260, 'xatol': 2e-3, 'fatol': 1e-2})
        b_hat, g_hat, tc_hat = map(float, res.x)
        nll = float(res.fun)

        out = {'success': bool(res.success), 'beta': b_hat, 'gamma': g_hat, 'Tc': tc_hat, 'nll': nll}
        self.results['gp_best_params'] = (b_hat, g_hat, tc_hat)
        self.results['gp_nll'] = nll
        self.results['gp_results'] = out

        if mcmc:
            samples, acc = self._metropolis(lambda p: -obj(p),
                                            x0=np.array([b_hat, g_hat, tc_hat], dtype=float),
                                            step=np.array([0.01, 0.02, 0.2], dtype=float),
                                            bounds=np.array([[0.05, 1.2], [0.2, 3.5], [np.nanmin(temps), np.nanmax(temps)]], dtype=float),
                                            n=int(n_mcmc), burn=int(burn))
            self.results['gp_posterior_samples'] = samples
            self.results['gp_mcmc_accept'] = acc
            ci = np.quantile(samples, [0.16, 0.5, 0.84], axis=0)
            self.results['gp_cred_int'] = {'beta': tuple(ci[:, 0]), 'gamma': tuple(ci[:, 1]), 'Tc': tuple(ci[:, 2])}
            out['posterior_ci_16_50_84'] = self.results['gp_cred_int']

        return out

    # ---------- 6) RG-inspired parametric EOS fit ----------

    def fit_eos_parametric(self, *, model_name: str, init_tc: Optional[float] = None,
                           fit_scales: bool = True,
                           max_points: int = 3000) -> Dict:
        """Fit non-universal scales (m0,h0) and Tc using a universal parametric EOS.

        Bugfixes / robustness:
          1) The previous version accidentally passed |t| into the u-solver, which
             forces the "t>0" branch even when T<Tc. This can make (u^2-1) negative
             and causes invalid R and NaNs downstream (fit crashes, chi2 becomes NaN).
          2) Handle very small |t| in a sign-preserving way.
          3) Optimize scales in log-space to enforce positivity.
          4) Always store a result (success=False + reason) so the GUI/report does not
             silently fall back to "None" / NaN.
        """
        if model_name not in self.eos_models:
            raise ValueError(f"Unknown EOS model: {model_name}")
        model = self.eos_models[model_name]

        # ---- 1) Collect (T,H,M) points ----
        pts: List[Tuple[float, float, float]] = []
        for d in self.data_files:
            H, M = self._get_filtered_data(d)
            if H.size < 6:
                continue
            T = float(d['T'])
            n = H.size
            idx = np.linspace(0, n - 1, min(40, n)).astype(int)
            for i in idx:
                pts.append((T, float(H[i]), float(M[i])))

        # dynamic minimum (keeps EOS usable on smaller datasets)
        min_pts = int(max(80, min(250, 10 * max(1, len(self.data_files)))))

        if len(pts) < min_pts:
            out = {'success': False, 'reason': f'too_few_points ({len(pts)}<{min_pts})'}
            self.results.setdefault('eos_fits', {})[model_name] = out
            return out

        if len(pts) > max_points:
            sel = np.random.choice(len(pts), size=max_points, replace=False)
            pts = [pts[i] for i in sel]

        temps = np.array([p[0] for p in pts], dtype=float)
        H_dat = np.array([p[1] for p in pts], dtype=float)
        M_dat = np.array([p[2] for p in pts], dtype=float)

        tmin, tmax = float(np.nanmin(temps)), float(np.nanmax(temps))
        tc0 = float(init_tc) if init_tc is not None else float(np.nanmedian(temps))

        # initial scales (rough)
        m0_0 = float(np.nanmedian(M_dat))
        h0_0 = float(np.nanmedian(H_dat))
        m0_0 = max(m0_0, 1e-12)
        h0_0 = max(h0_0, 1e-12)

        def solve_u_given_M_t(M: float, t: float, m0: float) -> float:
            """Solve M = m0 * R^beta * m(u) for u, given signed t."""
            b = float(model.beta)
            mfunc = model.m_func

            def f(u: float) -> float:
                u = float(u)
                if t > 0:
                    denom = 1.0 - u*u
                    if denom <= 0:
                        return 1e9
                    R = t / denom
                else:
                    denom = u*u - 1.0
                    if denom <= 0:
                        return 1e9
                    R = abs(t) / denom
                if R <= 0:
                    return 1e9
                return float(m0 * (R ** b) * mfunc(u) - M)

            # Above Tc: u in (0,1); below Tc: u in (1,u0)
            if t > 0:
                lo, hi = 1e-6, 0.999
            else:
                lo, hi = 1.001, min(float(model.u0) - 1e-4, 3.0)

            us = np.linspace(lo, hi, 80)
            fs = np.array([f(u) for u in us], dtype=float)

            # fallback: min |f|
            try:
                i0 = int(np.nanargmin(np.abs(fs)))
                u_best = float(us[i0])
            except Exception:
                u_best = float(0.5 * (lo + hi))

            # bisection if sign change exists
            for i in range(len(us) - 1):
                if np.sign(fs[i]) == 0:
                    return float(us[i])
                if np.sign(fs[i]) != np.sign(fs[i + 1]):
                    a, bnd = float(us[i]), float(us[i + 1])
                    fa, fb = float(fs[i]), float(fs[i + 1])
                    for _ in range(40):
                        mid = 0.5 * (a + bnd)
                        fm = f(mid)
                        if np.sign(fm) == 0:
                            return float(mid)
                        if np.sign(fa) != np.sign(fm):
                            bnd, fb = mid, fm
                        else:
                            a, fa = mid, fm
                    return float(0.5 * (a + bnd))
            return u_best

        def predict_H(T: float, M: float, m0: float, h0: float, Tc: float) -> float:
            t = (T - Tc) / Tc
            # sign-preserving small-|t|
            if abs(t) < 1e-10:
                t = (1.0 if t >= 0 else -1.0) * 1e-10

            u = solve_u_given_M_t(M, t, m0)

            if t > 0:
                denom = 1.0 - u*u
            else:
                denom = u*u - 1.0
            if denom <= 0:
                return np.nan

            R = abs(t) / denom
            if R <= 0:
                return np.nan

            return float(h0 * (R ** (float(model.beta) * float(model.delta))) * model.h_func(u))

        # ---- 2) Objective ----
        min_required = int(min(200, max(60, 0.25 * len(pts))))

        def _clip_tc(Tc: float) -> float:
            return float(np.clip(Tc, tmin + 1e-6, tmax - 1e-6))

        def obj(p):
            if fit_scales:
                log_m0, log_h0, Tc_raw = map(float, p)
                m0 = float(np.exp(log_m0))
                h0 = float(np.exp(log_h0))
                Tc = _clip_tc(Tc_raw)
            else:
                Tc = _clip_tc(float(p[0]))
                m0, h0 = m0_0, h0_0

            H_pred = np.array([predict_H(temps[i], M_dat[i], m0, h0, Tc)
                               for i in range(len(pts))], dtype=float)

            m = np.isfinite(H_pred) & (H_pred > 0) & (H_dat > 0)
            if int(np.sum(m)) < min_required:
                return 1e99

            r = np.log(H_pred[m]) - np.log(H_dat[m])
            s = np.nanmedian(np.abs(r)) + 1e-12
            rr = np.where(np.abs(r) < 3*s, r, 3*s*np.sign(r))
            return float(np.nanmean(rr**2))

        # ---- 3) Optimize ----
        try:
            if fit_scales:
                x0 = np.array([np.log(m0_0), np.log(h0_0), tc0], dtype=float)
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-3})
                log_m0_hat, log_h0_hat, tc_hat_raw = map(float, res.x)
                m0_hat, h0_hat = float(np.exp(log_m0_hat)), float(np.exp(log_h0_hat))
                tc_hat = _clip_tc(tc_hat_raw)
            else:
                x0 = np.array([tc0], dtype=float)
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 220, 'xatol': 1e-3, 'fatol': 1e-3})
                tc_hat = _clip_tc(float(res.x[0]))
                m0_hat, h0_hat = m0_0, h0_0

            chi2 = float(obj([np.log(m0_hat), np.log(h0_hat), tc_hat] if fit_scales else [tc_hat]))
        except Exception as e:
            out = {'success': False, 'reason': f'exception: {e}'}
            self.results.setdefault('eos_fits', {})[model_name] = out
            return out

        out = {
            'success': bool(res.success) and np.isfinite(chi2) and (chi2 < 1e50),
            'model': model_name,
            'Tc': float(tc_hat),
            'm0': float(m0_hat),
            'h0': float(h0_hat),
            'chi2_red': float(chi2),
            'n_points': int(len(pts)),
            'n_used': int(min_required)
        }

        # ---- 4) Store + pick best EOS model by minimum chi2_red (among successes) ----
        self.results.setdefault('eos_fits', {})[model_name] = out

        if out['success']:
            # keep best
            best_name = self.results.get('eos_best_model', None)
            best = self.results.get('eos_params', None)
            if (best is None) or (not isinstance(best, dict)) or (not best.get('success', False)) or (out['chi2_red'] < best.get('chi2_red', 1e99)):
                self.results['eos_best_model'] = model_name
                self.results['eos_params'] = out

        return out

    def get_effective_tc(self, prefer_kf: bool = True) -> float:
        """Return the Tc value that should be used consistently across plots/MCE.

        If prefer_kf=True and a finite Tc_KF exists, it is used (KF is typically more reliable).
        """
        tc_kf = self.results.get('Tc_KF', None)
        if prefer_kf and isinstance(tc_kf, (int, float)) and np.isfinite(tc_kf):
            return float(tc_kf)
        tc = self.results.get('Tc', np.nan)
        return float(tc) if np.isfinite(tc) else float('nan')

    # ---------- 7) MCE analysis with smoothing + uncertainty ----------

    def mce_analysis(self, advanced_smooth: bool = True, *,
                     grid_H: int = 80,
                     deriv_method: str = 'spline',
                     bootstrap_sigma: int = 60) -> Optional[Dict]:
        """Compute ΔS_M(T,H) via Maxwell relation using smoothed (∂M/∂T)_H.

        Maxwell relation (discrete form):
            ΔS_M(T,H) = ∫_0^H (∂M/∂T)_H' dH'

        Key bugfix (v19):
        - n_exp was previously computed at the *nearest* measured temperature to Tc.
          If Tc shifts by a few K, n_exp can be badly biased.
          We now interpolate ΔS(T,H) to Tc and ALSO compute a Tc-free exponent from ΔS_max(H).
        - Integration over field previously used a left-Riemann cumsum; now uses cumulative trapezoid.
        - Tc used for MCE is now the *effective Tc* (KF-preferred) to keep global consistency.
        """
        if not self.data_files:
            return None
        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        if temps.size < 6:
            return None

        # Ensure temperatures are strictly increasing for splines/interp
        t_order = np.argsort(temps)
        temps = temps[t_order]

        # Build common H grid
        Hmax = float(np.nanmax([np.nanmax(np.abs(d['H'])) for d in self.data_files]))
        if not np.isfinite(Hmax) or Hmax <= 0:
            return None
        H_grid = np.linspace(0.0, Hmax, int(grid_H))

        # Interpolate each isotherm to common H grid
        M_grid = np.zeros((len(self.data_files), len(H_grid)), dtype=float)
        for ii, i in enumerate(t_order):
            d = self.data_files[int(i)]
            H, M = _robust_sort_xy(np.abs(d['H']), d['M'])
            f = interpolate.interp1d(H, M, bounds_error=False, fill_value='extrapolate')
            M_grid[ii, :] = f(H_grid)

        # Smooth in T for each H, then differentiate
        dM_dT = np.zeros_like(M_grid)
        sigma_M = np.zeros_like(M_grid)

        if (deriv_method == 'gp') and SKLEARN_AVAILABLE:
            for j in range(len(H_grid)):
                y = M_grid[:, j]
                X = temps.reshape(-1, 1)
                kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                          RBF(length_scale=np.std(temps) + 1e-6, length_scale_bounds=(1e-3, 1e3)) +
                          WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 1.0)))
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                                              n_restarts_optimizer=1, random_state=0)
                try:
                    gp.fit(X, y)
                    y_pred, y_std = gp.predict(X, return_std=True)
                    sigma_M[:, j] = y_std
                    dM_dT[:, j] = np.gradient(y_pred, temps)
                except Exception:
                    dM_dT[:, j] = np.gradient(y, temps)
                    sigma_M[:, j] = np.nanstd(y - np.nanmedian(y))
        else:
            for j in range(len(H_grid)):
                y = M_grid[:, j]
                if advanced_smooth:
                    try:
                        # A light, data-driven smoothing (avoid exact interpolation of noise)
                        s = float((0.15 * (np.nanstd(y) + 1e-12)) ** 2 * len(temps))
                        spl = UnivariateSpline(temps, y, s=s)
                        y_fit = spl(temps)
                        dM_dT[:, j] = spl.derivative()(temps)
                        sigma_M[:, j] = np.nanstd(y - y_fit)
                    except Exception:
                        dM_dT[:, j] = np.gradient(y, temps)
                        sigma_M[:, j] = np.nanstd(y - np.nanmedian(y))
                else:
                    dM_dT[:, j] = np.gradient(y, temps)
                    sigma_M[:, j] = np.nanstd(y - np.nanmedian(y))

        # Integrate over H using cumulative trapezoid
        dS = np.zeros_like(M_grid)
        for i in range(len(temps)):
            for j in range(1, len(H_grid)):
                dH = (H_grid[j] - H_grid[j - 1])
                dS[i, j] = dS[i, j - 1] + 0.5 * (dM_dT[i, j] + dM_dT[i, j - 1]) * dH

        # Optional bootstrap uncertainty in T (quick estimate)
        sigma_dS = None
        if bootstrap_sigma and bootstrap_sigma > 0:
            rng = np.random.default_rng(0)
            dS_boot = np.zeros((int(bootstrap_sigma), len(temps), len(H_grid)), dtype=float)
            for b in range(int(bootstrap_sigma)):
                idx = rng.choice(len(temps), size=len(temps), replace=True)
                t_bs = temps[idx]
                order = np.argsort(t_bs)
                t_bs = t_bs[order]
                M_bs = M_grid[idx, :][order, :]
                dM = np.gradient(M_bs, t_bs, axis=0)
                dS_b = np.zeros_like(M_bs)
                for i in range(len(t_bs)):
                    for j in range(1, len(H_grid)):
                        dH = (H_grid[j] - H_grid[j - 1])
                        dS_b[i, j] = dS_b[i, j - 1] + 0.5 * (dM[i, j] + dM[i, j - 1]) * dH
                # interpolate back to original temps (nearest is ok for uncertainty)
                for iT, T in enumerate(temps):
                    k = int(np.argmin(np.abs(t_bs - T)))
                    dS_boot[b, iT, :] = dS_b[k, :]
            sigma_dS = np.nanstd(dS_boot, axis=0)

        self.mce_data = {'T': temps.tolist(), 'H': H_grid, 'dS': dS, 'sigma': sigma_dS}

        # ---- Exponent n from ΔS(Tc,H) (Tc-interpolated) + ΔS_max(H) (Tc-free) ----
        tc_use = self.get_effective_tc(prefer_kf=True)
        self.results['Tc_for_mce'] = float(tc_use) if np.isfinite(tc_use) else None

        n_exp_tc = None
        if np.isfinite(tc_use) and (np.nanmin(temps) <= tc_use <= np.nanmax(temps)):
            # interpolate ΔS at Tc for each H
            ds_at_tc = np.zeros(len(H_grid), dtype=float)
            for j in range(len(H_grid)):
                ds_at_tc[j] = float(np.interp(tc_use, temps, np.abs(dS[:, j])))
            m = (H_grid > 0.05 * Hmax) & np.isfinite(ds_at_tc) & (ds_at_tc > 1e-12)
            if np.sum(m) > 10:
                lr = _safe_linregress(np.log(H_grid[m]), np.log(ds_at_tc[m]), min_pts=10)
                if lr is not None:
                    n_exp_tc = float(lr.slope)

        # peak-based exponent (more robust to small Tc offset)
        n_exp_peak = None
        ds_peak = np.nanmax(np.abs(dS), axis=0)
        m2 = (H_grid > 0.05 * Hmax) & np.isfinite(ds_peak) & (ds_peak > 1e-12)
        if np.sum(m2) > 10:
            lr2 = _safe_linregress(np.log(H_grid[m2]), np.log(ds_peak[m2]), min_pts=10)
            if lr2 is not None:
                n_exp_peak = float(lr2.slope)

        # Theory n at Tc: n = 1 + (β-1)/(β+γ)
        b = float(self.results.get('beta', np.nan))
        g = float(self.results.get('gamma', np.nan))
        delta = 1.0 + g / b if (np.isfinite(b) and b != 0) else np.nan
        n_theo = 1.0 + (1.0 / delta) * (1.0 - 1.0 / b) if np.isfinite(delta) and np.isfinite(b) else np.nan

        # Choose primary experimental n
        n_exp = n_exp_tc if n_exp_tc is not None else n_exp_peak

        self.results['n_exp_tc'] = n_exp_tc
        self.results['n_exp_peak'] = n_exp_peak
        self.results['n_exp'] = n_exp
        self.results['delta'] = delta
        self.results['n_theo'] = n_theo

        if n_exp is None or (isinstance(n_exp, float) and not np.isfinite(n_exp)) or (not np.isfinite(n_theo)):
            self.results['n_status'] = 'N/A'
        else:
            diff = abs(float(n_exp) - float(n_theo))
            if diff < 0.03:
                self.results['n_status'] = 'Excellent (Diff < 0.03)'
            elif diff < 0.10:
                self.results['n_status'] = 'Acceptable (Diff < 0.10)'
            else:
                self.results['n_status'] = 'Poor Consistency (Diff > 0.10)'

        return self.mce_data


    # ---------- 8) Joint/global fit: magnetization collapse + MCE collapse ----------

    def optimize_joint_scaling(self, *,
                              init: Optional[Tuple[float, float, float]] = None,
                              w_mce: float = 0.6,
                              tc_fixed: Optional[float] = None) -> Dict:
        """Joint optimization on (beta, gamma, Tc) using:
        - magnetization collapse cost
        - MCE collapse cost (if mce_data exists)

        Enhancement:
        - supports tc_fixed to enforce Tc self-consistency (e.g. lock Tc to KF).
        - clips parameters inside objective (Nelder–Mead is unbounded).
        """
        if not self.data_files:
            raise RuntimeError("No data loaded")

        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        tmin = float(np.nanmin(temps) + 1e-6)
        tmax = float(np.nanmax(temps) - 1e-6)

        bmin, bmax = 0.05, 1.2
        gmin, gmax = 0.2, 3.5

        def _clip_tc(tc: float) -> float:
            return float(np.clip(float(tc), tmin, tmax))

        def _clip_bg(b: float, g: float) -> Tuple[float, float]:
            return float(np.clip(float(b), bmin, bmax)), float(np.clip(float(g), gmin, gmax))

        tc0 = float(np.nanmean(temps))
        b0 = float(self.results.get('beta', 0.365))
        g0 = float(self.results.get('gamma', 1.386))
        if init is not None:
            b0, g0, tc0 = map(float, init)

        if tc_fixed is not None and isinstance(tc_fixed, (int, float)) and np.isfinite(tc_fixed):
            tc0 = float(tc_fixed)

        def mce_cost(beta, gamma, tc):
            if self.mce_data is None:
                return 0.0
            T_arr = np.array(self.mce_data['T'], dtype=float)
            H_arr = np.asarray(self.mce_data['H'], dtype=float)
            dS_arr = np.asarray(self.mce_data['dS'], dtype=float)
            if not (np.nanmin(T_arr) < tc < np.nanmax(T_arr)):
                return 1e99
            delta = 1.0 + gamma / beta
            n_exp = 1.0 + (1.0 / delta) * (1.0 - 1.0 / beta)
            X, Y = [], []
            for i in range(len(T_arr)):
                T = T_arr[i]
                eps = (T - tc) / tc
                # do not over-prune: many experiments only span ~±2% around Tc
                if abs(T - tc) < 0.3:
                    continue
                ds = np.abs(dS_arr[i, :])
                m = (H_arr > 0) & np.isfinite(ds) & (ds > 1e-12)
                if np.sum(m) < 10:
                    continue
                h = H_arr[m]
                x = eps / (h ** (1.0 / (beta + gamma)))
                y = ds[m] / (h ** n_exp)
                X.append(x)
                Y.append(y)
            if not X:
                return 1e99
            X = np.concatenate(X)
            Y = np.concatenate(Y)
            m = np.isfinite(X) & np.isfinite(Y)
            X = X[m]; Y = Y[m]
            if X.size < 200:
                return 1e99
            xmin, xmax = np.nanpercentile(X, 2), np.nanpercentile(X, 98)
            bins = np.linspace(xmin, xmax, 80)
            bid = np.digitize(X, bins) - 1
            cs = []
            for bi in range(len(bins) - 1):
                sel = bid == bi
                if np.sum(sel) < 8:
                    continue
                yb = Y[sel]
                med = np.nanmedian(yb)
                scat = _mad(yb) + 1e-12
                cs.append(np.nanmean(((yb - med) / scat) ** 2))
            if not cs:
                return 1e99
            return float(np.nanmean(cs))

        # Optimize either (b,g) or (b,g,Tc)
        if tc_fixed is not None and isinstance(tc_fixed, (int, float)) and np.isfinite(tc_fixed):
            tc_fix = _clip_tc(float(tc_fixed))
            x0 = np.array([b0, g0], dtype=float)

            def obj_bg(p):
                b, g = _clip_bg(float(p[0]), float(p[1]))
                c1 = self._collapse_cost(b, g, tc_fix)
                c2 = mce_cost(b, g, tc_fix)
                if not np.isfinite(c1) or not np.isfinite(c2):
                    return 1e99
                return float(c1 + w_mce * c2)

            res = minimize(obj_bg, x0, method='Nelder-Mead', options={'maxiter': 420, 'xatol': 1e-4, 'fatol': 1e-4})
            b, g = _clip_bg(float(res.x[0]), float(res.x[1]))
            tc = tc_fix
            fun = float(obj_bg([b, g]))
        else:
            x0 = np.array([b0, g0, tc0], dtype=float)

            def obj(p):
                b, g = _clip_bg(float(p[0]), float(p[1]))
                tc = _clip_tc(float(p[2]))
                c1 = self._collapse_cost(b, g, tc)
                c2 = mce_cost(b, g, tc)
                if not np.isfinite(c1) or not np.isfinite(c2):
                    return 1e99
                return float(c1 + w_mce * c2)

            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 420, 'xatol': 1e-4, 'fatol': 1e-4})
            b, g = _clip_bg(float(res.x[0]), float(res.x[1]))
            tc = _clip_tc(float(res.x[2]))
            fun = float(obj([b, g, tc]))

        self.results['beta'] = float(b)
        self.results['gamma'] = float(g)
        self.results['Tc'] = float(tc)
        self.results['delta'] = 1.0 + float(g) / float(b)
        self.results['joint_cost'] = float(fun)
        return {'success': bool(res.success), 'beta': float(b), 'gamma': float(g), 'Tc': float(tc), 'cost': float(fun)}

    # ---------- 3) Uncertainty: bootstrap + MCMC ----------

    def bootstrap_analysis(self, n_iter: int = 80, *,
                           method: str = 'collapse',
                           callback: Optional[Callable[[int], None]] = None,
                           tc_fixed: Optional[float] = None,
                           fixed_params: bool = False) -> Dict:
        """Bootstrap resampling within each isotherm (WITHOUT changing final fit).

        Bugfix (v19):
        - The previous implementation called optimize_data_collapse() with update_results=True
          inside the bootstrap loop, which overwrote self.results['beta','gamma','Tc'] and
          caused *post-bootstrap* plots/summary to disagree with the KF intercept.
        - We now:
            (1) keep a copy of the final fitted parameters,
            (2) run bootstrap fits with update_results=False,
            (3) restore the final parameters at the end.

        tc_fixed:
            If provided (or if Tc_source=='KF'), each bootstrap replicate estimates Tc via KF
            (or uses tc_fixed), then optimizes only beta/gamma with Tc fixed.
            This matches the "Lock Tc to KF" workflow and avoids Tc drifting to a spurious value.
        fixed_params:
            If True, keep beta/gamma fixed during bootstrap (errors become ~0 for exponents).
        """
        if not self.data_files:
            return {}

        rng = np.random.default_rng(0)

        # ---- Preserve final fit ----
        b_keep = float(self.results.get('beta', np.nan))
        g_keep = float(self.results.get('gamma', np.nan))
        tc_keep = float(self.results.get('Tc', np.nan))
        tc_kf_keep = self.results.get('Tc_KF', None)
        tc_src_keep = self.results.get('Tc_source', None)
        delta_keep = self.results.get('delta', None)

        # backup raw data arrays
        orig = [{'H': d['H'].copy(), 'M': d['M'].copy()} for d in self.data_files]

        def estimate_tc_kf_local(beta: float, gamma: float) -> Optional[float]:
            """Lightweight KF Tc estimator (no mutation)."""
            map_res = []
            for d in self.data_files:
                H, M = self._get_filtered_data(d)
                if H.size < 6:
                    continue
                x = (H / M) ** (1.0 / gamma)
                y = (M) ** (1.0 / beta)
                r = _safe_linregress(x, y, min_pts=6)
                if r is None:
                    continue
                ms = r.intercept ** beta if r.intercept > 0 else np.nan
                chi_inv = (-r.intercept / r.slope) ** gamma if (r.intercept < 0 and r.slope != 0) else np.nan
                map_res.append({'T': float(d['T']), 'Ms': ms, 'ChiInv': chi_inv})
            if not map_res:
                return None
            temps = np.array([r['T'] for r in map_res], dtype=float)
            Ms = np.array([r['Ms'] for r in map_res], dtype=float)
            ChiInv = np.array([r['ChiInv'] for r in map_res], dtype=float)

            tc_ms_val = None
            m = np.isfinite(Ms)
            if np.sum(m) > 5:
                t = temps[m]; yv = Ms[m]
                order = np.argsort(t); t = t[order]; yv = yv[order]
                dy = np.gradient(yv, t)
                kf = np.where(np.abs(dy) > 1e-12, yv / dy, np.nan)
                mm = np.isfinite(kf)
                if np.sum(mm) > 3:
                    lr = _safe_linregress(t[mm], kf[mm], min_pts=4)
                    if (lr is not None) and (lr.slope != 0):
                        tc_ms_val = -lr.intercept / lr.slope

            tc_chi_val = None
            m = np.isfinite(ChiInv)
            if np.sum(m) > 5:
                t = temps[m]; yv = ChiInv[m]
                order = np.argsort(t); t = t[order]; yv = yv[order]
                dy = np.gradient(yv, t)
                kf = np.where(np.abs(dy) > 1e-12, yv / dy, np.nan)
                mm = np.isfinite(kf)
                if np.sum(mm) > 3:
                    lr = _safe_linregress(t[mm], kf[mm], min_pts=4)
                    if (lr is not None) and (lr.slope != 0):
                        tc_chi_val = -lr.intercept / lr.slope

            Tc_KF = None
            if tc_ms_val is not None and tc_chi_val is not None:
                Tc_KF = 0.5 * (tc_ms_val + tc_chi_val)
            elif tc_ms_val is not None:
                Tc_KF = tc_ms_val
            elif tc_chi_val is not None:
                Tc_KF = tc_chi_val

            if Tc_KF is None:
                return None

            # sanity: must lie within measured temperature window
            tmin, tmax = float(np.nanmin(temps)), float(np.nanmax(temps))
            if not (tmin < float(Tc_KF) < tmax):
                return None
            return float(Tc_KF)

        # Determine bootstrap mode (KF-locked or free)
        use_kf_lock = False
        if tc_fixed is not None and isinstance(tc_fixed, (int, float)) and np.isfinite(tc_fixed):
            use_kf_lock = True
        elif self.results.get('Tc_source', None) == 'KF':
            use_kf_lock = True
        elif isinstance(self.results.get('Tc_KF', None), (int, float)) and np.isfinite(self.results.get('Tc_KF')):
            use_kf_lock = True

        samples = []
        tc_kf_samples = []

        for i in range(int(n_iter)):
            if callback:
                callback(int((i / max(1, n_iter)) * 100))

            # resample each curve
            for k, d in enumerate(self.data_files):
                n = len(d['H'])
                if n <= 3:
                    continue
                idx = rng.choice(n, size=n, replace=True)
                d['H'] = orig[k]['H'][idx]
                d['M'] = orig[k]['M'][idx]

            try:
                if use_kf_lock:
                    # Tc from KF per bootstrap replicate (unless user provides tc_fixed)
                    if tc_fixed is not None and isinstance(tc_fixed, (int, float)) and np.isfinite(tc_fixed):
                        tc_i = float(tc_fixed)
                    else:
                        tc_i = estimate_tc_kf_local(b_keep, g_keep)
                        if tc_i is None:
                            tc_i = tc_keep if np.isfinite(tc_keep) else None
                    if tc_i is None or (not np.isfinite(tc_i)):
                        continue
                    tc_kf_samples.append(float(tc_i))

                    if fixed_params:
                        samples.append([b_keep, g_keep, float(tc_i)])
                    else:
                        r = self.optimize_data_collapse(b_keep, g_keep, fixed_params=False,
                                                       init_tc=float(tc_i), tc_fixed=float(tc_i),
                                                       update_results=False)
                        samples.append([float(r['beta']), float(r['gamma']), float(tc_i)])
                else:
                    # fully free collapse fit (bootstrap Tc too)
                    r = self.optimize_data_collapse(b_keep, g_keep, fixed_params=False,
                                                   init_tc=tc_keep if np.isfinite(tc_keep) else None,
                                                   update_results=False)
                    samples.append([float(r['beta']), float(r['gamma']), float(r['Tc'])])
            except Exception:
                pass

        # restore raw data
        for k, d in enumerate(self.data_files):
            d['H'] = orig[k]['H']
            d['M'] = orig[k]['M']

        # restore final fit parameters
        if np.isfinite(b_keep): self.results['beta'] = float(b_keep)
        if np.isfinite(g_keep): self.results['gamma'] = float(g_keep)
        if np.isfinite(tc_keep): self.results['Tc'] = float(tc_keep)
        if delta_keep is not None: self.results['delta'] = delta_keep
        if tc_kf_keep is not None: self.results['Tc_KF'] = tc_kf_keep
        if tc_src_keep is not None: self.results['Tc_source'] = tc_src_keep

        if not samples:
            return {}

        S = np.array(samples, dtype=float)
        ci = np.quantile(S, [0.16, 0.5, 0.84], axis=0)

        errs = {
            'beta': float(np.nanstd(S[:, 0])),
            'gamma': float(np.nanstd(S[:, 1])),
            'Tc': float(np.nanstd(S[:, 2])),
        }

        # v22 FIX: if Tc is locked (tc_fixed), bootstrap std of Tc is artificially 0.
        # Keep a realistic Tc uncertainty from KF/jackknife floor if available.
        prev_errs = self.results.get('errors', {}) if isinstance(self.results.get('errors', {}), dict) else {}
        if 'Tc_Arrott' in prev_errs:
            try:
                ta = float(prev_errs.get('Tc_Arrott'))
                if np.isfinite(ta) and ta > 0:
                    errs['Tc_Arrott'] = ta
            except Exception:
                pass
        if tc_fixed is not None and isinstance(tc_fixed, (int, float)) and np.isfinite(tc_fixed):
            tc_floor = None
            try:
                tt = np.sort(np.asarray([d['T'] for d in self.data_files], float))
                dtt = np.diff(tt[np.isfinite(tt)])
                dt_med = float(np.nanmedian(dtt)) if dtt.size else np.nan
                tc_floor = float(0.25 * dt_med) if np.isfinite(dt_med) and dt_med > 0 else 0.5
            except Exception:
                tc_floor = 0.5
            keep_tc = prev_errs.get('Tc', None)
            if isinstance(keep_tc, (int, float)) and np.isfinite(keep_tc) and keep_tc > 0:
                errs['Tc'] = float(keep_tc)
            else:
                errs['Tc'] = float(max(float(errs.get('Tc', 0.0)), float(tc_floor)))

        # If using KF-locked mode and we have Tc samples, also report Tc_KF uncertainty        if tc_kf_samples:
            errs['Tc_KF'] = float(np.nanstd(np.array(tc_kf_samples, dtype=float)))
            # avoid 0.0 Tc_KF uncertainty when Tc is effectively locked or bootstrap is degenerate
            if (not np.isfinite(errs['Tc_KF'])) or errs['Tc_KF'] <= 0:
                keep_kf = prev_errs.get('Tc_KF', None)
                if isinstance(keep_kf, (int, float)) and np.isfinite(keep_kf) and keep_kf > 0:
                    errs['Tc_KF'] = float(keep_kf)
                else:
                    errs['Tc_KF'] = 0.0
            # enforce a temperature-resolution floor (same floor as Tc)
            try:
                errs['Tc_KF'] = float(max(float(errs.get('Tc_KF', 0.0)), float(tc_floor)))
            except Exception:
                errs['Tc_KF'] = float(max(float(errs.get('Tc_KF', 0.0)), 0.5))

        self.results.setdefault('errors', {}).update(errs)
        self.results['bootstrap_ci_16_50_84'] = {
            'beta': tuple(ci[:, 0]),
            'gamma': tuple(ci[:, 1]),
            'Tc': tuple(ci[:, 2]),
        }

        return {'errors_std': errs, 'ci_16_50_84': self.results['bootstrap_ci_16_50_84'], 'n': int(S.shape[0])}


    def _metropolis(self, logp: Callable[[np.ndarray], float], *,
                    x0: np.ndarray,
                    step: np.ndarray,
                    bounds: np.ndarray,
                    n: int = 1500,
                    burn: int = 300) -> Tuple[np.ndarray, float]:
        rng = np.random.default_rng(1)
        x = x0.copy()
        lp = float(logp(x))
        samples = []
        acc = 0
        for i in range(n):
            prop = x + rng.normal(scale=step, size=x.shape)
            # bounds
            ok = True
            for j in range(len(prop)):
                if not (bounds[j, 0] <= prop[j] <= bounds[j, 1]):
                    ok = False
                    break
            if not ok:
                if i >= burn:
                    samples.append(x.copy())
                continue
            lp2 = float(logp(prop))
            if np.isfinite(lp2) and (math.log(rng.random()) < (lp2 - lp)):
                x = prop
                lp = lp2
                acc += 1
            if i >= burn:
                samples.append(x.copy())
        samples = np.asarray(samples, dtype=float)
        acc_rate = acc / max(1, n)
        return samples, float(acc_rate)

    def mcmc_uncertainty(self, *, target: str = 'collapse', n: int = 2000, burn: int = 400) -> Dict:
        """Simple Metropolis sampler to quantify uncertainty.

        target:
            - 'collapse': logp ~ -collapse_cost
            - 'gp': logp ~ GP log marginal likelihood
        """
        if not self.data_files:
            return {}
        temps = np.array([d['T'] for d in self.data_files], dtype=float)
        b0, g0, tc0 = float(self.results['beta']), float(self.results['gamma']), float(self.results['Tc'] or np.nanmean(temps))

        if target == 'gp':
            if not SKLEARN_AVAILABLE:
                return {}
            logp = lambda p: -self.gp_neg_log_marginal_likelihood(float(p[0]), float(p[1]), float(p[2]))
            step = np.array([0.008, 0.015, 0.15])
        else:
            logp = lambda p: -self._collapse_cost(float(p[0]), float(p[1]), float(p[2]))
            step = np.array([0.01, 0.02, 0.2])

        bounds = np.array([[0.05, 1.2], [0.2, 3.5], [np.nanmin(temps), np.nanmax(temps)]], dtype=float)
        samples, acc = self._metropolis(logp, x0=np.array([b0, g0, tc0]), step=step, bounds=bounds, n=int(n), burn=int(burn))
        if samples.size == 0:
            return {}
        ci = np.quantile(samples, [0.16, 0.5, 0.84], axis=0)
        out = {'accept': acc, 'ci_16_50_84': {'beta': tuple(ci[:, 0]), 'gamma': tuple(ci[:, 1]), 'Tc': tuple(ci[:, 2])},
               'mean': {'beta': float(np.nanmean(samples[:, 0])), 'gamma': float(np.nanmean(samples[:, 1])), 'Tc': float(np.nanmean(samples[:, 2]))},
               'n': int(samples.shape[0])}
        self.results[f'{target}_mcmc'] = out
        return out


# ==========================================
# Compatibility alias: external scripts may import MagneticCriticalEngine
MagneticCriticalEngine = PhysicsEngine

# GUI Frontend
# ==========================================

class AnalysisThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)
    log = pyqtSignal(str)

    def __init__(self, engine: PhysicsEngine, config: Dict):
        super().__init__()
        self.engine = engine
        self.config = config

    def run(self):
        try:
            self.progress.emit(1)
            self.log.emit("=== MagneticAnalyzer v24: self-consistent Tc (KF-locked) ===")

            # ---- config ----
            self.engine.hc_value = float(self.config.get('Bc', 0.0))
            init_beta = float(self.config.get('init_beta', 0.365))
            init_gamma = float(self.config.get('init_gamma', 1.386))
            fixed_params = bool(self.config.get('fixed_params', False))
            lock_tc_to_kf = bool(self.config.get('lock_tc_to_kf', True))
            kf_max_iter = int(self.config.get('kf_max_iter', 3))
            kf_tol = float(self.config.get('kf_tol', 0.1))

            # IMPORTANT: clear stale results from previous runs
            self.engine.results = {}

            # seed results
            temps = np.array([d['T'] for d in self.engine.data_files], dtype=float)
            tc_seed = float(np.nanmean(temps)) if temps.size else np.nan
            self.engine.results['beta'] = init_beta
            self.engine.results['gamma'] = init_gamma
            self.engine.results['Tc'] = tc_seed
            self.engine.results['Tc_source'] = 'seed'

            self.progress.emit(5)

            # ---- Step 1: KF + (optional) Tc-locked collapse refinement ----
            if lock_tc_to_kf:
                self.log.emit(">>> Step 1: Iterative Modified Arrott + Kouvel-Fisher (Tc from KF, no free-Tc collapse)")
                if not fixed_params:
                    # RS-based initial universality guess (more stable than a free collapse search)
                    try:
                        rs_init = self.engine.select_initial_universality_by_rs()
                        if rs_init is not None and isinstance(rs_init, dict):
                            self.log.emit(f"    RS initial guess: {rs_init.get('name')} (score={rs_init.get('score', np.nan):.4g})")
                            b0 = float(rs_init.get('beta', init_beta))
                            g0 = float(rs_init.get('gamma', init_gamma))
                            if np.isfinite(b0): self.engine.results['beta'] = float(np.clip(b0, 0.10, 0.80))
                            if np.isfinite(g0): self.engine.results['gamma'] = float(np.clip(g0, 0.80, 2.50))
                    except Exception:
                        pass

                tc_prev = None
                tc_lock = None
                for it in range(max(kf_max_iter, 1)):
                    self.log.emit(f"    MAP/KF iteration {it+1}/{max(kf_max_iter,1)} ...")
                    if fixed_params:
                        # Manual mode: keep exponents locked to GUI inputs
                        self.engine.results['beta'] = init_beta
                        self.engine.results['gamma'] = init_gamma
                    self.engine.perform_kf_and_correction(use_correction=False,
                                                         omega=self.config.get('omega', 0.5),
                                                         fit_omega=self.config.get('fit_omega', True),
                                                         update_results=True)

                    if fixed_params:

                        # Manual mode: do NOT iterate exponents; keep GUI values.

                        self.engine.results['beta'] = init_beta

                        self.engine.results['gamma'] = init_gamma

                    else:

                        # Update (beta,gamma) using KF slopes when available (damped to avoid oscillations)

                                            b_kf = self.engine.results.get('beta_KF', None)

                                            g_kf = self.engine.results.get('gamma_KF', None)

                                            try:

                                                b_prev = float(self.engine.results.get('beta', init_beta))

                                                g_prev = float(self.engine.results.get('gamma', init_gamma))

                                            except Exception:

                                                b_prev, g_prev = init_beta, init_gamma

                        

                                            if isinstance(b_kf, (int, float)) and np.isfinite(b_kf):

                                                b_new = 0.80 * float(b_kf) + 0.20 * float(b_prev)

                                                self.engine.results['beta'] = float(np.clip(b_new, 0.10, 0.80))

                                            if isinstance(g_kf, (int, float)) and np.isfinite(g_kf):

                                                g_new = 0.80 * float(g_kf) + 0.20 * float(g_prev)

                                                self.engine.results['gamma'] = float(np.clip(g_new, 0.80, 2.50))

                    tc_kf = self.engine.results.get('Tc_KF', None)
                    if tc_kf is None or (isinstance(tc_kf, float) and not np.isfinite(tc_kf)):
                        self.log.emit("    KF Tc not available -> fallback to free Tc collapse.")
                        tc_lock = None
                        break
                    tc_lock = float(tc_kf)

                    # lock Tc to KF for all downstream analyses
                    self.engine.results['Tc'] = tc_lock

                    # stop if converged
                    if tc_prev is not None and abs(tc_lock - tc_prev) < kf_tol:
                        break
                    tc_prev = tc_lock
                if tc_lock is not None and np.isfinite(tc_lock):
                    self.engine.results['Tc'] = float(tc_lock)
                    self.engine.results['Tc_source'] = 'KF'

                    # sanity check: KF can become unstable for small/noisy datasets
                    try:
                        b_kf_chk = float(self.engine.results.get('beta_KF', np.nan))
                        g_kf_chk = float(self.engine.results.get('gamma_KF', np.nan))
                        fits = self.engine.results.get('kf_fits', {})
                        tcse_ms = float(fits.get('Ms', {}).get('Tc_se', np.nan)) if isinstance(fits, dict) else np.nan
                        tcse_ch = float(fits.get('ChiInv', {}).get('Tc_se', np.nan)) if isinstance(fits, dict) else np.nan
                        tcse = float(np.nanmax([tcse_ms, tcse_ch]))
                        if ((not np.isfinite(b_kf_chk)) or (not np.isfinite(g_kf_chk)) or
                            (b_kf_chk < 0.25) or (b_kf_chk > 0.60) or
                            (g_kf_chk < 0.80) or (g_kf_chk > 2.20) or
                            (not np.isfinite(tcse)) or (tcse > 1.0)):
                            self.log.emit("    KF fit looks unreliable -> switching to discrete Tc collapse.")
                            self.engine.results['Tc_source'] = 'collapse'
                    except Exception:
                        self.engine.results['Tc_source'] = 'collapse'
                else:
                    self.engine.results['Tc_source'] = 'collapse'

            # If not locking (or KF failed), do standard free-Tc collapse
            if (not lock_tc_to_kf) or (self.engine.results.get('Tc_source') != 'KF'):
                self.log.emit(">>> Step 1: Discrete Tc scaling-collapse optimization (Tc chosen among measured isotherms)")
                self.engine.optimize_tc_discrete_by_collapse(init_beta, init_gamma, fixed_params=fixed_params, update_results=True)
                # cache collapse estimate explicitly
                try:
                    self.engine.results['Tc_collapse'] = float(self.engine.results.get('Tc', np.nan))
                    self.engine.results.setdefault('errors', {}).setdefault('Tc_collapse', float(self.engine.results.get('errors', {}).get('Tc', np.nan)))
                except Exception:
                    pass
                self.engine.results['Tc_source'] = 'collapse_discrete'

            # v22: refresh Modified Arrott Tc and KF diagnostics using the final (beta,gamma)
            try:
                self.engine.perform_kf_and_correction(use_correction=False,
                                                     omega=self.config.get('omega', 0.5),
                                                     fit_omega=self.config.get('fit_omega', True),
                                                     update_results=True)
            except Exception:
                pass

            # If KF is unavailable but MAP intercept crossing provides Tc, use it as a secondary Tc source
            # (Do NOT override the discrete-collapse Tc, which is already constrained to measured isotherms.)
            if self.engine.results.get('Tc_source') not in ('KF', 'collapse_discrete'):
                tc_a = self.engine.results.get('Tc_Arrott', None)
                if isinstance(tc_a, (int, float)) and np.isfinite(tc_a):
                    self.engine.results['Tc'] = float(tc_a)
                    self.engine.results['Tc_source'] = 'Arrott'
            self.progress.emit(25)

            # ---- Step 2: Optional correction-to-scaling (may refine beta/gamma) ----
            if self.config.get('use_correction', False):
                self.log.emit(">>> Step 2: Correction-to-scaling fit (omega)")
                self.engine.perform_kf_and_correction(use_correction=True,
                                                     override_exponents=(not fixed_params),
                                                     omega=self.config.get('omega', 0.5),
                                                     fit_omega=self.config.get('fit_omega', True),
                                                     update_results=True)
                if fixed_params:
                    # Manual mode: keep exponents locked even after correction fit
                    self.engine.results['beta'] = init_beta
                    self.engine.results['gamma'] = init_gamma
                # After correction, if Tc is KF-locked, enforce again
                if lock_tc_to_kf:
                    tc_kf = self.engine.results.get('Tc_KF', None)
                    if tc_kf is not None and isinstance(tc_kf, (int, float)) and np.isfinite(tc_kf):
                        self.engine.results['Tc'] = float(tc_kf)
                        self.engine.results['Tc_source'] = 'KF'
                        self.engine.optimize_data_collapse(float(self.engine.results['beta']),
                                                           float(self.engine.results['gamma']),
                                                           fixed_params=fixed_params,
                                                           init_tc=float(tc_kf),
                                                           tc_fixed=float(tc_kf),
                                                           update_results=True)


            # ---- v24.1 patch: always compute a diagnostic Tc from discrete collapse ----
            # Even when Tc is KF-locked, we still report "Tc (collapse)" in the dashboard/PDF
            # as an independent cross-check based on scaling-collapse optimization.
            try:
                tc_coll_existing = self.engine.results.get('Tc_collapse', None)
                if not (isinstance(tc_coll_existing, (int, float)) and np.isfinite(tc_coll_existing)):
                    diag = self.engine.optimize_tc_discrete_by_collapse(
                        float(self.engine.results.get('beta', init_beta)),
                        float(self.engine.results.get('gamma', init_gamma)),
                        fixed_params=fixed_params,
                        update_results=False
                    )
                    if isinstance(diag, dict):
                        tc_coll_val = diag.get('Tc', np.nan)
                        if isinstance(tc_coll_val, (int, float)) and np.isfinite(tc_coll_val):
                            self.engine.results['Tc_collapse'] = float(tc_coll_val)
                            # If bootstrap hasn't provided Tc_collapse uncertainty, estimate from temperature spacing.
                            tt = np.array([d.get('T', np.nan) for d in self.engine.data_files], dtype=float)
                            tt = np.unique(tt[np.isfinite(tt)])
                            tt.sort()
                            dt = np.diff(tt)
                            dt_med = float(np.nanmedian(dt)) if dt.size else np.nan
                            if np.isfinite(dt_med) and dt_med > 0:
                                self.engine.results.setdefault('errors', {}).setdefault('Tc_collapse', float(0.5 * dt_med))
            except Exception:
                pass

            self.progress.emit(40)

            # ---- Step 3: EOS fits (optional) ----
            if self.config.get('use_eos', True):
                self.log.emit(">>> Step 3: Universal EOS fits...")
                tc0 = float(self.engine.results.get('Tc', tc_seed))
                for name in self.engine.eos_models:
                    try:
                        self.engine.fit_eos_parametric(model_name=name, init_tc=tc0, fit_scales=True, max_points=2500)
                    except Exception:
                        pass

            self.progress.emit(55)

            # ---- Step 4: MCE analysis ----
            self.log.emit(">>> Step 4: MCE analysis...")
            self.engine.mce_analysis(advanced_smooth=self.config.get('use_mce_adv', True),
                                     deriv_method=self.config.get('mce_method', 'spline'))

            self.progress.emit(70)

            # ---- Step 5: Joint fit (optional) ----
            if (not fixed_params) and self.config.get('joint_fit', False) and self.engine.mce_data is not None:
                self.log.emit(">>> Step 5: Joint fit (M + MCE)...")
                tc_fix = float(self.engine.results.get('Tc', tc_seed)) if lock_tc_to_kf else None
                self.engine.optimize_joint_scaling(init=(float(self.engine.results['beta']),
                                                        float(self.engine.results['gamma']),
                                                        float(self.engine.results.get('Tc', tc_seed))),
                                                   w_mce=0.6,
                                                   tc_fixed=tc_fix)

            self.progress.emit(78)

            # ---- Step 6: GP scaling (optional) ----
            if (not fixed_params) and self.config.get('use_gp', False) and SKLEARN_AVAILABLE:
                self.log.emit(">>> Step 6: GP/Bayesian scaling...")
                self.engine.perform_gp_scaling(mcmc=self.config.get('gp_mcmc', True))

            self.progress.emit(86)

            # ---- Step 7: Universality scoring (does NOT overwrite beta/gamma/Tc) ----
            self.log.emit(">>> Step 7: Universality-class comparison...")
            self.engine.compare_universality_classes(include_eos=self.config.get('use_eos', True))

            self.progress.emit(90)

            # ---- Step 8: Uncertainties ----
            # Quick uncertainty (bootstrap-lite) even when full bootstrap is off
            if self.config.get('quick_uncertainty', True) and (not self.config.get('bootstrap', False)):
                self.log.emit(">>> Step 8a: Quick uncertainty (bootstrap-lite)...")
                self.engine.bootstrap_analysis(n_iter=self.config.get('quick_bootstrap_n', 40), callback=self.progress.emit,
                                              tc_fixed=(float(self.engine.results.get('Tc', tc_seed)) if lock_tc_to_kf else None),
                                              fixed_params=fixed_params)

            if self.config.get('bootstrap', False):
                self.log.emit(">>> Step 8b: Bootstrap uncertainty...")
                self.engine.bootstrap_analysis(n_iter=self.config.get('bootstrap_n', 80), callback=self.progress.emit,
                                              tc_fixed=(float(self.engine.results.get('Tc', tc_seed)) if lock_tc_to_kf else None),
                                              fixed_params=fixed_params)

            self.progress.emit(96)
            if (not fixed_params) and self.config.get('mcmc', False):
                self.log.emit(">>> Step 8b: MCMC uncertainty...")
                self.engine.mcmc_uncertainty(target=self.config.get('mcmc_target', 'collapse'),
                                             n=self.config.get('mcmc_n', 2000))

            self.progress.emit(100)
            self.finished.emit(True)
        except Exception as e:
            self.log.emit(f"Error: {e}")
            traceback.print_exc()
            self.finished.emit(False)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = PhysicsEngine()
        self.thread: Optional[AnalysisThread] = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Magnetic Analyzer v19 (KF-locked Tc)")
        self.resize(1420, 980)

        main_widget = QWidget(); self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(360)

        self.btn_load = QPushButton("Load Files")
        self.btn_load.clicked.connect(self.load_data)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.installEventFilter(self)

        # File list actions (remove / clear) + right-click menu
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self.remove_selected_files)
        self.btn_clear_files = QPushButton("Clear List")
        self.btn_clear_files.clicked.connect(self.clear_file_list)

        # Context menu (right click) using built-in ActionsContextMenu
        self.file_list.setContextMenuPolicy(Qt.ActionsContextMenu)
        act_remove = QAction("Remove selected", self)
        act_remove.triggered.connect(self.remove_selected_files)
        self.file_list.addAction(act_remove)
        act_clear = QAction("Clear list", self)
        act_clear.triggered.connect(self.clear_file_list)
        self.file_list.addAction(act_clear)

        left_layout.addWidget(self.btn_load)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear_files)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(self.file_list)

        gb_p = QGroupBox("Settings")
        form = QFormLayout()

        self.spin_bc = QDoubleSpinBox(); self.spin_bc.setRange(0, 1e6); self.spin_bc.setSingleStep(0.1)
        self.btn_auto = QPushButton("Auto Bc")
        self.btn_auto.clicked.connect(self.auto_bc)

        self.spin_beta = QDoubleSpinBox(); self.spin_beta.setRange(0.05, 2.0); self.spin_beta.setSingleStep(0.01); self.spin_beta.setValue(0.365)
        self.spin_gamma = QDoubleSpinBox(); self.spin_gamma.setRange(0.2, 4.0); self.spin_gamma.setSingleStep(0.01); self.spin_gamma.setValue(1.386)
        self.chk_manual = QCheckBox("Manual Mode (Fix beta/gamma)")

        form.addRow("Bc (same unit as H):", self.spin_bc)
        form.addRow("", self.btn_auto)
        form.addRow("Init beta:", self.spin_beta)
        form.addRow("Init gamma:", self.spin_gamma)
        form.addRow("", self.chk_manual)
        gb_p.setLayout(form)
        left_layout.addWidget(gb_p)

        gb_adv = QGroupBox("Advanced Features")
        vbox = QVBoxLayout()
        self.chk_corr = QCheckBox("Correction-to-scaling (omega fit)")
        self.chk_lock_tc = QCheckBox("Lock Tc to KF (recommended)")
        self.chk_lock_tc.setChecked(True)
        self.chk_boot = QCheckBox("Bootstrap errors")
        self.chk_mcmc = QCheckBox("MCMC errors")
        self.chk_gp = QCheckBox("GP/Bayesian scaling")
        self.chk_gp.setChecked(False)
        if not SKLEARN_AVAILABLE:
            self.chk_gp.setEnabled(False)
        self.chk_eos = QCheckBox("Universal EOS fits (Ising/Heisenberg/MF)")
        self.chk_eos.setChecked(True)
        self.chk_mce_adv = QCheckBox("Advanced MCE smoothing")
        self.chk_mce_adv.setChecked(True)
        self.chk_joint = QCheckBox("Joint fit: M + MCE (self-consistent)")
        self.chk_joint.setChecked(False)

        vbox.addWidget(self.chk_corr)
        vbox.addWidget(self.chk_lock_tc)
        vbox.addWidget(self.chk_gp)
        vbox.addWidget(self.chk_eos)
        vbox.addWidget(self.chk_mce_adv)
        vbox.addWidget(self.chk_joint)
        vbox.addWidget(self.chk_boot)
        vbox.addWidget(self.chk_mcmc)
        gb_adv.setLayout(vbox)
        left_layout.addWidget(gb_adv)

        self.btn_run = QPushButton("RUN ANALYSIS")
        self.btn_run.setStyleSheet("background:#4CAF50;color:white;padding:10px;font-weight:bold")
        self.btn_run.clicked.connect(self.start_analysis)
        self.prog = QProgressBar()
        left_layout.addWidget(self.btn_run)
        left_layout.addWidget(self.prog)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box)

        self.btn_pdf = QPushButton("Export PDF")
        self.btn_pdf.clicked.connect(self.export_pdf)
        left_layout.addWidget(self.btn_pdf)

        # Right: plots + dashboard
        right_splitter = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget(); self.canvases = {}
        for name in ["Arrott Plot", "Scaling Collapse", "Kouvel-Fisher", "EOS Fit", "GP Scaling", "MCE", "MCE Universal", "Scores"]:
            c = MplCanvas(self)
            self.tabs.addTab(c, name)
            self.canvases[name] = c
        right_splitter.addWidget(self.tabs)

        results_group = QGroupBox("Final Results Dashboard")
        res_layout = QVBoxLayout()
        self.results_display = QTextEdit(); self.results_display.setReadOnly(True)
        self.results_display.setStyleSheet("font-family: monospace; font-size: 11pt; background-color: #f6f6f6;")
        res_layout.addWidget(self.results_display)
        results_group.setLayout(res_layout)
        right_splitter.addWidget(results_group)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        layout.addWidget(left_panel)
        layout.addWidget(right_splitter)

    def log(self, t: str):
        self.log_box.append(t)

    def load_data(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open", "", "Data (*.dat *.txt *.csv)")
        if not files:
            return
        n = self.engine.load_files(files)
        self.refresh_file_list()
        self.results_display.clear()
        self.log(f"Loaded {n} isotherms.")

    def refresh_file_list(self):
        """Refresh the left file list from engine.data_files."""
        self.file_list.clear()
        for f in self.engine.data_files:
            self.file_list.addItem(f"{f['file']} ({f['T']} K)")

    def remove_selected_files(self):
        """Remove selected isotherms from the loaded list (no need to restart)."""
        items = self.file_list.selectedItems()
        if not items:
            return
        rows = sorted({self.file_list.row(it) for it in items}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.engine.data_files):
                del self.engine.data_files[r]
        self.refresh_file_list()
        # Clear stale results
        try:
            self.engine.results.clear()
        except Exception:
            pass
        self.results_display.clear()
        self.log(f"Removed {len(rows)} item(s).")

    def clear_file_list(self):
        """Clear all loaded isotherms."""
        self.engine.data_files.clear()
        self.refresh_file_list()
        try:
            self.engine.results.clear()
        except Exception:
            pass
        self.results_display.clear()
        self.log("Cleared file list.")

    def eventFilter(self, obj, event):
        """Enable Delete/Backspace to remove selected file(s) in the list."""
        if GUI_AVAILABLE and obj is self.file_list and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self.remove_selected_files()
                return True
        return super().eventFilter(obj, event)

    def auto_bc(self):
        bc = self.engine.auto_suggest_bc()
        self.spin_bc.setValue(bc)
        self.log(f"Auto Bc: {bc:.4g}")

    def start_analysis(self):
        if not self.engine.data_files:
            self.log("No data loaded.")
            return
        cfg = {
            'Bc': self.spin_bc.value(),
            'init_beta': self.spin_beta.value(),
            'init_gamma': self.spin_gamma.value(),
            'fixed_params': self.chk_manual.isChecked(),
            'use_correction': self.chk_corr.isChecked(),
            'lock_tc_to_kf': self.chk_lock_tc.isChecked(),
            'kf_max_iter': 3,
            'kf_tol': 0.1,
            'bootstrap': self.chk_boot.isChecked(),
            'use_gp': self.chk_gp.isChecked(),
            'use_eos': self.chk_eos.isChecked(),
            'use_mce_adv': self.chk_mce_adv.isChecked(),
            'joint_fit': self.chk_joint.isChecked(),
            # advanced knobs (kept simple)
            'mce_method': 'spline',
            'gp_mcmc': True,
            'bootstrap_n': 80,
            'mcmc': self.chk_mcmc.isChecked(),
            'mcmc_target': 'collapse',
            'mcmc_n': 2000,
            'omega': 0.5,
            'fit_omega': True,
        }

        self.thread = AnalysisThread(self.engine, cfg)
        self.thread.log.connect(self.log)
        self.thread.progress.connect(self.prog.setValue)
        self.thread.finished.connect(self.on_finished)
        self.btn_run.setEnabled(False)
        self.prog.setValue(0)
        self.log("==============================")
        self.thread.start()

    def on_finished(self, ok: bool):
        self.btn_run.setEnabled(True)
        self.prog.setValue(100)
        if ok:
            self.log("Done.")
            self.update_plots()
            self.print_summary()
            if not self.chk_manual.isChecked():
                self.spin_beta.setValue(float(self.engine.results['beta']))
                self.spin_gamma.setValue(float(self.engine.results['gamma']))
        else:
            self.log("Analysis failed. See traceback in console.")

    def update_plots(self):
        res = self.engine.results
        b = float(res.get('beta', np.nan))
        g = float(res.get('gamma', np.nan))
        tc = float(res.get('Tc_for_mce', res.get('Tc', np.nan)))

        # 1) Arrott Plot
        ax = self.canvases["Arrott Plot"].axes; ax.clear()
        for d in self.engine.data_files:
            H, M = self.engine._get_filtered_data(d)
            if H.size < 5:
                continue
            ax.plot((H / M) ** (1.0 / g), M ** (1.0 / b), '.', ms=2, label=f"{d['T']}K")
        tc_kf = self.engine.results.get('Tc_KF', None)
        tc_a = self.engine.results.get('Tc_Arrott', None)
        ttl = f"Modified Arrott Plot (beta={b:.4f}, gamma={g:.4f})"
        if isinstance(tc_kf,(int,float)) and np.isfinite(tc_kf):
            ttl += f"  |  Tc_KF={tc_kf:.3f}K"
        if isinstance(tc_a,(int,float)) and np.isfinite(tc_a):
            ttl += f"  |  Tc_Arrott={tc_a:.3f}K"
        ax.set_title(ttl)
        ax.set_xlabel(r"$(H/M)^{1/\gamma}$")
        ax.set_ylabel(r"$M^{1/\beta}$")
        ax.grid(True, alpha=0.25)
        self.canvases["Arrott Plot"].draw()

        # 2) Scaling Collapse
        ax = self.canvases["Scaling Collapse"].axes; ax.clear()
        for d in self.engine.data_files:
            T = float(d['T'])
            eps = (T - tc) / tc
            if abs(eps) < 1e-4:
                continue
            H, M = self.engine._get_filtered_data(d)
            if H.size < 5:
                continue
            hs = H / (abs(eps) ** (b + g))
            ms = M / (abs(eps) ** b)
            if eps < 0:
                hs = -hs
            ax.loglog(np.abs(hs), ms, '.', ms=2)
        ax.set_title(f"Scaled M vs H (Tc={tc:.3f})")
        ax.set_xlabel(r"$|H|/|\epsilon|^{\beta+\gamma}$")
        ax.set_ylabel(r"$M/|\epsilon|^{\beta}$")
        ax.grid(True, which='both', alpha=0.25)
        self.canvases["Scaling Collapse"].draw()

        # 3) Kouvel-Fisher
        ax = self.canvases["Kouvel-Fisher"].axes; ax.clear()
        map_res = res.get('kf_map', [])
        if map_res:
            temps = np.array([r['T'] for r in map_res], dtype=float)
            kf_lines = res.get('kf_lines', {}) or {}
            kf_ms = kf_lines.get('Ms', None)
            kf_chi = kf_lines.get('ChiInv', None)

            # fallback compute if not cached
            if kf_ms is None or kf_chi is None:
                Ms = np.array([r['Ms'] for r in map_res], dtype=float)
                ChiInv = np.array([r['ChiInv'] for r in map_res], dtype=float)
                try:
                    t_s, Ms_s = _robust_sort_xy(temps, Ms)
                    dMs = np.gradient(Ms_s, t_s)
                    kf_ms_tmp = np.full_like(temps, np.nan, dtype=float)
                    for ti, mi, di in zip(t_s, Ms_s, dMs):
                        if np.isfinite(di) and abs(di) > 1e-12 and np.isfinite(mi):
                            idx = np.where(np.isclose(temps, ti, rtol=0, atol=1e-9))[0]
                            if idx.size:
                                kf_ms_tmp[idx[0]] = mi / di
                    kf_ms = kf_ms_tmp
                except Exception:
                    kf_ms = np.full_like(temps, np.nan, dtype=float)
                try:
                    t_s, Chi_s = _robust_sort_xy(temps, ChiInv)
                    dChi = np.gradient(Chi_s, t_s)
                    kf_chi_tmp = np.full_like(temps, np.nan, dtype=float)
                    for ti, ci, di in zip(t_s, Chi_s, dChi):
                        if np.isfinite(di) and abs(di) > 1e-12 and np.isfinite(ci):
                            idx = np.where(np.isclose(temps, ti, rtol=0, atol=1e-9))[0]
                            if idx.size:
                                kf_chi_tmp[idx[0]] = ci / di
                    kf_chi = kf_chi_tmp
                except Exception:
                    kf_chi = np.full_like(temps, np.nan, dtype=float)

            ax.plot(temps, kf_ms, 's', ms=4, label=r"$M_s/(dM_s/dT)$")
            ax.plot(temps, kf_chi, 'o', ms=4, label=r"$\chi_0^{-1}/(d\chi_0^{-1}/dT)$")

            fits = res.get('kf_fits', {}) or {}
            for key, lab in [('Ms', 'Ms-KF fit'), ('ChiInv', 'ChiInv-KF fit')]:
                fit = fits.get(key, None)
                if isinstance(fit, dict) and np.isfinite(fit.get('slope', np.nan)) and np.isfinite(fit.get('intercept', np.nan)):
                    tline = np.linspace(np.nanmin(temps), np.nanmax(temps), 200)
                    yline = float(fit['slope']) * tline + float(fit['intercept'])
                    ax.plot(tline, yline, '-', lw=1, label=lab)
                    if np.isfinite(fit.get('Tc', np.nan)):
                        ax.axvline(float(fit['Tc']), ls='--', lw=1)

            # final Tc marker
            if np.isfinite(tc):
                ax.axvline(tc, ls=':', lw=1)
            ax.set_xlabel('T (K)')
            ax.set_ylabel('KF ordinate')
            ax.set_title(f"Kouvel–Fisher (Tc_final={tc:.3f}, Tc_KF={float(res.get('Tc_KF', np.nan) if res.get('Tc_KF', np.nan) is not None else np.nan):.3f})")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "KF data not available", ha='center', va='center')
        self.canvases["Kouvel-Fisher"].draw()

        # 4) EOS Fit
        ax = self.canvases["EOS Fit"].axes; ax.clear()
        eos_params = res.get('eos_params')
        if eos_params and eos_params.get('success', True):
            # Plot linearized Arrott-like EOS for the chosen best model (approx)
            for d in self.engine.data_files:
                H, M = self.engine._get_filtered_data(d)
                if H.size < 5:
                    continue
                ax.plot(M ** (1.0 / b), (H / M) ** (1.0 / g), '.', ms=2)
            ax.set_xlabel(r"$M^{1/\beta}$")
            ax.set_ylabel(r"$(H/M)^{1/\gamma}$")
            ax.set_title(f"EOS diagnostic (best={res.get('eos_best_model')})")
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "EOS fit not available", ha='center', va='center')
        self.canvases["EOS Fit"].draw()

        # 5) GP Scaling
        ax = self.canvases["GP Scaling"].axes; ax.clear()
        if SKLEARN_AVAILABLE and res.get('gp_best_params'):
            b_gp, g_gp, tc_gp = res['gp_best_params']
            X, y = self.engine._build_scaled_dataset(b_gp, g_gp, tc_gp)
            if X.shape[0] > 0:
                ax.plot(X[:, 0], y, '.', ms=2, alpha=0.3)
                # show GP mean
                try:
                    gp = self.engine._gp_cache.get('last_gp')
                    if gp is not None:
                        xs = np.linspace(np.nanpercentile(X[:, 0], 2), np.nanpercentile(X[:, 0], 98), 200)
                        mu, sd = gp.predict(xs.reshape(-1, 1), return_std=True)
                        ax.plot(xs, mu, '-', lw=1)
                        ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, alpha=0.2)
                except Exception:
                    pass
            ax.set_title("GP scaling function (log-log)")
            ax.set_xlabel("signed log10(x)")
            ax.set_ylabel("log10(y)")
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "GP not enabled/unavailable", ha='center', va='center')
        self.canvases["GP Scaling"].draw()

        # 6) MCE
        ax = self.canvases["MCE"].axes; ax.clear()
        if self.engine.mce_data:
            T_arr = np.array(self.engine.mce_data['T'], dtype=float)
            dS = np.asarray(self.engine.mce_data['dS'], dtype=float)
            # show -dS at max field
            ax.plot(T_arr, -dS[:, -1], 'o-', ms=3)
            ax.set_xlabel('T (K)')
            ax.set_ylabel(r'$-\Delta S_M$ (arb.)')
            ax.set_title('MCE')
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "MCE not available", ha='center', va='center')
        self.canvases["MCE"].draw()

        # 7) MCE Universal
        ax = self.canvases["MCE Universal"].axes; ax.clear()
        if self.engine.mce_data:
            T_arr = np.array(self.engine.mce_data['T'], dtype=float)
            H_arr = np.asarray(self.engine.mce_data['H'], dtype=float)
            dS = np.asarray(self.engine.mce_data['dS'], dtype=float)
            delta = float(res.get('delta', 1 + g / b))
            n_theo = float(res.get('n_theo', np.nan))
            for i, T in enumerate(T_arr):
                eps = (T - tc) / tc
                # do not over-prune: many experiments only span ~±2% around Tc
                if abs(T - tc) < 0.3:
                    continue
                m = (H_arr > 0) & (np.abs(dS[i, :]) > 1e-12)
                if np.sum(m) < 8:
                    continue
                x = eps / (H_arr[m] ** (1.0 / (b + g)))
                y = np.abs(dS[i, m]) / (H_arr[m] ** n_theo)
                ax.plot(x, y, '.', ms=2)
            ax.set_title(f"MCE scaling (n_theo={n_theo:.3f})")
            ax.set_xlabel(r"$\epsilon / H^{1/(\beta+\gamma)}$")
            ax.set_ylabel(r"$|\Delta S_M|/H^n$")
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "MCE scaling not available", ha='center', va='center')
        self.canvases["MCE Universal"].draw()

        # 8) Scores
        ax = self.canvases["Scores"].axes; ax.clear()
        sc = res.get('scores', {})
        if sc:
            names = list(sc.keys())
            vals = [sc[n]['S'] for n in names]
            ax.bar(names, vals)
            ax.set_title(f"Universality scores (best={res.get('best_model')})")
            ax.set_ylabel('S (lower better)')
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, axis='y', alpha=0.25)
        else:
            ax.text(0.5, 0.5, "Scores not available", ha='center', va='center')
        self.canvases["Scores"].draw()

    def print_summary(self):
        r = self.engine.results
        tc = r.get('Tc', None)
        tc_src = r.get('Tc_source', 'N/A')
        tc_kf = r.get('Tc_KF', None)
        tc_a = r.get('Tc_Arrott', None)
        tc_coll = r.get('Tc_collapse', None)
        beta = r.get('beta', None)
        gamma = r.get('gamma', None)
        delta = r.get('delta', None)
        best = r.get('best_model', None)

        def fmt(x, nd=4):
            return "N/A" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

        errs = r.get('errors', {})
        b_err = errs.get('beta', float('nan'))
        g_err = errs.get('gamma', float('nan'))
        tc_err = errs.get('Tc', float('nan'))
        tc_kf_err = errs.get('Tc_KF', float('nan'))
        tc_a_err = errs.get('Tc_Arrott', float('nan'))
        tc_coll_err = errs.get('Tc_collapse', float('nan'))

        n_exp = r.get('n_exp', None)
        n_theo = r.get('n_theo', None)
        n_status = r.get('n_status', 'N/A')

        eos_best = r.get('eos_best_model', None) or "N/A"
        eos_params = r.get('eos_params', {})

        html = f"""
<h3>Critical Parameters</h3>
<ul>
  <li><b>Tc (used):</b> {fmt(tc,3)} &plusmn; {fmt(tc_err,3)} K <span style='color:#666'>(source: {tc_src})</span></li>
  <li><b>Tc (Kouvel-Fisher):</b> {fmt(tc_kf,3)}{(f" &plusmn; {fmt(tc_kf_err,3)}" if isinstance(tc_kf_err, (int,float)) and np.isfinite(tc_kf_err) else "")} K</li>
  <li><b>Tc (Modified Arrott):</b> {fmt(tc_a,3)}{(f" &plusmn; {fmt(tc_a_err,3)}" if isinstance(tc_a_err,(int,float)) and np.isfinite(tc_a_err) else "")} K</li>
  <li><b>Tc (collapse):</b> {fmt(tc_coll,3)}{(f" &plusmn; {fmt(tc_coll_err,3)}" if isinstance(tc_coll_err,(int,float)) and np.isfinite(tc_coll_err) else "")} K</li>
  <li><b>beta:</b> {fmt(beta,4)} &plusmn; {fmt(b_err,4)}</li>
  <li><b>gamma:</b> {fmt(gamma,4)} &plusmn; {fmt(g_err,4)}</li>
  <li><b>delta (Widom):</b> {fmt(delta,4)}</li>
</ul>
<hr>
<h3>Universality</h3>
<ul>
  <li><b>Best match:</b> {best}</li>
  <li><b>EOS best model:</b> {eos_best} (chi2~{fmt(eos_params.get('chi2_red', float('nan')),4)})</li>
</ul>
<hr>
<h3>MCE Consistency</h3>
<ul>
  <li><b>Tc used for MCE:</b> {fmt(r.get('Tc_for_mce', None),3)} K</li>
  <li><b>n (exp, at Tc):</b> {fmt(r.get('n_exp_tc', None),4)}</li>
  <li><b>n (exp, from ΔS<sub>max</sub>):</b> {fmt(r.get('n_exp_peak', None),4)}</li>
  <li><b>n (theory):</b> {fmt(n_theo,4)}</li>
  <li><b>Status:</b> {n_status}</li>
</ul>
"""
        self.results_display.setHtml(html)

    def export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "Report.pdf", "PDF (*.pdf)")
        if not path:
            return
        with PdfPages(path) as pdf:
            # summary page
            fig = Figure(figsize=(8.5, 11))
            r = self.engine.results
            tc_src = r.get('Tc_source', 'N/A')
            errs = r.get('errors', {}) if isinstance(r.get('errors', {}), dict) else {}

            tc_used_val = r.get('Tc', float('nan'))
            tc_used_err = errs.get('Tc', float('nan'))
            if isinstance(tc_used_val, (int, float)) and np.isfinite(tc_used_val):
                if isinstance(tc_used_err, (int, float)) and np.isfinite(tc_used_err):
                    tc_used_line = f"Tc (used):    {tc_used_val:.5f} +/- {tc_used_err:.5f}  [source: {tc_src}]"
                else:
                    tc_used_line = f"Tc (used):    {tc_used_val:.5f}  [source: {tc_src}]"
            else:
                tc_used_line = f"Tc (used):    N/A  [source: {tc_src}]"

            tc_kf_val = r.get('Tc_KF', float('nan'))
            tc_kf_err = errs.get('Tc_KF', float('nan'))
            if isinstance(tc_kf_val, (int, float)) and np.isfinite(tc_kf_val):
                if isinstance(tc_kf_err, (int, float)) and np.isfinite(tc_kf_err):
                    tc_kf_line = f"Tc (KF):      {tc_kf_val:.5f} +/- {tc_kf_err:.5f}"
                else:
                    tc_kf_line = f"Tc (KF):      {tc_kf_val:.5f}"
            else:
                tc_kf_line = "Tc (KF):      N/A"

            tc_a_val = r.get('Tc_Arrott', float('nan'))
            tc_a_err = errs.get('Tc_Arrott', float('nan'))
            if isinstance(tc_a_val, (int, float)) and np.isfinite(tc_a_val):
                if isinstance(tc_a_err, (int, float)) and np.isfinite(tc_a_err):
                    tc_a_line = f"Tc (Arrott):  {tc_a_val:.5f} +/- {tc_a_err:.5f}"
                else:
                    tc_a_line = f"Tc (Arrott):  {tc_a_val:.5f}"
            else:
                tc_a_line = "Tc (Arrott):  N/A"

            # Classic Arrott (mean-field) Tc diagnostic
            tc_amf_val = r.get('Tc_Arrott_MF', float('nan'))
            tc_amf_err = errs.get('Tc_Arrott_MF', float('nan'))
            if isinstance(tc_amf_val, (int, float)) and np.isfinite(tc_amf_val):
                if isinstance(tc_amf_err, (int, float)) and np.isfinite(tc_amf_err):
                    tc_amf_line = f"Tc (Arrott MF): {tc_amf_val:.5f} +/- {tc_amf_err:.5f}"
                else:
                    tc_amf_line = f"Tc (Arrott MF): {tc_amf_val:.5f}"
            else:
                tc_amf_line = "Tc (Arrott MF): N/A"

            # Also show the raw collapse Tc (if available) and the selection rule
            tc_coll_val = r.get('Tc_collapse', float('nan'))
            tc_coll_err = errs.get('Tc_collapse', float('nan'))
            if isinstance(tc_coll_val, (int, float)) and np.isfinite(tc_coll_val):
                if isinstance(tc_coll_err, (int, float)) and np.isfinite(tc_coll_err):
                    tc_coll_line = f"Tc (collapse): {tc_coll_val:.5f} +/- {tc_coll_err:.5f}"
                else:
                    tc_coll_line = f"Tc (collapse): {tc_coll_val:.5f}"
            else:
                tc_coll_line = "Tc (collapse): N/A"

            tc_rule_line = "Tc selection: prefer KF > Arrott (MAP) > collapse (if previous not available)"

            beta_val = r.get('beta', float('nan'))
            beta_err = errs.get('beta', float('nan'))
            gamma_val = r.get('gamma', float('nan'))
            gamma_err = errs.get('gamma', float('nan'))

            beta_line = f"beta:        {beta_val:.5f} +/- {beta_err:.5f}" if (isinstance(beta_val,(int,float)) and np.isfinite(beta_val) and isinstance(beta_err,(int,float)) and np.isfinite(beta_err)) else (f"beta:        {beta_val:.5f} +/- N/A" if isinstance(beta_val,(int,float)) and np.isfinite(beta_val) else f"beta:        {beta_val}")
            gamma_line = f"gamma:       {gamma_val:.5f} +/- {gamma_err:.5f}" if (isinstance(gamma_val,(int,float)) and np.isfinite(gamma_val) and isinstance(gamma_err,(int,float)) and np.isfinite(gamma_err)) else (f"gamma:       {gamma_val:.5f} +/- N/A" if isinstance(gamma_val,(int,float)) and np.isfinite(gamma_val) else f"gamma:       {gamma_val}")

            txt = [
                "Comprehensive Magnetic Critical Analysis Report",
                "=================================================",
                "",
                tc_used_line,
                tc_kf_line,
                tc_a_line,
                tc_amf_line,
                tc_coll_line,
                tc_rule_line,
                beta_line,
                gamma_line,
                f"delta:       {r.get('delta', float('nan')):.5f}",
                "",
                f"Best universality: {r.get('best_model')}",
                f"EOS best: {r.get('eos_best_model')} (chi2~{r.get('eos_params', {}).get('chi2_red', float('nan'))})",
                "",
                f"Tc for MCE:  {r.get('Tc_for_mce', None)}",
                f"MCE n(exp, Tc): {r.get('n_exp_tc', None)}",
                f"MCE n(exp, peak): {r.get('n_exp_peak', None)}",
                f"MCE n(exp, used): {r.get('n_exp', None)}",
                f"MCE n(theo): {r.get('n_theo', None)}",
                f"MCE status: {r.get('n_status', None)}",
                "",
                "Generated by MagneticAnalyzer v24",
            ]
            fig.text(0.08, 0.95, "\n".join(txt), va='top', family='monospace', fontsize=11)
            pdf.savefig(fig)

            # plots
            for name, canvas in self.canvases.items():
                pdf.savefig(canvas.fig)
        self.log(f"Saved report to: {path}")



# Backward/forward compatible alias for CLI tools


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

