#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linear fitting with gammas and betas_v2.py

辅助诊断 / 初值选择工具（保留作为“辅助诊断/初值选择工具”）。

做什么
- 对候选 universality class (beta,gamma) 进行 Modified Arrott Plot (MAP) 高场线性诊断
- 在一组等温数据上 *汇总* 线性度 (R^2) 与 “截距接近 0” 程度，给出更稳健的初值建议

相对 v1 的改进
- 不再只用“代表单个温度文件”做诊断；改为对所有等温线取中位数评分，避免选到“碰巧线性”的那条。
- 评分更强调 |intercept|→0（这和 MAP 的“正确 beta/gamma”更相关），同时保留 R^2 作为线性度保障。
- 允许同时扫描多个高场阈值（|H|>=f*Hmax 或绝对阈值）。

输入
- 一组等温 M(H) 文件（自动从文件名识别温度）

输出
- 每个 universality model、每个高场规则下：
  - 各温度拟合的中位数 R^2
  - 归一化截距 |b|/(|m|+eps) 的中位数
  - 综合 score（越大越好）
- 推荐 top-N 组合（可直接作为 GUI_Final_v2 的 init_beta/init_gamma, Bc 初值的先验参考）

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
    from scipy.stats import linregress
except Exception:
    linregress = None


UNIVERSALITY_MODELS: Dict[str, Tuple[float, float]] = {
    "Mean-field": (0.5, 1.0),
    "3D Ising": (0.326, 1.237),
    "3D Heisenberg": (0.3689, 1.3960),
    "3D XY": (0.345, 1.316),
    "2D Ising": (0.125, 1.75),
    "2D XY": (0.231, 2.2),
}


@dataclass
class IsoFit:
    T: float
    n: int
    slope: float
    intercept: float
    r2: float


@dataclass
class ComboScore:
    model: str
    beta: float
    gamma: float
    rule: str
    r2_median: float
    bnorm_median: float
    score: float
    n_isotherms: int


def parse_temperature_from_filename(path: str) -> Optional[float]:
    name = os.path.basename(path)
    m = re.search(r"([0-9]+\.?[0-9]*)\s*[kK]", name)
    if m:
        return float(m.group(1))
    m = re.search(r"T\s*=\s*([0-9]+\.?[0-9]*)", name)
    if m:
        return float(m.group(1))
    nums = re.findall(r"([0-9]+\.?[0-9]*)", name)
    if nums:
        # last token is often the temperature
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None


def load_columns(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 2-column H, M (supports csv or whitespace)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if str(row[0]).strip().startswith("#"):
                    continue
                try:
                    h = float(row[0]); m = float(row[1])
                except Exception:
                    continue
                rows.append((h, m))
        arr = np.array(rows, dtype=float)
    else:
        try:
            arr = np.loadtxt(path, delimiter=None, comments="#")
        except Exception:
            arr = np.genfromtxt(path, delimiter=None, comments="#")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"Need at least 2 columns (H M): {path}")

    H = np.asarray(arr[:, 0], float)
    M = np.asarray(arr[:, 1], float)
    mask = np.isfinite(H) & np.isfinite(M)
    H, M = H[mask], M[mask]
    if H.size == 0:
        raise ValueError(f"No finite data: {path}")

    # sort by |H|
    idx = np.argsort(np.abs(H))
    return H[idx], M[idx]


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if x.size < 2:
        return float("nan"), float("nan"), 0.0
    if linregress is not None:
        res = linregress(x, y)
        return float(res.slope), float(res.intercept), float(res.rvalue ** 2)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def map_transform(H: np.ndarray, M: np.ndarray, beta: float, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """MAP coordinates: X=M^{1/beta}, Y=(H/M)^{1/gamma}."""
    eps = 1e-30
    Mabs = np.maximum(np.abs(M), eps)
    X = np.sign(M) * (Mabs ** (1.0 / beta))
    HM = H / np.sign(M) / Mabs
    Y = np.sign(HM) * (np.maximum(np.abs(HM), eps) ** (1.0 / gamma))
    return X, Y


def fit_one_isotherm(H: np.ndarray, M: np.ndarray, beta: float, gamma: float,
                     frac: float | None, Hmin_abs: float | None, min_points: int) -> Optional[IsoFit]:
    if H.size < min_points:
        return None

    Habs = np.abs(H)
    Hmax = float(np.nanmax(Habs))
    if not np.isfinite(Hmax) or Hmax <= 0:
        return None

    mask = np.isfinite(H) & np.isfinite(M) & (np.abs(M) > 1e-14)
    if frac is not None:
        mask = mask & (Habs >= frac * Hmax)
    if Hmin_abs is not None:
        mask = mask & (Habs >= Hmin_abs)

    if np.count_nonzero(mask) < min_points:
        return None

    X, Y = map_transform(H[mask], M[mask], beta, gamma)
    # remove any non-finite after transform
    m2 = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[m2], Y[m2]
    if X.size < min_points:
        return None

    slope, intercept, r2 = linear_fit(X, Y)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    return IsoFit(T=float("nan"), n=int(X.size), slope=slope, intercept=intercept, r2=r2)


def combo_score(isofits: List[IsoFit]) -> Tuple[float, float, float]:
    """Return (r2_median, bnorm_median, score). Higher score is better."""
    if not isofits:
        return 0.0, float("inf"), -1e18

    r2s = np.array([f.r2 for f in isofits], float)
    slopes = np.array([f.slope for f in isofits], float)
    intercepts = np.array([f.intercept for f in isofits], float)

    r2_med = float(np.nanmedian(r2s))
    bnorm = np.abs(intercepts) / (np.abs(slopes) + 1e-12)
    bnorm_med = float(np.nanmedian(bnorm))

    # Score: emphasize intercept≈0; keep R2 as a guard.
    # The log penalty keeps it from exploding.
    score = r2_med - 0.35 * np.log10(1.0 + bnorm_med)
    return r2_med, bnorm_med, float(score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="M(H) isotherm files")
    ap.add_argument("--frac", default="0.55,0.6,0.65,0.7,0.75,0.8", help="comma-separated Hmax fractions")
    ap.add_argument("--Hmin", type=float, default=None, help="absolute |H|>=Hmin cut (optional)")
    ap.add_argument("--min_points", type=int, default=10)
    ap.add_argument("--top", type=int, default=12, help="print top N combinations")
    args = ap.parse_args()

    frac_grid = [float(x) for x in args.frac.split(",") if x.strip()]

    # Load all datasets
    datasets = []
    for f in args.files:
        H, M = load_columns(f)
        T = parse_temperature_from_filename(f)
        datasets.append((float(T) if T is not None else float("nan"), f, H, M))
    datasets.sort(key=lambda x: x[0])

    combos: List[ComboScore] = []

    for model, (beta, gamma) in UNIVERSALITY_MODELS.items():
        for frac in frac_grid:
            rule = f"|H|>={frac:.2f}*Hmax"
            fits: List[IsoFit] = []
            for T, _f, H, M in datasets:
                fr = fit_one_isotherm(H, M, beta, gamma, frac=frac, Hmin_abs=args.Hmin, min_points=args.min_points)
                if fr is None:
                    continue
                fr.T = float(T)
                fits.append(fr)
            r2_med, bnorm_med, sc = combo_score(fits)
            combos.append(ComboScore(model, beta, gamma, rule, r2_med, bnorm_med, sc, len(fits)))

        if args.Hmin is not None:
            rule = f"|H|>={args.Hmin:g}"
            fits = []
            for T, _f, H, M in datasets:
                fr = fit_one_isotherm(H, M, beta, gamma, frac=None, Hmin_abs=args.Hmin, min_points=args.min_points)
                if fr is None:
                    continue
                fr.T = float(T)
                fits.append(fr)
            r2_med, bnorm_med, sc = combo_score(fits)
            combos.append(ComboScore(model, beta, gamma, rule, r2_med, bnorm_med, sc, len(fits)))

    combos.sort(key=lambda c: (c.score, c.n_isotherms), reverse=True)

    print("=" * 92)
    print("MAP linearity scan (aggregated over all isotherms)")
    print(f"N_files={len(datasets)}  min_points={args.min_points}  Hmin={args.Hmin}")
    print("Score = median(R^2) - 0.35*log10(1+median(|b|/|m|))  (higher is better)")
    print("=" * 92)

    topn = min(args.top, len(combos))
    print("rank\tmodel\tbeta\tgamma\trule\tNiso\tR2_med\tb_norm_med\tscore")
    for i in range(topn):
        c = combos[i]
        print(
            f"{i+1}\t{c.model}\t{c.beta:.4f}\t{c.gamma:.4f}\t{c.rule}\t"
            f"{c.n_isotherms}\t{c.r2_median:.4f}\t{c.bnorm_median:.4g}\t{c.score:.6f}"
        )

    best = combos[0] if combos else None
    if best:
        print("\nRecommended init values (feed into GUI_Final_v2 / main engine):")
        print(f"  init_beta  = {best.beta:.6f}")
        print(f"  init_gamma = {best.gamma:.6f}")
        print(f"  high-field rule = {best.rule}")
        print("(Then use GUI_Final_v2: auto_suggest_bc + data-collapse optimize for final.)")


if __name__ == "__main__":
    main()
