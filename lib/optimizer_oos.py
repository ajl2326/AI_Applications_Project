"""Point-in-time portfolio optimizer for walk-forward backtests.

The optimizer itself is deliberately date-agnostic. It receives a cross section
prepared by the caller with:
    Symbol
    TrainingReturn
    <factor columns...>

It never reads full-sample realized returns from Master.xlsx.

Factor exposures can currently be static/full-sample if that is all that is
available; that limitation is outside this module and should be disclosed in
research results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats.mstats import winsorize


@dataclass(frozen=True)
class OptimizerSettings:
    long_allocation: float = 1.0
    short_allocation: float = 1.0
    max_weight: float = 0.05
    precision: float = 1e-4
    winsor_tail: float = 0.02
    max_iterations: int = 2000
    min_assets: int = 25


def _winsorize(values: np.ndarray, tail: float) -> np.ndarray:
    if not 0 <= tail < 0.5:
        raise ValueError("winsor_tail must be in [0, 0.5).")
    return np.asarray(winsorize(values, limits=(tail, tail)), dtype=float)


def _prepare_cross_section(
    cross_section: pd.DataFrame,
    factor_targets: Mapping[str, float],
    settings: OptimizerSettings,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    factors = list(factor_targets)
    required = ["Symbol", "TrainingReturn", *factors]
    missing = [c for c in required if c not in cross_section.columns]
    if missing:
        raise KeyError(f"Cross section missing required columns: {missing}")

    data = cross_section[required].copy()
    data["Symbol"] = data["Symbol"].astype(str)
    for c in ["TrainingReturn", *factors]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if data["Symbol"].duplicated().any():
        dupes = data.loc[data["Symbol"].duplicated(), "Symbol"].unique().tolist()
        raise ValueError(f"Duplicate symbols in optimizer universe: {dupes[:10]}")
    if len(data) < settings.min_assets:
        raise ValueError(
            f"Only {len(data)} usable assets remain; need at least {settings.min_assets}."
        )

    # Robust estimation inputs.  Importantly, these winsorized exposures are used
    # ONLY for the cross-sectional regression.  Portfolio factor constraints below
    # use the raw exposures so the realized economic exposure hits the requested target.
    y = _winsorize(data["TrainingReturn"].to_numpy(dtype=float), settings.winsor_tail)
    X_est = np.column_stack([
        _winsorize(data[f].to_numpy(dtype=float), settings.winsor_tail)
        for f in factors
    ])

    # Raw factor matrix for economic constraints.
    X_constraint = data[factors].to_numpy(dtype=float)
    return data, y, X_est, X_constraint


def _fit_residuals(y: np.ndarray, X: np.ndarray):
    Xc = sm.add_constant(X, has_constant="add")
    result = sm.OLS(y, Xc).fit()
    return np.asarray(result.resid, dtype=float), result


def create_portfolio_weights_from_cross_section(
    cross_section: pd.DataFrame,
    exposure_target_dictionary: Mapping[str, float],
    settings: OptimizerSettings | None = None,
    *,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Optimize one portfolio from information supplied by the caller.

    Objective
    ---------
    Fit the trailing cross-sectional model

        TrainingReturn_i = a + beta_i' lambda + epsilon_i

    and maximize portfolio historical residual return

        max_w  epsilon' w

    subject to fixed gross long/short exposure, raw factor exposure targets,
    per-name bounds, and nonnegative portfolio residual return.

    The residual is a historical stock-selection signal, not guaranteed future alpha.
    """
    settings = settings or OptimizerSettings()
    targets = {str(k): float(v) for k, v in exposure_target_dictionary.items()}
    factors = list(targets)

    if settings.long_allocation < 0 or settings.short_allocation < 0:
        raise ValueError("Gross allocations must be non-negative.")
    if settings.max_weight <= 0 or settings.precision <= 0:
        raise ValueError("max_weight and precision must be positive.")

    data, y, X_est, X_constraint = _prepare_cross_section(
        cross_section, targets, settings
    )
    residuals, regression = _fit_residuals(y, X_est)
    n = len(data)

    min_long = int(np.ceil(settings.long_allocation / settings.max_weight - 1e-12))
    min_short = int(np.ceil(settings.short_allocation / settings.max_weight - 1e-12))
    if min_long + min_short > n:
        raise ValueError(
            f"Gross allocations require at least {min_long + min_short} names but only {n} are usable."
        )

    n_long = max(min_long, n // 2)
    n_long = min(n_long, n - min_short)
    n_short = n - n_long

    x0 = np.zeros(n)
    if settings.long_allocation:
        x0[:n_long] = settings.long_allocation / n_long
    if settings.short_allocation:
        x0[n_long:] = -settings.short_allocation / n_short

    def long_value(w):
        return float(w[w > 0].sum())

    def short_value(w):
        return float(-w[w < 0].sum())

    def residual_value(w):
        return float(residuals @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: long_value(w) - settings.long_allocation},
        {"type": "eq", "fun": lambda w: short_value(w) - settings.short_allocation},
        {"type": "ineq", "fun": residual_value},
    ]

    for j, factor in enumerate(factors):
        b = X_constraint[:, j].copy()
        t = targets[factor]
        constraints.append({
            "type": "eq",
            "fun": lambda w, b=b, t=t: float(b @ w - t),
        })

    start = time.perf_counter()
    sol = minimize(
        lambda w: -residual_value(w),
        x0,
        method="SLSQP",
        bounds=[(-settings.max_weight, settings.max_weight)] * n,
        constraints=constraints,
        options={
            "maxiter": settings.max_iterations,
            "ftol": max(1e-12, settings.precision * 0.1),
            "disp": False,
        },
    )
    runtime = time.perf_counter() - start
    w = np.asarray(sol.x, dtype=float)

    achieved = {
        factor: float(X_constraint[:, j] @ w)
        for j, factor in enumerate(factors)
    }
    errors = {
        "long": abs(long_value(w) - settings.long_allocation),
        "short": abs(short_value(w) - settings.short_allocation),
        **{f: abs(achieved[f] - targets[f]) for f in factors},
    }
    worst = max(errors.values())
    residual_ok = residual_value(w) >= -settings.precision
    bounds_ok = np.all(np.abs(w) <= settings.max_weight + settings.precision)

    if (not sol.success) or worst > settings.precision or not residual_ok or not bounds_ok:
        raise RuntimeError(
            "Invalid optimizer solution. "
            f"status={sol.message}; worst constraint error={worst:.3g}; "
            f"portfolio residual={residual_value(w):.6g}."
        )

    portfolio = pd.DataFrame({
        "Symbol": data["Symbol"].to_numpy(),
        "Weight": w,
    })

    diagnostics = {
        "n_assets": n,
        "runtime_seconds": runtime,
        "iterations": int(sol.nit),
        "regression_rsq": float(regression.rsquared),
        "regression_intercept": float(regression.params[0]),
        "portfolio_residual_return": residual_value(w),
        "long_allocation": long_value(w),
        "short_allocation": short_value(w),
        "max_abs_weight": float(np.abs(w).max()),
        **{f"exposure_{f}": achieved[f] for f in factors},
    }

    if verbose:
        print(
            f"Optimized {n} assets in {runtime:.2f}s; "
            f"cross-sectional R²={regression.rsquared:.4f}; "
            f"residual={residual_value(w):.6f}"
        )
        for f in factors:
            print(f"  {f}: {achieved[f]:.6f} / target {targets[f]:.6f}")

    return portfolio, diagnostics
