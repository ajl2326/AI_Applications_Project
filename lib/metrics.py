from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, annual_rf: float = 0.0) -> dict:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        raise ValueError("Need at least two daily returns for metrics.")
    ann_factor = 252.0
    daily_rf = (1.0 + annual_rf) ** (1.0 / ann_factor) - 1.0
    excess = r - daily_rf
    ann_vol = r.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = excess.mean() * ann_factor / ann_vol if ann_vol > 0 else np.nan
    wealth = (1.0 + r).cumprod()
    years = len(r) / ann_factor
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "n_days": int(len(r)),
        "cagr": float(cagr),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "ending_notional": float(wealth.iloc[-1]),
    }
