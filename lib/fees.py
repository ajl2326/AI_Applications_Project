from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd


HurdleType = Literal["hard", "soft"]
CrystallizationFrequency = Literal["annual", "quarterly", "monthly"]
ManagementFeeBasis = Literal["daily_nav", "period_start_nav"]


@dataclass(frozen=True)
class FeeConfig:
    """Hedge-fund fee assumptions."""

    starting_aum: float = 100_000_000.0
    management_fee_rate: float = 0.02
    performance_fee_rate: float = 0.20
    hurdle_rate: float = 0.05
    hurdle_type: HurdleType = "hard"
    high_water_mark: bool = True
    crystallization_frequency: CrystallizationFrequency = "annual"
    management_fee_basis: ManagementFeeBasis = "daily_nav"
    day_count: int = 252

    def __post_init__(self):
        if self.starting_aum <= 0:
            raise ValueError("starting_aum must be positive.")
        for name, value in [
            ("management_fee_rate", self.management_fee_rate),
            ("performance_fee_rate", self.performance_fee_rate),
            ("hurdle_rate", self.hurdle_rate),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.hurdle_type not in {"hard", "soft"}:
            raise ValueError("hurdle_type must be 'hard' or 'soft'.")
        if self.crystallization_frequency not in {"annual", "quarterly", "monthly"}:
            raise ValueError("crystallization_frequency must be annual, quarterly, or monthly.")
        if self.management_fee_basis not in {"daily_nav", "period_start_nav"}:
            raise ValueError("management_fee_basis must be daily_nav or period_start_nav.")
        if self.day_count <= 0:
            raise ValueError("day_count must be positive.")


def _normalize_equity_curve(gross_equity: pd.Series) -> pd.Series:
    if not isinstance(gross_equity, pd.Series):
        raise TypeError("gross_equity must be a pandas Series.")

    s = gross_equity.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    s = s.sort_index()

    if s.index.has_duplicates:
        raise ValueError("gross_equity contains duplicate dates.")

    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        raise ValueError("gross_equity is empty after cleaning.")
    if (s <= 0).any():
        raise ValueError("gross_equity must remain strictly positive.")

    return s / float(s.iloc[0])


def _period_key(index: pd.DatetimeIndex, frequency: str) -> pd.PeriodIndex:
    if frequency == "annual":
        return index.to_period("Y")
    if frequency == "quarterly":
        return index.to_period("Q")
    if frequency == "monthly":
        return index.to_period("M")
    raise ValueError(f"Unsupported crystallization frequency: {frequency}")


def _hurdle_for_period(start_date: pd.Timestamp, end_date: pd.Timestamp, annual_hurdle: float) -> float:
    days = max((end_date - start_date).days, 1)
    return (1.0 + annual_hurdle) ** (days / 365.0) - 1.0


def apply_fees(gross_equity: pd.Series, config: FeeConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply fees to one gross equity curve.

    Daily sequence:
      1. Apply gross strategy return to prior net AUM.
      2. Accrue management fee.
      3. If crystallization date, assess performance fee.
      4. Update high-water mark after fee deduction.
    """
    gross = _normalize_equity_curve(gross_equity)
    gross_returns = gross.pct_change().fillna(0.0)
    dates = gross.index
    n = len(dates)

    keys = _period_key(dates, config.crystallization_frequency)
    period_end = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == n - 1 or keys[i] != keys[i + 1]:
            period_end[i] = True

    gross_aum = gross * config.starting_aum
    net_aum = np.full(n, np.nan)
    mgmt_fee = np.zeros(n)
    perf_fee = np.zeros(n)
    hwm_series = np.full(n, np.nan)

    net_aum[0] = config.starting_aum
    high_water_mark = config.starting_aum
    period_start_nav = config.starting_aum
    period_start_date = dates[0]

    ledger_rows = []

    for i in range(1, n):
        nav_before_fees = net_aum[i - 1] * (1.0 + gross_returns.iloc[i])

        if config.management_fee_basis == "daily_nav":
            mgmt_base = nav_before_fees
        else:
            mgmt_base = period_start_nav

        mgmt_fee[i] = mgmt_base * (config.management_fee_rate / config.day_count)
        nav_after_mgmt = nav_before_fees - mgmt_fee[i]

        if nav_after_mgmt <= 0:
            raise ValueError(f"Fees drove NAV non-positive on {dates[i].date()}.")

        if period_end[i]:
            hurdle_return = _hurdle_for_period(period_start_date, dates[i], config.hurdle_rate)
            hurdle_nav = period_start_nav * (1.0 + hurdle_return)

            reference_nav = period_start_nav
            if config.high_water_mark:
                reference_nav = max(reference_nav, high_water_mark)

            if config.hurdle_type == "hard":
                fee_threshold = max(reference_nav, hurdle_nav)
                eligible_profit = max(nav_after_mgmt - fee_threshold, 0.0)
            else:
                clears_hurdle = nav_after_mgmt > hurdle_nav
                eligible_profit = max(nav_after_mgmt - reference_nav, 0.0) if clears_hurdle else 0.0

            perf_fee[i] = config.performance_fee_rate * eligible_profit
            nav_after_perf = nav_after_mgmt - perf_fee[i]

            if nav_after_perf <= 0:
                raise ValueError(f"Performance fee drove NAV non-positive on {dates[i].date()}.")

            mask = (dates >= period_start_date) & (dates <= dates[i])
            period_mgmt_fee = float(mgmt_fee[mask].sum())

            ledger_rows.append({
                "Period": str(keys[i]),
                "StartDate": period_start_date,
                "EndDate": dates[i],
                "StartNetAUM": period_start_nav,
                "EndNetAUMBeforePerformanceFee": nav_after_mgmt,
                "EndNetAUMAfterFees": nav_after_perf,
                "GrossReturn": float(gross.loc[dates[i]] / gross.loc[period_start_date] - 1.0) if dates[i] != period_start_date else 0.0,
                "NetReturn": float(nav_after_perf / period_start_nav - 1.0),
                "HurdleReturn": hurdle_return,
                "HurdleNAV": hurdle_nav,
                "HighWaterMarkBeforeFee": high_water_mark,
                "ManagementFee": period_mgmt_fee,
                "PerformanceFee": perf_fee[i],
                "TotalFee": period_mgmt_fee + perf_fee[i],
            })

            net_aum[i] = nav_after_perf

            if config.high_water_mark:
                high_water_mark = max(high_water_mark, nav_after_perf)

            period_start_nav = nav_after_perf
            if i + 1 < n:
                period_start_date = dates[i + 1]
        else:
            net_aum[i] = nav_after_mgmt

        hwm_series[i] = high_water_mark

    hwm_series[0] = config.starting_aum
    total_fee = mgmt_fee + perf_fee

    daily = pd.DataFrame({
        "GrossEquity": gross,
        "GrossAUM": gross_aum,
        "NetAUM": net_aum,
        "NetEquity": net_aum / config.starting_aum,
        "ManagementFee": mgmt_fee,
        "PerformanceFee": perf_fee,
        "TotalFee": total_fee,
        "CumulativeManagementFee": np.cumsum(mgmt_fee),
        "CumulativePerformanceFee": np.cumsum(perf_fee),
        "CumulativeTotalFee": np.cumsum(total_fee),
        "HighWaterMark": hwm_series,
    }, index=dates)

    return daily, pd.DataFrame(ledger_rows)


def fee_summary(daily: pd.DataFrame, ledger: pd.DataFrame, config: FeeConfig) -> Dict[str, float]:
    total_mgmt = float(daily["ManagementFee"].sum())
    total_perf = float(daily["PerformanceFee"].sum())
    total = total_mgmt + total_perf

    return {
        "starting_aum": config.starting_aum,
        "ending_gross_aum": float(daily["GrossAUM"].iloc[-1]),
        "ending_net_aum": float(daily["NetAUM"].iloc[-1]),
        "management_fees": total_mgmt,
        "performance_fees": total_perf,
        "total_fees": total,
        "fees_as_pct_starting_aum": total / config.starting_aum,
        "performance_fee_crystallizations": int((ledger["PerformanceFee"] > 0).sum()) if not ledger.empty else 0,
        "n_crystallization_periods": int(len(ledger)),
    }


def apply_fees_to_monte_carlo(
    equity_curves: Dict[float, list[pd.Series]],
    config: FeeConfig,
    baseline_rsq: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[float, list[pd.DataFrame]]]:
    """Apply fees to every path and compute run-matched marginal fees versus baseline R^2."""
    rows = []
    fee_paths: Dict[float, list[pd.DataFrame]] = {}

    for rsq, paths in equity_curves.items():
        fee_paths[rsq] = []

        for run_idx, gross_path in enumerate(paths, start=1):
            daily, ledger = apply_fees(gross_path, config)
            summary = fee_summary(daily, ledger, config)

            rows.append({
                "rsq": float(rsq),
                "run": int(run_idx),
                **summary,
            })
            fee_paths[rsq].append(daily)

    path_summary = pd.DataFrame(rows)

    baseline = (
        path_summary[path_summary["rsq"] == float(baseline_rsq)]
        [["run", "total_fees", "management_fees", "performance_fees"]]
        .rename(columns={
            "total_fees": "baseline_total_fees",
            "management_fees": "baseline_management_fees",
            "performance_fees": "baseline_performance_fees",
        })
    )

    if not baseline.empty:
        path_summary = path_summary.merge(baseline, on="run", how="left")
        path_summary["marginal_total_fees"] = path_summary["total_fees"] - path_summary["baseline_total_fees"]
        path_summary["marginal_management_fees"] = path_summary["management_fees"] - path_summary["baseline_management_fees"]
        path_summary["marginal_performance_fees"] = path_summary["performance_fees"] - path_summary["baseline_performance_fees"]

    return path_summary, fee_paths


def aggregate_fee_economics(path_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fee economics by R^2."""
    grouped = path_summary.groupby("rsq", as_index=False)

    summary = grouped.agg(
        runs=("run", "count"),
        mean_management_fees=("management_fees", "mean"),
        median_management_fees=("management_fees", "median"),
        mean_performance_fees=("performance_fees", "mean"),
        median_performance_fees=("performance_fees", "median"),
        mean_total_fees=("total_fees", "mean"),
        median_total_fees=("total_fees", "median"),
        total_fees_p05=("total_fees", lambda x: x.quantile(0.05)),
        total_fees_p95=("total_fees", lambda x: x.quantile(0.95)),
    )

    if "marginal_total_fees" in path_summary.columns:
        marginal = grouped.agg(
            mean_marginal_total_fees=("marginal_total_fees", "mean"),
            median_marginal_total_fees=("marginal_total_fees", "median"),
            marginal_total_fees_p05=("marginal_total_fees", lambda x: x.quantile(0.05)),
            marginal_total_fees_p95=("marginal_total_fees", lambda x: x.quantile(0.95)),
            mean_marginal_management_fees=("marginal_management_fees", "mean"),
            mean_marginal_performance_fees=("marginal_performance_fees", "mean"),
        )
        summary = summary.merge(marginal, on="rsq", how="left")

    return summary
