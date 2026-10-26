"""Walk-forward portfolio construction and backtesting helpers.

This module removes realized-return leakage from portfolio construction:
for each rebalance date t, trailing returns use only prices strictly before t.
Static factor exposures may still be contaminated if they were estimated with
future data; that is an explicit temporary limitation of the research design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json

import numpy as np
import pandas as pd

from .optimizer_oos import OptimizerSettings, create_portfolio_weights_from_cross_section


@dataclass(frozen=True)
class WalkForwardSettings:
    lookback_years: int = 3
    min_training_observations: int = 252
    min_training_coverage: float = 0.70
    cache_dir: Path = Path("WalkForwardPortfolios")
    rebuild_cache: bool = False


def load_static_exposures(
    master_path: str | Path,
    factors: list[str],
    sheet_name: str = "Raw",
) -> pd.DataFrame:
    """Load the current/static factor exposure snapshot.

    NOTE: This is the intentionally remaining contamination if these exposures
    were estimated using future information.
    """
    raw = pd.read_excel(master_path, sheet_name=sheet_name)
    required = ["Symbol", *factors]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"Master file missing columns: {missing}")

    out = raw[required].copy()
    out["Symbol"] = out["Symbol"].astype(str)
    for f in factors:
        out[f] = pd.to_numeric(out[f], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.drop_duplicates("Symbol", keep="last").set_index("Symbol").sort_index()
    return out


def load_universe_prices(
    hist_dir: str | Path,
    symbols: list[str],
) -> pd.DataFrame:
    """Load price histories into one wide matrix.

    Expected filename convention: '<SYMBOL> Hist.csv'.  If a CSV contains a
    column exactly matching the symbol it is used. Otherwise, if there is only
    one non-Date column, that column is used.
    """
    hist_dir = Path(hist_dir)
    series = []
    missing_files = []

    for symbol in symbols:
        path = hist_dir / f"{symbol} Hist.csv"
        if not path.exists():
            missing_files.append(symbol)
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        if "Date" not in df.columns:
            raise ValueError(f"{path} has no Date column.")

        candidates = [c for c in df.columns if c != "Date"]
        if symbol in df.columns:
            value_col = symbol
        elif len(candidates) == 1:
            value_col = candidates[0]
        else:
            numeric_candidates = []
            for c in candidates:
                x = pd.to_numeric(df[c], errors="coerce")
                if x.notna().sum() > 0:
                    numeric_candidates.append(c)
            if len(numeric_candidates) != 1:
                raise ValueError(
                    f"Cannot infer price column in {path}; candidates={candidates}."
                )
            value_col = numeric_candidates[0]

        s = pd.Series(
            pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(df["Date"]).normalize(),
            name=symbol,
        )
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series.append(s)

    if missing_files:
        print(f"Warning: {len(missing_files)} history files missing; first few: {missing_files[:10]}")
    if not series:
        raise ValueError("No price histories could be loaded.")

    return pd.concat(series, axis=1).sort_index()


def build_training_cross_section(
    rebalance_date: pd.Timestamp,
    prices: pd.DataFrame,
    static_exposures: pd.DataFrame,
    settings: WalkForwardSettings,
) -> pd.DataFrame:
    """Create point-in-time trailing returns using data strictly before t."""
    t = pd.Timestamp(rebalance_date).normalize()
    start = t - pd.DateOffset(years=settings.lookback_years)

    # Strictly before t prevents using the rebalance-date close to select weights
    # if the strategy is assumed to rebalance at that date's open / before close.
    window = prices.loc[(prices.index >= start) & (prices.index < t)].copy()
    if window.empty:
        raise ValueError(f"No training prices before {t.date()}.")

    rows = []
    for symbol in static_exposures.index.intersection(window.columns):
        s = window[symbol].dropna()
        if len(s) < settings.min_training_observations:
            continue
        coverage = len(s) / max(len(window), 1)
        if coverage < settings.min_training_coverage:
            continue
        first = float(s.iloc[0])
        last = float(s.iloc[-1])
        if not np.isfinite(first) or not np.isfinite(last) or first <= 0:
            continue
        rows.append((symbol, last / first - 1.0, len(s), coverage))

    if not rows:
        raise ValueError(f"No assets meet training-history requirements at {t.date()}.")

    cs = pd.DataFrame(rows, columns=["Symbol", "TrainingReturn", "TrainingObs", "TrainingCoverage"])
    cs = cs.join(static_exposures, on="Symbol", how="inner")
    return cs.reset_index(drop=True)


def _cache_paths(cache_dir: Path, date: pd.Timestamp) -> dict[str, Path]:
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    folder = cache_dir / d
    return {
        "folder": folder,
        "positive": folder / "positive.csv",
        "neutral": folder / "neutral.csv",
        "negative": folder / "negative.csv",
        "diagnostics": folder / "diagnostics.json",
    }


def build_walkforward_portfolios(
    rebalance_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    static_exposures: pd.DataFrame,
    base_targets: Mapping[str, float],
    optimizer_settings: OptimizerSettings,
    wf_settings: WalkForwardSettings,
    *,
    oil_factor: str = "OILP",
    positive_beta: float = 0.25,
    negative_beta: float = -0.10,
    verbose: bool = True,
) -> tuple[dict[pd.Timestamp, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Build/cache three point-in-time portfolios at every rebalance date."""
    wf_settings.cache_dir.mkdir(parents=True, exist_ok=True)
    schedule = {}
    diag_rows = []

    targets_by_regime = {
        "positive": {**base_targets, oil_factor: positive_beta},
        "neutral": {**base_targets, oil_factor: 0.0},
        "negative": {**base_targets, oil_factor: negative_beta},
    }

    # Last date has no next holding interval, so no portfolio is needed there.
    for idx, date in enumerate(pd.DatetimeIndex(rebalance_dates[:-1])):
        date = pd.Timestamp(date).normalize()
        cp = _cache_paths(wf_settings.cache_dir, date)
        have_cache = all(cp[r].exists() for r in ["positive", "neutral", "negative"])

        if have_cache and not wf_settings.rebuild_cache:
            portfolios = {
                r: pd.read_csv(cp[r]) for r in ["positive", "neutral", "negative"]
            }
            if cp["diagnostics"].exists():
                try:
                    cached_diag = json.loads(cp["diagnostics"].read_text())
                    diag_rows.extend(cached_diag)
                except Exception:
                    pass
            schedule[date] = portfolios
            if verbose and (idx % 12 == 0):
                print(f"{date.date()}: loaded cached portfolios")
            continue

        cs = build_training_cross_section(date, prices, static_exposures, wf_settings)
        portfolios = {}
        date_diags = []

        for regime, targets in targets_by_regime.items():
            p, d = create_portfolio_weights_from_cross_section(
                cs,
                targets,
                optimizer_settings,
                verbose=False,
            )
            portfolios[regime] = p
            d = {"Date": str(date.date()), "Regime": regime, **d}
            date_diags.append(d)
            diag_rows.append(d)

        cp["folder"].mkdir(parents=True, exist_ok=True)
        for regime, p in portfolios.items():
            p.to_csv(cp[regime], index=False)
        cp["diagnostics"].write_text(json.dumps(date_diags, indent=2))
        schedule[date] = portfolios

        if verbose and (idx % 6 == 0 or idx == len(rebalance_dates) - 2):
            print(f"{date.date()}: built portfolios from {len(cs)} training assets")

    return schedule, pd.DataFrame(diag_rows)


def _portfolio_return_for_period(
    returns: pd.DataFrame,
    portfolio: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    p = portfolio.set_index("Symbol") if "Symbol" in portfolio.columns else portfolio.copy()
    weights = p["Weight"].astype(float)
    missing_symbols = weights.index.difference(returns.columns)
    if len(missing_symbols):
        raise ValueError(f"Missing return histories for symbols: {missing_symbols[:10].tolist()}")

    period = returns.loc[(returns.index > start) & (returns.index <= end), weights.index].copy()
    if period.empty:
        raise ValueError(f"No returns in {start.date()} -> {end.date()}.")

    # Structural market-closed days: all names missing -> drop.
    period = period.loc[~period.isna().all(axis=1)]
    if period.empty:
        raise ValueError(f"No trading observations in {start.date()} -> {end.date()}.")

    # Partial missingness is a true data-quality problem.
    bad = period.isna().any(axis=1)
    if bad.any():
        dt = period.index[bad][0]
        syms = period.columns[period.loc[dt].isna()].tolist()
        raise ValueError(
            f"Partial missing returns on {dt.date()} within {start.date()} -> {end.date()}; "
            f"missing={syms[:10]}"
        )

    return period.mul(weights, axis=1).sum(axis=1)


def precompute_regime_returns(
    target: pd.DataFrame,
    portfolio_schedule: dict[pd.Timestamp, dict[str, pd.DataFrame]],
    asset_returns: pd.DataFrame,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Precompute + / 0 / - portfolio daily returns once per holding interval.

    Monte Carlo runs then only choose among these already-OOS return streams;
    the optimizer is never rerun inside the Monte Carlo loop.
    """
    out = {}
    for date, row in target.iterrows():
        start = pd.Timestamp(date).normalize()
        end = pd.Timestamp(row["NextDate"]).normalize()
        if start not in portfolio_schedule:
            raise KeyError(f"No walk-forward portfolios for {start.date()}.")

        cols = {}
        for regime in ["positive", "neutral", "negative"]:
            cols[regime] = _portfolio_return_for_period(
                asset_returns,
                portfolio_schedule[start][regime],
                start,
                end,
            )

        # Use only dates common to all three regimes to keep switching comparisons fair.
        df = pd.concat(cols, axis=1, join="inner").dropna()
        if df.empty:
            raise ValueError(f"No common regime returns for {start.date()} -> {end.date()}.")
        out[start] = df
    return out


def run_oos_backtest_from_precomputed(
    target: pd.DataFrame,
    signal_df: pd.DataFrame,
    regime_returns: dict[pd.Timestamp, pd.DataFrame],
    threshold: float,
    positive_beta: float = 0.25,
    negative_beta: float = -0.10,
) -> pd.DataFrame:
    """Apply one simulated timing signal to precomputed OOS regime returns."""
    pieces = []
    for date, row in target.iterrows():
        start = pd.Timestamp(date).normalize()
        decision = float(signal_df.at[start, "decision_signal"])
        if decision >= threshold:
            regime, beta = "positive", positive_beta
        elif decision <= -threshold:
            regime, beta = "negative", negative_beta
        else:
            regime, beta = "neutral", 0.0

        r = regime_returns[start][regime]
        piece = pd.DataFrame({
            "Change": r,
            "Regime": regime,
            "Beta": beta,
            "Signal": decision,
            "SignalDate": start,
        }, index=r.index)
        pieces.append(piece)

    result = pd.concat(pieces).sort_index()
    if result.index.duplicated().any():
        dupes = result.index[result.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate return dates across holding periods: {dupes[:10]}")
    result["Notional"] = (1.0 + result["Change"]).cumprod()
    return result


def make_market_rebalance_dates(
    prices: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    step_calendar_days: int = 30,
    min_market_coverage: float = 0.80,
) -> pd.DatetimeIndex:
    """Generate calendar-day anchors and snap each to the next real market date.

    A valid market date is one on which at least `min_market_coverage` of the
    loaded universe has a non-missing price. This prevents weekend/holiday
    rebalance dates even when a master skeleton contains calendar days.
    """
    if step_calendar_days <= 0:
        raise ValueError("step_calendar_days must be positive.")
    if not 0 < min_market_coverage <= 1:
        raise ValueError("min_market_coverage must be in (0, 1].")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        raise ValueError("end_date must be after start_date.")

    coverage = prices.notna().mean(axis=1)
    valid = pd.DatetimeIndex(coverage.index[coverage >= min_market_coverage]).sort_values()
    valid = valid[(valid >= start) & (valid <= end)]
    if len(valid) < 2:
        raise ValueError("Not enough valid market dates in requested sample.")

    anchors = pd.date_range(start, end, freq=f"{step_calendar_days}D")
    snapped = []
    for anchor in anchors:
        pos = valid.searchsorted(anchor, side="left")
        if pos < len(valid):
            snapped.append(valid[pos])

    # Ensure the sample end is represented by the last valid market date <= end.
    last_valid = valid[-1]
    if not snapped or snapped[-1] < last_valid:
        snapped.append(last_valid)

    snapped = pd.DatetimeIndex(pd.unique(snapped)).sort_values()
    if len(snapped) < 2:
        raise ValueError("Need at least two snapped rebalance dates.")
    return snapped
