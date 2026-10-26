from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    # Preferred when imported as part of alpha_research
    from .config import BacktestConfig
except ImportError:
    # Allows direct/local execution if needed
    from config import BacktestConfig


VALID_REGIMES = {"positive", "neutral", "negative"}


def choose_regime(
    signal: float,
    threshold: float,
) -> str:
    """
    Map a continuous decision signal into one of three portfolio regimes.

    positive:
        signal >= threshold

    negative:
        signal <= -threshold

    neutral:
        otherwise
    """

    if not np.isfinite(signal):
        raise ValueError(
            f"Decision signal must be finite; received {signal}."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must be non-negative."
        )

    if signal >= threshold:
        return "positive"

    if signal <= -threshold:
        return "negative"

    return "neutral"


def _validate_portfolios(
    portfolios: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
) -> None:
    """
    Validate regime portfolio structure and make sure every required
    security exists in the return matrix.
    """

    missing_regimes = VALID_REGIMES.difference(
        portfolios.keys()
    )

    if missing_regimes:
        raise ValueError(
            f"Missing portfolio regimes: "
            f"{sorted(missing_regimes)}"
        )

    for regime in VALID_REGIMES:

        portfolio = portfolios[regime]

        if "Weight" not in portfolio.columns:
            raise ValueError(
                f"{regime} portfolio is missing "
                f"'Weight' column."
            )

        if portfolio.index.has_duplicates:
            duplicates = (
                portfolio.index[
                    portfolio.index.duplicated()
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"{regime} portfolio contains duplicate "
                f"symbols: {duplicates[:10]}"
            )

        weights = pd.to_numeric(
            portfolio["Weight"],
            errors="coerce",
        )

        if weights.isna().any():
            raise ValueError(
                f"{regime} portfolio contains "
                f"non-numeric weights."
            )

        if not np.isfinite(weights.to_numpy()).all():
            raise ValueError(
                f"{regime} portfolio contains "
                f"non-finite weights."
            )

        missing_symbols = (
            portfolio.index
            .difference(returns.columns)
        )

        if len(missing_symbols) > 0:
            raise ValueError(
                f"{regime} portfolio contains symbols "
                f"missing from return matrix: "
                f"{missing_symbols[:10].tolist()}"
            )


def _prepare_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize and validate the return matrix index.
    """

    returns = returns.copy()

    returns.index = pd.DatetimeIndex(
        pd.to_datetime(returns.index)
    ).normalize()

    if returns.index.has_duplicates:
        duplicates = (
            returns.index[
                returns.index.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate dates in return matrix: "
            f"{duplicates[:10]}"
        )

    returns = returns.sort_index()

    return returns


def _extract_valid_period(
    returns: pd.DataFrame,
    weights: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    regime: str,
) -> pd.DataFrame:
    """
    Extract one holding-period return matrix.

    Rules
    -----
    1. Use dates:
           start_date < t <= end_date

       This prevents double-counting the rebalance date across
       consecutive holding periods.

    2. If ALL selected securities are missing on a date, treat
       that date as a structural non-trading day and remove it.

       Typical examples:
           weekend
           exchange holiday

    3. If only SOME securities are missing on an otherwise valid
       date, raise an error rather than silently assuming zero
       return or deleting the observation.
    """

    symbols = weights.index

    period = returns.loc[
        (returns.index > start_date)
        & (returns.index <= end_date),
        symbols,
    ].copy()

    if period.empty:
        raise ValueError(
            f"No return observations found for "
            f"{start_date.date()} -> {end_date.date()}."
        )

    # ---------------------------------------------------------
    # Rows where every selected security is missing
    #
    # These are normally weekends / holidays created by the
    # master skeleton.
    # ---------------------------------------------------------
    all_missing = period.isna().all(axis=1)

    period = period.loc[~all_missing].copy()

    if period.empty:
        raise ValueError(
            f"No trading-day observations remain for "
            f"{regime} portfolio from "
            f"{start_date.date()} -> {end_date.date()}."
        )

    # ---------------------------------------------------------
    # Partial missingness is different.
    #
    # At least one security traded / has data, but at least one
    # selected security does not. Treat this as a data-quality
    # problem rather than silently dropping or zero-filling.
    # ---------------------------------------------------------
    partial_missing = period.isna().any(axis=1)

    if partial_missing.any():

        bad_date = period.index[
            partial_missing
        ][0]

        missing_symbols = (
            period.loc[
                bad_date,
                period.loc[bad_date].isna(),
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"Partial missing security returns in "
            f"{regime} portfolio from "
            f"{start_date.date()} -> {end_date.date()}; "
            f"first bad date={bad_date.date()}, "
            f"missing symbols={missing_symbols[:10]}"
        )

    return period


def run_one_backtest(
    target: pd.DataFrame,
    signal_df: pd.DataFrame,
    returns: pd.DataFrame,
    portfolios: Dict[str, pd.DataFrame],
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    Run one complete regime-switching backtest.

    Parameters
    ----------
    target
        DataFrame indexed by rebalance date.
        Must contain:
            NextDate

    signal_df
        DataFrame indexed by rebalance date.
        Must contain:
            decision_signal

    returns
        Daily security return matrix.
        Columns are security symbols.

    portfolios
        Dictionary containing:
            positive
            neutral
            negative

        Each portfolio must be indexed by Symbol and contain
        a Weight column.

    cfg
        BacktestConfig containing at least:
            threshold
            positive_beta
            negative_beta

    Returns
    -------
    pd.DataFrame

        Daily strategy observations with columns:
            Change
            Beta
            Regime
            Signal
            SignalDate
            Notional
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------
    if "NextDate" not in target.columns:
        raise ValueError(
            "target must contain a 'NextDate' column."
        )

    if "decision_signal" not in signal_df.columns:
        raise ValueError(
            "signal_df must contain a "
            "'decision_signal' column."
        )

    if target.empty:
        raise ValueError(
            "target is empty."
        )

    returns = _prepare_returns(
        returns
    )

    _validate_portfolios(
        portfolios=portfolios,
        returns=returns,
    )


    # ---------------------------------------------------------
    # 2. Normalize target / signal indices
    # ---------------------------------------------------------
    target = target.copy()
    signal_df = signal_df.copy()

    target.index = pd.DatetimeIndex(
        pd.to_datetime(target.index)
    ).normalize()

    signal_df.index = pd.DatetimeIndex(
        pd.to_datetime(signal_df.index)
    ).normalize()

    target = target.sort_index()
    signal_df = signal_df.sort_index()


    # ---------------------------------------------------------
    # 3. Make sure every rebalance date has a signal
    # ---------------------------------------------------------
    missing_signal_dates = (
        target.index
        .difference(signal_df.index)
    )

    if len(missing_signal_dates) > 0:
        raise ValueError(
            "Missing decision signals for rebalance dates: "
            f"{missing_signal_dates[:10].tolist()}"
        )


    # ---------------------------------------------------------
    # 4. Run each holding period
    # ---------------------------------------------------------
    pieces = []

    for date, row in target.iterrows():

        start_date = pd.Timestamp(
            date
        ).normalize()

        end_date = pd.Timestamp(
            row["NextDate"]
        ).normalize()

        if end_date <= start_date:
            raise ValueError(
                f"Invalid holding period: "
                f"{start_date.date()} -> "
                f"{end_date.date()}"
            )

        decision = float(
            signal_df.at[
                start_date,
                "decision_signal",
            ]
        )

        regime = choose_regime(
            signal=decision,
            threshold=cfg.threshold,
        )


        # -----------------------------------------------------
        # Portfolio weights
        # -----------------------------------------------------
        weights = (
            portfolios[regime]["Weight"]
            .astype(float)
            .copy()
        )


        # -----------------------------------------------------
        # Extract valid daily returns
        # -----------------------------------------------------
        period = _extract_valid_period(
            returns=returns,
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            regime=regime,
        )


        # -----------------------------------------------------
        # Portfolio daily return
        #
        # r_p,t = sum_i w_i r_i,t
        # -----------------------------------------------------
        port_ret = (
            period
            .mul(weights, axis=1)
            .sum(axis=1)
        )


        # -----------------------------------------------------
        # Store intended oil beta for diagnostics
        # -----------------------------------------------------
        if regime == "positive":
            beta = cfg.positive_beta

        elif regime == "negative":
            beta = cfg.negative_beta

        else:
            beta = 0.0


        # -----------------------------------------------------
        # Store holding-period observations
        # -----------------------------------------------------
        piece = pd.DataFrame(
            {
                "Change": port_ret,
                "Beta": beta,
                "Regime": regime,
                "Signal": decision,
                "SignalDate": start_date,
            },
            index=port_ret.index,
        )

        pieces.append(piece)


    # ---------------------------------------------------------
    # 5. Combine holding periods
    # ---------------------------------------------------------
    if not pieces:
        raise ValueError(
            "Backtest produced no return periods."
        )

    out = (
        pd.concat(pieces)
        .sort_index()
    )


    # ---------------------------------------------------------
    # 6. Guard against accidentally counting a date twice
    # ---------------------------------------------------------
    if out.index.duplicated().any():

        duplicate_dates = (
            out.index[
                out.index.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate daily return dates detected across "
            f"rebalance periods: {duplicate_dates[:10]}"
        )


    # ---------------------------------------------------------
    # 7. Final return validation
    # ---------------------------------------------------------
    if out["Change"].isna().any():
        raise ValueError(
            "NaN portfolio returns remain after "
            "backtest construction."
        )

    if not np.isfinite(
        out["Change"].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite portfolio returns detected."
        )


    # ---------------------------------------------------------
    # 8. Compound wealth
    # ---------------------------------------------------------
    out["Notional"] = (
        1.0 + out["Change"]
    ).cumprod()


    return out