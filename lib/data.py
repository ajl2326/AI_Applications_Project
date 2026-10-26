from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_calendar(path: Path, start: str, end: str) -> pd.DatetimeIndex:
    df = pd.read_excel(path, parse_dates=["Date"])
    dates = pd.DatetimeIndex(df["Date"]).normalize().sort_values().unique()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = dates[(dates >= start_ts) & (dates <= end_ts)]
    if len(dates) < 2:
        raise ValueError("Calendar has fewer than two dates in the requested sample.")
    return pd.DatetimeIndex(dates)


def make_rebalance_dates(
    calendar: pd.DatetimeIndex,
    start: str,
    end: str,
    step_calendar_days: int,
) -> pd.DatetimeIndex:
    nominal = pd.date_range(start=start, end=end, freq=f"{step_calendar_days}D")
    snapped: list[pd.Timestamp] = []
    for d in nominal:
        pos = calendar.searchsorted(d, side="left")
        if pos < len(calendar):
            snapped.append(calendar[pos])
    out = pd.DatetimeIndex(list(dict.fromkeys(snapped)))
    if len(out) < 2:
        raise ValueError("Need at least two rebalance dates.")
    return out


def prepare_forward_target(
    oil_path,
    calendar,
    rebalance_dates,
    winsor_tail=0.02,
):
    """
    Build standardized forward oil returns over the exact rebalance intervals.

    Parameters
    ----------
    oil_path : str or pathlib.Path
        CSV containing columns:
            Date
            Factor Level

    calendar : pd.DatetimeIndex
        Daily/trading-date skeleton used by the backtest.

    rebalance_dates : pd.DatetimeIndex
        Rebalance dates. Forward returns are computed from each date
        to the next rebalance date.

    winsor_tail : float, default 0.02
        Fraction winsorized in each tail before standardization.

    Returns
    -------
    pd.DataFrame
        Indexed by rebalance start date with columns:
            NextDate
            ForwardReturn
            ForwardReturnWinsor
            z
    """

    import numpy as np
    import pandas as pd

    # ---------------------------------------------------------
    # 1. Load raw monthly oil data
    # ---------------------------------------------------------
    raw = pd.read_csv(
        oil_path,
        parse_dates=["Date"],
    )

    required_cols = {"Date", "Factor Level"}

    missing_cols = required_cols.difference(raw.columns)

    if missing_cols:
        raise ValueError(
            f"Oil file is missing required columns: "
            f"{sorted(missing_cols)}"
        )


    # ---------------------------------------------------------
    # 2. Clean monthly observations
    # ---------------------------------------------------------
    oil_raw = (
        raw[["Date", "Factor Level"]]
        .rename(columns={"Factor Level": "Oil"})
        .copy()
    )

    oil_raw["Date"] = (
        oil_raw["Date"]
        .dt.normalize()
    )

    oil_raw["Oil"] = pd.to_numeric(
        oil_raw["Oil"],
        errors="coerce",
    )

    oil_raw = (
        oil_raw
        .dropna(subset=["Date", "Oil"])
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .set_index("Date")
        .sort_index()
    )


    # ---------------------------------------------------------
    # 3. Normalize calendar / rebalance dates
    # ---------------------------------------------------------
    calendar = pd.DatetimeIndex(
        pd.to_datetime(calendar)
    ).normalize()

    rebalance_dates = pd.DatetimeIndex(
        pd.to_datetime(rebalance_dates)
    ).normalize()


    # ---------------------------------------------------------
    # 4. IMPORTANT:
    #
    # Interpolate on the UNION of:
    #
    #   - actual monthly oil observation dates
    #   - skeleton/calendar dates
    #
    # This prevents observations such as 2017-01-01 from being
    # discarded simply because they are not skeleton dates.
    # ---------------------------------------------------------
    interpolation_index = (
        oil_raw.index
        .union(calendar)
        .union(rebalance_dates)
        .sort_values()
    )

    oil_daily = oil_raw.reindex(
        interpolation_index
    )


    # ---------------------------------------------------------
    # 5. Time interpolation between monthly observations
    # ---------------------------------------------------------
    oil_daily["Oil"] = (
        oil_daily["Oil"]
        .interpolate(
            method="time",
            limit_area="inside",
        )
    )


    # ---------------------------------------------------------
    # 6. Validate rebalance dates
    #
    # Because rebalance dates are included explicitly in the
    # interpolation index, we can safely query them directly.
    # ---------------------------------------------------------
    missing_rebalance_dates = [
        date
        for date in rebalance_dates
        if (
            date not in oil_daily.index
            or pd.isna(oil_daily.at[date, "Oil"])
        )
    ]

    if missing_rebalance_dates:
        raise ValueError(
            "Oil interpolation failed for rebalance dates: "
            f"{missing_rebalance_dates[:10]}"
        )


    # ---------------------------------------------------------
    # 7. Compute forward oil returns over exact holding periods
    # ---------------------------------------------------------
    rows = []

    for start_date, end_date in zip(
        rebalance_dates[:-1],
        rebalance_dates[1:],
    ):

        start_level = oil_daily.at[
            start_date,
            "Oil",
        ]

        end_level = oil_daily.at[
            end_date,
            "Oil",
        ]

        if pd.isna(start_level) or pd.isna(end_level):
            raise ValueError(
                "Missing oil level for interval "
                f"{start_date.date()} -> "
                f"{end_date.date()}"
            )

        forward_return = (
            end_level / start_level
        ) - 1.0

        rows.append(
            {
                "Date": start_date,
                "NextDate": end_date,
                "ForwardReturn": forward_return,
            }
        )


    target = pd.DataFrame(rows)

    if target.empty:
        raise ValueError(
            "No forward-return intervals were created."
        )

    target = target.set_index("Date")


    # ---------------------------------------------------------
    # 8. Winsorize forward returns
    # ---------------------------------------------------------
    if not 0.0 <= winsor_tail < 0.5:
        raise ValueError(
            "winsor_tail must satisfy "
            "0 <= winsor_tail < 0.5"
        )

    lower = target[
        "ForwardReturn"
    ].quantile(
        winsor_tail
    )

    upper = target[
        "ForwardReturn"
    ].quantile(
        1.0 - winsor_tail
    )

    target[
        "ForwardReturnWinsor"
    ] = target[
        "ForwardReturn"
    ].clip(
        lower=lower,
        upper=upper,
    )


    # ---------------------------------------------------------
    # 9. Standardize target
    # ---------------------------------------------------------
    mean_return = target[
        "ForwardReturnWinsor"
    ].mean()

    std_return = target[
        "ForwardReturnWinsor"
    ].std(
        ddof=0
    )

    if (
        not np.isfinite(std_return)
        or std_return <= 0
    ):
        raise ValueError(
            "Forward-return target has zero or invalid "
            "standard deviation."
        )

    target["z"] = (
        target[
            "ForwardReturnWinsor"
        ] - mean_return
    ) / std_return


    # ---------------------------------------------------------
    # 10. Final validation
    # ---------------------------------------------------------
    if target[
        [
            "ForwardReturn",
            "ForwardReturnWinsor",
            "z",
        ]
    ].isna().any().any():
        raise ValueError(
            "NaNs remain in prepared forward target."
        )


    return target


def load_portfolio(path: Path) -> pd.DataFrame:
    p = pd.read_excel(path)
    required = {"Symbol", "Weight"}
    missing = required.difference(p.columns)
    if missing:
        raise ValueError(f"Portfolio {path} missing columns: {sorted(missing)}")
    p = p[["Symbol", "Weight"]].copy()
    p["Symbol"] = p["Symbol"].astype(str)
    p["Weight"] = pd.to_numeric(p["Weight"], errors="raise")
    if p["Symbol"].duplicated().any():
        dupes = p.loc[p["Symbol"].duplicated(), "Symbol"].tolist()
        raise ValueError(f"Duplicate symbols in {path}: {dupes}")
    if not np.isfinite(p["Weight"]).all():
        raise ValueError(f"Non-finite weights in {path}")
    return p.set_index("Symbol")


def _read_symbol_price(hist_dir: Path, symbol: str) -> pd.Series:
    path = hist_dir / f"{symbol} Hist.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.normalize()
    candidate_cols = [c for c in df.columns if c != "Date"]
    if symbol in df.columns:
        value_col = symbol
    elif len(candidate_cols) == 1:
        value_col = candidate_cols[0]
    else:
        numeric = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) != 1:
            raise ValueError(f"Cannot infer price column for {symbol}. Columns: {df.columns.tolist()}")
        value_col = numeric[0]
    s = (
        df[["Date", value_col]]
        .drop_duplicates("Date", keep="last")
        .set_index("Date")[value_col]
        .sort_index()
    )
    s.name = symbol
    return s


def load_asset_returns(
    hist_dir: Path,
    calendar: pd.DatetimeIndex,
    portfolios: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    symbols = sorted(set().union(*(p.index for p in portfolios.values())))
    prices = pd.concat([_read_symbol_price(hist_dir, s) for s in symbols], axis=1)
    prices = prices.reindex(calendar)
    return prices.pct_change(fill_method=None)
