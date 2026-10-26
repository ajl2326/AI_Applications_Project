from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SignalMode = Literal["score", "forecast"]


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2017-01-01"
    end_date: str = "2023-01-01"
    rebalance_calendar_days: int = 30
    threshold: float = 0.75
    positive_beta: float = 0.25
    negative_beta: float = -0.10
    rsq_values: tuple[float, ...] = (0.01,)
    n_runs: int = 10
    seed: int = 12345
    signal_mode: SignalMode = "score"
    winsor_tail: float = 0.02
    annual_rf: float = 0.0175


@dataclass(frozen=True)
class OptimizerConfig:
    long_allocation: float = 1.0
    short_allocation: float = 1.0
    max_weight: float = 0.05
    precision: float = 1e-4
    winsor_tail: float = 0.02
    maxiter: int = 1000
    method: str = "SLSQP"
    return_column: str = "Period Realized Returns"
    symbol_column: str = "Symbol"
    sheet_name: str = "Raw"


@dataclass(frozen=True)
class ExposureTargets:
    market: float = 0.8
    credit: float = 0.0
    oil: float = 0.0
    consumer: float = 0.0
    employment: float = 0.0
    factor_names: tuple[str, ...] = field(
        default=("MKTB", "CCDB", "OILP", "CONS", "EMPR")
    )

    def as_dict(self) -> dict[str, float]:
        vals = (self.market, self.credit, self.oil, self.consumer, self.employment)
        return dict(zip(self.factor_names, vals))


@dataclass(frozen=True)
class Paths:
    skeleton: Path
    oil_prices: Path
    hist_dir: Path
    master: Path
    portfolio_dir: Path
    output_dir: Path

    @property
    def positive_portfolio(self) -> Path:
        return self.portfolio_dir / "positive-beta.xlsx"

    @property
    def neutral_portfolio(self) -> Path:
        return self.portfolio_dir / "beta-neutral.xlsx"

    @property
    def negative_portfolio(self) -> Path:
        return self.portfolio_dir / "negative-beta.xlsx"
