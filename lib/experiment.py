from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .backtest import run_one_backtest
from .config import BacktestConfig, Paths
from .data import load_asset_returns, load_calendar, load_portfolio, make_rebalance_dates, prepare_forward_target
from .metrics import performance_metrics
from .signals import signal_diagnostics, simulate_signal


def run_experiment(paths: Paths, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    calendar = load_calendar(paths.skeleton, cfg.start_date, cfg.end_date)
    rebalance_dates = make_rebalance_dates(calendar, cfg.start_date, cfg.end_date, cfg.rebalance_calendar_days)
    target = prepare_forward_target(paths.oil_prices, calendar, rebalance_dates, cfg.winsor_tail)

    portfolios = {
        "positive": load_portfolio(paths.positive_portfolio),
        "neutral": load_portfolio(paths.neutral_portfolio),
        "negative": load_portfolio(paths.negative_portfolio),
    }
    returns = load_asset_returns(paths.hist_dir, calendar, portfolios)

    rows = []
    seed_sequence = np.random.SeedSequence(cfg.seed)
    seed_iter = iter(seed_sequence.spawn(len(cfg.rsq_values) * cfg.n_runs))

    for rsq in cfg.rsq_values:
        for run in range(1, cfg.n_runs + 1):
            rng = np.random.default_rng(next(seed_iter))
            signal = simulate_signal(target["z"], rsq, rng, cfg.signal_mode)
            sig_diag = signal_diagnostics(signal, rsq)
            result = run_one_backtest(target, signal, returns, portfolios, cfg)
            perf = performance_metrics(result["Change"], cfg.annual_rf)

            decision = signal["decision_signal"]
            regime_share = {
                "positive_share": float((decision >= cfg.threshold).mean()),
                "neutral_share": float((decision.abs() < cfg.threshold).mean()),
                "negative_share": float((decision <= -cfg.threshold).mean()),
            }

            stem = f"rsq-{rsq:g}_run-{run:03d}"
            result.to_csv(paths.output_dir / f"{stem}_daily.csv", index_label="Date")
            signal.join(target[["NextDate", "ForwardReturn", "ForwardReturnWinsor"]]).to_csv(
                paths.output_dir / f"{stem}_signal.csv", index_label="Date"
            )
            rows.append({"rsq": rsq, "run": run, "signal_mode": cfg.signal_mode, **sig_diag, **regime_share, **perf})

    summary = pd.DataFrame(rows)
    summary.to_csv(paths.output_dir / "summary_by_run.csv", index=False)
    aggregate = (
        summary.groupby(["rsq", "signal_mode"], as_index=False)
        .agg(
            runs=("run", "count"),
            median_sharpe=("sharpe", "median"),
            mean_sharpe=("sharpe", "mean"),
            sharpe_p05=("sharpe", lambda x: x.quantile(0.05)),
            sharpe_p95=("sharpe", lambda x: x.quantile(0.95)),
            median_cagr=("cagr", "median"),
            median_max_drawdown=("max_drawdown", "median"),
            mean_realized_corr2=("realized_corr2_score", "mean"),
            mean_positive_share=("positive_share", "mean"),
            mean_neutral_share=("neutral_share", "mean"),
            mean_negative_share=("negative_share", "mean"),
        )
    )
    aggregate.to_csv(paths.output_dir / "summary_aggregate.csv", index=False)

    metadata = {
        "config": cfg.__dict__,
        "paths": {k: str(v) for k, v in paths.__dict__.items()},
        "n_rebalance_intervals": len(target),
    }
    with open(paths.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return summary, aggregate
