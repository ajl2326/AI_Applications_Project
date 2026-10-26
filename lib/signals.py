from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SignalMode


def simulate_signal(
    z: pd.Series,
    target_rsq: float,
    rng: np.random.Generator,
    mode: SignalMode = "score",
) -> pd.DataFrame:
    """Generate a fixed-scale score or calibrated forecast with controlled R^2."""
    if not 0.0 <= target_rsq <= 1.0:
        raise ValueError("target_rsq must lie in [0, 1].")
    z_arr = z.to_numpy(dtype=float)
    if np.isnan(z_arr).any():
        raise ValueError("z contains NaNs.")

    eta = rng.standard_normal(len(z_arr))
    score = np.sqrt(target_rsq) * z_arr + np.sqrt(1.0 - target_rsq) * eta
    if mode == "score":
        decision = score
    elif mode == "forecast":
        decision = np.sqrt(target_rsq) * score
    else:
        raise ValueError(f"Unknown signal mode: {mode}")

    return pd.DataFrame(
        {
            "z_true": z_arr,
            "noise": eta,
            "score": score,
            "decision_signal": decision,
        },
        index=z.index,
    )


def signal_diagnostics(signal_df: pd.DataFrame, target_rsq: float) -> dict:
    z = signal_df["z_true"].to_numpy()
    score = signal_df["score"].to_numpy()
    decision = signal_df["decision_signal"].to_numpy()
    corr = np.corrcoef(z, score)[0, 1] if len(z) > 1 else np.nan
    mse_rsq = 1.0 - np.mean((z - decision) ** 2) / np.var(z, ddof=0)
    return {
        "target_rsq": float(target_rsq),
        "realized_corr2_score": float(corr**2),
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score, ddof=0)),
        "decision_mean": float(np.mean(decision)),
        "decision_std": float(np.std(decision, ddof=0)),
        "mse_rsq_decision": float(mse_rsq),
    }
