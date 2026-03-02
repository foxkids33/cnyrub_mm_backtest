from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_std(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    s = float(np.std(x, ddof=1))
    return s if s > 1e-12 else 0.0


def pnl_to_step_returns(pnl: pd.Series) -> np.ndarray:
    p = pnl.to_numpy(dtype=float)
    if p.size < 2:
        return np.array([], dtype=float)
    return np.diff(p)


def pnl_to_time_returns(df: pd.DataFrame, freq: str = "1s") -> np.ndarray:

    if "datetime" not in df.columns or "pnl" not in df.columns:
        raise ValueError("df must contain datetime and pnl")
    s = df.set_index(pd.to_datetime(df["datetime"]))["pnl"].astype(float)
    s2 = s.resample(freq).last().ffill()
    return np.diff(s2.to_numpy(dtype=float))


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = float(np.mean(r))
    sd = _safe_std(r)
    return float(mu / sd) if sd > 0 else 0.0


def sortino(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = float(np.mean(r))
    downside = r[r < 0]
    dd = _safe_std(downside)
    return float(mu / dd) if dd > 0 else 0.0