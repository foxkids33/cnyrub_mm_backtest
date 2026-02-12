from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_sorted_trades(df: pd.DataFrame) -> pd.DataFrame:
    required = {"TIME", "SIDE", "TRADE", "VOLUME", "log_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in trades: {sorted(missing)}")

    out = df.copy()
    out["log_date"] = pd.to_datetime(out["log_date"], errors="coerce")
    if out["log_date"].isna().any():
        raise ValueError("Failed to parse log_date to datetime.")

    out["TIME"] = pd.to_numeric(out["TIME"], errors="coerce")
    out["TRADE"] = pd.to_numeric(out["TRADE"], errors="coerce")
    out["VOLUME"] = pd.to_numeric(out["VOLUME"], errors="coerce")
    out["SIDE"] = pd.to_numeric(out["SIDE"], errors="coerce")

    out = out.dropna(subset=["log_date", "TIME", "TRADE", "VOLUME", "SIDE"]).copy()
    out = out.sort_values(["log_date", "TIME"]).reset_index(drop=True)
    return out


def make_datetime_index(log_date: pd.Series, time_us: pd.Series) -> pd.Series:
    td = pd.to_timedelta(time_us.astype("int64"), unit="us")
    return pd.to_datetime(log_date.dt.date) + td


def infer_tick_size(prices: np.ndarray, max_samples: int = 200_000) -> float:
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return 0.0001
    if p.size > max_samples:
        p = p[:max_samples]

    diffs = np.abs(np.diff(p))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0001

    est = float(np.quantile(diffs, 0.05))
    if not np.isfinite(est) or est <= 0:
        return 0.0001

    decimals = int(max(0, min(10, round(-np.log10(est) + 2))))
    est_r = round(est, decimals)
    return float(est_r) if est_r > 0 else float(est)


def round_to_tick(price: float, tick: float, side: str = "nearest") -> float:
    if tick <= 0:
        return float(price)
    x = price / tick
    if side == "down":
        return float(np.floor(x) * tick)
    if side == "up":
        return float(np.ceil(x) * tick)
    return float(np.round(x) * tick)
