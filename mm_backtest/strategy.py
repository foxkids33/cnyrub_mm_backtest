from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import round_to_tick


@dataclass
class Quotes:
    bid: float
    ask: float
    mid: float
    mark: float
    spread_ticks: int


@dataclass
class SimpleMarketMaker:
    tick: float

    base_spread_ticks: int = 2
    min_spread_ticks: int = 1
    max_spread_ticks: int = 20

    vol_window: int = 50
    vol_mult: float = 3.0

    qty: float = 1_000_000.0
    inv_limit: float = 3_000_000.0
    inv_skew_ticks: float = 1.0
    mark_window: int = 5

    def compute_mark(self, last_prices: np.ndarray) -> float:
        if last_prices.size == 0:
            return float("nan")
        w = min(self.mark_window, last_prices.size)
        return float(np.median(last_prices[-w:]))

    def _estimate_sigma(self, last_prices: np.ndarray) -> float:
        if last_prices.size < 3:
            return 0.0
        w = min(self.vol_window, last_prices.size)
        p = last_prices[-w:]
        p = p[p > 0]
        if p.size < 3:
            return 0.0
        r = np.diff(np.log(p))
        if r.size < 2:
            return 0.0
        return float(np.std(r, ddof=1))

    def compute_spread_ticks(self, last_prices: np.ndarray) -> int:
        base = int(self.base_spread_ticks)
        sigma = self._estimate_sigma(last_prices)
        mark = float(last_prices[-1]) if last_prices.size else 0.0
        extra = 0
        if sigma > 0 and self.tick > 0 and mark > 0:
            extra = int(round((self.vol_mult * mark * sigma) / self.tick))
        spread = base + extra
        spread = max(int(self.min_spread_ticks), min(int(self.max_spread_ticks), spread))
        return int(spread)

    def compute_quotes(
        self,
        mark_price: float,
        position: float,
        spread_ticks: int,
        *,
        flow: float = 0.0,
        flow_skew_ticks: float = 0.0,
    ) -> Quotes:
        if not np.isfinite(mark_price):
            mark_price = 0.0

        f = float(np.clip(flow, -1.0, 1.0))
        ref = mark_price + f * float(flow_skew_ticks) * self.tick

        mid = round_to_tick(ref, self.tick, side="nearest")

        half_spread = int(spread_ticks) * self.tick
        inv_frac = 0.0
        if self.inv_limit > 0:
            inv_frac = float(np.clip(position / self.inv_limit, -1.0, 1.0))
        skew_px = inv_frac * float(self.inv_skew_ticks) * self.tick

        if skew_px >= 0:
            bid = mid - half_spread - 2.0 * skew_px
            ask = mid + half_spread - 1.0 * skew_px
        else:
            bid = mid - half_spread - 1.0 * skew_px
            ask = mid + half_spread - 2.0 * skew_px

        bid = round_to_tick(bid, self.tick, side="down")
        ask = round_to_tick(ask, self.tick, side="up")

        if np.isfinite(bid) and np.isfinite(ask) and bid >= ask:
            ask = round_to_tick(bid + self.tick, self.tick, side="up")

        if position >= self.inv_limit:
            bid = float("nan")
        if position <= -self.inv_limit:
            ask = float("nan")

        return Quotes(
            bid=float(bid) if np.isfinite(bid) else float("nan"),
            ask=float(ask) if np.isfinite(ask) else float("nan"),
            mid=float(mid),
            mark=float(mark_price),
            spread_ticks=int(spread_ticks),
        )