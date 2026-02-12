from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

from .strategy import SimpleMarketMaker
from .utils import ensure_sorted_trades, infer_tick_size, make_datetime_index


@dataclass
class Fees:
    maker: float = 0.000005
    taker: float = 0.000045


def _fee(notional: float, rate: float) -> float:
    return float(abs(notional) * rate)


def backtest(
    trades: pd.DataFrame,
    *,
    base_spread_ticks: int = 2,
    qty: float = 1_000_000.0,
    inv_limit: float = 3_000_000.0,
    inv_skew_ticks: float = 1.0,
    mark_window: int = 5,
    fees: Fees = Fees(),
    tick_size: Optional[float] = None,
    fill_full_qty_on_touch: bool = True,
    use_side_filter: bool = True,
    quote_every_n: int = 1,
    # Dynamic spread
    min_spread_ticks: int = 1,
    max_spread_ticks: int = 20,
    vol_window: int = 50,
    vol_mult: float = 3.0,
    # Inventory risk management (taker unwind)
    enable_taker_unwind: bool = True,
    unwind_to: float = 0.0,
    taker_slip_ticks: int = 1,
) -> pd.DataFrame:
    """Step-by-step tape-only backtest.

    Execution rule (from task):
      A posted limit order is considered filled (maker) if at least one trade occurs at exactly the quote price.

    To avoid pathological double-fills when bid==ask, we optionally use `SIDE`:
      SIDE=0 (aggressive buy) -> trade likely hits ask (liquidity sells) -> only our ask can fill.
      SIDE=1 (aggressive sell) -> trade likely hits bid (liquidity buys) -> only our bid can fill.

    Mark-to-market:
      mark_price = median of last `mark_window` trades.

    Taker unwind (optional):
      If |position| > inv_limit, we send an immediate "unwind" trade at price = mark ± taker_slip_ticks*tick
      (conservative approximation without order book), paying taker fee.
    """
    df = ensure_sorted_trades(trades)

    dt = make_datetime_index(df["log_date"], df["TIME"])
    prices = df["TRADE"].to_numpy(dtype=float)
    sides = df["SIDE"].to_numpy(dtype=int)
    vols = df["VOLUME"].to_numpy(dtype=float)

    if tick_size is None:
        tick_size = infer_tick_size(prices)

    strat = SimpleMarketMaker(
        tick=float(tick_size),
        base_spread_ticks=int(base_spread_ticks),
        min_spread_ticks=int(min_spread_ticks),
        max_spread_ticks=int(max_spread_ticks),
        vol_window=int(vol_window),
        vol_mult=float(vol_mult),
        qty=float(qty),
        inv_limit=float(inv_limit),
        inv_skew_ticks=float(inv_skew_ticks),
        mark_window=int(mark_window),
    )

    cash = 0.0
    pos = 0.0
    last_prices: List[float] = []

    # Quote state
    cur_bid = float("nan")
    cur_ask = float("nan")
    cur_mid = float("nan")
    cur_mark = float("nan")
    cur_spread_ticks = int(base_spread_ticks)

    records: List[Dict[str, float]] = []

    n = len(df)
    for i in tqdm(range(n), total=n, desc="backtest", smoothing=0.01):
        trade_price = float(prices[i])
        last_prices.append(trade_price)

        mark = strat.compute_mark(np.asarray(last_prices, dtype=float))
        spread_ticks = strat.compute_spread_ticks(np.asarray(last_prices, dtype=float))

        # Update quotes every N trades (throttle)
        if i % max(1, int(quote_every_n)) == 0 or not np.isfinite(cur_mark):
            q = strat.compute_quotes(mark_price=mark, position=pos, spread_ticks=spread_ticks)
            cur_bid, cur_ask, cur_mid, cur_mark, cur_spread_ticks = q.bid, q.ask, q.mid, q.mark, q.spread_ticks

        # Maker fills on exact touch
        n_maker_fills = 0
        maker_notional = 0.0
        maker_fee = 0.0

        def fill_qty_for_trade() -> float:
            if fill_full_qty_on_touch:
                return float(strat.qty)
            return float(min(strat.qty, float(vols[i])))

        # SIDE filter: only one side can fill on this trade
        if use_side_filter:
            if sides[i] == 1:  # aggressive sell => hits bid => we buy
                if np.isfinite(cur_bid) and trade_price == cur_bid:
                    fq = fill_qty_for_trade()
                    notional = cur_bid * fq
                    fee = _fee(notional, fees.maker)
                    cash -= notional + fee
                    pos += fq
                    n_maker_fills += 1
                    maker_notional += notional
                    maker_fee += fee
            else:  # aggressive buy => hits ask => we sell
                if np.isfinite(cur_ask) and trade_price == cur_ask:
                    fq = fill_qty_for_trade()
                    notional = cur_ask * fq
                    fee = _fee(notional, fees.maker)
                    cash += notional
                    cash -= fee
                    pos -= fq
                    n_maker_fills += 1
                    maker_notional += notional
                    maker_fee += fee
        else:
            if np.isfinite(cur_bid) and trade_price == cur_bid:
                fq = fill_qty_for_trade()
                notional = cur_bid * fq
                fee = _fee(notional, fees.maker)
                cash -= notional + fee
                pos += fq
                n_maker_fills += 1
                maker_notional += notional
                maker_fee += fee

            if np.isfinite(cur_ask) and trade_price == cur_ask:
                fq = fill_qty_for_trade()
                notional = cur_ask * fq
                fee = _fee(notional, fees.maker)
                cash += notional
                cash -= fee
                pos -= fq
                n_maker_fills += 1
                maker_notional += notional
                maker_fee += fee

        # Optional taker unwind if inventory breaks limit
        n_taker_fills = 0
        taker_fee = 0.0
        taker_notional = 0.0

        if enable_taker_unwind and inv_limit > 0 and np.isfinite(mark):
            if pos > inv_limit:
                # sell to reduce to unwind_to
                target = float(unwind_to)
                sell_qty = max(0.0, pos - target)
                # conservative: cross with slip
                px = mark - int(taker_slip_ticks) * float(tick_size)
                px = float(px)
                notional = px * sell_qty
                fee = _fee(notional, fees.taker)
                cash += notional
                cash -= fee
                pos -= sell_qty
                n_taker_fills += 1
                taker_fee += fee
                taker_notional += notional
            elif pos < -inv_limit:
                target = float(unwind_to)
                buy_qty = max(0.0, target - pos)  # since pos is negative
                px = mark + int(taker_slip_ticks) * float(tick_size)
                px = float(px)
                notional = px * buy_qty
                fee = _fee(notional, fees.taker)
                cash -= notional + fee
                pos += buy_qty
                n_taker_fills += 1
                taker_fee += fee
                taker_notional += notional

        pnl = cash + pos * mark

        records.append(
            {
                "datetime": dt.iloc[i],
                "pnl": pnl,
                "cash": cash,
                "position": pos,
                "mark_price": mark,
                "mid": cur_mid,
                "bid": cur_bid,
                "ask": cur_ask,
                "tick": float(tick_size),
                "spread_ticks": float(cur_spread_ticks),
                "maker_fills": float(n_maker_fills),
                "taker_fills": float(n_taker_fills),
                "maker_fee": float(maker_fee),
                "taker_fee": float(taker_fee),
                "maker_notional": float(maker_notional),
                "taker_notional": float(taker_notional),
            }
        )

    return pd.DataFrame.from_records(records)
