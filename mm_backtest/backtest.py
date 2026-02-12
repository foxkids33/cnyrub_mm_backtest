from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

from .strategy import SimpleMarketMaker
from .utils import ensure_sorted_trades, infer_tick_size, make_datetime_index, price_to_tick_int


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
    min_spread_ticks: int = 1,
    max_spread_ticks: int = 20,
    vol_window: int = 50,
    vol_mult: float = 3.0,
    enable_taker_unwind: bool = False,
    unwind_to: float = 0.0,
    taker_slip_ticks: int = 1,
    flow_window: int = 50,
    flow_skew_ticks: float = 0.0,
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

    dt = make_datetime_index(df["log_date"], df["TIME"]).to_numpy()
    prices = df["TRADE"].to_numpy(dtype=np.float64)
    sides = df["SIDE"].to_numpy(dtype=np.int8)
    vols = df["VOLUME"].to_numpy(dtype=np.float64)

    if tick_size is None:
        tick_size = infer_tick_size(prices)
    tick_size = float(tick_size)

    # Work in integer ticks for robust fill matching
    price_ticks = np.rint(prices / tick_size).astype(np.int64)

    strat = SimpleMarketMaker(
        tick=tick_size,
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

    n = len(df)

    mark_buf = deque(maxlen=int(mark_window))  # stores int ticks
    vol_buf = deque(maxlen=int(vol_window))
    flow_buf = deque(maxlen=int(flow_window))

    cur_bid = np.nan
    cur_ask = np.nan
    cur_mid = np.nan
    cur_mark = np.nan
    cur_spread_ticks = int(base_spread_ticks)

    cur_bid_tick: Optional[int] = None
    cur_ask_tick: Optional[int] = None

    cash = 0.0
    pos = 0.0

    # ---- preallocated outputs (fast + low memory) ----
    pnl_arr = np.empty(n, dtype=np.float64)
    cash_arr = np.empty(n, dtype=np.float64)
    pos_arr = np.empty(n, dtype=np.float64)
    mark_arr = np.empty(n, dtype=np.float64)
    mid_arr = np.empty(n, dtype=np.float64)
    bid_arr = np.empty(n, dtype=np.float64)
    ask_arr = np.empty(n, dtype=np.float64)
    spread_arr = np.empty(n, dtype=np.float32)
    maker_fills_arr = np.empty(n, dtype=np.int8)
    taker_fills_arr = np.empty(n, dtype=np.int8)
    maker_fee_arr = np.empty(n, dtype=np.float64)
    taker_fee_arr = np.empty(n, dtype=np.float64)

    quote_every_n = max(1, int(quote_every_n))

    def compute_mark() -> float:
        # Median of last `mark_window` trades, aligned to tick.
        if len(mark_buf) == 0:
            return float("nan")
        med_tick = float(np.median(np.fromiter(mark_buf, dtype=np.float64)))
        return float(med_tick * tick_size)

    def compute_spread_ticks() -> int:
        # compute ONLY from vol_buf (size <= vol_window, constant)
        base = int(base_spread_ticks)

        if vol_mult <= 0 or len(vol_buf) < 3 or tick_size <= 0:
            return max(int(min_spread_ticks), min(int(max_spread_ticks), base))

        p = np.fromiter(vol_buf, dtype=np.float64)
        p = p[p > 0]
        if p.size < 3:
            return max(int(min_spread_ticks), min(int(max_spread_ticks), base))

        r = np.diff(np.log(p))
        if r.size < 2:
            return max(int(min_spread_ticks), min(int(max_spread_ticks), base))

        sigma = float(np.std(r, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            return max(int(min_spread_ticks), min(int(max_spread_ticks), base))

        extra = int(round((float(vol_mult) * p[-1] * sigma) / tick_size))
        spread = base + extra
        spread = max(int(min_spread_ticks), min(int(max_spread_ticks), spread))
        return int(spread)

    for i in tqdm(range(n), total=n, desc="backtest", mininterval=0.5):
        trade_price = float(prices[i])
        trade_tick = int(price_ticks[i])

        n_maker_fills = 0
        maker_fee = 0.0

        if fill_full_qty_on_touch:
            fq = float(strat.qty)
        else:
            fq = float(min(strat.qty, float(vols[i])))

        # ------------------------------------------------------------------
        # 1) Execute against quotes that were posted BEFORE this trade.
        #    (No look-ahead: do not update quotes/mark from current trade first.)
        # ------------------------------------------------------------------
        if use_side_filter:
            if sides[i] == 1:  # aggressive sell -> hits bid -> we buy
                if cur_bid_tick is not None and trade_tick == cur_bid_tick:
                    notional = cur_bid * fq
                    fee = _fee(notional, fees.maker)
                    cash -= notional + fee
                    pos += fq
                    n_maker_fills = 1
                    maker_fee = fee
            else:  # aggressive buy -> hits ask -> we sell
                if cur_ask_tick is not None and trade_tick == cur_ask_tick:
                    notional = cur_ask * fq
                    fee = _fee(notional, fees.maker)
                    cash += notional
                    cash -= fee
                    pos -= fq
                    n_maker_fills = 1
                    maker_fee = fee
        else:
            if cur_bid_tick is not None and trade_tick == cur_bid_tick:
                notional = cur_bid * fq
                fee = _fee(notional, fees.maker)
                cash -= notional + fee
                pos += fq
                n_maker_fills += 1
                maker_fee += fee
            if cur_ask_tick is not None and trade_tick == cur_ask_tick:
                notional = cur_ask * fq
                fee = _fee(notional, fees.maker)
                cash += notional
                cash -= fee
                pos -= fq
                n_maker_fills += 1
                maker_fee += fee

        # ------------------------------------------------------------------
        # 2) Update buffers with the current trade (mark/vol/flow).
        # ------------------------------------------------------------------
        mark_buf.append(trade_tick)
        vol_buf.append(trade_price)

        # Flow: +volume for aggressive buys (SIDE!=1), -volume for aggressive sells (SIDE==1)
        sgn = -1.0 if sides[i] == 1 else 1.0
        flow_buf.append(sgn * float(vols[i]))

        mark = compute_mark()

        flow = 0.0
        if flow_skew_ticks != 0.0 and len(flow_buf) >= 3:
            fb = np.fromiter(flow_buf, dtype=np.float64)
            denom = float(np.sum(np.abs(fb)))
            if denom > 0:
                flow = float(np.clip(np.sum(fb) / denom, -1.0, 1.0))

        # ---- taker unwind (optional) ----
        n_taker_fills = 0
        taker_fee = 0.0

        if enable_taker_unwind and inv_limit > 0 and np.isfinite(mark):
            if pos > inv_limit:
                sell_qty = max(0.0, pos - float(unwind_to))
                px = mark - int(taker_slip_ticks) * tick_size
                notional = px * sell_qty
                fee = _fee(notional, fees.taker)
                cash += notional
                cash -= fee
                pos -= sell_qty
                n_taker_fills = 1
                taker_fee = fee
            elif pos < -inv_limit:
                buy_qty = max(0.0, float(unwind_to) - pos)
                px = mark + int(taker_slip_ticks) * tick_size
                notional = px * buy_qty
                fee = _fee(notional, fees.taker)
                cash -= notional + fee
                pos += buy_qty
                n_taker_fills = 1
                taker_fee = fee

        # ------------------------------------------------------------------
        # 3) (Re)quote for FUTURE trades.
        # ------------------------------------------------------------------
        if (i % quote_every_n == 0) or (not np.isfinite(cur_mark)):
            spread_ticks = compute_spread_ticks()
            q = strat.compute_quotes(
                mark_price=mark,
                position=pos,
                spread_ticks=spread_ticks,
                flow=flow,
                flow_skew_ticks=float(flow_skew_ticks),
            )
            cur_bid, cur_ask, cur_mid, cur_mark, cur_spread_ticks = q.bid, q.ask, q.mid, q.mark, q.spread_ticks

            cur_bid_tick = None if not np.isfinite(cur_bid) else price_to_tick_int(cur_bid, tick_size, side="nearest")
            cur_ask_tick = None if not np.isfinite(cur_ask) else price_to_tick_int(cur_ask, tick_size, side="nearest")

        pnl = cash + pos * mark

        # ---- write outputs ----
        pnl_arr[i] = pnl
        cash_arr[i] = cash
        pos_arr[i] = pos
        mark_arr[i] = mark
        mid_arr[i] = cur_mid
        bid_arr[i] = cur_bid
        ask_arr[i] = cur_ask
        spread_arr[i] = float(cur_spread_ticks)
        maker_fills_arr[i] = n_maker_fills
        taker_fills_arr[i] = n_taker_fills
        maker_fee_arr[i] = maker_fee
        taker_fee_arr[i] = taker_fee

    out = pd.DataFrame(
        {
            "datetime": dt,
            "pnl": pnl_arr,
            "cash": cash_arr,
            "position": pos_arr,
            "mark_price": mark_arr,
            "mid": mid_arr,
            "bid": bid_arr,
            "ask": ask_arr,
            "tick": float(tick_size),
            "spread_ticks": spread_arr,
            "maker_fills": maker_fills_arr,
            "taker_fills": taker_fills_arr,
            "maker_fee": maker_fee_arr,
            "taker_fee": taker_fee_arr,
        }
    )
    return out