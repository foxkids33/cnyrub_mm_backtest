from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import Fees, backtest
from .metrics import pnl_to_step_returns, pnl_to_time_returns, sharpe, sortino


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tape-only market making backtest for CNYRUB_TOM trades.parquet")
    p.add_argument("--trades", type=str, required=True, help="Path to trades.parquet")
    p.add_argument("--out_pnl", type=str, default="pnl.csv", help="Output path for pnl csv")

    # Strategy parameters
    p.add_argument("--base_spread_ticks", type=int, default=2)
    p.add_argument("--min_spread_ticks", type=int, default=1)
    p.add_argument("--max_spread_ticks", type=int, default=20)
    p.add_argument("--vol_window", type=int, default=50)
    p.add_argument("--vol_mult", type=float, default=3.0)

    p.add_argument("--qty", type=float, default=1_000_000.0)
    p.add_argument("--inv_limit", type=float, default=3_000_000.0)
    p.add_argument("--inv_skew_ticks", type=float, default=1.0)
    p.add_argument("--mark_window", type=int, default=5)

    # Execution controls
    p.add_argument("--quote_every_n", type=int, default=1, help="Update quotes every N trades.")
    p.add_argument("--use_side_filter", action="store_true", help="Use SIDE to avoid double fills (recommended).")
    p.set_defaults(use_side_filter=True)

    p.add_argument("--maker_fee", type=float, default=0.000005)
    p.add_argument("--taker_fee", type=float, default=0.000045)
    p.add_argument("--tick_size", type=float, default=None, help="Override tick size; otherwise inferred from trades.")
    p.add_argument("--fill_full_qty_on_touch", action="store_true", help="Fill full qty if any trade touches the quote.")
    p.set_defaults(fill_full_qty_on_touch=True)

    # Taker unwind
    p.add_argument("--enable_taker_unwind", action="store_true", help="Enable taker unwind when inventory breaks limit.")
    p.set_defaults(enable_taker_unwind=True)
    p.add_argument("--unwind_to", type=float, default=0.0, help="Target inventory after unwind (default 0).")
    p.add_argument("--taker_slip_ticks", type=int, default=1, help="Conservative slip in ticks for taker unwind.")

    # Metrics
    p.add_argument("--metric_freq", type=str, default="1S", help="Resample frequency for time-based Sharpe/Sortino, e.g. 1S, 10S, 1min.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    df = pd.read_parquet(args.trades)
    pnl_df = backtest(
        df,
        base_spread_ticks=int(args.base_spread_ticks),
        min_spread_ticks=int(args.min_spread_ticks),
        max_spread_ticks=int(args.max_spread_ticks),
        vol_window=int(args.vol_window),
        vol_mult=float(args.vol_mult),
        qty=float(args.qty),
        inv_limit=float(args.inv_limit),
        inv_skew_ticks=float(args.inv_skew_ticks),
        mark_window=int(args.mark_window),
        quote_every_n=int(args.quote_every_n),
        use_side_filter=bool(args.use_side_filter),
        fees=Fees(maker=float(args.maker_fee), taker=float(args.taker_fee)),
        tick_size=None if args.tick_size is None else float(args.tick_size),
        fill_full_qty_on_touch=bool(args.fill_full_qty_on_touch),
        enable_taker_unwind=bool(args.enable_taker_unwind),
        unwind_to=float(args.unwind_to),
        taker_slip_ticks=int(args.taker_slip_ticks),
    )

    out = Path(args.out_pnl)
    out.parent.mkdir(parents=True, exist_ok=True)
    pnl_df.to_csv(out, index=False)

    step_rets = pnl_to_step_returns(pnl_df["pnl"])
    time_rets = pnl_to_time_returns(pnl_df, freq=str(args.metric_freq))

    print(f"Saved: {out} (rows={len(pnl_df)})")
    print(f"Final PnL: {pnl_df['pnl'].iloc[-1]:.6f}")
    print(f"Sharpe (per-trade-step diffs): {sharpe(step_rets):.4f}")
    print(f"Sortino (per-trade-step diffs): {sortino(step_rets):.4f}")
    print(f"Sharpe (resampled {args.metric_freq}): {sharpe(time_rets):.4f}")
    print(f"Sortino (resampled {args.metric_freq}): {sortino(time_rets):.4f}")


if __name__ == "__main__":
    main()
